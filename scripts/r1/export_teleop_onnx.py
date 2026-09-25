#!/usr/bin/env python3
"""Export a teleop-only SONIC R1 checkpoint to ONNX on CPU, without Isaac Lab.

``eval_agent_trl.py ++export_onnx_only=True`` needs a live Isaac env (its example observations
come from ``env.reset_all()``) and exports every encoder of the three-encoder model. The teleop
model (``sonic_r1_dex3_teleop*``) has one encoder (VR 3-point) and one decoder (``g1_dyn``), and
every dimension it needs is in ``layouts/r1_dex3_teleop/layout.json``. This script rebuilds the
actor from the run's ``config.yaml`` and the layout, loads ``policy_state_dict`` and calls SONIC's
own ``inference_helpers.export_universal_token_module_as_onnx``. Next to ``<name>.onnx`` it writes:

* ``<name>.json``: what the runtime needs to build the input and apply the output (input term
  order and sizes, history length, Isaac Lab and MuJoCo joint orders, default pose, action scale,
  PD gains, effort limits, action clip, control period, VR 3-point bodies and offsets);
* ``<name>_check.npz``: random observations and the actions of the *training-time* forward
  (``Actor.forward`` on the tokenizer vector including ``encoder_index``), so
  ``scripts/r1/teleop/policy.py --check`` verifies the flat ONNX layout end to end (gate G0).

Runs in the training env (``sonic-train``: tensordict, trl, onnx), on CPU:

    CUDA_VISIBLE_DEVICES= python scripts/r1/export_teleop_onnx.py \\
        --run logs_rl/TRL_R1_Track/manager/universal_token/all_modes/<run> \\
        --checkpoint model_step_002000.pt   # -> <run>/exported/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ENCODER, DECODER = "teleop", "g1_dyn"


class _LenientUnpickler(pickle.Unpickler):
    """Checkpoints also pickle trainer objects; stub classes that are not importable here."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {"__setstate__": lambda self, state: None})


class _LenientPickle:
    Unpickler = _LenientUnpickler
    load = pickle.load


def per_joint(patterns: dict[str, float], joint_names: list[str], default=None) -> list[float]:
    """Resolve an Isaac Lab style {regex: value} dict for each joint (first match wins)."""
    values = []
    for joint in joint_names:
        match = next((v for p, v in patterns.items() if re.fullmatch(p, joint)), default)
        if match is None:
            raise KeyError(f"no pattern matches {joint}")
        values.append(float(match))
    return values


def gains(joint_names: list[str]) -> dict[str, list[float]]:
    from gear_sonic.utils.embodiment import r1_spec

    out = {"kp": [], "kd": [], "effort_limit": []}
    for joint in joint_names:
        group = next(
            g
            for g in r1_spec.ACTUATOR_GROUPS.values()
            if any(re.fullmatch(p, joint) for p in g["joints"])
        )
        out["kp"].append(float(group["stiffness"]))
        out["kd"].append(float(group["damping"]))
        out["effort_limit"].append(float(group["effort_limit"]))
    return out


def build_policy(cfg, layout: dict):
    from omegaconf import OmegaConf

    from gear_sonic.trl.utils.common import custom_instantiate

    tokenizer = layout["groups"]["tokenizer"]
    env_config = OmegaConf.create(
        {
            "obs": {
                "obs_dims": layout["obs_dims"],
                "group_obs_dims": {"tokenizer": {t["name"]: t["dims"] for t in tokenizer}},
                "group_obs_names": {"tokenizer": [t["name"] for t in tokenizer]},
            },
            "robot": {
                "type": layout["robot_type"],
                "actions_dim": layout["num_actions"],
                "algo_obs_dim_dict": layout["obs_dims"],
            },
        }
    )
    return custom_instantiate(
        cfg.algo.config.actor,
        env_config=env_config,
        algo_config=cfg.algo.config,
        module_dim_dict=cfg.algo.config.get("module_dim", {}),
        backbone_kwargs={},
        _resolve=False,
    )


def metadata(cfg, layout: dict, module, checkpoint: Path) -> dict:
    from gear_sonic.utils.embodiment import r1_spec

    joints = layout["joint_names_isaaclab"]
    required = set(module.encoder_input_features[ENCODER]) | set(
        module.decoder_input_features[DECODER]
    )
    tokenizer = [
        {"name": t["name"], "dim": int(np.prod(t["dims"]))}
        for t in layout["groups"]["tokenizer"]
        if t["name"] in required
    ]
    actor = [
        {
            "name": t["name"],
            "dim": int(np.prod(t["dims"])) // t["history"],
            "history": t["history"],
        }
        for t in layout["groups"]["policy"]
    ]
    motion = cfg.manager_env.commands.motion
    env = cfg.manager_env.config
    return {
        "checkpoint": str(checkpoint),
        "encoder": ENCODER,
        "decoder": DECODER,
        "input": {
            "layout": "[tokenizer terms | actor terms], each actor term = its history, oldest first",
            "tokenizer": tokenizer,
            "actor_obs": actor,
            "size": sum(t["dim"] for t in tokenizer) + layout["obs_dims"]["actor_obs"],
        },
        "joint_names_isaaclab": joints,
        "joint_names_mujoco": list(r1_spec.MJCF_JOINT_ORDER),
        "default_joint_pos": per_joint(r1_spec.INIT_JOINT_POS, joints, default=0.0),
        "action_scale": per_joint(r1_spec.ACTION_SCALE, joints),
        **gains(joints),
        "action_clip": float(env.action_clip_value),
        "control_dt": float(env.sim_dt) * int(env.decimation),
        "num_future_frames": int(motion.num_future_frames),
        "dt_future_ref_frames": float(motion.dt_future_ref_frames),
        "vr_3point_body": list(motion.vr_3point_body),
        "vr_3point_body_offset": [list(map(float, o)) for o in motion.vr_3point_body_offset],
        "init_pos_z": float(r1_spec.INIT_POS_Z),
    }


def check_data(policy, meta: dict, layout: dict, n: int = 64, seed: int = 0) -> dict:
    """Random observations and the training-time forward's actions (tokenizer incl. encoder_index)."""
    import torch

    rng = np.random.default_rng(seed)
    tokenizer_terms = layout["groups"]["tokenizer"]
    encoders = list(policy.actor_module.encoder_sample_probs)
    tok_parts, onnx_tok_parts = [], []
    for t in tokenizer_terms:
        dim = int(np.prod(t["dims"]))
        if t["name"] == "encoder_index":
            part = np.zeros((n, dim), np.float32)
            part[:, encoders.index(ENCODER)] = 1.0
        else:
            part = rng.normal(size=(n, dim)).astype(np.float32)
            onnx_tok_parts.append(part)
        tok_parts.append(part)
    actor_obs = rng.normal(size=(n, layout["obs_dims"]["actor_obs"])).astype(np.float32)
    names = [t["name"] for t in tokenizer_terms if t["name"] != "encoder_index"]
    assert names == [t["name"] for t in meta["input"]["tokenizer"]], (names, meta["input"])
    obs = {
        "actor_obs": torch.from_numpy(actor_obs)[:, None],
        "tokenizer": torch.from_numpy(np.concatenate(tok_parts, axis=1))[:, None],
    }
    with torch.no_grad():
        actions = policy.forward(obs)[:, -1].numpy()
    return {
        "onnx_input": np.concatenate(onnx_tok_parts + [actor_obs], axis=1),
        "actions": actions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True, help="training run directory")
    ap.add_argument("--checkpoint", default="last.pt", help="file name inside --run")
    ap.add_argument("--layout", type=Path, default=REPO / "layouts/r1_dex3_teleop/layout.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/exported")
    ap.add_argument("--name", default=None, help="default: <run name>_<checkpoint stem>")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    import torch

    from gear_sonic.utils import config_utils, inference_helpers

    config_utils.register_rl_resolvers()
    cfg = OmegaConf.load(args.run / "config.yaml")
    layout = json.loads(args.layout.read_text())
    policy = build_policy(cfg, layout)
    checkpoint = args.run / args.checkpoint
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=False, pickle_module=_LenientPickle
    )
    policy.load_state_dict(state["policy_state_dict"], strict=True)
    policy.eval()

    module = policy.actor_module
    meta = metadata(cfg, layout, module, checkpoint)
    name = args.name or f"{args.run.name}_{Path(args.checkpoint).stem}"
    args.out = args.out or args.run / "exported"
    args.out.mkdir(parents=True, exist_ok=True)
    inference_helpers.export_universal_token_module_as_onnx(
        module, ENCODER, DECODER, str(args.out), f"{name}.onnx"
    )
    (args.out / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    np.savez(args.out / f"{name}_check.npz", **check_data(policy, meta, layout))
    print(f"input size {meta['input']['size']}: {args.out / name}.{{onnx,json}} + _check.npz")


if __name__ == "__main__":
    main()
