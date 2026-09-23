"""Dump the observation/network layout of a live training run (for checkpoint surgery).

Called from ``train_agent_trl.py`` when ``++dump_layout_dir=<dir>`` is given: writes
``layout.json`` (term names/dims/history per observation group, joint/body names, encoder
inputs, token dims) and ``template.pt`` (randomly-initialised policy/value state dicts in the
checkpoint format used by ``ModelSaveCallback``), then the caller exits.

Run it once per embodiment:

    # G1 (source of the pretrained weights)
    python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_v1_1 \
        num_envs=1 headless=True ++dump_layout_dir=layouts/g1_v1_1 ...
    # R1 (target)
    python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
        num_envs=1 headless=True ++dump_layout_dir=layouts/r1_dex3 ...

Then: ``python scripts/r1/surgery_g1_to_r1_checkpoint.py --src sonic_v1_1/last.pt
--src-layout layouts/g1_v1_1/layout.json --tgt layouts/r1_dex3/template.pt
--tgt-layout layouts/r1_dex3/layout.json --out r1_init/last.pt``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _to_list(x):
    try:
        return [int(v) for v in x]
    except TypeError:
        return int(x)


def build_layout(env, config, policy) -> dict:
    """Collect the layout dictionary (no file I/O). Isaac Lab objects accessed defensively."""
    robot = env.env.scene["robot"]
    om = env.env.observation_manager

    groups = {}
    for group, names in om.active_terms.items():
        dims = om.group_obs_term_dim[group]
        cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(group, [None] * len(names))
        terms = []
        for name, dim, cfg in zip(names, dims, cfgs):
            hist = int(getattr(cfg, "history_length", 0) or 0) if cfg is not None else 0
            terms.append({"name": name, "dims": _to_list(dim), "history": max(hist, 1)})
        groups[group] = terms

    backbone = getattr(policy, "backbone", None)
    enc_cfg = config.algo.config.actor.backbone.get("encoders", {})
    encoder_inputs = {k: list(v.get("inputs", [])) for k, v in enc_cfg.items() if hasattr(v, "get")}
    tok_cfg = config.manager_env.observations.get("tokenizer", {})
    wrist_cfg = (
        tok_cfg.get("joint_pos_multi_future_wrist_for_smpl", {}) if hasattr(tok_cfg, "get") else {}
    )
    wrist_idx = (
        list(wrist_cfg.get("params", {}).get("joints_idx", [])) if hasattr(wrist_cfg, "get") else []
    )

    layout = {
        "robot_type": str(config.manager_env.config.robot.type),
        "num_actions": int(env.env.action_space.shape[-1]),
        "joint_names_isaaclab": list(robot.joint_names),
        "body_names_isaaclab": list(robot.body_names),
        "groups": groups,
        "tokenizer_term_dims": {
            k: _to_list(v)
            for k, v in env.config["obs"]["group_obs_dims"].get("tokenizer", {}).items()
        },
        "tokenizer_term_order": list(env.config["obs"]["group_obs_names"].get("tokenizer", [])),
        "encoder_inputs": encoder_inputs,
        "proprioception_features": list(
            getattr(backbone, "proprioception_features", ["actor_obs"])
        ),
        "token_dim": int(getattr(backbone, "token_dim", 0) or 0),
        "max_num_tokens": int(getattr(backbone, "max_num_tokens", 0) or 0),
        "wrist_for_smpl_joints_idx": wrist_idx,
        "obs_dims": {k: int(v) for k, v in env.config["obs"]["obs_dims"].items()},
    }
    return layout


def dump_layout_and_template(env, config, policy, value_model, out_dir: str | Path) -> Path:
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layout = build_layout(env, config, policy)
    (out / "layout.json").write_text(json.dumps(layout, indent=2))
    template = {
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value_model.state_dict() if value_model is not None else None,
    }
    torch.save(template, out / "template.pt")
    return out
