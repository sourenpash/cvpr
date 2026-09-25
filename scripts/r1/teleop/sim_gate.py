#!/usr/bin/env python3
"""One-shot MuJoCo gate for an exported checkpoint: tracking success, smoothness, closed loop.

1. ``play_clips.py`` on ``data/r1_reference/eval100`` with the measured R1 actuators (#51):
   success overall / planner / mocap, MPJPE-L, VR 3-point error.
2. The full runtime (``run.py``) with the scripted operator, 60 s ``walk_reach`` (walk, stop,
   turn in place, sidestep, back, waving) and 20 s ``stand``: falls, palm error, and jitter:
   the RMS of the joint-velocity content above 5 Hz per group (rad/s) and the share of
   joint-velocity power above 5 Hz (power-weighted), against the IK reference for the arms.
   Calm (unrequested foot motion, D18): while a reference foot is planted, the robot foot's mean
   speed (``foot_slip_mm_s``), the share of that time it is lifted > 2 cm (``foot_lift_frac``)
   and the number of such lifts of 0.1 s or more with the reference foot planted for the 0.3 s
   before (``unplanned_steps``: balance steps, not a late reference step); and
   ``idle_foot_drift_mm``, how far a foot moves while the reference stands completely still.

Gates (docs/r1/SMOOTHNESS.md): arm high-frequency power <= 2 %, no falls, success within a
point of the previous checkpoint.

    python scripts/r1/teleop/sim_gate.py --onnx <export>.onnx --out logs_rl/sim2sim/<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def hf(q: np.ndarray, fps: float = 50.0, cutoff: float = 5.0) -> tuple[float, float]:
    """(RMS rad/s of the >cutoff part of joint velocity, its share of the total power)."""
    v = np.diff(q, axis=0) * fps
    X = np.fft.rfft(v - v.mean(0), axis=0)
    high = np.fft.rfftfreq(len(v), 1.0 / fps) > cutoff
    power = np.abs(X) ** 2
    X[~high] = 0
    rms = float(np.sqrt(np.mean(np.fft.irfft(X, n=len(v), axis=0) ** 2)))
    return rms, float(power[high].sum() / max(power.sum(), 1e-12))


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_metrics(model, z, names: list[str], skip: int = 50) -> dict[str, float]:
    """Robot foot motion the reference does not ask for (see the module docstring).

    Reference feet: forward kinematics of the planner's root (G1 pelvis; horizontal motion only
    matters) and the tokenizer's R1 legs at +0 s; planted = slower than 0.1 m/s.
    """
    import mujoco

    data = mujoco.MjData(model)
    jid = {
        n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names
    }
    legs = [model.jnt_qposadr[j] for j in range(1, 13)]  # MuJoCo order: the 12 leg joints first
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]

    def fk(root: np.ndarray, joints: dict[int, float]) -> np.ndarray:
        data.qpos[:] = model.qpos0
        data.qpos[:7] = root
        for adr, value in joints.items():
            data.qpos[adr] = value
        mujoco.mj_kinematics(model, data)
        return data.xpos[feet].copy()

    robot = np.array([
        fk(np.concatenate([p, q]), {jid[n]: v for n, v in zip(names, qs)})
        for p, q, qs in zip(z["robot_pos"], z["robot_quat"], z["robot_q"])
    ])  # fmt: skip
    ref = np.array([
        fk(r[:7], dict(zip(legs, o[6:18]))) for r, o in zip(z["ref_g1_qpos"], z["obs"])
    ])  # fmt: skip
    robot, ref = robot[skip:], ref[skip:]
    v_robot = np.linalg.norm(np.diff(robot[..., :2], axis=0), axis=-1) * 50.0  # (T-1, 2)
    planted = np.linalg.norm(np.diff(ref, axis=0), axis=-1) * 50.0 < 0.1
    lifted = (robot[1:, :, 2] - robot[:, :, 2].min(0)) > 0.02
    still = planted.all(1) & (np.abs(np.diff(z["obs"][skip:, 6:18], axis=0)).max(1) < 1e-6)
    drift = 0.0  # net foot displacement within each still stretch (>= 0.5 s)
    for seg in _runs(still, 25):
        xy = robot[1:][seg, :, :2]
        drift = max(drift, float(np.linalg.norm(xy - xy[0], axis=-1).max()))
    return {
        "foot_slip_mm_s": 1e3 * float(v_robot[planted].mean()) if planted.any() else 0.0,
        "foot_lift_frac": float(lifted[planted].mean()) if planted.any() else 0.0,
        "unplanned_steps": int(
            sum(  # lifts while the reference foot stood for 0.3 s (not lag)
                planted[max(0, r.start - 15) : r.stop, k].all()
                for k in range(2)
                for r in _runs(lifted[:, k] & planted[:, k], 5)
            )
        ),
        "idle_foot_drift_mm": 1e3 * drift,
    }


def _runs(mask: np.ndarray, min_len: int) -> list[slice]:
    """Stretches of consecutive True at least ``min_len`` long."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", default=str(REPO / "data/r1_reference/eval100"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-clips", action="store_true", help="closed-loop runs only")
    ap.add_argument("--run-args", default="", help="extra run.py arguments, e.g. calm limits")
    args = ap.parse_args()

    import mujoco

    py = sys.executable
    meta = json.loads(Path(args.onnx).with_suffix(".json").read_text())
    names = meta["joint_names_isaaclab"]
    groups = {
        "arms": [
            i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist"))
        ],
        "legs": [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))],
    }
    model = mujoco.MjModel.from_xml_path(
        str(REPO / "gear_sonic/data/assets/robot_description/mjcf/r1_dex3.xml")
    )
    qadr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names
    ]
    result = {"onnx": args.onnx, "run_args": args.run_args}
    clips_json = args.out.with_suffix(".clips.json")
    if not args.skip_clips:
        subprocess.run(
            [py, str(HERE / "play_clips.py"), "--onnx", args.onnx, "--clips", args.clips, "--profile", "issue51",
             "--workers", str(args.workers), "--threads", "1", "--out", str(clips_json)],
            check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        result["clips"] = json.loads(clips_json.read_text())["summary"]
    for program, seconds in (("walk_reach", 60), ("stand", 20)):
        rec = args.out.with_suffix(f".{program}.npz")
        out = subprocess.run(
            [py, str(HERE / "run.py"), "--onnx", args.onnx, "--input", f"script:{program}",
             "--seconds", str(seconds), "--sync-planner", "--record", str(rec), *args.run_args.split()],
            check=True, capture_output=True, text=True,
        )  # fmt: skip
        summary = json.loads(out.stdout.strip().splitlines()[-1])
        z = np.load(rec)
        q = z["robot_q"][50:]
        entry = {k: summary[k] for k in ("fell", "palm_err_p50_mm", "palm_err_p95_mm")}
        for g, idx in groups.items():
            entry[f"{g}_hf_rms"], entry[f"{g}_hf_share"] = hf(q[:, idx])
        entry["arms_ref_hf_share"] = hf(z["q_ik"][50:][:, qadr][:, groups["arms"]])[1]
        entry.update(foot_metrics(model, z, names))
        result[program] = entry
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
