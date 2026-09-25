#!/usr/bin/env python3
"""One-shot MuJoCo gate for an exported checkpoint: tracking success, smoothness, closed loop.

1. ``play_clips.py`` on ``data/r1_reference/eval100`` with the measured R1 actuators (#51):
   success overall / planner / mocap, MPJPE-L, VR 3-point error.
2. The full runtime (``run.py``) with the scripted operator, 60 s ``walk_reach`` (walk, stop,
   turn in place, sidestep, back, waving) and 20 s ``stand``: falls, palm error, and jitter:
   the RMS of the joint-velocity content above 5 Hz per group (rad/s) and the share of
   joint-velocity power above 5 Hz (power-weighted), against the IK reference for the arms.

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", default=str(REPO / "data/r1_reference/eval100"))
    ap.add_argument("--workers", type=int, default=12)
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
    result = {"onnx": args.onnx}
    clips_json = args.out.with_suffix(".clips.json")
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
             "--seconds", str(seconds), "--sync-planner", "--record", str(rec)],
            check=True, capture_output=True, text=True,
        )  # fmt: skip
        summary = json.loads(out.stdout.strip().splitlines()[-1])
        z = np.load(rec)
        q = z["robot_q"][50:]
        entry = {k: summary[k] for k in ("fell", "palm_err_p50_mm", "palm_err_p95_mm")}
        for g, idx in groups.items():
            entry[f"{g}_hf_rms"], entry[f"{g}_hf_share"] = hf(q[:, idx])
        entry["arms_ref_hf_share"] = hf(z["q_ik"][50:][:, qadr][:, groups["arms"]])[1]
        result[program] = entry
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
