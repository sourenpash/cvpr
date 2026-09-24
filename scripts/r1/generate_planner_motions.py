#!/usr/bin/env python3
"""Generate R1 locomotion references with SONIC's kinematic planner, as the Quest demo drives it.

At deployment (SONIC ``VR_3PT`` mode) the teleop encoder's lower-body input comes from the
joystick-driven planner. Training on the planner's own output closes that gap: this script
drives ``planner_loop.PlannerLoop`` with random stick scripts that follow SONIC's gamepad
semantics (``gamepad.hpp``: left stick = walking direction relative to the facing, right stick
turns the facing at up to 2 rad/s, per-mode speed bands, centred stick = idle), records the 50 Hz
reference the policy would track, and converts each clip to an R1 motion_lib PKL with the same
joint-name transfer as the BONES-SEED data (``transfer_g1_motion_lib_to_r1.py``).

    python scripts/r1/generate_planner_motions.py --output data/motion_lib_r1/planner_v2 \
        --num-clips 200 --duration 20 --workers 6 --threads 2

The planner's side-steps and turns exceed the R1's (and G1's) ankle-roll limit of 0.262 rad; the
transfer clamps them, and the clips are kept by default because the deployed policy meets exactly
these clamped references (the demo runtime must apply the same clamp).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "gear_sonic/data_process"))

import planner_loop as pl  # noqa: E402
from transfer_g1_motion_lib_to_r1 import G1ToR1Transfer  # noqa: E402

# Direction of the left stick relative to the facing (rad) and its probability.
DIRECTIONS = [
    (0.0, 0.45), (np.pi, 0.15), (np.pi / 2, 0.08), (-np.pi / 2, 0.08),
    (np.pi / 4, 0.06), (-np.pi / 4, 0.06), (3 * np.pi / 4, 0.06), (-3 * np.pi / 4, 0.06),
]  # fmt: skip
MODES = [(pl.IDLE, 0.25), (pl.SLOW_WALK, 0.40), (pl.WALK, 0.35)]


def _choice(rng, options):
    values, probs = zip(*options)
    return values[rng.choice(len(values), p=np.asarray(probs) / np.sum(probs))]


def stick_script(rng: np.random.Generator, duration: float, dt: float = 0.02):
    """Yield one :class:`planner_loop.Command` per 50 Hz tick for ``duration`` seconds."""
    facing, t = 0.0, 0.0
    first = True
    while t < duration:
        seg = rng.uniform(1.0, 2.0) if first else rng.uniform(1.5, 5.0)
        mode = pl.IDLE if first else _choice(rng, MODES)  # start from standing, like the demo
        first = False
        direction = _choice(rng, DIRECTIONS)
        band = pl.SPEED_BANDS.get(mode)
        speed = float(rng.uniform(*band)) if band and rng.random() < 0.5 else -1.0
        yaw_rate = (
            0.0 if rng.random() < 0.55 else float(rng.choice([-1, 1]) * rng.uniform(0.3, 1.5))
        )
        for _ in range(int(round(seg / dt))):
            facing += yaw_rate * dt
            face = np.array([np.cos(facing), np.sin(facing), 0.0])
            if mode == pl.IDLE:
                yield pl.Command(mode=pl.IDLE, face_dir=face)
            else:
                move = np.array([np.cos(facing + direction), np.sin(facing + direction), 0.0])
                yield pl.Command(mode=mode, speed=speed, move_dir=move, face_dir=face)
            t += dt


def generate_clip(loop_model: pl.PlannerModel, rng: np.random.Generator, duration: float) -> dict:
    loop = pl.PlannerLoop(loop_model)
    frames = np.array([loop.tick(cmd) for cmd in stick_script(rng, duration)])
    root_rot_xyzw = frames[:, [4, 5, 6, 3]]
    return {
        "root_trans_offset": frames[:, :3].astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "dof": frames[:, 7:].astype(np.float32),
        "fps": 50.0,
        "num_replans": loop.num_replans,
    }


def _worker(job):
    """Generate, transfer and save clips ``job['indices']`` (one process, bounded threads)."""
    model = pl.PlannerModel(job["planner"], threads=job["threads"])
    transfer = G1ToR1Transfer()
    out_dir, stats = Path(job["output"]), []
    for i in job["indices"]:
        rng = np.random.default_rng([job["seed"], i])  # independent of the worker split
        model.rng = np.random.default_rng([job["seed"], i, 1])
        name = f"planner_v2_{job['seed']:03d}_{i:04d}"
        g1 = generate_clip(model, rng, job["duration"])
        # The G1 clip too (motion_lib format), so the R1 transfer can be re-run without the planner:
        # transfer_g1_motion_lib_to_r1.py --input <output>_g1 --max-clamp-frac 1.0
        g1_entry = {k: g1[k] for k in ("root_trans_offset", "root_rot", "dof", "fps")}
        joblib.dump({name: g1_entry}, Path(job["output"] + "_g1") / f"{name}.pkl")
        out, st = transfer.transfer_entry(name, g1)
        out["smpl_joints"] = np.zeros((out["dof"].shape[0], 24, 3), np.float32)  # S0 format
        st.dropped = st.clamp_frac > job["max_clamp_frac"] or st.vel_frac > job["max_vel_frac"]
        st.reason = (
            "clamp" if st.clamp_frac > job["max_clamp_frac"] else ("velocity" if st.dropped else "")
        )
        stats.append({**asdict(st), "num_replans": g1["num_replans"]})
        if not st.dropped:
            joblib.dump({name: out}, out_dir / f"{name}.pkl")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-clips", type=int, default=200)
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--planner", type=Path, default=None, help="planner_sonic.onnx (default: HF cache)"
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads per worker")
    ap.add_argument(
        "--max-clamp-frac", type=float, default=1.0, help="drop above (default: keep all)"
    )
    ap.add_argument("--max-vel-frac", type=float, default=0.05)
    args = ap.parse_args()

    planner = str(args.planner or pl.find_planner_onnx())
    args.output.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + "_g1").mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "indices": list(range(w, args.num_clips, args.workers)), "planner": planner,
            "threads": args.threads, "seed": args.seed, "duration": args.duration,
            "output": str(args.output), "max_clamp_frac": args.max_clamp_frac,
            "max_vel_frac": args.max_vel_frac,
        }
        for w in range(args.workers)
    ]  # fmt: skip
    with mp.get_context("spawn").Pool(args.workers) as pool:
        stats = sorted(
            (s for part in pool.map(_worker, jobs) for s in part), key=lambda s: s["name"]
        )
    kept = [s for s in stats if not s["dropped"]]
    report = {
        "planner": planner,
        "num_clips": len(stats),
        "num_kept": len(kept),
        "kept_hours": sum(s["num_frames"] for s in kept) / 50.0 / 3600.0,
        "mean_clamp_frac_kept": float(np.mean([s["clamp_frac"] for s in kept])) if kept else 0.0,
        "leg_scale": G1ToR1Transfer().scale,
        "motions": stats,
    }
    (args.output / "planner_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "motions"}, indent=1))


if __name__ == "__main__":
    main()
