#!/usr/bin/env python3
"""Gate G1: track training clips with the exported policy in MuJoCo (sim-to-sim).

Each clip (``export_r1_reference.py``) is played like one evaluation episode in Isaac Lab: the
robot starts in the reference's first frame, the runtime builds the observation
(``observations.py``) from the MuJoCo state and the reference, the ONNX policy acts at 50 Hz, and
the episode fails on the training terminations (``base_adaptive_strict_ori_foot_xyz``):

* pelvis height error > 0.15 m;
* pelvis orientation error^2 > 0.2 rad^2 (full orientation, heading included);
* ankle or wrist height error > 0.15 m;
* ankle position error > 0.2 m, with the reference re-anchored at the robot's pelvis (x, y) and
  heading (``body_pos_relative_w``).

A clip succeeds if it reaches its end. The Isaac Lab ``eval/`` success rate of the same
checkpoint is the bar. ``--profile issue51`` adds the measured R1 armature and joint friction.

    python scripts/r1/teleop/play_clips.py --onnx <export>.onnx --clips data/r1_reference/eval100 \\
        --profile nominal --workers 8
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observations as ob  # noqa: E402
from policy import TeleopPolicy  # noqa: E402
from robot_mujoco import MujocoR1  # noqa: E402
import rotations as rot  # noqa: E402

FRAME_SKIP = 5  # 0.1 s between future frames at 50 Hz
TERMINATION_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]
FEET = TERMINATION_BODIES[:2]
TRACKED = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link",
    "left_elbow_link", "left_wrist_roll_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_roll_link",
]  # fmt: skip


class ClipReference:
    """Reference terms at time step t from a clip exported by export_r1_reference.py."""

    def __init__(self, path: str, meta: dict):
        z = np.load(path)
        self.name = str(z["name"])
        assert list(z["joint_names"]) == meta["joint_names_isaaclab"], "joint order mismatch"
        self.body_names = list(z["body_names"])
        self.joint_pos, self.joint_vel = z["joint_pos"], z["joint_vel"]
        self.body_pos, self.body_quat = z["body_pos_w"], z["body_quat_w"]
        self.num_frames = len(self.joint_pos)
        isaac = meta["joint_names_isaaclab"]
        self.legs = [isaac.index(n) for n in meta["joint_names_mujoco"][:12]]
        self.pelvis = self.body_names.index("pelvis")
        self.vr_bodies = [self.body_names.index(n) for n in meta["vr_3point_body"]]
        self.vr_offsets = np.asarray(meta["vr_3point_body_offset"])
        self.num_future = int(meta["num_future_frames"])

    def anchor(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.body_pos[t, self.pelvis], self.body_quat[t, self.pelvis]

    def vr_points(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        quat = self.body_quat[t, self.vr_bodies]
        return self.body_pos[t, self.vr_bodies] + rot.apply(quat, self.vr_offsets), quat

    def tokenizer(self, t: int, robot_quat: np.ndarray) -> dict[str, np.ndarray]:
        frames = np.minimum(t + FRAME_SKIP * np.arange(self.num_future), self.num_frames - 1)
        anchor_pos, anchor_quat = self.anchor(t)
        vr_pos, vr_quat = ob.vr_3point_local(anchor_pos, anchor_quat, *self.vr_points(t))
        return {
            "motion_anchor_ori_heading": ob.anchor_ori_heading(robot_quat, anchor_quat),
            "command_multi_future_lower_body": ob.lower_body_command(
                self.joint_pos[frames][:, self.legs], self.joint_vel[frames][:, self.legs]
            ),
            "vr_3point_local_target": vr_pos,
            "vr_3point_local_orn_target": vr_quat,
        }


def relative_reference(ref: ClipReference, t: int, robot_pos, robot_quat) -> np.ndarray:
    """Reference bodies placed at the robot's pelvis (x, y) and heading (body_pos_relative_w)."""
    anchor_pos, anchor_quat = ref.anchor(t)
    delta = rot.heading(rot.multiply(robot_quat, rot.inverse(anchor_quat)))
    base = np.array([robot_pos[0], robot_pos[1], anchor_pos[2]])
    return base + rot.apply(delta[None], ref.body_pos[t] - anchor_pos[None])


def play(
    ref: ClipReference, policy: TeleopPolicy, meta: dict, profile: str, delay_s: float
) -> dict:
    robot = MujocoR1(meta, profile=profile, delay_s=delay_s)
    anchor_pos, anchor_quat = ref.anchor(0)
    root_lin_vel = (ref.body_pos[1, ref.pelvis] - anchor_pos) * 50.0 if ref.num_frames > 1 else 0
    robot.reset(anchor_pos, anchor_quat, ref.joint_pos[0], ref.joint_vel[0])
    robot.data.qvel[:3] = root_lin_vel
    builder = ob.ObservationBuilder(meta)
    default = np.asarray(meta["default_joint_pos"])
    scale = np.asarray(meta["action_scale"])
    clip = float(meta["action_clip"])
    last_action = np.zeros(len(default))
    body_ids = {n: ref.body_names.index(n) for n in TRACKED}
    errors, vr_errors, infer = [], [], []
    failed, reason, t = False, "", 0
    while t < ref.num_frames - 1:
        tok = ref.tokenizer(t, robot.root_quat)
        proprio = builder.proprioception(
            robot.joint_pos, robot.joint_vel, robot.root_quat, robot.gyro, last_action
        )
        obs = builder.build(proprio, tok)
        t0 = time.perf_counter()
        action = np.clip(policy(obs), -clip, clip)
        infer.append(time.perf_counter() - t0)
        last_action = action
        robot.step(default + scale * action)
        t += 1
        # terminations at the new time step, as Isaac Lab evaluates them after the physics step
        pos, quat = robot.root_pos, robot.root_quat
        ref_pos, ref_quat = ref.anchor(t)
        rel = relative_reference(ref, t, pos, quat)
        robot_bodies = {n: robot.body_pose(n)[0] for n in TRACKED}
        if abs(ref_pos[2] - pos[2]) > 0.15:
            failed, reason = True, "anchor_height"
        elif float(np.sum(rot_error(ref_quat, quat) ** 2)) > 0.2:
            failed, reason = True, "anchor_ori"
        elif any(abs(rel[body_ids[n], 2] - robot_bodies[n][2]) > 0.15 for n in TERMINATION_BODIES):
            failed, reason = True, "ee_height"
        elif any(np.linalg.norm(rel[body_ids[n]] - robot_bodies[n]) > 0.2 for n in FEET):
            failed, reason = True, "foot_pos"
        # local (pelvis-relative) errors, as SONIC's MPJPE-L
        ref_local = ref.body_pos[t, [body_ids[n] for n in TRACKED]] - ref_pos
        rob_local = np.array([robot_bodies[n] for n in TRACKED]) - pos
        errors.append(np.linalg.norm(ref_local - rob_local, axis=-1).mean())
        vr_ref, _ = ref.vr_points(t)
        vr_rob = [
            robot.body_pose(n)[0] + rot.apply(robot.body_pose(n)[1], o)
            for n, o in zip(meta["vr_3point_body"], ref.vr_offsets)
        ]
        vr_errors.append(
            np.linalg.norm((vr_ref - ref_pos) - (np.array(vr_rob) - pos), axis=-1).mean()
        )
        if failed:
            break
    return {
        "name": ref.name,
        "success": not failed,
        "reason": reason,
        "progress": t / max(ref.num_frames - 1, 1),
        "mpjpe_l_mm": 1e3 * float(np.mean(errors)) if errors else float("nan"),
        "vr_3point_mm": 1e3 * float(np.mean(vr_errors)) if vr_errors else float("nan"),
        "infer_ms": 1e3 * float(np.median(infer)) if infer else float("nan"),
    }


def rot_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Axis-angle of q1 * q2^-1 (isaaclab quat_box_minus)."""
    d = rot.positive_w(rot.multiply(q1, rot.inverse(q2)))
    s = np.linalg.norm(d[1:])
    return np.zeros(3) if s < 1e-9 else d[1:] / s * 2.0 * np.arctan2(s, d[0])


def _worker(job) -> list[dict]:
    policy = TeleopPolicy(job["onnx"], threads=job["threads"])
    out = []
    for path in job["clips"]:
        ref = ClipReference(path, policy.meta)
        out.append(play(ref, policy, policy.meta, job["profile"], job["delay_s"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--clips", type=Path, required=True, help="dir of export_r1_reference npz")
    ap.add_argument("--profile", choices=["nominal", "issue51"], default="nominal")
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write per-clip results (json)")
    args = ap.parse_args()

    clips = sorted(glob.glob(str(args.clips / "*.npz")))[: args.max_clips]
    jobs = [
        {"onnx": args.onnx, "clips": clips[w :: args.workers], "profile": args.profile,
         "delay_s": args.delay_ms / 1e3, "threads": args.threads}
        for w in range(args.workers)
    ]  # fmt: skip
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = sorted(
            (r for part in pool.map(_worker, jobs) for r in part), key=lambda r: r["name"]
        )
    summary = {"profile": args.profile, "delay_ms": args.delay_ms, "num_clips": len(results)}
    for group in ("all", "planner", "mocap"):
        rs = [
            r for r in results
            if group == "all" or r["name"].startswith("planner_") == (group == "planner")
        ]  # fmt: skip
        if rs:
            summary[f"success_{group}"] = float(np.mean([r["success"] for r in rs]))
            summary[f"num_{group}"] = len(rs)
    summary["mpjpe_l_mm"] = float(np.nanmean([r["mpjpe_l_mm"] for r in results]))
    summary["vr_3point_mm"] = float(np.nanmean([r["vr_3point_mm"] for r in results]))
    summary["infer_ms"] = float(np.nanmedian([r["infer_ms"] for r in results]))
    summary["reasons"] = {
        k: sum(r["reason"] == k for r in results) for k in {r["reason"] for r in results} if k
    }
    summary["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "clips": results}, indent=1))


if __name__ == "__main__":
    main()
