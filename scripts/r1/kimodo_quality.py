#!/usr/bin/env python3
"""Filter transferred Kimodo clips (R1 motion_lib PKLs) before they go into training.

Text-to-motion samples vary; trackers trained on generated data filter it (PHUMA 2510.26236:
physically implausible clips hurt; RLPF 2506.12769: 43-48 % of raw text-to-motion clips were
trackable). A clip is kept when its R1 reference is

* smooth: RMS of the >5 Hz part of the joint velocities <= 0.3 rad/s and root jerk p95
  <= 200 m/s^3 (roughly the 90th percentiles of our mocap clips, S0: 0.27 rad/s, 171 m/s^3;
  medians: Kimodo 0.09 rad/s / 13 m/s^3, S0 0.09 / 28, gesture mocap 0.26 / 68, planner 0.20 / 485);
* free of self-penetration: arm/hand collision geoms off the torso and legs in >= 98 % of
  frames (MuJoCo contacts of ``r1_dex3.xml``, adjacent bodies excluded as in the model);
* on the ground: soles within 5 cm of the floor in >= 95 % of frames for clips that should stand
  (entries without walk/step/turn in their name);
* doing what it was asked (gestures): the named hand rises above the shoulder for >= 0.4 s for
  wave / beckon / point_up / arms_up / high_five / salute / reach_up entries.

Writes ``<out>/<clip>.pkl`` symlinks for kept clips and ``quality.json`` with every metric.

    python scripts/r1/kimodo_quality.py --clips data/motion_lib_r1/K1 --manifest data/kimodo_g1/K1/manifest.json \\
        --out data/motion_lib_r1/K1_ok
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/r1_dex3.xml"
RAISE = {  # entry -> hands that must go above the shoulder
    "wave_right": ["right"], "wave_left": ["left"], "wave_both": ["left", "right"],
    "wave_big": ["right"], "beckon": ["right"], "point_up": ["right"], "arms_up": ["left", "right"],
    "high_five": ["right"], "salute": ["right"], "reach_up": ["right"], "walk_wave": ["right"],
    "walk_then_wave": ["right"], "wave_then_walk": ["left"],
}  # fmt: skip
MOVING = ("walk", "step", "turn", "sidestep")


def hf_velocity_rms(q: np.ndarray, fps: float, cutoff: float = 5.0) -> float:
    """RMS (rad/s) over joints and time of the joint-velocity content above ``cutoff`` Hz."""
    v = np.diff(q, axis=0) * fps
    X = np.fft.rfft(v - v.mean(0), axis=0)
    X[np.fft.rfftfreq(len(v), 1.0 / fps) <= cutoff] = 0
    return float(np.sqrt(np.mean(np.fft.irfft(X, n=len(v), axis=0) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clips", type=Path, required=True, help="dir of transferred R1 PKLs")
    ap.add_argument("--manifest", type=Path, required=True, help="generate_kimodo_motions manifest")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    entries = {c["clip"]: c["entry"] for c in json.loads(args.manifest.read_text())}
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    geom_body = m.geom_bodyid
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    arm = {
        b
        for b, n in enumerate(names)
        if any(k in n for k in ("shoulder", "elbow", "wrist", "hand"))
    }
    trunk = {
        b
        for b, n in enumerate(names)
        if any(k in n for k in ("pelvis", "torso", "waist", "hip", "knee", "ankle"))
    }
    sole = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
    wrist = {
        s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_roll_link")
        for s in ("left", "right")
    }
    shoulder = {
        s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_roll_link")
        for s in ("left", "right")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report, kept = {}, 0
    for path in sorted(glob.glob(str(args.clips / "**/*.pkl"), recursive=True)):
        name = Path(path).stem
        e = joblib.load(path)[name]
        fps = float(e["fps"])
        dof, trans = np.asarray(e["dof"]), np.asarray(e["root_trans_offset"])
        quat = np.asarray(e["root_rot"])[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        entry = entries.get(name, "")
        pen, grounded, above = 0, 0, {s: 0 for s in ("left", "right")}
        for t in range(len(dof)):
            d.qpos[:3], d.qpos[3:7], d.qpos[7:] = trans[t], quat[t], dof[t]
            mujoco.mj_forward(m, d)
            hit = any(
                (geom_body[c.geom1] in arm and geom_body[c.geom2] in trunk)
                or (geom_body[c.geom2] in arm and geom_body[c.geom1] in trunk)
                for c in d.contact[: d.ncon]
                if c.dist < -0.01
            )
            pen += hit
            grounded += min(d.site_xpos[s][2] for s in sole) < 0.05
            for s in above:
                above[s] += d.xpos[wrist[s]][2] > d.xpos[shoulder[s]][2]
        T = len(dof)
        jerk = np.diff(trans, n=3, axis=0) * fps**3
        metrics = {
            "entry": entry,
            "frames": T,
            "hf_velocity_rms": hf_velocity_rms(dof, fps),
            "root_jerk_p95": (
                float(np.percentile(np.linalg.norm(jerk, axis=1), 95)) if T > 4 else 0.0
            ),
            "self_penetration_frac": pen / T,
            "grounded_frac": grounded / T,
            "raised_s": {s: n / fps for s, n in above.items()},
        }
        reasons = []
        if metrics["hf_velocity_rms"] > 0.3:
            reasons.append("jitter")
        if metrics["root_jerk_p95"] > 200.0:
            reasons.append("root jerk")
        if metrics["self_penetration_frac"] > 0.02:
            reasons.append("self penetration")
        if not any(k in entry for k in MOVING) and metrics["grounded_frac"] < 0.95:
            reasons.append("not grounded")
        for s in RAISE.get(entry, []):
            if metrics["raised_s"][s] < 0.4:
                reasons.append(f"{s} hand not raised")
        metrics["kept"], metrics["reasons"] = not reasons, reasons
        report[name] = metrics
        if not reasons:
            link = args.out / f"{name}.pkl"
            if not link.exists():
                link.symlink_to(Path(path).resolve())
            kept += 1
    summary = {"clips": len(report), "kept": kept}
    for why in ("jitter", "root jerk", "self penetration", "not grounded", "hand not raised"):
        summary[why] = sum(any(why in r for r in m_["reasons"]) for m_ in report.values())
    (args.out / "quality.json").write_text(
        json.dumps({"summary": summary, "clips": report}, indent=1)
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
