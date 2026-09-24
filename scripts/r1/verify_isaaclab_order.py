#!/usr/bin/env python3
"""Verify the inferred Isaac Lab joint/body ordering of the R1 against a live Isaac Lab run.

``r1_ordering.py`` is generated on a machine without Isaac Lab using a breadth-first rule
(validated on G1/H2). This script checks that rule against the *actual* articulation order
reported by Isaac Lab, taken from the ``layout.json`` written by:

    python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
        num_envs=1 headless=True ++dump_layout_dir=layouts/r1_dex3 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=<any small R1 motion dir>

Usage:
    python scripts/r1/verify_isaaclab_order.py --layout layouts/r1_dex3/layout.json

Exit code 0 = ordering matches; 1 = mismatch (regenerate r1_ordering.py from the printed
lists, or fix gear_sonic/utils/embodiment/ordering.py and re-run build_r1_assets.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.utils.embodiment import r1_spec  # noqa: E402

ORDERING = runpy.run_path(str(REPO / "gear_sonic/envs/manager_env/robots/r1_ordering.py"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--layout", required=True, help="layout.json from ++dump_layout_dir")
    args = ap.parse_args()
    layout = json.loads(Path(args.layout).read_text())

    live_joints = list(layout["joint_names_isaaclab"])
    live_bodies = list(layout["body_names_isaaclab"])
    expected_bodies = list(ORDERING["R1_ISAACLAB_JOINTS"])
    expected_joints = [r1_spec.MJCF_JOINT_ORDER[i] for i in ORDERING["R1_MUJOCO_TO_ISAACLAB_DOF"]]

    ok = True
    if live_joints != expected_joints:
        ok = False
        print("JOINT ORDER MISMATCH (Isaac Lab vs r1_ordering.py):")
        for i, (a, b) in enumerate(zip(live_joints, expected_joints)):
            flag = "" if a == b else "   <-- differs"
            print(f"  {i:2d}  live={a:32s} inferred={b}{flag}")
    else:
        print(f"joint order OK ({len(live_joints)} DOF)")

    if live_bodies[0] != r1_spec.ROOT_BODY:
        ok = False
        print(f"ROOT BODY MISMATCH: live={live_bodies[0]!r} expected={r1_spec.ROOT_BODY!r}")
    if live_bodies != expected_bodies:
        ok = False
        print("DOF-BODY ORDER MISMATCH:")
        print("  live    :", live_bodies)
        print("  inferred:", expected_bodies)
    else:
        print(f"DOF body order OK ({len(expected_bodies)} bodies)")

    if int(layout["num_actions"]) != r1_spec.NUM_DOF:
        ok = False
        print(f"ACTION DIM MISMATCH: live={layout['num_actions']} expected={r1_spec.NUM_DOF}")

    # Isaac Lab fuses fixed URDF links (head and palms) into their parent bodies.
    # Check the articulation bodies used by the experiment preset.
    required_bodies = set(
        r1_spec.TRACKED_BODY_NAMES
        + r1_spec.VR_3POINT_BODY
        + r1_spec.REWARD_POINT_BODY_3PT
        + r1_spec.EE_TERMINATION_BODIES
        + r1_spec.ANTI_SHAKE_BODIES
    )
    for name in sorted(required_bodies):
        if name not in live_bodies:
            ok = False
            print(f"BODY MISSING in Isaac Lab articulation: {name}")

    print("RESULT:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
