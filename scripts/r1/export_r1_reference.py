#!/usr/bin/env python3
"""Export R1 motion_lib clips as the training environment sees them, one ``.npz`` per clip.

``MotionLibRobot`` resamples each clip to 50 Hz, fixes its height, runs forward kinematics on
``r1_dex3.xml`` and reorders everything to Isaac Lab order, exactly as in training. The runtime
(``scripts/r1/teleop/``) reads the result without torch or Isaac: MuJoCo clip playback (gate G1)
and checks of its own reference processing (gate G0). Kinematics only, CPU; runs in the
``sonic-train`` env (the motion library needs torch and wandb, not Isaac Sim):

    CUDA_VISIBLE_DEVICES= python scripts/r1/export_r1_reference.py \\
        --motion-file data/motion_lib_r1/B --out data/r1_reference/eval100 --num-clips 100

Each ``.npz``: ``joint_pos``/``joint_vel`` (T, 24), ``body_pos_w`` (T, 25, 3), ``body_quat_w``
(T, 25, 4, w x y z), ``joint_names``/``body_names`` (Isaac Lab order), ``fps``, ``name``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile

import easydict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def r1_ordering():
    """Load the generated ordering tables without the robots package (which imports Isaac Lab)."""
    path = REPO / "gear_sonic/envs/manager_env/robots/r1_ordering.py"
    spec = importlib.util.spec_from_file_location("r1_ordering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def motion_lib_cfg(motion_file: str) -> easydict.EasyDict:
    o = r1_ordering()
    return easydict.EasyDict(
        {
            "motion_file": motion_file,
            "smpl_motion_file": None,
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": "r1_dex3.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": 50,
            "multi_thread": False,
            "use_parallel_fk": False,
            "robot_type": "r1_dex3",
            "body_indexes_data": list(range(len(o.R1_MUJOCO_TO_ISAACLAB_BODY))),
            "mujoco_to_isaaclab_body": o.R1_MUJOCO_TO_ISAACLAB_BODY,
            "mujoco_to_isaaclab_dof": o.R1_MUJOCO_TO_ISAACLAB_DOF,
            "isaaclab_to_mujoco_body": o.R1_ISAACLAB_TO_MUJOCO_BODY,
            "isaaclab_to_mujoco_dof": o.R1_ISAACLAB_TO_MUJOCO_DOF,
        }
    )


def pick_clips(motion_file: Path, num_clips: int, seed: int) -> list[Path]:
    """Deterministic, half planner clips and half mocap clips when both exist."""
    # glob follows the dataset's directory symlinks (Path.rglob does not before Python 3.13)
    files = sorted(Path(f) for f in glob.glob(str(motion_file / "**/*.pkl"), recursive=True))
    files = [p for p in files if "report" not in p.name]
    assert files, f"no clips under {motion_file}"
    planner = [p for p in files if p.name.startswith("planner_")]
    mocap = [p for p in files if not p.name.startswith("planner_")]
    rng = random.Random(seed)
    k = min(len(planner), num_clips // 2) if mocap else min(len(planner), num_clips)
    chosen = rng.sample(planner, k) + rng.sample(mocap, min(len(mocap), num_clips - k))
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--motion-file", type=Path, required=True, help="R1 motion_lib PKL dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-clips", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from gear_sonic.utils.motion_lib import motion_lib_robot

    o = r1_ordering()
    clips = pick_clips(args.motion_file, args.num_clips, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for p in clips:  # the motion lib loads a directory; link exactly the chosen clips
            (Path(tmp) / p.name).symlink_to(p.resolve())
        lib = motion_lib_robot.MotionLibRobot(motion_lib_cfg(tmp), num_envs=1, device="cpu")
        lib.load_motions_for_training(max_num_seqs=len(clips))
        names = [str(k) for k in lib.curr_motion_keys]
        assert len(set(names)) == len(clips), f"loaded {len(set(names))} of {len(clips)} clips"
        layout = json.loads((REPO / "layouts/r1_dex3_teleop/layout.json").read_text())
        body_names, joint_names = layout["body_names_isaaclab"], layout["joint_names_isaaclab"]
        # r1_ordering's "ISAACLAB_JOINTS" are the bodies (SONIC's naming)
        assert body_names == list(o.R1_ISAACLAB_JOINTS) and lib.body_pos_w.shape[1] == 25
        with torch.no_grad():
            for i, name in enumerate(names):
                start, n = int(lib.length_starts[i]), int(lib._motion_num_frames[i])  # noqa: SLF001
                sl = slice(start, start + n)
                np.savez(
                    args.out / f"{name}.npz",
                    name=name,
                    fps=50.0,
                    joint_pos=lib.dof_pos[sl].cpu().numpy(),
                    joint_vel=lib.dof_vel[sl].cpu().numpy(),
                    body_pos_w=lib.body_pos_w[sl].cpu().numpy(),
                    body_quat_w=lib.body_quat_w[sl].cpu().numpy(),
                    joint_names=np.array(joint_names),
                    body_names=np.array(body_names),
                )
    manifest = {"source": str(args.motion_file), "clips": names}
    (args.out / "clips.json").write_text(json.dumps(manifest, indent=1))
    print(f"exported {len(names)} clips to {args.out}")


if __name__ == "__main__":
    main()
