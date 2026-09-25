"""The lower-body reference at run time: SONIC's kinematic planner driven by the thumbsticks.

SONIC's ``VR_3PT`` mode feeds the teleop encoder the planner's legs while the VR 3 points drive the
upper body. Per 50 Hz control tick this module returns what the tokenizer needs from the planner:

* ``command_multi_future_lower_body`` (240): the R1 legs at the cursor and every 0.1 s after it
  (10 frames; the plan's last frame repeats past its end, as the motion library clamps at a
  clip's end), 120 positions then 120 velocities (50 Hz forward differences), MuJoCo leg order;
* the reference pelvis orientation for ``motion_anchor_ori_heading``, rotated into the robot
  IMU's yaw frame: q_ref = delta * q_planner with delta = heading(q_imu) heading(q_planner)^-1
  taken when control engages (otherwise the robot turns to the planner's initial yaw).

The planner outputs G1 motion. Legs map to the R1 exactly as the training clips were made
(``transfer_g1_motion_lib_to_r1.py``: joint by name, clamped to the R1 joint ranges; the motion
library leaves 50 Hz joint angles untouched, verified in PLAN.md Q1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/r1"))
sys.path.insert(0, str(REPO / "gear_sonic/data_process"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import planner_loop as pl  # noqa: E402
import rotations as rot  # noqa: E402

FRAME_SKIP = 5  # 0.1 s at 50 Hz
NUM_FUTURE = 10
NUM_LEGS = 12  # R1 MuJoCo order starts with the 12 leg joints


class LegMap:
    """G1 planner joints -> R1 joints (MuJoCo order), the training transfer's map and clamp."""

    def __init__(self):
        from transfer_g1_motion_lib_to_r1 import G1ToR1Transfer

        t = G1ToR1Transfer(contact_fix=False)
        assert tuple(t.g1_joints) == pl.G1_JOINTS, "planner qpos order != G1 MJCF order"
        self.index, self.lower, self.upper = t.g1_to_r1_idx, t.lower, t.upper
        self.r1_joints = list(t.r1_joints)

    def __call__(self, g1_dof: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(g1_dof)[..., self.index], self.lower, self.upper)


@dataclass
class StickState:
    """SONIC ``gamepad.hpp`` planner mode, continuous: sticks in [-1, 1], up = +y."""

    facing: float = 0.0  # rad, integrated from the right stick
    max_yaw_rate: float = 1.0  # rad/s at full right-stick deflection
    dead_zone: float = 0.15
    mode_moving: int = pl.SLOW_WALK
    speed_range: tuple[float, float] = (0.2, 0.8)  # m/s over the stick deflection (slow walk band)

    def command(self, lx: float, ly: float, rx: float, dt: float) -> pl.Command:
        if abs(rx) > self.dead_zone:
            self.facing -= self.max_yaw_rate * rx * dt  # right = clockwise, as gamepad.hpp
        face = np.array([np.cos(self.facing), np.sin(self.facing), 0.0])
        mag = min(1.0, float(np.hypot(lx, ly)))
        if mag <= self.dead_zone:  # released: stand (and turn in place)
            return pl.Command(mode=pl.IDLE, face_dir=face)
        direction = np.arctan2(ly, lx) - np.pi / 2 + self.facing  # up = forward
        lo, hi = self.speed_range
        speed = lo + (hi - lo) * (mag - self.dead_zone) / (1.0 - self.dead_zone)
        move = np.array([np.cos(direction), np.sin(direction), 0.0])
        return pl.Command(mode=self.mode_moving, speed=float(speed), move_dir=move, face_dir=face)


class PlannerReference:
    """``hold``: stand in the default pose until the sticks first move (``PlannerLoop.hold``)."""

    def __init__(
        self,
        planner_onnx: str | None = None,
        device: str = "cpu",
        threads: int = 4,
        hold: bool = True,
    ):
        onnx = planner_onnx or str(pl.find_planner_onnx())
        self.model = pl.PlannerModel(onnx, threads=threads, device=device)
        self.loop = pl.AsyncPlannerLoop(self.model)
        if hold:
            self.loop.hold()
        self.legs = LegMap()
        self.delta = np.array([1.0, 0.0, 0.0, 0.0])

    def engage(self, q_imu: np.ndarray) -> None:
        """Align the planner's yaw with the robot's (call when control starts)."""
        q_plan = self.loop.motion[min(self.loop.cur, len(self.loop.motion) - 1), 3:7]
        self.delta = rot.multiply(rot.heading(q_imu), rot.inverse(rot.heading(q_plan)))

    def step(self, cmd: pl.Command) -> dict[str, np.ndarray]:
        """Advance one 50 Hz tick; returns the tokenizer's lower-body term and the reference."""
        frame = self.loop.tick(cmd)  # may merge a plan (rebasing the buffer) before reading
        motion = self.loop.motion
        idx = (self.loop.cur - 1) + FRAME_SKIP * np.arange(NUM_FUTURE)  # frame 0 = ``frame``
        last = len(motion) - 1
        q = self.legs(motion[np.minimum(idx, last), 7:])[:, :NUM_LEGS]
        q_next = self.legs(motion[np.minimum(idx + 1, last), 7:])[:, :NUM_LEGS]
        q_prev = self.legs(motion[np.minimum(idx - 1, last).clip(0), 7:])[:, :NUM_LEGS]
        # forward difference; at the plan's end the motion library repeats the last difference
        at_end = (idx + 1 > last)[:, None]
        dq = np.where(at_end, q - q_prev, q_next - q) * 50.0
        return {
            "command_multi_future_lower_body": np.concatenate([q.ravel(), dq.ravel()]),
            "ref_root_quat": rot.multiply(self.delta, frame[3:7]),
            "g1_qpos": frame,
        }

    def wait(self) -> None:
        """Block until a pending plan is ready (an instantaneous planner, for offline sim tests)."""
        if self.loop._pending is not None:
            self.loop._pending[0].result()

    def close(self) -> None:
        self.loop.close()
