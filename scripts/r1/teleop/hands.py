"""Dex3-1 finger targets: semi-closed by default, grasp and point from the Quest controller.

The policy does not move the fingers. Its model has them fixed in ``r1_spec.DEX3_HOLD_POSE`` (a
relaxed, semi-closed fist), and the real hands hold that pose while the arms move. The operator
shapes each hand with the controller as in Meta's hand presence:

* nothing pressed: hold (semi-closed);
* grip (middle finger): point: middle finger and thumb close, the index stays out;
* grip + trigger: fist;
* trigger alone: pinch: thumb and index close more than the middle finger;
* A (right) / X (left) held while engaged: open flat hand (a waving "hello").

Poses blend continuously with the trigger and grip values and reach the hands through a
first-order filter (``tau``), so a button never snaps the fingers.

The fingers' DDS topics (unitree_sdk2, xr_teleoperate ``robot_hand_unitree.py``) are
``rt/dex3/{left,right}/cmd`` with a ``unitree_hg`` ``HandCmd_`` of 7 motors. Each motor gets
``mode`` = RIS byte (motor id | status 0x01 << 4 | timeout << 7), kp 1.5, kd 0.2. The motor
order differs per hand (``MOTOR_JOINTS``). VERIFY on the real hands: order, signs and the
look of each pose.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from gear_sonic.utils.embodiment import r1_spec  # noqa: E402

FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
MOTOR_JOINTS = {  # HandCmd_ motor order (xr_teleoperate Dex3_1_{Left,Right}_JointIndex)
    "left": ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
    "right": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
}
KP, KD = 1.5, 0.2

# Left hand; unitree_ros dex3_1_l.urdf limits: thumb_0 +-1.05, thumb_1 [-0.61, 1.05],
# thumb_2 [0, 1.75], middle/index_0 [-1.57, 0], middle/index_1 [-1.75, 0] (closing is negative).
# The right hand mirrors every joint but thumb_0.
_LEFT = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "fist": (0.0, 0.5, 1.1, -1.35, -1.45, -1.35, -1.45),
    "point": (0.0, 0.5, 1.1, -1.35, -1.45, -0.1, -0.1),
    "pinch": (0.0, 0.5, 1.0, -0.8, -0.9, -1.1, -1.0),
}


def pose(name: str, side: str) -> np.ndarray:
    """A named pose as 7 joint angles in :data:`FINGERS` order."""
    if name == "hold":
        return np.array([r1_spec.DEX3_HOLD_POSE[f"{side}_hand_{f}_joint"] for f in FINGERS])
    q = np.array(_LEFT[name], dtype=float)
    return q if side == "left" else q * np.array([1.0, -1, -1, -1, -1, -1, -1])


def to_motor_order(q: np.ndarray, side: str) -> np.ndarray:
    """:data:`FINGERS` order -> the hand's ``HandCmd_`` motor order."""
    return np.asarray(q)[[FINGERS.index(j) for j in MOTOR_JOINTS[side]]]


def ris_mode(motor: int, status: int = 0x01, timeout: int = 0) -> int:
    """The Dex3 motor ``mode`` byte: id (4 bits) | status (3 bits) << 4 | timeout (1 bit) << 7."""
    return (motor & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


class HandShaper:
    """Controller inputs -> per-hand finger targets (:data:`FINGERS` order), filtered."""

    def __init__(self, tau: float = 0.1):
        self.tau = tau
        self.poses = {s: {n: pose(n, s) for n in ("hold", "open", *_LEFT)} for s in MOTOR_JOINTS}
        self.q = {s: self.poses[s]["hold"].copy() for s in MOTOR_JOINTS}

    def target(self, side: str, trigger: float, grip: float, open_hand: bool = False):
        p = self.poses[side]
        t, g = float(np.clip(trigger, 0, 1)), float(np.clip(grip, 0, 1))
        base = p["hold"] + g * (p["point"] - p["hold"])  # grip: point
        closed = p["pinch"] + g * (p["fist"] - p["pinch"])  # trigger: pinch, with grip: fist
        q = base + t * (closed - base)
        return p["open"].copy() if open_hand else q

    def __call__(self, sample, dt: float, engaged: bool = True) -> dict[str, np.ndarray]:
        """Filtered targets for both hands from a ``QuestSample`` (hold until engaged)."""
        a = 1.0 - np.exp(-dt / self.tau) if self.tau > 0 else 1.0
        for side, trig, grip, button in (
            ("left", sample.left_trigger, sample.left_grip, "X"),
            ("right", sample.right_trigger, sample.right_grip, "A"),
        ):
            if engaged:
                goal = self.target(side, trig, grip, bool(sample.buttons.get(button)))
            else:
                goal = self.poses[side]["hold"]
            self.q[side] = self.q[side] + a * (goal - self.q[side])
        return {s: q.copy() for s, q in self.q.items()}
