"""Drive the robot from the terminal keyboard, without a Quest: walk by velocity command, gesture.

A virtual operator in the calibration pose (as ``run.py``'s scripted operator) whose thumbsticks
and hands follow keys typed in the terminal running ``run.py``:

    w / s       faster forward / backward (left stick, 4 presses to full speed)
    a / d       sidestep left / right
    q / e       turn left / right (right stick; again to turn faster)
    space       stop: sticks centered (the planner stands; facing is kept)
    1 2 3 4 5   hands: rest, wave (right), point (right), both hands up, reach forward
    g / t / o   fingers of both hands: point / fist / open (toggles; only with run.py --hands)
    Ctrl-C      quit

The sticks drive SONIC's kinematic planner exactly as the Quest's do (``reference.StickState``):
the left stick is a velocity command (direction relative to the robot's facing, speed within the
slow-walk band), the right stick a yaw rate.

    python scripts/r1/teleop/run.py --onnx <export>.onnx --input keys --viewer --seconds 3600
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
from quest import FACING_X, QuestSample, device_pose

STEP = 0.25  # stick change per key press
HOLD_S = 0.8  # gesture transition time
# operator hand displacements (m, robot-convention world: x forward, y left, z up) per gesture
GESTURES = {
    "rest": {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
    "wave": {"left": (0.0, 0.0, 0.0), "right": (0.10, -0.12, 0.72)},
    "point": {"left": (0.0, 0.0, 0.0), "right": (0.38, 0.04, 0.40)},
    "up": {"left": (0.0, 0.06, 0.85), "right": (0.0, -0.06, 0.85)},
    "reach": {"left": (0.35, -0.02, 0.22), "right": (0.35, 0.02, 0.22)},
}
KEYS_GESTURE = {"1": "rest", "2": "wave", "3": "point", "4": "up", "5": "reach"}


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)  # minimum-jerk profile


class KeyboardOperator:
    def __init__(self, engage_at: float = 0.0):
        if not sys.stdin.isatty():
            raise RuntimeError("--input keys reads the terminal: run run.py in a terminal")
        self.engage_at = engage_at
        self.head0 = np.array([0.0, 0.0, 1.60])
        self.hand0 = {"left": np.array([0.30, 0.22, 1.02]), "right": np.array([0.30, -0.22, 1.02])}
        self.left_stick, self.right_stick = np.zeros(2), np.zeros(2)
        self.fingers = ""  # "", "g" (point), "t" (fist), "o" (open)
        self.gesture, self.previous, self.t_gesture = "rest", "rest", -HOLD_S
        self.t = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_keys, daemon=True, name="keys")
        self.thread.start()
        print(__doc__.split("\n\n")[1], flush=True)

    # -- keyboard thread ----------------------------------------------------------------
    def _read_keys(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.running:
                key = sys.stdin.read(1)
                if key:
                    self._press(key.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _press(self, key: str) -> None:
        with self.lock:
            ls, rs = self.left_stick, self.right_stick
            moves = {"w": (1, STEP), "s": (1, -STEP), "a": (0, -STEP), "d": (0, STEP)}
            if key in moves:
                axis, delta = moves[key]
                ls[axis] = float(np.clip(ls[axis] + delta, -1.0, 1.0))
            elif key in "qe":
                rs[0] = float(np.clip(rs[0] + (-2 * STEP if key == "q" else 2 * STEP), -1, 1))
            elif key == " ":
                ls[:], rs[:] = 0.0, 0.0
            elif key in KEYS_GESTURE:
                self.previous = self._current_gesture_name()
                self.gesture, self.t_gesture = KEYS_GESTURE[key], self.t
            elif key in "gto":
                self.fingers = "" if self.fingers == key else key
            else:
                return
            fingers = {"": "hold", "g": "point", "t": "fist", "o": "open"}[self.fingers]
            print(
                f"\r[keys] left stick {ls.round(2)}  right stick {rs.round(2)}  "
                f"hands {self.gesture}  fingers {fingers}      ",
                end="",
                flush=True,
            )

    def _current_gesture_name(self) -> str:
        return self.gesture if self.t - self.t_gesture >= HOLD_S else self.previous

    # -- the operator -------------------------------------------------------------------
    def hands(self, t: float) -> dict[str, np.ndarray]:
        a = _smooth((t - self.t_gesture) / HOLD_S)
        out = {}
        for side in ("left", "right"):
            d0 = np.array(GESTURES[self.previous][side])
            d1 = np.array(GESTURES[self.gesture][side])
            out[side] = self.hand0[side] + (1 - a) * d0 + a * d1
        if self.gesture == "wave" and a >= 1.0:  # side to side at 1.2 Hz
            out["right"][1] += 0.10 * np.sin(2 * np.pi * 1.2 * (t - self.t_gesture - HOLD_S))
        return out

    def read(self, t: float) -> QuestSample:
        press = self.engage_at <= t < self.engage_at + 0.2
        with self.lock:
            self.t = t
            hands = self.hands(t)
            ls, rs, fingers = self.left_stick.copy(), self.right_stick.copy(), self.fingers
        grip = 1.0 if fingers in ("g", "t") else 0.0
        trigger = 1.0 if fingers == "t" else 0.0
        buttons = {"A": press, "X": press}
        if fingers == "o" and not press:
            buttons = {"A": True, "X": True} if t > self.engage_at + 1.0 else buttons
        return QuestSample(
            t=time.monotonic(), head=device_pose(self.head0, FACING_X),
            left=device_pose(hands["left"], FACING_X), right=device_pose(hands["right"], FACING_X),
            left_stick=ls, right_stick=rs, left_trigger=trigger, right_trigger=trigger,
            left_grip=grip, right_grip=grip, buttons=buttons,
        )  # fmt: skip

    def close(self) -> None:
        self.running = False
