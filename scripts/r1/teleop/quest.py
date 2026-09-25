"""Meta Quest 3 input through televuer (Unitree xr_teleoperate's WebXR front end), controllers only.

The Quest's browser opens televuer's page (no app): over Wi-Fi ``https://<pc>:8012/?ws=wss://<pc>:8012``,
or over USB after ``adb reverse tcp:8012 tcp:8012`` at ``https://localhost:8012/?ws=wss://localhost:8012``.
A self-signed certificate (``openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem
-out cert.pem``) goes in ``~/.config/xr_teleoperate/``. Display mode ``pass-through``: the
operator sees the room and the robot.

Poses arrive in WebXR's frame (right-handed, x right, y up, z backward). :func:`xr_to_robot`
converts them to the robot convention (x forward, y left, z up); nothing else here interprets
them (``calibration.py`` does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

# WebXR (x right, y up, z back) -> robot (x forward, y left, z up)
XR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# WebXR device axes in robot-convention world coordinates for a device facing +x (forward = -z)
FACING_X = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def device_pose(pos, R=None) -> np.ndarray:
    """4x4 pose of a virtual device (scripted and keyboard operators)."""
    m = np.eye(4)
    m[:3, :3] = np.eye(3) if R is None else R
    m[:3, 3] = pos
    return m


def xr_to_robot(pose: np.ndarray) -> np.ndarray:
    """4x4 pose of an XR device -> the same device pose with robot-convention world axes.

    The device's own axes are kept (a Quest grip frame stays the grip frame): p' = T p, R' = T R.
    """
    out = np.eye(4)
    out[:3, :3] = XR_TO_ROBOT @ pose[:3, :3]
    out[:3, 3] = XR_TO_ROBOT @ pose[:3, 3]
    return out


@dataclass
class QuestSample:
    t: float
    head: np.ndarray  # 4x4, robot-convention world axes
    left: np.ndarray
    right: np.ndarray
    left_stick: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (x right, y up)
    right_stick: np.ndarray = field(default_factory=lambda: np.zeros(2))
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
    buttons: dict = field(default_factory=dict)  # A, B (right), X, Y (left), stick clicks

    @property
    def valid(self) -> bool:
        """False until the headset and both controllers have reported a pose."""
        return all(
            abs(np.linalg.det(m[:3, :3]) - 1.0) < 1e-2 for m in (self.head, self.left, self.right)
        )


class QuestInput:
    def __init__(self, cert_file: str | None = None, key_file: str | None = None):
        from televuer import TeleVuer

        self.tv = TeleVuer(
            use_hand_tracking=False,
            binocular=False,
            img_shape=(480, 640),
            display_mode="pass-through",
            cert_file=cert_file,
            key_file=key_file,
        )

    def read(self) -> QuestSample:
        tv = self.tv
        # WebXR thumbsticks: x right, y *down*; flip y so pushing forward is +1
        ls = np.array(tv.left_ctrl_thumbstickValue_shared[:]) * np.array([1.0, -1.0])
        rs = np.array(tv.right_ctrl_thumbstickValue_shared[:]) * np.array([1.0, -1.0])
        return QuestSample(
            t=time.monotonic(),
            head=xr_to_robot(tv.head_pose),
            left=xr_to_robot(tv.left_arm_pose),
            right=xr_to_robot(tv.right_arm_pose),
            left_stick=ls,
            right_stick=rs,
            left_trigger=float(tv.left_ctrl_triggerValue_shared.value),
            right_trigger=float(tv.right_ctrl_triggerValue_shared.value),
            left_grip=float(tv.left_ctrl_squeezeValue_shared.value),
            right_grip=float(tv.right_ctrl_squeezeValue_shared.value),
            buttons={
                "A": bool(tv.right_ctrl_aButton_shared.value),
                "B": bool(tv.right_ctrl_bButton_shared.value),
                "X": bool(tv.left_ctrl_aButton_shared.value),
                "Y": bool(tv.left_ctrl_bButton_shared.value),
                "LS": bool(tv.left_ctrl_thumbstick_shared.value),
                "RS": bool(tv.right_ctrl_thumbstick_shared.value),
            },
        )

    def close(self) -> None:
        self.tv.close()
