"""Run-time safety checks: any failure switches the robot to damping (kp = 0, kd > 0).

Thresholds follow SONIC's C++ deploy (|dq| > 35 rad/s) and the demo plan (PLAN.md Q4): stale
``lowstate``, joint speed, pelvis tilt, non-finite observations or actions, motor temperature,
and the operator's kill inputs (remote B / select, Quest B + Y, Ctrl-C). A stale Quest stream is
not a fault: the runtime freezes the targets and releases the sticks instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Limits:
    state_age_s: float = 0.04  # lowstate is 500 Hz; two missed policy ticks = link lost
    joint_speed: float = 35.0  # rad/s (SONIC deploy)
    tilt_deg: float = 45.0
    temperature_c: float = 90.0
    quest_stale_s: float = 0.5


def tilt_deg(q_wxyz: np.ndarray) -> float:
    """Angle between the pelvis z axis and the world z axis."""
    w, x, y, _ = q_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


class Watchdog:
    def __init__(self, limits: Limits | None = None):
        self.limits = limits or Limits()
        self.reason = ""

    def check(self, state: dict, obs=None, action=None, state_age: float = 0.0) -> str:
        """Empty string if safe, else the first failed check (also kept in ``reason``)."""
        lim = self.limits
        checks = (
            (state_age > lim.state_age_s, f"lowstate {1e3 * state_age:.0f} ms old"),
            (not np.isfinite(state["q"]).all() or not np.isfinite(state["quat"]).all(), "non-finite state"),
            (np.abs(state["dq"]).max() > lim.joint_speed, f"joint speed {np.abs(state['dq']).max():.1f} rad/s"),
            (tilt_deg(state["quat"]) > lim.tilt_deg, f"tilt {tilt_deg(state['quat']):.0f} deg"),
            ("temperature" in state and state["temperature"].max() >= lim.temperature_c,
             f"motor temperature {state.get('temperature', np.zeros(1)).max():.0f} C"),
            (obs is not None and not np.isfinite(obs).all(), "non-finite observation"),
            (action is not None and not np.isfinite(action).all(), "non-finite action"),
        )  # fmt: skip
        for failed, why in checks:
            if failed:
                self.reason = why
                return why
        remote = state.get("remote") or {}
        if remote.get("B") or remote.get("select"):
            self.reason = "remote kill (B / select)"
            return self.reason
        return ""
