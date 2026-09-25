"""Quest poses -> the teleop encoder's VR 3-point targets, projected onto what the R1 can reach.

Calibration (operator standing in the R1's default pose, button press) fixes an operator frame:
the headset's heading and the controllers' poses. Afterwards:

* hand positions: the robot's default palm points plus the controllers' displacement relative to
  the headset, in the calibration heading frame, times ``scale`` (arm-length ratio; one-shot
  calibration as in SPOT 2609.07933 / ExtremControl 2602.11321). The frame does not follow the
  live head yaw, so looking around does not move the hands;
* hand orientations: the controllers' world-frame rotation since calibration, applied to the
  robot's default wrist orientations (no assumption that grip and wrist frames align);
* head: the headset's yaw and roll since calibration, clamped, drive the waist (the R1 has no
  waist pitch, so a free head position would lie outside the training data; SONIC also derives
  the head point from the head orientation).

The R1's arms have 5 DOF and no wrist pitch/yaw, so a raw controller orientation is often
unreachable. A damped-least-squares IK on the upper body (pelvis fixed) projects the targets; the
policy then gets the forward kinematics of that solution: palm points on the wrist roll links,
the head point on the torso, and the body orientations, all in the pelvis frame, as in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rotations as rot  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/r1_dex3.xml"
sys.path.insert(0, str(REPO))
from gear_sonic.utils.embodiment import r1_spec  # noqa: E402

ARMS = {
    side: [f"{side}_{j}_joint" for j in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
    for side in ("left", "right")
}  # fmt: skip


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def log_so3(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-7:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - angle < 1e-4:  # near pi: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        axis *= np.sign(v + (v == 0))
        return angle * axis / np.linalg.norm(axis)
    return angle * v / (2.0 * np.sin(angle))


class UpperBodyIK:
    """R1 upper body with the pelvis fixed at the origin (the reference pelvis frame)."""

    def __init__(self, orientation_weight: float = 0.2, damping: float = 0.05):
        self.m = mujoco.MjModel.from_xml_path(str(MJCF))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.w_ori, self.damping = orientation_weight, damping
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.q0 = np.zeros(m.nq)
        self.q0[3] = 1.0
        for j in range(1, m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            for pattern, value in r1_spec.INIT_JOINT_POS.items():
                if re.fullmatch(pattern, name):
                    self.q0[m.jnt_qposadr[j]] = value
        self.arm_q = {s: np.array([m.jnt_qposadr[jid(n)] for n in ARMS[s]]) for s in ARMS}
        self.arm_v = {s: np.array([m.jnt_dofadr[jid(n)] for n in ARMS[s]]) for s in ARMS}
        self.arm_lo = {s: m.jnt_range[[jid(n) for n in ARMS[s]], 0] for s in ARMS}
        self.arm_hi = {s: m.jnt_range[[jid(n) for n in ARMS[s]], 1] for s in ARMS}
        self.waist_q = {n: m.jnt_qposadr[jid(f"waist_{n}_joint")] for n in ("roll", "yaw")}
        self.waist_range = {n: m.jnt_range[jid(f"waist_{n}_joint")] for n in ("roll", "yaw")}
        self.wrist = {s: bid(f"{s}_wrist_roll_link") for s in ARMS}
        self.torso = bid("torso_link")
        self.palm_offset = {
            "left": np.array(r1_spec.LEFT_HAND_POINT_OFFSET),
            "right": np.array(r1_spec.RIGHT_HAND_POINT_OFFSET),
        }
        self.head_offset = np.array(r1_spec.VR_HEAD_POINT_OFFSET)
        self.q = self.q0.copy()
        self._jacp, self._jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    # -- kinematics --------------------------------------------------------------------
    def _forward(self, q: np.ndarray) -> None:
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.wrist[side]
        R = self.d.xmat[b].reshape(3, 3).copy()  # a copy: d.xmat changes with the next FK call
        return self.d.xpos[b] + R @ self.palm_offset[side], R

    def fk(self, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Palm points, wrist rotations, head point and torso rotation in the pelvis frame."""
        self._forward(self.q if q is None else q)
        out = {}
        for side in ARMS:
            out[f"{side}_palm"], out[f"{side}_wrist_R"] = self.palm(side)
        R_torso = self.d.xmat[self.torso].reshape(3, 3).copy()
        out["head"] = self.d.xpos[self.torso] + R_torso @ self.head_offset
        out["torso_R"] = R_torso
        return out

    # -- solve -------------------------------------------------------------------------
    def set_waist(self, roll: float, yaw: float) -> None:
        for name, value in (("roll", roll), ("yaw", yaw)):
            lo, hi = self.waist_range[name]
            self.q[self.waist_q[name]] = np.clip(value, lo, hi)

    def solve_arm(self, side: str, palm_target: np.ndarray, R_target: np.ndarray, iters: int = 8):
        qa, va = self.arm_q[side], self.arm_v[side]
        for _ in range(iters):
            self._forward(self.q)
            p, R = self.palm(side)
            err = np.concatenate([palm_target - p, self.w_ori * log_so3(R_target @ R.T)])
            mujoco.mj_jac(self.m, self.d, self._jacp, self._jacr, p, self.wrist[side])
            J = np.vstack([self._jacp[:, va], self.w_ori * self._jacr[:, va]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), err)
            self.q[qa] = np.clip(self.q[qa] + dq, self.arm_lo[side], self.arm_hi[side])

    def reset(self) -> None:
        self.q = self.q0.copy()


@dataclass
class Calibration:
    """Operator reference captured in the calibration pose (robot-convention world axes)."""

    head: np.ndarray  # 4x4
    left: np.ndarray
    right: np.ndarray

    @property
    def yaw(self) -> float:
        forward = self.head[:3, :3] @ np.array([0.0, 0.0, -1.0])  # WebXR device forward = -z
        return float(np.arctan2(forward[1], forward[0]))


class VRTargets:
    """Calibrated Quest poses -> IK-projected VR 3-point observation terms."""

    def __init__(
        self,
        scale: float = 0.65,
        max_head_yaw: float = 0.6,
        max_head_roll: float = 0.25,
        waist_gain: float = 1.0,
    ):
        self.ik = UpperBodyIK()
        self.scale, self.max_yaw, self.max_roll = scale, max_head_yaw, max_head_roll
        self.waist_gain = waist_gain
        default = self.ik.fk(self.ik.q0)
        self.palm0 = {s: default[f"{s}_palm"].copy() for s in ARMS}
        self.wrist_R0 = {s: default[f"{s}_wrist_R"].copy() for s in ARMS}
        self.calib: Calibration | None = None

    def calibrate(self, head: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self.calib = Calibration(head.copy(), left.copy(), right.copy())
        self.ik.reset()

    def solve(self, head: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
        """IK solution for the current Quest poses (calibrated); returns :meth:`terms`."""
        c = self.calib
        assert c is not None, "calibrate() first"
        to_op = yaw_matrix(-c.yaw)  # world -> operator heading frame (x forward, z up)
        # head: rotation since calibration in the operator frame -> clamped yaw / roll on the waist
        dR_head = to_op @ head[:3, :3] @ c.head[:3, :3].T @ to_op.T
        yaw = np.clip(np.arctan2(dR_head[1, 0], dR_head[0, 0]), -self.max_yaw, self.max_yaw)
        roll = np.clip(np.arctan2(dR_head[2, 1], dR_head[2, 2]), -self.max_roll, self.max_roll)
        self.ik.set_waist(self.waist_gain * roll, self.waist_gain * yaw)
        for side, now, ref in (("left", left, c.left), ("right", right, c.right)):
            disp = (now[:3, 3] - head[:3, 3]) - (ref[:3, 3] - c.head[:3, 3])
            target = self.palm0[side] + self.scale * (to_op @ disp)
            dR = to_op @ now[:3, :3] @ ref[:3, :3].T @ to_op.T
            self.ik.solve_arm(side, target, dR @ self.wrist_R0[side])
        return self.terms()

    def terms(self, q: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """``vr_3point_local_target`` (9) and ``vr_3point_local_orn_target`` (12), w >= 0."""
        f = self.ik.fk(q)
        pos = np.concatenate([f["left_palm"], f["right_palm"], f["head"]])
        quats = [f["left_wrist_R"], f["right_wrist_R"], f["torso_R"]]
        quat = np.concatenate([rot.positive_w(matrix_to_quat(R)) for R in quats])
        return {
            "vr_3point_local_target": pos,
            "vr_3point_local_orn_target": quat,
            "q": self.ik.q.copy(),
        }


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q
