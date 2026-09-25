"""The teleop policy's observation, built exactly as the Isaac Lab training environment builds it.

The exported ONNX takes one flat vector (``export_teleop_onnx.py`` writes the layout next to it):

* tokenizer terms, current frame only, in ``meta["input"]["tokenizer"]`` order:
  ``motion_anchor_ori_heading`` (6), ``command_multi_future_lower_body`` (240),
  ``vr_3point_local_target`` (9), ``vr_3point_local_orn_target`` (12);
* actor terms, each with its own history of ``history`` steps, oldest first, in
  ``meta["input"]["actor_obs"]`` order: ``base_ang_vel`` (3), ``joint_pos`` (24), ``joint_vel``
  (24), ``actions`` (24), ``gravity_dir`` (3).

Term definitions (``gear_sonic/envs/manager_env/mdp/observations.py`` / ``commands.py``):

* ``motion_anchor_ori_heading``: first two columns of R(heading(q_robot)^-1 * q_ref), row by row;
  heading() zeroes the quaternion's x, y and renormalizes (``torch_transform.get_heading_q``).
* ``command_multi_future_lower_body``: the reference's 12 leg joints (MuJoCo order) at the current
  frame and every 0.1 s after it (10 frames), 120 positions then 120 velocities.
* ``vr_3point_local_target`` / ``_orn_target``: the VR 3 points (palm points on the wrist roll
  links and the head point on the torso) relative to the *reference* pelvis, in its full frame.
* ``base_ang_vel``: pelvis angular velocity in the pelvis frame (IMU gyro); ``joint_pos``: q minus
  the default pose; ``joint_vel``: dq; ``actions``: the last raw action (clipped to +-20);
  ``gravity_dir``: R(q_pelvis)^T [0, 0, -1]. All in Isaac Lab joint order.

Isaac Lab's history buffer fills every slot with the first value after a reset (the action term's
first value is 0), and the policy's observation noise is a training-only corruption.
"""

from __future__ import annotations

import numpy as np

try:
    from . import rotations as rot
except ImportError:  # run as a script
    import rotations as rot

DOWN = np.array([0.0, 0.0, -1.0])


def anchor_ori_heading(q_robot: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """(6,) reference pelvis orientation seen from the robot pelvis' heading frame."""
    rel = rot.multiply(rot.inverse(rot.heading(q_robot)), q_ref)
    return rot.to_matrix(rel)[..., :2].reshape(*rel.shape[:-1], 6)


def lower_body_command(joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """(240,) from (10, 12) future leg positions and velocities (MuJoCo leg order)."""
    return np.concatenate([np.ravel(joint_pos), np.ravel(joint_vel)])


def vr_3point_local(
    anchor_pos: np.ndarray, anchor_quat: np.ndarray, points_pos: np.ndarray, points_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(9,), (12,): world 3-point poses relative to the reference pelvis (full frame)."""
    inv = rot.inverse(anchor_quat)
    pos = rot.apply(inv[None], np.asarray(points_pos) - np.asarray(anchor_pos)[None])
    quat = rot.multiply(inv[None], np.asarray(points_quat))
    return pos.ravel(), quat.ravel()


def gravity_dir(q_pelvis: np.ndarray) -> np.ndarray:
    return rot.apply(rot.inverse(q_pelvis), DOWN)


class ObservationBuilder:
    """Assembles the flat ONNX input and keeps the actor terms' histories."""

    def __init__(self, meta: dict):
        self.tokenizer_terms = [(t["name"], int(t["dim"])) for t in meta["input"]["tokenizer"]]
        self.actor_terms = [
            (t["name"], int(t["dim"]), int(t["history"])) for t in meta["input"]["actor_obs"]
        ]
        self.size = int(meta["input"]["size"])
        self.default_joint_pos = np.asarray(meta["default_joint_pos"], dtype=np.float64)
        self.history: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        """Start a new episode: the next :meth:`build` fills every history slot."""
        self.history = None

    def proprioception(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        q_pelvis: np.ndarray,
        gyro: np.ndarray,
        last_action: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Actor terms for one step (joint arrays in Isaac Lab order)."""
        return {
            "base_ang_vel": np.asarray(gyro, dtype=np.float64),
            "joint_pos": np.asarray(joint_pos, dtype=np.float64) - self.default_joint_pos,
            "joint_vel": np.asarray(joint_vel, dtype=np.float64),
            "actions": np.asarray(last_action, dtype=np.float64),
            "gravity_dir": gravity_dir(q_pelvis),
        }

    def build(self, proprio: dict[str, np.ndarray], tokenizer: dict[str, np.ndarray]) -> np.ndarray:
        if self.history is None:
            self.history = {
                name: np.repeat(np.asarray(proprio[name], np.float64)[None], hist, axis=0)
                for name, _, hist in self.actor_terms
            }
        else:
            for name, _, _ in self.actor_terms:
                buf = self.history[name]
                buf[:-1] = buf[1:]
                buf[-1] = proprio[name]
        parts = []
        for name, dim in self.tokenizer_terms:
            value = np.asarray(tokenizer[name], dtype=np.float64).ravel()
            assert value.size == dim, (name, value.size, dim)
            parts.append(value)
        for name, dim, _ in self.actor_terms:
            assert self.history[name].shape[1] == dim, name
            parts.append(self.history[name].ravel())
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.size == self.size, (obs.size, self.size)
        return obs
