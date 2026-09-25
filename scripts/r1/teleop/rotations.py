"""Quaternion helpers for the R1 runtime, numpy only. Quaternions are (w, x, y, z), as in Isaac Lab.

Each function takes arrays of shape (..., 4) / (..., 3) and matches the torch function the
training environment uses (``isaaclab.utils.math``, ``gear_sonic.trl.utils.torch_transform``).
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


inverse = conjugate  # unit quaternions


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by ``q`` (``isaaclab.utils.math.quat_apply``)."""
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(normalize(q), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def heading(q: np.ndarray) -> np.ndarray:
    """Yaw part as training computes it (``torch_transform.get_heading_q``): zero x, y; normalize."""
    q = np.array(q, dtype=np.float64)
    q[..., 1:3] = 0.0
    return normalize(q)


def from_yaw(yaw: np.ndarray | float) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack(
        [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1
    )


def yaw_of(q: np.ndarray) -> np.ndarray:
    """Angle of :func:`heading` (rad)."""
    h = heading(q)
    return 2.0 * np.arctan2(h[..., 3], h[..., 0])


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [1, 2, 3, 0]]


def positive_w(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)
