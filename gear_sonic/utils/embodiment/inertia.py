"""Rigid-body inertia utilities for fusing fixed URDF subtrees into a single MJCF body.

Used to add the Dex3-1 hand mass to the R1 wrist bodies (and to validate the approach
against mjlab's fused knee/torso bodies). Pure numpy/scipy; no simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class RigidBody:
    """Mass properties of one rigid body expressed in a common reference frame."""

    mass: float
    com: np.ndarray  # (3,) position of the centre of mass in the reference frame
    inertia_com: np.ndarray  # (3,3) inertia tensor about the COM, axes of the reference frame

    def transformed(self, rot: np.ndarray, trans: np.ndarray) -> "RigidBody":
        """Express this body in a parent frame: p_parent = rot @ p_local + trans."""
        return RigidBody(
            mass=self.mass,
            com=rot @ self.com + trans,
            inertia_com=rot @ self.inertia_com @ rot.T,
        )


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF roll-pitch-yaw (extrinsic x-y-z, i.e. R = Rz(y) Ry(p) Rx(r)) to a rotation matrix."""
    r, p, y = rpy
    return R.from_euler("xyz", [r, p, y]).as_matrix()


def urdf_inertial_to_body(inertial_el: ET.Element) -> RigidBody:
    """Parse a URDF ``<inertial>`` element into a RigidBody expressed in the link frame."""
    origin = inertial_el.find("origin")
    xyz = (
        np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        if origin is not None
        else np.zeros(3)
    )
    rpy = (
        [float(v) for v in origin.get("rpy", "0 0 0").split()]
        if origin is not None
        else [0.0, 0.0, 0.0]
    )
    mass = float(inertial_el.find("mass").get("value"))
    i = inertial_el.find("inertia")
    ixx, iyy, izz = (float(i.get(k)) for k in ("ixx", "iyy", "izz"))
    ixy, ixz, iyz = (float(i.get(k, "0")) for k in ("ixy", "ixz", "iyz"))
    inertia_local = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot = rpy_to_matrix(rpy)
    return RigidBody(mass=mass, com=xyz, inertia_com=rot @ inertia_local @ rot.T)


def mjcf_inertial_to_body(inertial_el: ET.Element) -> RigidBody:
    """Parse a MuJoCo ``<inertial pos quat mass diaginertia>`` element (quat is wxyz)."""
    pos = np.array([float(v) for v in inertial_el.get("pos", "0 0 0").split()])
    w, x, y, z = (float(v) for v in inertial_el.get("quat", "1 0 0 0").split())
    rot = R.from_quat([x, y, z, w]).as_matrix()
    mass = float(inertial_el.get("mass"))
    if inertial_el.get("diaginertia") is not None:
        diag = np.array([float(v) for v in inertial_el.get("diaginertia").split()])
        inertia = rot @ np.diag(diag) @ rot.T
    else:  # fullinertia = ixx iyy izz ixy ixz iyz
        ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in inertial_el.get("fullinertia").split())
        inertia = rot @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ rot.T
    return RigidBody(mass=mass, com=pos, inertia_com=inertia)


def combine(bodies: list[RigidBody]) -> RigidBody:
    """Combine rigid bodies expressed in the same frame (parallel-axis theorem)."""
    mass = sum(b.mass for b in bodies)
    if mass <= 0:
        raise ValueError("Total mass must be positive")
    com = sum(b.mass * b.com for b in bodies) / mass
    inertia = np.zeros((3, 3))
    for b in bodies:
        d = b.com - com
        inertia += b.inertia_com + b.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return RigidBody(mass=mass, com=com, inertia_com=inertia)


def body_to_mjcf_inertial_attrs(body: RigidBody) -> dict[str, str]:
    """Return ``pos``, ``quat`` (wxyz), ``mass``, ``diaginertia`` attributes for an MJCF ``<inertial>``."""
    sym = 0.5 * (body.inertia_com + body.inertia_com.T)
    evals, evecs = np.linalg.eigh(sym)
    if np.linalg.det(evecs) < 0:  # keep a right-handed frame
        evecs[:, 2] *= -1
    x, y, z, w = R.from_matrix(evecs).as_quat()
    fmt = lambda a: " ".join(f"{v:.8g}" for v in a)  # noqa: E731
    return {
        "pos": fmt(body.com),
        "quat": fmt([w, x, y, z]),
        "mass": f"{body.mass:.8g}",
        "diaginertia": fmt(np.maximum(evals, 1e-12)),
    }
