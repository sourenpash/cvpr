"""Compute IsaacLab <-> MuJoCo joint/body ordering maps from a URDF + MJCF pair.

SONIC needs four index arrays per robot (see ``gear_sonic/envs/manager_env/robots/g1.py``):

* ``isaaclab_to_mujoco_dof``  : ``data_mj = data_il[..., isaaclab_to_mujoco_dof]``
* ``mujoco_to_isaaclab_dof``  : ``data_il = data_mj[..., mujoco_to_isaaclab_dof]``
* ``isaaclab_to_mujoco_body`` / ``mujoco_to_isaaclab_body`` : same, for the DOF-bearing bodies
  with the root body at index 0 in both orders.

Conventions (verified against the shipped G1 and H2 constants, see
``gear_sonic/tests/r1/test_ordering_matches_upstream.py``):

* **Isaac Lab order** = breadth-first traversal of the URDF kinematic tree, expanding each
  level's bodies in order and appending their children in URDF joint-definition order.
  Only DOF-bearing bodies (children of non-fixed joints) plus the root are listed.
* **MuJoCo order** = document (depth-first) order of ``<body>`` / ``<joint>`` elements in the
  MJCF ``<worldbody>``, excluding the free joint.

The Isaac Lab rule is an *inference*; the authoritative order is whatever
``Articulation.joint_names`` / ``body_names`` report after import. Always confirm on a
machine with Isaac Lab (``scripts/r1/verify_isaaclab_order.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

_DOF_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass
class UrdfTree:
    root: str
    # child link -> (joint name, joint type, parent link), in URDF joint-definition order
    joints: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(
        default_factory=dict
    )  # parent -> [child links] (URDF order)

    def isaaclab_order(self) -> tuple[list[str], list[str]]:
        """Return (dof_bodies, dof_joints) in inferred Isaac Lab order, root body first.

        The root body is included in the body list (index 0); joints exclude the root.
        """
        bodies = [self.root]
        joints: list[str] = []
        queue = deque([self.root])
        while queue:
            parent = queue.popleft()
            for child in self.children.get(parent, []):
                jname, jtype, _ = self.joints[child]
                if jtype in _DOF_JOINT_TYPES:
                    bodies.append(child)
                    joints.append(jname)
                queue.append(child)
        return bodies, joints

    def dof_joint_to_body(self) -> dict[str, str]:
        return {j: c for c, (j, t, _) in self.joints.items() if t in _DOF_JOINT_TYPES}


def parse_urdf(path: str | Path) -> UrdfTree:
    root_el = ET.parse(str(path)).getroot()
    links = [el.get("name") for el in root_el.findall("link")]
    joints: dict[str, tuple[str, str, str]] = {}
    children: dict[str, list[str]] = {}
    child_links = set()
    for j in root_el.findall("joint"):
        name, jtype = j.get("name"), j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joints[child] = (name, jtype, parent)
        children.setdefault(parent, []).append(child)
        child_links.add(child)
    roots = [name for name in links if name not in child_links]
    # A stray ``world`` link with no joints (e.g. commented-out floating joint) is harmless.
    roots = [r for r in roots if r in children] or roots
    if len(roots) != 1:
        raise ValueError(f"URDF {path}: expected exactly one root link, found {roots}")
    return UrdfTree(root=roots[0], joints=joints, children=children)


@dataclass
class MjcfTree:
    bodies: list[str]  # document order, root first
    joints: list[str]  # document order, excluding free joint
    joint_to_body: dict[str, str]
    body_to_joint: dict[str, str]
    joint_axes: dict[str, tuple[int, int, int]]
    motors: list[str]  # <actuator> children, document order


def parse_mjcf(path: str | Path) -> MjcfTree:
    root_el = ET.parse(str(path)).getroot()
    wb = root_el.find("worldbody")
    if wb is None:
        raise ValueError(f"MJCF {path}: no <worldbody>")
    bodies, joints = [], []
    joint_to_body, body_to_joint, axes = {}, {}, {}

    def visit(body_el):
        bname = body_el.get("name")
        bodies.append(bname)
        for j in body_el.findall("joint"):
            if j.get("type") == "free":
                continue
            jname = j.get("name")
            joints.append(jname)
            joint_to_body[jname] = bname
            body_to_joint[bname] = jname
            ax = tuple(int(float(a)) for a in j.get("axis", "0 0 1").split())
            axes[jname] = ax  # type: ignore[assignment]
        for child in body_el.findall("body"):
            visit(child)

    for b in wb.findall("body"):
        visit(b)
    act = root_el.find("actuator")
    motors = [m.get("joint") or m.get("name") for m in act] if act is not None else []
    return MjcfTree(bodies, joints, joint_to_body, body_to_joint, axes, motors)


@dataclass
class OrderingMaps:
    isaaclab_joints: list[
        str
    ]  # DOF-bearing body names, Isaac Lab order, root first (SONIC's misnomer)
    isaaclab_dof_names: list[str]
    mujoco_dof_names: list[str]
    mujoco_body_names: list[str]
    isaaclab_to_mujoco_dof: list[int]
    mujoco_to_isaaclab_dof: list[int]
    isaaclab_to_mujoco_body: list[int]
    mujoco_to_isaaclab_body: list[int]

    def as_dict(self) -> dict:
        return {
            "isaaclab_joints": self.isaaclab_joints,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }

    def to_python(self, prefix: str) -> str:
        """Render the constants the way ``robots/g1.py`` / ``robots/h2.py`` write them."""

        def fmt_list(name, items, quote=False):
            body = "\n".join(f'    "{x}",' if quote else f"    {x}," for x in items)
            return f"{name} = [\n{body}\n]\n"

        out = fmt_list(f"{prefix}_ISAACLAB_JOINTS", self.isaaclab_joints, quote=True)
        out += "\n" + fmt_list(f"{prefix}_ISAACLAB_TO_MUJOCO_DOF", self.isaaclab_to_mujoco_dof)
        out += "\n" + fmt_list(f"{prefix}_MUJOCO_TO_ISAACLAB_DOF", self.mujoco_to_isaaclab_dof)
        out += "\n" + fmt_list(f"{prefix}_ISAACLAB_TO_MUJOCO_BODY", self.isaaclab_to_mujoco_body)
        out += "\n" + fmt_list(f"{prefix}_MUJOCO_TO_ISAACLAB_BODY", self.mujoco_to_isaaclab_body)
        return out


def compute_ordering_maps(urdf_path: str | Path, mjcf_path: str | Path) -> OrderingMaps:
    """Compute the four SONIC ordering maps for a URDF/MJCF pair describing the same robot."""
    urdf = parse_urdf(urdf_path)
    mjcf = parse_mjcf(mjcf_path)

    il_bodies, il_joints = urdf.isaaclab_order()
    mj_joints = mjcf.joints
    mj_bodies_dof = [mjcf.bodies[0]] + [mjcf.joint_to_body[j] for j in mj_joints]

    if set(il_joints) != set(mj_joints):
        only_urdf = sorted(set(il_joints) - set(mj_joints))
        only_mjcf = sorted(set(mj_joints) - set(il_joints))
        raise ValueError(
            f"URDF/MJCF DOF joint sets differ.\n  only in URDF: {only_urdf}\n  only in MJCF: {only_mjcf}"
        )
    j2b = urdf.dof_joint_to_body()
    il_bodies_from_joints = [urdf.root] + [j2b[j] for j in il_joints]
    if il_bodies_from_joints != il_bodies:  # sanity: one joint per DOF body
        raise ValueError("Internal error: body/joint order mismatch in URDF traversal")
    # Body names must agree too (after renames), else the maps are ambiguous.
    mj_body_for_joint = {j: mjcf.joint_to_body[j] for j in mj_joints}
    mismatched = [
        (j, j2b[j], mj_body_for_joint[j]) for j in il_joints if j2b[j] != mj_body_for_joint[j]
    ]
    if mismatched or urdf.root != mjcf.bodies[0]:
        raise ValueError(
            f"Body names differ between URDF and MJCF (root {urdf.root!r} vs {mjcf.bodies[0]!r}); "
            f"first mismatches: {mismatched[:5]}"
        )

    il_idx = {j: i for i, j in enumerate(il_joints)}
    mj_idx = {j: i for i, j in enumerate(mj_joints)}
    isaaclab_to_mujoco_dof = [il_idx[j] for j in mj_joints]
    mujoco_to_isaaclab_dof = [mj_idx[j] for j in il_joints]

    il_bidx = {b: i for i, b in enumerate(il_bodies)}
    mj_bidx = {b: i for i, b in enumerate(mj_bodies_dof)}
    isaaclab_to_mujoco_body = [il_bidx[b] for b in mj_bodies_dof]
    mujoco_to_isaaclab_body = [mj_bidx[b] for b in il_bodies]

    return OrderingMaps(
        isaaclab_joints=il_bodies,
        isaaclab_dof_names=il_joints,
        mujoco_dof_names=mj_joints,
        mujoco_body_names=mj_bodies_dof,
        isaaclab_to_mujoco_dof=isaaclab_to_mujoco_dof,
        mujoco_to_isaaclab_dof=mujoco_to_isaaclab_dof,
        isaaclab_to_mujoco_body=isaaclab_to_mujoco_body,
        mujoco_to_isaaclab_body=mujoco_to_isaaclab_body,
    )
