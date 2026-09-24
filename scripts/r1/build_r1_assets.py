#!/usr/bin/env python3
"""Build the SONIC assets for Unitree R1 + Dex3-1 from the upstream Unitree sources.

Deterministic transformation (re-runnable; outputs are committed):

  third_party_assets/unitree_ros/robots/r1_description/R1.urdf (+meshes)
  third_party_assets/unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf (+meshes)
  third_party_assets/unitree_rl_mjlab/src/assets/robots/unitree_r1/xmls/r1.xml
        |
        v
  gear_sonic/data/assets/robot_description/urdf/r1/r1_dex3.urdf      (Isaac Lab physics model)
  gear_sonic/data/assets/robot_description/urdf/r1/meshes/*.STL      (R1 + Dex3 meshes, LFS)
  gear_sonic/data/assets/robot_description/mjcf/r1_dex3.xml          (motion-library FK model)
  gear_sonic/envs/manager_env/robots/r1_ordering.py                  (IsaacLab<->MuJoCo maps)
  gear_sonic/data/assets/robot_description/urdf/r1/R1_DEX3_DERIVED.json (measured constants)

URDF changes: rename pelvis_link->pelvis and waist_yaw_link->torso_link; head joints fixed;
Dex3-1 hands attached as fixed subtrees on the wrist roll links (fingers fixed at the open
pose); wrist collision mesh (which contains the stock fist) replaced by a forearm cylinder;
wrist visual mesh cut at the Dex3 mount (``*_wrist_roll_link_forearm.STL``, fist removed).
MJCF changes: meshdir; Dex3 mass fused into the wrist bodies (parallel-axis); forearm visual
mesh; hand collision/palm site extended; dangling <exclude> entries removed; <actuator> block
added (required by SONIC's Humanoid_Batch).

Usage:
    python scripts/r1/fetch_upstream_assets.py      # once
    python scripts/r1/build_r1_assets.py [--no-meshes]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np

from gear_sonic.utils.embodiment import r1_spec as spec
from gear_sonic.utils.embodiment.inertia import (
    RigidBody,
    body_to_mjcf_inertial_attrs,
    combine,
    mjcf_inertial_to_body,
    rpy_to_matrix,
    urdf_inertial_to_body,
)
from gear_sonic.utils.embodiment.ordering import compute_ordering_maps

REPO = Path(__file__).resolve().parents[2]
TP = REPO / "third_party_assets"
SRC_R1_URDF = TP / "unitree_ros/robots/r1_description/R1.urdf"
SRC_R1_MESHES = TP / "unitree_ros/robots/r1_description/meshes"
SRC_DEX3_DIR = TP / "unitree_ros/robots/dexterous_hand_description/dex3_1"
SRC_MJLAB_XML = TP / "unitree_rl_mjlab/src/assets/robots/unitree_r1/xmls/r1.xml"

ASSET_ROOT = REPO / "gear_sonic/data/assets/robot_description"
OUT_URDF_DIR = ASSET_ROOT / "urdf/r1"
OUT_URDF = OUT_URDF_DIR / "r1_dex3.urdf"
OUT_MESH_DIR = OUT_URDF_DIR / "meshes"
OUT_MJCF = ASSET_ROOT / "mjcf" / spec.MJCF_FILE_NAME
OUT_ORDERING_PY = REPO / "gear_sonic/envs/manager_env/robots/r1_ordering.py"
OUT_DERIVED_JSON = OUT_URDF_DIR / "R1_DEX3_DERIVED.json"

WRIST_FOREARM_CYLINDER = dict(radius=0.03, length=0.08)  # forearm tube before the Dex3 flange
HAND_COLLISION_CAPSULE = dict(x0=0.05, x1=0.24, radius=0.035)  # covers forearm end + Dex3 hand
PALM_SITE_X = spec.DEX3_MOUNT_XYZ[0] + 0.06  # Dex3 palm COM is ~0.062 beyond the palm base


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


# --------------------------------------------------------------------------------------
# Forearm visual: the stock wrist mesh minus the stock fist that the Dex3 replaces
# --------------------------------------------------------------------------------------
_STL_RECORD = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def _read_stl(path: Path) -> np.ndarray:
    """Triangles (n, 3, 3) of a binary or ASCII STL."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        verts = [
            [float(v) for v in line.split()[1:4]]
            for line in data.decode().splitlines()
            if line.strip().startswith("vertex")
        ]
        return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    n = int.from_bytes(data[80:84], "little")
    return np.frombuffer(data, dtype=_STL_RECORD, count=n, offset=84)["v"].astype(np.float64)


def _write_stl(path: Path, tris: np.ndarray) -> None:
    rec = np.zeros(len(tris), dtype=_STL_RECORD)
    rec["v"] = tris
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    rec["n"] = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
    path.write_bytes(b"\0" * 80 + len(tris).to_bytes(4, "little") + rec.tobytes())


def _clip_below_x(tris: np.ndarray, x_max: float) -> np.ndarray:
    """Keep the part of every triangle with x <= x_max (Sutherland-Hodgman, fan-triangulated)."""
    out = []
    for tri in tris:
        poly = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            if a[0] <= x_max:
                poly.append(a)
            if (a[0] <= x_max) != (b[0] <= x_max):
                poly.append(a + (x_max - a[0]) / (b[0] - a[0]) * (b - a))
        out += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
    return np.asarray(out, dtype=np.float64).reshape(-1, 3, 3)


def forearm_mesh_name(wrist_link: str) -> str:
    return f"{wrist_link}_forearm"


def write_forearm_meshes(mesh_dir: Path) -> None:
    """``<side>_wrist_roll_link_forearm.STL``: the wrist mesh up to the Dex3 mount plane.

    The upstream wrist_roll_link mesh is the forearm tube (x < 0.080) plus the stock fist
    (0.080 < x < 0.143), which would otherwise render through the Dex3 palm (as BruteForce's
    r1_dex3_constants.py does, the fist geometry is removed; its mass stays in the link).
    """
    for wrist in (spec.LEFT_EE_BODY, spec.RIGHT_EE_BODY):
        tris = _clip_below_x(_read_stl(SRC_R1_MESHES / f"{wrist}.STL"), spec.DEX3_MOUNT_XYZ[0])
        _write_stl(mesh_dir / f"{forearm_mesh_name(wrist)}.STL", tris)


def _pretty(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    txt = minidom.parseString(raw).toprettyxml(indent="  ")
    return "\n".join(line for line in txt.splitlines() if line.strip()) + "\n"


# --------------------------------------------------------------------------------------
# URDF
# --------------------------------------------------------------------------------------
def _rename_links(root: ET.Element, renames: dict[str, str]) -> None:
    for link in root.findall("link"):
        if link.get("name") in renames:
            link.set("name", renames[link.get("name")])
    for joint in root.findall("joint"):
        for tag in ("parent", "child"):
            el = joint.find(tag)
            if el is not None and el.get("link") in renames:
                el.set("link", renames[el.get("link")])


def _make_fixed(joint: ET.Element) -> None:
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "safety_controller", "calibration", "mimic"):
        for el in joint.findall(tag):
            joint.remove(el)


def _load_dex3(side: str) -> tuple[list[ET.Element], list[ET.Element]]:
    """Return (links, joints) of one Dex3-1 URDF with the floating base removed and joints fixed."""
    root = ET.parse(SRC_DEX3_DIR / f"dex3_1_{side}.urdf").getroot()
    links = [copy.deepcopy(el) for el in root.findall("link") if el.get("name") != "world"]
    joints = []
    for j in root.findall("joint"):
        if j.get("type") == "floating" or j.find("parent").get("link") == "world":
            continue
        j = copy.deepcopy(j)
        _make_fixed(j)
        joints.append(j)
    return links, joints


def build_urdf() -> ET.Element:
    root = ET.parse(SRC_R1_URDF).getroot()
    root.set("name", "r1_dex3")
    for el in root.findall("mujoco"):
        root.remove(el)

    _rename_links(root, spec.LINK_RENAMES)

    # Head joints -> fixed (policy does not control the head; mass retained).
    for joint in root.findall("joint"):
        if joint.get("name") in spec.FIXED_JOINTS:
            _make_fixed(joint)

    # Wrist collision and visual: the upstream mesh includes the stock fist which the Dex3
    # replaces; collide with a forearm cylinder and draw the mesh cut at the Dex3 mount.
    for link in root.findall("link"):
        if link.get("name") in (spec.LEFT_EE_BODY, spec.RIGHT_EE_BODY):
            for mesh in link.findall("visual/geometry/mesh"):
                mesh.set("filename", f"{forearm_mesh_name(link.get('name'))}.STL")
            for col in link.findall("collision"):
                link.remove(col)
            col = ET.SubElement(link, "collision")
            ET.SubElement(
                col,
                "origin",
                xyz=_fmt([WRIST_FOREARM_CYLINDER["length"] / 2, 0, 0]),
                rpy=_fmt([0, np.pi / 2, 0]),
            )
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(
                geom,
                "cylinder",
                radius=str(WRIST_FOREARM_CYLINDER["radius"]),
                length=str(WRIST_FOREARM_CYLINDER["length"]),
            )

    # Attach Dex3-1 hands.
    for side, parent, palm in (
        ("l", spec.LEFT_EE_BODY, spec.LEFT_PALM_BODY),
        ("r", spec.RIGHT_EE_BODY, spec.RIGHT_PALM_BODY),
    ):
        links, joints = _load_dex3(side)
        mount = ET.Element("joint", name=f"{palm.replace('_link', '')}_joint", type="fixed")
        ET.SubElement(mount, "origin", xyz=_fmt(spec.DEX3_MOUNT_XYZ), rpy=_fmt(spec.DEX3_MOUNT_RPY))
        ET.SubElement(mount, "parent", link=parent)
        ET.SubElement(mount, "child", link=palm)
        root.append(
            ET.Comment(
                f" Dex3-1 {side.upper()} hand (unitree_ros dex3_1_{side}.urdf), fixed at the open pose "
            )
        )
        root.append(mount)
        for el in links + joints:
            root.append(el)

    # Mesh paths: keep the H2 convention (relative 'meshes/<file>' next to the URDF).
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", "meshes/" + Path(fn).name)
    return root


# --------------------------------------------------------------------------------------
# MJCF
# --------------------------------------------------------------------------------------
def _urdf_subtree_bodies(urdf_root: ET.Element, top_link: str, stop_link: str) -> list[RigidBody]:
    """All link inertias in the fixed subtree rooted at ``top_link``, expressed in ``stop_link``'s frame."""
    links = {el.get("name"): el for el in urdf_root.findall("link")}
    joints_by_child = {j.find("child").get("link"): j for j in urdf_root.findall("joint")}
    children: dict[str, list[str]] = {}
    for c, j in joints_by_child.items():
        children.setdefault(j.find("parent").get("link"), []).append(c)

    def pose_in(link: str) -> tuple[np.ndarray, np.ndarray]:
        rot, trans = np.eye(3), np.zeros(3)
        cur = link
        while cur != stop_link:
            j = joints_by_child[cur]
            o = j.find("origin")
            xyz = np.array(
                [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            )
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            rj = rpy_to_matrix(rpy)
            rot, trans = rj @ rot, rj @ trans + xyz
            cur = j.find("parent").get("link")
        return rot, trans

    out, stack = [], [top_link]
    while stack:
        name = stack.pop()
        inertial = links[name].find("inertial")
        if inertial is not None:
            rot, trans = pose_in(name)
            out.append(urdf_inertial_to_body(inertial).transformed(rot, trans))
        stack.extend(children.get(name, []))
    return out


def _dex3_visual_geoms(
    urdf_root: ET.Element, palm: str, wrist: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(mesh name, pos, quat_wxyz) for each Dex3 link mesh in the wrist frame."""
    from scipy.spatial.transform import Rotation as R

    links = {el.get("name"): el for el in urdf_root.findall("link")}
    joints_by_child = {j.find("child").get("link"): j for j in urdf_root.findall("joint")}
    children: dict[str, list[str]] = {}
    for c, j in joints_by_child.items():
        children.setdefault(j.find("parent").get("link"), []).append(c)
    out, stack = [], [palm]
    while stack:
        name = stack.pop()
        rot, trans = np.eye(3), np.zeros(3)
        cur = name
        while cur != wrist:
            j = joints_by_child[cur]
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            rj = rpy_to_matrix(rpy)
            rot, trans = rj @ rot, rj @ trans + xyz
            cur = j.find("parent").get("link")
        vis = links[name].find("visual")
        if vis is not None and vis.find("geometry/mesh") is not None:
            vo = vis.find("origin")
            vxyz = np.array(
                [float(v) for v in (vo.get("xyz", "0 0 0") if vo is not None else "0 0 0").split()]
            )
            vrpy = [
                float(v) for v in (vo.get("rpy", "0 0 0") if vo is not None else "0 0 0").split()
            ]
            rv = rpy_to_matrix(vrpy)
            rot_g, trans_g = rot @ rv, rot @ vxyz + trans
            x, y, z, w = R.from_matrix(rot_g).as_quat()
            out.append(
                (
                    Path(vis.find("geometry/mesh").get("filename")).stem,
                    trans_g,
                    np.array([w, x, y, z]),
                )
            )
        stack.extend(children.get(name, []))
    return out


def build_mjcf(urdf_root: ET.Element) -> ET.Element:
    root = ET.parse(SRC_MJLAB_XML).getroot()
    root.set("model", "r1_dex3")
    root.find("compiler").set("meshdir", "../urdf/r1/meshes/")

    # Dangling contact excludes (bodies that do not exist in the 24-DOF model).
    contact = root.find("contact")
    if contact is not None:
        body_names = {b.get("name") for b in root.iter("body")}
        for ex in list(contact):
            if ex.get("body1") not in body_names or ex.get("body2") not in body_names:
                contact.remove(ex)

    asset = root.find("asset")
    bodies = {b.get("name"): b for b in root.iter("body")}
    for wrist, palm in (
        (spec.LEFT_EE_BODY, spec.LEFT_PALM_BODY),
        (spec.RIGHT_EE_BODY, spec.RIGHT_PALM_BODY),
    ):
        wbody = bodies[wrist]
        # Draw the forearm only; the stock fist is replaced by the Dex3 (write_forearm_meshes).
        wrist_mesh = asset.find(f"mesh[@name='{wrist}']")
        if wrist_mesh is not None:
            wrist_mesh.set("file", f"{forearm_mesh_name(wrist)}.STL")
        # Fuse Dex3 mass into the wrist body.
        hand_parts = _urdf_subtree_bodies(urdf_root, palm, wrist)
        fused = combine([mjcf_inertial_to_body(wbody.find("inertial"))] + hand_parts)
        for k, v in body_to_mjcf_inertial_attrs(fused).items():
            wbody.find("inertial").set(k, v)
        # Visual meshes for the hand (nice renders, no physics role in the motion lib).
        for mesh_name, pos, quat in _dex3_visual_geoms(urdf_root, palm, wrist):
            if asset.find(f"mesh[@name='{mesh_name}']") is None:
                ET.SubElement(asset, "mesh", name=mesh_name, file=f"{mesh_name}.STL")
            ET.SubElement(
                wbody,
                "geom",
                **{"class": "visual"},
                pos=_fmt(pos),
                quat=_fmt(quat),
                rgba="0.3 0.3 0.3 1",
                mesh=mesh_name,
            )
        # Collision capsule and palm site covering the Dex3 hand instead of the stock fist.
        side = "left" if wrist.startswith("left") else "right"
        for g in wbody.findall("geom"):
            if g.get("name") == f"{side}_hand_collision":
                c = HAND_COLLISION_CAPSULE
                g.set("fromto", _fmt([c["x0"], 0, 0, c["x1"], 0, 0]))
                g.set("size", str(c["radius"]))
        for s in wbody.findall("site"):
            if s.get("name") == f"{side}_palm":
                s.set("pos", _fmt([PALM_SITE_X, 0, 0]))

    # Actuators (SONIC's Humanoid_Batch requires one <motor> per DOF; names == joint names).
    for act in root.findall("actuator"):
        root.remove(act)
    act = ET.SubElement(root, "actuator")
    mjcf_joints = [
        j.get("name") for j in root.find("worldbody").iter("joint") if j.get("type") != "free"
    ]
    if tuple(mjcf_joints) != spec.MJCF_JOINT_ORDER:
        raise RuntimeError(
            f"mjlab joint order changed; update r1_spec.MJCF_JOINT_ORDER:\n{mjcf_joints}"
        )
    for j in mjcf_joints:
        ET.SubElement(act, "motor", name=j, joint=j)
    return root


# --------------------------------------------------------------------------------------
# Derived constants
# --------------------------------------------------------------------------------------
def derive_constants(mjcf_path: Path) -> dict:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)  # zero pose (straight legs)

    def body_pos(name):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    hip = body_pos("left_hip_pitch_link")
    ankle = body_pos(spec.LEFT_FOOT_BODY)
    # Lowest foot collision point at zero pose (capsule centres minus radius).
    foot_geoms = [
        i
        for i in range(m.ngeom)
        if m.geom_bodyid[i] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.LEFT_FOOT_BODY)
        and m.geom_contype[i] | m.geom_conaffinity[i]
        or "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
    ]
    sole_z = (
        min(float(d.geom_xpos[i][2] - m.geom_size[i][0]) for i in foot_geoms)
        if foot_geoms
        else float("nan")
    )
    total_mass = float(
        m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.ROOT_BODY)]
    )
    return {
        "num_dof": int(m.nu),
        "num_bodies": int(m.nbody - 1),  # minus world
        "total_mass_kg": round(total_mass, 4),
        "hip_pitch_to_ankle_roll_m_straight": round(float(np.linalg.norm(hip - ankle)), 5),
        "pelvis_height_straight_legs_feet_flat_m": round(
            float(body_pos(spec.ROOT_BODY)[2] - sole_z), 5
        ),
        "sole_below_ankle_roll_m": round(float(ankle[2] - sole_z), 5),
        "dex3_mount_xyz": list(spec.DEX3_MOUNT_XYZ),
        "hand_point_offset_left": list(spec.LEFT_HAND_POINT_OFFSET),
        "hand_point_offset_right": list(spec.RIGHT_HAND_POINT_OFFSET),
        "vr_head_point_offset": list(spec.VR_HEAD_POINT_OFFSET),
        "wrist_mass_with_dex3_kg": round(
            float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.LEFT_EE_BODY)]), 4
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-meshes", action="store_true", help="skip copying STL meshes")
    args = ap.parse_args()
    for p in (SRC_R1_URDF, SRC_DEX3_DIR, SRC_MJLAB_XML):
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/r1/fetch_upstream_assets.py first")

    OUT_URDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MJCF.parent.mkdir(parents=True, exist_ok=True)

    urdf_root = build_urdf()
    OUT_URDF.write_text(_pretty(urdf_root))
    print(f"wrote {OUT_URDF.relative_to(REPO)}")

    mjcf_root = build_mjcf(urdf_root)
    OUT_MJCF.write_text(_pretty(mjcf_root))
    print(f"wrote {OUT_MJCF.relative_to(REPO)}")

    if not args.no_meshes:
        OUT_MESH_DIR.mkdir(exist_ok=True)
        n = 0
        for src_dir in (SRC_R1_MESHES, SRC_DEX3_DIR / "meshes"):
            for f in sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".stl"):
                shutil.copy2(f, OUT_MESH_DIR / f.name)
                n += 1
        print(f"copied {n} meshes -> {OUT_MESH_DIR.relative_to(REPO)}")
    OUT_MESH_DIR.mkdir(exist_ok=True)
    write_forearm_meshes(OUT_MESH_DIR)
    print(
        f"wrote forearm meshes (cut at x={spec.DEX3_MOUNT_XYZ[0]}) -> {OUT_MESH_DIR.relative_to(REPO)}"
    )

    maps = compute_ordering_maps(OUT_URDF, OUT_MJCF)
    header = (
        '"""AUTO-GENERATED by scripts/r1/build_r1_assets.py - do not edit by hand.\n\n'
        "IsaacLab <-> MuJoCo ordering for Unitree R1 (24 DOF, 25 bodies). Isaac Lab order is the\n"
        "inferred breadth-first URDF order (rule validated on G1/H2); confirm with\n"
        "scripts/r1/verify_isaaclab_order.py on a machine with Isaac Lab.\n"
        '"""\n\n'
    )
    OUT_ORDERING_PY.write_text(
        header + maps.to_python("R1") + "\nR1_ISAACLAB_TO_MUJOCO_MAPPING = {\n"
        '    "isaaclab_joints": R1_ISAACLAB_JOINTS,\n'
        '    "isaaclab_to_mujoco_dof": R1_ISAACLAB_TO_MUJOCO_DOF,\n'
        '    "mujoco_to_isaaclab_dof": R1_MUJOCO_TO_ISAACLAB_DOF,\n'
        '    "isaaclab_to_mujoco_body": R1_ISAACLAB_TO_MUJOCO_BODY,\n'
        '    "mujoco_to_isaaclab_body": R1_MUJOCO_TO_ISAACLAB_BODY,\n'
        "}\n"
    )
    n_dof, n_bodies = len(maps.isaaclab_dof_names), len(maps.isaaclab_joints)
    print(f"wrote {OUT_ORDERING_PY.relative_to(REPO)} ({n_dof} DOF, {n_bodies} bodies)")

    derived = derive_constants(OUT_MJCF)
    OUT_DERIVED_JSON.write_text(json.dumps(derived, indent=2) + "\n")
    print(f"wrote {OUT_DERIVED_JSON.relative_to(REPO)}:\n{json.dumps(derived, indent=2)}")


if __name__ == "__main__":
    main()
