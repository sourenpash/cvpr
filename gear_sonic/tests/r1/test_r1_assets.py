"""Consistency tests for the generated R1 + Dex3 assets (URDF, MJCF, ordering constants)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gear_sonic.tests.r1.conftest import R1_MJCF, R1_URDF, REPO, load_module_constants
from gear_sonic.utils.embodiment import r1_spec as spec
from gear_sonic.utils.embodiment.ordering import compute_ordering_maps, parse_mjcf, parse_urdf

ORDERING_PY = REPO / "gear_sonic/envs/manager_env/robots/r1_ordering.py"


@pytest.fixture(scope="module")
def urdf_root():
    assert R1_URDF.exists(), "run scripts/r1/build_r1_assets.py"
    return ET.parse(R1_URDF).getroot()


@pytest.fixture(scope="module")
def mjcf_tree():
    return parse_mjcf(R1_MJCF)


# --------------------------------------------------------------------------------------
# URDF structure
# --------------------------------------------------------------------------------------
def test_urdf_actuated_joints_match_spec(urdf_root):
    tree = parse_urdf(R1_URDF)
    _, il_joints = tree.isaaclab_order()
    assert set(il_joints) == set(spec.MJCF_JOINT_ORDER)
    assert len(il_joints) == spec.NUM_DOF == 24
    assert tree.root == spec.ROOT_BODY


def test_urdf_renames_and_fixed_head(urdf_root):
    links = {el.get("name") for el in urdf_root.findall("link")}
    assert "pelvis" in links and "pelvis_link" not in links
    assert "torso_link" in links and "waist_yaw_link" not in links
    joints = {j.get("name"): j for j in urdf_root.findall("joint")}
    for j in spec.FIXED_JOINTS:
        assert joints[j].get("type") == "fixed"
    assert spec.HEAD_BODY in links


def test_urdf_has_dex3_hands_fixed(urdf_root):
    links = {el.get("name") for el in urdf_root.findall("link")}
    joints = {j.get("name"): j for j in urdf_root.findall("joint")}
    for side, ee, palm in (
        ("left", spec.LEFT_EE_BODY, spec.LEFT_PALM_BODY),
        ("right", spec.RIGHT_EE_BODY, spec.RIGHT_PALM_BODY),
    ):
        assert palm in links
        mount = joints[f"{side}_hand_palm_joint"]
        assert mount.get("type") == "fixed"
        assert mount.find("parent").get("link") == ee
        assert mount.find("child").get("link") == palm
        xyz = [float(v) for v in mount.find("origin").get("xyz").split()]
        assert np.allclose(xyz, spec.DEX3_MOUNT_XYZ)
        for finger in (
            "thumb_0",
            "thumb_1",
            "thumb_2",
            "index_0",
            "index_1",
            "middle_0",
            "middle_1",
        ):
            assert f"{side}_hand_{finger}_link" in links
            assert joints[f"{side}_hand_{finger}_joint"].get("type") == "fixed"


def test_urdf_wrist_collision_is_forearm_cylinder(urdf_root):
    for ee in (spec.LEFT_EE_BODY, spec.RIGHT_EE_BODY):
        link = [el for el in urdf_root.findall("link") if el.get("name") == ee][0]
        cols = link.findall("collision")
        assert len(cols) == 1 and cols[0].find("geometry/cylinder") is not None


def test_urdf_mesh_paths_relative_and_present(urdf_root):
    mesh_dir = R1_URDF.parent / "meshes"
    for mesh in urdf_root.iter("mesh"):
        fn = mesh.get("filename")
        assert fn.startswith("meshes/"), fn
        assert (R1_URDF.parent / fn).exists(), f"missing mesh {fn} (git lfs pull?)"
    n_meshes = len([p for p in mesh_dir.iterdir() if p.suffix.lower() == ".stl"])
    assert n_meshes >= 43 + 16, n_meshes  # 43 R1 (incl. torso_collision.stl) + 16 Dex3


# --------------------------------------------------------------------------------------
# MJCF structure
# --------------------------------------------------------------------------------------
def test_mjcf_joint_order_and_motors(mjcf_tree):
    assert tuple(mjcf_tree.joints) == spec.MJCF_JOINT_ORDER
    assert tuple(mjcf_tree.motors) == spec.MJCF_JOINT_ORDER, "one <motor> per joint, same order"
    assert mjcf_tree.bodies[0] == spec.ROOT_BODY
    assert len(mjcf_tree.bodies) == spec.NUM_BODIES
    # legs first: commands.py hard-codes lower_joint_indices_mujoco = range(12)
    assert all(("hip" in j or "knee" in j or "ankle" in j) for j in mjcf_tree.joints[:12])


def test_mjcf_wrist_indices_match_spec(mjcf_tree):
    assert [mjcf_tree.joints.index(j) for j in spec.WRIST_JOINTS] == spec.WRIST_MUJOCO_DOF_INDICES


def test_mjcf_loads_in_mujoco_and_has_dex3_mass():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(R1_MJCF))
    assert m.nu == spec.NUM_DOF
    assert m.nbody - 1 == spec.NUM_BODIES
    wrist_mass = m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.LEFT_EE_BODY)]
    assert 0.9 < wrist_mass < 1.2, "wrist should carry stock wrist (0.32 kg) + Dex3 (~0.7 kg)"
    total = m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.ROOT_BODY)]
    assert 29.5 < total < 31.5, f"total mass {total:.2f} kg (spec: ~29 kg body + 2x0.7 kg hands)"


# --------------------------------------------------------------------------------------
# Ordering constants
# --------------------------------------------------------------------------------------
def test_ordering_file_is_up_to_date():
    maps = compute_ordering_maps(R1_URDF, R1_MJCF)
    shipped = load_module_constants(ORDERING_PY, "R1")
    for key, value in maps.as_dict().items():
        assert (
            value == shipped[f"R1_{key.upper()}"]
        ), f"{key} stale: re-run scripts/r1/build_r1_assets.py"
    assert len(maps.isaaclab_dof_names) == spec.NUM_DOF
    # Isaac Lab DOF indices of the wrists (used by the SMPL-encoder wrist observation)
    assert [
        maps.isaaclab_dof_names.index(j) for j in spec.WRIST_JOINTS
    ] == spec.WRIST_ISAACLAB_DOF_INDICES


# --------------------------------------------------------------------------------------
# Kinematics: SONIC's Humanoid_Batch FK must agree with MuJoCo on the same MJCF
# --------------------------------------------------------------------------------------
def test_humanoid_batch_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    torch = pytest.importorskip("torch")
    omegaconf = pytest.importorskip("omegaconf")
    from scipy.spatial.transform import Rotation as R

    from gear_sonic.utils.motion_lib import torch_humanoid_batch as thb

    m = mujoco.MjModel.from_xml_path(str(R1_MJCF))
    d = mujoco.MjData(m)
    cfg = omegaconf.OmegaConf.create(
        {
            "asset": {
                "assetRoot": str(R1_MJCF.parent) + "/",
                "assetFileName": R1_MJCF.name,
                "urdfFileName": "",
            },
            "extend_config": [],
        }
    )
    hb = thb.Humanoid_Batch(cfg)
    assert hb.num_dof == spec.NUM_DOF and hb.num_bodies == spec.NUM_BODIES
    assert hb.body_names[0] == spec.ROOT_BODY

    rng = np.random.default_rng(0)
    T = 20
    dof = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1], size=(T, spec.NUM_DOF))
    root_rot = R.random(T, random_state=1)
    root_pos = rng.normal(size=(T, 3))
    pose_aa = np.zeros((T, spec.NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0] = root_rot.as_rotvec()
    pose_aa[:, 1:] = hb.dof_axis.numpy()[None] * dof[:, :, None]
    out = hb.fk_batch(
        torch.tensor(pose_aa[None]),
        torch.tensor(root_pos[None], dtype=torch.float32),
        return_full=True,
    )
    hb_pos = out["global_translation"][0].numpy()

    body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in hb.body_names]
    mj_pos = np.zeros_like(hb_pos)
    for t in range(T):
        q = root_rot[t].as_quat()  # xyzw
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[7:] = dof[t]
        mujoco.mj_kinematics(m, d)
        mj_pos[t] = d.xpos[body_ids]
    assert np.abs(hb_pos - mj_pos).max() < 1e-4


def test_home_pose_feet_clearance():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(R1_MJCF))
    d = mujoco.MjData(m)
    d.qpos[2] = spec.INIT_POS_Z
    d.qpos[3:7] = [1, 0, 0, 0]
    import re

    for j_idx in range(1, m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j_idx)
        for pattern, val in spec.INIT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                d.qpos[m.jnt_qposadr[j_idx]] = val
    mujoco.mj_kinematics(m, d)
    foot_geoms = [
        i
        for i in range(m.ngeom)
        if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
    ]
    lowest = min(float(d.geom_xpos[i][2] - m.geom_size[i][0]) for i in foot_geoms)
    assert 0.0 < lowest < 0.03, f"feet should spawn slightly above ground, got {lowest:.4f} m"
