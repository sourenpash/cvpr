"""The R1 experiment preset and robot config must agree with r1_spec and with the URDF."""

from __future__ import annotations

import importlib
import re
import xml.etree.ElementTree as ET

import pytest
import yaml

from gear_sonic.tests.r1.conftest import PRESET, R1_URDF, REPO
from gear_sonic.utils.embodiment import r1_spec as spec

MOTION_YAML = REPO / "gear_sonic/config/manager_env/commands/terms/motion.yaml"


@pytest.fixture(scope="module")
def preset() -> dict:
    return yaml.safe_load(PRESET.read_text())


@pytest.fixture(scope="module")
def urdf_bodies() -> set[str]:
    return {el.get("name") for el in ET.parse(R1_URDF).getroot().findall("link")}


def test_preset_robot_type_and_asset(preset):
    assert preset["manager_env"]["config"]["robot"]["type"] == spec.ROBOT_TYPE
    motion = preset["manager_env"]["commands"]["motion"]
    assert motion["motion_lib_cfg"]["asset"]["assetFileName"] == spec.MJCF_FILE_NAME
    assert motion["motion_lib_cfg"]["robot_type"] == spec.ROBOT_TYPE
    assert motion["motion_lib_cfg"]["wrist_mujoco_dof_indices"] == spec.WRIST_MUJOCO_DOF_INDICES
    assert (
        "upper_body_augment_prefixes" not in motion["motion_lib_cfg"]
    ), "G1-specific motion-name prefixes"


def test_preset_tracking_points_match_spec(preset):
    motion = preset["manager_env"]["commands"]["motion"]
    assert motion["vr_3point_body"] == spec.VR_3POINT_BODY
    assert motion["vr_3point_body_offset"] == spec.VR_3POINT_BODY_OFFSET
    assert motion["reward_point_body"] == spec.REWARD_POINT_BODY_3PT
    assert motion["reward_point_body_offset"] == spec.REWARD_POINT_BODY_OFFSET_3PT
    assert motion["body_names"] == spec.TRACKED_BODY_NAMES
    assert len(motion["body_names"]) == 14


def test_preset_body_name_overrides_match_spec(preset):
    me = preset["manager_env"]
    assert me["rewards"]["anti_shake_ang_vel"]["params"]["body_names"] == spec.ANTI_SHAKE_BODIES
    assert me["rewards"]["tracking_vr_2wrists_local_ori"]["params"]["body_names"] == [
        spec.LEFT_EE_BODY,
        spec.RIGHT_EE_BODY,
    ]
    assert me["rewards"]["undesired_contacts"]["params"]["sensor_cfg"]["body_names"] == [
        spec.UNDESIRED_CONTACT_EXCLUDE_REGEX
    ]
    assert me["terminations"]["ee_body_pos"]["params"]["body_names"] == spec.EE_TERMINATION_BODIES
    assert (
        me["events"]["randomize_rigid_body_mass"]["params"]["asset_cfg"]["body_names"]
        == spec.UPPER_BODY_EVENT_BODY_REGEX
    )
    assert (
        me["observations"]["tokenizer"]["joint_pos_multi_future_wrist_for_smpl"]["params"][
            "joints_idx"
        ]
        == spec.WRIST_ISAACLAB_DOF_INDICES
    )


def test_all_referenced_bodies_exist_in_urdf(preset, urdf_bodies):
    me = preset["manager_env"]
    motion = me["commands"]["motion"]
    names = (
        set(motion["vr_3point_body"]) | set(motion["reward_point_body"]) | set(motion["body_names"])
    )
    names |= set(me["rewards"]["anti_shake_ang_vel"]["params"]["body_names"])
    names |= set(me["rewards"]["tracking_vr_2wrists_local_ori"]["params"]["body_names"])
    names |= set(me["terminations"]["ee_body_pos"]["params"]["body_names"])
    base_motion = yaml.safe_load(MOTION_YAML.read_text())["motion"]
    names.add(base_motion["anchor_body"])
    # foot_pos_xyz termination and undesired_contacts exclusions use the ankle roll links
    names |= {spec.LEFT_FOOT_BODY, spec.RIGHT_FOOT_BODY}
    missing = sorted(n for n in names if n not in urdf_bodies)
    assert not missing, f"bodies referenced by the R1 preset but absent from the URDF: {missing}"
    # regex-based selections must match the intended bodies and nothing G1-specific
    rx = re.compile(me["events"]["randomize_rigid_body_mass"]["params"]["asset_cfg"]["body_names"])
    matched = sorted(b for b in urdf_bodies if rx.fullmatch(b))
    assert matched == sorted([spec.LEFT_EE_BODY, spec.RIGHT_EE_BODY, spec.TORSO_BODY])
    rx_contacts = re.compile(spec.UNDESIRED_CONTACT_EXCLUDE_REGEX)
    penalised = {b for b in urdf_bodies if rx_contacts.fullmatch(b)}
    for allowed in (
        spec.LEFT_FOOT_BODY,
        spec.RIGHT_FOOT_BODY,
        spec.LEFT_EE_BODY,
        spec.RIGHT_EE_BODY,
        spec.LEFT_PALM_BODY,
        "right_hand_thumb_2_link",
    ):
        assert allowed not in penalised
    assert spec.TORSO_BODY in penalised and "left_knee_link" in penalised


def test_robot_config_module_builds_action_scale(isaaclab_stubs):
    """Import robots/r1.py with stubbed Isaac Lab and check the derived dicts."""
    r1 = importlib.import_module("gear_sonic.envs.manager_env.robots.r1")
    assert r1.R1_DEX3_CFG.init_state.pos == (0.0, 0.0, spec.INIT_POS_Z)
    assert set(r1.R1_DEX3_CFG.actuators) == set(spec.ACTUATOR_GROUPS)
    # every policy joint is covered by exactly one actuator group and gets an action scale
    joints = spec.MJCF_JOINT_ORDER
    for j in joints:
        groups = [
            g
            for g, cfg in spec.ACTUATOR_GROUPS.items()
            if any(re.fullmatch(p, j) for p in cfg["joints"])
        ]
        assert len(groups) == 1, f"{j} matched actuator groups {groups}"
        scales = [s for p, s in r1.R1_DEX3_ACTION_SCALE.items() if re.fullmatch(p, j)]
        assert len(scales) == 1 and scales[0] > 0
    # sanity on magnitudes vs G1 (G1 legs: 0.25*139/99 ~ 0.35; R1 legs: 0.25*60/100 = 0.15)
    assert abs(r1.R1_DEX3_ACTION_SCALE[".*_knee_joint"] - 0.15) < 1e-9
    assert r1.R1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_joints"][0] == spec.ROOT_BODY


def test_order_converter_registry():
    from gear_sonic.trl.utils import order_converter as oc

    conv = oc.get_converter(spec.ROBOT_TYPE)
    assert isinstance(conv, oc.R1Converter)
    assert conv.num_dof == spec.NUM_DOF
    assert conv.VR_3POINTS_BODY_NAMES == [spec.TORSO_BODY, spec.LEFT_EE_BODY, spec.RIGHT_EE_BODY]
    import torch

    qpos = torch.randn(4, 7 + spec.NUM_DOF)
    assert torch.allclose(conv.to_isaaclab(conv.to_mujoco(qpos)), qpos)
    assert isinstance(oc.get_converter("g1_model_12_dex"), oc.G1Converter)
    with pytest.raises(KeyError):
        oc.get_converter("not_a_robot")
