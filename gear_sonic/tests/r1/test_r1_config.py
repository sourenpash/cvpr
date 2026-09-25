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
TELEOP_PRESET = PRESET.with_name("sonic_r1_dex3_teleop.yaml")
EVENTS = REPO / "gear_sonic/envs/manager_env/mdp/events.py"
R1_EVENTS = REPO / "gear_sonic/envs/manager_env/mdp/r1_events.py"


@pytest.fixture(scope="module", params=[PRESET, TELEOP_PRESET], ids=lambda p: p.stem)
def preset(request) -> dict:
    return yaml.safe_load(request.param.read_text())


@pytest.fixture(scope="module")
def urdf_bodies() -> set[str]:
    return {el.get("name") for el in ET.parse(R1_URDF).getroot().findall("link")}


def test_preset_robot_type_and_asset(preset):
    assert preset["manager_env"]["config"]["robot"]["type"] == spec.ROBOT_TYPE
    motion = preset["manager_env"]["commands"]["motion"]
    assert motion["motion_lib_cfg"]["asset"]["assetFileName"] == spec.MJCF_FILE_NAME
    assert motion["motion_lib_cfg"]["robot_type"] == spec.ROBOT_TYPE
    assert motion["motion_lib_cfg"]["wrist_mujoco_dof_indices"] == spec.WRIST_MUJOCO_DOF_INDICES
    # Upper-body grafting may only target our planner clips, never NVIDIA's G1 clip names.
    prefixes = motion["motion_lib_cfg"].get("upper_body_augment_prefixes", [])
    assert all(p.startswith("planner_") for p in prefixes), prefixes


def test_preset_tracking_points_match_spec(preset):
    motion = preset["manager_env"]["commands"]["motion"]
    assert motion["vr_3point_body"] == spec.VR_3POINT_BODY
    assert motion["vr_3point_body_offset"] == spec.VR_3POINT_BODY_OFFSET
    assert motion["reward_point_body"] == spec.REWARD_POINT_BODY_3PT
    assert motion["reward_point_body_offset"] == spec.REWARD_POINT_BODY_OFFSET_3PT
    # The EE position reward must score the same palm points the teleop encoder targets.
    assert motion["reward_point_body_offset"][1:] == motion["vr_3point_body_offset"][:2]
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
    if "observations" in me:  # the SMPL encoder's wrist input (three-encoder preset only)
        assert (
            me["observations"]["tokenizer"]["joint_pos_multi_future_wrist_for_smpl"]["params"][
                "joints_idx"
            ]
            == spec.WRIST_ISAACLAB_DOF_INDICES
        )


def test_teleop_preset_builds_only_the_quest_path():
    """sonic_r1_dex3_teleop: VR 3-point encoder + action decoder, nothing else (PLAN.md D13)."""
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    from gear_sonic.utils import config_utils

    config_utils.register_rl_resolvers()
    with hydra.initialize_config_dir(
        config_dir=str(REPO / "gear_sonic/config"), version_base="1.1"
    ):
        cfg = hydra.compose(
            config_name="base",
            overrides=["+exp=manager/universal_token/all_modes/sonic_r1_dex3_teleop"],
        )
    backbone = cfg.algo.config.actor.backbone
    assert list(backbone.encoders) == ["teleop"] and list(backbone.active_encoders) == ["teleop"]
    assert list(backbone.decoders) == ["g1_dyn"] and list(backbone.active_decoders) == ["g1_dyn"]
    assert not backbone.get("aux_loss_func") and not cfg.algo.config.actor.has_aux_loss
    probs = OmegaConf.to_container(cfg.manager_env.commands.motion.encoder_sample_probs)
    assert probs == {"g1": 0.0, "teleop": 1.0}  # commands.py requires the g1 key
    tokenizer = {k for k in cfg.manager_env.observations.tokenizer if k.endswith(("_target", "_b"))}
    assert set(backbone.encoders.teleop.inputs) <= set(cfg.manager_env.observations.tokenizer)
    assert not any("smpl" in k for k in cfg.manager_env.observations.tokenizer), tokenizer
    assert cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file == "dummy"
    # Reach-while-walking: planner clips get the upper body of a random mocap clip.
    motion = cfg.manager_env.commands.motion
    assert motion.cat_upper_body_poses
    assert list(motion.motion_lib_cfg.upper_body_augment_prefixes) == ["planner_"]


def test_teleop_robust_preset_adds_only_robustness_terms():
    """sonic_r1_dex3_teleop_robust = the teleop preset + latency, armature and gain randomization."""
    hydra = pytest.importorskip("hydra")

    from gear_sonic.utils import config_utils

    config_utils.register_rl_resolvers()
    exp = "+exp=manager/universal_token/all_modes/sonic_r1_dex3_teleop"
    with hydra.initialize_config_dir(
        config_dir=str(REPO / "gear_sonic/config"), version_base="1.1"
    ):
        base = hydra.compose(config_name="base", overrides=[exp])
        robust = hydra.compose(config_name="base", overrides=[exp + "_robust"])
    assert robust.algo.config.actor.backbone == base.algo.config.actor.backbone
    assert robust.manager_env.observations == base.manager_env.observations
    assert robust.manager_env.commands == base.manager_env.commands
    action = robust.manager_env.actions.joint_pos
    assert action._target_.endswith("delayed_actions.DelayedJointPositionActionCfg")
    assert list(action.delay_substeps_range) == [0, 4]  # 0-20 ms at sim.dt = 5 ms
    events = robust.manager_env.events
    legs = events.r1_leg_armature.params.armature_distribution_params
    ankles = events.r1_ankle_armature.params.armature_distribution_params
    assert legs[0] < 0.05 < legs[1] and ankles[0] < 0.10 < ankles[1]  # unitree_rl_mjlab #51
    assert robust.callbacks.periodic_eval.eval_frequency == 1000
    # EventCfg only accepts declared fields: every added term needs one in R1RobustEventCfg.
    assert events._target_.endswith("r1_events.R1RobustEventCfg")
    declared = set(re.findall(r"^    (\w+) = None$", R1_EVENTS.read_text(), re.M)) | set(
        re.findall(r"^    (\w+) = None$", EVENTS.read_text().split("def ")[0], re.M)
    )
    assert set(events) - {"_target_"} <= declared, set(events) - declared
    friction = events.r1_joint_friction
    assert friction.func.endswith("joint_friction:randomize_joint_friction_torque")
    assert friction.params.torque[".*_knee_joint"] == 2.5 and friction.mode == "startup"


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
    assert r1.R1_ISAACLAB_TO_MUJOCO_MAPPING["isaaclab_joints"][0] == spec.ROOT_BODY


def _lookup(patterns: dict, joint: str):
    hits = [v for p, v in patterns.items() if re.fullmatch(p, joint)]
    assert len(hits) <= 1, f"{joint} matched {len(hits)} patterns"
    return hits[0] if hits else 0.0


def test_action_parameterization_matches_g1(isaaclab_stubs):
    """Warm start (D4/D10): default pose and action scale equal G1's for every shared joint."""
    g1 = importlib.import_module("gear_sonic.envs.manager_env.robots.g1")
    r1 = importlib.import_module("gear_sonic.envs.manager_env.robots.r1")
    g1_pose = g1.G1_CYLINDER_MODEL_12_DEX_CFG.init_state.joint_pos
    for j in spec.MJCF_JOINT_ORDER:
        g1_scale = _lookup(g1.G1_MODEL_12_ACTION_SCALE, j)
        assert abs(_lookup(r1.R1_DEX3_ACTION_SCALE, j) - g1_scale) < 1e-9, j
        assert _lookup(r1.R1_DEX3_CFG.init_state.joint_pos, j) == _lookup(g1_pose, j), j


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
