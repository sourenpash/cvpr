"""The R1 joint-friction event (unitree_rl_mjlab #51 values) writes torques per joint group."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.tests.r1.conftest import REPO
from gear_sonic.utils.embodiment import r1_spec

# Loaded by path: the mdp package __init__ imports Isaac Lab.
_spec = importlib.util.spec_from_file_location(
    "joint_friction", REPO / "gear_sonic/envs/manager_env/mdp/joint_friction.py"
)
jf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jf)

ISSUE_51 = {  # N m, MuJoCo frictionloss fitted on a real R1
    ".*_hip_.*_joint": 2.5,
    ".*_knee_joint": 2.5,
    "waist_.*_joint": 2.5,
    ".*_ankle_.*_joint": 1.5,
    ".*_shoulder_(pitch|roll)_joint": 2.5,
    ".*_shoulder_yaw_joint": 0.2,
    ".*_elbow_joint": 0.2,
    ".*_wrist_roll_joint": 0.2,
}


def test_every_r1_joint_gets_its_group_torque():
    joints = list(r1_spec.MJCF_JOINT_ORDER)
    torques = dict(zip(joints, jf.joint_friction_torques(joints, ISSUE_51).tolist()))
    assert torques["left_hip_pitch_joint"] == 2.5 and torques["waist_yaw_joint"] == 2.5
    assert torques["right_ankle_roll_joint"] == 1.5
    assert torques["left_shoulder_roll_joint"] == 2.5 and torques[
        "right_elbow_joint"
    ] == pytest.approx(0.2)
    assert all(t > 0 for t in torques.values()), torques


def test_event_writes_static_equal_dynamic_and_no_viscous():
    writes = {}
    joints = list(r1_spec.MJCF_JOINT_ORDER)
    asset = SimpleNamespace(
        joint_names=joints,
        device="cpu",
        write_joint_friction_coefficient_to_sim=lambda **kw: writes.update(kw),
    )

    class Scene(dict):
        num_envs = 64

    env = SimpleNamespace(scene=Scene(robot=asset))
    jf.randomize_joint_friction_torque(
        env, None, SimpleNamespace(name="robot"), ISSUE_51, scale_range=(1.0, 2.0)
    )
    base = jf.joint_friction_torques(joints, ISSUE_51)
    static = writes["joint_friction_coeff"]
    assert static.shape == (64, len(joints))
    assert torch.equal(static, writes["joint_dynamic_friction_coeff"])
    assert torch.all(writes["joint_viscous_friction_coeff"] == 0)
    ratio = static / base
    assert ratio.min() >= 1.0 and ratio.max() <= 2.0
    assert len(writes["env_ids"]) == 64
