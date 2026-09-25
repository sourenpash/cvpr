"""The R1 smoothness and "calm" reward terms (``mdp/smoothness.py``, PLAN.md D17/D18) on toy tensors."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from gear_sonic.tests.r1.conftest import REPO

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


@pytest.fixture(scope="module")
def sm():
    """``smoothness.py`` loaded by path with stand-ins for its two Isaac Lab imports."""
    stubs = {}
    try:  # real Isaac Lab, or another test's partial stub without ``managers``
        importlib.import_module("isaaclab.managers").ManagerTermBase  # noqa: B018
    except Exception:  # noqa: BLE001

        class ManagerTermBase:
            def __init__(self, cfg, env):
                self.cfg = cfg

        stubs["isaaclab"] = ModuleType("isaaclab")
        stubs["isaaclab.managers"] = ModuleType("isaaclab.managers")
        stubs["isaaclab.managers"].ManagerTermBase = ManagerTermBase
    rewards = ModuleType("gear_sonic.envs.manager_env.mdp.rewards")  # its package imports Isaac
    rewards._get_body_indexes = lambda command, names: [
        command.cfg.body_names.index(n) for n in names
    ]
    stubs[rewards.__name__] = rewards
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "r1_smoothness", REPO / "gear_sonic/envs/manager_env/mdp/smoothness.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def _feet_env(ref_z, ref_vel, robot_vel):
    """One env; bodies [pelvis, left foot, right foot]; per-foot reference height and velocities."""
    n = len(ref_z)
    pos = torch.zeros(1, 3, 3)
    pos[0, 1:, 2] = torch.tensor(ref_z)
    ref_v, rob_v = torch.zeros(1, 3, 3), torch.zeros(1, 3, 3)
    ref_v[0, 1 : n + 1] = torch.tensor(ref_vel)
    rob_v[0, 1 : n + 1] = torch.tensor(robot_vel)
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["pelvis", *FEET]),
        body_pos_w=pos,
        body_lin_vel_w=ref_v,
        robot_body_lin_vel_w=rob_v,
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
    )


def test_stance_foot_motion_penalizes_a_balance_step_only(sm):
    # left reference foot planted, robot lifts it at 0.5 m/s; right reference foot swings (free)
    env = _feet_env([0.058, 0.15], [[0, 0, 0], [0.8, 0, 0.3]], [[0, 0, 0.5], [0.2, 0, 0]])
    assert sm.stance_foot_motion(env, "motion", FEET).item() == pytest.approx(0.25)
    # a reference toe roll (slow ankle motion) that the robot copies costs nothing
    env = _feet_env([0.07, 0.058], [[0.1, 0, 0.05], [0, 0, 0]], [[0.1, 0, 0.05], [0, 0, 0]])
    assert sm.stance_foot_motion(env, "motion", FEET).item() == pytest.approx(0.0)
    # a reference foot sliding faster than ``speed`` is not "standing"
    env = _feet_env([0.058, 0.058], [[0.3, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0.1, 0]])
    assert sm.stance_foot_motion(env, "motion", FEET).item() == pytest.approx(0.01)


def test_joint_vel_error_counts_the_legs_only(sm):
    command = SimpleNamespace(
        joint_vel=torch.tensor([[1.0, 0.0, 2.0, 0.0]]),
        robot_joint_vel=torch.tensor([[1.5, 3.0, 2.0, 0.0]]),
        lower_joint_isaaclab_indices=[0, 2],
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    assert sm.joint_vel_error_l2(env, "motion").item() == pytest.approx(0.25)
    assert sm.joint_vel_error_l2(env, "motion", legs_only=False).item() == pytest.approx(9.25)


def test_action_acc_is_a_masked_second_difference(sm):
    manager = SimpleNamespace(action=torch.zeros(2, 3), prev_action=torch.zeros(2, 3))
    env = SimpleNamespace(action_manager=manager, num_envs=2, device="cpu")
    term = sm.action_acc_l2(SimpleNamespace(params={}), env)
    out = []
    for a in ([1.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0], [4.0, 0, 0]):  # a_t, one joint moving
        manager.prev_action, manager.action = manager.action, torch.tensor([a, a])
        out.append(term(env)[0].item())
    # first two steps masked; then (4 - 2*2 + 1)^2 = 1 and (4 - 2*4 + 2)^2 = 4
    assert out == pytest.approx([0.0, 0.0, 1.0, 4.0])
    term.reset(torch.tensor([1]))
    manager.prev_action, manager.action = manager.action, torch.zeros(2, 3)
    assert term(env).tolist() == pytest.approx([(0 - 8 + 4) ** 2, 0.0])
