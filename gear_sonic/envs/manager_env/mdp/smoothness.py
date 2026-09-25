"""Smoothness terms for the R1 teleop policy (PLAN.md D17).

In deterministic MuJoCo rollouts of the B+ policy, ~14 % of arm joint-velocity power lies above
5 Hz against 0.5 % in the reference (jitter). Three reward-side fixes, besides the L2C2 loss
(``trl/trainer/ppo_trainer_smooth.py``) and a smaller exploration std:

* ``action_acc_l2``: sum (a_t - 2 a_{t-1} + a_{t-2})^2. Per unit power it penalizes 10 Hz ~480x
  more than 2 Hz (a first difference: ~22x), so 1-3 Hz gestures are left mostly alone.
* ``action_rate_l2_scaled``: action rate with per-joint multipliers. The policy's action scale
  is G1's (PLAN.md D10), so a radian of target change on the R1's hips, knees and waist costs
  5-13x less than with the R1's own 0.25 * effort / kp scale.
* ``anti_shake_rel_ang_vel``: wrist and torso angular velocity *relative to the reference*
  beyond a deadzone. SONIC's version penalizes |omega| > 1.5 rad/s, which misses small fast
  jitter (+-0.02 rad at 8 Hz is ~1 rad/s) and penalizes an intended 2 Hz wave (~6 rad/s).

"Calm" terms (stage B3, D18): as little motion as the reference does not ask for.

* ``stance_foot_motion``: a robot foot moving while the reference foot stands: balance steps,
  shuffles and sliding.
* ``joint_vel_error_l2``: leg joint velocity beyond (or short of) the reference's.
"""

from __future__ import annotations

import re

from isaaclab.managers import ManagerTermBase
import torch

from gear_sonic.envs.manager_env.mdp.rewards import _get_body_indexes


class action_acc_l2(ManagerTermBase):  # noqa: N801  (Isaac Lab term naming)
    """Squared second difference of the policy action; zero on an episode's first two steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.prev2 = torch.zeros_like(env.action_manager.action)
        self.age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self.age[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        a, a1 = env.action_manager.action, env.action_manager.prev_action
        acc = a - 2.0 * a1 + self.prev2
        out = torch.sum(acc * acc, dim=1) * (self.age >= 2)
        self.prev2 = a1.clone()
        self.age += 1
        return out


class action_rate_l2_scaled(ManagerTermBase):  # noqa: N801
    """sum_j m_j (a_t - a_{t-1})_j^2, ``m`` from ``{joint regex: multiplier}`` (default 1)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        term = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
        names = list(term._joint_names)  # noqa: SLF001
        multipliers = dict(cfg.params.get("multipliers", {}))
        m = [next((v for p, v in multipliers.items() if re.fullmatch(p, n)), 1.0) for n in names]
        self.m = torch.tensor(m, dtype=torch.float32, device=env.device)

    def __call__(self, env, multipliers=None, action_name="joint_pos"):  # noqa: ARG002
        da = env.action_manager.action - env.action_manager.prev_action
        return torch.sum(self.m * da * da, dim=1)


def stance_foot_motion(
    env,
    command_name: str,
    foot_names: list[str],
    height: float = 0.09,
    speed: float = 0.15,
) -> torch.Tensor:
    """sum over feet of |v_foot - v_foot_ref|^2 while the *reference* foot stands (weight < 0).

    A reference foot stands when its link is within ``height`` of the floor (standing, the R1's
    ankle-roll link is 0.055-0.06 m up) and moves slower than ``speed`` horizontally: 73 % of
    the eval clips' foot frames. The robot should then not lift or slide that foot: no balance
    steps or shuffles the reference does not take (user request 2026-09-25). Relative to the
    reference, so a heel or toe roll the reference makes costs nothing; the feet-slip and
    contact-mismatch terms of legged_gym, ASAP 2502.01143 and PBHC 2506.12851 play this role.
    """
    command = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, foot_names)
    ref_pos, ref_vel = command.body_pos_w[:, idx], command.body_lin_vel_w[:, idx]
    ground = env.scene.env_origins[:, 2:3]
    stance = ((ref_pos[..., 2] - ground) < height) & (ref_vel[..., :2].norm(dim=-1) < speed)
    err = command.robot_body_lin_vel_w[:, idx] - ref_vel
    return torch.sum(stance * torch.sum(err * err, dim=-1), dim=1)


def joint_vel_error_l2(env, command_name: str, legs_only: bool = True) -> torch.Tensor:
    """sum (dq - dq_ref)^2 over the leg joints (or all): motion the reference does not ask for."""
    command = env.command_manager.get_term(command_name)
    err = command.robot_joint_vel - command.joint_vel
    if legs_only:
        err = err[:, command.lower_joint_isaaclab_indices]
    return torch.sum(err * err, dim=1)


def anti_shake_rel_ang_vel(
    env, command_name: str, threshold: float = 0.5, body_names: list[str] | None = None
) -> torch.Tensor:
    """sum over bodies of max(|omega - omega_ref| - threshold, 0)^2 (use a negative weight)."""
    command = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    rel = command.robot_body_ang_vel_w[:, idx] - command.body_ang_vel_w[:, idx]
    excess = torch.clamp(rel.norm(dim=-1) - threshold, min=0.0)
    return torch.sum(excess * excess, dim=1)
