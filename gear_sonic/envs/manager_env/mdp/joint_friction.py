"""Joint Coulomb friction as a torque (N m), randomized per environment: an R1 sim-to-real term.

unitree_rl_mjlab issue #51 fitted a real R1 (hanging unloaded; per-joint step responses and sine
sweeps, ~822k ``lowstate`` samples) with MuJoCo ``frictionloss``: legs and waist 2.5 N m, ankles
1.5, shoulder pitch/roll 2.5, shoulder yaw / elbow / wrist roll 0.2 (armature: legs/waist 0.05,
ankles 0.10, arms 0.01). Without it the simulated joints track PD targets exactly where the real
ones reach 0.68-0.95 of a step. Unloaded backdriving gives a lower bound for loaded legs, so #51
randomizes friction x1.0-2.0 upwards.

Isaac Lab's ``randomize_joint_parameters`` draws static, dynamic *and viscous* friction from the
same range, which would add a viscous term of the same magnitude. This term writes static =
dynamic = ``torque[joint] * U(scale_range)`` and viscous = 0. In Isaac Sim >= 5 these are joint
friction efforts; ``scripts/r1/isaac_joint_friction_test.py`` checks that they act in N m.
"""

from __future__ import annotations

import re

import torch


def joint_friction_torques(joint_names: list[str], torque: dict[str, float]) -> torch.Tensor:
    """Per-joint friction (N m) from ``{joint regex: torque}``; the first matching pattern wins."""
    values = []
    for name in joint_names:
        match = next((v for p, v in torque.items() if re.fullmatch(p, name)), None)
        values.append(0.0 if match is None else float(match))
    return torch.tensor(values)


def randomize_joint_friction_torque(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg,
    torque: dict[str, float],
    scale_range: tuple[float, float] = (1.0, 2.0),
):
    """Event term (use ``mode: startup``): static = dynamic friction = torque x U(scale), viscous 0."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    base = joint_friction_torques(list(asset.joint_names), dict(torque)).to(asset.device)
    lo, hi = scale_range
    scale = lo + (hi - lo) * torch.rand((len(env_ids), base.numel()), device=asset.device)
    friction = base[None] * scale
    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=friction,
        joint_dynamic_friction_coeff=friction,
        joint_viscous_friction_coeff=torch.zeros_like(friction),
        env_ids=env_ids,
    )
