#!/usr/bin/env python3
"""Which unit does Isaac Sim 5.1 use for articulation joint friction? A one-joint test.

unitree_rl_mjlab issue #51 fits the real R1 with MuJoCo ``frictionloss`` (a torque, N m): legs and
waist 2.5, ankles 1.5, shoulder pitch/roll 2.5, distal arm 0.2. Isaac Lab 2.3's actuator docstring
calls joint friction "unitless" (a coefficient times the spatial force the joint transmits), while
its ``write_joint_friction_coefficient_to_sim`` describes Isaac Sim >= 5 as static/dynamic friction
"efforts". This settles it before friction goes into training.

The robot hangs fixed at the pelvis with gravity off, all joints hold their pose, and the left elbow
gets kp = 10, kd = 1 with a target 0.1 rad away: a 1 N m PD torque at rest. With static = dynamic
friction f written to the elbow:

* friction is a torque   -> the elbow stops where kp * error = f: moves 0.1 - f / kp
  (f = 0.5 -> 0.05 rad, f >= 1 -> 0);
* friction is a coefficient of the transmitted force (~0 N here, no gravity, no load)
  -> the elbow moves the full 0.1 rad for every f.

    python scripts/r1/isaac_joint_friction_test.py --device cuda:1   # sonic-train env, ~2 min
"""

import argparse

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--device", default="cuda:0", help="an RTX GPU (asblab: cuda:1)")
DEVICE = _parser.parse_args().device
simulation_app = AppLauncher(headless=True, device=DEVICE).app

from isaaclab.assets import Articulation  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

from gear_sonic.envs.manager_env.robots.r1 import R1_DEX3_CFG  # noqa: E402

JOINT, KP, KD, OFFSET = "left_elbow_joint", 10.0, 1.0, 0.1


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=DEVICE))
    cfg = R1_DEX3_CFG.replace(prim_path="/World/Robot")
    cfg.spawn.fix_base = True
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.init_state.pos = (0.0, 0.0, 1.5)
    robot = Articulation(cfg)
    sim.reset()
    j = robot.joint_names.index(JOINT)
    q0 = robot.data.default_joint_pos.clone()
    kp = robot.data.joint_stiffness.clone()
    kd = robot.data.joint_damping.clone()
    kp[:, j], kd[:, j] = KP, KD
    robot.write_joint_stiffness_to_sim(kp)
    robot.write_joint_damping_to_sim(kd)
    print(
        f"Isaac Sim {'.'.join(map(str, sim.get_version()[:3]))}; {JOINT}: kp {KP}, target +{OFFSET}"
    )
    for f in (0.0, 0.5, 0.8, 2.0):
        friction = torch.zeros_like(q0)
        friction[:, j] = f
        robot.write_joint_friction_coefficient_to_sim(
            joint_friction_coeff=friction,
            joint_dynamic_friction_coeff=friction,
            joint_viscous_friction_coeff=torch.zeros_like(q0),
        )
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        target = q0.clone()
        target[:, j] += OFFSET
        robot.set_joint_position_target(target)
        for _ in range(400):  # 2 s
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        moved = float(robot.data.joint_pos[0, j] - q0[0, j])
        torque_model = max(0.0, OFFSET - f / KP)
        print(
            f"friction {f:.1f}: moved {moved:.4f} rad | torque model {torque_model:.4f}"
            f" | coefficient model {OFFSET:.4f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
