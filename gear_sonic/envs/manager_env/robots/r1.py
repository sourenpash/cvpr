"""Unitree R1 (Basic/EDU body, 24 policy DOF) with two Dex3-1 hands - Isaac Lab robot config.

Structure mirrors ``g1.py`` / ``h2.py``. All geometric/naming constants come from
``gear_sonic.utils.embodiment.r1_spec`` and the ordering maps from the auto-generated
``r1_ordering.py`` (``scripts/r1/build_r1_assets.py``).

Actuator gains: unitree_rl_mjlab's real-robot-validated PD values (see r1_spec.ACTUATOR_GROUPS).
SONIC's G1 config instead derives kp/kd from per-motor armature and a 10 Hz natural frequency;
we do not have R1 rotor inertias, so mjlab's numbers are the starting point. If training is
unstable at num_envs=16 (robot explodes / falls immediately), reduce stiffness or increase
damping per group (docs/source/user_guide/new_embodiments.md, "KP/KD tuning").

Default pose and action scale are G1's for the shared joints (r1_spec.INIT_JOINT_POS /
ACTION_SCALE, PLAN.md D10): the policy is warm-started from G1 weights gathered by joint name.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from gear_sonic.envs.manager_env.robots.r1_ordering import (  # noqa: F401  (re-exported)
    R1_ISAACLAB_JOINTS,
    R1_ISAACLAB_TO_MUJOCO_BODY,
    R1_ISAACLAB_TO_MUJOCO_DOF,
    R1_ISAACLAB_TO_MUJOCO_MAPPING,
    R1_MUJOCO_TO_ISAACLAB_BODY,
    R1_MUJOCO_TO_ISAACLAB_DOF,
)
from gear_sonic.utils.embodiment import r1_spec

ASSET_DIR = "gear_sonic/data/assets"


def _actuator(group: dict) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=list(group["joints"]),
        effort_limit_sim=group["effort_limit"],
        velocity_limit_sim=group["velocity_limit"],
        stiffness=group["stiffness"],
        damping=group["damping"],
        armature=group["armature"],
    )


R1_DEX3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/{r1_spec.URDF_REL_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, r1_spec.INIT_POS_Z),
        joint_pos=dict(r1_spec.INIT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={name: _actuator(group) for name, group in r1_spec.ACTUATOR_GROUPS.items()},
)

# action_scale[joint_expr]: G1's value for the same joint (r1_spec.ACTION_SCALE, PLAN.md D10),
# so the warm-started G1 output layer commands the same joint targets.
R1_DEX3_ACTION_SCALE = dict(r1_spec.ACTION_SCALE)
