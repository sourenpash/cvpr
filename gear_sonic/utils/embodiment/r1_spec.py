"""Unitree R1 (Basic/EDU body) + Dex3-1 hands: geometric and naming constants for SONIC.

Single source of truth shared by ``scripts/r1/build_r1_assets.py``, the robot config
(``gear_sonic/envs/manager_env/robots/r1.py``), the experiment preset and the tests.
No simulator imports.

Every value marked ``VERIFY`` is an estimate derived from public meshes/docs and must be
confirmed against the physical robot or Unitree CAD before hardware use (see PLAN.md §7).
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------
ROBOT_TYPE = "r1_dex3"  # key in modular_tracking_env_cfg.robot_mapping and robot.type in the preset
URDF_REL_PATH = "robot_description/urdf/r1/r1_dex3.urdf"
MJCF_FILE_NAME = "r1_dex3.xml"  # under gear_sonic/data/assets/robot_description/mjcf/

# Upstream URDF link names -> SONIC names (matches unitree_rl_mjlab's MJCF and SONIC's G1/H2).
LINK_RENAMES = {
    "pelvis_link": "pelvis",
    "waist_yaw_link": "torso_link",
}
ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
HEAD_BODY = "head_yaw_link"  # fixed URDF link; Isaac Lab fuses it into TORSO_BODY
LEFT_EE_BODY = "left_wrist_roll_link"
RIGHT_EE_BODY = "right_wrist_roll_link"
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
LEFT_PALM_BODY = "left_hand_palm_link"  # Dex3, fixed to LEFT_EE_BODY (exists in URDF only)
RIGHT_PALM_BODY = "right_hand_palm_link"

# Joints that exist on the R1 URDF but are held fixed for the policy (D1 in PLAN.md).
FIXED_JOINTS = ("head_pitch_joint", "head_yaw_joint")

# The 24 policy-controlled joints, in MuJoCo/MJCF (mjlab) order. Legs first is REQUIRED:
# commands.py hard-codes lower_joint_indices_mujoco = range(12).
MJCF_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)
NUM_DOF = len(MJCF_JOINT_ORDER)  # 24
NUM_BODIES = NUM_DOF + 1  # 25 (pelvis + one body per DOF)

# Wrist joints (roll only on R1). Indices are asserted against r1_ordering.py in the tests.
WRIST_JOINTS = ("left_wrist_roll_joint", "right_wrist_roll_joint")
WRIST_MUJOCO_DOF_INDICES = [
    18,
    23,
]  # motion_lib_cfg.wrist_mujoco_dof_indices (randomize_wrist_poses)
WRIST_ISAACLAB_DOF_INDICES = [
    22,
    23,
]  # observations.tokenizer.joint_pos_multi_future_wrist_for_smpl.joints_idx

# G1 joints (SONIC's g1_29dof MJCF order) that the R1 does not have. Used by the data
# transfer and checkpoint surgery. All 24 R1 joints exist in G1 under the same names.
G1_ONLY_JOINTS = (
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# --------------------------------------------------------------------------------------
# Dex3-1 mounting (VERIFY on hardware)
# --------------------------------------------------------------------------------------
# Palm frame origin in the wrist_roll_link frame. Derived from the R1 wrist mesh profile:
# the forearm tube (r~0.028 m) ends and the stock fist begins at x~0.080 m; the Dex3 palm
# base ring is r~0.028 m. G1 uses the analogous rule (palm at the wrist mesh flange face).
DEX3_MOUNT_XYZ = (0.080, 0.0, 0.0)
DEX3_MOUNT_RPY = (0.0, 0.0, 0.0)  # VERIFY: hand roll about the forearm axis may differ from G1

# Hand "grasp point" used for vr_3point / reward_point tracking, in the wrist_roll_link frame.
# Mirrors SONIC's G1 convention (wrist_yaw_link + [0.18, -/+0.025, 0]) where the palm base
# sits at x=0.0415: i.e. 0.1385 m beyond the palm base, 0.025 m lateral (thumb side).
_G1_HAND_POINT_BEYOND_PALM_BASE = 0.18 - 0.0415
HAND_POINT_X = round(DEX3_MOUNT_XYZ[0] + _G1_HAND_POINT_BEYOND_PALM_BASE, 4)  # 0.2185
LEFT_HAND_POINT_OFFSET = (HAND_POINT_X, -0.025, 0.0)
RIGHT_HAND_POINT_OFFSET = (HAND_POINT_X, 0.025, 0.0)

# Approximate Dex3-1 total mass per hand (palm + 7 finger links) from the Unitree URDF;
# spec sheets quote 0.65-0.71 kg. The stock-fist mass remains inside wrist_roll_link
# (conservative: slightly heavier forearm). VERIFY if Unitree publishes EDU wrist inertials.

# --------------------------------------------------------------------------------------
# Head / torso reference points (VERIFY; tunable)
# --------------------------------------------------------------------------------------
# In torso_link frame: head_pitch joint at z=0.255, head_yaw joint at z=0.3715, head top ~0.44.
# G1 uses torso_link + [0,0,0.35] for the VR head point and +0.5 for the reward torso point.
VR_HEAD_POINT_OFFSET = (0.0, 0.0, 0.32)
REWARD_TORSO_POINT_OFFSET = (0.0, 0.0, 0.45)

# --------------------------------------------------------------------------------------
# Default standing pose (unitree_rl_mjlab HOME_KEYFRAME; real-robot validated)
# --------------------------------------------------------------------------------------
# Feet-flat pelvis height at this pose is 0.7335 m (MuJoCo FK on r1_dex3.xml); spawn ~1 cm above.
INIT_POS_Z = 0.745
INIT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
}

# --------------------------------------------------------------------------------------
# Actuators (unitree_rl_mjlab r1_constants.py / deploy.yaml; real-robot validated for a
# velocity policy). Effort/velocity limits from the Unitree R1 URDF.
# --------------------------------------------------------------------------------------
ACTUATOR_GROUPS = {
    "legs": dict(
        joints=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
        stiffness=100.0,
        damping=2.0,
        effort_limit=60.0,
        velocity_limit=18.8,
        armature=0.01,
    ),
    "feet": dict(
        joints=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=50.0,
        velocity_limit=30.0,
        armature=0.01,
    ),
    "waist": dict(
        joints=["waist_roll_joint", "waist_yaw_joint"],
        stiffness=100.0,
        damping=2.0,
        effort_limit=60.0,
        velocity_limit=18.8,
        armature=0.01,
    ),
    "shoulders": dict(
        joints=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=60.0,
        velocity_limit=18.8,
        armature=0.01,
    ),
    "distal_arm": dict(
        joints=[".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
        stiffness=20.0,
        damping=1.0,
        effort_limit=33.0,
        velocity_limit=33.4,
        armature=0.01,
    ),
}
ACTION_SCALE_FACTOR = 0.25  # action_scale = factor * effort_limit / stiffness (as in g1.py)

# --------------------------------------------------------------------------------------
# Body sets used by the tracking configs (R1 equivalents of motion.yaml / preset entries)
# --------------------------------------------------------------------------------------
VR_3POINT_BODY = [LEFT_EE_BODY, RIGHT_EE_BODY, TORSO_BODY]
VR_3POINT_BODY_OFFSET = [
    list(LEFT_HAND_POINT_OFFSET),
    list(RIGHT_HAND_POINT_OFFSET),
    list(VR_HEAD_POINT_OFFSET),
]

REWARD_POINT_BODY_5PT = [ROOT_BODY, LEFT_EE_BODY, RIGHT_EE_BODY, LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
REWARD_POINT_BODY_OFFSET_5PT = [
    [0.0, 0.0, 0.0],
    list(LEFT_HAND_POINT_OFFSET),
    list(RIGHT_HAND_POINT_OFFSET),
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
# The v1.1 / release presets override reward_point_body with a 3-point set.
REWARD_POINT_BODY_3PT = [TORSO_BODY, LEFT_EE_BODY, RIGHT_EE_BODY]
REWARD_POINT_BODY_OFFSET_3PT = [list(REWARD_TORSO_POINT_OFFSET), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

# 14 tracked bodies (motion.yaml body_names) with G1 wrist_yaw -> R1 wrist_roll.
TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]
EE_TERMINATION_BODIES = [LEFT_FOOT_BODY, RIGHT_FOOT_BODY, LEFT_EE_BODY, RIGHT_EE_BODY]
ANTI_SHAKE_BODIES = [LEFT_EE_BODY, RIGHT_EE_BODY, TORSO_BODY]
UPPER_BODY_EVENT_BODY_REGEX = ".*wrist_roll.*|torso_link"
# Bodies whose contacts are NOT penalised: feet, end-effectors, elbows (as G1) plus the Dex3
# links, which are the intended contact surface for manipulation.
UNDESIRED_CONTACT_EXCLUDE_REGEX = (
    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    "(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
    "(?!left_elbow_link$)(?!right_elbow_link$)"
    "(?!left_hand_.*)(?!right_hand_.*).+$"
)
