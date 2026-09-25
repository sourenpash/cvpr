"""Reward terms of the R1 teleop smoothness stage (``sonic_r1_dex3_teleop_smooth.yaml``).

``RewardsCfg`` is a configclass with a fixed set of term fields, so the preset's extra terms need
fields of their own (the same pattern as ``r1_events.R1RobustEventCfg``).
"""

from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.rewards import RewardsCfg


@configclass
class R1SmoothRewardsCfg(RewardsCfg):
    """:class:`RewardsCfg` plus the smoothness and gesture terms (``mdp/smoothness.py``)."""

    action_acc_l2 = None
    action_rate_l2_scaled = None
    anti_shake_rel_ang_vel = None
    tracking_wrist_linvel = None
    stance_foot_motion = None
    leg_joint_vel_error = None
