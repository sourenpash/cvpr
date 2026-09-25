"""Event terms of the R1 teleop robustness stage (``sonic_r1_dex3_teleop_robust.yaml``).

``EventCfg`` is a configclass with a fixed set of term fields, so the preset's extra terms need
fields of their own: measured armature (legs/waist, ankles), joint friction and PD-gain scaling.
"""

from isaaclab.utils import configclass

from gear_sonic.envs.manager_env.mdp.events import EventCfg


@configclass
class R1RobustEventCfg(EventCfg):
    """:class:`EventCfg` plus the R1 sim-to-real terms (all startup events)."""

    r1_leg_armature = None
    r1_ankle_armature = None
    r1_joint_friction = None
    actuator_gains = None
