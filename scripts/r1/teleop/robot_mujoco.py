"""The R1 + Dex3 in MuJoCo, driven like the real robot: joint PD targets from the policy.

This is the sim-to-sim plant for the runtime (PLAN.md gates G1/G2). ``r1_dex3.xml`` is the model
the motion library uses for kinematics; a floor is added. The motor driver's PD law runs at every
physics step (1 kHz by default), ``tau = kp (q* - q) - kd dq`` clipped to the effort limit, with the
training gains. Two actuator profiles:

* ``nominal``: what Isaac Lab trained with (armature 0.01, no joint friction, no passive damping);
* ``issue51``: unitree_rl_mjlab issue #51's fit of a real R1 (MuJoCo ``frictionloss``, N m):
  legs and waist armature 0.05 / friction 2.5, ankles 0.10 / 1.5, shoulder pitch and roll
  0.01 / 2.5, shoulder yaw, elbow and wrist roll 0.01 / 0.2.

Joint arrays at the interface are in Isaac Lab order (the policy's), quaternions (w, x, y, z).
"""

from __future__ import annotations

from pathlib import Path
import re

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MJCF = REPO / "gear_sonic/data/assets/robot_description/mjcf/r1_dex3.xml"

PROFILES = {
    "nominal": {".*": (0.01, 0.0)},
    "issue51": {  # joint regex: (armature, frictionloss N m)
        ".*_hip_.*_joint|.*_knee_joint|waist_.*_joint": (0.05, 2.5),
        ".*_ankle_.*_joint": (0.10, 1.5),
        ".*_shoulder_(pitch|roll)_joint": (0.01, 2.5),
        ".*_shoulder_yaw_joint|.*_elbow_joint|.*_wrist_roll_joint": (0.01, 0.2),
    },
}


def build_model(
    sim_dt: float = 0.001, floor_friction: float = 1.0, impratio: float = 10.0
) -> mujoco.MjModel:
    """The R1 on a floor. Friction: elliptic cones with ``impratio`` 10 (MuJoCo's default,
    pyramidal with impratio 1, is soft: a standing policy's steady sideways foot force made the
    feet creep apart ~3 mm/s, 56 mm in 20 s; PhysX and rubber soles hold). impratio 1 restores the
    soft contacts the gates used before 2026-09-25."""
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = sim_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if impratio > 1.0:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = impratio
    floor = spec.worldbody.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.friction = [floor_friction, 0.005, 0.0001]
    light = spec.worldbody.add_light()
    light.pos, light.dir = [0, 0, 4], [0, 0, -1]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    return spec.compile()


class MujocoR1:
    def __init__(
        self,
        meta: dict,
        profile: str = "nominal",
        sim_dt: float = 0.001,
        delay_s: float = 0.0,
        floor_friction: float = 1.0,
        kp_scale: float = 1.0,
    ):
        self.model = build_model(sim_dt, floor_friction)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.isaac_joints = list(meta["joint_names_isaaclab"])
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.isaac_joints]
        assert min(jid) >= 0, "joint missing from the MJCF"
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.isaac_joints]
        assert min(aid) >= 0, "actuator missing from the MJCF"
        self.act = np.array(aid)
        self.kp = np.asarray(meta["kp"], dtype=np.float64) * kp_scale
        self.kd = np.asarray(meta["kd"], dtype=np.float64)
        self.effort = np.asarray(meta["effort_limit"], dtype=np.float64)
        self.control_dt = float(meta["control_dt"])
        self.substeps = int(round(self.control_dt / sim_dt))
        self.delay_steps = int(round(delay_s / sim_dt))
        m.dof_damping[:] = 0.0
        for name, vadr in zip(self.isaac_joints, self.vadr):
            armature, friction = next(
                v for p, v in PROFILES[profile].items() if re.fullmatch(p, name)
            )
            m.dof_armature[vadr], m.dof_frictionloss[vadr] = armature, friction
        self.target = np.zeros(len(jid))
        self.previous_target = np.zeros(len(jid))
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # -- state -------------------------------------------------------------------------
    def reset(self, root_pos, root_quat, joint_pos, joint_vel=None, root_vel=None) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3], d.qpos[3:7] = root_pos, root_quat
        d.qpos[self.qadr] = joint_pos
        if joint_vel is not None:
            d.qvel[self.vadr] = joint_vel
        if root_vel is not None:  # (lin world, ang body)
            d.qvel[:6] = root_vel
        self.target[:] = joint_pos
        self.previous_target[:] = joint_pos
        mujoco.mj_forward(self.model, d)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    @property
    def root_pos(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    @property
    def root_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    @property
    def gyro(self) -> np.ndarray:
        """Pelvis angular velocity in the pelvis frame (free-joint qvel[3:6] is body-frame)."""
        return self.data.qvel[3:6].copy()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    # -- control -----------------------------------------------------------------------
    def step(self, target: np.ndarray) -> None:
        """Hold the joint targets for one control period (delayed by ``delay_s``)."""
        self.previous_target, self.target = self.target, np.asarray(target, dtype=np.float64)
        d = self.data
        for k in range(self.substeps):
            q_star = self.previous_target if k < self.delay_steps else self.target
            tau = self.kp * (q_star - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(tau, -self.effort, self.effort)
            mujoco.mj_step(self.model, d)
