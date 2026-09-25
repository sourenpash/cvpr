#!/usr/bin/env python3
"""The R1 teleop runtime: Quest (or a scripted operator) -> planner + VR targets -> policy -> robot.

At 50 Hz:

1. input: headset + controller poses, thumbsticks, buttons (``quest.py``, or ``ScriptedOperator``
   for unattended sim tests, the terminal keyboard (``keyboard.py``), or a recorded session);
2. lower body: the left/right sticks drive SONIC's kinematic planner (``reference.py``), a
   velocity command (direction and speed relative to the robot's facing) plus a yaw rate;
3. upper body: calibrated Quest poses -> IK-projected VR 3-point targets (``targets.py``),
   smoothed and speed-limited in joint space;
4. the observation, exactly as in training (``observations.py``), the ONNX policy
   (``policy.py``), joint targets ``default + scale * clip(a)``;
5. the robot: MuJoCo (``robot_mujoco.py``) or the real R1 (``robot_unitree.py``), with the
   Dex3 fingers semi-closed or shaped by the controllers' grip and trigger (``hands.py``).

Calm defaults (user request 2026-09-25): walking at up to 0.5 m/s and turning at 0.6 rad/s
(SONIC's gamepad: 0.8 m/s, 1 rad/s), arm joints at up to 3 rad/s, and the planner holds the
default stance until the sticks first move (its first plan would shuffle the feet otherwise).

Operator protocol: stand in the robot's default pose (upper arms down, forearms forward-down),
press A + X to calibrate and engage; targets ramp in over 1 s; B + Y disengages (damping on the
real robot). Face the same direction the whole time and turn the robot with the right stick.

    python scripts/r1/teleop/run.py --onnx <export>.onnx --robot mujoco --input script:walk_reach \\
        --seconds 60 --record logs_rl/sessions/test.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hands import HandShaper  # noqa: E402
import observations as ob  # noqa: E402
from policy import TeleopPolicy  # noqa: E402
from quest import FACING_X, QuestSample, device_pose  # noqa: E402
import reference as rf  # noqa: E402
import rotations as rot  # noqa: E402
import targets as tg  # noqa: E402

RAMP_S = 1.0


# ------------------------------------------------------------------------------------------
# A scripted operator (a virtual Quest), for closed-loop tests without a headset
# ------------------------------------------------------------------------------------------


@dataclass
class ScriptedOperator:
    """Standing operator in the calibration pose; hands and sticks follow a named program.

    Programs: ``stand``, ``reach`` (reach grid, both hands), ``walk`` (forward, stop, turn in
    place, sidestep, back), ``walk_reach`` (walking while waving/reaching), ``demo``.
    """

    program: str = "walk_reach"
    seed: int = 0
    engage_at: float = 0.0  # A + X pressed for 0.2 s here; the program's clock starts then
    head0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.60]))
    hand0: dict = field(
        default_factory=lambda: {
            "left": np.array([0.30, 0.22, 1.02]),
            "right": np.array([0.30, -0.22, 1.02]),
        }
    )

    def sticks(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros(2)
        if self.program in ("stand", "reach"):
            return z, z
        # (t_start, left stick, right stick)
        schedule = [
            (0.0, z, z), (3.0, np.array([0.0, 0.8]), z), (7.0, z, z), (9.0, z, np.array([0.6, 0.0])),
            (12.0, z, z), (13.0, np.array([0.7, 0.0]), z), (16.0, z, z), (17.0, np.array([0.0, -0.6]), z),
            (20.0, z, z), (21.0, np.array([0.0, 0.9]), np.array([-0.3, 0.0])), (26.0, z, z),
        ]  # fmt: skip
        period = 28.0
        tt = t % period
        left, right = z, z
        for t0, ls, rs in schedule:
            if tt >= t0:
                left, right = ls, rs
        return left, right

    def hands(self, t: float) -> dict[str, np.ndarray]:
        out = {s: self.hand0[s].copy() for s in ("left", "right")}
        if self.program in ("reach", "walk_reach", "demo"):
            phase = 2 * np.pi * t / 6.0
            # right hand: wave up-down and in-out; left hand: slower forward/up reach
            out["right"] += [
                0.18 * (1 - np.cos(phase)),
                -0.05 * np.sin(phase),
                0.30 * (1 - np.cos(phase)) / 2,
            ]
            out["left"] += [
                0.25 * (1 - np.cos(phase / 2)) / 2,
                0.08 * np.sin(phase / 2),
                0.15 * np.sin(phase / 2),
            ]
        return out

    def read(self, t: float) -> QuestSample:
        press = self.engage_at <= t < self.engage_at + 0.2
        t = max(0.0, t - self.engage_at)
        hands = self.hands(t)
        ls, rs = self.sticks(t)
        yaw = 0.15 * np.sin(2 * np.pi * t / 8.0) if self.program in ("walk_reach", "demo") else 0.0
        head = device_pose(self.head0, tg.yaw_matrix(yaw) @ FACING_X)
        return QuestSample(
            t=time.monotonic(), head=head, left=device_pose(hands["left"], FACING_X),
            right=device_pose(hands["right"], FACING_X), left_stick=ls, right_stick=rs,
            buttons={"A": press, "X": press},
        )  # fmt: skip


class ReplayOperator:
    """A recorded session (``run.py --record``) played back tick by tick, calibration included."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.z = {k: z[k] for k in ("head", "left", "right", "left_stick", "right_stick")}
        self.calib = {k: z[f"calib_{k}"] for k in ("head", "left", "right")}
        self.n = len(self.z["head"])
        self.seeds = list(z["planner_seeds"]) if "planner_seeds" in z.files else []
        self.k = -1  # first the calibration sample (A + X), then the recorded ticks

    def read(self, t: float) -> QuestSample:
        k, self.k = self.k, self.k + 1
        if k < 0:
            c = self.calib
            return QuestSample(
                t=time.monotonic(), head=c["head"], left=c["left"], right=c["right"],
                buttons={"A": True, "X": True},
            )  # fmt: skip
        k, z = min(k, self.n - 1), self.z
        return QuestSample(
            t=time.monotonic(), head=z["head"][k], left=z["left"][k], right=z["right"][k],
            left_stick=z["left_stick"][k], right_stick=z["right_stick"][k],
        )  # fmt: skip


# ------------------------------------------------------------------------------------------
# Robots
# ------------------------------------------------------------------------------------------
class MujocoRobot:
    """``robot_mujoco.MujocoR1`` behind the runtime's robot interface (simulated time)."""

    realtime = False

    def __init__(
        self, meta: dict, profile: str = "issue51", delay_ms: float = 10.0, viewer: bool = False
    ):
        from robot_mujoco import MujocoR1

        self.sim = MujocoR1(meta, profile=profile, delay_s=delay_ms / 1e3)
        default = np.asarray(meta["default_joint_pos"])
        self.sim.reset([0, 0, float(meta["init_pos_z"])], [1, 0, 0, 0], default)
        self.viewer = None
        if viewer:
            from mujoco import viewer as mj_viewer

            self.viewer = mj_viewer.launch_passive(self.sim.model, self.sim.data)

    def state(self) -> dict[str, np.ndarray]:
        s = self.sim
        return {
            "q": s.joint_pos,
            "dq": s.joint_vel,
            "quat": s.root_quat,
            "gyro": s.gyro,
            "pos": s.root_pos,
        }

    def command(self, target: np.ndarray) -> None:
        self.sim.step(target)
        if self.viewer is not None:
            self.viewer.sync()

    def hands(self, left: np.ndarray, right: np.ndarray) -> None:
        pass  # the model's fingers are fixed in the hold pose

    def show_targets(self, vr_local: np.ndarray, offsets: np.ndarray) -> None:
        """Viewer markers: VR targets (red, placed at the robot's pelvis) and the palms (blue)."""
        if self.viewer is None:
            return
        import mujoco

        pos, quat = self.sim.root_pos, self.sim.root_quat
        points = [(pos + rot.apply(quat, p), (1, 0, 0, 0.8)) for p in vr_local.reshape(3, 3)]
        for side, off in zip(("left", "right"), offsets):
            p, q = self.sim.body_pose(f"{side}_wrist_roll_link")
            points.append((p + rot.apply(q, off), (0, 0.3, 1, 0.8)))
        scn = self.viewer.user_scn
        scn.ngeom = 0
        for p, rgba in points:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                np.asarray(p, dtype=np.float64), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32),
            )  # fmt: skip
            scn.ngeom += 1

    def palms_in_pelvis(self, offsets: np.ndarray) -> np.ndarray:
        pos, quat = self.sim.root_pos, self.sim.root_quat
        out = []
        for side, off in zip(("left", "right"), offsets):
            p, q = self.sim.body_pose(f"{side}_wrist_roll_link")
            out.append(rot.apply(rot.inverse(quat), p + rot.apply(q, off) - pos))
        return np.array(out)

    def fallen(self) -> bool:
        up = rot.apply(self.sim.root_quat, np.array([0.0, 0.0, 1.0]))[2]
        return bool(self.sim.root_pos[2] < 0.45 or up < np.cos(np.radians(50)))

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


class UnitreeRobot:
    """``robot_unitree.UnitreeR1`` (the R1, or ``dds_sim.py``) behind the same interface."""

    realtime = True

    def __init__(self, meta: dict, interface: str, domain: int = 0):
        from robot_unitree import UnitreeR1

        self.r1 = UnitreeR1(meta, interface, domain)
        self.r1.wait_for_state()

    def state(self) -> dict[str, np.ndarray]:
        return self.r1.state()

    @property
    def state_age(self) -> float:
        return self.r1.state_age

    def command(self, target: np.ndarray) -> None:
        self.r1.command(target)

    def hands(self, left: np.ndarray, right: np.ndarray) -> None:
        self.r1.hands(left, right)

    def damping(self) -> None:
        self.r1.damping()

    def palms_in_pelvis(self, offsets: np.ndarray):
        return None  # no base pose on the robot

    def fallen(self) -> bool:
        return False  # the watchdog's tilt check covers the real robot

    def close(self) -> None:
        self.r1.close()


# ------------------------------------------------------------------------------------------
# The runtime
# ------------------------------------------------------------------------------------------
class Runtime:
    def __init__(
        self,
        policy: TeleopPolicy,
        robot,
        planner_device: str = "cpu",
        scale: float = 0.65,
        sync_planner: bool = False,
        max_speed: float = 0.5,
        max_yaw_rate: float = 0.6,
        smooth_tau: float = 0.04,
        max_arm_speed: float = 3.0,
        max_waist_speed: float = 1.0,
        planner_hold: bool = True,
    ):
        self.policy, self.meta, self.robot = policy, policy.meta, robot
        self.builder = ob.ObservationBuilder(self.meta)
        self.default = np.asarray(self.meta["default_joint_pos"])
        self.action_scale = np.asarray(self.meta["action_scale"])
        self.clip = float(self.meta["action_clip"])
        self.dt = float(self.meta["control_dt"])
        self.reference = rf.PlannerReference(device=planner_device, hold=planner_hold)
        lo = rf.StickState.speed_range[0]
        self.sticks = rf.StickState(max_yaw_rate=max_yaw_rate, speed_range=(lo, max(lo, max_speed)))
        self.targets = tg.VRTargets(
            scale=scale,
            smooth_tau=smooth_tau,
            max_arm_speed=max_arm_speed,
            max_waist_speed=max_waist_speed,
        )
        self.hands = HandShaper()
        self.default_vr = self.targets.terms(self.targets.ik.q0)
        self.sync_planner = sync_planner
        self.engaged = False
        self.t_engage = 0.0
        self.last_action = np.zeros(len(self.default))
        self.log: dict[str, list] = {}

    def engage(self, sample: QuestSample, t: float) -> None:
        self.targets.calibrate(sample.head, sample.left, sample.right)
        self.reference.engage(self.robot.state()["quat"])
        self.builder.reset()
        self.last_action[:] = 0.0
        self.engaged, self.t_engage = True, t

    def step(self, sample: QuestSample, t: float) -> dict:
        state = self.robot.state()
        cmd = self.sticks.command(*sample.left_stick, sample.right_stick[0], self.dt)
        if self.sync_planner:
            self.reference.wait()
        ref = self.reference.step(cmd)
        vr = self.targets.solve(sample.head, sample.left, sample.right, dt=self.dt)
        alpha = float(np.clip((t - self.t_engage) / RAMP_S, 0.0, 1.0))
        if alpha < 1.0:  # ramp the targets in from the default pose (joint space of the IK)
            q = (1 - alpha) * self.targets.ik.q0 + alpha * vr["q"]
            vr = self.targets.terms(q)
        fingers = self.hands(sample, self.dt)
        tok = {
            "motion_anchor_ori_heading": ob.anchor_ori_heading(state["quat"], ref["ref_root_quat"]),
            "command_multi_future_lower_body": ref["command_multi_future_lower_body"],
            "vr_3point_local_target": vr["vr_3point_local_target"],
            "vr_3point_local_orn_target": vr["vr_3point_local_orn_target"],
        }
        proprio = self.builder.proprioception(
            state["q"], state["dq"], state["quat"], state["gyro"], self.last_action
        )
        obs = self.builder.build(proprio, tok)
        t0 = time.perf_counter()
        action = np.clip(self.policy(obs), -self.clip, self.clip)
        infer_ms = 1e3 * (time.perf_counter() - t0)
        self.last_action = action
        target = self.default + self.action_scale * action
        rec = {
            "t": t, "head": sample.head, "left": sample.left, "right": sample.right,
            "left_stick": sample.left_stick, "right_stick": sample.right_stick,
            "cmd_mode": cmd.mode, "cmd_speed": cmd.speed, "cmd_move": cmd.move_dir, "cmd_face": cmd.face_dir,
            "ref_g1_qpos": ref["g1_qpos"], "obs": obs, "action": action, "q_ik": vr["q"],
            "q_ik_raw": vr["q_ik"], "hand_left": fingers["left"], "hand_right": fingers["right"],
            "robot_q": state["q"], "robot_dq": state["dq"], "robot_quat": state["quat"], "robot_pos": state["pos"],
            "infer_ms": infer_ms, "planner_latency_ticks": self.reference.loop.last_latency_ticks,
            "target": target, "wall": time.perf_counter(),
        }  # fmt: skip
        for k, v in rec.items():
            self.log.setdefault(k, []).append(np.asarray(v))
        return rec

    def save(self, path: Path, extra: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self.log.items()}
        arrays["planner_seeds"] = np.asarray(self.reference.loop.seeds)
        c = self.targets.calib
        if c is not None:
            arrays.update(calib_head=c.head, calib_left=c.left, calib_right=c.right)
        np.savez_compressed(path, meta=json.dumps(extra), **arrays)

    def close(self) -> None:
        self.reference.close()


STAND_UP_S, BLEND_S = 3.0, 2.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--robot", choices=["mujoco", "unitree"], default="mujoco")
    ap.add_argument("--interface", default="lo", help="unitree: network interface to the robot")
    ap.add_argument("--domain", type=int, default=1, help="unitree: DDS domain (robot: 0)")
    ap.add_argument(
        "--input",
        default="script:walk_reach",
        help="quest | keys | script:<program> | replay:<npz>",
    )
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--profile", choices=["nominal", "issue51"], default="issue51")
    ap.add_argument("--delay-ms", type=float, default=10.0)
    ap.add_argument("--planner-device", default="cpu", help="cpu | cuda:<i>")
    ap.add_argument("--policy-device", default="cpu")
    ap.add_argument(
        "--sync-planner", action="store_true", help="wait for each plan (offline sim tests)"
    )
    ap.add_argument("--realtime", action="store_true", help="pace the sim at wall-clock 50 Hz")
    ap.add_argument("--shadow-s", type=float, default=0.0, help="compute but do not send actions")
    ap.add_argument(
        "--auto-start", action="store_true", help="unitree: stand up without the remote's start"
    )
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--record", type=Path, default=None)
    calm = ap.add_argument_group("calm motion (0 disables a limit)")
    calm.add_argument("--max-speed", type=float, default=0.5, help="walking, m/s")
    calm.add_argument("--max-yaw-rate", type=float, default=0.6, help="turning, rad/s")
    calm.add_argument("--smooth-tau", type=float, default=0.04, help="arm/waist target filter, s")
    calm.add_argument("--max-arm-speed", type=float, default=3.0, help="arm targets, rad/s")
    calm.add_argument("--max-waist-speed", type=float, default=1.0, help="waist targets, rad/s")
    calm.add_argument(
        "--no-planner-hold", action="store_true", help="use the planner's first idle plan"
    )
    args = ap.parse_args()
    for name in ("max_arm_speed", "max_waist_speed"):
        if getattr(args, name) <= 0:
            setattr(args, name, np.inf)

    from safety import Watchdog

    policy = TeleopPolicy(args.onnx, device=args.policy_device)
    if args.robot == "mujoco":
        robot = MujocoRobot(policy.meta, args.profile, args.delay_ms, viewer=args.viewer)
        robot.realtime = args.realtime or args.viewer or args.input in ("quest", "keys")
    else:
        robot = UnitreeRobot(policy.meta, args.interface, args.domain)
    if args.input == "quest":
        from quest import QuestInput

        quest = QuestInput()
        read = lambda t: quest.read()  # noqa: E731
    elif args.input.startswith("replay:"):
        operator = ReplayOperator(Path(args.input.split(":", 1)[1]))
        read = operator.read
    elif args.input == "keys":
        from keyboard import KeyboardOperator

        operator = KeyboardOperator(engage_at=0.0 if args.robot == "mujoco" else STAND_UP_S + 1.0)
        read = operator.read
    else:
        operator = ScriptedOperator(
            program=args.input.split(":", 1)[1],
            engage_at=0.0 if args.robot == "mujoco" else STAND_UP_S + 1.0,
        )
        read = operator.read
    runtime = Runtime(
        policy, robot, planner_device=args.planner_device, sync_planner=args.sync_planner,
        max_speed=args.max_speed if args.max_speed > 0 else 0.8,
        max_yaw_rate=args.max_yaw_rate if args.max_yaw_rate > 0 else 1.0,
        smooth_tau=args.smooth_tau, max_arm_speed=args.max_arm_speed,
        max_waist_speed=args.max_waist_speed, planner_hold=not args.no_planner_hold,
    )  # fmt: skip
    if args.input.startswith("replay:") and operator.seeds:  # the recorded planner seeds
        seeds = iter(operator.seeds)
        rng = runtime.reference.model.rng
        runtime.reference.model.rng = type(
            "Replay",
            (),
            {"integers": lambda self, *a, **k: next(seeds, int(rng.integers(0, 2**31 - 1)))},
        )()
    watchdog = Watchdog()
    offsets = np.asarray(policy.meta["vr_3point_body_offset"])[:2]
    default = runtime.default
    phase = "damping" if args.robot == "unitree" else "hold"
    t, palm_err, fell, stopped = 0.0, [], False, ""
    t_phase, stand_from, last_sample = 0.0, None, None
    wall0 = time.perf_counter()
    print(f"[run] phase {phase}", flush=True)

    def enter(new: str) -> None:
        nonlocal phase, t_phase
        phase, t_phase = new, t
        print(f"[run] {t:6.2f} s: phase {new}", flush=True)

    try:
        while t < args.seconds:
            sample = read(t)
            if last_sample is not None and robot.realtime and time.monotonic() - sample.t > 0.5:
                # Quest stream stale: freeze the targets, release the sticks
                sample = last_sample
                sample.left_stick, sample.right_stick = np.zeros(2), np.zeros(2)
            last_sample = sample
            state = robot.state()
            if args.robot == "unitree":
                why = watchdog.check(state, state_age=robot.state_age)
                if why and phase != "damping":
                    robot.damping()
                    stopped = why
                    print(f"[run] SAFETY: {why} -> damping", flush=True)
                    break
            if sample.buttons.get("B") and sample.buttons.get("Y") and runtime.engaged:
                stopped = "operator disengaged (B + Y)"
                if args.robot == "unitree":
                    robot.damping()
                break
            if phase == "damping":
                robot.damping()
                if args.auto_start or state.get("remote", {}).get("start"):
                    stand_from = state["q"].copy()
                    enter("stand_up")
            elif phase == "stand_up":
                a = min(1.0, (t - t_phase) / STAND_UP_S)
                robot.command((1 - a) * stand_from + a * default)
                robot.hands(*runtime.hands(sample, runtime.dt, engaged=False).values())
                if a >= 1.0:
                    enter("hold")
            elif phase == "hold":
                robot.command(default)
                robot.hands(*runtime.hands(sample, runtime.dt, engaged=False).values())
                if sample.valid and sample.buttons.get("A") and sample.buttons.get("X"):
                    runtime.engage(sample, t)
                    enter(
                        "shadow"
                        if args.shadow_s > 0
                        else ("blend" if args.robot == "unitree" else "run")
                    )
            else:
                rec = runtime.step(sample, t)
                if args.robot == "unitree":
                    why = watchdog.check(state, rec["obs"], rec["action"], robot.state_age)
                    if why:
                        robot.damping()
                        stopped = why
                        print(f"[run] SAFETY: {why} -> damping", flush=True)
                        break
                if phase != "shadow":
                    robot.hands(rec["hand_left"], rec["hand_right"])
                if phase == "shadow":
                    robot.command(default)
                    if t - t_phase >= args.shadow_s:
                        enter("blend" if args.robot == "unitree" else "run")
                elif phase == "blend":
                    b = min(1.0, (t - t_phase) / BLEND_S)
                    robot.command((1 - b) * default + b * rec["target"])
                    if b >= 1.0:
                        enter("run")
                else:
                    robot.command(rec["target"])
                if hasattr(robot, "show_targets"):
                    robot.show_targets(rec["obs"][246:255], offsets)
                palms = robot.palms_in_pelvis(offsets)
                if palms is not None and t - runtime.t_engage > RAMP_S:
                    targets = rec["obs"][246:252].reshape(2, 3)  # vr_3point_local_target palms
                    palm_err.append(np.linalg.norm(palms - targets, axis=1).max())
            if robot.fallen():
                fell = True
                break
            t += runtime.dt
            if robot.realtime:
                time.sleep(max(0.0, wall0 + t - time.perf_counter()))
    except KeyboardInterrupt:
        stopped = "Ctrl-C"
        if args.robot == "unitree":
            robot.damping()
    log = runtime.log
    summary = {
        "robot": args.robot, "input": args.input, "profile": args.profile, "delay_ms": args.delay_ms,
        "seconds": round(t, 2), "phase": phase, "fell": fell, "stopped": stopped,
        "palm_err_p50_mm": 1e3 * float(np.median(palm_err)) if palm_err else None,
        "palm_err_p95_mm": 1e3 * float(np.percentile(palm_err, 95)) if palm_err else None,
        "infer_ms_p50": float(np.median(log["infer_ms"])) if log else None,
        "planner_latency_ticks_max": int(np.max(log["planner_latency_ticks"])) if log else None,
        "wall_s": round(time.perf_counter() - wall0, 1),
    }  # fmt: skip
    print(json.dumps(summary))
    if args.record and log:
        runtime.save(args.record, summary)
    runtime.close()
    robot.close()


if __name__ == "__main__":
    main()
