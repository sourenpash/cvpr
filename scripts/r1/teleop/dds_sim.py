#!/usr/bin/env python3
"""The R1 in MuJoCo behind ``rt/lowcmd`` / ``rt/lowstate``: the real-robot code path without the robot.

``run.py --robot unitree --interface lo --domain 1`` talks to this process exactly as it talks
to the R1: the same DDS topics, message types, R1 motor slots (``robot_unitree.R1_SLOTS``; the
head slots are reported at zero) and CRC. Physics runs in real time at 1 kHz; each step applies
the motor driver's law ``tau = kp (q* - q) + kd (dq* - dq) + tau_ff`` from the latest LowCmd,
clipped to the effort limits; ``lowstate`` is published at 500 Hz. ``--gantry`` holds the pelvis
on an elastic band (as unitree_mujoco's), released after ``--release-after`` seconds.

    python scripts/r1/teleop/dds_sim.py --onnx-meta <export>.json --profile issue51 --gantry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_mujoco import MujocoR1  # noqa: E402
from robot_unitree import HEAD_SLOTS, R1_SLOTS, dds_init  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx-meta", type=Path, required=True, help="the export's .json")
    ap.add_argument("--profile", choices=["nominal", "issue51"], default="issue51")
    ap.add_argument("--interface", default="lo")
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--gantry", action="store_true")
    ap.add_argument("--release-after", type=float, default=1e9)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    meta = json.loads(args.onnx_meta.read_text())
    robot = MujocoR1(meta, profile=args.profile)
    m, d = robot.model, robot.data
    names = meta["joint_names_isaaclab"]
    slots = np.array([R1_SLOTS[n] for n in names])
    robot.reset(
        [0, 0, float(meta["init_pos_z"]) + 0.02],
        [1, 0, 0, 0],
        np.asarray(meta["default_joint_pos"]),
    )

    dds_init(args.domain, args.interface)
    lock = threading.Lock()
    latest = {"cmd": None}

    def on_cmd(msg):
        with lock:
            latest["cmd"] = msg

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(on_cmd, 10)
    pub = ChannelPublisher("rt/lowstate", LowState_)
    pub.Init()
    state = unitree_hg_msg_dds__LowState_()
    crc = CRC()
    band = np.array([0.0, 0.0, float(meta["init_pos_z"]) + 0.06])  # stiff band: feet on the floor
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(m, d)
    print(
        f"dds_sim: {args.profile} actuators, domain {args.domain} on {args.interface}", flush=True
    )
    t0 = time.perf_counter()
    k = 0
    print("dds_sim: waiting for the first lowcmd (physics paused)", flush=True)
    while latest["cmd"] is None:  # publish the initial state so the runtime can connect
        for i, s in enumerate(slots):
            state.motor_state[s].q = float(d.qpos[robot.qadr[i]])
        state.imu_state.quaternion = [float(x) for x in d.qpos[3:7]]
        state.mode_machine = 5
        state.crc = crc.Crc(state)
        pub.Write(state)
        time.sleep(0.002)
    t0 = time.perf_counter() - d.time
    while True:
        with lock:
            cmd = latest["cmd"]
        q, dq = d.qpos[robot.qadr], d.qvel[robot.vadr]
        tau = np.zeros(len(slots))
        if cmd is not None:
            mc = cmd.motor_cmd
            for i, s in enumerate(slots):
                c = mc[s]
                if c.mode:
                    tau[i] = c.kp * (c.q - q[i]) + c.kd * (c.dq - dq[i]) + c.tau
        d.ctrl[robot.act] = np.clip(tau, -robot.effort, robot.effort)
        d.xfrc_applied[robot.pelvis] = 0.0
        if args.gantry and d.time < args.release_after:  # harness: pelvis position and uprightness
            d.xfrc_applied[robot.pelvis, :3] = 5000.0 * (band - d.qpos[:3]) - 300.0 * d.qvel[:3]
            w, x, y, z = d.qpos[3:7]
            tilt = 2.0 * np.array([w * x + y * z, w * y - x * z, 0.0])  # small-angle roll/pitch
            omega_w = d.xmat[robot.pelvis].reshape(3, 3) @ d.qvel[3:6]
            d.xfrc_applied[robot.pelvis, 3:] = -2000.0 * tilt - 100.0 * omega_w
        if cmd is None and not args.gantry:
            d.qvel[:] = 0.0  # frozen until the runtime's first lowcmd
            ahead = 0.0
        mujoco.mj_step(m, d)
        k += 1
        if k % 2 == 0:  # 500 Hz lowstate
            for i, s in enumerate(slots):
                ms = state.motor_state[s]
                ms.q, ms.dq, ms.tau_est = (
                    float(d.qpos[robot.qadr[i]]),
                    float(d.qvel[robot.vadr[i]]),
                    float(tau[i]),
                )
            for s in HEAD_SLOTS:
                state.motor_state[s].q = 0.0
            state.imu_state.quaternion = [float(x) for x in d.qpos[3:7]]
            state.imu_state.gyroscope = [float(x) for x in d.qvel[3:6]]
            state.mode_machine = 5
            state.tick = k
            state.crc = crc.Crc(state)
            pub.Write(state)
        if viewer is not None and k % 20 == 0:
            viewer.sync()
        if k % 2000 == 0:
            rtf = d.time / max(1e-9, time.perf_counter() - t0)
            print(f"dds_sim: t {d.time:.1f} s, real-time factor {rtf:.2f}", flush=True)
        # real time
        ahead = d.time - (time.perf_counter() - t0)
        if ahead > 0:
            time.sleep(ahead)


if __name__ == "__main__":
    main()
