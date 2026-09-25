"""The real Unitree R1 over DDS: ``rt/lowcmd`` / ``rt/lowstate`` (unitree_hg IDL) in debug mode.

In debug mode the R1's own balance controller is off and this process owns every motor
(BruteForce ``HARDWARE.md``). ``rt/arm_sdk`` is not an option: it stops the robot from walking
(xr_teleoperate #319). Enter debug mode from damping with L2 + R2 on the remote, or release the
motion service (``MotionSwitcherClient.ReleaseMode``) with the robot on the gantry.

Motor slots follow Unitree's R1 ``JointIndex`` (unitree_sdk2
``include/unitree/dds_wrapper/robots/r1/defines.h``): legs 0-11, waist roll 12, waist yaw 13,
left arm 15-19, right arm 22-26, head pitch 29, head yaw 30; slots 14, 20, 21, 27, 28 are empty
(unitree_rl_mjlab #52 is the bug of mapping the 24 joints contiguously). The head is not in the
policy's model; it is held at zero with low gains. VERIFY on the robot: move each joint by hand
in zero torque and watch ``python scripts/r1/teleop/robot_unitree.py --check <iface>``.

All DDS traffic runs in a child process: at 500 Hz it takes ``lowstate`` and publishes
``lowcmd`` (training PD gains, ``mode_pr`` = PR for the ankles, ``mode_machine`` copied from
``lowstate``, CRC), and at 100 Hz the Dex3 finger targets on ``rt/dex3/{left,right}/cmd``
(``hands.py``; the fingers hold a semi-closed pose unless the operator shapes them, and go limp
in damping). Python (de)serialization of these messages costs ~0.5 ms each, which in the
policy process would starve the 50 Hz loop of the GIL. The processes share the latest state
and the joint targets through shared memory. IMU quaternion (w, x, y, z), gyro in the body frame.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
import time

import numpy as np

R1_SLOTS = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_roll_joint": 12, "waist_yaw_joint": 13,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18, "left_wrist_roll_joint": 19,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25, "right_wrist_roll_joint": 26,
}  # fmt: skip
HEAD_SLOTS = (29, 30)  # pitch, yaw
MODE_PR = 0
PASSIVE, DAMPING, POSITION = 0, 1, 2

# unitree_sdk2py.utils.joystick: wireless_remote[2] / [3] bits, LSB first
REMOTE_BITS = {
    2: ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2"),
    3: ("A", "B", "X", "Y", "up", "right", "down", "left"),
}


def dds_init(domain: int, interface: str) -> None:
    """``ChannelFactoryInitialize``; on loopback (not multicast-capable) discover peers by unicast."""
    from unitree_sdk2py.core import channel, channel_config
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if interface == "lo":
        peers = (
            "<Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>20"
            '</MaxAutoParticipantIndex><Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
        )
        config = channel_config.ChannelConfigHasInterface.replace(
            "</General>", "</General>" + peers
        )
        channel.ChannelConfigHasInterface = config  # the factory reads the module-level template
    ChannelFactoryInitialize(domain, interface)


def parse_remote(raw) -> dict:
    """Buttons and sticks of the R1's wireless remote from ``lowstate.wireless_remote``."""
    b = bytes(bytearray(raw))
    out = {}
    for byte, names in REMOTE_BITS.items():
        for bit, name in enumerate(names):
            out[name] = bool((b[byte] >> bit) & 1)
    out["lx"], out["rx"], out["ry"] = struct.unpack("3f", b[4:16])
    out["ly"] = struct.unpack("f", b[20:24])[0]
    return out


CTX = mp.get_context("spawn")  # shared objects and the child must come from the same context


class _Shared:
    """Shared memory between the policy process and the DDS process."""

    def __init__(self, n: int):
        self.lock = CTX.Lock()
        self.target = CTX.Array("d", n, lock=False)
        self.command = CTX.Array("d", 2, lock=False)  # mode, kp scale
        self.q = CTX.Array("d", n, lock=False)
        self.dq = CTX.Array("d", n, lock=False)
        self.tau = CTX.Array("d", n, lock=False)
        self.temperature = CTX.Array("d", n, lock=False)
        self.imu = CTX.Array("d", 7, lock=False)  # quat (4), gyro (3)
        self.head_q = CTX.Array("d", 2, lock=False)
        self.remote = CTX.Array("B", 40, lock=False)
        self.hands = CTX.Array("d", 14, lock=False)  # Dex3 targets, motor order, left then right
        self.hand_mode = CTX.Value("i", PASSIVE, lock=False)
        self.t_state = CTX.Value("d", 0.0, lock=False)
        self.stop = CTX.Event()


def _dds_process(shared: _Shared, slots, kp, kd, head_kp, head_kd, damping_kd, domain, interface):
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from hands import KD as HAND_KD, KP as HAND_KP, ris_mode
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelPublisher
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__HandCmd_,
        unitree_hg_msg_dds__LowCmd_,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    dds_init(domain, interface)
    participant = ChannelFactory()._ChannelFactory__participant  # the SDK's configured participant
    # A plain reader, polled without blocking; keep-last-1 so only the newest state is decoded.
    reader = DataReader(
        participant, Topic(participant, "rt/lowstate", LowState_), Qos(Policy.History.KeepLast(1))
    )
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd, crc = unitree_hg_msg_dds__LowCmd_(), CRC()
    hand_pubs, hand_cmds = {}, {}
    for side in ("left", "right"):
        hand_pubs[side] = ChannelPublisher(f"rt/dex3/{side}/cmd", HandCmd_)
        hand_pubs[side].Init()
        hand_cmds[side] = unitree_hg_msg_dds__HandCmd_()
    state = None
    period, t_next, tick = 0.002, time.perf_counter(), 0
    while not shared.stop.is_set():
        samples = [x for x in reader.take(N=4) if isinstance(x, LowState_)]
        msg = samples[-1] if samples else None
        if msg is not None:
            state = msg
            ms = msg.motor_state
            with shared.lock:
                for k, s in enumerate(slots):
                    shared.q[k], shared.dq[k], shared.tau[k] = ms[s].q, ms[s].dq, ms[s].tau_est
                    shared.temperature[k] = max(ms[s].temperature)
                shared.imu[:4] = list(msg.imu_state.quaternion)
                shared.imu[4:] = list(msg.imu_state.gyroscope)
                shared.head_q[:] = [ms[s].q for s in HEAD_SLOTS]
                shared.remote[:] = bytes(bytearray(msg.wireless_remote))
                shared.t_state.value = time.monotonic()
        if state is not None:
            with shared.lock:
                mode, scale = int(shared.command[0]), shared.command[1]
                target = np.frombuffer(shared.target, dtype=np.float64).copy()
            if mode != PASSIVE:
                cmd.mode_pr, cmd.mode_machine = MODE_PR, state.mode_machine
                for k, s in enumerate(slots):
                    m = cmd.motor_cmd[s]
                    m.mode, m.dq, m.tau = 1, 0.0, 0.0
                    if mode == POSITION:
                        m.q, m.kp, m.kd = float(target[k]), float(scale * kp[k]), float(kd[k])
                    else:  # damping
                        m.q, m.kp, m.kd = float(state.motor_state[s].q), 0.0, damping_kd
                for s in HEAD_SLOTS:  # not in the policy: hold straight, damp in damping mode
                    m = cmd.motor_cmd[s]
                    m.mode, m.q, m.dq, m.tau, m.kd = 1, 0.0, 0.0, 0.0, head_kd
                    m.kp = head_kp if mode == POSITION else 0.0
                cmd.crc = crc.Crc(cmd)
                pub.Write(cmd)
        hand_mode = shared.hand_mode.value
        if hand_mode != PASSIVE and tick % 5 == 0:  # 100 Hz
            with shared.lock:
                hand_q = list(shared.hands[:])
            for h, side in enumerate(("left", "right")):
                for i, m in enumerate(hand_cmds[side].motor_cmd):
                    m.mode, m.dq, m.tau, m.kd = ris_mode(i), 0.0, 0.0, HAND_KD
                    m.q = float(hand_q[7 * h + i])
                    m.kp = HAND_KP if hand_mode == POSITION else 0.0
                hand_pubs[side].Write(hand_cmds[side])
        tick += 1
        t_next += period
        time.sleep(max(0.0, t_next - time.perf_counter()))


class UnitreeR1:
    realtime = True

    def __init__(
        self,
        meta: dict,
        interface: str,
        domain: int = 0,
        head_kp: float = 8.0,
        head_kd: float = 0.5,
        damping_kd: float = 3.0,
    ):
        names = meta["joint_names_isaaclab"]
        self.slots = [R1_SLOTS[n] for n in names]  # policy (Isaac) order -> motor slot
        self.n = len(self.slots)
        self.shared = _Shared(self.n)
        self.shared.command[:] = [PASSIVE, 1.0]
        self.process = CTX.Process(
            target=_dds_process,
            args=(self.shared, self.slots, list(map(float, meta["kp"])), list(map(float, meta["kd"])),
                  head_kp, head_kd, damping_kd, domain, interface),
            daemon=True, name="r1_dds",
        )  # fmt: skip
        self.process.start()

    def wait_for_state(self, timeout: float = 10.0) -> None:
        t0 = time.monotonic()
        while self.shared.t_state.value == 0.0:
            if time.monotonic() - t0 > timeout or not self.process.is_alive():
                raise TimeoutError("no rt/lowstate: robot on, cable in, right interface/domain?")
            time.sleep(0.01)

    @property
    def state_age(self) -> float:
        return time.monotonic() - self.shared.t_state.value

    def state(self) -> dict[str, np.ndarray]:
        s = self.shared
        with s.lock:
            out = {
                "q": np.array(s.q[:]),
                "dq": np.array(s.dq[:]),
                "tau": np.array(s.tau[:]),
                "temperature": np.array(s.temperature[:]),
                "quat": np.array(s.imu[:4]),
                "gyro": np.array(s.imu[4:]),
                "head_q": np.array(s.head_q[:]),
                "remote": bytes(s.remote[:]),
            }
        out["remote"] = parse_remote(out["remote"])
        out["pos"] = np.full(3, np.nan)  # no base position on the robot
        return out

    def command(self, target: np.ndarray, kp_scale: float = 1.0) -> None:
        with self.shared.lock:
            self.shared.target[:] = [float(x) for x in target]
            self.shared.command[:] = [POSITION, kp_scale]

    def hands(self, left: np.ndarray, right: np.ndarray) -> None:
        """Dex3 finger targets (``hands.FINGERS`` order per hand); starts the hand commands."""
        from hands import to_motor_order

        q = np.concatenate([to_motor_order(left, "left"), to_motor_order(right, "right")])
        with self.shared.lock:
            self.shared.hands[:] = [float(x) for x in q]
            self.shared.hand_mode.value = POSITION

    def damping(self) -> None:
        with self.shared.lock:
            self.shared.command[0] = DAMPING
            if self.shared.hand_mode.value != PASSIVE:
                self.shared.hand_mode.value = DAMPING

    def close(self) -> None:
        self.damping()
        time.sleep(0.05)
        self.shared.stop.set()
        self.process.join(timeout=2.0)


def check(interface: str, domain: int, seconds: float) -> None:
    """Read-only (nothing is commanded): joint angles by name and the IMU, twice a second."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    names = json.loads((root / "layouts/r1_dex3_teleop/layout.json").read_text())[
        "joint_names_isaaclab"
    ]
    robot = UnitreeR1(
        {"joint_names_isaaclab": names, "kp": [0] * 24, "kd": [0] * 24}, interface, domain
    )
    robot.wait_for_state()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        s = robot.state()
        joints = " ".join(f"{n.removesuffix('_joint')}={q:+.2f}" for n, q in zip(names, s["q"]))
        print(f"{joints} | quat {np.round(s['quat'], 3)} gyro {np.round(s['gyro'], 2)}", flush=True)
        time.sleep(0.5)
    robot.shared.stop.set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", metavar="IFACE", required=True, help="network interface to the robot"
    )
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    check(a.check, a.domain, a.seconds)
