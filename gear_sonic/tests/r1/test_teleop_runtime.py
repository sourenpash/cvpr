"""The R1 teleop runtime (scripts/r1/teleop): observation math, targets IK, planner leg map."""

from __future__ import annotations

import glob
import json
import sys

import joblib
import mujoco
import numpy as np
import pytest

from gear_sonic.tests.r1.conftest import REPO

sys.path.insert(0, str(REPO / "scripts/r1/teleop"))
import observations as ob  # noqa: E402
import rotations as rot  # noqa: E402

LAYOUT = REPO / "layouts/r1_dex3_teleop/layout.json"


def _random_quats(n, seed=0):
    return rot.normalize(np.random.default_rng(seed).normal(size=(n, 4)))


def test_quaternion_helpers_match_mujoco():
    for q1, q2 in zip(_random_quats(20, 1), _random_quats(20, 2)):
        ref = np.zeros(4)
        mujoco.mju_mulQuat(ref, q1, q2)
        assert np.allclose(rot.multiply(q1, q2), ref)
        v, out = np.array([0.3, -0.2, 0.9]), np.zeros(3)
        mujoco.mju_rotVecQuat(out, v, q1)
        assert np.allclose(rot.apply(q1, v), out)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q1)
        assert np.allclose(rot.to_matrix(q1), mat.reshape(3, 3))


def test_heading_is_training_get_heading_q():
    """torch_transform.get_heading_q: zero the x, y components of (w, x, y, z), renormalize."""
    q = _random_quats(10)
    h = rot.heading(q)
    assert np.allclose(h[:, 1:3], 0) and np.allclose(np.linalg.norm(h, axis=1), 1)
    assert np.allclose(
        h[:, [0, 3]], q[:, [0, 3]] / np.linalg.norm(q[:, [0, 3]], axis=1, keepdims=True)
    )


def test_anchor_heading_term():
    ident = np.array([1.0, 0, 0, 0])
    assert np.allclose(ob.anchor_ori_heading(ident, ident), [1, 0, 0, 1, 0, 0])
    # the robot's own yaw is removed: same reference yaw as the robot -> identity
    q = rot.from_yaw(1.1)
    assert np.allclose(ob.anchor_ori_heading(q, q), [1, 0, 0, 1, 0, 0])
    # row-major first two columns of R_z(0.3)
    c, s = np.cos(0.3), np.sin(0.3)
    assert np.allclose(ob.anchor_ori_heading(ident, rot.from_yaw(0.3)), [c, -s, s, c, 0, 0])


def test_vr_terms_are_in_the_reference_pelvis_frame():
    anchor_q = rot.from_yaw(np.pi / 2)
    pos, quat = ob.vr_3point_local(
        np.array([1.0, 2.0, 0.7]),
        anchor_q,
        np.array([[1.0, 2.5, 0.7]] * 3),
        np.tile(anchor_q, (3, 1)),
    )
    assert np.allclose(pos.reshape(3, 3), [[0.5, 0, 0]] * 3)  # +y world = +x of a pelvis facing +y
    assert np.allclose(np.abs(quat.reshape(3, 4)[:, 0]), 1)


@pytest.mark.skipif(not LAYOUT.exists(), reason="layout.json is written on the GPU box")
def test_observation_builder_layout_and_history():
    layout = json.loads(LAYOUT.read_text())
    meta = {
        "input": {
            "tokenizer": [
                {"name": t["name"], "dim": int(np.prod(t["dims"]))}
                for t in layout["groups"]["tokenizer"]
                if t["name"] != "encoder_index"
            ],
            "actor_obs": [
                {"name": t["name"], "dim": t["dims"][0] // t["history"], "history": t["history"]}
                for t in layout["groups"]["policy"]
            ],
            "size": 267 + layout["obs_dims"]["actor_obs"],
        },
        "default_joint_pos": np.zeros(24).tolist(),
    }
    b = ob.ObservationBuilder(meta)
    tok = {name: np.zeros(dim) for name, dim in b.tokenizer_terms}
    first = b.proprioception(
        np.full(24, 1.0), np.zeros(24), [1, 0, 0, 0], np.zeros(3), np.zeros(24)
    )
    obs1 = b.build(first, tok)
    assert obs1.size == 1047
    jp = slice(267 + 30, 267 + 30 + 240)  # after base_ang_vel's 10 x 3
    assert np.allclose(obs1[jp], 1.0)  # every history slot filled with the first frame
    second = dict(first, joint_pos=np.full(24, 2.0))
    obs2 = b.build(second, tok)
    assert np.allclose(obs2[jp][:-24], 1.0) and np.allclose(obs2[jp][-24:], 2.0)  # oldest first
    assert np.allclose(obs2[-30:].reshape(10, 3), [0, 0, -1])  # gravity_dir, upright


def test_targets_calibration_identity_and_smooth_tracking():
    targets = pytest.importorskip("targets")
    vt = targets.VRTargets()
    head = np.eye(4)
    head[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])  # WebXR device facing +x
    head[:3, 3] = [0, 0, 1.6]
    left, right = np.eye(4), np.eye(4)
    left[:3, 3], right[:3, 3] = [0.3, 0.25, 1.0], [0.3, -0.25, 1.0]
    vt.calibrate(head, left, right)
    default = vt.terms(vt.ik.q0)
    assert np.allclose(
        vt.solve(head, left, right)["vr_3point_local_target"], default["vr_3point_local_target"]
    )
    # a slow reach: FK targets of a smooth joint trajectory are tracked by the warm-started IK
    ik = vt.ik
    ik.reset()
    errors = []
    for k in range(100):
        q = ik.q0.copy()
        q[ik.arm_q["left"]] += 0.6 * np.sin(np.pi * k / 100) * np.array([-1.0, 0.5, 0.3, 0.5, 0.4])
        f = ik.fk(q)
        ik.solve_arm("left", f["left_palm"], f["left_wrist_R"])
        errors.append(np.linalg.norm(ik.fk()["left_palm"] - f["left_palm"]))
    assert max(errors) < 0.005, max(errors)


@pytest.mark.skipif(
    not glob.glob(str(REPO / "data/motion_lib_r1/planner_v3_g1/*.pkl")), reason="planner data"
)
def test_runtime_leg_map_reproduces_the_training_clips():
    reference = pytest.importorskip("reference")
    legs = reference.LegMap()
    g1_path = sorted(glob.glob(str(REPO / "data/motion_lib_r1/planner_v3_g1/*.pkl")))[0]
    name = g1_path.split("/")[-1][:-4]
    g1 = joblib.load(g1_path)[name]
    r1 = joblib.load(REPO / f"data/motion_lib_r1/planner_v3/{name}.pkl")[name]
    assert np.allclose(legs(g1["dof"]), r1["dof"], atol=1e-6)


def test_r1_motor_slots_follow_unitrees_joint_index():
    """unitree_sdk2 dds_wrapper/robots/r1/defines.h; slots 14, 20, 21, 27, 28 are empty (#52)."""
    ru = pytest.importorskip("robot_unitree")
    from gear_sonic.utils.embodiment import r1_spec

    assert set(ru.R1_SLOTS) == set(r1_spec.MJCF_JOINT_ORDER)
    slots = [ru.R1_SLOTS[n] for n in r1_spec.MJCF_JOINT_ORDER]
    assert slots == [*range(0, 14), *range(15, 20), *range(22, 27)]
    assert ru.R1_SLOTS["waist_roll_joint"] == 12 and ru.R1_SLOTS["waist_yaw_joint"] == 13
    assert ru.HEAD_SLOTS == (29, 30) and not set(ru.HEAD_SLOTS) & set(slots)


def test_remote_parser_bits():
    ru = pytest.importorskip("robot_unitree")
    raw = bytearray(40)
    raw[2] = 0b0000_0100  # start
    raw[3] = 0b0000_0011  # A, B
    raw[4:8] = np.float32(0.5).tobytes()
    remote = ru.parse_remote(raw)
    assert remote["start"] and remote["A"] and remote["B"] and not remote["select"]
    assert remote["lx"] == pytest.approx(0.5)


def test_watchdog_trips():
    safety = pytest.importorskip("safety")
    w = safety.Watchdog()
    ok = {
        "q": np.zeros(24),
        "dq": np.zeros(24),
        "quat": np.array([1.0, 0, 0, 0]),
        "temperature": np.full(24, 40.0),
    }
    assert w.check(ok) == ""
    assert "joint speed" in w.check(dict(ok, dq=np.full(24, 40.0)))
    assert "tilt" in w.check(dict(ok, quat=rot.normalize(np.array([0.9, 0.44, 0, 0]))))
    assert "old" in w.check(ok, state_age=0.1)
    assert "non-finite" in w.check(ok, obs=np.array([np.nan]))
    assert "remote" in w.check(dict(ok, remote={"B": True}))


def test_joint_smoother_is_speed_limited_and_settles_without_overshoot():
    targets = pytest.importorskip("targets")
    dt = 0.02
    for tau in (0.04, 0.0):
        s = targets.JointSmoother(tau, np.array([3.0]))
        s.reset(np.zeros(1))
        q = np.array([s(np.ones(1), dt)[0] for _ in range(100)])
        steps = np.diff(np.r_[0.0, q])
        assert steps.max() <= 3.0 * dt + 1e-12 and q.max() <= 1.0 + 1e-9, (tau, steps.max())
        assert abs(q[-1] - 1.0) < 1e-3, (tau, q[-1])
    # a 1.5 Hz wave within the speed limit keeps most of its amplitude; 6 Hz shake does not
    s = targets.JointSmoother(0.04, np.array([10.0]))
    t = np.arange(400) * dt
    for hz, lo, hi in ((1.5, 0.8, 1.0), (6.0, 0.0, 0.5)):
        s.reset(np.zeros(1))
        out = np.array([s(np.array([0.3 * np.sin(2 * np.pi * hz * x)]), dt)[0] for x in t])
        assert lo < np.abs(out[200:]).max() / 0.3 < hi, (hz, np.abs(out[200:]).max())


def test_hand_shaper_holds_semi_closed_and_follows_grip_and_trigger():
    hands = pytest.importorskip("hands")
    from quest import QuestSample

    shaper = hands.HandShaper(tau=0.0)
    idle = QuestSample(t=0.0, head=np.eye(4), left=np.eye(4), right=np.eye(4))
    q = shaper(idle, 0.02)
    assert np.allclose(q["left"], hands.pose("hold", "left"))
    assert np.allclose(q["right"], hands.pose("hold", "right"))
    for trigger, grip, name in ((1.0, 1.0, "fist"), (0.0, 1.0, "point"), (1.0, 0.0, "pinch")):
        sample = QuestSample(
            t=0.0, head=np.eye(4), left=np.eye(4), right=np.eye(4),
            right_trigger=trigger, right_grip=grip, buttons={"X": True},
        )  # fmt: skip
        q = shaper(sample, 0.02)
        assert np.allclose(q["right"], hands.pose(name, "right")), name
        assert np.allclose(q["left"], hands.pose("open", "left"))  # X held: open hand
        assert np.allclose(
            shaper(sample, 0.02, engaged=False)["right"], hands.pose("hold", "right")
        )
    # filtered: a button press moves the fingers gradually
    shaper = hands.HandShaper(tau=0.1)
    fist = QuestSample(
        t=0.0, head=np.eye(4), left=np.eye(4), right=np.eye(4), right_trigger=1, right_grip=1
    )
    first = shaper(fist, 0.02)["right"]
    hold, goal = hands.pose("hold", "right"), hands.pose("fist", "right")
    assert 0.1 < np.linalg.norm(first - hold) / np.linalg.norm(goal - hold) < 0.3
    # DDS motor order and the RIS mode byte (xr_teleoperate robot_hand_unitree.py)
    q = np.arange(7.0)
    assert list(hands.to_motor_order(q, "left")) == list(q)
    assert list(hands.to_motor_order(q, "right")) == [0, 1, 2, 5, 6, 3, 4]
    assert hands.ris_mode(3) == 0x13 and hands.ris_mode(6, timeout=1) == 0x96


def test_hand_poses_within_the_dex3_limits():
    hands = pytest.importorskip("hands")
    import xml.etree.ElementTree as ET

    src = REPO / "third_party_assets/unitree_ros/robots/dexterous_hand_description/dex3_1"
    if not src.exists():
        pytest.skip("unitree_ros Dex3 description not checked out")
    for side in ("left", "right"):
        root = ET.parse(src / f"dex3_1_{side[0]}.urdf").getroot()
        limits = {
            j.get("name"): (
                float(j.find("limit").get("lower")),
                float(j.find("limit").get("upper")),
            )
            for j in root.iter("joint")
            if j.get("type") == "revolute"
        }
        for name in ("hold", "open", "fist", "point", "pinch"):
            for f, value in zip(hands.FINGERS, hands.pose(name, side)):
                lo, hi = limits[f"{side}_hand_{f}_joint"]
                assert lo - 1e-9 <= value <= hi + 1e-9, (side, name, f, value)


def test_planner_hold_keeps_the_stance_until_the_sticks_move():
    reference = pytest.importorskip("reference")
    pl = reference.pl
    try:
        pl.find_planner_onnx()
    except FileNotFoundError:
        pytest.skip("planner ONNX not downloaded")
    pytest.importorskip("onnxruntime")
    ref = reference.PlannerReference(device="cpu", threads=2, hold=True)
    try:
        idle = pl.Command(mode=pl.IDLE, face_dir=np.array([1.0, 0.0, 0.0]))
        terms = []
        for _ in range(60):
            ref.wait()
            terms.append(ref.step(idle)["command_multi_future_lower_body"])
        terms = np.array(terms)
        assert np.ptp(terms, axis=0).max() == 0.0 and np.abs(terms[:, 120:]).max() == 0.0
        walk = pl.Command(
            mode=pl.SLOW_WALK, speed=0.4, move_dir=np.array([1.0, 0, 0]), face_dir=idle.face_dir
        )
        moved = []
        for _ in range(60):
            ref.wait()
            moved.append(ref.step(walk)["command_multi_future_lower_body"])
        assert np.abs(moved[-1][:120] - terms[-1][:120]).max() > 0.05  # the legs step
    finally:
        ref.close()
