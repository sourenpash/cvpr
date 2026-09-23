"""Unit tests for the G1 -> R1 motion transfer on synthetic motions (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import transform

from gear_sonic.tests.r1.conftest import G1_MJCF, R1_MJCF, REPO
from gear_sonic.utils.embodiment import r1_spec as spec

mujoco = pytest.importorskip("mujoco")

from gear_sonic.data_process.transfer_g1_motion_lib_to_r1 import G1ToR1Transfer  # noqa: E402
from gear_sonic.utils.embodiment.ordering import parse_mjcf  # noqa: E402


@pytest.fixture(scope="module")
def tr() -> G1ToR1Transfer:
    return G1ToR1Transfer()


def make_g1_entry(T: int = 60, fps: float = 30.0, walk_speed: float = 0.6) -> dict:
    """Synthetic G1 motion_lib entry: standing pose, arm swing, forward walk translation."""
    g1 = parse_mjcf(G1_MJCF)
    t = np.arange(T) / fps
    dof = np.zeros((T, 29), dtype=np.float32)
    ji = {j: i for i, j in enumerate(g1.joints)}
    for side in ("left", "right"):
        dof[:, ji[f"{side}_hip_pitch_joint"]] = -0.3
        dof[:, ji[f"{side}_knee_joint"]] = 0.6
        dof[:, ji[f"{side}_ankle_pitch_joint"]] = -0.3
        dof[:, ji[f"{side}_elbow_joint"]] = 0.5 + 0.3 * np.sin(2 * np.pi * t)
    dof[:, ji["left_shoulder_pitch_joint"]] = 0.4 * np.sin(2 * np.pi * t)
    dof[:, ji["waist_pitch_joint"]] = 0.2  # G1-only joint, must be dropped
    dof[:, ji["left_wrist_yaw_joint"]] = 1.0  # G1-only joint, must be dropped
    dof[:, ji["left_wrist_roll_joint"]] = 0.25  # shared joint, distinctive value
    root_rot = transform.Rotation.from_euler("z", 0.3).as_quat().astype(np.float32)  # xyzw
    trans = np.zeros((T, 3), dtype=np.float32)
    trans[:, 0] = 1.0 + walk_speed * t
    trans[:, 1] = -2.0
    trans[:, 2] = 0.79
    return {
        "root_trans_offset": trans,
        "pose_aa": np.zeros((T, 30, 3), dtype=np.float32),
        "dof": dof,
        "root_rot": np.repeat(root_rot[None], T, 0),
        "smpl_joints": np.random.default_rng(0).normal(size=(T, 24, 3)).astype(np.float32),
        "fps": fps,
    }


def test_scale_and_joint_maps(tr):
    assert 0.85 < tr.scale < 0.95  # 0.5985 / 0.6564
    assert tr.r1_joints == list(spec.MJCF_JOINT_ORDER)
    g1 = parse_mjcf(G1_MJCF)
    for r1_i, g1_i in enumerate(tr.g1_to_r1_idx):
        assert g1.joints[g1_i] == tr.r1_joints[r1_i]
    dropped = {g1.joints[i] for i in range(29)} - set(tr.r1_joints)
    assert dropped == set(spec.G1_ONLY_JOINTS)


def test_transfer_shapes_and_passthrough(tr):
    entry = make_g1_entry()
    out, st = tr.transfer_entry("synthetic", entry)
    T = entry["dof"].shape[0]
    assert out["dof"].shape == (T, spec.NUM_DOF)
    assert out["pose_aa"].shape == (T, spec.NUM_BODIES, 3)
    assert out["root_trans_offset"].shape == (T, 3)
    assert np.array_equal(out["root_rot"], entry["root_rot"])
    assert np.array_equal(out["smpl_joints"], entry["smpl_joints"])
    assert out["fps"] == entry["fps"]
    assert st.num_frames == T and not st.dropped
    # joint values travel by name
    r1_i = tr.r1_joints.index("left_wrist_roll_joint")
    assert np.allclose(out["dof"][:, r1_i], 0.25)
    r1_e = tr.r1_joints.index("left_elbow_joint")
    g1_e = parse_mjcf(G1_MJCF).joints.index("left_elbow_joint")
    assert np.allclose(out["dof"][:, r1_e], entry["dof"][:, g1_e])


def test_pose_aa_consistent_with_dof_and_root(tr):
    entry = make_g1_entry(T=5)
    out, _ = tr.transfer_entry("synthetic", entry)
    # body 0 = root rotvec; bodies 1..24 = axis * angle in MJCF order
    rv = transform.Rotation.from_quat(entry["root_rot"]).as_rotvec()
    assert np.allclose(out["pose_aa"][:, 0], rv, atol=1e-6)
    m = mujoco.MjModel.from_xml_path(str(R1_MJCF))
    for j_i in range(spec.NUM_DOF):
        axis = m.jnt_axis[j_i + 1]
        assert np.allclose(
            out["pose_aa"][:, j_i + 1], axis[None] * out["dof"][:, j_i : j_i + 1], atol=1e-6
        )


def test_root_translation_scaled_about_first_frame_and_grounded(tr):
    entry = make_g1_entry(T=30)
    out, st = tr.transfer_entry("synthetic", entry)
    src, dst = entry["root_trans_offset"], out["root_trans_offset"]
    assert np.allclose(dst[0, :2], src[0, :2])  # start position kept
    disp_src = src[-1, :2] - src[0, :2]
    disp_dst = dst[-1, :2] - dst[0, :2]
    assert np.allclose(disp_dst, tr.scale * disp_src, atol=1e-5)
    # frame 0 stands on the ground (lowest foot capsule surface at z=0)
    z_low = tr.lowest_foot_z(dst[0], out["root_rot"][0], out["dof"][0])
    assert abs(z_low) < 1e-4
    assert abs(st.ground_shift_m) < 0.2


def test_cli_end_to_end_on_pkl_tree(tmp_path):
    """Run the CLI on a tiny synthetic G1 PKL tree; check outputs, layout mirroring and report."""
    import json
    import subprocess
    import sys

    import joblib

    src = tmp_path / "g1"
    (src / "sessionA").mkdir(parents=True)
    (src / "sessionB").mkdir(parents=True)
    joblib.dump({"walk_001": make_g1_entry(T=20)}, src / "sessionA" / "walk_001.pkl")
    bad = make_g1_entry(T=20)
    bad["dof"][
        :, parse_mjcf(G1_MJCF).joints.index("left_knee_joint")
    ] = 3.0  # always clamped -> dropped
    joblib.dump({"bad_001": bad}, src / "sessionB" / "bad_001.pkl")
    out = tmp_path / "r1"
    cmd = [
        sys.executable,
        str(REPO / "gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py"),
        "--input",
        str(src),
        "--output",
        str(out),
        "--num_workers",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert (out / "sessionA" / "walk_001.pkl").exists()
    assert not (out / "sessionB" / "bad_001.pkl").exists()
    report = json.loads((out / "transfer_report.json").read_text())
    assert report["num_motions"] == 2 and report["num_kept"] == 1 and report["num_dropped"] == 1
    entry = joblib.load(out / "sessionA" / "walk_001.pkl")["walk_001"]
    assert entry["dof"].shape == (20, spec.NUM_DOF) and entry["pose_aa"].shape == (
        20,
        spec.NUM_BODIES,
        3,
    )


def test_clamping_and_velocity_statistics(tr):
    entry = make_g1_entry(T=40)
    g1 = parse_mjcf(G1_MJCF)
    k = g1.joints.index("left_knee_joint")
    entry["dof"][:20, k] = 3.0  # beyond R1 knee limit (2.426) for half the frames
    out, st = tr.transfer_entry("clamped", entry)
    r1_k = tr.r1_joints.index("left_knee_joint")
    assert out["dof"][:, r1_k].max() <= tr.upper[r1_k] + 1e-6
    assert abs(st.clamp_frac - 0.5) < 1e-6
    assert "left_knee_joint" in st.clamped_joints
    assert st.max_limit_excess_rad > 0.5
    # a single-frame jump in a shared joint trips the velocity check
    entry2 = make_g1_entry(T=40)
    e = g1.joints.index("right_elbow_joint")
    entry2["dof"][20, e] += 2.0  # 2 rad in 1/30 s = 60 rad/s > 33.4 rad/s
    _, st2 = tr.transfer_entry("fast", entry2)
    assert st2.vel_frac > 0 and st2.max_vel_ratio > 1.0
