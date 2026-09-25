#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201
"""Transfer Unitree G1 (29-DOF) motion_lib data to Unitree R1 (24-DOF) by joint name.

All 24 R1 joints exist on the G1 with the same names and axis conventions, so a fast,
key-preserving first-pass R1 dataset is obtained by:

1. selecting the 24 shared joint angles (G1 MJCF order -> R1 MJCF order),
2. clamping to the R1 joint limits,
3. scaling root translation by the leg-length ratio (R1 0.5985 m / G1 0.6564 m, hip pitch to
   ankle roll, straight leg) about the first frame,
4. contact correction (default; ``--no-contact-fix`` keeps the old frame-0 grounding): foot
   contacts are detected on the G1 source (heel/toe sole point below 2.5 cm and slower than
   0.15 m/s, ProtoMotions' SEED converter / PHUMA 2510.26236), the root's horizontal velocity is
   taken from the stance feet so they do not slide (Kovar et al. 2002 footskate cleanup), and
   the root height is set per frame so stance soles touch z=0 (PBHC 2506.12851, ProtoMotions),
   smoothed, with no sole below -5 mm; per-clip quality metrics go to the report,
5. rebuilding ``pose_aa`` from the R1 joint axes,
6. passing ``root_rot``, ``smpl_joints`` and ``fps`` through unchanged.

The same leg proportions that make identical joint angles produce different foot paths on the
R1 (thigh/shin ratios differ from G1's) left 7.5 % of frames with a sole > 2 cm below ground
and 6x the G1 source's stance skating before the contact correction.

Motion keys are preserved, so the Bones-SEED SMPL data (``smpl_filtered``) still lines up
for the SMPL encoder, and ``filter_and_copy_bones_data.py`` keyword filters still apply.
Motions that need heavy clamping or exceed R1 joint-velocity limits are reported and
(optionally) dropped.

Inputs (auto-detected):
  * a directory tree of G1 motion_lib PKLs (joblib dict {name: entry}); the relative layout is
    preserved in the output directory;
  * Bones-SEED G1 CSVs (flat directory or parent of session directories), converted on the fly
    with the upstream ``convert_soma_csv_to_motion_lib`` loaders.

Usage:
    python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \
        --input data/motion_lib_bones_seed/robot_filtered --output data/motion_lib_r1/robot \
        --num_workers 16 --max-clamp-frac 0.05 --max-vel-frac 0.05

    python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \
        --input /path/to/bones_seed/g1/csv --output data/motion_lib_r1/robot \
        --fps 30 --fps_source 120 --num_workers 16
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import sys

import joblib
import numpy as np
from scipy.spatial import transform

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "gear_sonic/data/assets/robot_description"
G1_MJCF = ASSETS / "mjcf/g1_29dof_rev_1_0.xml"
R1_MJCF = ASSETS / "mjcf/r1_dex3.xml"


@dataclass
class TransferStats:
    name: str
    num_frames: int
    fps: float
    clamp_frac: float  # fraction of frames with at least one clamped joint
    max_limit_excess_rad: float  # largest |excess| beyond the R1 limits before clamping
    vel_frac: float  # fraction of frames with at least one joint over the R1 velocity limit
    max_vel_ratio: float  # max |qdot| / limit
    ground_shift_m: float  # applied z shift after leg scaling (frame 0)
    height_fix_max_m: float = 0.0  # largest per-frame root height correction (contact fix)
    contact_frac: float = 0.0  # frames with at least one foot in contact (G1 source)
    frac_below_2cm: float = 0.0  # frames with a sole > 2 cm below ground (result)
    min_sole_z_m: float = 0.0  # lowest sole point (result)
    skate_median_mps: float = 0.0  # horizontal speed of contact points while in contact (result)
    skate_p95_mps: float = 0.0
    stance_float_frac: float = 0.0  # stance frames whose lowest contact point is > 1 cm up
    dropped: bool = False
    reason: str = ""
    clamped_joints: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Foot-contact correction
# --------------------------------------------------------------------------------------
CONTACT_HEIGHT_M = 0.025  # sole point below this ... (PHUMA 2510.26236 contact band)
CONTACT_SPEED_MPS = 0.15  # ... and slower than this (ProtoMotions' BONES-SEED G1 converter)
MIN_GAP_S, MIN_SEGMENT_S = 0.05, 0.10  # fill shorter gaps, drop shorter contact runs
SOFT_WINDOW_S = 0.17  # contact weights: centred mean of the labels
HEIGHT_SMOOTH_S = 0.04  # Gaussian sigma of the per-frame height correction
PENETRATION_TOL_M = 0.005  # no sole point below -5 mm after the correction


def _odd(n: float) -> int:
    return max(1, int(round(n)) | 1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _clean_contacts(c: np.ndarray, fps: float) -> np.ndarray:
    """Fill gaps shorter than MIN_GAP_S, then drop contact runs shorter than MIN_SEGMENT_S."""
    c = c.copy()
    for s, e in _runs(~c):
        if s > 0 and e < len(c) and (e - s) < MIN_GAP_S * fps:
            c[s:e] = True
    for s, e in _runs(c):
        if (e - s) < MIN_SEGMENT_S * fps:
            c[s:e] = False
    return c


def foot_sole_points(m, side: str) -> tuple[int, np.ndarray]:
    """(ankle-roll body id, [heel, toe] sole points in its frame) from the foot's collision geoms."""
    import mujoco

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, size, down = m.geom_pos[g], m.geom_size[g], np.array([0.0, 0.0, m.geom_size[g][0]])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        rot = rot.reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(c - down)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts += [c + rot[:, 2] * size[1] - down, c - rot[:, 2] * size[1] - down]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append(c + rot @ (np.array([sx, sy, -1.0]) * size))
    if not pts:
        raise ValueError(f"no collision geoms on {side}_ankle_roll_link")
    pts = np.asarray(pts)
    z = pts[:, 2].min()
    return bid, np.array([[pts[:, 0].min(), 0.0, z], [pts[:, 0].max(), 0.0, z]])


class G1ToR1Transfer:
    """Joint-name transfer of G1 motion_lib entries to the R1 (24 DOF) MJCF."""

    def __init__(
        self,
        g1_mjcf: Path = G1_MJCF,
        r1_mjcf: Path = R1_MJCF,
        soft_limit_factor: float = 1.0,
        contact_fix: bool = True,
    ):
        import mujoco

        from gear_sonic.utils.embodiment import r1_spec
        from gear_sonic.utils.embodiment.ordering import parse_mjcf

        self._mujoco = mujoco
        g1 = parse_mjcf(g1_mjcf)
        r1 = parse_mjcf(r1_mjcf)
        assert tuple(r1.joints) == r1_spec.MJCF_JOINT_ORDER
        missing = [j for j in r1.joints if j not in g1.joints]
        if missing:
            raise ValueError(f"R1 joints missing from G1 MJCF: {missing}")
        self.r1_joints = list(r1.joints)
        self.g1_joints = list(g1.joints)
        self.g1_to_r1_idx = np.array([g1.joints.index(j) for j in r1.joints], dtype=np.int64)
        self.r1_axes = np.array([r1.joint_axes[j] for j in r1.joints], dtype=np.float32)  # (24, 3)

        self.m_r1 = mujoco.MjModel.from_xml_path(str(r1_mjcf))
        self.d_r1 = mujoco.MjData(self.m_r1)
        m_g1 = mujoco.MjModel.from_xml_path(str(g1_mjcf))
        # Joint limits (R1), optionally tightened.
        rng = self.m_r1.jnt_range[1:].copy()  # skip free joint
        mid, half = rng.mean(1), 0.5 * (rng[:, 1] - rng[:, 0])
        self.lower = mid - soft_limit_factor * half
        self.upper = mid + soft_limit_factor * half
        # Velocity limits per joint from the actuator groups in r1_spec.
        self.vel_limit = np.zeros(len(self.r1_joints), dtype=np.float32)
        for j_i, j in enumerate(self.r1_joints):
            for grp in r1_spec.ACTUATOR_GROUPS.values():
                if any(re.fullmatch(p, j) for p in grp["joints"]):
                    self.vel_limit[j_i] = grp["velocity_limit"]
        assert (self.vel_limit > 0).all()

        self.scale = self._leg_length(self.m_r1) / self._leg_length(m_g1)
        self._foot_geoms = [
            i
            for i in range(self.m_r1.ngeom)
            if "foot" in (mujoco.mj_id2name(self.m_r1, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        ]
        assert self._foot_geoms, "R1 MJCF has no *foot* collision geoms"
        self.contact_fix = contact_fix
        self.m_g1, self.d_g1 = m_g1, mujoco.MjData(m_g1)
        self.feet_g1 = [foot_sole_points(m_g1, side) for side in ("left", "right")]
        self.feet_r1 = [foot_sole_points(self.m_r1, side) for side in ("left", "right")]

    def _leg_length(self, m) -> float:
        mujoco = self._mujoco
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        hip = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")]
        ankle = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")]
        return float(np.linalg.norm(hip - ankle))

    def lowest_foot_z(self, root_pos, root_quat_xyzw, dof) -> float:
        """Lowest foot-capsule surface height for one frame (MuJoCo FK)."""
        mujoco, m, d = self._mujoco, self.m_r1, self.d_r1
        q = root_quat_xyzw
        d.qpos[:3] = root_pos
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[7:] = dof
        mujoco.mj_kinematics(m, d)
        return min(float(d.geom_xpos[i][2] - m.geom_size[i][0]) for i in self._foot_geoms)

    def _sole_trajectory(self, m, d, feet, root_pos, root_quat_xyzw, dof) -> np.ndarray:
        """World heel/toe sole points per frame, shape (T, 2 feet, 2 points, 3)."""
        mujoco = self._mujoco
        quat = np.asarray(root_quat_xyzw, dtype=np.float64)[:, [3, 0, 1, 2]]
        out = np.empty((len(root_pos), 2, 2, 3))
        n = dof.shape[1]
        for t in range(len(root_pos)):
            d.qpos[:3], d.qpos[3:7], d.qpos[7 : 7 + n] = root_pos[t], quat[t], dof[t]
            mujoco.mj_kinematics(m, d)
            for f, (bid, pts) in enumerate(feet):
                out[t, f] = d.xpos[bid] + pts @ d.xmat[bid].reshape(3, 3).T
        return out

    def contact_fix_root(
        self, entry, trans, root_rot, dof_r1, fps
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Pin stance feet and ground stance soles by moving only the root (see module doc).

        Returns the corrected root translation (T, 3), the per-frame height correction (T,) and
        quality metrics of the result.
        """
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d

        g1_dof = np.asarray(entry["dof"], dtype=np.float64)[:, : self.m_g1.nq - 7]
        src = np.asarray(entry["root_trans_offset"], dtype=np.float64)
        sole_g1 = self._sole_trajectory(self.m_g1, self.d_g1, self.feet_g1, src, root_rot, g1_dof)
        pos = uniform_filter1d(sole_g1, _odd(0.04 * fps), axis=0, mode="nearest")
        speed = np.linalg.norm(np.gradient(pos[..., :2], 1.0 / fps, axis=0), axis=-1)
        contact = (pos[..., 2] < CONTACT_HEIGHT_M) & (speed < CONTACT_SPEED_MPS)  # (T, 2, 2)
        for f in range(2):
            for k in range(2):
                contact[:, f, k] = _clean_contacts(contact[:, f, k], fps)
        w = uniform_filter1d(contact.astype(np.float64), _odd(SOFT_WINDOW_S * fps), axis=0)

        trans = np.asarray(trans, dtype=np.float64)
        sole = self._sole_trajectory(self.m_r1, self.d_r1, self.feet_r1, trans, root_rot, dof_r1)
        # Horizontal: a stance foot's world position root + r must not move, so the root moves by
        # -dr (forward differences: exact pinning in single support); swing phases keep the scaled
        # source root velocity; double support averages the feet.
        dr = np.diff(sole[..., :2] - trans[:, None, None, :2], axis=0)  # (T-1, 2, 2, 2)
        wm = 0.5 * (w[1:] + w[:-1])  # (T-1, 2, 2)
        foot_w = wm.max(axis=2)  # (T-1, 2)
        foot_dr = (wm[..., None] * dr).sum(axis=2) / np.maximum(wm.sum(axis=2), 1e-9)[..., None]
        total = foot_w.sum(axis=1)
        step = -(foot_w[..., None] * foot_dr).sum(axis=1) / np.maximum(total, 1.0)[:, None]
        step += (1.0 - np.minimum(total, 1.0))[:, None] * np.diff(trans[:, :2], axis=0)
        new = trans.copy()
        new[1:, :2] = trans[0, :2] + np.cumsum(step, axis=0)

        # Vertical: stance frames put their lowest stance point on z=0; swing frames interpolate
        # (a jump keeps its arc); smoothed; lifted wherever a sole would still be below -5 mm.
        z = sole[..., 2]
        stance = contact.any(axis=(1, 2))
        frames = np.arange(len(z))
        if stance.any():
            lowest = np.where(contact, z, np.inf).min(axis=(1, 2))
            d = np.interp(frames, frames[stance], -lowest[stance])
        else:
            d = np.full(len(z), -z.min())
        sigma = max(HEIGHT_SMOOTH_S * fps, 0.5)
        d = gaussian_filter1d(d, sigma, mode="nearest")
        z_min = z.min(axis=(1, 2))
        lift = np.maximum(0.0, -(z_min + d) - PENETRATION_TOL_M)
        if lift.any():
            lift = maximum_filter1d(lift, _odd(0.2 * fps), mode="nearest")
            d = d + gaussian_filter1d(lift, sigma, mode="nearest")
        d = np.maximum(d, -z_min - PENETRATION_TOL_M)
        new[:, 2] = trans[:, 2] + d

        sole_new = sole + (new - trans)[:, None, None, :]
        z_new = sole_new[..., 2]
        slide = np.linalg.norm(np.gradient(sole_new[..., :2], 1.0 / fps, axis=0), axis=-1)[contact]
        low_contact = np.where(contact, z_new, np.inf).min(axis=(1, 2))
        quality = {
            "height_fix_max_m": float(np.abs(d - d[0]).max()),
            "contact_frac": float(stance.mean()),
            "frac_below_2cm": float((z_new.min(axis=(1, 2)) < -0.02).mean()),
            "min_sole_z_m": float(z_new.min()),
            "skate_median_mps": float(np.median(slide)) if slide.size else 0.0,
            "skate_p95_mps": float(np.percentile(slide, 95)) if slide.size else 0.0,
            "stance_float_frac": (
                float((low_contact[stance] > 0.01).mean()) if stance.any() else 0.0
            ),
        }
        return new.astype(np.float32), d, quality

    def transfer_entry(self, name: str, entry: dict) -> tuple[dict, TransferStats]:
        dof_g1 = np.asarray(entry["dof"], dtype=np.float32)
        if dof_g1.shape[1] < len(self.g1_joints):
            raise ValueError(
                f"{name}: expected >= {len(self.g1_joints)} G1 DOFs, got {dof_g1.shape}"
            )
        T = dof_g1.shape[0]
        fps = float(entry.get("fps", 30.0))
        root_rot = np.asarray(entry["root_rot"], dtype=np.float32)  # xyzw
        trans = np.asarray(entry["root_trans_offset"], dtype=np.float32).copy()

        # 1) joint selection by name
        dof = dof_g1[:, self.g1_to_r1_idx]

        # 2) joint limits
        excess = np.maximum(dof - self.upper, 0) + np.maximum(self.lower - dof, 0)
        clamped_mask = excess > 1e-6
        dof = np.clip(dof, self.lower, self.upper)
        clamp_frac = float(clamped_mask.any(1).mean())
        clamped_joints = [self.r1_joints[j] for j in np.where(clamped_mask.any(0))[0]]

        # velocity feasibility (finite differences at source fps)
        if T > 1:
            qdot = np.abs(np.diff(dof, axis=0)) * fps
            ratio = qdot / self.vel_limit[None]
            vel_frac = float((ratio > 1.0).any(1).mean())
            max_vel_ratio = float(ratio.max())
        else:
            vel_frac, max_vel_ratio = 0.0, 0.0

        # 3) root translation: scale displacement about frame 0 (xy) and height (z), then either
        #    the contact correction or (legacy) re-grounding of frame 0
        xy0 = trans[0, :2].copy()
        trans[:, :2] = xy0 + self.scale * (trans[:, :2] - xy0)
        trans[:, 2] = self.scale * trans[:, 2]
        quality = {}
        if self.contact_fix and T > 1:
            trans, height_fix, quality = self.contact_fix_root(entry, trans, root_rot, dof, fps)
            z_low = -float(height_fix[0])
        else:
            z_low = self.lowest_foot_z(trans[0], root_rot[0], dof[0])
            trans[:, 2] -= z_low

        # 4) pose_aa from R1 axes
        pose_aa = np.zeros((T, len(self.r1_joints) + 1, 3), dtype=np.float32)
        pose_aa[:, 0] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
        pose_aa[:, 1:] = self.r1_axes[None] * dof[:, :, None]

        out = {
            "root_trans_offset": trans.astype(np.float32),
            "pose_aa": pose_aa,
            "dof": dof.astype(np.float32),
            "root_rot": root_rot,
            "fps": fps,
        }
        if "smpl_joints" in entry:
            out["smpl_joints"] = np.asarray(entry["smpl_joints"], dtype=np.float32)
        for k in ("beta", "gender"):
            if k in entry:
                out[k] = entry[k]
        stats = TransferStats(
            name=name,
            num_frames=T,
            fps=fps,
            clamp_frac=clamp_frac,
            max_limit_excess_rad=float(excess.max()) if excess.size else 0.0,
            vel_frac=vel_frac,
            max_vel_ratio=max_vel_ratio,
            ground_shift_m=float(-z_low),
            clamped_joints=clamped_joints,
            **quality,
        )
        return out, stats


# --------------------------------------------------------------------------------------
# Input discovery
# --------------------------------------------------------------------------------------
def _iter_pkl_files(root: Path):
    for p in sorted(root.rglob("*.pkl")):
        yield p


def _iter_bones_csv_files(root: Path):
    for p in sorted(root.rglob("*.csv")):
        yield p


def _load_qpos_csv(csv_path: Path, fps: int) -> dict:
    """MuJoCo qpos rows (root xyz, root quat wxyz, 29 G1 joints; Kimodo-G1's CSV) -> G1 entry."""
    rows = []
    for line in csv_path.read_text().splitlines():
        try:
            rows.append([float(x) for x in line.split(",")])
        except ValueError:  # a header line
            continue
    qpos = np.asarray(rows, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected (T, 36) qpos rows, got {qpos.shape}")
    return {
        csv_path.stem: {
            "root_trans_offset": qpos[:, :3],
            "root_rot": qpos[:, [4, 5, 6, 3]],  # wxyz -> xyzw
            "dof": qpos[:, 7:],
            "fps": float(fps),
        }
    }


def _load_g1_entries_from_csv(
    csv_path: Path, fps: int, fps_source: int | None, csv_format: str = "bones"
) -> dict:
    if csv_format == "qpos":
        return _load_qpos_csv(csv_path, fps)
    sys.path.insert(0, str(REPO / "gear_sonic/data_process"))
    import convert_soma_csv_to_motion_lib as conv  # noqa: PLC0415

    seq = conv.load_bones_csv(str(csv_path))
    entry = conv.convert_sequence(seq, fps_source or fps)
    if fps_source and fps_source != fps:
        entry = conv.downsample_sequence(entry, fps_source, fps)
    return {csv_path.stem: entry}


def _process_file(args_tuple):
    src, rel_out, mode, fps, fps_source, max_clamp, max_vel, dry_run, csv_format = args_tuple
    import warnings

    warnings.filterwarnings("ignore")
    tr = _process_file.transfer  # type: ignore[attr-defined]
    if mode == "pkl":
        data = joblib.load(src)
    else:
        data = _load_g1_entries_from_csv(Path(src), fps, fps_source, csv_format)
    out_entries, stats_list = {}, []
    for name, entry in data.items():
        try:
            r1_entry, st = tr.transfer_entry(name, entry)
        except Exception as e:  # noqa: BLE001
            stats_list.append(
                asdict(TransferStats(name, 0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, True, f"error: {e}"))
            )
            continue
        if st.clamp_frac > max_clamp:
            st.dropped, st.reason = True, f"clamp_frac {st.clamp_frac:.3f} > {max_clamp}"
        elif st.vel_frac > max_vel:
            st.dropped, st.reason = True, f"vel_frac {st.vel_frac:.3f} > {max_vel}"
        stats_list.append(asdict(st))
        if not st.dropped:
            out_entries[name] = r1_entry
    if out_entries and not dry_run:
        Path(rel_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(out_entries, rel_out, compress=True)
    return stats_list


def _init_worker(contact_fix: bool = True):
    _process_file.transfer = G1ToR1Transfer(contact_fix=contact_fix)  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="G1 motion_lib PKL tree or Bones-SEED CSV tree")
    ap.add_argument("--output", required=True, help="output directory (layout mirrors input)")
    ap.add_argument("--fps", type=int, default=30, help="output fps for CSV inputs")
    ap.add_argument(
        "--fps_source", type=int, default=None, help="source fps for CSV inputs (Bones-SEED: 120)"
    )
    ap.add_argument(
        "--csv-format",
        choices=["bones", "qpos"],
        default="bones",
        help="bones: Bones-SEED CSV (deg, cm, Euler); qpos: MuJoCo qpos rows (Kimodo-G1)",
    )
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--max-clamp-frac",
        type=float,
        default=0.05,
        help="drop motions clamped in more than this fraction of frames",
    )
    ap.add_argument(
        "--max-vel-frac",
        type=float,
        default=0.05,
        help="drop motions over R1 velocity limits in more than this fraction of frames",
    )
    ap.add_argument(
        "--no-contact-fix",
        action="store_true",
        help="legacy grounding (frame 0 only) instead of the per-frame contact correction",
    )
    ap.add_argument("--dry-run", action="store_true", help="compute statistics only")
    ap.add_argument("--limit", type=int, default=None, help="process at most N files (debug)")
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    pkls = list(_iter_pkl_files(inp))
    csvs = [] if pkls else list(_iter_bones_csv_files(inp))
    mode = "pkl" if pkls else "csv"
    files = pkls or csvs
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no .pkl or .csv files under {inp}")
    tr = G1ToR1Transfer(contact_fix=not args.no_contact_fix)
    print(
        f"G1->R1 transfer: {len(files)} {mode} files; leg scale {tr.scale:.4f}; "
        f"contact fix {tr.contact_fix}; output {out}"
    )

    jobs = []
    for f in files:
        rel = f.relative_to(inp).with_suffix(".pkl")
        jobs.append(
            (
                str(f),
                str(out / rel),
                mode,
                args.fps,
                args.fps_source,
                args.max_clamp_frac,
                args.max_vel_frac,
                args.dry_run,
                args.csv_format,
            )
        )

    import multiprocessing as mp

    all_stats: list[dict] = []
    if args.num_workers <= 1:
        _init_worker(not args.no_contact_fix)
        for j in jobs:
            all_stats.extend(_process_file(j))
    else:
        with mp.Pool(
            args.num_workers, initializer=_init_worker, initargs=(not args.no_contact_fix,)
        ) as pool:
            for i, res in enumerate(pool.imap_unordered(_process_file, jobs, chunksize=4)):
                all_stats.extend(res)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(jobs)} files")

    n = len(all_stats)
    dropped = [s for s in all_stats if s["dropped"]]
    kept = [s for s in all_stats if not s["dropped"]]
    report = {
        "input": str(inp),
        "output": str(out),
        "mode": mode,
        "leg_scale": tr.scale,
        "num_motions": n,
        "num_kept": len(kept),
        "num_dropped": len(dropped),
        "drop_reasons": {},
        "kept_frames_total": int(sum(s["num_frames"] for s in kept)),
        "kept_hours_total": round(
            sum(s["num_frames"] / max(s["fps"], 1e-6) for s in kept) / 3600.0, 3
        ),
        "mean_clamp_frac_kept": float(np.mean([s["clamp_frac"] for s in kept])) if kept else None,
        "mean_vel_frac_kept": float(np.mean([s["vel_frac"] for s in kept])) if kept else None,
        "contact_fix": tr.contact_fix,
        "quality_kept": {
            k: float(np.mean([s[k] for s in kept])) if kept else None
            for k in ("frac_below_2cm", "min_sole_z_m", "skate_median_mps", "skate_p95_mps",
                      "stance_float_frac", "height_fix_max_m", "contact_frac")
        },  # fmt: skip
        "most_clamped_joints": {},
        "motions": all_stats,
    }
    for s in dropped:
        key = s["reason"].split(" ")[0]
        report["drop_reasons"][key] = report["drop_reasons"].get(key, 0) + 1
    for s in all_stats:
        for j in s["clamped_joints"]:
            report["most_clamped_joints"][j] = report["most_clamped_joints"].get(j, 0) + 1
    report["most_clamped_joints"] = dict(
        sorted(report["most_clamped_joints"].items(), key=lambda kv: -kv[1])[:10]
    )

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / ("transfer_report_dryrun.json" if args.dry_run else "transfer_report.json")
    report_path.write_text(json.dumps(report, indent=1))
    summary = {k: v for k, v in report.items() if k != "motions"}
    print(json.dumps(summary, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
