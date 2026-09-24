#!/usr/bin/env python3
"""Render a rollout recorded by ``RolloutVideoCallback`` to an mp4 with MuJoCo (no Isaac Sim).

Layout copies BruteForce's training videos (``bruteforce/tracking/mdp/lane_viz.py``): the
reference motion as a translucent green ghost one lane to one side of the policy's robot, the
SMPL human the clip was retargeted from as an orange stick figure one lane to the other side.
On the policy's robot: the VR 3-point targets (red: palms and head) re-anchored to the robot's
root the way SONIC's tracking terms compare them (``body_pos_relative_w``) and the robot's own
points (blue).

Runs standalone (numpy, mujoco, imageio; opencv optional for the caption) so the training
process can launch it with a clean environment on a GPU other than the training one:

    MUJOCO_GL=egl python scripts/r1/render_rollout_video.py --npz rollout.npz --out rollout.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

GHOST_RGBA = (0.2, 0.8, 0.3, 0.35)
HUMAN_RGBA = (0.95, 0.55, 0.15, 1.0)
TARGET_RGBA = (0.95, 0.15, 0.15, 0.55)
PALM_RGBA = (0.15, 0.35, 0.95, 1.0)
# SMPL kinematic tree (24 joints).
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
SMPL_FOOT_JOINTS = (7, 8, 10, 11)
SMPL_HIPS = (1, 2)  # left, right


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q_wxyz, dtype=np.float64))
    return m.reshape(3, 3)


def yaw_of(q_wxyz: np.ndarray) -> float:
    r = quat_to_mat(q_wxyz)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_model(mjcf: str) -> mujoco.MjModel:
    """The robot MJCF plus a checker floor, skybox and a light (the training scene is flat)."""
    spec = mujoco.MjSpec.from_file(mjcf)
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.12, 0.22, 0.34],
        rgb2=[0.20, 0.32, 0.46],
        width=512,
        height=512,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.4, 0.5, 0.6],
    )
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.45, 0.6],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=512,
    )
    mat = spec.add_material(name="grid", texrepeat=[1, 1], reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(
        name="rollout_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="grid",
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        pos=[0, 0, 4],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
    )
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    return spec.compile()


def qpos_indexer(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(qpos addresses, source columns) for the recorded joint order -> this model's hinges."""
    adr, cols = [], []
    for col, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            adr.append(model.jnt_qposadr[jid])
            cols.append(col)
    missing = sorted(set(joint_names) - {joint_names[c] for c in cols})
    if missing:
        print(f"[rollout_video] joints not in the MJCF (left at 0): {missing}")
    return np.asarray(adr, dtype=int), np.asarray(cols, dtype=int)


def add_geom(scene: mujoco.MjvScene, gtype, size, pos, mat, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        gtype,
        np.asarray(size, float),
        np.asarray(pos, float),
        np.asarray(mat, float).reshape(-1),
        np.asarray(rgba, np.float32),
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba) -> None:
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3), rgba)


def add_capsule(scene, p0, p1, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    add_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, [radius, 0, 0], np.zeros(3), np.eye(3), rgba)
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p0, float),
        np.asarray(p1, float),
    )


def human_frame(
    joints: np.ndarray, ref_root_pos: np.ndarray, ref_root_quat: np.ndarray, lane: np.ndarray
) -> np.ndarray:
    """SMPL joints (root-relative, z-up) at true size, hips over the reference pelvis + lane, heading aligned."""
    j = joints - joints[0]
    hips = j[SMPL_HIPS[0]] - j[SMPL_HIPS[1]]  # human's left direction
    human_yaw = np.arctan2(hips[1], hips[0]) - np.pi / 2  # left axis -> forward heading
    j = j @ rot_z(yaw_of(ref_root_quat) - human_yaw).T
    j[:, :2] += ref_root_pos[:2] + lane[:2]
    j[:, 2] += 0.07 - j[list(SMPL_FOOT_JOINTS), 2].min()  # ankle/foot joints ~7 cm above the floor
    return j


def render(
    npz_path: str,
    out_path: str,
    mjcf: str | None,
    width: int,
    height: int,
    lane_m: float,
    azimuth: float | None,
    distance: float,
    elevation: float,
) -> dict:
    rec = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(rec["meta"]))
    mjcf = mjcf or meta["mjcf"]
    model = build_model(mjcf)
    pol, ref = mujoco.MjData(model), mujoco.MjData(model)
    adr, cols = qpos_indexer(model, list(meta["joint_names"]))
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    root = model.jnt_qposadr[free[0]] if free else 0

    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=4000)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0  # no sites (IMU, feet markers)
    pert = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if azimuth is None:  # side view of the robot's initial heading: it walks left -> right
        azimuth = np.degrees(yaw_of(rec["robot_root"][0, 3:7])) + 90.0
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    az = np.radians(azimuth)
    lane = lane_m * np.array(
        [-np.sin(az), np.cos(az), 0.0]
    )  # perpendicular to the view (lane_viz.py)

    robot_root, robot_q = rec["robot_root"], rec["robot_q"]
    ref_root, ref_q = rec["ref_root"], rec["ref_q"]
    targets, palms = rec["target_points"], rec["robot_points"]
    smpl = rec["smpl_joints"] if "smpl_joints" in rec.files else None
    encoders = meta.get("encoder_names", [])
    has_human = smpl is not None and bool(np.abs(smpl).sum() > 0)
    legend = "ghost: reference   " + ("orange: human   " if has_human else "")
    legend += "red: VR 3-point target   blue: robot point"
    T = robot_q.shape[0]
    frames = []
    look = robot_root[0, :3].copy()
    try:
        import cv2
    except ImportError:  # captions are optional
        cv2 = None
    for t in range(T):
        for data, rt, q in ((pol, robot_root[t], robot_q[t]), (ref, ref_root[t], ref_q[t])):
            data.qpos[:] = 0.0
            data.qpos[root : root + 7] = rt
            data.qpos[adr] = q[cols]
        ref.qpos[root : root + 3] += lane
        mujoco.mj_kinematics(model, pol)
        mujoco.mj_kinematics(model, ref)
        look = 0.9 * look + 0.1 * robot_root[t, :3] if t else look
        cam.lookat[:] = [look[0], look[1], 0.75]
        renderer.update_scene(pol, camera=cam, scene_option=opt)
        scene = renderer.scene
        n0 = scene.ngeom
        mujoco.mjv_addGeoms(model, ref, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for i in range(n0, scene.ngeom):
            scene.geoms[i].rgba[:] = GHOST_RGBA
        # Palm targets re-anchored to the robot (SONIC body_pos_relative_w) and the robot's palm points.
        d_yaw = yaw_of(robot_root[t, 3:7]) - yaw_of(ref_root[t, 3:7])
        anchor = np.array([robot_root[t, 0], robot_root[t, 1], ref_root[t, 2]])
        for k in range(targets.shape[1]):  # left palm, right palm, head (the VR 3 points)
            add_sphere(
                scene, anchor + rot_z(d_yaw) @ (targets[t, k] - ref_root[t, :3]), 0.04, TARGET_RGBA
            )
            add_sphere(scene, palms[t, k], 0.022, PALM_RGBA)
        if smpl is not None and np.abs(smpl[t]).sum() > 0:
            hj = human_frame(smpl[t], ref_root[t, :3], ref_root[t, 3:7], -lane)
            for j, p in enumerate(SMPL_PARENTS):
                add_sphere(scene, hj[j], 0.03, HUMAN_RGBA)
                if p >= 0:
                    add_capsule(scene, hj[p], hj[j], 0.015, HUMAN_RGBA)
        img = renderer.render().copy()
        if cv2 is not None:
            enc = rec["encoder"][t] if "encoder" in rec.files else None
            active = "+".join(n for n, on in zip(encoders, enc) if on) if enc is not None else "?"
            names = meta.get("motion_names", [])
            mid = int(rec["motion_id"][t]) if "motion_id" in rec.files else -1
            clip = names[mid] if 0 <= mid < len(names) else str(mid)
            lines = [
                f"iter {meta.get('iteration', '?')}  t={t * meta['dt']:.2f}s  encoder: {active}",
                clip[:60],
            ]
            for li, text in enumerate(lines):
                cv2.putText(
                    img,
                    text,
                    (8, 20 + 20 * li),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                img,
                legend,
                (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            if "reset" in rec.files and rec["reset"][t]:
                cv2.putText(
                    img,
                    "RESET",
                    (width - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 60, 255),
                    2,
                    cv2.LINE_AA,
                )
        frames.append(img)
    import imageio.v2 as imageio

    fps = int(round(1.0 / meta["dt"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=16)
    return {"frames": len(frames), "fps": fps, "out": out_path}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mjcf", default=None, help="robot MJCF (default: the path recorded in the npz)"
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--lane", type=float, default=1.0, help="lane spacing in m (BruteForce: 1.0)")
    ap.add_argument(
        "--azimuth", type=float, default=None, help="deg; default: side view of the robot"
    )
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=4.2)
    args = ap.parse_args()
    npz, out = Path(args.npz), Path(args.out)
    if not npz.exists() and Path(npz.name).exists():  # relative to another cwd: use the file here
        npz, out = Path(npz.name), Path(out.name)
    info = render(
        str(npz),
        str(out),
        args.mjcf,
        args.width,
        args.height,
        args.lane,
        args.azimuth,
        args.distance,
        args.elevation,
    )
    print(json.dumps(info), flush=True)
    os._exit(0)  # skip EGL context teardown noise at interpreter exit


if __name__ == "__main__":
    main()
