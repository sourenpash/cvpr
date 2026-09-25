#!/usr/bin/env python3
"""SONIC's kinematic-planner loop in Python, ported from gear_sonic_deploy (C++).

The planner (``planner_sonic.onnx``, V2) turns locomotion commands into a whole-body Unitree G1
reference. In SONIC's ``VR_3PT`` teleop mode its lower body feeds the teleop encoder while the VR
3 points (head + hands) drive the upper body. This module reproduces the deployed loop so that
(1) R1 training references match what the deployed planner produces and (2) the R1 demo runtime
can drive the planner:

* control ticks at 50 Hz advance a playback cursor through a 50 Hz motion buffer;
* a planner tick every 5 control ticks (10 Hz) replans when the mode, facing or height changed,
  or, in moving modes, when the speed or direction changed or the replan interval (1 s; 0.1 s
  running) elapsed while moving;
* the context is 4 frames sampled at 30 Hz from the playing motion, 2 control frames ahead;
* the 30 Hz output is resampled to 50 Hz (linear, slerp for the root) and cross-faded into the
  playing motion over 8 frames; the cursor is rebased to 0.

Sources: ``include/localmotion_kplanner.hpp`` (Initialize, UpdatePlanning,
UpdateContextFromMotion, ResampleGeneratedSequence50Hz), ``src/g1_deploy_onnx_ref.cpp``
(CurrentFrameAdvancement: blending), ``docs/source/references/planner_onnx.md`` (replan rules),
``include/input_interface/gamepad.hpp`` (stick semantics, speed bands). Joint order everywhere
is the planner's qpos order: the G1 29-DOF MuJoCo order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IDLE, SLOW_WALK, WALK, RUN = 0, 1, 2, 3
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 9})  # idle, squat, kneel x2, lying, idle boxing
DEFAULT_HEIGHT = 0.788740  # localmotion_kplanner.hpp PlannerConfig::default_height
LOOK_AHEAD = 2  # PlannerConfig::motion_look_ahead_steps (50 Hz frames)
BLEND_FRAMES = 8  # g1_deploy_onnx_ref.cpp CurrentFrameAdvancement
PLANNER_EVERY = 5  # planner thread 10 Hz vs control 50 Hz
SPEED_BANDS = {SLOW_WALK: (0.2, 0.8), WALK: (0.8, 1.5), RUN: (1.5, 3.0)}  # gamepad.hpp

#: G1 29-DOF MuJoCo joint order of the planner's qpos (gear_sonic mdp/actions.py G1_MUJOCO_ORDER).
G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)  # fmt: skip
#: G1 default standing pose (g1.py G1_CYLINDER_MODEL_12_DEX_CFG.init_state; the R1 uses it too, D10).
G1_DEFAULT_POSE = {
    "hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
    "left_shoulder_roll": 0.2, "right_shoulder_roll": -0.2, "shoulder_pitch": 0.2,
}  # fmt: skip


def default_joint_positions() -> np.ndarray:
    q = np.zeros(len(G1_JOINTS))
    for i, name in enumerate(G1_JOINTS):
        for key, val in G1_DEFAULT_POSE.items():
            if name.endswith(key + "_joint"):
                q[i] = val
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Shortest-path slerp of wxyz quaternions; broadcasts over leading dims."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    t = np.asarray(t, float)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def interpolate_qpos(motion: np.ndarray, f: np.ndarray) -> np.ndarray:
    """qpos at fractional frame indices ``f`` (clamped): linear for positions/joints, slerp root."""
    f = np.clip(np.asarray(f, float), 0.0, len(motion) - 1)
    f0 = np.floor(f).astype(int)
    f1 = np.minimum(f0 + 1, len(motion) - 1)
    w = (f - f0)[..., None]
    out = (1.0 - w) * motion[f0] + w * motion[f1]
    out[..., 3:7] = slerp(motion[f0, 3:7], motion[f1, 3:7], w[..., 0])
    return out


def resample_30_to_50(qpos30: np.ndarray) -> np.ndarray:
    n50 = int(np.floor(len(qpos30) / 30.0 * 50.0))
    return interpolate_qpos(qpos30, np.arange(n50) / 50.0 * 30.0)


@dataclass
class Command:
    """One locomotion command, as SONIC's input interfaces send it (world-frame directions)."""

    mode: int = IDLE
    speed: float = -1.0  # m/s; <= 0 uses the mode's default speed
    move_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_dir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    height: float = -1.0

    @property
    def moving(self) -> bool:
        return self.mode not in STATIC_MODES and float(np.linalg.norm(self.move_dir[:2])) > 1e-5


class PlannerModel:
    """``planner_sonic.onnx`` (V2, 11 inputs) on onnxruntime."""

    def __init__(self, onnx_path: str | Path, seed: int = 0, threads: int = 0, device: str = "cpu"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads > 0:  # 0 = onnxruntime default (all cores); ~70 ms/call on 4 cores
            options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):  # onnxruntime-gpu (r1rt env): ~18 ms on the TITAN V
            if hasattr(ort, "preload_dlls"):  # CUDA / cuDNN from the nvidia-* pip wheels
                ort.preload_dlls()
            index = int(device.split(":")[1]) if ":" in device else 0
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": index}))
        self.session = ort.InferenceSession(str(onnx_path), options, providers=providers)
        names = [i.name for i in self.session.get_inputs()]
        assert "allowed_pred_num_tokens" in names, f"expected a V1/V2 planner, got inputs {names}"
        self.k = self.session.get_inputs()[names.index("allowed_pred_num_tokens")].shape[1]
        self.rng = np.random.default_rng(seed)

    def __call__(self, context: np.ndarray, cmd: Command, seed: int | None = None) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**31 - 1)) if seed is None else seed
        feeds = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.array([cmd.speed], np.float32),
            "mode": np.array([cmd.mode], np.int64),
            "movement_direction": np.asarray(cmd.move_dir, np.float32)[None],
            "facing_direction": np.asarray(cmd.face_dir, np.float32)[None],
            "random_seed": np.array([seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.ones((1, self.k), np.int64),
            "height": np.array([cmd.height], np.float32),
        }
        qpos, n = self.session.run(None, feeds)
        qpos = qpos[0, : int(np.asarray(n).reshape(-1)[0])].astype(np.float64)
        if not np.isfinite(qpos).all():
            raise FloatingPointError("planner returned NaN/inf")
        return qpos


class PlannerLoop:
    """The deployed planner loop: call :meth:`tick` at 50 Hz, get the reference qpos (36,) back."""

    def __init__(self, model: PlannerModel, joints: np.ndarray | None = None):
        self.model = model
        joints = default_joint_positions() if joints is None else np.asarray(joints, float)
        context = np.zeros((4, 36))
        context[:, 2] = DEFAULT_HEIGHT
        context[:, 3] = 1.0
        context[:, 7:] = joints
        self.last = Command()
        self.context_frame = context[-1].copy()
        self.motion = resample_30_to_50(self.model(context, self.last))
        self.cur = 0
        self.ticks = 0
        self.since_replan = 0.0
        self.num_replans = 1

    def hold(self) -> None:
        """Stand still in the initial context frame until a command replans.

        The first IDLE plan from the default pose settles into the planner's own stance: the
        feet move up to ~10 cm over 2 s and one lifts ~3 cm. A robot that stood up into the
        default pose would shuffle when control engages. Static modes do not replan, so the
        hold lasts until the sticks move, and that plan starts from the held frame.
        """
        self.motion, self.cur = self.context_frame[None].copy(), 0

    def _needs_replan(self, cmd: Command) -> bool:
        last = self.last
        if (
            cmd.mode != last.mode
            or cmd.height != last.height
            or not np.allclose(cmd.face_dir, last.face_dir, atol=1e-6)
        ):
            return True
        if cmd.mode in STATIC_MODES:
            return False
        if cmd.speed != last.speed or not np.allclose(cmd.move_dir, last.move_dir, atol=1e-6):
            return True
        interval = 0.1 if cmd.mode == RUN else 1.0
        return cmd.moving and self.since_replan >= interval - 1e-9

    def _context(self) -> tuple[int, np.ndarray]:
        """Generation frame (LOOK_AHEAD ahead of the cursor) and the 4-frame 30 Hz context there."""
        gen = self.cur + LOOK_AHEAD
        return gen, interpolate_qpos(self.motion, (gen / 50.0 + np.arange(4) / 30.0) * 50.0)

    def _merge(self, new: np.ndarray, gen: int) -> None:
        """Cross-fade the 50 Hz plan ``new`` (starting at buffer frame ``gen``) in from the cursor.

        If the cursor has passed ``gen`` (an asynchronous plan arrived late), the plan's elapsed
        frames are skipped and the cross-fade starts at the cursor.
        """
        length = gen - self.cur + len(new)
        if length > 0:
            f = np.arange(length)
            f_old = np.clip(f + self.cur, 0, len(self.motion) - 1)
            f_new = np.clip(f + self.cur - gen, 0, len(new) - 1)
            w = np.clip((f - max(0, gen - self.cur)) / BLEND_FRAMES, 0.0, 1.0)
            out = (1.0 - w[:, None]) * self.motion[f_old] + w[:, None] * new[f_new]
            out[:, 3:7] = slerp(self.motion[f_old, 3:7], new[f_new, 3:7], w)
            self.motion, self.cur = out, 0

    def _replan(self, cmd: Command) -> None:
        gen, context = self._context()
        self._merge(resample_30_to_50(self.model(context, cmd)), gen)
        self.last = cmd
        self.since_replan = 0.0
        self.num_replans += 1

    def tick(self, cmd: Command) -> np.ndarray:
        if self.ticks % PLANNER_EVERY == 0:
            if self._needs_replan(cmd):
                self._replan(cmd)
            else:
                self.since_replan += PLANNER_EVERY / 50.0
        frame = self.motion[min(self.cur, len(self.motion) - 1)].copy()
        self.cur += 1
        self.ticks += 1
        return frame


class AsyncPlannerLoop(PlannerLoop):
    """The planner loop with the model call on a worker thread, as the deployed C++ loop runs it.

    :meth:`tick` never blocks on the planner: a replan request takes its context at the cursor
    plus LOOK_AHEAD and is merged on the first tick after the result arrives (late frames are
    skipped, :meth:`PlannerLoop._merge`). Training references (``generate_planner_motions.py``)
    were generated with an instantaneous planner, so keep the latency under LOOK_AHEAD ticks
    (40 ms; a GPU planner takes a few ms). ``last_latency_ticks`` reports it; ``seeds`` logs the
    planner's random seeds for replay.
    """

    def __init__(self, model: PlannerModel, joints: np.ndarray | None = None):
        from concurrent.futures import ThreadPoolExecutor

        super().__init__(model, joints)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
        self._pending = None  # (future, gen, cmd, tick requested)
        self.last_latency_ticks = 0
        self.seeds: list[int] = []

    def _plan(self, context: np.ndarray, cmd: Command, seed: int) -> np.ndarray:
        return resample_30_to_50(self.model(context, cmd, seed=seed))

    def tick(self, cmd: Command) -> np.ndarray:
        if self._pending is not None and self._pending[0].done():
            future, gen, _, requested = self._pending
            self._pending = None
            self._merge(future.result(), gen)  # raises the planner's exception, if any
            self.last_latency_ticks = self.ticks - requested
        if self.ticks % PLANNER_EVERY == 0:
            if self._pending is None and self._needs_replan(cmd):
                gen, context = self._context()
                seed = int(self.model.rng.integers(0, 2**31 - 1))
                self.seeds.append(seed)
                future = self._pool.submit(self._plan, context, cmd, seed)
                self._pending = (future, gen, cmd, self.ticks)
                self.last = cmd
                self.since_replan = 0.0
                self.num_replans += 1
            else:
                self.since_replan += PLANNER_EVERY / 50.0
        frame = self.motion[min(self.cur, len(self.motion) - 1)].copy()
        self.cur += 1
        self.ticks += 1
        return frame

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def find_planner_onnx() -> Path:
    """The planner downloaded by ``download_from_hf.py`` (Hugging Face cache) or the deploy tree."""
    candidates = sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/*/planner_sonic.onnx"
        )
    ) + [
        Path(__file__).resolve().parents[2]
        / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("planner_sonic.onnx not found; run `python download_from_hf.py`")
