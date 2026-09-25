"""Training videos on W&B, in the format of BruteForce's runs (``scripts/run_a.sh``: ``--video``).

Every ``every_n_iterations`` iterations, record ``num_frames`` consecutive control steps of env 0
(robot state, reference motion, palm targets, SMPL joints), render them with MuJoCo in a
subprocess on another GPU (``scripts/r1/render_rollout_video.py``: reference ghost | policy |
human, palm targets) and log the mp4 under the W&B key ``video``. The npz and mp4 stay in
``video_dir``. Recording wraps ``env.step`` on the main process only; it reads a few tensors of
env 0 per step while armed and nothing otherwise, so training is not slowed down.

Training rollouts sample actions from the policy's Gaussian (std up to 0.5 action units), so
they look far noisier than the deterministic policy. With ``arm_on_eval=True`` the recorder
instead arms on the iterations where ``PeriodicEvalCallback`` evaluates (list it before that
callback): the video then shows the deterministic policy that gets deployed.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from loguru import logger
import numpy as np
import torch
from transformers import TrainerCallback
import wandb

from gear_sonic.trl.utils.common import wandb_run_exists

RENDER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "r1" / "render_rollout_video.py"


class RolloutVideoCallback(TrainerCallback):
    """Record env 0 periodically and log a rendered video to W&B."""

    def __init__(
        self,
        video_dir: str,
        every_n_iterations: int = 250,
        first_iteration: int = 1,
        num_frames: int = 500,
        egl_device: int = 0,
        wandb_key: str = "video",
        render_args: list | None = None,
        render_timeout_s: float = 900.0,
        arm_on_eval: bool = False,
        eval_frequency: int = 1000,
    ):
        super().__init__()
        self.video_dir = Path(video_dir).resolve()
        self.every = int(every_n_iterations)
        self.first = -1 if first_iteration is None else int(first_iteration)
        self.num_frames = int(num_frames)
        self.egl_device = str(egl_device)
        self.wandb_key = wandb_key
        self.render_args = [str(a) for a in (render_args or [])]
        self.render_timeout_s = render_timeout_s
        self.arm_on_eval, self.eval_frequency = arm_on_eval, int(eval_frequency)
        self._command = None
        self._recording = None  # dict of lists while armed
        self._rec_iteration = 0
        self._jobs = []  # (Popen, mp4 path, recorded iteration, start time, log file)

    # ------------------------------------------------------------------ setup
    def on_train_begin(self, args, state, control, env=None, **kwargs):
        off = self.every <= 0 or (self.arm_on_eval and self.eval_frequency <= 0)
        if env is None or not state.is_world_process_zero or off:
            return
        base = env.env.unwrapped
        self._command = base.command_manager.get_term("motion")
        self._origins = base.scene.env_origins
        self._dt = float(base.step_dt)
        lib = self._command.motion_lib
        self._mjcf = str(
            (Path(lib.m_cfg.asset.assetRoot) / lib.m_cfg.asset.assetFileName).resolve()
        )
        probs = getattr(self._command, "encoder_sample_probs_dict", None)
        self._encoder_names = list(probs.keys()) if probs else []
        self._has_smpl = getattr(lib, "_motion_smpl_joints", None) is not None
        self._has_encoders = getattr(self._command, "encoder_index", None) is not None
        self.video_dir.mkdir(parents=True, exist_ok=True)

        step = env.step

        def recording_step(*a, **kw):
            out = step(*a, **kw)
            if self._recording is not None and len(self._recording["robot_q"]) < self.num_frames:
                self._capture()
            return out

        env.step = recording_step
        logger.info(
            f"RolloutVideoCallback: {self.num_frames} steps of env 0 every {self.every} iterations "
            f"-> {self.video_dir} and W&B '{self.wandb_key}'"
        )

    # ------------------------------------------------------------------ capture
    @torch.no_grad()
    def _capture(self):
        c, o = self._command, self._origins[0]
        robot = c.robot.data
        rec = self._recording
        rec["robot_root"].append(torch.cat([robot.root_pos_w[0] - o, robot.root_quat_w[0]]))
        rec["robot_q"].append(robot.joint_pos[0].clone())
        rec["ref_root"].append(torch.cat([c.body_pos_w[0, 0] - o, c.body_quat_w[0, 0]]))
        rec["ref_q"].append(c.joint_pos[0].clone())
        rec["target_points"].append(c.vr_3point_body_pos_w[0] - o)
        rec["robot_points"].append(c.robot_vr_3point_pos_w[0] - o)
        if self._has_smpl:
            rec["smpl_joints"].append(c.smpl_joints[0].clone())
        if self._has_encoders:
            rec["encoder"].append(c.encoder_index[0].clone())
        rec["motion_id"].append(c.motion_ids[0].clone())
        rec["reset"].append(c.time_steps[0] <= 1)

    def _save(self) -> Path:
        rec = {k: torch.stack(v).cpu().numpy() for k, v in self._recording.items()}
        lib = self._command.motion_lib
        meta = {
            "mjcf": self._mjcf,
            "joint_names": list(self._command.robot.joint_names),
            "dt": self._dt,
            "iteration": self._rec_iteration,
            "encoder_names": self._encoder_names,
            "motion_names": [str(k) for k in getattr(lib, "curr_motion_keys", [])],
        }
        path = self.video_dir / f"rollout_it{self._rec_iteration:06d}.npz"
        np.savez_compressed(path, meta=json.dumps(meta), **rec)
        return path

    def _launch(self, npz: Path, state):
        mp4 = npz.with_suffix(".mp4")
        env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG") if k in os.environ}
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=self.egl_device)
        log = open(npz.with_suffix(".log"), "w")  # noqa: SIM115
        cmd = [
            sys.executable,
            str(RENDER_SCRIPT),
            "--npz",
            str(npz),
            "--out",
            str(mp4),
            *self.render_args,
        ]
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(npz.parent)
        )
        self._jobs.append((proc, mp4, self._rec_iteration, time.monotonic(), log))

    # ------------------------------------------------------------------ per iteration
    def _collect(self, step: int, wait: bool = False):
        """Log finished renders at ``step``; with ``wait``, block until every pending render ends."""
        for job in list(self._jobs):
            proc, mp4, rec_it, started, log = job
            if wait:
                try:
                    proc.wait(
                        timeout=max(1.0, self.render_timeout_s - (time.monotonic() - started))
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
            elif proc.poll() is None:
                if time.monotonic() - started < self.render_timeout_s:
                    continue
                proc.kill()
            proc.wait()
            log.close()
            self._jobs.remove(job)
            if proc.returncode == 0 and mp4.exists():
                if wandb_run_exists():
                    video = wandb.Video(str(mp4), format="mp4", caption=f"iteration {rec_it}")
                    wandb.log({self.wandb_key: video}, step=step)
                logger.info(f"RolloutVideoCallback: rendered {mp4.name}")
            else:
                logger.warning(
                    f"RolloutVideoCallback: render failed, see {mp4.with_suffix('.log')}"
                )

    def on_step_end(self, args, state, control, **kwargs):
        if self._command is None:
            return
        it = state.global_step
        # The trainer returns without on_train_end once max_steps is reached: drain here.
        last = control.should_training_stop or (state.max_steps and it >= state.max_steps)
        self._collect(it, wait=bool(last))
        if last:
            return
        if self._recording is None:
            if self.arm_on_eval:  # the evaluation that follows in this on_step_end round
                due = it == self.first or (it > 0 and it % self.eval_frequency == 0)
            else:
                due = it >= self.first and (it - self.first) % self.every == 0
            if due:
                self._recording = defaultdict(list)
                self._rec_iteration = it
        elif len(self._recording["robot_q"]) >= self.num_frames:
            npz = self._save()
            self._recording = None
            self._launch(npz, state)

    def on_train_end(self, args, state, control, **kwargs):
        if self._command is not None:
            self._collect(state.global_step, wait=True)
