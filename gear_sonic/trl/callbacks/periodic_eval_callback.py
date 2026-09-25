"""Uniform-coverage evaluation during training, logged to W&B under ``eval/``.

Adaptive sampling concentrates training episodes on the clip segments the policy fails, so
the training curves (time-out rate, tracking rewards) drift down while the policy improves.
Every ``eval_frequency`` iterations this callback runs SONIC's ``ImEvalCallback.evaluate_policy``
in the training process: every loaded clip is played once from its start with the
deterministic policy and the environment's own terminations (a clip "succeeds" if it reaches
its end), then training resumes. Success rate and MPJPE-style errors (all bodies, legs, VR
3-point) go to W&B at the current iteration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import wandb

from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback
from gear_sonic.trl.utils.common import wandb_run_exists


def _register_metrics_only_smpl_sim() -> None:
    """Make ``smpl_sim`` importable without running its package ``__init__``.

    ImEvalCallback lazily imports ``smpl_sim.smpllib.smpl_eval.compute_metrics_lite`` (numpy /
    torch only). The package ``__init__`` imports its environments -> ``mujoco.viewer`` -> ``glfw``,
    whose cffi FFI fails inside the Isaac Sim process: Kit's pip_prebundle puts cffi 1.17.1 on
    sys.path while the env's compiled ``_cffi_backend`` is 2.1.1 ("Version mismatch"). A bare
    package entry pointing at the real directory skips only that ``__init__``.
    """
    if "smpl_sim" in sys.modules:
        return
    spec = importlib.util.find_spec("smpl_sim")
    if spec is None or not spec.submodule_search_locations:
        return
    package = types.ModuleType("smpl_sim")
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules["smpl_sim"] = package


def _scalar(value):
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    if hasattr(value, "numel") and value.numel() == 1:  # torch scalar
        return float(value.item())
    return None


class PeriodicEvalCallback(ImEvalCallback):
    """ImEvalCallback that only runs on evaluation iterations and logs its scalars to W&B."""

    def __init__(self, eval_frequency: int = 500, first_iteration: int | None = None, **kwargs):
        super().__init__(eval_frequency=eval_frequency, eval_only=False, **kwargs)
        self.first_iteration = first_iteration
        _register_metrics_only_smpl_sim()

    # clip-name prefix -> group; everything else is "mocap"
    GROUPS = {"planner_": "planner", "kimodo_": "kimodo"}

    @classmethod
    def group_of(cls, key: str) -> str:
        return next((g for prefix, g in cls.GROUPS.items() if key.startswith(prefix)), "mocap")

    def _post_evaluate_policy(self, eval_res):
        """Add success rates per clip group: planner-generated, Kimodo-generated, mocap."""
        metrics = super()._post_evaluate_policy(eval_res)
        outcome = [(k, 0.0) for k in eval_res.get("failed_keys", [])]
        outcome += [(k, 1.0) for k in eval_res.get("success_keys", [])]
        self._outcome = {str(k): bool(ok) for k, ok in outcome}
        for group in ("planner", "kimodo", "mocap"):
            values = [ok for k, ok in outcome if self.group_of(str(k)) == group]
            if values:
                metrics[f"eval/success/success_rate_{group}"] = float(np.mean(values))
                metrics[f"eval/success/num_clips_{group}"] = len(values)
        return metrics

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        due = step > 0 and step % self.eval_frequency == 0
        if not (due or step == self.first_iteration) or control.should_training_stop:
            return
        self.env = kwargs.get("env")
        self.model = kwargs.get("model")
        self.accelerator = kwargs.get("accelerator")
        self.device = self.accelerator.device
        self.args = args
        metrics = self.evaluate_policy()
        scalars = {k: v for k, v in ((k, _scalar(v)) for k, v in metrics.items()) if v is not None}
        if self.accelerator.is_main_process and wandb_run_exists():
            wandb.log(scalars, step=step)
        rates = {k.split("/")[-1]: round(v, 4) for k, v in scalars.items() if "success_rate" in k}
        print(f"[PeriodicEval] iteration {step}: {rates}")
        # Per-clip outcomes, e.g. to compare with MuJoCo sim-to-sim (scripts/r1/teleop/play_clips.py)
        out_dir = getattr(args, "output_dir", None)
        if self.accelerator.is_main_process and out_dir and getattr(self, "_outcome", None):
            os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)
            with open(os.path.join(out_dir, "eval", f"iteration_{step:06d}.json"), "w") as f:
                json.dump({"iteration": step, **rates, "clips": self._outcome}, f, indent=1)
