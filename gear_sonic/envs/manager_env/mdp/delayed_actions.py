"""Joint position targets that reach the actuators late: actuation-latency randomization.

On the robot, a policy output reaches the motors after sensing, inference and bus latency.
Trackers deployed on Unitree humanoids randomize it (OmniH2O 2406.08858 and HOVER 2410.21229:
20-60 ms; ASAP 2502.01143: 20-40 ms; GMT 2506.14770 and CLONE 2506.08931: 0-20 ms), and
BeyondMimic 2508.08241 saw a policy trained without it fall with 5-10 ms of injected delay.
SONIC's released environment has none.

Isaac Lab applies actions at every physics substep (``decimation`` per control step). This term
holds the previous control step's target for the first ``d`` substeps of each control step and
applies the new one afterwards, with ``d`` drawn per environment at every reset from
``delay_substeps_range`` (substeps of ``sim.dt``: 0-3 at 5 ms = 0-15 ms). The policy's action
observations are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
import torch


class DelayedJointPositionAction(JointPositionAction):
    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._previous = self._processed_actions.clone()
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep = 0
        self._sample(torch.arange(self.num_envs, device=self.device))

    def _sample(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.delay_substeps_range
        self._delay[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        previous = self._processed_actions.clone()
        super().process_actions(actions)
        # A reset env has no previous target of its own episode: apply its first one undelayed.
        self._previous = torch.where(self._fresh[:, None], self._processed_actions, previous)
        self._fresh[:] = False
        self._substep = 0

    def apply_actions(self):
        late = (self._substep < self._delay)[:, None]
        target = torch.where(late, self._previous, self._processed_actions)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._substep += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._fresh[ids] = True
        self._sample(ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """:class:`JointPositionActionCfg` with per-env actuation latency (see module doc)."""

    class_type: type = DelayedJointPositionAction

    delay_substeps_range: tuple[int, int] = (0, 3)
    """Inclusive range of the per-env delay, in physics substeps (``sim.dt`` each)."""
