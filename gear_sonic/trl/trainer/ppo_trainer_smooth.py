"""PPO with a temporal smoothness loss on the action mean (L2C2).

L2C2 (Kobayashi, "L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth
Reinforcement Learning", IROS 2022) asks the policy mean to change little between consecutive
observations: with x~ = x_t + u (x_{t+1} - x_t), u ~ U(0, 1),

    L = coef * || mu(x~) - mu(x_t) ||^2      (pairs t -> t+1 inside one episode)

Humanoid trackers report it removes high-frequency action oscillation without the lag of an
action filter (AGILE 2603.20147: a G1 oscillated audibly without it; HoST 2502.08378). The
micro-batches of this trainer hold whole rollouts of the selected environments ([env, time, ...]),
so the consecutive observations are already there; ``l2c2_fraction`` of the environments per
micro-batch get the extra actor forward pass (cost: that fraction of one actor forward/backward).

Config (``algo.config``): ``l2c2_policy_coef`` (0 = off), ``l2c2_fraction`` (default 0.5),
``l2c2_keys`` (observation keys to interpolate; the others stay at x_t; default: all). Logged
as ``loss/l2c2_avg``.

With SONIC's token model, interpolate only the proprioception (``["actor_obs"]``). Stage B2
interpolated the tokenizer observations too, which asks the FSQ encoder to change its token
less as the reference moves. Its uniform success fell from 0.845 to 0.706 in 1000 iterations,
walking clips from 0.75 to 0.47 (PLAN.md Q6). The jitter is in the feedback from the robot's
state, which the proprioception carries.
"""

from __future__ import annotations

import torch

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer


class TRLSmoothPPOTrainer(TRLAuxLossPPOTrainer):
    _tag_names = ["trl", "aux_loss_ppo", "l2c2"]

    def _init_config(self):
        super()._init_config()
        self.l2c2_coef = float(self.config.get("l2c2_policy_coef", 0.0))
        self.l2c2_fraction = float(self.config.get("l2c2_fraction", 0.5))
        keys = self.config.get("l2c2_keys", None)
        self.l2c2_keys = None if keys is None else set(keys)

    def _register_stats_buffer(self):
        super()._register_stats_buffer()
        args = self.args
        shape = (args.num_ppo_epochs, args.num_mini_batches, args.num_micro_batches)
        self.l2c2_stats = torch.zeros(shape, device=self.accelerator.device)

    def _forward_model(self, model, mb_rollout_data):
        results = super()._forward_model(model, mb_rollout_data)
        if self.l2c2_coef <= 0:
            return results
        obs = mb_rollout_data["mb_obs_dict"]
        dones = mb_rollout_data["mb_dones"]
        num_envs, num_steps = dones.shape[:2]
        if num_steps < 2:
            return results
        n = max(1, int(round(num_envs * self.l2c2_fraction)))
        idx = torch.randperm(num_envs, device=dones.device)[:n]
        u = torch.rand(n, num_steps - 1, device=dones.device)
        interp = {}
        for key, value in obs.items():
            x0, x1 = value[idx, :-1], value[idx, 1:]
            if value.is_floating_point() and (self.l2c2_keys is None or key in self.l2c2_keys):
                w = u.view(n, num_steps - 1, *([1] * (value.dim() - 2)))
                interp[key] = x0 + w * (x1 - x0)
            else:
                interp[key] = x0
        policy = self.accelerator.unwrap_model(model).policy
        results["l2c2"] = (idx, policy.forward(interp))
        return results

    def _compute_loss(self, forward_results, mb_rollout_data):
        loss_dict = super()._compute_loss(forward_results, mb_rollout_data)
        if "l2c2" in forward_results:
            idx, mu_tilde = forward_results["l2c2"]
            mu = forward_results["policy_results"]["action_mean"][idx, :-1]
            valid = ~mb_rollout_data["mb_dones"][idx, :-1].bool()  # t -> t+1 in one episode
            err = torch.square(mu_tilde - mu).mean(dim=-1)
            l2c2 = (err * valid).sum() / valid.sum().clamp(min=1)
            loss_dict["loss"] = loss_dict["loss"] + self.l2c2_coef * l2c2
            loss_dict["l2c2"] = l2c2.detach()
        return loss_dict

    def _update_stats_buffer(
        self,
        ppo_epoch_idx,
        minibatch_idx,
        microbatch_idx,
        loss_dict,
        forward_results,
        mb_rollout_data,
    ):
        super()._update_stats_buffer(
            ppo_epoch_idx,
            minibatch_idx,
            microbatch_idx,
            loss_dict,
            forward_results,
            mb_rollout_data,
        )
        if "l2c2" in loss_dict:
            self.l2c2_stats[ppo_epoch_idx, minibatch_idx, microbatch_idx] = loss_dict["l2c2"]

    def _get_train_metrics(self):
        metrics = super()._get_train_metrics()
        if self.l2c2_coef > 0:
            metrics["loss/l2c2_avg"] = (
                self.accelerator.gather_for_metrics(self.l2c2_stats).mean().item()
            )
            metrics["l2c2_policy_coef"] = self.l2c2_coef
        return metrics
