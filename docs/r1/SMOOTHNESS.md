# Smoother tracking and better gestures: measurements, literature, changes

## 1. Measurements (deterministic policy in MuJoCo, #51 actuators, 60 s walk + wave)

Share of joint-velocity power above 5 Hz:

| | arms | legs |
|---|---|---|
| reference (IK targets / planner) | 0.5 % | – |
| A2 iteration 2000 | 15 % | 11 % |
| B+ iteration 4500 | 14 % | 14 % |

- **Exploration std at the cap.** The action std sits at its 0.5 cap on every arm joint (mean 0.48) with entropy 0.01. Training rollouts, and the W&B `video` recorded from them, are much noisier than the deployed (mean) policy. `video_eval` now records the deterministic policy during each periodic evaluation.
- **Leg and waist action rate is barely penalized.** `action_rate_l2` works in action units, and the action scale is G1's (D10). So a radian of joint-target change costs 5.5× less on hips/knees and 8.5–13× less on waist/hip yaw than with the R1's own scale (0.25·effort/kp). Ankles: 2× less; arms: 1.1–1.4× less.
- **SONIC's anti-shake misses jitter.** It penalizes |ω| > 1.5 rad/s, which lets small fast jitter through (±0.02 rad at 8 Hz ≈ 1 rad/s). It also penalizes an intended wave: a 2 Hz ±0.5 rad wave peaks at 6.3 rad/s.

## 2. Literature (survey of 2025–2026 trackers; arXiv IDs)

- **L2C2**, a temporal smoothness loss on the action mean (Kobayashi 2022):
  - AGILE 2603.20147: without it, a G1 showed audible high-frequency oscillation.
  - HoST 2502.08378: G1 20/20 vs 11/20 successes, smoothness score 4.01 vs 6.54.
  - Athena-WBC 2607.04837 (withdrawn): a second-difference variant gave action rate 0.56 vs 1.46, with success 98.95 % vs 98.18 % for SONIC-style penalties.
- **Action rate and acceleration weights:**
  - Xie 2602.18312: action-rate weight 0.01 / 0.1 / 1 gives 9.2 / 5.5 / 1.5 % of power above 10 Hz.
  - Weights in use: BeyondMimic −0.1; ASAP and PBHC −0.5; OmniH2O and HOVER −0.625 upper / −3 lower.
  - A second difference penalizes 10 Hz about 480× more than 2 Hz (a first difference: about 22×), so 1–3 Hz gestures survive.
- **Std and entropy:**
  - PHC and AMP train with a fixed small std and no entropy bonus.
  - BeyondMimic: learned std, entropy 0.005.
  - SONIC: std clamp [0.001, 0.5], entropy 0.01.
- **Filters and Lipschitz penalties:**
  - Action low-pass filters hide jitter rather than remove it, and add latency.
  - LCP 2410.11825, a gradient penalty, won only in simulation, and independent re-tests found it weak.
  - None of these beat a filter head to head on a real humanoid.
- **Generated data:**
  - Keep only physically plausible, trackable clips: PHUMA 2510.26236; RLPF 2506.12769 (43–48 % of raw text-to-motion clips were trackable).
  - Keep generated clips to at most about half of the mix.

## 3. What stage B2 changes

Preset `sonic_r1_dex3_teleop_smooth.yaml`, warm-started from B+.

| Change | Setting |
|---|---|
| Exploration | entropy 0, std cap 0.25 |
| L2C2 on the action mean | `TRLSmoothPPOTrainer`, coefficient 1.0, on half of each micro-batch |
| Action second difference | −0.03 |
| Per-joint action rate | −0.1 × {hips/knees 3, hip yaw and waist 4, ankles 1.5, arms 1.2} (softened G1-scale correction) |
| Anti-shake | on wrist/torso ω relative to the reference, deadzone 0.5 rad/s, −0.02 |
| Wrist linear-velocity tracking | 0.5, σ 0.5 m/s (crisper gestures) |
| Data | B + S2 (628 real gesture mocap clips, mirrored included) + Kimodo-G1 gestures and locomotion (filtered by `kimodo_quality.py`) |

**Gates:**
- High-frequency power ratio ≤ 2 % per joint group in the MuJoCo walk + wave test.
- Eval success down by at most 1 point versus B+.
- VR 3-point and MPJPE-L errors up by at most 5 %.

## 4. Kimodo on this machine

Kimodo's text encoder is LLM2Vec on Llama-3-8B, run on the CPU.
- **Wrong CPU results on this machine.** Under torch 2.6 and 2.7 (the `kimodo` and `sonic-train` envs), some CPU kernels return wrong values: `bool_tensor.sum()` with 16 threads returns 2^40 + n; the int64 sum and the single-thread sum are correct.
- **Effect on the encoder.** Under torch 2.6 its activations overflow at layer 3 (mean |h| ≈ 4e30) and every prompt encodes to an all-zero vector. That happens with Kimodo's bundled copy (transformers 5.1) and with the original package (transformers 4.44) alike. Every prompt then yields the same unconditioned motion.
- **torch 2.8 is correct.** The same weights and code give valid, distinct embeddings (norms 127–137, max cosine between prompts 0.92).
- **Consequence.** This is likely why earlier Kimodo samples on this machine rarely matched their prompt.

The fix:
- Compute the embeddings under torch 2.8 with the original `llm2vec` package (`scripts/r1/kimodo_text_embeddings.py`, env `llm2vec`).
- Pass them to `generate_kimodo_motions.py --text-embeddings`.

With real embeddings, every wave, point, beckon and arms-up sample raised the requested hand.
