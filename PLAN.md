# R1 + Dex3-1 SONIC tracking policy — execution runbook

This document is the single source of truth for the project. It is written so that a
person or an agent on a different machine can pick up the work with no other context.
Update the **Status ledger** whenever a task changes state, and keep task IDs stable.

Companion files: `AGENTS.md` (conventions for agents), `gear_sonic/utils/embodiment/r1_spec.py`
(every R1 constant), `gear_sonic/tests/r1/` (what "done" means for the Mac-side work).

---

## 0. Goal and scope

**Goal.** A GEAR-SONIC whole-body controller for the **Unitree R1 (Basic/EDU body, 24 policy
DOF) with two Dex3-1 hands**, fine-tuned in Isaac Lab from NVIDIA's released G1 weights, that
tracks **end-effector (hand) pose targets** supplied by an upstream planner through SONIC's
teleop encoder. Validated in simulation; exported to ONNX with a written input contract.

**In scope:** R1 assets, robot/experiment configuration, R1 motion data, checkpoint warm
start, single-GPU RL fine-tuning, evaluation (SONIC metrics + EE-tracking metrics), MuJoCo
sim-to-sim sanity check, ONNX export + interface doc.

**Out of scope (do not start):** the upstream EE-pose generator, VR teleop, VLA data
collection, real-robot deployment (`gear_sonic_deploy`), finger/head control.

---

## 1. Machines and roles

| Machine | Role | Environment |
|---|---|---|
| **Mac** (this work so far) | asset authoring, data-transfer tooling, checkpoint surgery, unit tests | conda env `sonic` (`environment.yml`): gear_sonic editable + MuJoCo; **no Isaac Lab**, **no datasets** |
| **asblab** (Ubuntu 24.04, RTX 2080 Ti 11 GB; TITAN V lacks RTX support) | GPU bring-up and small training trials | conda env `sonic-train`: Isaac Sim 5.1, Isaac Lab 2.3.1, `gear_sonic[training]`; set `ACCELERATE_TORCH_DEVICE=cuda:1` |
| **24 GB+ GPU box** (future, if full-scale training does not fit on asblab) | 4096-env training, evaluation, export | Isaac Sim + Isaac Lab ≥ 2.3 Python env (Isaac Sim 5.x is mandatory on a 5090) |

Everything under `data/`, `logs_rl/`, `layouts/`, `r1_init/`, `sonic_v1_1/`, `third_party_assets/` is
git-ignored and lives on the machine that produced it.

---

## 2. Status ledger

Legend: `[x]` done + verified · `[~]` done, needs verification on the GPU box · `[ ]` open.
Task IDs (M = Mac-side, G = GPU-box) are referenced from §5.

### Mac-side (complete; 32 tests pass: `pytest gear_sonic/tests/r1 -q`)

- [x] **M0** Repo cloned (`NVlabs/GR00T-WholeBodyControl@7f15131`, 2026-09-15), LFS pulled; `.gitignore` fixed so `gear_sonic/data/` is trackable; `environment.yml` (conda env `sonic`).
- [x] **M1** Upstream sources fetched with pinned commits (`scripts/r1/fetch_upstream_assets.py`, manifest in `third_party_assets/MANIFEST.json`; provenance in `gear_sonic/data/assets/robot_description/R1_PROVENANCE.md`).
- [x] **M2** R1+Dex3 assets generated deterministically by `scripts/r1/build_r1_assets.py`:
  `urdf/r1/r1_dex3.urdf` (+59 STL), `mjcf/r1_dex3.xml`, `robots/r1_ordering.py`, `urdf/r1/R1_DEX3_DERIVED.json`. Verified: MuJoCo loads (24 actuators, 25 bodies, 30.3 kg), SONIC's `Humanoid_Batch` loads, FK agrees with MuJoCo to 3e-4 mm, home pose stands ~1 cm above ground.
- [x] **M3** Ordering-map generator (`utils/embodiment/ordering.py`) reproduces NVIDIA's shipped G1 (29 DOF) and H2 (31 DOF) constants bit-for-bit; G2 confirmed the generated R1 order against live Isaac Lab.
- [x] **M4** Robot config `robots/r1.py` (mjlab gains), `R1Converter` + registry in `order_converter.py`, `robot_mapping["r1_dex3"]`, preset `config/exp/.../sonic_r1_dex3.yaml` (v1.1 recipe, all body names/indices overridden), G1 hard-codes made configurable (`wrist_mujoco_dof_indices`, `joint_pos_multi_future_wrist_for_smpl.joints_idx`, converter selection in `token_losses.py` / `commands.create_offline`).
- [x] **M5** Motion transfer tool `data_process/transfer_g1_motion_lib_to_r1.py` (G1 PKL tree or Bones-SEED CSV → R1 PKL; leg-scale 0.912; clamp/velocity report; drop thresholds). Tested on synthetic motions incl. CLI end-to-end.
- [x] **M6** Checkpoint surgery `scripts/r1/surgery_g1_to_r1_checkpoint.py` + layout dump hook (`++dump_layout_dir=`) in `train_agent_trl.py`. Tested on synthetic layouts/state dicts.
- [x] **M7** Git remote setup: `origin` points to `sourenpash/cvpr`; NVIDIA is retained as `upstream`.

### GPU-box — do in order

- [x] **G1** Environment + G1 baseline smoke test (§5.G1). On `asblab` (2026-09-24),
  conda envs `sonic` (development) and `sonic-train` (Isaac Sim 5.1 / Isaac Lab 2.3.1) are
  installed; the training preflight and 32 R1 tests pass. The G1 preset completed 5 learning
  iterations with 16 envs and printed reward metrics (~3.3 s/iteration). Host ROS sets
  `PYTHONPATH` to Python 3.12 packages, so unset it for tests and training.
- [x] **G2** Isaac Lab ordering verification for R1 (§5.G2). On `asblab` (2026-09-24),
  `num_envs=1` produced `layouts/r1_dex3/layout.json`; the verifier printed `RESULT: OK`
  for 24 joints and 25 bodies. Isaac Lab fuses the fixed head and palm links into parents;
  the anti-shake reward now names the live torso body. This check passed before training.
- [ ] **G3** Data: BONES-SEED gated access granted. Downloaded the 23 GB G1 archive and
  metadata on `asblab`; `curate_bones_s0.py` selected 500 non-mirrored stand, idle, slow/normal
  walk, arm gesture, and reach clips. G1→R1 transfer kept **472/500 (94.4%, 1.261 h)** at
  30 Hz with 5% clamp/velocity thresholds; clamps are dominated by shoulders/elbows, not
  legs. Downloaded NVIDIA's 31 GB split SMPL archive and selectively extracted all **472**
  matching SMPL PKLs. Names match exactly and durations differ by at most 0.0334 s.
  Aligned three-encoder training check, S1/S2 curation, and replay remain (§5.G3–G4).
- [ ] **G4** R1 env bring-up: 16-env random-init trial completed 20 iterations
  (~2.9 s/iteration, ~4.9 GB RTX VRAM). Warm-start sample runs completed at 256 envs
  (~4.2 s/iteration, 5.9 GB), 512 envs (~5.7 s/iteration, 6.9 GB), and 1024 envs
  (~8.6 s/iteration, 8.5 GB). The 3-point tracking reward now scores palms instead of
  wrist origins. A 3-iteration warm-start smoke run completed and synced 132 metrics to the
  new W&B project `sourenpashangpour-university-of-toronto/TRL_R1_Track`; the training
  entrypoint now flushes W&B before Isaac's forced exit. 2026-09-24: the R1 now uses G1's
  default pose and per-joint action scale (D10); the Dex3 visual no longer overlaps the stock
  fist (wrist mesh cut at the mount, D11); training logs BruteForce-style videos to W&B (D12).
  512 envs: 2.4 s rollout + 3.5 s PPO update per iteration (the 91M-parameter nets, not the
  simulator, dominate). Replay check and full-scale throughput remain (§5.G4).
- [x] **G5** Live G1/R1 layouts → surgery → `r1_init/last.pt` (§5.G5). The report shows
  69 tensors copied, 11 gathered, zero template-init; R1 loaded it at step 0 and completed
  five 16-env learning iterations. The checkpoint is local and ignored by Git.
- [ ] **G6** Fine-tuning stages A → B → C (§5.G6). Stage A v1 (`stage_a_s0`, W&B
  `rhldriet`, 512 envs, S0) stopped at iteration ~459/500 when its parent agent session
  ended; it plateaued (time-out 0.36, foot-position terminations 0.51, 3-point reward 0.42,
  rising action std) because the G1 weights ran on R1-specific default pose / action scales
  (leg targets shifted and 2.3–3.7× smaller). A/B from `r1_init` with D10, 120 iterations:
  time-out 0.62 vs 0.37, foot terminations 0.20 vs 0.45, 3-point reward 0.74 vs 0.45,
  body-position reward 0.47 vs 0.31 (iterations 90–120, same data and envs). **Stage A v2**
  (`stage_a_s0_g1param`, W&B `lezotaeg`) ran 100 iterations from `r1_init/last.pt` with
  D10–D12, then was stopped for the teleop-only model (D13); its iteration-100 checkpoint is
  `r1_init/stage_a_v2_it100.pt`. **Teleop Stage A** (`sonic_r1_dex3_teleop_stage_a_s0`, W&B
  `2xvwnnrh`, 1024 envs, S0, no SMPL) runs from that checkpoint; the layout check
  (`layouts/r1_dex3_teleop/template.pt` vs the checkpoint) found every teleop encoder,
  decoder and critic tensor present with matching shapes. Console log
  `logs_rl/console/teleop_stage_a_s0.log`. Launch detached (`setsid nohup … &`) so a
  closing terminal or agent session cannot kill it; Isaac Sim ignores SIGTERM (stop with
  `kill -KILL`). It ran 250 iterations (`r1_init/teleop_a_it250.pt`); its training curves drifted
  down while the adaptive sampler concentrated (mean segment failure rate 0.68 -> 0.15), so
  progress is now measured with uniform evaluation (`PeriodicEvalCallback`, W&B `eval/`).
  **Teleop Stage A2** (`sonic_r1_dex3_teleop_stage_a2`, W&B `m01c6sgc`; `4080i32e` was restarted to add per-group eval rates; the first launch `34g9bnba` crashed at its first eval: `smpl_sim`'s package init -> `mujoco.viewer` -> `glfw` cffi clashes with Isaac Sim's bundled cffi, now bypassed in `PeriodicEvalCallback`; 1024 envs) runs from
  that checkpoint on `data/motion_lib_r1/A2` = contact-corrected S0 (472) + 200 planner
  clips (1.2 h, D14), with VR target noise, evaluating at iteration 1 and every 500.
  **Baseline eval (iteration 1, the 250-iteration teleop policy, training terminations):
  success 76.5 % overall, 89.8 % on mocap clips but 45.0 % on planner-generated walking**
  (the generator gap of 2604.17335); MPJPE-L 30.1 mm, VR 3-point 23.9 mm.
  Console log `logs_rl/console/teleop_stage_a2.log`. Stage B (robustness,
  `sonic_r1_dex3_teleop_robust.yaml`, D15) is prepared. Open items and the literature behind
  each fix: `docs/r1/DEMO_READINESS.md`.
- [ ] **G7** Evaluation incl. EE-tracking metrics and MuJoCo sim-to-sim (§5.G7)
- [ ] **G8** ONNX export + `docs/r1/INTERFACE.md` (§5.G8)

### Quest demo track (real-R1 video in 5 days; plan approved 2026-09-25, D16)

- [x] **Q0 Final training run B+** (`sonic_r1_dex3_teleop_stage_bplus`, W&B `oarg3uey`, log
  `logs_rl/console/teleop_stage_bplus.log`, launched 2026-09-25 01:03 from A2's iteration 2000 =
  `r1_init/teleop_a2_it2000.pt`). A2's uniform eval, success overall / planner / mocap:

  | A2 iteration | 1000 | 1500 | 2000 | 2500 |
  |---|---|---|---|---|
  | overall | 0.877 | 0.896 | **0.909** | 0.865 |
  | planner | 0.790 | 0.800 | **0.875** | 0.770 |
  | mocap | 0.913 | 0.936 | **0.924** | 0.905 |

  B+ = `sonic_r1_dex3_teleop_robust.yaml` on `data/motion_lib_r1/B` (1875 clips):
  - Data: S0_cf 472, S1 803, planner_v2_cf 200, planner_v3 400.
    - S1 (`curate_bones_s1.py`): 1000 standing gesture, in-place manipulation and reach clips; 803 kept, 1.56 h.
    - planner_v3 (`generate_planner_motions.py --profile demo`): mostly idle / slow walk; more starts, stops, turns in place and reversals; 2.4 h.
  - Upper-body grafting on planner clips (`upper_body_augment_prefixes: ["planner_"]`).
  - Robustness terms:
    - measured R1 armature ±30 %;
    - #51 joint friction ×0.75–2.0 (`joint_friction.py`);
    - PD gains ×0.9–1.1;
    - 0–20 ms actuation latency.
  - Uniform eval every 1000 iterations; per-clip outcomes in `<run>/eval/iteration_*.json`.
  - Run directory `sonic_r1_dex3_teleop_robust_stage_bplus-20260925_011642`. Two earlier launches crashed:
    - `EventCfg` rejected the new event terms, fixed by `R1RobustEventCfg`;
    - CUDA OOM while loading the second eval pass, fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - **Iteration-1 eval** (A2 weights under B+'s randomization, on all 1875 clips): 0.719 overall, planner 0.56, mocap 0.794; MPJPE-L 32.0 mm, VR 3-point 25.5 mm.
  - No joint locks up, so Isaac Sim 5.1 applies the friction as a torque, not as a coefficient × reaction force.
- [x] **Q1 ONNX export without Isaac** (`scripts/r1/export_teleop_onnx.py`, `sonic-train` env, CPU):
  - Rebuilds the actor from the run config + `layouts/r1_dex3_teleop/layout.json` and calls SONIC's exporter.
  - Writes `<run>/exported/<name>.{onnx,json}`: 1047 inputs = tokenizer 267 + actor 780.
  - **G0 passed**: `scripts/r1/teleop/policy.py --check` gives max |ONNX − training forward| 4.2e-6 over 64 random observations.
  - The motion library's reference equals the clip joint angles in MuJoCo→Isaac order (`R1_MUJOCO_TO_ISAACLAB_DOF` indexing) with 50 Hz forward-difference velocities (2e-7).
- [~] **Q2 MuJoCo sim-to-sim (gate G1)**.
  - Tools:
    - `scripts/r1/export_r1_reference.py`: motion lib → `.npz`; `data/r1_reference/eval100` holds 50 planner + 50 mocap clips of B.
    - `scripts/r1/teleop/{observations,robot_mujoco,play_clips}.py`: runtime observation builder, PD plant at 1 kHz, the training terminations.
  - A2 iteration 2000 on eval100 (per-clip results in `logs_rl/sim2sim/`):
    - nominal actuators: 81 % (planner 78, mocap 84), MPJPE-L 29 mm;
    - #51 actuators (armature + friction): 76 % (planner 70, mocap 82).
  - Failures:
    - planner: foot position on the new v3 turn/start-stop clips;
    - mocap: wrist height on reaching up/far.
  - Same 100 clips, Isaac with B+'s randomization (iteration-1 eval JSON): 61 %. MuJoCo does not do worse than Isaac on this checkpoint:
    - with nominal actuators, 24 clips pass only in MuJoCo and 4 only in Isaac;
    - with #51 actuators, 23 only in MuJoCo and 8 only in Isaac.
- [~] **Q3 Runtime with the Quest** (`scripts/r1/teleop/`; operator guide `docs/r1/QUEST_TELEOP.md`; conda env `r1rt` from `requirements.txt`).
  - **Planner.** `reference.py` + `planner_loop.AsyncPlannerLoop`:
    - the planner runs on a worker thread and a late plan skips its elapsed frames;
    - on the TITAN V it takes 18 ms per call, merging within LOOK_AHEAD = 2 ticks; on CPU, 60–80 ms (too slow);
    - the TITAN V needs cuDNN 9.5; newer versions fail on sm_70;
    - `LegMap` reproduces the training clips exactly (test).
  - **Targets.** `targets.py`:
    - calibration at A + X fixes the operator frame (heading + poses);
    - hand displacements relative to the headset × 0.65;
    - controller rotations since calibration are applied in the world frame;
    - head yaw/roll clamped onto the waist;
    - a damped-least-squares IK projects everything onto the R1's 5-DOF arms (1.5 ms).
  - **Quest.** `quest.py` wraps televuer (WebXR, pass-through). The server answers on `https://<pc>:8012`; it needed `params-proto<3`.
  - **Recording.** `run.py --record` / `--input replay:`. On CPU the replay is bit-identical.
  - **Closed loop in MuJoCo**, #51 actuators, 10 ms delay, A2 iteration 2000, scripted operator:
    - 60 s real time with forward walk / stop / turn in place / sidestep / back + waving: no fall;
    - palm error p50 55 mm, p95 91 mm; standing on the default targets, the palms sit 2–5 cm forward of the target;
    - policy 1.3 ms on the TITAN V.
  - Open: the live-Quest part of gate G2 (needs the operator).
- [~] **Q4 Real-robot interface.**
  - `robot_unitree.py`: `rt/lowcmd` / `rt/lowstate` in a child process (500 Hz; Python deserialization costs 0.5 ms per message, which starved the policy loop of the GIL in-process).
    - Motor slots from Unitree's R1 `JointIndex` (unitree_sdk2 `dds_wrapper/robots/r1/defines.h`): legs 0–11, **waist roll 12, yaw 13** (no swap, unlike G1), left arm 15–19, right arm 22–26, **head 29/30**.
    - The head is held at 0.
  - `safety.py` watchdog. `run.py` phases: damping → stand-up 3 s → hold → A + X → shadow → blend 2 s → run.
  - `dds_sim.py`: MuJoCo behind the same DDS topics, with a gantry band. On loopback, discovery uses a unicast peer.
  - **DDS closed loop:** 45 s of stand-up, shadow, blend, walking + reaching, gantry released at 9 s. No trips; tick period p50 20.0 ms, p95 21.0 ms, max 23.2 ms; max tilt 12°.
  - Open: the hardware checks (`robot_unitree.py --check`), a Dex3 hold pose.
- [ ] **Q5** Real-robot bring-up and demo takes.

---

## 3. Facts the plan rests on (verified unless marked VERIFY)

### 3.1 Robots

| | G1 (`g1_model_12_dex`, SONIC default) | R1 Basic/EDU + Dex3-1 (ours, `r1_dex3`) |
|---|---|---|
| Height / mass | 1.32 m / ~35 kg | 1.23 m / 28.8 kg body + 2×0.7 kg hands = 30.3 kg (MJCF) |
| Actuated DOF | 29 (legs 6×2, waist y/r/p, arms 7×2) | URDF 26; **policy 24** (legs 6×2, waist roll+yaw, arms **5×2**: shoulder p/r/y, elbow, wrist roll). Head pitch/yaw fixed. |
| MJCF bodies | 30 | 25 |
| Root / torso body | `pelvis` / `torso_link` | `pelvis` / `torso_link` (renamed from `pelvis_link` / `waist_yaw_link`) |
| End effector body | `*_wrist_yaw_link` + `[0.18, ∓0.025, 0]` | `*_wrist_roll_link` + `[0.2185, ∓0.025, 0]` (Dex3 mount at 0.080 + G1's 0.1385 beyond palm base) |
| Head reference | `head_link` (fixed) | `head_yaw_link` (fixed) |
| Leg (hip pitch→ankle roll, straight) | 0.6564 m | 0.5985 m → **root scale 0.912** |
| Pelvis height, feet flat | 0.792 m straight; spawn 0.76 | 0.743 m straight; 0.7335 at mjlab home, 0.7094 at G1's default pose; **spawn 0.72** |
| Default pose / action scale | knee 0.669, hip −0.312, ankle −0.363 …; 0.25·effort/kp | **G1's for all 24 joints** (D10); mjlab HOME kept as `MJLAB_HOME_JOINT_POS` |
| Effort limits | hips 88/139, knee 139, ankles 50, waist 88/50, arm 25, wrist p/y 5 Nm | hips/knee/waist/shoulder p+r 60, ankles 50, shoulder-yaw/elbow/wrist 33 Nm |
| PD gains (ours) | armature·(2π·10 Hz)², ζ=2 | mjlab real-robot values: legs/waist 100/2, ankles 40/2, shoulders 40/2, distal arm 20/1, armature 0.01 |
| Motor layout on hardware | 29 slots | same 29-slot LowCmd; slots 14, 20, 21, 27, 28 are phantom (`unitree_mujoco R1_C++.xml`) |

**All 24 R1 joints exist in G1 with identical names and axes**; G1's extra joints are
`waist_pitch`, `left/right_wrist_pitch`, `left/right_wrist_yaw`. This is what makes the data
transfer (M5) and the weight gather (M6) pure re-indexing with no invented parameters.

### 3.2 Dex3-1 mount (VERIFY on hardware)

Palm frame = `wrist_roll_link + [0.080, 0, 0]`, rpy 0. Derived from the wrist mesh: the
forearm tube (r≈0.028 m) ends and the stock fist begins at x≈0.080; Dex3's palm base ring is
r≈0.028; G1 uses the same "palm at flange" rule. Roll about the forearm axis may differ from
G1 — check the "L/R" marking orientation on the physical hand. The stock-fist mass stays in
`wrist_roll_link` (Unitree publishes no EDU wrist inertials); its geometry is removed from
the wrist visual (D11). BruteForce (`bruteforce/tracking/assets/r1_dex3_constants.py`)
mounts the Dex3 at `(0.0415, ±0.003, 0)`, G1's wrist-yaw-link value, 3.9 cm closer to the
wrist joint; one measurement on the robot settles both projects.

### 3.3 SONIC code paths that depend on the embodiment (all handled)

| Where | What | Handling |
|---|---|---|
| `robots/<robot>.py`, `robot_mapping` | ArticulationCfg, action scale, index maps | `robots/r1.py`, `r1_ordering.py`, `modular_tracking_env_cfg.py` |
| `order_converter.py` | Isaac↔MuJoCo converter | `R1Converter`, `get_converter(robot_type)` registry |
| `commands.py:176` | `lower_joint_indices_mujoco = range(12)` | R1 MJCF lists legs first (asserted in tests) |
| `commands.py:622` (`create_offline`) | hard-coded `G1Converter()` | now `motion_lib_cfg.robot_type` |
| `token_losses.py` ×3 (optional kinematic losses) | hard-coded `G1Converter()` | now `kwargs["robot_type"]` (default g1) |
| `motion_lib_base.py:362` | wrist DOF indices for `randomize_wrist_poses` | `motion_lib_cfg.wrist_mujoco_dof_indices` (R1: `[18, 23]`) |
| `observations/terms/joint_pos_multi_future_wrist_for_smpl.yaml` | Isaac Lab wrist DOF indices `[23..28]` | preset override `[22, 23]` |
| `motion.yaml`, `terminations/ee_body_pos*`, `rewards/anti_shake_ang_vel`, `rewards/tracking_vr_2wrists_local_ori`, `rewards/undesired_contacts`, `events/level0_4` | body names / regexes | all overridden in `sonic_r1_dex3.yaml` |
| `commands.py:4198-4218` (`ForceTrackingCommand`, 6×17 Jacobian) | G1 arm joint lists | **not on our path**; would need R1 lists if force tracking is ever enabled |
| `im_eval_callback.py` (VR 3-point eval subset) | `left/right_wrist_yaw_link` by name | last arm link present (`wrist_yaw`, else `wrist_roll`); G1 unchanged |
| `data_process/convert_soma_csv_to_motion_lib.py` | hard-coded G1 axes | use `transfer_g1_motion_lib_to_r1.py` instead (accepts the same CSVs) |

### 3.4 Network shapes that change (drives M6)

Policy obs (`local_dir_hist`): gravity(3), ang-vel(3), joint_pos(29→24), joint_vel(29→24),
actions(29→24), each ×10 history → 930 → 780. Critic obs (`privileged_mf_hist`): adds
`command_multi_future` (2×10×29 → 2×10×24), 14-body pos/ori (unchanged), etc. Tokenizer:
`command_multi_future_nonflat` changes (10×58 → 10×48) and
`joint_pos_multi_future_wrist_for_smpl` changes (10×6 → 10×2). The G1 and SMPL encoder
first layers, the kinematic decoder's last layer, the dynamic decoder's first and last
layers, `std`, the critic's first layer and the obs normalisers are gathered by index;
the teleop encoder transfers verbatim.

---

## 4. Repository conventions and setup

- Upstream: `https://github.com/NVlabs/GR00T-WholeBodyControl` (remote `upstream`). **M7:**
  ```bash
  git remote rename origin upstream
  git remote add origin https://github.com/sourenpash/cvpr.git
  git push -u origin main
  ```
- Keep upstream files minimally touched (new robot = new files + one mapping entry + one
  preset). All R1 constants live in `gear_sonic/utils/embodiment/r1_spec.py`; the preset and
  tests assert they stay in sync. New tooling under `scripts/r1/`, tests under `gear_sonic/tests/r1/`.
- On `asblab`, activate `sonic-train` and run Isaac Lab commands with
  `env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ACCELERATE_TORCH_DEVICE=cuda:1 python ...`.
  Host ROS sets a Python 3.12 `PYTHONPATH`; GPU 0 is a non-RTX TITAN V, while GPU 1 is the
  RTX 2080 Ti. Do not mask devices with `CUDA_VISIBLE_DEVICES=1`: AppLauncher then selects
  the TITAN V as `cuda:0`.
- Re-generate assets only via `scripts/r1/build_r1_assets.py` (never hand-edit the URDF/MJCF/ordering file).
- Lint: `ruff check` + `black` (line length 115/100 per `pyproject.toml`) on files you touch; do not reformat upstream files wholesale.
- Housekeeping on the Mac: a stray duplicate env exists at `/opt/homebrew/Caskroom/miniforge/base/envs/sonic`; remove with `/opt/homebrew/bin/mamba env remove -n sonic -y`.

---

## 5. Task cards

Each card: purpose → commands → acceptance. Paths are relative to the repo root; `$DATA` is
the data root on the GPU box (default `data/`).

### G1 — Environment and G1 baseline smoke test

Purpose: prove the Isaac Lab stack before touching R1.

```bash
# Ubuntu 22.04, NVIDIA driver per Isaac Sim version (5.x + driver >= 570 for RTX 5090)
# 1) Install Isaac Sim + Isaac Lab >= 2.3 (docs/source/getting_started/installation_training.md)
# 2) In the Isaac Lab python env:
git clone <our-repo-url> && cd <repo> && git lfs pull
pip install -e "gear_sonic/[training]"
python check_environment.py --training
python -m pytest gear_sonic/tests/r1 -q          # Mac-side tests must also pass here
# 3) Baseline
python download_from_hf.py --sample               # sample_data/robot_filtered + smpl_filtered
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_v1_1 \
  num_envs=16 headless=True ++algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```
Acceptance: metrics print for 5 iterations; `pytest gear_sonic/tests/r1` passes.

### G2 — Verify Isaac Lab ordering for R1 (blocking)

Purpose: confirm the inferred breadth-first ordering in `r1_ordering.py` against Isaac Lab.
Needs any small R1 motion directory; if G3 is not done yet, create one from the shipped G1
sample data:

```bash
python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \
  --input sample_data/robot_filtered --output $DATA/motion_lib_r1/sample --num_workers 4
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  num_envs=1 headless=True ++dump_layout_dir=layouts/r1_dex3 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$DATA/motion_lib_r1/sample
python scripts/r1/verify_isaaclab_order.py --layout layouts/r1_dex3/layout.json
```
Acceptance: `RESULT: OK`. If MISMATCH: paste the printed live order into
`utils/embodiment/ordering.py` (fix the traversal rule so G1/H2 still reproduce), re-run
`build_r1_assets.py`, re-run the tests, repeat. Also confirms every body referenced by the
preset exists (a missing body raises during env construction).

### G3 — Data

```bash
# S0: selective download/extraction fits on asblab's SSD. Checkpoint is already local.
hf download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv \
  --repo-type dataset --local-dir $DATA/bones_seed
python scripts/r1/curate_bones_s0.py --dest $DATA/bones_seed/s0_source_curated
python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \
  --input $DATA/bones_seed/s0_source_curated/g1/csv --output $DATA/motion_lib_r1/S0 \
  --fps 30 --fps_source 120 --num_workers 8 --max-clamp-frac 0.05 --max-vel-frac 0.05
hf download nvidia/GEAR-SONIC bones_seed_smpl/bones_seed_smpl.tar.part_a{a,b,c,d,e,f,g} \
  --local-dir $DATA/sonic_smpl_source
python scripts/r1/extract_s0_smpl.py # kept R1 names -> data/smpl_filtered/S0
```

For S1/S2, extend the metadata-driven selection and extract only the needed archive
members; full extraction is too large for `asblab`. Keep G1 CSV, R1 PKL, and SMPL PKL
names aligned. Subset goals:
- **S0 bring-up** (~500): idle/stand, slow walking, standing arm gestures/reaching.
- **S1 manipulation-centric** (~5–10K): Interactions + Everyday + upper-body Communication + locomotion basics.
- **S2 broad** (~30–50K incl. mirrored `_M`).

Acceptance: `transfer_report.json` shows ≥ 85 % motions kept; `most_clamped_joints` is
dominated by shoulder roll / wrist roll (expected) and not by legs; replay (G4) looks right.
Record counts/hours in `$DATA/motion_lib_r1/REPORT.md`.

Track 2 (quality, parallel, optional for the first policy): IK retargeting from Bones-SEED BVH
with GMR (R1 config exists in the fork `FredericAS1231/video2humanoid@R1_adaptation`,
`smplx_to_r1.json`) for S1; convert with a `--robot r1` path added to
`convert_soma_csv_to_motion_lib.py` (mirror `transfer_g1_motion_lib_to_r1.py`'s axis/pose_aa code).
`/home/asblab10/BruteForce` was inspected read-only: its R1 GMR retargeter documents an
elbow T-pose calibration fix (G1 elbow reference at π/2, not zero), which matters when
porting IK retargeting. The direct G1-CSV joint-name transfer used for S0 does not apply
that GMR calibration and should not blindly copy its code or motion files.

### G4 — R1 environment bring-up

```bash
M=$DATA/motion_lib_r1/S0; S=data/smpl_filtered
# replay reference motions on the R1 (visual check: feet on ground, arms plausible)
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  ++replay=True num_envs=4 headless=False \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$M
# stability at small scale (if the robot explodes/falls at t=0: lower kp / raise kd in r1_spec.ACTUATOR_GROUPS)
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  num_envs=16 headless=True ++algo.config.num_learning_iterations=20 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$M \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$S
# throughput / VRAM at full scale (drop to 2048 on a 3090 if PhysX OOMs)
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  num_envs=4096 headless=True ++algo.config.num_learning_iterations=20 use_wandb=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$M \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$S
```
Acceptance: replay correct; 20 iterations at 4096 envs without OOM; record s/iteration and
VRAM in the ledger (this sizes G6).

### G5 — Warm start (layout dumps → surgery)

```bash
# G1 layout (source weights are sonic_v1_1)
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_v1_1 \
  num_envs=1 headless=True ++dump_layout_dir=layouts/g1_v1_1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
# R1 layout + template (same as G2, but with SMPL data so the smpl encoder is instantiated)
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  num_envs=1 headless=True ++dump_layout_dir=layouts/r1_dex3 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=$M \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$S
python scripts/r1/surgery_g1_to_r1_checkpoint.py \
  --src sonic_v1_1/last.pt --src-layout layouts/g1_v1_1/layout.json \
  --tgt layouts/r1_dex3/template.pt --tgt-layout layouts/r1_dex3/layout.json \
  --out r1_init/last.pt --dry-run     # inspect, then without --dry-run
```
Acceptance: the report lists **no** `kept_template_init` entries other than genuinely new
modules; gathered tensors include G1/SMPL encoder first layers, `g1_dyn` first + last layer,
`g1_kin` last layer, `std`/`log_std`, critic first layer, obs normalisers. If a tensor is "ambiguous"/"no map",
inspect its name and layout and extend `MapBuilder` (add a test in `test_surgery.py`).

### G6 — Fine-tuning

Common flags: `checkpoint=r1_init/last.pt` (loads policy+critic, no optimizer), data paths,
`headless=True`. Use 512 envs on `asblab`'s 11 GB RTX 2080 Ti (1024 only if the full data
fits); 4096 is for a 24 GB+ GPU. W&B (metrics + `video` every 250 it, D12) is enabled by the
R1 preset. Save every 500 it (`++callbacks.model_save.save_frequency=500`). Stage A v2:

```bash
setsid nohup env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ACCELERATE_TORCH_DEVICE=cuda:1 \
  python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3 \
  num_envs=512 headless=True checkpoint=r1_init/last.pt ++algo.config.num_learning_iterations=10000 \
  ++callbacks.model_save.save_frequency=500 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_r1/S0 \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered/S0 \
  exp_var=stage_a_s0_g1param > logs_rl/console/stage_a_s0_g1param.log 2>&1 < /dev/null & disown
```

Teleop-only (D13, the current path; no SMPL data):

```bash
setsid nohup env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES ACCELERATE_TORCH_DEVICE=cuda:1 \
  python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_r1_dex3_teleop \
  num_envs=1024 headless=True checkpoint=r1_init/stage_a_v2_it100.pt \
  ++algo.config.num_learning_iterations=10000 ++callbacks.model_save.save_frequency=500 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_r1/S0 \
  exp_var=stage_a_s0 > logs_rl/console/teleop_stage_a_s0.log 2>&1 < /dev/null & disown
```

S1 for the Quest path: locomotion variety that matches the planner's modes (turning,
side-stepping, backward walking, stop/start, slow and normal walks) plus standing and walking
arm reaches/manipulation; no SMPL extraction needed.

| Stage | Data | Purpose | Overrides | Stop when |
|---|---|---|---|---|
| A | S0 | R1 dynamics, balance, stepping | defaults | `Episode_Reward/time_out > 0.8`, no falls at start |
| B | S1 | manipulation-centric, teleop-heavy | `++manager_env.commands.motion.teleop_sample_prob_when_smpl=0.7` | `tracking_vr_5point_local > 0.75` |
| C | S2 | breadth, robustness | defaults (pushes, terrain on) | plateau or budget |

Monitor (`docs/source/user_guide/training.md`): `tracking_vr_5point_local > 0.80`,
`tracking_relative_body_pos > 0.44`, `time_out > 0.90`; `adp_samp/failure_rate_max` must
separate from the mean after a few hundred iterations. Budget guide: NVIDIA trains from
scratch on 64+ GPUs for 100K it; with the warm start and subsets expect days per stage on one
GPU — refine from the s/iteration measured in G4.

### G7 — Evaluation

```bash
python gear_sonic/eval_agent_trl.py +checkpoint=logs_rl/TRL_R1_Track/<run>/last.pt +headless=True \
  ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=128 "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512"
```
Targets: `success_rate > 0.95`, `mpjpe_l < 35 mm` on held-out S1/S2 (G1 reaches 0.97 / 30 mm).
Then add **EE-tracking eval** (new script under `scripts/r1/eval_ee_tracking.py`, to write on
the GPU box): run only through the teleop encoder with an idle lower-body reference; sweep a
grid of palm targets over the reachable workspace; report palm position error (mean / p95),
palm-normal angle error, settling time after a step change; produce a reachability map for
the upstream team. Robustness: pushes on, hand payload 0–0.5 kg. Sim-to-sim: export (G8) and
run `gear_sonic/scripts/run_sim_loop.py` with an R1 `wbc_configs` entry on the
`unitree_mujoco` R1 scene.

### G8 — Export and hand-off

```bash
python gear_sonic/eval_agent_trl.py +checkpoint=<best>.pt +headless=True ++num_envs=1 +export_onnx_only=true
```
Deliver `*_teleop.onnx`, `*_encoder.onnx`, `*_decoder.onnx` (24 actions) and write
`docs/r1/INTERFACE.md`: frames (heading-normalised pelvis anchor), units, point order
(L palm, R palm, head point), quaternion convention (wxyz), 50 Hz, future-frame stride,
how lower-body intent is supplied, palm offset used, reachability map, and the 5-DOF-arm
caveat (position tracked strongly; orientation reduced to palm-normal).

---

## 6. Design decisions (why)

- **D1 24-DOF policy, head fixed, Dex3 as passive mass** — matches Unitree's own R1 RL model; SONIC's recipe never assumes a head DOF; head/fingers can be driven by other modules later.
- **D2 Recipe `sonic_v1_1`** — heading-normalised targets + `randomize_wrist_poses` were built for model-generated hand targets (our case). Fallback `sonic_release` (smaller MLPs) if 24 GB is tight.
- **D3 Data: name-based transfer first, IK retargeting second** — unblocks training in days and keeps SMPL alignment; IK later for hand-placement fidelity on manipulation clips.
- **D4 Warm start by index gather** — every R1 joint exists in G1, so no parameter is invented; the only way to converge on one GPU.
- **D5 mjlab PD gains** — real-robot validated; SONIC's armature formula is unusable without R1 rotor inertias.
- **D6 EE contract for a 5-DOF arm** — track palm position strongly, orientation weakly (palm normal); waist yaw/roll free for reach.
- **D7 mjlab naming** in both URDF and MJCF; identical trees (asserted).
- **D8 Single-GPU regime** — 4096 envs, curated subsets, staged curriculum; 5090 ⇒ Isaac Sim 5.x.
- **D9 Palm reward alignment** — SONIC's position reward and the teleop encoder must measure
  the same R1 palm points; the orientation term stays weak because R1 arms have 5 DOF.
- **D10 G1 action parameterization** — SONIC's action is `default_pose + scale·a` and the
  policy observes `q − default_pose`, so the gathered G1 weights (D4) only mean the same
  thing on the R1 with G1's default pose and per-joint scale (`r1_spec.INIT_JOINT_POS`,
  `ACTION_SCALE`; checked against `g1.py` in the tests). Gains stay mjlab's (D5). Measured
  effect in §2 G6.
- **D11 Dex3 as SONIC's G1 carries it** — NVIDIA's `g1_model_12_dex` attaches the Dex3 with
  every finger joint fixed; the R1 does the same (fused into `wrist_roll_link`: mass, convex
  hull collision, meshes). The stock fist's geometry is cut from the wrist visual
  (`*_wrist_roll_link_forearm.STL`), as BruteForce deletes it; its mass stays.
- **D12 Training videos** — BruteForce's W&B format (`video` key, 640×480, 50 fps, 10 s of
  env 0): reference ghost | policy | SMPL human, plus palm targets and palm points,
  rendered by MuJoCo on the TITAN V in a subprocess (`RolloutVideoCallback`,
  `scripts/r1/render_rollout_video.py`); mp4/npz in `<run>/videos/train/`.
- **D13 Quest teleop, teleop-only SONIC** (user decision 2026-09-24) — the robot is driven by
  a Meta Quest: head + two hands through SONIC's VR 3-point (teleop) encoder, walking through
  its lower-body command, which SONIC's deploy stack fills from the joystick-driven kinematic
  planner in `VR_3PT` mode (`docs/source/tutorials/vr_wholebody_teleop.md`). Fingers are
  handled separately. Only the teleop encoder and the action decoder are built and trained
  (`sonic_r1_dex3_teleop.yaml`, `actor_critic/universal_token/teleop_mlp_v1.yaml`): actor
  40.2M instead of 52.1M parameters, no robot-motion/SMPL encoder, no kinematic decoder, no
  latent-alignment losses, no SMPL data. The decoder keeps its pretrained size (the warm start
  is the point); the critic (39M, training only) is unchanged. The G1 planner's output maps to
  the R1's lower-body joints by name, as the training data does — to verify in G7.
- **D14 Planner-in-the-loop data and contact-corrected transfer** — SONIC's VR_3PT mode feeds
  the teleop encoder planner-generated legs, and a tracker trained only on clean mocap breaks
  on generator artifacts (2604.17335: 0.23 -> ~0.99 success with the generator in the loop).
  `scripts/r1/planner_loop.py` ports the deployed planner loop; `generate_planner_motions.py`
  drives it with Quest-like stick scripts and transfers the G1 output like BONES-SEED. The
  transfer now pins stance feet and grounds soles per frame (contacts detected on the G1
  source; Kovar 2002, PHUMA 2510.26236, PBHC 2506.12851, ProtoMotions): S0 frames > 2 cm under
  ground 7.5 % -> 0 %, stance skating 1.25 -> 0.61 cm/s. `--no-contact-fix` = old behaviour.
- **D15 Robustness as a separate stage** — measured R1 leg armature (unitree_rl_mjlab #51),
  PD gains x0.9-1.1 and 0-15 ms actuation latency (`delayed_actions.py`) are applied as a
  fine-tune from A2 (`sonic_r1_dex3_teleop_robust.yaml`) so their effect on `eval/` is
  measurable; joint friction waits until its Isaac Sim 5.1 units are verified.
- **D16 One final run for the real-robot demo** (superseded D15's staging; plan of 2026-09-25).
  - With one GPU and the robot needed on days 3–4, the remaining GPU time goes to the configuration that ships: B+ = robustness terms + grafting + S1 + planner v3, from A2's best checkpoint. A2 is the baseline and the fallback.
  - The #51 fit comes from a comment by a user who measured one R1, not from Unitree. Per group (armature / friction N m):
    - legs and waist 0.05 / 2.5;
    - ankles 0.10 / 1.5;
    - shoulder pitch/roll 0.01 / 2.5;
    - shoulder yaw, elbow, wrist roll 0.01 / 0.2.
  - Upward friction randomization, because unloaded backdriving under-measures loaded legs. The commenter's sim-trained policy walked on the real R1 on the first try.
  - Friction goes in as Isaac Sim 5 static = dynamic efforts. The one-joint unit test (`scripts/r1/isaac_joint_friction_test.py`) hung at startup. Instead, B+'s iteration-1 eval checks the units: a coefficient × joint reaction force would lock the legs.
  - `EventCfg` only accepts declared terms, hence `mdp/r1_events.py:R1RobustEventCfg`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Isaac Lab ordering differs from inferred rule | G2 is blocking; fix rule, regenerate, tests keep G1/H2 honest |
| Dex3 mount transform / roll wrong | single constant in `r1_spec`; VERIFY on hardware; palm offset re-derived automatically |
| Distal arm (33 Nm, 0.7 kg hands) too weak for dynamic arm clips | transfer report velocity/clamp filters; relax `ee_body_pos` termination early in stage A |
| Track-1 data foot skating | root scaled by leg ratio; loader re-grounds frame 0; Track 2 for manipulation clips. Measured on 80 S0 clips (MuJoCo FK): frames with a sole > 2 cm below ground 0.9 % (G1 source) → 7.5 % (R1; p90 clip 25 %), median stance skating 0.2 → 1.3 cm/s. Candidate fix: per-frame ground correction in `transfer_g1_motion_lib_to_r1.py` |
| Surgery leaves a critical layer randomly initialised | dry-run report must be clean before training; extend `MapBuilder` |
| PhysX OOM at 4096 envs on 24 GB | 2048 envs; `sonic_release` decoder |
| Upstream moves weekly | `upstream` remote; minimal diffs; `pytest gear_sonic/tests/r1` after each merge |

## 8. Open questions for the team

1. R1 tier: **EDU** confirmed (only tier with secondary development + Dex3)? Physical robot available later?
2. Dex3 mount: same wrist interface as G1? RGB-camera variant (extra mass/offset)?
3. Who owns head gaze and fingers at runtime (D1 assumes not SONIC)?
4. Which GPU/box (decides Isaac Sim version, env count)?
5. Upstream targets: full 6-D palm poses, or position + approach direction? (D6 weighting)
6. ~~Lower-body intent~~ Answered (D13): Meta Quest head + hands, walking via SONIC's
   joystick-driven kinematic planner (VR_3PT mode).
7. ~~Encoders~~ Answered (D13): teleop encoder only; robot-motion and SMPL encoders dropped.

## 9. File map (what was added or changed)

```
AGENTS.md, PLAN.md, environment.yml, .gitignore (re-include gear_sonic/data/, ignore scratch dirs)
scripts/r1/fetch_upstream_assets.py          pinned download of unitree_ros / unitree_rl_mjlab / unitree_mujoco sources
scripts/r1/build_r1_assets.py                URDF + MJCF + ordering + derived constants (deterministic)
scripts/r1/surgery_g1_to_r1_checkpoint.py    G1 -> R1 weight gather (needs layout.json x2 + template.pt)
scripts/r1/verify_isaaclab_order.py          compare live Isaac Lab order with r1_ordering.py
gear_sonic/utils/embodiment/{r1_spec,ordering,inertia,layout_dump}.py
gear_sonic/envs/manager_env/robots/{r1,r1_ordering}.py (+ __init__.py import)
gear_sonic/envs/manager_env/modular_tracking_env_cfg.py   robot_mapping["r1_dex3"]
gear_sonic/trl/utils/order_converter.py      R1Converter, CONVERTERS, get_converter()
gear_sonic/trl/losses/token_losses.py        converter via kwargs["robot_type"]
gear_sonic/envs/manager_env/mdp/commands.py  create_offline converter via motion_lib_cfg.robot_type
gear_sonic/utils/motion_lib/motion_lib_base.py   wrist_mujoco_dof_indices configurable
gear_sonic/train_agent_trl.py                ++dump_layout_dir hook
gear_sonic/config/exp/manager/universal_token/all_modes/sonic_r1_dex3.yaml
gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py
gear_sonic/data/assets/robot_description/{urdf/r1/*, mjcf/r1_dex3.xml, R1_PROVENANCE.md}
gear_sonic/tests/r1/*                        34 tests (assets, ordering, config, transfer, surgery)
gear_sonic/trl/callbacks/rollout_video_callback.py + config/callbacks/rollout_video.yaml   W&B videos (D12)
scripts/r1/render_rollout_video.py           MuJoCo renderer for the recorded rollouts (standalone)
scripts/r1/{curate_bones_s0,extract_s0_smpl}.py   S0 selection from BONES-SEED metadata / matching SMPL
docs/r1/TRAINING.md                          reward contract, data, logging
```

## 10. References

- Repo docs: `docs/source/user_guide/new_embodiments.md`, `training.md`, `training_data.md`, `references/training_code.md`, `getting_started/installation_training.md`
- Unitree R1: https://support.unitree.com/home/en/R1_developer/about_R1 · URDF https://github.com/unitreerobotics/unitree_ros/tree/master/robots/r1_description · RL model https://github.com/unitreerobotics/unitree_rl_mjlab · MuJoCo https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/r1
- Dex3-1 URDFs: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/dexterous_hand_description/dex3_1
- Data/weights: https://huggingface.co/datasets/bones-studio/seed · https://huggingface.co/nvidia/GEAR-SONIC
- Methods and reward rationale: https://arxiv.org/abs/2511.07820 (SONIC) ·
  https://arxiv.org/abs/2509.16757 (HDMI) · https://arxiv.org/abs/2511.15200 (VIRAL) ·
  `docs/r1/TRAINING.md`
- Retargeting: https://github.com/YanjieZe/GMR · https://github.com/NVIDIA/soma-retargeter
