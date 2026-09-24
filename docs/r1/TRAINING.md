# R1 + Dex3-1 training choices

The R1 preset follows the SONIC v1.1 motion-tracking recipe and keeps its three encoders:
robot motion, teleop targets, and SMPL motion. The G1 checkpoint is gathered into the R1
24-DOF layout before fine-tuning. `PLAN.md` contains the stage commands and acceptance
criteria.

## Quest teleop model (current)

The robot is driven by a Meta Quest: the headset and two controllers give SONIC's VR 3-point
input (head and both palms), and walking comes from the teleop encoder's lower-body command,
which SONIC's deploy stack fills from its joystick-driven kinematic planner (`VR_3PT` mode).
The fingers are controlled separately. `sonic_r1_dex3_teleop.yaml` therefore builds only the
teleop encoder and the action decoder (`teleop_mlp_v1.yaml`): a 40.2M-parameter actor instead
of 52.1M, no SMPL data, and no latent-alignment losses. The rewards, terminations, robot, and
data are the same as the three-encoder preset's, and its checkpoints warm-start the teleop
model: policy weights load by name, and the unused encoders are ignored.

## Reward contract

The reward uses SONIC's dense reference tracking rather than an object-manipulation task
reward. The active preset is
`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_r1_dex3.yaml`, which selects
`tracking/local_feet_acc_energy_5pt`. The configured terms are:

| Term | Weight | Purpose |
| --- | ---: | --- |
| root position / orientation | 0.5 / 0.5 | Keep the reference anchor |
| relative body position / orientation | 1.0 / 1.0 | Track the 14 named R1 links |
| body linear / angular velocity | 1.0 / 1.0 | Match motion timing |
| `tracking_vr_5point_local` | 2.0 | Track torso point and **two palm points** in the local root frame |
| wrist orientation | 0.4 | Track two wrist frames; palm mount has zero relative rotation |
| action rate / joint limit / undesired contact | -0.1 / -10 / -0.1 | Smooth, feasible, non-colliding motion |
| wrist/torso anti-shake / feet acceleration / energy | -0.005 / -2.5e-6 / -1e-4 | Reduce jitter and effort |

The inherited reward function is named `tracking_vr_5point_local`, but the R1 v1.1 preset
supplies three points: torso and two palms. Its wrist offsets are the same
`[0.2185, ±0.025, 0]` values used by the teleop command. The constant is in
`r1_spec.py`; the preset and test assert that the command and reward agree. The Dex3 mount
and its palm offset are **VERIFY** estimates until CAD or hardware measurement.

SONIC's [paper](https://arxiv.org/abs/2511.07820) and released v1.1 configuration motivate
the reference-tracking and smoothness terms. [HDMI](https://arxiv.org/abs/2509.16757)
adds a contact and object-state interaction reward for object-aware references.
[VIRAL](https://arxiv.org/abs/2511.15200) uses stage-dependent task rewards for a
vision-conditioned manipulation teacher. Neither interaction setup is present in this
motion-only training environment: there are no object poses, desired contact points, or
contact-force annotations to score. Add those terms only with corresponding data and an
object-aware task, after the tracking policy works.

## Warm-start parameterization

SONIC's action is `target = default_pose + action_scale * a`, and the policy observes
`q - default_pose`. The G1 checkpoint is gathered into the R1 layout by joint name, so the R1
uses G1's default pose and G1's per-joint action scale for all 24 joints
(`r1_spec.INIT_JOINT_POS`, `r1_spec.ACTION_SCALE`); the PD gains stay unitree_rl_mjlab's.
The first Stage A run used mjlab's standing pose and the R1 rule `0.25 * effort / kp`: the
transferred policy's knee target at zero action was 0.30 rad instead of 0.669, and leg
offsets were 2.3–3.7 times smaller. It plateaued with half of all episodes ending on foot
position. With the G1 parameterization, the same data reached a 0.74 three-point reward
and 0.62 time-out rate within 120 iterations (PLAN.md §2 G6).

## Robot model

The Dex3 hands are attached the way NVIDIA's released G1 carries them: every finger joint is
fixed, and Isaac Lab fuses the hand into `wrist_roll_link` (mass, convex-hull collision,
meshes). The stock fist that the Dex3 replaces is cut from the wrist's visual mesh.

## Data and logging

`scripts/r1/curate_bones_s0.py` selects 500 non-mirrored clips from the BONES-SEED metadata:
100 neutral stands, 100 idle transitions, 150 basic walks, 100 standing arm gestures, and
50 reaches. It extracts only those G1 CSVs from the 23 GB archive. Transfer them with
`transfer_g1_motion_lib_to_r1.py` at 30 Hz and inspect `transfer_report.json` for joint
clamping and velocity violations before stage A. The NVIDIA SMPL archive supplies the
third encoder's data; the shipped sample SMPL clips are only suitable for smoke tests.

The R1 preset logs online to the `TRL_R1_Track` W&B project in the authenticated
`sourenpashangpour-university-of-toronto` entity. Run files stay in `logs_rl/wandb` and
checkpoints in `logs_rl/TRL_R1_Track`. Record the stage in `exp_var` so runs can be
distinguished (for example, `exp_var=stage_a_s0`).

Every 250 iterations the run also logs a 10-second video of env 0 under the W&B key `video`,
in the format of BruteForce's runs: the reference motion as a green ghost, the policy's
robot, and the SMPL human side by side. Red spheres mark the palm targets, re-anchored to the
robot's root as SONIC's tracking terms compare them; blue spheres mark the robot's palms. The
caption gives the iteration, clip, and active encoder. MuJoCo renders the video on the
TITAN V in a subprocess, so training is not slowed; the `.npz` recording and `.mp4` stay in
`<run>/videos/train/`. Change the interval with
`++callbacks.rollout_video.every_n_iterations=N`.
