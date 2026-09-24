# Quest teleop demo: open items and paper-backed solutions

Goal: a Meta Quest operator drives the Unitree R1 + Dex3 (head, both hands and walking) with the
teleop-only SONIC policy (`sonic_r1_dex3_teleop`), reliably enough for a live research-talk demo.
Fingers are handled separately. This page lists what was open, what the literature says works, and
what is implemented. Commands and run history are in `PLAN.md`.

## 1. Walking from the Quest (lower-body command)

**Problem.** The teleop encoder takes head and hand targets plus the next 10 frames of lower-body
joint positions and velocities. The Quest provides head, hands and thumbsticks.

**Solution.** Use SONIC's `VR_3PT` mode:
- Thumbsticks drive SONIC's kinematic planner (`planner_sonic.onnx`, which outputs G1 motion).
- The planner's legs are mapped to the R1 by joint name, with the same leg scaling and ankle-roll
  clamp as the training data.
- The VR 3 points drive the upper body.

Keep this rather than switching the encoder to a velocity command:
- Planner+SONIC survived 98.5 % vs 43 % for a velocity-command controller (SONIC 2511.07820).
- Velocity controllers take 3.5x more, smaller steps (OpenHLM 2606.22174).
- Simplified commands cannot express complex leg patterns (ULC 2507.06905).

**Train with the planner in the loop.** A tracker trained on clean mocap reached 0.23 success on a
generator's output. Fine-tuning with the generator in the loop brought it to ~0.99 (2604.17335).

**Implemented:**
- `scripts/r1/planner_loop.py` ports the deployed C++ planner loop: 10 Hz replanning, 30->50 Hz
  resampling, 8-frame cross-fade.
- `scripts/r1/generate_planner_motions.py` generates Quest-like stick scripts: idle, slow walk and
  walk; forward, backward, sideways and diagonal; turning; start and stop. It produced 200 R1 clips
  (1.2 h) in `data/motion_lib_r1/planner_v2_cf`, used in Stage A2.
- The planner's side-steps exceed the R1 and G1 ankle-roll limit (0.262 rad). The transfer clamps
  them, and the demo runtime must apply the same clamp.

**For the demo:**
- Use idle + slow walk (0.2-0.8 m/s). SONIC's own 3-point interface exposed two walk speeds.
- Releasing the stick returns to idle.
- Run the planner on a GPU: it takes 70 ms on 4 CPU cores against a 100 ms replan tick (SONIC: < 5 ms
  on a laptop GPU).

**Measured gap.** The first teleop policy, trained on mocap only, completed 89.8 % of mocap clips
but only 45.0 % of planner clips (uniform eval, 20 s clips; W&B `m01c6sgc`, iteration 1). Stage A2
trains on both; `eval/success/success_rate_planner` tracks the gap.

**Open:** a closed-loop sim test of the whole chain (Quest log -> planner -> R1 map -> policy) with
start/stop, reversals, turning in place, and reaching while walking.

## 2. Reference data quality (G1 -> R1 transfer)

**Problem.** Copying G1 joint angles onto the R1's differently proportioned legs caused:
- 7.5 % of frames with a sole > 2 cm below ground (G1 source: 0.9 %);
- 6x the source's stance-foot skating.

**Solution.** `transfer_g1_motion_lib_to_r1.py`, on by default:
- Detect contacts on the G1 source: sole below 2.5 cm (PHUMA 2510.26236) and slower than 0.15 m/s
  (ProtoMotions' BONES-SEED converter).
- Pin stance feet by moving the root (Kovar et al. 2002, footskate cleanup).
- Set the root height per frame so stance soles touch z=0, with no sole below -5 mm (PBHC
  2506.12851, ProtoMotions).
- Record per-clip quality metrics in the transfer report.

Contact-preserving retargeting raises RL success (OmniRetarget 2509.26633: 82 % vs 51 % for GMR).

**Result on S0:**
- Frames > 2 cm under ground: 7.5 % -> 0 %.
- Median stance skating: 1.25 -> 0.61 cm/s.
- The residual is double support, where the R1's leg proportions move both planted feet.

**Open, optional:**
- Leg IK polish for the double-support residual.
- Simulation-based clip filtering (H2O 2406.08858: dropping ~15 % of clips gave +4.6 pts success).

## 3. Sim-to-real robustness (Stage B, `sonic_r1_dex3_teleop_robust.yaml`)

Fine-tune from Stage A2 and compare `eval/` metrics.

- **Actuator model.** Leg and waist armature 0.05, ankles 0.10, measured on one R1 (unitree_rl_mjlab
  issue #51). With Unitree's 0.01, the simulated hip overshoots ~30 % vs ~2 % on the real robot.
  Joint friction (#51: 2.5 / 1.5 N m) is **not** transferred yet: Isaac Lab 2.3 documents joint
  friction as a unitless coefficient, not MuJoCo's torque. Verify with a one-joint test first.
- **PD gains x0.9-1.1** (KungfuBot 2506.12851; ASAP 2502.01143).
- **Actuation latency 0-15 ms** (`gear_sonic/envs/manager_env/mdp/delayed_actions.py`).
  - Other trackers randomize 0-60 ms (OmniH2O 2406.08858, HOVER 2410.21229, ASAP, GMT 2506.14770,
    CLONE 2506.08931).
  - BeyondMimic 2508.08241: a policy trained without latency fell with 5-10 ms injected.
- **VR target noise.** ±1 cm and ±0.02 on the quaternion components, already in A2 (CLONE; OmniH2O).
- **Mass.** SONIC already randomizes wrist and torso mass x0.8-2.5, which covers hand payloads.

**Open:**
- MuJoCo sim-to-sim gate with the measured actuator model. SONIC's bar: success > 0.97 and MPJPE-L
  < 30 mm. This needs the R1 runtime (policy ONNX + observations + planner) in MuJoCo.
- Verify joint friction in Isaac Sim 5.1 units.

## 4. Evaluation

Adaptive sampling concentrates episodes on failing clip segments, so training curves drift down as
the policy improves. In the first teleop run:
- the mean segment failure rate fell 0.68 -> 0.15;
- the time-out rate fell 0.66 -> 0.54.

`PeriodicEvalCallback` plays every clip once every 500 iterations with the deterministic policy and
logs success rate and MPJPE (all bodies, legs, VR 3 points) to W&B under `eval/`. The success rate is
SONIC's, but it is scored against the training terminations, which are stricter than SONIC's eval
set (`tracking/eval`). For reported numbers, use `eval_agent_trl.py` with
`+manager_env/terminations=tracking/eval`.

## 5. Demo protocol

Sources: SONIC docs and 2511.07820 §S1.2, SPOT 2609.07933, Teleopit 2608.01834, TWIST2 2511.02832,
CLONE 2506.08931, BruteForce `HARDWARE.md`.

**Calibration and engagement**
- Calibrate in the reference pose: upright, arms down.
- Align operator and robot before entering VR_3PT, and recalibrate the wrists on each entry.
- Ramp targets in over ~1 s.

**Network and input dropouts**
- Keep latency under 10 ms (under 30 ms at worst) with a dedicated access point or a Link cable.
- If controllers drop out, hold the last value. After a timeout, fall back to frozen upper body + idle.

**Startup and safety**
- Interpolate to standing over 3 s at startup.
- A joint-velocity watchdog switches to damping mode.
- Two e-stops, a 3 m clear zone, and a gantry for the first sessions.

**Robot interface.** On the real R1, command all joints with `rt/lowcmd` in debug mode.
`rt/arm_sdk` puts the robot in a state where it will not walk (xr_teleoperate #319).

**Training data from the Quest.** Record 50-100 Quest sessions and add them to training and
evaluation. TWIST2 found 73 in-device clips "essential" to bridge the gap.

## 6. Hardware facts to measure

- Dex3 mount offset on the R1 wrist: 0.080 m here vs 0.0415 m in BruteForce; both are estimates.
- Joint friction per group, in the units Isaac Sim 5.1 uses.
