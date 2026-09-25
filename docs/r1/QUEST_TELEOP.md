# Quest teleop runtime: operator guide and runbook

A Meta Quest 3 with controllers drives the R1 + Dex3 through the teleop-only SONIC policy:
- the headset and both hands drive the VR 3-point input;
- the thumbsticks drive SONIC's kinematic planner (walking);
- fingers are handled separately.

The code is `scripts/r1/teleop/`. Design and gates are in `PLAN.md` (Quest demo track, D16). Background is in `docs/r1/DEMO_READINESS.md`.

## 1. Environments

| env | used for |
|---|---|
| `sonic-train` | training; `scripts/r1/export_teleop_onnx.py` (CPU, no Isaac); `scripts/r1/export_r1_reference.py` |
| `r1rt` | the runtime: MuJoCo, onnxruntime-gpu, DDS, televuer (`scripts/r1/teleop/requirements.txt`) |

```bash
mamba create -y -n r1rt python=3.10
~/miniforge3/envs/r1rt/bin/pip install -r scripts/r1/teleop/requirements.txt
~/miniforge3/envs/r1rt/bin/pip install --no-deps -e external_dependencies/unitree_sdk2_python
```

On `asblab`:
- The policy and planner run on the TITAN V (`cuda:0` for onnxruntime).
- The planner takes ~18 ms per call there and 60–80 ms on CPU. The CPU is too slow: the training references assume a plan arrives within 40 ms.
- The TITAN V needs cuDNN 9.5 (pinned in the requirements).

## 2. Export a checkpoint

```bash
R=logs_rl/TRL_R1_Track/manager/universal_token/all_modes/<run>
CUDA_VISIBLE_DEVICES= env -u PYTHONPATH ~/miniforge3/envs/sonic-train/bin/python \
    scripts/r1/export_teleop_onnx.py --run $R --checkpoint model_step_00XXXX.pt   # -> $R/exported/
M=$R/exported/<run>_model_step_00XXXX
env -u PYTHONPATH ~/miniforge3/envs/r1rt/bin/python scripts/r1/teleop/policy.py --check $M.onnx --device cuda:0
```

The second command is gate G0: ONNX vs the training forward pass must agree within 1e-5.

## 3. Checks in simulation, in order

```bash
PY="env -u PYTHONPATH $HOME/miniforge3/envs/r1rt/bin/python"
# G1: track 100 training clips in MuJoCo (nominal, then the measured R1 actuators of #51)
$PY scripts/r1/teleop/play_clips.py --onnx $M.onnx --clips data/r1_reference/eval100 --profile issue51 --workers 12 --threads 1
# the full runtime with a scripted operator (walk, stop, turn in place, sidestep, back, waving)
$PY scripts/r1/teleop/run.py --onnx $M.onnx --input script:walk_reach --seconds 60 --realtime \
    --planner-device cuda:0 --policy-device cuda:0 --viewer
# the real-robot code path: DDS + R1 motor slots + 500 Hz writer, against MuJoCo
$PY scripts/r1/teleop/dds_sim.py --onnx-meta $M.json --gantry --release-after 9 &
$PY scripts/r1/teleop/run.py --onnx $M.onnx --robot unitree --interface lo --domain 1 --auto-start \
    --shadow-s 1 --input script:walk_reach --seconds 45 --planner-device cuda:0 --policy-device cuda:0
```

In the viewer:
- red spheres are the VR targets (palms, head);
- blue spheres are the robot's palms.

**Without a Quest: drive it from the keyboard.** `--input keys` replaces the headset with a
virtual operator whose sticks and hands follow keys typed in the terminal (click the terminal,
watch the viewer):

```bash
$PY scripts/r1/teleop/run.py --onnx $M.onnx --input keys --viewer --seconds 3600 \
    --planner-device cuda:0 --policy-device cuda:0
```

| Keys | Action |
|---|---|
| w / s | faster forward / backward (4 presses to full speed) |
| a / d | sidestep left / right |
| q / e | turn left / right |
| space | stop (the robot stands; it keeps its facing) |
| 1 2 3 4 5 | hands: rest, wave (right), point (right), both hands up, reach forward |
| g / t / o | fingers: point / fist / open (only with `--hands`; the MuJoCo fingers are fixed) |

**Calm motion (defaults, D18).** Every run applies these limits; `0` turns one off:

| Flag | Default | What it does |
|---|---|---|
| `--max-speed` | 0.5 m/s | top walking speed at full stick (SONIC's gamepad: 0.8) |
| `--max-yaw-rate` | 0.6 rad/s | turning rate at full stick (gamepad: 1.0) |
| `--smooth-tau` | 0.04 s | critically damped filter on the arm/waist targets (joint space) |
| `--max-arm-speed` | 3 rad/s | speed limit on each arm target joint |
| `--max-waist-speed` | 1 rad/s | speed limit on the waist targets |
| `--no-planner-hold` | off | by default the legs hold the stand-up pose until the sticks first move |

Without the hold, the planner's first idle plan moves the feet up to 10 cm over 2 s, lifting one
foot 3 cm, before the operator has done anything.

`sim_gate.py` also reports balance steps: `unplanned_steps` counts robot foot lifts of 0.1 s or
more while the reference foot has stood for 0.3 s. `idle_foot_drift_mm` is how far a foot moves
while the reference stands completely still.

## 4. The Quest

1. Certificate, created once. It is already in `~/.config/xr_teleoperate/` on `asblab`:
   `openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem -subj /CN=r1-teleop`
2. Connect the headset, either way:
   - **Wi-Fi.** Quest and PC on the same network; allow TCP port 8012 in the firewall. Open `https://<pc-ip>:8012/?ws=wss://<pc-ip>:8012` in the Quest browser; `asblab` is `192.168.13.79`. Accept the certificate warning once.
   - **USB** (no Wi-Fi lag). Enable developer mode on the headset, then `adb reverse tcp:8012 tcp:8012` and open `https://localhost:8012/?ws=wss://localhost:8012`.
3. Start the runtime with `--input quest`, then press "Enter VR" (pass-through: you see the room and the robot).
4. **Controls:**
   - Stand in the calibration pose and press **A + X**. Targets ramp in over 1 s.
   - **B + Y** stops. On the robot this is damping.

   | Thumbstick | Action |
   |---|---|
   | Left | walk in the pushed direction, relative to the robot's facing; speed grows with deflection (0.2–0.5 m/s, `--max-speed`) |
   | Right, sideways | turn the robot (up to 0.6 rad/s, `--max-yaw-rate`) |
   | Released | the robot stands |

   The left stick is a velocity command: SONIC's kinematic planner turns it into leg motion.

   **Fingers: not used for now** (decision 2026-09-25). The Dex3 hands are attached but not
   commanded, and nothing is sent on `rt/dex3/*`. `--hands` turns on finger control:

   | Fingers (Dex3, each hand, `--hands` only) | Pose |
   |---|---|
   | nothing pressed | semi-closed (the hold pose the policy was trained with) |
   | grip | point: index out, middle finger and thumb closed |
   | grip + trigger | fist |
   | trigger | pinch |
   | A (right) / X (left), after engaging | open flat hand |

   Poses blend with the analog trigger and grip, filtered over 0.1 s (`hands.py`).

5. **Calibration pose** (the R1's default pose): stand upright, upper arms along the body, forearms angled forward-down about 50°, controllers level. Face the same direction the whole session and turn the robot with the right stick. The hand frame is fixed at calibration, so looking around does not move the hands.
6. **Head:** your head yaw and roll (clamped to ±0.6 / ±0.25 rad) turn and tilt the robot's torso through the waist. The R1 has no waist pitch.
7. **Hands:**
   - Hand displacements are scaled by 0.65, the arm-length ratio.
   - Orientations follow the controllers' rotation since calibration, but only as far as the R1's 5-DOF arms can reach: an IK projects every target onto the robot's workspace.

## 5. Recording and replay

- `--record logs_rl/sessions/<name>.npz` saves, for every tick:
  - the Quest poses, sticks and buttons, plus the calibration;
  - the planner commands and seeds;
  - the observation, action, joint targets and robot state.
- `--input replay:<file>.npz --sync-planner` replays a session in MuJoCo, bit-identically on CPU.
- Record 8–10 sessions of 1–2 min before the talk, each starting in the calibration pose:
  - reach grid;
  - gestures;
  - stick walking (forward, stop, turn 90° in place, back, sidestep);
  - walking while waving;
  - the demo script.

## 6. The real robot (gantry at all times; two people)

1. **Network.** Ethernet to the robot's 192.168.123.x network; find the interface name with `ip -4 addr`.
2. **Joint map, read-only.** Put the robot in zero torque and run `$PY scripts/r1/teleop/robot_unitree.py --check <iface>`.
   - Move each joint by hand. The right name must change, with the sign and zero of the MJCF.
   - The slot map is Unitree's R1 `JointIndex`: legs 0–11, waist roll 12, waist yaw 13, left arm 15–19, right arm 22–26, head 29/30.
   - Also yaw the waist while holding the pelvis: the IMU quaternion must not change, which confirms the IMU is in the pelvis.
3. **Debug mode:** from damping, press L2 + R2 on the remote. Never use `rt/arm_sdk`: it stops walking (#319).
4. **Run:**

   ```bash
   $PY scripts/r1/teleop/run.py --onnx $M.onnx --robot unitree --interface <iface> --domain 0 --input quest \
       --shadow-s 5 --planner-device cuda:0 --policy-device cuda:0 --record logs_rl/sessions/robot_<n>.npz
   ```

   Phases:
   1. damping;
   2. **start** on the remote stands up to the default pose over 3 s;
   3. hold;
   4. **A + X** calibrates and engages;
   5. shadow for 5 s (actions computed, not sent: check they are small);
   6. blend over 2 s;
   7. run.
5. **Stop.** Each of these switches to damping:
   - remote **B** or **select**;
   - Quest **B + Y**;
   - Ctrl-C;
   - the watchdog: lowstate > 40 ms old, |dq| > 35 rad/s, tilt > 45°, NaN, motor ≥ 90 °C.

   A Quest stream stale for more than 0.5 s freezes the targets and releases the sticks.
6. **Order of the first sessions:**
   1. hanging, standing idle;
   2. feet loaded, rope slack;
   3. standing teleop;
   4. stepping;
   5. 1 m walks;
   6. off the rope with spotters.

   Each stage takes 3 × 1 min without a trip before the next.

**Dex3 hands: attached, not commanded** (default; nothing is published on `rt/dex3/*`).
- With `--hands`, the DDS process publishes `rt/dex3/{left,right}/cmd` at 100 Hz from stand-up on:
  - the hold pose, then the operator's finger poses once engaged;
  - xr_teleoperate's parameters: kp 1.5, kd 0.2, RIS mode byte;
  - limp fingers in damping (kp 0).
- Before using `--hands`, VERIFY with the robot hanging:
  - each hand's motor order: left thumb 0–2, middle 0–1, index 0–1; right thumb 0–2, index 0–1, middle 0–1;
  - that closing is negative on the left and positive on the right.
- The policy's model has the fingers fixed semi-closed. Note what pose the uncommanded fingers
  actually sit in; the model can match it at the next training restart (`r1_spec.DEX3_HOLD_POSE`).

Not done yet:
- Driving the head motors from the headset. They are held at 0.
