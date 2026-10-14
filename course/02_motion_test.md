# Stage 02 — Motion test without AI

**Goal:** before running a trained policy, check with hand-made motions that the calibrated robot
moves the way the simulation expects: stand, crouch, poses, a short trot.

**Date:** 2026-10-05

## Files

| File | Role |
|---|---|
| `hardware/motion_test.py` | keyboard motion test (new in this stage; uses `robot_io.py` from stage 01) |

```
python hardware/motion_test.py --dry-run      # practice, no hardware
python hardware/motion_test.py --port COM4    # real robot
```

## How it works

- Every motion is built as **sim joint angles** (rad) and converted with `robot_io.rad_to_us()`, so the
  stage-01 calibration and its limits are applied automatically. The policy will use the same path.
- Poses are given as **foot positions** and turned into hip/knee angles with 2-link inverse kinematics
  (thigh 44 mm, shank 38 mm, from `dog_mujoco/robot.xml`). Standing = foot 76.6 mm below the hip.
- 50 updates per second, smooth interpolation between poses.
- First command after power-up moves the joints to standing **one at a time** (smaller current spike).
- Any key during a motion stops it and **holds** the pose (`r` relaxes all servos).

## Keys

| Key | Motion |
|---|---|
| `1` / `2` | stand / crouch (all feet up to 64 mm depth, knee ≈ 78°) |
| `3` / `4` | dog sit (rear low) / play bow (front low, reaching forward) |
| `i` `k` `j` `l` | lean forward / back / left / right (≈ 7–10°) |
| `b` | bounce ×3 |
| `u` | hop attempt |
| `w` / `e` | trot forward / in place |
| `[` `]` | steps −/+ (1–20, default 5) — also **while walking** |
| `+` `-` | slower / faster (0.3–0.8 s per step) — also while walking |
| `p` | right-front paw wave; `z`/`x` sideways and `c`/`v` backward weight shift |

## Trot

Diagonal legs move together: **fl + br** lift while **fr + bl** push, then the other way round. One
step = one diagonal pair lifts. Stride 16 mm per step, lift 10 mm, walking height 74 mm. The first and
last steps are half steps so the walk starts and ends in the standing pose.

## Why a real jump is not possible

| | |
|---|---|
| mass | ~0.31 kg (design estimate) — torque is not the limit (~1/8 of MG90S stall torque needed) |
| leg extension, crouch → straight | ~17 mm |
| servo speed | ~0.1 s per 60° unloaded → ~0.12 s to straighten the knee 70° |
| expected "jump" | 0–3 mm: a bounce, not a jump |

The hop (`u`) commands straight legs in one update (top servo speed), then bends back after 0.15 s to
cushion the landing.

## Paw wave without an IMU (estimated balance)

The estimated center of mass (from `design_summary.json`) is the middle of the body, which lies exactly
on the **fl–br diagonal**. Lifting fr straight away would tip the robot onto it. So the body is first
moved toward the rear-left, inside the triangle fl–bl–br:

| Shift | Default | Effect |
|---|---|---|
| roll joints slide all feet to the robot's right | 8 mm (~4°; left legs can roll inward only 6°) | body goes left |
| left legs 6 mm shorter | ~4° tilt | body goes further left |
| all feet 10 mm toward the head | | body goes back |
| **estimated** | CoM 16 mm left, 10 mm back of center | **~19 mm margin** to the fl–br edge |

Then: lift fr gently → knee 70° → swing the hip −45° ↔ −10° three times → put it down → stand. The
estimate ignores the battery/wires and servo sag, so the shift is adjustable live.

## Results on the real robot

| Test | Result |
|---|---|
| stand / crouch / trot on the stand and floor | works well (user report) |
| walking on the floor | **drifts to one side after a few steps** — expected for open loop (no feedback). Possible causes: small calibration differences between legs, servo differences under load, foot slip, CoM offset |
| poses, bounce, hop | _to fill in_ |
| paw wave: shift values that balanced | _to fill in (side __ mm, back __ mm)_ |

## Lessons

- Open loop = the robot does not know what its body is doing. Small errors add up (the drift).
- Balance-critical motions (paw wave) need an estimate of the CoM plus manual tuning — or an IMU.
- The trained policy's inputs are **IMU gyro + accel + its own last two commands**
  (`train/env.py`), so the next stage is the IMU.
