# MG90S Robot Dog — Course Log

A record of bringing the simulated MG90S quadruped (MuJoCo, trained on a DGX Spark) onto the real
robot, one stage at a time. Each stage lists what was done, the real problems hit, the measured
numbers, and how they were solved.

**Rule for this course:** a new stage adds **new files** and does not edit the code of earlier
stages, so every stage can still be repeated exactly as it was.

| Stage | Document | New files in this stage | Status |
|---|---|---|---|
| 01 | [Servo calibration](01_calibration.md) | `hardware/calibrate.py`, `hardware/robot_io.py`, `hardware/esp32_servo_bridge/`, `hardware_calibration.json` | done 2026-10-05 |
| 02 | [Motion test without AI](02_motion_test.md) | `hardware/motion_test.py` | done 2026-10-05 (paw wave not yet tried on the floor) |
| 03 | [IMU (BNO085)](03_imu.md) | `hardware/esp32_imu_bridge/`, `hardware/imu_link.py`, `hardware/imu_test.py`, `hardware_imu.json` | in progress |
| 04 | Run the trained policy on the robot | (planned) | — |

## Hardware

- 12 × MG90S servos, PCA9685 PWM board, ESP32 (classic, ESP32-D0WD-V3) on USB (COM4)
- Servo power: separate 5 V, 5 A supply on the PCA9685 V+ terminal
- IMU (stage 03): GY-BNO085 9-axis module
- Calibration stand (`calibration_stand/`) and a phone inclinometer app for angles

## Joint names

`fl` / `fr` / `bl` / `br` = front-left / front-right / back-left / back-right (robot's own left and right).
`_1` = roll (hip sideways), `_2` = hip (thigh), `_3` = knee. PCA9685 channel = position in
`fl_1, fl_2, fl_3, fr_1, … br_3` (0–11).
