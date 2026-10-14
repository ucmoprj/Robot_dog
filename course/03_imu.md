# Stage 03 — IMU (BNO085)

**Goal:** give the real robot the same body-motion sensing the policy had in the simulation, in the
same axes and units.

**Started:** 2026-10-06

## Why the policy needs it

The trained policy (`train/env.py`, `_get_obs`) sees 30 numbers every 20 ms (50 Hz):

| Input | Count | Sim source | Real source |
|---|---|---|---|
| gyro, rad/s | 3 | `imu_gyro` at the `imu` site | BNO085 calibrated gyroscope |
| accel, m/s² (gravity included: level & still = (0, 0, +9.81)) | 3 | `imu_accel` | BNO085 accelerometer |
| its own last 2 commands | 24 | action history | remembered on the PC |

Its output is 12 joint angles in the same sim frame stage 01 calibrated, so the output side is already
done. Without the IMU, the policy would be walking blind.

Sim IMU axes (`imu` site in `dog_mujoco/robot.xml`, 26 mm above the body origin): **x = HEAD,
y = robot's LEFT, z = UP**.

## Files (all new; stage 01–02 files unchanged)

| File | Role |
|---|---|
| `hardware/esp32_imu_bridge/esp32_imu_bridge.ino` | firmware: same servo commands as stage 01 + `S` (IMU status) and `I` (read IMU) |
| `hardware/imu_link.py` | `ImuLink` = stage-01 `ServoLink` + IMU reading; raw → body axes |
| `hardware/imu_test.py` | status, live view, axis wizard (no servo moves) |
| `hardware_imu.json` | result: mounting (raw → body axes) and gyro bias |
| `hardware/imu_motion_test.py` | motion_test.py moves + IMU read every frame: summary per move, tilt-direction check for lean/sit/bow, tilt stop (±35°), CSV log in `imu_logs/`. Imports motion_test.py and imu_link.py unchanged |
| `hardware/imu_feedback_test.py` | closed loop: the same moves, leg depths corrected every frame so the body keeps the tilt the move intends (level for stand/trot, the designed lean for i k j l 3 4), also between commands. PI + gyro damping per axis, ±10° limit, keys f (on/off), 0 (level reference), n/o (gain). Builds on imu_motion_test.py's ImuRobot. Simulated with 80 ms servo lag: 6° floor → within 1° in ~2 s; KD 0.03 oscillated, 0.005 is stable up to gain ×2.7 |
| `docs/imu_viewer.html` | learning page: live 3D board, gyro bars, spirit level, charts (Web Serial in Chrome/Edge, or demo). 12 frozen snapshots + comparison table + Capture (`#snap=N` opens one). Raw sensor axes |

## 1. Wiring

Same I2C bus as the PCA9685 (PCA9685 = 0x40, BNO085 = 0x4A or 0x4B, no conflict).

| BNO085 pin | ESP32 |
|---|---|
| VCC / VIN | 3V3 |
| GND | GND |
| SDA | GPIO21 (shared with PCA9685 SDA) |
| SCL | GPIO22 (shared with PCA9685 SCL) |
| RST (optional) | GPIO4, then set `IMU_RST_PIN = 4` in the sketch |
| PS0, PS1 | must be LOW (I2C mode, board default) |

Mounting: on the `imu_bridge` at the middle of the body, board **flat** (or exactly on its side).
Which way it points does not matter — the axis wizard finds it.

## 2. Flash the firmware

1. Arduino IDE → Library Manager → install **Adafruit BNO08x** (and its dependencies).
2. Open `hardware/esp32_imu_bridge/esp32_imu_bridge.ino`, board = ESP32 Dev Module, port = COM4, Upload.
3. Serial Monitor (115200, newline): `?` → `READY`, `S` → `IMU OK 0x4A` (or 0x4B).
   Close the Serial Monitor before running Python.

The stage-01 tools (`calibrate.py`, `motion_test.py`) still work with this firmware: it keeps the same
servo commands.

## 3. Check and set up

```
python hardware/imu_test.py --port COM4
```

| Key | What | Expected |
|---|---|---|
| `s` | status | `IMU OK 0x4A` |
| `l` | live view (raw axes until the wizard has run) | still: gyro ≈ 0 deg/s; one axis reads ≈ ±9.8 m/s², \|a\| ≈ 9.8 |
| `a` | axis wizard: 1) level & still 2) nose up ~45° 3) turn left ~90° → `y` saves | `✔ left turn = +z, as in the sim` |
| `l` | live view again (body axes) | level: accel ≈ (0, 0, +9.81); nose up → `nose-up` positive; left side up → `left-side-up` positive; turn left → gyro z positive |

## Results

| Check | Result |
|---|---|
| IMU found (`S`) | `IMU OK 0x4B` (breadboard test, ESP32 + BNO085 only, 2026-10-12) |
| still: gyro noise / bias | _to fill in_ |
| raw → body mapping found by the wizard | _to fill in_ |
| live view after the wizard matches the sim signs | _to fill in_ |
| problems hit | First flash: gyro always 0.00000 and age_ms = time since boot, while accel (az ≈ -9.93, board upside down) and quaternion worked. Cause: the library's `getSensorEvent()` keeps only the last report of each I2C read, and the BNO085 packs same-rate reports together, so the gyro report was always overwritten. Fix: own `sh2_setSensorCallback` handler that stores every report. After the fix: sensor 1 gave gyro data (age 2-9 ms) but once age 3301 ms while shaken on the breadboard. Sensor 2 (swapped in, also 0x4B): 5 samples while turning, age 1-7 ms, gyro and accel follow the motion. 2026-10-13: `imu_test.py` printed `ERR unknown command` = the ESP32 still had the servo firmware. The IMU sketch did not compile: `Adafruit_PWMServoDriver.h: No such file` although the library is installed, because the library folder is `OneDrive\문서\Arduino\libraries` (Korean path). Fix: copy the libraries to an English-only folder and build with `arduino-cli compile --upload -p COM4 --fqbn esp32:esp32:esp32 --libraries <that folder> hardware\esp32_imu_bridge` → `S` = `IMU OK 0x4B`. Then the live view froze: same values, age_ms counting up (robot still). Bench poll: reports stopped after ~48 s at 400 kHz I2C and never came back until the ESP32 restarted. Fix in the sketch: I2C 100 kHz (`I2C_HZ`) + watchdog that restarts the IMU after 1 s without a gyro report + `R` command (restart count). After the fix: 150 s, max age 10 ms, 0 restarts. Later the same day, with the PCA9685 + servos connected: `IMU NONE` after every boot. Cause 1 (my watchdog bug): it fired before the first report (timer started at boot) and restarted with a second `begin_I2C()`, which always fails (the library has one link slot) → fixed: timer starts at `startImu()`, `sh2_close()` before a new begin, retry every 2 s. Then RST → GPIO4 (D4) and INT → GPIO27 (D27) were wired; firmware uses RST for a hardware reset on restart and reads only when INT is LOW (+20 ms fallback). Diagnostic sketches showed wiring OK (scan 0x40 + 0x4B, RST LOW silences 0x4B, INT LOW 96 ms after reset, product IDs read: 3.2.13) but **the sensor reports never start**: `enableReport` returns OK, `getSensorConfig(accel)` stays at interval 0, no samples, INT never LOW again. Shorter SDA/SCL and power wires and removing RST/INT did not help (data for 7-20 s, then none; once none at all), so it was not the wiring. **Fix: `sh2_devOn()` (wake from sleep)**: a diagnostic sent it before enableReport and the reports started at once (it then also cleared the saved calibration with `sh2_clearDcdAndReset`, which was not needed). The firmware now sends `sh2_devOn()` after every begin and every IMU reset. After the fix (RST/INT unplugged, PCA9685 on the bus): 240 s, 0 restarts, one 379 ms gap that recovered by itself. |

## Next (stage 04)

Export the trained actor network on the DGX Spark, then a new `hardware/run_policy.py`: every 20 ms read
the IMU (body axes), build the 30 inputs, run the network, send the 12 angles through
`rad_to_us()` — with a tilt cutoff and any-key stop. First on the stand, then on the floor held by hand.
Expect a sim-to-real gap: training had no domain randomization (servo strength/latency, IMU noise,
mass, friction), so a retrain with it may be needed.
