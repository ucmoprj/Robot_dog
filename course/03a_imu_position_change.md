# Stage 03a — When the IMU Moves: What to Change and Why

**Situation (2026-10-14):** the original design put the IMU on a small bridge **above** the body
(26 mm above the body origin). On the real robot the BNO085 was mounted on the **inside floor of the
body, at the middle**. This page explains which files depend on the IMU position, how each one has to
change, and how to check the result. It is written so the same procedure can be repeated whenever the
IMU moves again.

---

## 1. Why the position matters at all

The policy sees 30 numbers every 20 ms (see `train/env.py`, `_get_obs`):

| Input | Count | Depends on IMU position? |
|---|---|---|
| gyro (rad/s) | 3 | **No.** Every point of a rigid body rotates with the same angular velocity. |
| accel (m/s², gravity included) | 3 | **Yes**, while the body rotates (see below). |
| last 2 commands | 24 | No. |

An accelerometer at a point P, a distance **r** from the body origin O, measures

```
a_P = a_O + α × r + ω × (ω × r)     (+ gravity term, the same everywhere)
```

- `α × r` — tangential part, from angular acceleration α (rad/s²)
- `ω × (ω × r)` — centripetal part, from angular velocity ω (rad/s)

When the robot stands still, ω = α = 0, so the position does not matter and every point reads
(0, 0, +9.81). When the body rocks, the two extra terms grow with |r|.

**Example for this robot:** the IMU moved from z = +26 mm to z = −17 mm, a change of 43 mm.

| Motion | Extra accel from moving the IMU 43 mm |
|---|---|
| body rocking at ω = 3 rad/s | ω²·r ≈ 9 × 0.043 ≈ **0.4 m/s²** |
| quick pitch jerk, α = 50 rad/s² | α·r ≈ 50 × 0.043 ≈ **2.2 m/s²** |

The second number is comparable to the tilt signal the policy uses to balance (sin 10° × 9.81 ≈ 1.7
m/s²). A policy trained with the IMU in the wrong place gets the wrong accel exactly when it needs
the signal most, during fast corrections.

**Orientation is a separate question.** Which way the board points (rotated, upside down) is handled
on the hardware side by `hardware_imu.json` (see section 4.6). The simulation only needs to know
**where** the sensor is, as long as the board is flat or exactly on its side.

---

## 2. Coordinate frame used everywhere

- **Body origin:** the roll-axis height, at the center of the body, front-to-back and left-to-right.
  It is 119.6 mm above the ground in the stand pose (`target_height: 0.1196` in
  `train/tasks/stand.yaml`).
- **Axes (body frame):** x = toward the HEAD, y = robot's LEFT, z = UP. These are the same axes as the
  sim `imu` site and as `hardware_imu.json` after the axis wizard.
- **Units:** MuJoCo and URDF files use **meters**. CAD files in `onshape/` use **millimeters**.

---

## 3. Finding the new position

### 3.1 From the CAD (what was done here)

The body plate is `onshape/print_stl/body_base.stl` (mm, body frame; `body_base_v3.stl` is the same
at the center). A vertical ray through the body center (x = 0, y = 0) hits two faces:

| Surface | z (mm) |
|---|---|
| top of the body floor (inside the body) | −19.0 |
| underside of the body floor (outside) | −21.0 |

The sensing element is the BNO085 chip, not the module's PCB, so add the stack height:

```
chip center z = floor top + PCB thickness + chip height / 2
              = −19.0     + 1.6           + 0.55
              ≈ −16.9 mm   →  used: −17 mm  =  −0.017 m
```

If the module stands on spacers, add their height. If double-sided tape is used, add its thickness
(often about 1 mm; small enough to ignore).

How the faces were found: read the STL triangles, keep those whose (x, y) projection contains
(0, 0), and solve each triangle's plane for z. Any STL tool with a "section" or "measure" feature
gives the same numbers.

### 3.2 By measuring the real robot

With the robot in the stand pose on flat ground:

```
z (mm) = (height of the chip center above the ground) − 119.6
x (mm) = distance from the body center toward the head (behind = negative)
y (mm) = distance from the body center toward the robot's left (right = negative)
```

A few millimeters of error are harmless: for 3 mm, the extra accel is under 0.2 m/s² even in the
fast-jerk case above.

---

## 4. Files that depend on the IMU position

Summary first, then details.

| # | File | Needed for MuJoCo training? | What changes |
|---|---|---|---|
| 4.1 | `dog_mujoco/robot*.xml` → new `robot_imu.xml`, `robot_gait_imu.xml` | **Yes** | `<site name="imu" pos=...>` |
| 4.2 | same files | Visual only | `IMU_envelope` geom `pos` |
| 4.3 | `dog_mujoco/scene*.xml` → new `scene_imu.xml`, `scene_gait_imu.xml` | **Yes** | `<include file=...>` |
| 4.4 | `train/vec_env_imu.py`, `train/train_imu.py`, `train/play_imu.py` (new) | **Yes** | which scene file is loaded |
| 4.5 | checkpoints in `train/logs/` | **Yes** | retrain; old ones saw the old IMU |
| 4.6 | `hardware_imu.json` | Only if the orientation changed | re-run the axis wizard |
| 4.7 | `robot.xml` `<inertial>` | Only if the mass moved a lot | body mass / center of mass |
| 4.8 | `dog_urdf/robot.urdf` | No (Isaac Sim only) | `imu_fixed` joint `origin` |
| 4.9 | `isaaclab_dog/` (USD, `robot_cfg.py`) | No (Isaac Lab only) | regenerate from the URDF |
| — | `train/env.py`, `vec_env.py`, `reward_terms*.py`, `tasks/*.yaml`, `ppo_config.py`, `hardware/imu_link.py` | **No change** | — |

**Course rule:** a new stage adds new files and does not edit earlier stages' files. So instead of
editing `robot.xml`, copies with an `_imu` suffix were made. The old training and checkpoints still
work exactly as before and can be compared side by side.

### 4.1 The sensor site (the one line that really matters)

`dog_mujoco/robot.xml`, line 64, inside `<body name="body">`:

```xml
<!-- original: on the bridge above the body -->
<site name="imu" pos="0 0 0.026" size="0.002" rgba="1 0.6 0 1" />

<!-- robot_imu.xml / robot_gait_imu.xml: inside floor, body center -->
<site name="imu" pos="0 0 -0.017" size="0.002" rgba="1 0.6 0 1" />
```

Why only this line: the sensors at the end of the file are attached to the site by name:

```xml
<gyro          name="imu_gyro"  site="imu" />
<accelerometer name="imu_accel" site="imu" />
<framequat     name="ground_truth_orientation" objtype="site" objname="imu" />
```

Moving the site moves all three sensors. The training code looks the sensors up by name
(`_sensor_slice(model, "imu_gyro")` in `train/vec_env.py`), so it needs no change.

- `pos` is `"x y z"` in meters, in the body frame (section 2).
- If the board is ever mounted **tilted** (not flat and not exactly on its side), also give the site
  a `quat` or `euler` so its axes match the board. A flat board needs nothing: the wizard (4.6)
  handles 90°/180° turns.
- The gait model `robot_gait.xml` is a copy of `robot.xml` with foot-contact sensors added, so it has
  the same line and needs the same change.

### 4.2 The visual IMU box (optional)

```xml
<geom name="IMU_envelope" type="mesh" mesh="IMU_envelope" pos="0 0 -0.043" ... mass="0" />
```

The mesh `assets/IMU_envelope.stl` was exported at its old place (z = 24.5…27.5 mm, center 26 mm).
A geom `pos` shifts it: −0.043 m = −17 mm − 26 mm, so the green box shows up where the chip now is.
It has `mass="0"` and `contype="0" conaffinity="0"` (no mass, no collisions), so it does not change
the physics. The old `imu_bridge` geom on top was left as it is; it is also visual only.

### 4.3 The scene files

`scene.xml` only adds the floor and lights and includes the robot:

```xml
<include file="robot.xml" />        <!-- scene.xml -->
<include file="robot_imu.xml" />    <!-- scene_imu.xml -->
<include file="robot_gait_imu.xml" /> <!-- scene_gait_imu.xml -->
```

The code always loads a **scene** file, so each new robot file needs its own scene file.

The two new robot files and two new scene files were made with `sed` from the originals, so the
diffs are exactly the lines above (plus the model name). To check:

```bash
diff dog_mujoco/robot.xml dog_mujoco/robot_imu.xml
diff dog_mujoco/scene.xml dog_mujoco/scene_imu.xml
```

### 4.4 Training and playback code

The original env classes already accept an `xml_path=` argument, but `train.py` and `play.py` always
use the default (`scene.xml`). New files point them at the IMU scenes:

| New file | What it does |
|---|---|
| `train/vec_env_imu.py` | `ROBOT_XML_IMU = scene_imu.xml`, `ROBOT_XML_GAIT_IMU = scene_gait_imu.xml`; classes `VecQuadrupedEnvImu` and `VecQuadrupedEnvGaitImu` subclass the existing envs and pass these paths. They also add IMU noise (section 5). |
| `train/train_imu.py` | same as `train.py`; `--gait` switches to the gait-reward version (like `train_gait.py`) |
| `train/play_imu.py` | same as `play.py`, but builds `QuadrupedEnv(cfg, xml_path=ROBOT_XML_IMU)` |

If the IMU moves **again**, only the XML files from 4.1/4.3 change. The Python files keep pointing at
`scene_imu.xml` / `scene_gait_imu.xml`.

### 4.5 Checkpoints must be retrained

A trained network learned "this pattern of accel values means I am tipping forward". With the IMU 43
mm lower, the pattern during fast motion is different, so old checkpoints (`train/logs/live_demo`,
`walk_demo`, `walk_gait_demo`) were trained for a sensor the real robot does not have. Train new
runs with `train_imu.py` and give them new names (`stand_imu`, `walk_gait_imu`). The observation size
did not change (30), so `ppo_config.py` stays the same and the new checkpoints load the same way.

### 4.6 `hardware_imu.json` — orientation, not position

```json
"raw_to_body": [[1,0,0],[0,1,0],[0,0,1]],
"gyro_bias_rad_s": [0.00057, -0.002523, -0.00297]
```

This file says how the board's raw axes map to the body axes, plus the gyro bias. It is written by
the axis wizard (`python hardware/imu_test.py --port COM4`, key `a`) and applied on the real robot by
`hardware/imu_link.to_body()`. It **contains no position**, so moving the IMU does not change it,
**unless the board was also rotated** while being moved. After any remount:

1. run the live view (`l`): level and still must read accel ≈ (0, 0, +9.81)
2. if not, run the wizard again (`a`) and save (`y`)

Here the matrix is the identity: the chip faces up and its x axis points to the head, the same as
the sim axes.

### 4.7 Body mass and center of mass (usually skip)

`robot.xml` line 65 has the body's total `<inertial>` (mass 0.161 kg, center of mass, inertia),
computed from the CAD with the IMU allowance (`imu_mass_kg: 0.002` in `parameters.json`) at the old
place. Moving 2–3 g by 43 mm shifts the center of mass of a ~0.31 kg robot by well under 1 mm. That
is far below the other model errors, so it was left alone.

Update it only when something heavy moves (battery, boards), or when the body plate itself changed
mass (e.g. body_base v2/v3). Best source: weigh the real robot and correct the mass, keeping the
inertia proportional.

### 4.8 URDF (Isaac Sim only)

`dog_urdf/robot.urdf`, end of file:

```xml
<link name="imu_link" />
<joint name="imu_fixed" type="fixed">
  <parent link="body" />
  <child link="imu_link" />
  <origin xyz="0 0 0.026" rpy="0 0 0" />   <!-- change to "0 0 -0.017" -->
</joint>
```

MuJoCo training never reads this file. Change it (as a new copy, e.g. `robot_imu.urdf`) only when
training or testing in Isaac Sim. `rpy` plays the role of the MJCF site `quat` for a tilted board.

### 4.9 Isaac Lab (`isaaclab_dog/`)

The USD files in `isaaclab_dog/usd/` are converted from the URDF. After changing the URDF, convert
it again. Also check whether `isaaclab_dog/mg90s_dog/robot_cfg.py` reads the IMU from `imu_link` or
from a fixed offset. Again: not needed for MuJoCo.

---

## 5. Making the sim IMU behave like the real one

Moving the site makes the sim sensor read the right physics. The real BNO085 also has errors the
sim sensor does not have. Without them, a policy can learn to rely on perfectly clean signals and
then shake or fall on the real robot. `vec_env_imu.py` adds three effects to the **observation
only** (rewards and fall checks still use the clean sim state):

| Effect | Default | Why this value |
|---|---|---|
| white noise, gyro | 0.005 rad/s (std) | typical BNO085 calibrated gyro noise |
| white noise, accel | 0.05 m/s² (std) | typical BNO085 accelerometer noise |
| bias per episode, gyro | uniform ±0.01 rad/s | measured bias was ≈0.003 rad/s; `to_body()` removes it, but it drifts with temperature, so a few times that value |
| bias per episode, accel | uniform ±0.1 m/s² | uncorrected accel offset and small mounting tilt |
| latency per episode | 0 or 1 step (0 or 20 ms) | BNO085 report age (1–10 ms) + serial link + PC loop |

Override them per task with an `imu_noise:` block in the task yaml:

```yaml
imu_noise:
  gyro_noise_std: 0.008
  max_latency_steps: 2
```

Better numbers come from the real sensor: hold the robot still, log ~10 s with the live view
(`imu_test.py`, key `l`) or `imu_motion_test.py` (CSV in `imu_logs/`), and use the standard
deviation of each axis as the noise std.

---

## 6. Checklist for the next time the IMU moves

1. **Measure** the new chip position in the body frame (section 3). Write it down in mm.
2. **Edit the site** in `dog_mujoco/robot_imu.xml` **and** `dog_mujoco/robot_gait_imu.xml`:
   `pos="x y z"` in meters. Shift the `IMU_envelope` geom by (new − 0.026) in z, and by x, y if those
   changed.
3. **If the board was rotated:** re-run the axis wizard and check level = (0, 0, +9.81) (4.6). If it
   is tilted at an odd angle, also set the site `quat`.
4. **Check the model:**
   ```bash
   source .venv/bin/activate
   python -m mujoco.viewer --mjcf=dog_mujoco/scene_imu.xml   # orange dot = imu site
   ```
   Or numerically, after letting the robot settle:
   ```python
   import mujoco
   m = mujoco.MjModel.from_xml_path("dog_mujoco/scene_imu.xml"); d = mujoco.MjData(m)
   for _ in range(1500): mujoco.mj_step(m, d)
   sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
   a = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")]
   print(m.site_pos[sid], d.sensordata[a:a + 3])   # expect [x y z], ≈ (0, 0, +9.81)
   ```
   Note: right after loading (t = 0) the accel reads 0, because the robot is in free fall until the
   feet touch the floor. Step it first.
5. **Retrain** with new run names:
   ```bash
   python train/train_imu.py --task train/tasks/stand.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name stand_imu
   python train/train_imu.py --gait --task train/tasks/walk_forward_gait.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name walk_gait_imu
   ```
6. **Play back** in the sim:
   ```bash
   python train/play_imu.py --task train/tasks/stand.yaml --checkpoint train/logs/stand_imu/model_<N>.pt
   ```
7. **Optional:** update the URDF (4.8) and Isaac Lab assets (4.9) if those simulators are used.
8. **Record** the new position, the date, and the reason in this page's history table below.

---

## 7. Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| mm entered in the XML (`pos="0 0 -17"`) | IMU 17 m below the robot; huge accel while rotating; training fails | MJCF/URDF use meters: `-0.017` |
| measured from the ground, not the body origin | z off by ~0.12 m | subtract 119.6 mm (section 3.2) |
| edited `robot_imu.xml` but not `robot_gait_imu.xml` | gait runs still use the old IMU | change both |
| edited only the URDF | MuJoCo training unchanged | MuJoCo reads only the MJCF (`.xml`) |
| moved the board and kept the old `hardware_imu.json` | real accel signs flipped; policy falls immediately | re-run the axis wizard |
| played a new checkpoint with `play.py` | runs on `scene.xml` (old IMU) | use `play_imu.py` |
| used an old checkpoint on the real robot | policy trained for the old sensor position | retrain with `train_imu.py` |

---

## 8. History

| Date | IMU chip position (body frame) | Mounting | Model files |
|---|---|---|---|
| design R1 | (0, 0, +26 mm) | bridge above the body | `robot.xml`, `robot_gait.xml` |
| 2026-10-14 | (0, 0, −17 mm) | inside body floor, center, chip up, x to head | `robot_imu.xml`, `robot_gait_imu.xml` |
