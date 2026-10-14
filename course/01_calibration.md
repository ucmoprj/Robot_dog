# Stage 01 — Servo calibration

**Goal:** for each of the 12 servos, find how a joint angle in the simulation (radians) maps to a
servo pulse (µs), so that a pose computed in MuJoCo looks the same on the real robot.

**Dates:** 2026-10-02 to 2026-10-05

## Files

| File | Role |
|---|---|
| `hardware/esp32_servo_bridge/esp32_servo_bridge.ino` | ESP32 firmware: serial commands → PCA9685 pulses (`P`, `M`, `X`, `?`) |
| `hardware/robot_io.py` | joint specs, calibration load/save, angle → pulse, serial link |
| `hardware/calibrate.py` | interactive per-joint calibration (keyboard) |
| `hardware_calibration.json` | the result |

## The model

```
us = center_us + direction * us_per_rad * (q - zero_rad)      clamped to [min_us, max_us]
```

| Field | Meaning | How it is measured |
|---|---|---|
| `zero_rad` | the sim's standing angle for this joint | from `onshape/onshape_mates.csv` (roll 0°, hip −19.4°, knee 42°) |
| `center_us` | pulse that puts the joint at the standing angle | **pose 1**: phone reads the standing angle → key `1` |
| `direction`, `us_per_rad` | which way and how fast the pulse turns the joint | **pose 2**: turn exactly 15° (roll) or 30° (hip, knee) → key `2` |
| `min_us`, `max_us` | physical limits | both ends, a few degrees before anything touches → keys `3`, `4` |

## Procedure (per joint)

```
python hardware/calibrate.py --port COM4 --joint <name>
```

1. `a`/`d` (±10 µs), `z`/`c` (±2 µs) until the phone reads the pose-1 angle → `1`
2. Turn by the reference angle in the direction shown → `2`. Check: the pulse moved 146–194 µs (15°)
   or 293–388 µs (30°).
3. One end → `3`, other end → `4`. Check: `sim range ... fits inside the physical limits ✔`
4. `s` save, `t` sweep test, `n` next joint.

How to measure (phone inclinometer):
- roll (`_1`): stand in front, phone upright against the outer face of the shoulder bracket, 0° = vertical
- hip (`_2`): stand at the robot's side, phone along the thigh (hip screw → knee screw), 0° = vertical
- knee (`_3`): phone along the shank (knee screw → foot). The thigh must be holding its standing pose,
  so start from `--joint <leg>_2` and press `n`.

Standing pose shape (all four legs, front and back alike): the thigh's knee end points toward the
**HEAD**, the shank's foot end points toward the **TAIL** — the knee sticks out toward the head.

## Final result (2026-10-05)

| Joint | center µs | dir | µs/rad | limits µs | sim range needs µs | |
|---|---|---|---|---|---|---|
| fl_1 | 1260 | +1 | 649.4 | 1170–1770 | 1192–1464 | ✔ |
| fl_2 | 1220 | +1 | 668.5 | 550–2030 | 886–1680 | ✔ |
| fl_3 | 1500 | +1 | 706.6 | 760–2200 | 1044–2030 | ✔ |
| fr_1 | 1550 | +1 | 687.5 | 1090–1640 | 1334–1622 | ✔ |
| fr_2 | 1910 | −1 | 725.7 | 890–2350 | 1411–2272 | ✔ |
| fr_3 | 970 | −1 | 611.2 | 540–2000 | 511–1365 | ⚠ bends to 82.3° (sim 85°), accepted |
| bl_1 | 1390 | −1 | 611.2 | 820–1490 | 1198–1454 | ✔ |
| bl_2 | 1150 | +1 | 687.5 | 550–2180 | 807–1623 | ✔ |
| bl_3 | 1400 | +1 | 706.6 | 690–2050 | 944–1930 | ✔ (horn re-seated) |
| br_1 | 1510 | −1 | 687.5 | 1410–2070 | 1438–1726 | ✔ |
| br_2 | 1540 | −1 | 687.5 | 580–2220 | 1067–1883 | ✔ |
| br_3 | 1360 | −1 | 706.6 | 770–2480 | 830–1816 | ✔ (bends to ~90°) |

These MG90S servos measured **611–726 µs/rad** (about 11–12.7 µs per degree).

## Problems hit and how they were solved

| # | Joint | What happened | Cause | Fix |
|---|---|---|---|---|
| 1 | fl_3 | center 1960 µs, knee could only bend ~65° (sim needs 85°) | horn put on the spline at a bad angle, so the bent end ran into the servo's 2500 µs end | re-seated the horn; center became ~1500 |
| 2 | knees | phone targets for the knee did not match | the thigh actually reads ~21° (not 19.4°) when it holds its standing pulse | knee targets = **42 − X** (pose 1) and **72 − X** (pose 2), X = measured thigh angle |
| 3 | fl_1 | its saved values were wiped (min = max = 1260, dir/µs/rad empty) | key `1` was pressed on a finished joint (this restarts the joint) and then saved | restored the known values (center unchanged). **Never press `1` on a joint marked `✔ done`** |
| 4 | fr_2 | 840 µs/rad (440 µs for "30°") | turned ~38° instead of 30° | redone: 725.7 µs/rad. Use only `z`/`c` near the target |
| 5 | fr_3 | pose 1 at 1500 had the shank 22° toward the **HEAD** | read the number but not the direction | moved past vertical to 22.6° toward the TAIL: center 970. Bend limited to 82.3° — accepted |
| 6 | bl_2 | 382 µs/rad (200 µs for "30°") | stopped at about 17° | redone: 687.5 |
| 7 | bl_3 | center 2150, bent end stopped at 2500 µs → only ~72° | same as #1 | re-seated the horn ~35° (2 teeth) more bent while holding 2150 µs; new center 1400, bends to ~95° |
| 8 | br_1 | 802 µs/rad (210 µs for "15°") | slightly over-turned | redone: 687.5 |
| 9 | br_2 | 878.5 µs/rad and dir **+1** (fr_2 is −1) | over-turned AND turned toward the head instead of the tail | redone: −1, 687.5 (now mirrors fr_2) |
| 10 | br_3 | measured while br_2's center was 1590; br_2 later became 1540 (~4°) | the knee's reference depends on the thigh pose | redone after br_2 |

## Lessons

- Always check **direction** (HEAD/TAIL, INWARD/OUTWARD) as well as the number on the phone.
- The pose-2 pulse-change check catches most mistakes: if it is outside the band, the angle is wrong.
- If the standing pulse is far from 1500 (below ~1000 or above ~2000), the joint will likely run out
  of servo range: re-seat the horn **before** measuring the rest.
- Redo a knee whenever its thigh is redone.
- There is no automatic backup: copy `hardware_calibration.json` before risky changes.
