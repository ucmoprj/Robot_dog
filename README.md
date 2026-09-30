# MG90S Dog R1

A mechanical design for review that rebuilds the **4 legs × 3 joints** layout of Rhoban's `dog_mujoco` and `dog_urdf` around MG90S servos. The joint names and kinematic structure of both originals were checked, and new geometry, masses, inertias and joint limits were made for the MG90S. It is not a simple scaled-down copy of the original CAD.

Generated output is in `out/mg90s_dog`. This design is separate from the `leg_*.step` files and `sim` model of the earlier ESP32 quadruped.

## Design constraints

- 12 × MG90S: 4 body-side roll, 4 hip pitch, 4 knee pitch.
- Body plate 116 × 66 mm, shoulder vertical offset 37 mm, lateral offset 16 mm.
- Upper leg 44 mm between axes, 38 mm from the lower-leg axis to the foot center, foot radius 5 mm.
- Default knee angle 42°. The hip angle is computed from the two link lengths so the foot sits under the hip.
- Roll is limited to 18° outward and 6° inward. Tilting too far inward makes the leg hit the opposite knee servo, so left and right legs get limits of opposite sign.
- Estimated mass follows the generated `design_summary.json` and `bom.csv`: a conditional estimate of about 0.31 kg.
- Includes battery 30 g, PCA9685 12 g, controller board 10 g, IMU 2 g, wiring/regulator 10 g, fasteners 10 g. Extra payload 0 g.
- The effective print density of 900 kg/m³ is an assumption. Replace it with the slicer's filament usage and the measured weight.

From the manufacturer's MG90S spec: mass 13.4 g, overall size 22.8 × 12.2 × 28.5 mm, stall torque about 0.1765 N·m at 4.8 V, 60°/0.10 s. **The 0.06 N·m design limit is a conservative assumption, not a continuous torque guaranteed by the manufacturer.** Mounting ears, output shaft position and horn hole sizes are adjustable values that need checking against the real part.

Source: https://towerpro.com.tw/product/mg90s-3/

## Opening in Onshape

1. To check the shape, import `onshape/mg90s_dog_assembly.step`. It is saved in the standing pose.
2. To build a movable Assembly, **`onshape/mg90s_dog_zero.step`** is easier. The body origin is at (0,0,0) and every joint is at its geometric reference angle of 0°. This pose is the assembly reference, not a walking pose.
3. The STEP hierarchy has 13 rigid groups such as `body`, `fl_shoulder`, `fl_thigh`, `fl_shank`. Fix the parts of each group so they move together.
4. Create 12 Revolute Mates following the origins and axes in `onshape/onshape_mates.csv`. Align each Mate connector's Z axis with the rotation axis in the table. Name them `dof_fl_1` through `dof_br_3`.
5. Apply the rotation limits and standing angles from the CSV. `*_1` is roll, `*_2` hip pitch, `*_3` knee pitch.
6. The IMU frame is at (0,0,26) mm in the body frame, X forward, Y left, Z up. It can be added as `frame_imu`.

STEP carries shape, names and placement. **It does not create Onshape Mates or the original sketch/feature history.** Depending on how the hierarchy imports, group the parts of each rigid body with Group or Fastened Mates. The CAD model was generated locally and did not modify the user's Onshape documents directly.

If you build mates directly on the standing STEP, the current pose may become 0°. The CSV's absolute angles would then be wrong, so the zero STEP is recommended. When exporting again with `onshape-to-robot`, compare the joint axes, zeros and masses with the current MJCF. Do not let Onshape estimate the MG90S internal mass from a generic material density; apply the rigid-body masses and inertias from `design_summary.json` separately.

Source: https://onshape-to-robot.readthedocs.io/en/latest/design.html

## Printing and assembly

`onshape/parts` holds a STEP per printed part and `onshape/print_stl` holds STLs in mm. Part files keep their rigid-body coordinates, so place a flat face on the bed in the slicer. Some parts are mirrored for left/right and front/back mounting; keep the file names.

First check the servo case clearance, ear hole spacing and pilot holes with `servo_fit_coupon.step` or the STL of the same name. Use the horns that ship with the MG90S; the round horn in the model is a stand-in for mass and space. Do not assume a printed spline will mate with the servo shaft. If the real horn shape and the 6 mm-radius screw holes differ, change `horn_screw_radius` and related values.

Screw the servo ears to the frame and the horn to the rotating bracket. There is an access hole for the horn's center screw. After checking plate/frame and screw fit, assemble one leg and check its motion. The current design relies on the MG90S's own output shaft support; it is not a double-supported design with external bearings. Drop/impact durability and horn/pilot-hole pull-out strength need testing on the real part.

Electronics are held with slots and straps instead of a specific seller's hole pattern. The green parts are space reservations: PCA9685 62×25×5 mm, controller board 50×25×7 mm, IMU 20×16×3 mm. Also check real terminal and cable bend space. The IMU bridge is a draft fixed to the support in the center channel with adhesive or tape; a screw pattern for a specific module is not decided.

## Load and height

`load_screen` in `validation_report.json` computes the cases of weight shared equally by all four feet and by two diagonal feet. It uses each joint's Jacobian with masses and inertias summed from the CAD, multiplies the required torque by 1.5 and compares it with 0.06 N·m. The two-foot case is a load-sharing condition during walking, not a claim that the robot is statically stable on two feet.

Height and torque are recorded for knee angles of 25°, 42°, 60° and 75°. **Bending the legs more to lower the body can increase knee torque.** If you need a lower body, calculate shortening `upper_leg`/`lower_leg` and changing the bend angle as separate options. Shorter legs require re-checking servo/horn/frame interference.

The report's mass limit is an approximation that scales the load at the same pose; it is not a guaranteed payload. If the battery or part weights change, edit `parameters.json`, regenerate and re-validate. Stall torque, heat, gear backlash, voltage drop and impact contact forces are not validated by this load calculation alone.

## Simulation

From the project root:

```powershell
python design/mg90s_dog/build.py
python design/mg90s_dog/validate.py
python design/mg90s_dog/check_cad.py
python design/mg90s_dog/run.py --seconds 10
python design/mg90s_dog/run.py --viewer
```

- MuJoCo: `dog_mujoco/scene.xml`. Internal units m·kg·s, 1 ms step.
- URDF for Isaac Sim and similar: `dog_urdf/robot.urdf`. Keep it together with the `assets` folder. Import as a floating base and set the articulation root to body. It has not been run in Isaac Sim.
- URDF effort/velocity values alone do not reproduce the real actuator. In Isaac Sim, start with a position drive, max force 0.06 N·m, stiffness 1.8 N·m/rad and damping 0.035 N·m·s/rad, and apply the 50 Hz command rate and velocity limit in the controller. Check import options and drive units for your Isaac Sim version.

MuJoCo's direct position actuator only has a static torque limit and PD. Using `ServoBridge` in `control.py` adds a 50 Hz command rate, one-cycle command delay and a 4 rad/s command rate limit. This rate limit does not guarantee real shaft speed or torque-speed behavior. Initial PD, friction and joint armature are also assumptions until identified on hardware.

Tested package versions are recorded in `requirements-tested.txt`.

## Sim-to-real / real-to-sim

MG90S servos and a PCA9685 give no joint angle feedback. The default policy observation has 30 values: 3 gyro, 3 accelerometer, and 24 for the current and previous commands. Exact joint angles, joint velocities, foot contacts, world position and a perfect quaternion are not in the policy observation. The `ground_truth_orientation` sensor is for evaluation in simulation.

Measure and apply the real angle zeros, rotation signs, PWM pulse range and frequency, latency, speed, sag under load, and IMU bias and axis orientation. The proposed PCA9685 channels are FL 0/1/2, FR 3/4/5, BL 6/7/8, BR 9/10/11, which differ from the earlier ESP32 robot. Fill the nulls in `hardware_calibration.template.json` with measured values before use (see `hardware/calibrate.py` and `docs/servo_bench.html`). The model itself does not send I2C commands.

Still to do before transfer to hardware: identify mass, friction, torque, latency, IMU noise and zero error, and randomize them during training. A standing success in the validation files does not mean walking or sim-to-real transfer will succeed.

## Sources and limits

- Original: https://github.com/Rhoban/onshape-to-robot-examples — `source_audit.json` records the commit, Onshape URL, and comparison of original masses and joints. The original MIT license is kept in `LICENSE.Rhoban`.
- MuJoCo: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- Isaac Sim URDF: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/import_urdf.html

The current result is **a runnable simulation and CAD design R1 that can be reviewed in Onshape**. It is not a production-ready design validated for real MG90S fit, hardware mounting, heat, durability or print-orientation strength.
