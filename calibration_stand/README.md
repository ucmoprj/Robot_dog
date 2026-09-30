# Small Quadruped Calibration Stand R1

A **design for review and test printing** for MG90S-DOG-R1. It is a center-support stand built around a P1S printer, PLA, and M3 screws and nuts. Other small robots of similar weight can use it by adjusting the support width or swapping only the supports. Real print stiffness, fastener holding force and interference with real cables have not been verified yet.

## Files

- `output/robot_on_stand.step`: the whole layout with the current robot on the stand. Open this in Onshape first.
- `output/stand_assembly.step`: assembly STEP of the stand only.
- `output/parts_step/`: one STEP per part, keeping assembly coordinates.
- `output/print_stl/`: STLs in mm, in print orientation. One file per part; the coupon is an optional test piece.
- `output/preview_robot.png`, `preview_stand.png`: previews rendered from the actual CAD geometry.
- `output/validation.json`: interference check and print size results.
- `build.py`, `parameters.json`: the editable, regenerable CadQuery source.

**This was not created directly as an Onshape document.** In a new document, use + → Import to bring in the STEP. Imported geometry has no original sketch/feature history or Mates. Add Mates or edit based on the part names and placement. To rebuild it with native features, use the dimension table below.

## Dimensions and adaptability

| Item | Value |
|---|---|
| Base plate | 180 × 140 × 6 mm, corner R8 |
| Fixed mast overall | 64 × 50 × 110 mm, mounted on the base plate |
| Mast vertical plate | 40 wide × 10 thick |
| Carriage vertical part | 28 wide × 10 thick × 80 long |
| Top platform | 44 × 90 × 6 mm |
| Body underside support height | 136 / 146 / **156** / 166 / 176 mm |
| Current body plate width | 66 mm, side clearance 0.3 mm each side |
| Body plate width range | 50–80 mm recommended; adjusted in the slots of the same parts |
| Front-back contact length of the supports | 40 mm, not adjustable |
| Cap inner contact tongue | 16 wide × 3 mm protrusion |
| Body base plate thickness the current cap targets | 2 mm |
| M3 clearance hole | Ø3.4 mm |
| Nut pocket | 5.7 mm across flats; check against real nuts and print tolerance |

The default `platform_height=150` is the top of the platform; the actual support surface is +6 mm. Set it from 130 to 170 mm in 10 mm steps. `body_width` can be set from 50 to 80 mm. After editing the source, regenerate with `python calibration_stand/build.py`.

"Adaptable" does not mean every robot's shape fits. The body needs an accessible 40 mm front-back support area in the middle and base plate edges on both sides. If the base plate is not 2 mm thick, or there are cables, a battery or other protrusions, modify the jaw/cap adapters. Front-back movement of the robot is held only by the cap's clamping friction. There is no separate stopper at the front or back.

The robot's BREP was inspected and the underside of the center lower plate at z=-21 mm is used as the support surface. The battery underside is at z=-16 mm, so the supports never touch the battery. The cap's center tongue avoids the body support posts near x=±15 mm.

## Parts and fasteners

There are **7** printed parts: base, mast, carriage, jaw_left/right, cap_left/right. The test piece fit_coupon is separate.

| Joint | Screw | Nut | Qty | Notes |
|---|---|---|---|---|
| Base plate ↔ mast | M3 × 12 mm | M3 standard hex | 4 sets | 2 mm counterbore on top of the mast feet, nut pocket under the base plate |
| Fixed mast ↔ carriage | M3 × 25 mm | M3 standard hex | 2 sets | uses two holes 20 mm apart; the rear nuts are exposed |
| Platform ↔ left/right supports | M3 × 12 mm | M3 standard hex | 4 sets | inserted from under the platform, nut pocket on top of the support |
| Support ↔ clamp cap | M3 × 10 mm | M3 standard hex | 4 sets | inserted from above the cap, deep nut pocket under the support |
| Total | 12 mm × 8 / 25 mm × 2 / 10 mm × 4 | 14 | 14 sets | screw length is under-head length |

Screw heads are assumed cylindrical/socket head fitting a Ø6.4 mm counterbore. The design is not for countersunk heads. Nuts are plain nuts, not inserts or heat-set nuts. The cap nut pockets are 4.6 mm deep; push the nut up against the pocket ceiling. Check hole depths and real screw/nut heights with the coupon and during assembly. Screws and nuts are not included in the STEP.

1. Use the coupon to test the M3 clearance hole, nut fit and the 2.3 mm gap.
2. Put the nuts under the base plate and fix the mast with four screws.
3. Hold the carriage against the front of the mast and fix it at the desired height with the two screws 20 mm apart. At the default height, the holes at z=70 and 90 mm from the base can be used. Take the robot off before adjusting.
4. Pre-insert the four support nuts and the four cap nuts. Loosely fasten the four screws under the platform. The two 4 mm ribs under the carriage were each moved 6 mm toward the center so they do not block the slots. The vertical plate edge that overlapped the bolt head at the inner end of the slots was also cut back by up to 1.8 mm. All four slots have Ø7 mm head/tool access over the full travel of the bolt center. The center gap between the ribs is 8 mm, and the original slot positions, the platform's head-bearing surface and the width range are unchanged.
5. Place the robot in the center, fit both supports to the body width, then tighten the screws under the platform.
6. Fasten the caps so their center tongue covers the body plate. There is 0.3 mm of space under the outer side of the cap, so it clamps the plate. Do not fully tighten the caps with no robot in place, or they will deform.
7. First, with power off, check joint motion, cables and screw access. This stand is for zeroing and unloaded joint tests. It is not for walking tests where the feet push against the ground.

## PLA print starting values

A 0.4 mm nozzle, 0.2 mm layers, 5 walls, 5 top/bottom layers and about 30% infill are suggested as **starting values for a test print**. They are not validated strength numbers. Use the filament maker's PLA profile.

- base: put the large flat face on the bed. Check the short bridges over the nut pockets in the slicer.
- mast/carriage: the supplied STLs lie on their side so vertical loads don't act only in the layer-separation direction. The protruding feet and platform need supports; check the slicer preview.
- jaw: the supplied STL is upside down. Check supports under the step faces and remove any support left inside the nut pockets.
- cap/coupon: put the large face down.

Every part is checked to fit within 220 mm in XY in its print orientation. The official nominal P1S build volume is 256 × 256 × 256 mm. Keep the slicer's real exclusion zones and brim margin.

## Scope of the checks

Use the latest results in `validation.json`. An OpenCascade solid intersection above 0.05 mm³ is recorded as interference. Checked: every part pair inside the stand, the robot's default pose, and 5 points across each joint's limit range (other joints at the standing pose). 61 poses in total.

`carriage_slot_fastener_access` checks, for each of the four slots of the carriage alone, that the space below for a Ø7 mm head/tool and the platform pass-through are clear along the full 25 mm travel of the bolt center. The real tool handle, access blocked by other assembled parts, and strength after fastening need separate checking.

The checks apply to the default 66 mm width and the default 156 mm support height. Passing is not guaranteed at other settings or for other robots. Not included: continuous swept volume of simultaneous multi-joint motion, real screw heads, cables, print strength and long-term creep, fastening friction, and tip-over testing. Keep CAD passes and real-world verification separate.

## References

- [Importing and editing STEP in Onshape](https://www.onshape.com/en/resource-center/tech-tips/import-edit-step-iges-parasolid-stl)
- [Importing Assemblies in Onshape](https://www.onshape.com/en/resource-center/tech-tips/tech-tip-importing-assemblies)
- [Official P1S specifications](https://store.bblcdn.com/63ec128d3b8f4f32b7d60fe4dd112ed3.pdf)
