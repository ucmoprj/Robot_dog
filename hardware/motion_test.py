"""Keyboard motion test for the real robot, without the AI policy: poses, bounce, hop and a short trot.

Use it after calibration (hardware_calibration.json status MEASURED) to check that the calibrated
joints move the legs the way the simulation expects, before running a trained policy.

Usage (from the project root):
    python hardware/motion_test.py --port COM4      # real hardware
    python hardware/motion_test.py --dry-run        # practice the keys without hardware

Keys (shown again in the program):
    1 : stand            2 : crouch (all four legs fold, body goes down)
    3 : dog sit (rear legs fold)      4 : play bow (front low, rear high)
    i / k : lean forward / back       j / l : lean left / right
    b : bounce 3 times                u : hop attempt (crouch, then extend as fast as the servos can)
    p : paw wave: shift the weight onto fl, bl, br, then wave the right front paw back and forth
    z / x : less / more sideways weight shift for p      c / v : less / more backward weight shift
    w : trot forward                  e : trot in place
    [ / ] : fewer / more steps (1-20, starts at 5)
    + / - : slower / faster steps and bounces     r : relax all servos (the robot drops!)
    q : quit (all servos relax)
    While walking, [ ] + - still work (the walk keeps going with the new value).
    Any OTHER key stops the motion; the robot holds the pose where it stopped.

Trot = diagonal legs move together: fl+br lift while fr+bl push, then the other way round.
One step = one diagonal pair lifts and moves forward. A walk is N steps (5 unless changed), then the robot stands.

Leg geometry (from dog_mujoco/robot.xml), side view, x = toward the HEAD, d = depth below the hip axis:
    foot x = -L_THIGH*sin(hip) - L_SHANK*sin(hip + knee)
    foot d =  L_THIGH*cos(hip) + L_SHANK*cos(hip + knee)
The roll joints (_1) stay at their standing angle (0) except in the paw wave, where they slide the feet
sideways to move the body over the three standing feet.

Paw wave without an IMU (open loop, estimated balance):
    The estimated center of mass (design_summary.json) is at the middle of the body, which lies exactly on
    the fl-br diagonal, so lifting fr straight away would tip the robot onto it. Before the lift, the body is
    moved toward the rear-left, inside the triangle fl-bl-br:
      sideways: roll joints slide all feet to the robot's RIGHT (body goes left) + left legs a bit shorter
                (body tilts left, which also carries it left)
      backward: all feet step toward the HEAD (body goes back)
    Real robots differ from the estimate (battery, wires), so z / x and c / v change the shift.
    If the robot tips toward fr when the paw lifts, press x (more left) or v (more back).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robot_io import (  # noqa: E402
    CALIB_PATH,
    JOINT_ORDER,
    ServoLink,
    channel_cfg,
    is_complete,
    load_calibration,
    load_joint_specs,
    rad_to_us,
)

L_THIGH = 0.044  # m, hip axis → knee axis (fl_shank pos in robot.xml)
L_SHANK = 0.038  # m, knee axis → foot site (fl_foot pos in robot.xml)
LEGS = ("fl", "fr", "bl", "br")
PAIRS = (("fl", "br"), ("fr", "bl"))  # trot: diagonal legs move together

RATE_HZ = 50          # pose updates per second
POSE_TIME = 1.5       # s, stand ↔ sit transition
SIT_DEPTH = 0.064     # m, foot depth when sitting (knee ≈ 78°, inside fr_3's 82° hardware limit)
WALK_DEPTH = 0.074    # m, slightly lower than standing so the legs have reach for the stride
STRIDE = 0.016        # m, foot travel per full step (±8 mm around the hip)
LIFT = 0.010          # m, foot lift during swing
DEFAULT_STEPS = 5
STEPS_RANGE = (1, 20)  # changed with [ / ]
STEP_TIME_RANGE = (0.3, 0.8)  # s per step (and per bounce), changed with + / -

LOW_DEPTH = 0.066     # m, lowered legs in the lean poses
HIGH_DEPTH = 0.080    # m, extended legs (knee ≈ 25°; full stretch would be 0.082)
LEAN_TIME = 0.8       # s, transition into a lean / bow / dog-sit pose
BOUNCE_DEPTH = 0.068  # m, bottom of a bounce
BOUNCE_REPS = 3
HOP_PUSH_TIME = 0.15  # s, legs commanded straight before they bend again to catch the landing

ROLL_ARM = 0.037      # m, roll axis → hip pitch axis (fl_thigh pos in robot.xml); foot lever = ROLL_ARM + d
PAW_DEPTH = 0.074     # m, standing depth during the paw wave (a little lower = steadier)
PAW_SIDE = 0.008      # m, default body shift to the LEFT by the roll joints (left legs roll in only 6°)
PAW_TILT = 0.006      # m, left legs this much shorter than right legs (tilts the body left, ~4°)
PAW_BACK = 0.010      # m, default body shift toward the TAIL (feet step toward the HEAD)
PAW_SIDE_RANGE = (0.0, 0.011)
PAW_BACK_RANGE = (0.0, 0.020)
PAW_KNEE = math.radians(70)  # lifted paw: knee bend kept fixed while the hip swings
PAW_HIP_FWD, PAW_HIP_BACK = math.radians(-45), math.radians(-10)  # hip swing ends (sim range -48°..20°)
PAW_WAVES = 3
PAW_WAVE_TIME = 0.7   # s per back-and-forth

# Feet (x, d) per leg for each held pose. Small x (toward the HEAD) keeps the feet under the body
# when it tilts. Pitch ≈ depth difference / 108 mm hip spacing, roll ≈ depth difference / 80 mm.
POSES = {
    "3": ("dog sit (rear legs fold, body tilts back)",
          {"fl": (0.0, HIGH_DEPTH), "fr": (0.0, HIGH_DEPTH), "bl": (0.006, SIT_DEPTH), "br": (0.006, SIT_DEPTH)}),
    "4": ("play bow (front low and reaching forward, rear high)",
          {"fl": (0.010, SIT_DEPTH), "fr": (0.010, SIT_DEPTH), "bl": (0.0, HIGH_DEPTH), "br": (0.0, HIGH_DEPTH)}),
    "i": ("lean forward (nose down)",
          {"fl": (0.0, LOW_DEPTH), "fr": (0.0, LOW_DEPTH), "bl": (0.0, HIGH_DEPTH), "br": (0.0, HIGH_DEPTH)}),
    "k": ("lean back (nose up)",
          {"fl": (0.0, HIGH_DEPTH), "fr": (0.0, HIGH_DEPTH), "bl": (0.0, LOW_DEPTH), "br": (0.0, LOW_DEPTH)}),
    "j": ("lean left",
          {"fl": (0.0, LOW_DEPTH), "bl": (0.0, LOW_DEPTH), "fr": (0.0, HIGH_DEPTH), "br": (0.0, HIGH_DEPTH)}),
    "l": ("lean right",
          {"fl": (0.0, HIGH_DEPTH), "bl": (0.0, HIGH_DEPTH), "fr": (0.0, LOW_DEPTH), "br": (0.0, LOW_DEPTH)}),
}


# ---------- leg kinematics ----------

def leg_fk(hip: float, knee: float) -> tuple[float, float]:
    x = -L_THIGH * math.sin(hip) - L_SHANK * math.sin(hip + knee)
    d = L_THIGH * math.cos(hip) + L_SHANK * math.cos(hip + knee)
    return x, d


def leg_ik(x: float, d: float) -> tuple[float, float]:
    """Foot position → (hip, knee) in sim radians, knee bent forward like the standing pose."""
    c = (x * x + d * d - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    knee = math.acos(max(-1.0, min(1.0, c)))
    hip = math.atan2(-x, d) - math.atan2(L_SHANK * math.sin(knee), L_THIGH + L_SHANK * math.cos(knee))
    return hip, knee


# ---------- keyboard ----------

class Keys:
    """get(): wait for one key. hit(): a key if one was pressed, else None (never waits)."""

    def __enter__(self):
        self.tty = sys.stdin.isatty()
        if self.tty and os.name != "nt":
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self.tty and os.name != "nt":
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def get(self) -> str:
        if not self.tty:  # piped input (for tests): skip whitespace, quit at end of input
            while True:
                c = sys.stdin.read(1)
                if c == "":
                    return "q"
                if not c.isspace():
                    return c
        if os.name == "nt":
            import msvcrt
            return msvcrt.getwch()
        return sys.stdin.read(1)

    def hit(self) -> str | None:
        if not self.tty:
            return None
        if os.name == "nt":
            import msvcrt
            return msvcrt.getwch() if msvcrt.kbhit() else None
        import select
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


# ---------- motion ----------

class Robot:
    def __init__(self, link: ServoLink, keys: Keys):
        self.link, self.keys = link, keys
        self.calib = load_calibration()
        self.specs = load_joint_specs()
        self.q: dict[str, float] | None = None  # last pose sent (None = servos relaxed / unknown)
        self.step_time = 0.4
        self.walk_steps = DEFAULT_STEPS
        self.paw_side, self.paw_back = PAW_SIDE, PAW_BACK
        stand = {j: self.specs[j].stand_rad for j in JOINT_ORDER}
        self.stand_pose = stand
        _, self.stand_depth = leg_fk(stand["fl_2"], stand["fl_3"])

    def pose_from_feet(self, feet: dict[str, tuple]) -> dict[str, float]:
        """feet[leg] = (x, d) or (x, d, roll). Roll defaults to the standing angle."""
        q = {}
        for leg in LEGS:
            x, d, *roll = feet[leg]
            hip, knee = leg_ik(x, d)
            q[f"{leg}_1"] = roll[0] if roll else self.specs[f"{leg}_1"].stand_rad
            q[f"{leg}_2"], q[f"{leg}_3"] = hip, knee
        return q

    def to_us(self, q: dict[str, float]) -> list[int]:
        us = []
        for j in JOINT_ORDER:
            spec = self.specs[j]
            qj = min(max(q[j], spec.min_rad), spec.max_rad)  # sim range first, then hardware limits
            us.append(rad_to_us(channel_cfg(self.calib, j), qj))
        return us

    def send(self, q: dict[str, float]) -> None:
        us = self.to_us(q)
        if not self.link.dry_run:
            self.link.set_all_us(us)
        self.q = q

    def wake(self) -> None:
        """First command after power-up: servo positions are unknown, so bring the joints to the
        standing pose one at a time (smaller current spike than all 12 jumping together)."""
        print("  Waking up: moving each joint to the standing pose, one at a time...")
        us = self.to_us(self.stand_pose)
        for j, u in zip(JOINT_ORDER, us):
            if not self.link.dry_run:
                self.link.set_us(channel_cfg(self.calib, j)["channel"], u)
            time.sleep(0.15)
        self.q = dict(self.stand_pose)

    def adjust(self, key: str) -> bool:
        """[ ] change the step count, + - the step time. Returns False for any other key."""
        if key in "[]":
            lo, hi = STEPS_RANGE
            self.walk_steps = min(hi, max(lo, self.walk_steps + (1 if key == "]" else -1)))
            print(f"  steps: {self.walk_steps}")
        elif key in "+=-_":
            lo, hi = STEP_TIME_RANGE
            self.step_time = round(min(hi, max(lo, self.step_time + (0.05 if key in "+=" else -0.05))), 2)
            print(f"  step time: {self.step_time:.2f} s")
        else:
            return False
        return True

    def play(self, frames, live_keys: bool = False) -> bool:
        """Send each pose at RATE_HZ. Returns False if a key stopped it (the last pose is held).
        live_keys=True: [ ] + - adjust the running walk instead of stopping it."""
        dt = 1 / RATE_HZ
        t_next = time.perf_counter()
        for q in frames:
            key = self.keys.hit()
            if key is not None and not (live_keys and self.adjust(key)):
                print("  STOPPED. Holding this pose.")
                return False
            self.send(q)
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
        return True

    def move_to(self, target: dict[str, float], duration: float) -> bool:
        if self.q is None:
            self.wake()
        start = dict(self.q)
        n = max(1, round(duration * RATE_HZ))

        def frames():
            for i in range(1, n + 1):
                s = i / n
                s = s * s * (3 - 2 * s)  # smoothstep: starts and ends gently
                yield {j: start[j] + (target[j] - start[j]) * s for j in JOINT_ORDER}
        return self.play(frames())

    def stand(self) -> bool:
        print("  -> stand")
        return self.move_to(self.stand_pose, POSE_TIME)

    def level(self, depth: float) -> dict[str, float]:
        """All four feet under the hips at the same depth."""
        return self.pose_from_feet({leg: (0.0, depth) for leg in LEGS})

    def hold(self, seconds: float) -> bool:
        return self.play(dict(self.q) for _ in range(max(1, round(seconds * RATE_HZ))))

    def sit(self) -> bool:
        print("  -> crouch")
        return self.move_to(self.level(SIT_DEPTH), POSE_TIME)

    def pose(self, key: str) -> bool:
        name, feet = POSES[key]
        print(f"  -> {name}  (1 = back to standing)")
        return self.move_to(self.pose_from_feet(feet), LEAN_TIME)

    def bounce(self) -> bool:
        print(f"  -> bounce x{BOUNCE_REPS}, {self.step_time:.2f} s each (+ / - in the menu to change)")
        if self.q != self.stand_pose and not self.stand():
            return False
        for i in range(BOUNCE_REPS):
            print(f"    bounce {i + 1}/{BOUNCE_REPS}")
            if not (self.move_to(self.level(BOUNCE_DEPTH), self.step_time / 2)
                    and self.move_to(self.stand_pose, self.step_time / 2)):
                return False
        return True

    def hop(self) -> bool:
        """Crouch slowly, then command straight legs in ONE update so each servo turns at its
        top speed, then bend back to standing right away so the legs cushion the landing.
        Expect a small bounce rather than a real jump: the legs only extend about 16 mm."""
        print("  -> hop attempt: crouch... (hold the robot loosely the first time)")
        if not self.move_to(self.level(SIT_DEPTH), 1.0) or not self.hold(0.4):
            return False
        print("    push!")
        self.send(self.level(HIGH_DEPTH))
        time.sleep(HOP_PUSH_TIME)
        self.send(self.stand_pose)
        return self.hold(0.5)

    def trot(self, stride: float) -> bool:
        what = "forward" if stride > 0 else "in place"
        print(f"  -> trot {what}, {self.walk_steps} steps, {self.step_time:.2f} s per step")
        print("    ([ ] + - change it while walking; any other key stops)")
        print("    (a step = one diagonal pair lifts: fl+br, then fr+bl, ...)")
        if not self.stand():
            return False
        if not self.move_to(self.pose_from_feet({leg: (0.0, WALK_DEPTH) for leg in LEGS}), 0.5):
            return False
        if not self.play(self._trot_frames(stride), live_keys=True):
            return False
        return self.stand()

    def _trot_frames(self, stride: float):
        """First step starts from feet under the hips and the last step ends there, so the
        walk begins and ends in the standing pose. Stance feet slide toward the TAIL
        (which pushes the body forward); swing feet lift and move toward the HEAD.
        walk_steps and step_time are read at the start of every step, so [ ] + - take
        effect from the next step (lowering the count below the steps done ends with the next step)."""
        half = stride / 2
        x = {leg: 0.0 for leg in LEGS}
        step = 0
        while True:
            swing, stance = PAIRS[step % 2], PAIRS[(step + 1) % 2]
            last = step >= self.walk_steps - 1
            n = max(1, round(self.step_time * RATE_HZ))
            start = dict(x)
            goal = {**{leg: 0.0 if last else half for leg in swing},
                    **{leg: 0.0 if last else -half for leg in stance}}
            print(f"    step {step + 1}/{max(self.walk_steps, step + 1)}: {'+'.join(swing)} lift")
            for i in range(1, n + 1):
                u = i / n
                feet = {}
                for leg in swing:
                    s = (1 - math.cos(math.pi * u)) / 2
                    feet[leg] = (start[leg] + (goal[leg] - start[leg]) * s,
                                 WALK_DEPTH - LIFT * math.sin(math.pi * u))
                for leg in stance:
                    feet[leg] = (start[leg] + (goal[leg] - start[leg]) * u, WALK_DEPTH)
                yield self.pose_from_feet(feet)
            x = goal
            if last:
                return
            step += 1

    def shifted_feet(self) -> dict[str, tuple]:
        """Feet for the paw wave: body moved toward the rear-left, all four feet still down.
        Roll by phi moves a foot sideways by (ROLL_ARM + d)*sin(phi) and lifts it a little, so d is
        lengthened by 1/cos(phi) to keep the foot at the same height."""
        feet = {}
        for leg in LEGS:
            depth = PAW_DEPTH + (-PAW_TILT / 2 if leg[1] == "l" else PAW_TILT / 2)
            arm = ROLL_ARM + depth
            phi = -math.asin(self.paw_side / arm)  # − = foot toward the robot's right
            feet[leg] = (self.paw_back, arm / math.cos(phi) - ROLL_ARM, phi)
        return feet

    def paw_wave(self) -> bool:
        print(f"  -> paw wave: weight shift left {self.paw_side * 1000:.0f} mm (+ tilt), "
              f"back {self.paw_back * 1000:.0f} mm  (z/x, c/v change it)")
        print("    Watch the robot when the paw lifts. If it tips toward the right front: press any key,")
        print("    then 1 to stand, and x (more left) or v (more back) before trying again.")
        if self.q != self.stand_pose and not self.stand():
            return False
        feet = self.shifted_feet()
        down = self.pose_from_feet(feet)
        print("    1/5 shifting the weight onto fl, bl, br")
        if not (self.move_to(down, 1.2) and self.hold(0.4)):
            return False
        x, d, roll = feet["fr"]
        print("    2/5 lifting the right front paw")
        unload = self.pose_from_feet({**feet, "fr": (x, d - 0.008, roll)})  # lift gently first
        if not (self.move_to(unload, 0.5) and self.hold(0.3)):
            return False
        mid = (PAW_HIP_FWD + PAW_HIP_BACK) / 2
        amp = (PAW_HIP_BACK - PAW_HIP_FWD) / 2
        up = {**unload, "fr_2": mid, "fr_3": PAW_KNEE}
        if not self.move_to(up, 0.5):
            return False
        print(f"    3/5 waving x{PAW_WAVES}")
        n = round(PAW_WAVES * PAW_WAVE_TIME * RATE_HZ)
        # starts at mid and swings toward the HEAD first (hip more negative)
        wave = ({**up, "fr_2": mid - amp * math.sin(2 * math.pi * PAW_WAVES * i / n)} for i in range(1, n + 1))
        if not self.play(wave):
            return False
        print("    4/5 putting the paw down")
        if not (self.move_to(unload, 0.5) and self.move_to(down, 0.4) and self.hold(0.3)):
            return False
        print("    5/5 back to standing")
        return self.move_to(self.stand_pose, 1.0)

    def adjust_paw(self, key: str) -> None:
        if key in "zx":
            lo, hi = PAW_SIDE_RANGE
            self.paw_side = round(min(hi, max(lo, self.paw_side + (0.001 if key == "x" else -0.001))), 3)
        else:
            lo, hi = PAW_BACK_RANGE
            self.paw_back = round(min(hi, max(lo, self.paw_back + (0.002 if key == "v" else -0.002))), 3)
        print(f"  paw wave weight shift: left {self.paw_side * 1000:.0f} mm, back {self.paw_back * 1000:.0f} mm")

    def relax(self) -> None:
        if not self.link.dry_run:
            self.link.relax_all()
        self.q = None
        print("  Relaxed: all servos limp (the next motion wakes them up again)")


MENU = """
  1 : stand            2 : crouch          3 : dog sit          4 : play bow
  i / k : lean forward / back      j / l : lean left / right
  b : bounce x3        u : hop attempt (expect only a small bounce)
  p : paw wave (right front)     z / x : sideways shift -/+     c / v : backward shift -/+
  w : trot forward ({n} steps)    e : trot in place ({n} steps)
  [ / ] : fewer / more steps     + / - : slower / faster (now {t:.2f} s per step / bounce)
  r : relax all servos (the robot drops!)       q : quit
  While walking: [ ] + - change the walk live. Any other key stops and holds the pose.
"""


def check_ready(calib: dict) -> bool:
    missing = [ch["joint"] for ch in calib["channels"] if not is_complete(ch)]
    if missing:
        print(f"Not calibrated yet: {', '.join(missing)}. Finish hardware/calibrate.py first ({CALIB_PATH}).")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: shows what would be sent")
    args = parser.parse_args()
    if not args.port and not args.dry_run:
        parser.error("--port or --dry-run is required")

    if not check_ready(load_calibration()):
        sys.exit(1)
    link = ServoLink(args.port, dry_run=args.dry_run)
    try:
        with Keys() as keys:
            robot = Robot(link, keys)
            print(MENU.format(n=robot.walk_steps, t=robot.step_time))
            while True:
                key = keys.get().lower()
                if key == "1":
                    robot.stand()
                elif key == "2":
                    robot.sit()
                elif key in POSES:
                    robot.pose(key)
                elif key == "b":
                    robot.bounce()
                elif key == "u":
                    robot.hop()
                elif key == "p":
                    robot.paw_wave()
                elif key in "zxcv":
                    robot.adjust_paw(key)
                    continue
                elif key == "w":
                    robot.trot(STRIDE)
                elif key == "e":
                    robot.trot(0.0)
                elif robot.adjust(key):
                    continue
                elif key == "r":
                    robot.relax()
                elif key in ("q", "\x03"):
                    break
                elif key == "h":
                    print(MENU.format(n=robot.walk_steps, t=robot.step_time))
                    continue
                else:
                    continue
                if robot.q is not None and link.dry_run:
                    print(f"  [dry-run] now: {robot.to_us(robot.q)}")
    finally:
        link.close()
        print("All servos relaxed. Bye.")


if __name__ == "__main__":
    main()
