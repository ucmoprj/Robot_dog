"""Calibrate the 12 servos one at a time and save the result to hardware_calibration.json.

Setup: flash hardware/esp32_servo_bridge to the ESP32, put the robot on the stand
       (legs in the air), connect servo power (5 V, 5 A or more).

Usage (from the project root):
    pip install pyserial
    python hardware/calibrate.py --port COM5          # real hardware
    python hardware/calibrate.py --dry-run            # practice the keys without hardware
    python hardware/calibrate.py --port COM5 --joint fr_3   # start from a given joint

Per joint (the "NEXT:" line under each status line spells out every step: where to look,
what the phone should read, and which key moves which way):
    1) use a/d (±10 µs) and z/c (±2 µs) to put the leg at the standing-pose angle → 1
    2) turn it by the reference angle (15° or 30°) in the direction shown → 2
       (direction and µs/rad are computed and the pulse change is checked)
    3) one end, a few degrees before it hits anything → 3 / the other end → 4   (physical limits)
    4) s to save, then t sweeps the full sim range slowly to check → n for the next joint
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
    CALIB_PATH, JOINT_ORDER, US_ABS_MAX, US_ABS_MIN, ServoLink, channel_cfg, is_complete,
    load_calibration, load_joint_specs, rad_to_us, rad_to_us_unclamped, save_calibration,
)

SWEEP_SPEED = 0.5  # rad/s, speed of the check sweep
US_PER_RAD = 650.0  # what this robot's MG90S servos measured (about 11.3 µs per degree)

LEG_NAMES = {"fl": "front-left", "fr": "front-right", "bl": "back-left", "br": "back-right"}
KIND_NAMES = {"1": "roll: swings the whole leg sideways",
              "2": "hip: swings the thigh toward the head or tail",
              "3": "knee: bends the shank"}
PART = {"1": "bracket", "2": "thigh", "3": "shank"}


def ref_delta_deg(joint: str) -> float:
    """Angle to turn for reference mark 2. Roll uses 15° (narrow range); right-side roll only has room in the − direction."""
    leg, idx = joint.split("_")
    if idx == "1":
        return 15.0 if leg in ("fl", "bl") else -15.0
    return 30.0


def signed_deg(v: float) -> str:
    """-19.4 -> "-19.4°", 10.6 -> "+10.6°", 0 -> "0.0°"."""
    return "0.0°" if abs(v) < 0.05 else f"{v:+.1f}°"


def side_words(joint: str) -> tuple[str, str]:
    """(what a − change looks like, what a + change looks like), in plain words."""
    leg, kind = joint.split("_")
    if kind == "1":  # + moves the foot to the robot's left: outward on left legs, inward on right legs
        return ("INWARD", "OUTWARD") if leg[1] == "l" else ("OUTWARD", "INWARD")
    if kind == "2":
        return "toward the HEAD", "toward the TAIL"
    return "straighter", "more bent"


def how_to_measure(joint: str) -> str:
    leg, kind = joint.split("_")
    side = "LEFT" if leg[1] == "l" else "RIGHT"
    if kind == "1":
        return ("Stand IN FRONT of the robot. Hold the phone upright, flat against the outer face "
                "of the shoulder bracket. 0° = bracket vertical.")
    if kind == "2":
        return (f"Stand at the robot's {side} side. Lay the phone along the thigh, "
                "from the hip screw to the knee screw. 0° = thigh vertical.")
    return (f"Stand at the robot's {side} side. Lay the phone along the shank, from the knee screw "
            "to the foot. 0° = shank vertical. The thigh must be holding its standing pose.")


def phone_reading(joint: str, specs: dict, angle_deg: float) -> str:
    """What the phone should show when this joint is at angle_deg (sim angle)."""
    leg, kind = joint.split("_")
    if kind == "1":
        if abs(angle_deg) < 0.05:
            return "0.0° (bracket vertical, leg straight down)"
        minus, plus = side_words(joint)
        return f"{abs(angle_deg):.1f}°, bottom of the bracket tilted {plus if angle_deg > 0 else minus}"
    if kind == "2":
        if abs(angle_deg) < 0.05:
            return "0.0° (thigh vertical)"
        return f"{abs(angle_deg):.1f}°, knee end of the thigh toward the {'TAIL' if angle_deg > 0 else 'HEAD'}"
    tilt = math.degrees(specs[f"{leg}_2"].stand_rad) + angle_deg  # shank tilt = hip + knee
    return (f"{abs(tilt):.1f}°, foot end of the shank toward the {'TAIL' if tilt > 0 else 'HEAD'} "
            f"(= {angle_deg:.0f}° bend between thigh and shank)")


def expected_delta_us(joint: str) -> tuple[int, int]:
    """Usual pulse change between pose 1 and pose 2 for these MG90S servos (±14%)."""
    nominal = math.radians(abs(ref_delta_deg(joint))) * US_PER_RAD
    return round(nominal * 0.86), round(nominal * 1.14)


def joint_header(joint: str, idx: int, ch: dict, specs: dict, done: int, dirty: bool) -> list[str]:
    """The lines printed when a joint is selected."""
    leg, kind = joint.split("_")
    spec = specs[joint]
    stand = math.degrees(spec.stand_rad)
    delta = ref_delta_deg(joint)
    minus, plus = side_words(joint)
    return [
        f"===== {joint}  (PCA9685 channel {ch['channel']}) =====  "
        f"[joint {idx + 1} of {len(JOINT_ORDER)} · {done}/{len(JOINT_ORDER)} complete"
        f"{' · unsaved changes' if dirty else ''}]",
        f"  {LEG_NAMES[leg]} leg · {KIND_NAMES[kind]}",
        f"  sim range: {math.degrees(spec.min_rad):.1f}° to {math.degrees(spec.max_rad):.1f}°",
        f"  measure: {how_to_measure(joint)}",
        f"  pose 1 (key 1): phone reads {phone_reading(joint, specs, stand)}",
        f"  pose 2 (key 2): phone reads {phone_reading(joint, specs, stand + delta)}",
        f"                  = pose 1 turned {abs(delta):.0f}° {plus if delta > 0 else minus}",
    ]


def next_step_hint(joint: str, ch: dict, specs: dict, marks: dict, dirty: bool, tested: bool) -> str:
    """The one thing to do next for this joint, from what has been marked in this session."""
    kind = joint.split("_")[1]
    part = PART[kind]
    stand = math.degrees(specs[joint].stand_rad)
    delta = ref_delta_deg(joint)
    minus, plus = side_words(joint)
    if "center" not in marks:
        if is_complete(ch):
            return "Already calibrated. g: check the standing pose · t: test · n: next joint. To redo it, reach pose 1 and press 1."
        return (f"POSE 1. Get the phone to read {phone_reading(joint, specs, stand)}. "
                "Press d (+10 µs) or a (-10 µs) and watch the number: if it moves away from the target, "
                "switch keys. Finish with c / z (±2 µs). Then press 1.")
    if "ref" not in marks:
        turn = plus if delta > 0 else minus
        lo, hi = expected_delta_us(joint)
        return (f"POSE 2. Turn the {part} {turn} by {abs(delta):.0f}° until the phone reads "
                f"{phone_reading(joint, specs, stand + delta)}. Which key? Press d 3 times: if the {part} "
                f"moves {turn}, keep using d / c; if it moves the other way, use a / z. "
                f"The pulse should end about {lo}-{hi} µs away from {ch['center_us']} µs. Then press 2.")
    # direction is known now, so the keys for each side are known
    k_plus, k_minus = ("d", "a") if ch["direction"] > 0 else ("a", "d")
    f_plus, f_minus = ("c", "z") if ch["direction"] > 0 else ("z", "c")
    side_of = {k: ("plus" if (v - ch["center_us"]) * ch["direction"] > 0 else "minus")
               for k, v in marks.items() if k in ("limit3", "limit4")}
    have3, have4 = "limit3" in marks, "limit4" in marks
    moved = abs(marks["ref"] - marks["center"])
    lo, hi = expected_delta_us(joint)
    if not (have3 or have4) and not lo <= moved <= hi:
        return (f"CHECK POSE 2. The pulse moved {moved} µs, but {lo}-{hi} µs is usual. Look at the phone: "
                f"it should read {phone_reading(joint, specs, stand + delta)}. Adjust with a / d / z / c and "
                f"press 2 again. If you are sure the angle is right, go on to the limits: {k_minus} moves "
                f"the {part} {minus}.")
    if not (have3 and have4):
        todo = "minus" if not side_of or "plus" in side_of.values() else "plus"
        word, k_go, f_go, k_back = ((minus, k_minus, f_minus, k_plus) if todo == "minus"
                                    else (plus, k_plus, f_plus, k_minus))
        key = "3" if not have3 else "4"
        return (f"LIMIT. Press {k_go} slowly: the {part} moves {word} (use {f_go} near the end). When it "
                f"touches anything or the servo buzzes, press {k_back} 3-5 times to back off. Then press {key}.")
    sweep = f"During t the {part} should move {minus} first, then {plus}, then back to pose 1."
    if dirty:
        return "Press s to save." if tested else f"Press s to save, then t to test. {sweep}"
    if not tested:
        return (f"Press t to test. {sweep} If it moves {plus} first, pose 2 was turned the wrong way: "
                "redo this joint from pose 1.")
    return "This joint is done. Press n for the next joint, or q to quit."


def make_getch():
    if not sys.stdin.isatty():  # piped input (for tests): skip whitespace, quit at end of input
        def getch():
            while True:
                c = sys.stdin.read(1)
                if c == "":
                    return "q"
                if not c.isspace():
                    return c
        return getch
    if os.name == "nt":
        import msvcrt
        return msvcrt.getwch
    import termios
    import tty

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return getch


HELP = """
  a/d : −10/+10 µs      z/c : −2/+2 µs       g : go to center
  1   : now = standing-pose angle (center)
  2   : now = standing pose + reference angle (computes direction and µs/rad)
  3/4 : now = physical limit, one end / the other end
  t   : slowly sweep the full sim range (after calibration)
  r   : relax this servo    n/p : next/previous joint
  s   : save                h   : help             q : save and quit (all servos relax)
"""


class Calibrator:
    def __init__(self, link: ServoLink, start_joint: str):
        self.link = link
        self.calib = load_calibration()
        self.specs = load_joint_specs()
        self.idx = JOINT_ORDER.index(start_joint)
        self.us = 1500
        self.marks: dict[str, int] = {}
        self.dirty = False    # calibration changed since the last save
        self.tested = False   # this joint passed a t sweep since it was last changed

    @property
    def joint(self) -> str:
        return JOINT_ORDER[self.idx]

    @property
    def ch(self) -> dict:
        return channel_cfg(self.calib, self.joint)

    def send(self, us: int) -> None:
        self.us = max(US_ABS_MIN, min(US_ABS_MAX, us))
        self.link.set_us(self.ch["channel"], self.us)

    def select(self, idx: int) -> None:
        self.idx = idx % len(JOINT_ORDER)
        self.marks = {}
        self.tested = False
        ch = self.ch
        done = sum(is_complete(c) for c in self.calib["channels"])
        print()
        for line in joint_header(self.joint, self.idx, ch, self.specs, done, self.dirty):
            print(line)
        self.send(ch["center_us"] if ch["center_us"] is not None else 1500)
        self.status()

    def status(self) -> None:
        ch = self.ch
        print(f"  now {self.us} µs | center={ch['center_us']} dir={ch['direction']} "
              f"us/rad={ch['us_per_rad']} min={ch['min_us']} max={ch['max_us']}"
              f"{'  ✔ done' if is_complete(ch) else ''}")
        hint = next_step_hint(self.joint, ch, self.specs, self.marks, self.dirty, self.tested)
        print(f"  NEXT: {hint}")

    def mark_center(self) -> None:
        self.ch["center_us"] = self.us
        self.ch["zero_rad"] = self.specs[self.joint].stand_rad
        self.marks = {"center": self.us}  # a new center starts this joint over:
        for k in ("direction", "us_per_rad", "min_us", "max_us"):  # old values may be from a horn
            self.ch[k] = None                                       # position that no longer exists
        self.dirty, self.tested = True, False
        print(f"  center = {self.us} µs recorded (in memory; s writes it to the file)")

    def mark_ref(self) -> None:
        center = self.ch["center_us"]
        if center is None:
            print("  Save 1 (center) first")
            return
        delta_us = self.us - center
        if abs(delta_us) < 50:
            print("  Too close to center. Make sure you turned the full reference angle")
            return
        delta_rad = math.radians(ref_delta_deg(self.joint))
        slope = delta_us / delta_rad
        self.ch["direction"] = 1 if slope > 0 else -1
        self.ch["us_per_rad"] = round(abs(slope), 1)
        self.marks["ref"] = self.us
        self.dirty, self.tested = True, False
        print(f"  direction={self.ch['direction']}, us_per_rad={self.ch['us_per_rad']} "
              f"({abs(slope) * math.pi / 180:.2f} µs/deg)")
        lo, hi = expected_delta_us(self.joint)
        ref = abs(ref_delta_deg(self.joint))
        print(f"  pose 2 is {abs(delta_us)} µs from pose 1. An MG90S usually needs {lo}-{hi} µs for {ref:.0f}°.")
        if not lo <= abs(delta_us) <= hi:
            print(f"  ⚠ Outside that range: pose 2 is probably not exactly {ref:.0f}° from pose 1 "
                  "(stopped early or went too far). Fix the position and press 2 again, "
                  "or press g and redo from pose 1.")

    def mark_limit(self, key: str) -> None:
        other = self.marks.get("limit4" if key == "limit3" else "limit3")
        center = self.ch["center_us"]
        if other is not None and center is not None and (self.us - center) * (other - center) > 0:
            print("  ⚠ Not recorded: both limits would be on the same side of pose 1. "
                  "The two limits must be on opposite sides: go back past pose 1 to the other end.")
            return
        self.marks[key] = self.us
        print(f"  limit {key[-1]} = {self.us} µs recorded")
        if "limit3" in self.marks and "limit4" in self.marks:
            lo, hi = sorted((self.marks["limit3"], self.marks["limit4"]))
            self.ch["min_us"], self.ch["max_us"] = lo, hi
            self.dirty, self.tested = True, False
            print(f"  min_us={lo}, max_us={hi}")
            self.check_range()

    def check_range(self) -> None:
        """Check that the whole sim joint range fits inside the physical limits."""
        ch, spec = self.ch, self.specs[self.joint]
        if not is_complete(ch):
            return
        ends = sorted(rad_to_us_unclamped(ch, q) for q in (spec.min_rad, spec.max_rad))
        if ends[0] < ch["min_us"] or ends[1] > ch["max_us"]:
            print(f"  ⚠ The sim range needs {ends[0]:.0f}-{ends[1]:.0f} µs but the hardware only allows "
                  f"{ch['min_us']}-{ch['max_us']} µs. This joint will be clamped and move less than "
                  f"in sim (re-seat the horn or narrow the range in robot.xml)")
        else:
            print(f"  sim range {ends[0]:.0f}-{ends[1]:.0f} µs fits inside the physical limits ✔")

    def sweep(self) -> None:
        ch, spec = self.ch, self.specs[self.joint]
        if not is_complete(ch):
            print("  Available after saving marks 1-4")
            return
        print("  Sweeping: standing → min → max → standing")
        dt = 1 / 50
        q = spec.stand_rad
        for target in (spec.min_rad, spec.max_rad, spec.stand_rad):
            while abs(target - q) > 1e-6:
                q += max(-SWEEP_SPEED * dt, min(SWEEP_SPEED * dt, target - q))
                self.link.set_us(ch["channel"], rad_to_us(ch, q))
                if not self.link.dry_run:
                    time.sleep(dt)
            time.sleep(0.3)
        self.us = rad_to_us(ch, spec.stand_rad)
        self.tested = True

    def run(self) -> None:
        getch = make_getch()
        print(HELP)
        self.select(self.idx)
        while True:
            key = getch()
            if key in "adzc":
                self.send(self.us + {"a": -10, "d": 10, "z": -2, "c": 2}[key])
                print(f"  {self.us} µs", end="\r" if sys.stdin.isatty() else "\n", flush=True)
                continue
            if key == "1":
                self.mark_center()
            elif key == "2":
                self.mark_ref()
            elif key in "34":
                self.mark_limit(f"limit{key}")
            elif key == "g" and self.ch["center_us"] is not None:
                self.send(self.ch["center_us"])
            elif key == "t":
                self.sweep()
            elif key == "r":
                self.link.relax(self.ch["channel"])
                print("  Relaxed (press a/d/g to hold again)")
            elif key in ("n", "p"):
                if self.dirty:
                    print("  Note: unsaved changes are kept in memory. Press s to write them to the file.")
                self.select(self.idx + (1 if key == "n" else -1))
                continue
            elif key == "s":
                save_calibration(self.calib)
                self.dirty = False
                print(f"  Saved: {CALIB_PATH}")
            elif key == "h":
                print(HELP)
            elif key in ("q", "\x03"):
                save_calibration(self.calib)
                done = sum(is_complete(c) for c in self.calib["channels"])
                print(f"\nSaved: {CALIB_PATH}  ({done}/12 joints done)")
                return
            else:
                continue
            self.status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM5, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="print commands only, no hardware")
    parser.add_argument("--joint", default="fl_1", choices=JOINT_ORDER, help="joint to start from")
    args = parser.parse_args()
    if not args.dry_run and not args.port:
        parser.error("--port or --dry-run is required")

    link = ServoLink(args.port, dry_run=args.dry_run)
    try:
        Calibrator(link, args.joint).run()
    finally:
        link.close()


if __name__ == "__main__":
    main()
