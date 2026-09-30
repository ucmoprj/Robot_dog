"""Calibrate the 12 servos one at a time and save the result to hardware_calibration.json.

Setup: flash hardware/esp32_servo_bridge to the ESP32, put the robot on the stand
       (legs in the air), connect servo power (5 V, 5 A or more).

Usage (from the project root):
    pip install pyserial
    python hardware/calibrate.py --port COM5          # real hardware
    python hardware/calibrate.py --dry-run            # practice the keys without hardware
    python hardware/calibrate.py --port COM5 --joint fr_3   # start from a given joint

Per joint:
    1) use a/d (±10 µs) and z/c (±2 µs) to put the leg at the standing-pose angle → 1
    2) turn it by the reference angle (15° or 30°) in the direction shown → 2
       (direction and µs/rad are computed automatically)
    3) one end, a few degrees before it hits anything → 3 / the other end → 4   (physical limits)
    4) t sweeps the full sim range slowly to check → n for the next joint
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

# What a + change of the sim joint angle looks like on the robot (robot.xml axis directions)
PLUS_MEANING = {
    "1": "foot moves to the robot's left (outward on left legs, inward on right legs)",
    "2": "thigh swings back (foot toward the tail)",
    "3": "knee bends more (shank folds back)",
}
STAND_MEANING = {
    "1": "roll 0°: shoulder bracket square to the body, leg straight down",
    "2": "hip −19.4°: thigh tilted about 19° forward of vertical",
    "3": "knee 42°: thigh and shank bent 42° from straight (foot right under the hip)",
}
SWEEP_SPEED = 0.5  # rad/s, speed of the check sweep


def ref_delta_deg(joint: str) -> float:
    """Angle to turn for reference mark 2. Roll uses 15° (narrow range); right-side roll only has room in the − direction."""
    leg, idx = joint.split("_")
    if idx == "1":
        return 15.0 if leg in ("fl", "bl") else -15.0
    return 30.0


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
        ch, spec = self.ch, self.specs[self.joint]
        kind = self.joint.split("_")[1]
        print(f"\n===== {self.joint}  (PCA9685 channel {ch['channel']}) =====")
        print(f"  sim range: {math.degrees(spec.min_rad):.1f}° to {math.degrees(spec.max_rad):.1f}°")
        print(f"  pose 1: {STAND_MEANING[kind]}")
        print(f"  pose 2: pose 1 {ref_delta_deg(self.joint):+.0f}° "
              f"(+ direction = {PLUS_MEANING[kind]})")
        self.send(ch["center_us"] if ch["center_us"] is not None else 1500)
        self.status()

    def status(self) -> None:
        ch = self.ch
        print(f"  now {self.us} µs | center={ch['center_us']} dir={ch['direction']} "
              f"us/rad={ch['us_per_rad']} min={ch['min_us']} max={ch['max_us']}"
              f"{'  ✔ done' if is_complete(ch) else ''}")

    def mark_center(self) -> None:
        self.ch["center_us"] = self.us
        self.ch["zero_rad"] = self.specs[self.joint].stand_rad
        self.marks["center"] = self.us
        print(f"  center = {self.us} µs saved")

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
        print(f"  direction={self.ch['direction']}, us_per_rad={self.ch['us_per_rad']} "
              f"({abs(slope) * math.pi / 180:.2f} µs/deg)")
        if not 450 <= abs(slope) <= 800:
            print("  ⚠ Far from the usual MG90S value (about 570-630 µs/rad). Re-check your angle measurement")

    def mark_limit(self, key: str) -> None:
        self.marks[key] = self.us
        print(f"  limit {key} = {self.us} µs saved")
        if "limit3" in self.marks and "limit4" in self.marks:
            lo, hi = sorted((self.marks["limit3"], self.marks["limit4"]))
            self.ch["min_us"], self.ch["max_us"] = lo, hi
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
            elif key == "n":
                self.select(self.idx + 1)
                continue
            elif key == "p":
                self.select(self.idx - 1)
                continue
            elif key == "s":
                save_calibration(self.calib)
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
