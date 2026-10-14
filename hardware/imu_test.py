"""BNO085 IMU test and mounting setup (course stage 03). No servo moves in this program.

Usage (from the project root, after flashing hardware/esp32_imu_bridge):
    python hardware/imu_test.py --port COM4
    python hardware/imu_test.py --dry-run        # practice the keys without hardware

Keys:
    s : IMU status (found? which I2C address?)
    l : live view, 10 lines per second (any key stops)
    a : axis wizard: finds how the board is mounted and the gyro bias → hardware_imu.json
    q : quit

Why the axis wizard: the policy was trained with the sim's IMU axes (x HEAD, y LEFT, z UP). The real
board can be mounted turned or upside down, so its raw x/y/z can be any of the body axes, with any sign.
The wizard asks for two still poses and reads gravity:
    1) robot level      → the raw axis reading about ±9.8 is body z  (sign chosen so it reads +9.8)
    2) robot nose UP    → the raw axis that gains the most is body x (sign chosen so it is positive)
    body y = z × x (right-handed). 3) a left turn checks it: body gyro z must go positive.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_link import G, IMU_PATH, ImuLink, load_imu_config, mat_vec, save_imu_config, to_body  # noqa: E402


# ---------- keyboard (same idea as motion_test.py, kept local so each stage stands alone) ----------

def getch() -> str:
    if not sys.stdin.isatty():  # piped input (tests): quit at end of input
        while True:
            c = sys.stdin.read(1)
            if c == "":
                return "q"
            if not c.isspace():
                return c
    if os.name == "nt":
        import msvcrt
        return msvcrt.getwch()
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def key_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    if os.name == "nt":
        import msvcrt
        if msvcrt.kbhit():
            msvcrt.getwch()
            return True
        return False
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)
        return True
    return False


# ---------- helpers ----------

def average(link: ImuLink, seconds: float) -> tuple[list[float], list[float]]:
    """Mean raw (gyro, accel) over a still period."""
    n, gs, accs = 0, [0.0] * 3, [0.0] * 3
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end or n == 0:
        s = link.read_raw()
        gs = [a + b for a, b in zip(gs, s.gyro)]
        accs = [a + b for a, b in zip(accs, s.accel)]
        n += 1
        time.sleep(0.01)
    return [g / n for g in gs], [a / n for a in accs]


def cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def fmt(v: list[float], digits: int = 2) -> str:
    return "(" + ", ".join(f"{x:+.{digits}f}" for x in v) + ")"


def wait_key(prompt: str) -> bool:
    print(prompt)
    print("    press SPACE when ready (q cancels)")
    return getch().lower() != "q"


# ---------- modes ----------

def live(link: ImuLink) -> None:
    cfg = load_imu_config()
    if cfg is None:
        print("  No hardware_imu.json yet: showing RAW sensor axes. Run the axis wizard (a) for body axes.")
    else:
        print(f"  Body axes (x HEAD, y LEFT, z UP) from {IMU_PATH.name}, measured {cfg['measured']}")
    print("  Level and still should read: gyro ~(0, 0, 0) deg/s, accel ~(0.00, 0.00, +9.81) m/s²")
    print("  Any key stops.\n")
    t_next = time.perf_counter()
    while not key_pressed():
        s = link.read_raw()
        if cfg is None:
            gyro, accel = s.gyro, s.accel
            tilt = ""
        else:
            gyro, accel = to_body(s, cfg)
            nose_up = math.degrees(math.atan2(accel[0], accel[2]))
            left_up = math.degrees(math.atan2(accel[1], accel[2]))
            tilt = f"  nose-up {nose_up:+5.1f}°  left-side-up {left_up:+5.1f}°"
        gyro_deg = [math.degrees(g) for g in gyro]
        print(f"  gyro {fmt(gyro_deg, 1)} deg/s  accel {fmt(accel)} |a|={math.hypot(*accel):.2f}{tilt}"
              f"  age {s.age_ms} ms")
        t_next += 0.1
        time.sleep(max(0.0, t_next - time.perf_counter()))


def axis_wizard(link: ImuLink) -> None:
    print("\n  AXIS WIZARD (3 steps). Hold the robot by the body; legs can hang or rest on the stand.")

    # 1) level → body z
    if not wait_key("\n  STEP 1/3. Hold the robot LEVEL (as if standing on flat ground) and STILL."):
        return
    gyro_bias_raw, a_level = average(link, 2.0)
    k_z = max(range(3), key=lambda i: abs(a_level[i]))
    z_raw = [0, 0, 0]
    z_raw[k_z] = 1 if a_level[k_z] > 0 else -1
    print(f"    raw accel {fmt(a_level)} → body z = {'+' if z_raw[k_z] > 0 else '-'}raw {'xyz'[k_z]}")
    if not 0.9 * G < abs(a_level[k_z]) < 1.1 * G:
        print(f"    ⚠ {abs(a_level[k_z]):.2f} m/s² is far from 9.81: the robot was moving or not level. Try again.")
        return
    if math.hypot(*[a for i, a in enumerate(a_level) if i != k_z]) > 0.25 * G:
        print("    ⚠ The board seems tilted on the robot (another axis also reads a lot of gravity).")
        print("      The board must be mounted flat (or exactly on its side). Check the mounting.")
        return

    # 2) nose up → body x
    if not wait_key("\n  STEP 2/3. Tilt the robot NOSE UP (head higher than tail) by about 45° and hold STILL."):
        return
    _, a_nose = average(link, 2.0)
    k_x = max((i for i in range(3) if i != k_z), key=lambda i: abs(a_nose[i]))
    x_raw = [0, 0, 0]
    x_raw[k_x] = 1 if a_nose[k_x] > 0 else -1
    tilt = math.degrees(math.asin(min(1.0, abs(a_nose[k_x]) / G)))
    print(f"    raw accel {fmt(a_nose)} → body x = {'+' if x_raw[k_x] > 0 else '-'}raw {'xyz'[k_x]}"
          f"  (tilt about {tilt:.0f}°)")
    if tilt < 20:
        print("    ⚠ Less than 20° of tilt: tilt more and run the wizard again.")
        return
    y_raw = cross(z_raw, x_raw)
    raw_to_body = [x_raw, [int(v) for v in y_raw], z_raw]
    print(f"    body y = z × x → raw {fmt(y_raw, 0)}")

    # 3) left turn → check gyro z sign
    bias_body = mat_vec(raw_to_body, gyro_bias_raw)
    cfg = {"raw_to_body": raw_to_body, "gyro_bias_rad_s": bias_body}
    if not wait_key("\n  STEP 3/3. Put it level again. After SPACE, turn the robot to ITS LEFT "
                    "(nose swings left, seen from above) by about 90° within 2 seconds."):
        return
    peak = 0.0
    t_end = time.perf_counter() + 2.5
    while time.perf_counter() < t_end:
        gyro, _ = to_body(link.read_raw(), cfg)
        if abs(gyro[2]) > abs(peak):
            peak = gyro[2]
        time.sleep(0.01)
    print(f"    peak body gyro z = {math.degrees(peak):+.0f} deg/s")
    if abs(math.degrees(peak)) < 30:
        print("    ⚠ Hardly any turn was measured. Turn faster/larger and run the wizard again.")
        return
    if peak < 0:
        print("    ⚠ Negative: either you turned RIGHT, or the board's axes are left-handed (should not")
        print("      happen with a BNO085). Turn left and run the wizard again.")
        return
    print("    ✔ left turn = +z, as in the sim")

    print(f"\n  raw → body: {raw_to_body}   gyro bias (body): {fmt([math.degrees(b) for b in bias_body], 2)} deg/s")
    print("  Save to hardware_imu.json? y = save, anything else = discard")
    if getch().lower() == "y":
        save_imu_config(raw_to_body, bias_body, "axis wizard: level, nose-up, left-turn")
        print(f"  Saved: {IMU_PATH}")
    else:
        print("  Not saved.")


MENU = """
  s : IMU status     l : live view (any key stops)     a : axis wizard (→ hardware_imu.json)     q : quit
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: fake level-and-still readings")
    args = parser.parse_args()
    if not args.port and not args.dry_run:
        parser.error("--port or --dry-run is required")

    link = ImuLink(args.port, dry_run=args.dry_run)
    try:
        print(f"  {link.status()}")
        print(MENU)
        while True:
            key = getch().lower()
            if key == "s":
                print(f"  {link.status()}")
            elif key == "l":
                live(link)
            elif key == "a":
                axis_wizard(link)
            elif key in ("q", "\x03"):
                break
            else:
                continue
            print(MENU)
    finally:
        link.close()


if __name__ == "__main__":
    main()
