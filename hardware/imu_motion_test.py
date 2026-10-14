"""Motion test + IMU (course stage 03): the same moves and keys as motion_test.py, with the BNO085 read on
every motion frame (50 Hz). Needs hardware/esp32_imu_bridge flashed (servo + IMU in one firmware).

motion_test.py and imu_link.py are imported unchanged; this file only adds the IMU on top:
  - every frame: send the 12 servo pulses (M), then read one IMU sample (I)
  - after every move: an IMU summary (tilt before → after, wobble, gyro peaks, |a|, IMU age)
  - lean / sit / bow poses: checks that the body tilted the way the leg geometry says it should
    (a quick test that hardware_imu.json maps the axes right: nose-up = +x tilt, left-side-up = +y tilt)
  - tilt stop: a move stops (holds the pose) if the body tilts more than TILT_STOP_DEG
  - every sample goes to a CSV file in imu_logs/ (open it in a spreadsheet, or plot it later)

Usage (from the project root, after imu_test.py's axis wizard has written hardware_imu.json):
    python hardware/imu_motion_test.py --port COM4
    python hardware/imu_motion_test.py --dry-run      # practice the keys; the IMU is faked (flat, still)

Keys: all of motion_test.py (1 2 3 4 i k j l b u p z x c v w e [ ] + - r q h), plus
    m : IMU live view while the robot holds its pose (any key stops)
    t : tilt stop on / off

Tilt (body axes, from accel, same formulas as imu_test.py's live view):
    nose-up      = atan2(accel x, accel z)   + = nose up
    left-side-up = atan2(accel y, accel z)   + = left side up
They are only right while the body is not being shaken; during a trot they show the wobble.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_link import IMU_PATH, ImuLink, load_imu_config, to_body  # noqa: E402
from motion_test import MENU, POSES, RATE_HZ, STRIDE, Keys, Robot, check_ready  # noqa: E402
from robot_io import ROOT, load_calibration  # noqa: E402

HIP_DX = 0.108          # m, front ↔ rear hip spacing (pitch estimate, as in motion_test.py)
HIP_DY = 0.080          # m, left ↔ right hip spacing (roll estimate)
TILT_STOP_DEG = 35.0    # a move stops if nose-up or left-side-up goes past this
CHECK_MIN_DEG = 2.0     # a pose check needs at least this much tilt in the right direction
STALL_MS = 100          # IMU age above this = the IMU stopped sending
LOG_DIR = ROOT / "imu_logs"

IMU_MENU = """  m : IMU live view (robot holds its pose)    t : tilt stop on/off (now {stop})
"""


@dataclass
class Frame:
    t: float            # s since the program started
    gyro: list[float]   # deg/s
    accel: list[float]  # m/s²
    nose: float         # deg, nose-up (nan without hardware_imu.json)
    left: float         # deg, left-side-up
    age: int            # ms


def tilts(accel: list[float]) -> tuple[float, float]:
    return math.degrees(math.atan2(accel[0], accel[2])), math.degrees(math.atan2(accel[1], accel[2]))


def expected_tilt(feet: dict[str, tuple]) -> tuple[float, float]:
    """Body tilt the leg lengths should give: longer legs lift that end / side."""
    d = {leg: feet[leg][1] for leg in feet}
    nose = math.degrees(math.atan2((d["fl"] + d["fr"]) / 2 - (d["bl"] + d["br"]) / 2, HIP_DX))
    left = math.degrees(math.atan2((d["fl"] + d["bl"]) / 2 - (d["fr"] + d["br"]) / 2, HIP_DY))
    return nose, left


class ImuRobot(Robot):
    def __init__(self, link: ImuLink, keys: Keys, cfg: dict | None, log):
        super().__init__(link, keys)
        self.cfg, self.log = cfg, log
        self.rec: list[Frame] | None = None   # frames of the move in progress
        self.motion = "idle"
        self.last: Frame | None = None
        self.ref = (0.0, 0.0)                 # nose-up, left-side-up measured while standing
        self.tilt_stop = cfg is not None
        self.t0 = time.perf_counter()

    # ---------- IMU ----------

    def read_imu(self) -> Frame | None:
        try:
            s = self.link.read_raw()
        except RuntimeError as e:
            print(f"  IMU read failed: {e}")
            return None
        if self.cfg is None:
            gyro, accel = s.gyro, s.accel
            nose = left = math.nan
        else:
            gyro, accel = to_body(s, self.cfg)
            nose, left = tilts(accel)
        f = Frame(time.perf_counter() - self.t0, [math.degrees(g) for g in gyro], list(accel), nose, left, s.age_ms)
        if self.rec is not None:
            self.rec.append(f)
        if self.log is not None:
            self.log.writerow([f"{f.t:.3f}", self.motion, *(f"{v:.2f}" for v in f.gyro),
                               *(f"{v:.3f}" for v in f.accel), f"{f.nose:.2f}", f"{f.left:.2f}", f.age])
        self.last = f
        return f

    def send(self, q: dict[str, float]) -> None:
        super().send(q)
        self.read_imu()

    def play(self, frames, live_keys: bool = False) -> bool:
        """Robot.play plus the tilt stop (checked after every frame)."""
        dt = 1 / RATE_HZ
        t_next = time.perf_counter()
        for q in frames:
            key = self.keys.hit()
            if key is not None and not (live_keys and self.adjust(key)):
                print("  STOPPED. Holding this pose.")
                return False
            self.send(q)
            if self.tipping():
                return False
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
        return True

    def tipping(self) -> bool:
        f = self.last
        if not self.tilt_stop or f is None or math.isnan(f.nose):
            return False
        if abs(f.nose) > TILT_STOP_DEG or abs(f.left) > TILT_STOP_DEG:
            print(f"  TILT STOP: nose-up {f.nose:+.0f}°, left-side-up {f.left:+.0f}° "
                  f"(limit ±{TILT_STOP_DEG:.0f}°). Holding this pose.")
            print("  Put the robot back on its feet, then 1 (stand) or r (relax). t turns the tilt stop off.")
            return True
        return False

    # ---------- one move, recorded ----------

    def run(self, key: str, name: str, move) -> None:
        start = self.read_imu()
        self.rec, self.motion = [], name
        ok = move()
        frames, self.rec, self.motion = self.rec, None, "idle"
        if frames:
            self.summary(key, name, start or frames[0], frames, ok)
            if ok and key == "1" and self.cfg is not None:
                tail = frames[-10:]
                self.ref = (statistics.fmean(f.nose for f in tail), statistics.fmean(f.left for f in tail))

    def summary(self, key: str, name: str, start: Frame, fr: list[Frame], ok: bool) -> None:
        print(f"  ── IMU: {name} ({'done' if ok else 'stopped'}) · {fr[-1].t - fr[0].t:.1f} s, {len(fr)} samples ──")
        if self.cfg is not None:
            for label, get in (("nose-up", lambda f: f.nose), ("left-side-up", lambda f: f.left)):
                vals = [get(f) for f in fr]
                end = statistics.fmean(vals[-10:])
                print(f"     {label:<13}{get(start):+6.1f}° → {end:+6.1f}°   range {min(vals):+.1f} .. {max(vals):+.1f}°"
                      f"   wobble (sd) {statistics.pstdev(vals):.1f}°")
        peak = [max(abs(f.gyro[i]) for f in fr) for i in range(3)]
        mags = [math.hypot(*f.accel) for f in fr]
        print(f"     gyro peak    x {peak[0]:4.0f}   y {peak[1]:4.0f}   z {peak[2]:4.0f} deg/s"
              f"{'' if self.cfg else '   (RAW sensor axes)'}")
        print(f"     |a|          {min(mags):.2f} .. {max(mags):.2f} m/s²")
        age = max(f.age for f in fr)
        print(f"     IMU age max  {age} ms" + ("   ← the IMU stopped sending for a while!" if age > STALL_MS else ""))
        if ok and key in POSES:
            self.check(key, fr)

    def check(self, key: str, fr: list[Frame]) -> None:
        if self.cfg is None:
            return
        if self.link.dry_run:
            print("     check: skipped (dry-run, the IMU is fake)")
            return
        exp_nose, exp_left = expected_tilt(POSES[key][1])
        got_nose = statistics.fmean(f.nose for f in fr[-10:]) - self.ref[0]
        got_left = statistics.fmean(f.left for f in fr[-10:]) - self.ref[1]
        bad = False
        for label, exp, got in (("nose-up", exp_nose, got_nose), ("left-side-up", exp_left, got_left)):
            if abs(exp) < 1.0:
                continue
            good = got * exp > 0 and abs(got) >= CHECK_MIN_DEG
            bad |= not good
            print(f"     check {label} vs standing: {got:+.1f}°  (leg geometry ≈ {exp:+.1f}°)  {'✔' if good else '✖'}")
        if bad:
            print("     ✖ = wrong direction or too small. Did the robot stand (1) first? Is the IMU loose?"
                  " If the sign is wrong, run imu_test.py's axis wizard (a) again.")

    # ---------- live view ----------

    def monitor(self) -> None:
        frame = "body axes (x HEAD, y LEFT, z UP)" if self.cfg else "RAW sensor axes (no hardware_imu.json)"
        print(f"  IMU live, {frame}. The robot holds its pose. Any key stops.")
        t_next = time.perf_counter()
        for i in range(10 ** 9):
            if self.keys.hit() is not None or (not self.keys.tty and i >= 5):
                break
            f = self.read_imu()
            if f is not None:
                tilt = "" if math.isnan(f.nose) else f"  nose-up {f.nose:+5.1f}°  left-side-up {f.left:+5.1f}°"
                g = ", ".join(f"{v:+6.1f}" for v in f.gyro)
                a = ", ".join(f"{v:+6.2f}" for v in f.accel)
                print(f"  gyro ({g}) deg/s  accel ({a}) |a|={math.hypot(*f.accel):.2f}{tilt}  age {f.age} ms")
            t_next += 0.1
            time.sleep(max(0.0, t_next - time.perf_counter()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: shows what would be sent")
    parser.add_argument("--no-log", action="store_true", help="do not write the CSV file")
    args = parser.parse_args()
    if not args.port and not args.dry_run:
        parser.error("--port or --dry-run is required")
    sys.stdout.reconfigure(errors="replace")  # a console without ° → ✔ prints '?' instead of crashing

    if not check_ready(load_calibration()):
        sys.exit(1)
    cfg = load_imu_config()
    if cfg is None:
        print(f"No {IMU_PATH.name} yet: the IMU is shown in RAW sensor axes, with no tilt, tilt stop or checks.")
        print("Run the axis wizard first for the full test:  python hardware/imu_test.py --port COM4  → a\n")
    else:
        print(f"IMU body axes from {IMU_PATH.name} (measured {cfg['measured']}).")

    link = ImuLink(args.port, dry_run=args.dry_run)
    status = link.status()
    if not status.startswith("IMU OK"):
        link.close()
        hint = ("the ESP32 has the servo-only firmware: flash hardware/esp32_imu_bridge"
                if "unknown command" in status else "check the BNO085 wiring (3V3, GND, SDA→GPIO21, SCL→GPIO22)")
        sys.exit(f"IMU status: {status!r} → {hint}")
    print(f"IMU status: {status}")

    log_file = log = None
    if not args.no_log:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / time.strftime("imu_motion_%Y%m%d_%H%M%S.csv")
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        log.writerow(["t_s", "motion", "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s", "accel_x", "accel_y",
                      "accel_z", "nose_up_deg", "left_side_up_deg", "age_ms"])
        print(f"Logging every IMU sample to {log_path.relative_to(ROOT)}")

    def menu(robot: ImuRobot) -> None:
        print(MENU.format(n=robot.walk_steps, t=robot.step_time) + IMU_MENU.format(stop="on" if robot.tilt_stop else "off"))

    try:
        with Keys() as keys:
            robot = ImuRobot(link, keys, cfg, log)
            moves = {
                "1": ("stand", robot.stand),
                "2": ("crouch", robot.sit),
                "b": ("bounce", robot.bounce),
                "u": ("hop", robot.hop),
                "p": ("paw wave", robot.paw_wave),
                "w": ("trot forward", lambda: robot.trot(STRIDE)),
                "e": ("trot in place", lambda: robot.trot(0.0)),
                **{k: (POSES[k][0], (lambda k=k: robot.pose(k))) for k in POSES},
            }
            menu(robot)
            while True:
                key = keys.get().lower()
                if key in moves:
                    name, move = moves[key]
                    robot.run(key, name, move)
                elif key in "zxcv":
                    robot.adjust_paw(key)
                    continue
                elif robot.adjust(key):
                    continue
                elif key == "m":
                    robot.monitor()
                    continue
                elif key == "t":
                    if cfg is None:
                        print("  Tilt stop needs hardware_imu.json (run the axis wizard first).")
                    else:
                        robot.tilt_stop = not robot.tilt_stop
                        print(f"  tilt stop: {'on' if robot.tilt_stop else 'off'} (±{TILT_STOP_DEG:.0f}°)")
                    continue
                elif key == "r":
                    robot.relax()
                elif key in ("q", "\x03"):
                    break
                elif key == "h":
                    menu(robot)
                    continue
                else:
                    continue
                if robot.q is not None and link.dry_run:
                    print(f"  [dry-run] now: {robot.to_us(robot.q)}")
    finally:
        link.close()
        if log_file is not None:
            log_file.close()
        print("All servos relaxed. Bye.")


if __name__ == "__main__":
    main()
