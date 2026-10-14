"""IMU feedback motion test (course stage 03): the motion_test.py moves, now CORRECTED by the IMU.

imu_motion_test.py only watches the IMU. Here the IMU closes the loop: every frame (50 Hz) the body tilt is
measured and the leg lengths are changed so the body keeps the tilt the move intends:
  - stand / crouch / bounce / trot: body kept LEVEL, also on a tilted floor
  - lean i k j l, dog sit 3, play bow 4: body kept at the lean the legs are designed for
  - paw wave p: the three standing legs balance, the waving paw is left alone
  - between commands the robot keeps balancing (tilt the board it stands on and watch)
Nothing in motion_test.py, imu_motion_test.py or imu_link.py is changed; this file builds on them.

Usage (from the project root; needs hardware_imu.json from imu_test.py's axis wizard):
    python hardware/imu_feedback_test.py --port COM4
    python hardware/imu_feedback_test.py --dry-run     # practice the keys (fake IMU, nothing moves)

Start like this (feedback is OFF at start):
    1  stand, on FLAT ground
    0  "this is level": remembers the tilt the IMU reads now (absorbs a slightly crooked mount)
    f  feedback on      then any move, or tilt the ground board by hand and watch the legs
Keys: all of motion_test.py (1 2 3 4 i k j l b u p z x c v w e [ ] + - r q h), plus
    f : feedback on / off          0 : set level reference (robot standing on flat ground)
    n / o : weaker / stronger feedback (gain x0.7 / x1.4)
    m : live view of tilt, target and correction (keeps balancing; any key stops)
    t : tilt stop on / off (from imu_motion_test.py, ±35°)

The controller (one per axis, pitch = nose-up, roll = left-side-up, degrees):
    error      = measured tilt - level reference - tilt the commanded legs should give
    correction = KP*error + KI*(error added up over time) + KD*(tilt rate from the gyro)
    The correction tilts the body the other way by changing leg depths:
    pitch: front legs - HIP_DX/2*tan(c), rear legs + HIP_DX/2*tan(c)     roll: left/right with HIP_DY
    It is limited to ±MAX_CORR_DEG, and the legs to D_MIN..D_MAX.
Tilt comes from the BNO085's own fusion (game rotation vector: gyro + accel), which stays steady while
walking; if it does not match the accel when 0 is pressed, the accel (smoothed) is used instead.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_link import IMU_PATH, ImuLink, ImuSample, load_imu_config, mat_vec  # noqa: E402
from imu_motion_test import HIP_DX, HIP_DY, TILT_STOP_DEG, ImuRobot, tilts  # noqa: E402
from motion_test import LEGS, MENU, POSES, RATE_HZ, STRIDE, Keys, check_ready, leg_fk, leg_ik  # noqa: E402
from robot_io import load_calibration  # noqa: E402

KP = 0.3               # correction per degree of error
KI = 1.0               # 1/s: removes a steady error (tilted floor) in 1-2 s
KD = 0.005             # s: damping from the gyro rate. Keep small: at 50 Hz a 1° step in one frame is 50°/s,
                       # and 0.03 (tried in simulation) made the loop swing between the ±10° limits
MAX_CORR_DEG = 10.0    # largest body tilt correction
DEADBAND_DEG = 0.5     # errors smaller than this are ignored (no jitter while standing)
GAIN_RANGE = (0.25, 3.0)
D_MIN, D_MAX = 0.056, 0.081  # m, corrected foot depth limits (0.082 = leg fully straight)
MATCH_DEG = 5.0        # fused tilt must agree with the accel this well when 0 is pressed
ACCEL_SMOOTH = 0.15    # accel fallback: low-pass factor per frame

FEEDBACK_MENU = """  f : feedback on/off (now {fb})   0 : level reference (now {ref})   n / o : weaker / stronger (gain x{gain:.2f})
  m : live tilt / target / correction view     t : tilt stop on/off (now {stop})
  Start: 1 (stand on flat ground) → 0 → f
"""


# ---------- small vector helpers ----------

def qrot(q: list[float], v: list[float]) -> list[float]:
    """Rotate v by the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]


def up_tilt(up: list[float]) -> tuple[float, float]:
    """(nose-up, left-side-up) in degrees from the body-frame 'up' direction."""
    return math.degrees(math.atan2(up[0], up[2])), math.degrees(math.atan2(up[1], up[2]))


def group_tilt(feet: dict[str, tuple[float, float]], legs: set[str]) -> tuple[float, float]:
    """Body tilt the leg depths should give, using only the legs in 'legs' (the ones on the ground)."""
    def mean(group):
        ds = [feet[g][1] for g in group if g in legs]
        return sum(ds) / len(ds) if ds else None
    front, back, left, right = mean(("fl", "fr")), mean(("bl", "br")), mean(("fl", "bl")), mean(("fr", "br"))
    nose = math.degrees(math.atan2(front - back, HIP_DX)) if front is not None and back is not None else 0.0
    side = math.degrees(math.atan2(left - right, HIP_DY)) if left is not None and right is not None else 0.0
    return nose, side


class RecordingLink(ImuLink):
    """ImuLink that keeps the last raw sample (the feedback needs its quaternion)."""

    last: ImuSample | None = None

    def read_raw(self) -> ImuSample:
        self.last = super().read_raw()
        return self.last


class Axis:
    """PI + gyro-damping controller for one tilt axis (degrees)."""

    def __init__(self):
        self.i = 0.0

    def reset(self) -> None:
        self.i = 0.0

    def step(self, error: float, rate: float, dt: float, gain: float) -> float:
        if abs(error) < DEADBAND_DEG:
            error = 0.0
        ki = KI * gain
        if ki > 0:  # integrator limited so it alone can never ask for more than the max correction
            lim = MAX_CORR_DEG / ki
            self.i = max(-lim, min(lim, self.i + error * dt))
        c = gain * (KP * error + KD * rate) + ki * self.i
        return max(-MAX_CORR_DEG, min(MAX_CORR_DEG, c))


class FeedbackRobot(ImuRobot):
    def __init__(self, link: RecordingLink, keys: Keys, cfg: dict, log):
        super().__init__(link, keys, cfg, log)
        self.feedback = False
        self.level: tuple[float, float] | None = None  # (nose-up, left-up) read standing on flat ground
        self.gain = 1.0
        self.pitch, self.roll = Axis(), Axis()
        self.source = "fused"           # "fused" (BNO085 game rotation vector) or "accel"
        self.quat_flip = False          # which way the quaternion turns sensor → world (chosen at 0)
        self.smooth: tuple[float, float] | None = None
        self.lifted: set[str] = set()   # legs left out of the feedback (the waving paw)
        self.t_prev: float | None = None
        self.corr = (0.0, 0.0)          # last correction (pitch, roll), deg
        self.target = (0.0, 0.0)
        self.tilt = (0.0, 0.0)
        self.peak = [0.0, 0.0]

    # ---------- measuring the tilt ----------

    def fused_tilt(self, flip: bool | None = None) -> tuple[float, float] | None:
        s = self.link.last
        if s is None:
            return None
        q = s.quat if (self.quat_flip if flip is None else flip) else [s.quat[0], -s.quat[1], -s.quat[2], -s.quat[3]]
        return up_tilt(mat_vec(self.cfg["raw_to_body"], qrot(q, [0.0, 0.0, 1.0])))

    def measured_tilt(self, f) -> tuple[float, float] | None:
        if f is None:
            return None
        if self.source == "fused":
            return self.fused_tilt()
        a = (f.nose, f.left)
        if self.smooth is None:
            self.smooth = a
        self.smooth = tuple(s + ACCEL_SMOOTH * (v - s) for s, v in zip(self.smooth, a))
        return self.smooth

    # ---------- feedback ----------

    def send(self, q: dict[str, float]) -> None:
        """Read the IMU, correct the commanded pose, send it. self.q keeps the UNcorrected command, so the
        next move starts from what was commanded and the correction is never counted twice."""
        f = self.read_imu()
        q_out = self.correct(q, f)
        if not self.link.dry_run:
            self.link.set_all_us(self.to_us(q_out))
        self.q = q

    def correct(self, q: dict[str, float], f) -> dict[str, float]:
        now = time.perf_counter()
        dt = min(0.1, now - self.t_prev) if self.t_prev is not None else 1 / RATE_HZ
        self.t_prev = now
        tilt = self.measured_tilt(f)
        if not self.feedback or self.level is None or tilt is None:
            self.corr = (0.0, 0.0)
            return q
        feet = {leg: leg_fk(q[f"{leg}_2"], q[f"{leg}_3"]) for leg in LEGS}
        ground = set(LEGS) - self.lifted
        self.target = group_tilt(feet, ground)
        self.tilt = (tilt[0] - self.level[0], tilt[1] - self.level[1])
        nose_rate, left_rate = -f.gyro[1], f.gyro[0]  # deg/s: turning about +y lowers the nose, about +x lifts the left
        cp = self.pitch.step(self.tilt[0] - self.target[0], nose_rate, dt, self.gain)
        cr = self.roll.step(self.tilt[1] - self.target[1], left_rate, dt, self.gain)
        self.corr = (cp, cr)
        self.peak = [max(self.peak[0], abs(cp)), max(self.peak[1], abs(cr))]
        dp = HIP_DX / 2 * math.tan(math.radians(cp))  # + = nose too high → shorter front, longer rear legs
        dr = HIP_DY / 2 * math.tan(math.radians(cr))  # + = left too high → shorter left, longer right legs
        out = dict(q)
        for leg in ground:
            x, d = feet[leg]
            d += (-dp if leg[0] == "f" else dp) + (-dr if leg[1] == "l" else dr)
            out[f"{leg}_2"], out[f"{leg}_3"] = leg_ik(x, max(D_MIN, min(D_MAX, d)))
        return out

    def balance_step(self) -> bool:
        """One frame of balancing while no move runs. False if the tilt stop fired."""
        if self.q is None or not self.feedback:
            return True
        self.send(dict(self.q))
        return not self.tipping()

    # ---------- keys ----------

    def set_reference(self) -> None:
        f = self.read_imu()
        if f is None:
            return
        accel = (f.nose, f.left)
        cands = {flip: self.fused_tilt(flip) for flip in (False, True)}
        err = {flip: max(abs(c[0] - accel[0]), abs(c[1] - accel[1])) for flip, c in cands.items()}
        best = min(err, key=err.get)
        if err[best] <= MATCH_DEG:
            self.source, self.quat_flip = "fused", best
            tilt = cands[best]
        else:
            self.source, self.smooth = "accel", None
            tilt = accel
            print(f"  (fused tilt and accel disagree by {err[best]:.0f}°: using the accel, smoothed)")
        # the level reference is the tilt read now minus the tilt the current legs are meant to give
        if self.q is not None:
            feet = {leg: leg_fk(self.q[f"{leg}_2"], self.q[f"{leg}_3"]) for leg in LEGS}
            meant = group_tilt(feet, set(LEGS))
        else:
            meant = (0.0, 0.0)
        self.level = (tilt[0] - meant[0], tilt[1] - meant[1])
        self.pitch.reset(); self.roll.reset()
        print(f"  Level reference: nose-up {self.level[0]:+.1f}°, left-side-up {self.level[1]:+.1f}° "
              f"(source: {self.source}). Now f turns the feedback on.")

    def toggle_feedback(self) -> None:
        if self.q is None:
            print("  Stand first (1): feedback works on a standing robot.")
            return
        if not self.feedback and self.level is None:
            print("  No level reference yet: taking it now. (Better: stand on FLAT ground and press 0 first.)")
            self.set_reference()
        self.feedback = not self.feedback
        self.pitch.reset(); self.roll.reset()
        print(f"  feedback: {'ON' if self.feedback else 'OFF'}  (gain x{self.gain:.2f})")

    def change_gain(self, key: str) -> None:
        lo, hi = GAIN_RANGE
        self.gain = round(min(hi, max(lo, self.gain * (1.4 if key == "o" else 0.7))), 2)
        print(f"  feedback gain x{self.gain:.2f}  (n weaker, o stronger; shaking = too strong)")

    def relax(self) -> None:
        super().relax()
        self.pitch.reset(); self.roll.reset()

    def paw_wave(self) -> bool:
        self.lifted = {"fr"}  # the waving paw is left out; the other three legs balance
        try:
            return super().paw_wave()
        finally:
            self.lifted = set()

    def run(self, key: str, name: str, move) -> None:
        self.peak = [0.0, 0.0]
        super().run(key, name, move)
        if self.feedback:
            print(f"     feedback     largest correction: pitch {self.peak[0]:.1f}°, roll {self.peak[1]:.1f}°"
                  f"{'   ← at the limit: the floor is too steep or the gain too low' if max(self.peak) >= MAX_CORR_DEG - 0.1 else ''}")

    def monitor(self) -> None:
        print("  Live: tilt vs level reference, target from the legs, correction. Keeps balancing. Any key stops.")
        if not self.feedback:
            print("  (feedback is OFF: the correction stays 0)")
        t_next = time.perf_counter()
        for i in range(10 ** 9):
            if self.keys.hit() is not None or (not self.keys.tty and i >= 25):
                break
            if self.q is not None and self.feedback:
                if not self.balance_step():
                    break
            else:
                f = self.read_imu()
                t = self.measured_tilt(f)
                if t is not None and self.level is not None:
                    self.tilt = (t[0] - self.level[0], t[1] - self.level[1])
            if i % 5 == 0:
                print(f"  nose-up {self.tilt[0]:+5.1f}° (target {self.target[0]:+5.1f})  "
                      f"left-up {self.tilt[1]:+5.1f}° (target {self.target[1]:+5.1f})  "
                      f"correction pitch {self.corr[0]:+5.1f}° roll {self.corr[1]:+5.1f}°")
            t_next += 1 / RATE_HZ
            time.sleep(max(0.0, t_next - time.perf_counter()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: fake IMU, nothing moves")
    args = parser.parse_args()
    if not args.port and not args.dry_run:
        parser.error("--port or --dry-run is required")
    sys.stdout.reconfigure(errors="replace")

    if not check_ready(load_calibration()):
        sys.exit(1)
    cfg = load_imu_config()
    if cfg is None:
        sys.exit(f"No {IMU_PATH.name}: run the axis wizard first:  python hardware/imu_test.py --port COM4  → a")
    print(f"IMU body axes from {IMU_PATH.name} (measured {cfg['measured']}).")

    link = RecordingLink(args.port, dry_run=args.dry_run)
    status = link.status()
    if not status.startswith("IMU OK"):
        link.close()
        sys.exit(f"IMU status: {status!r}. Flash hardware/esp32_imu_bridge and check the BNO085 wiring.")
    print(f"IMU status: {status}")

    def menu(robot: FeedbackRobot) -> None:
        ref = "not set" if robot.level is None else f"{robot.level[0]:+.1f}°/{robot.level[1]:+.1f}°"
        print(MENU.format(n=robot.walk_steps, t=robot.step_time) + FEEDBACK_MENU.format(
            fb="ON" if robot.feedback else "OFF", ref=ref, gain=robot.gain, stop="on" if robot.tilt_stop else "off"))

    try:
        with Keys() as keys:
            robot = FeedbackRobot(link, keys, cfg, None)
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
            t_next = time.perf_counter()
            while True:
                key = keys.hit() if keys.tty else keys.get()
                if key is None:  # no key: keep balancing at 50 Hz
                    if not robot.balance_step():
                        robot.feedback = False
                        print("  feedback OFF after the tilt stop. f turns it on again.")
                    t_next = max(t_next + 1 / RATE_HZ, time.perf_counter() - 0.1)
                    time.sleep(max(0.0, t_next - time.perf_counter()))
                    continue
                key = key.lower()
                if key in moves:
                    name, move = moves[key]
                    robot.run(key, name, move)
                elif key in "zxcv":
                    robot.adjust_paw(key)
                elif robot.adjust(key):
                    pass
                elif key == "f":
                    robot.toggle_feedback()
                elif key == "0":
                    robot.set_reference()
                elif key in "no":
                    robot.change_gain(key)
                elif key == "m":
                    robot.monitor()
                elif key == "t":
                    robot.tilt_stop = not robot.tilt_stop
                    print(f"  tilt stop: {'on' if robot.tilt_stop else 'off'} (±{TILT_STOP_DEG:.0f}°)")
                elif key == "r":
                    robot.relax()
                    robot.feedback = False
                elif key in ("q", "\x03"):
                    break
                elif key == "h":
                    menu(robot)
                t_next = time.perf_counter()
    finally:
        link.close()
        print("All servos relaxed. Bye.")


if __name__ == "__main__":
    main()
