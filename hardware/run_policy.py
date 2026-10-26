"""Run a trained walking policy on the real robot (course stage 04). Needs hardware/esp32_imu_bridge flashed,
hardware_calibration.json (stage 01) and hardware_imu.json (stage 03 axis wizard).

The policy comes from train/export_policy.py (a .npz file: network weights + task settings), so this PC
needs only numpy + pyserial. Every 20 ms (50 Hz, the sim's control step) it does what
train/env_obstacle.py does in the simulation:
    1. read the BNO085 (I) and rotate it to body axes (imu_link.to_body)
    2. build the 35 inputs, in this order:
         gyro 3 (rad/s) | accel 3 (m/s², gravity included) | last raw action 12 | the one before 12 |
         gait clock [sin 2πφ, cos 2πφ] | yaw error from the start heading [sin, cos] | vx_cmd × 10
    3. run the network → 12 raw actions
    4. target = stand + scale × clip(raw), at most max_command_rate × 20 ms from the last target,
       clamped to the joint range → rad_to_us() → M command (12 pulses)
    5. advance the clock: φ += f(vx_cmd) × 20 ms
The yaw error uses the game rotation vector (no magnetometer): yaw at the moment you press g = heading 0.

Usage (from the project root):
    python hardware/run_policy.py --policy policy_stable.npz --port COM4
    python hardware/run_policy.py --policy policy_stable.npz --dry-run      # no hardware, fake flat IMU

Keys:
    g : go (start the policy from the standing pose; the current heading becomes "straight ahead")
    w / s : faster / slower (vx_cmd ± 0.01 m/s, within the training range; works while walking)
    1 : stand             r : relax all servos (the robot drops!)       q : quit (servos relax)
    While walking, any other key stops the policy and the robot goes back to the standing pose.
Safety: the policy stops by itself if the body tilts more than --tilt-stop degrees, if the IMU sample is
older than 100 ms, or after --seconds. Every step goes to imu_logs/policy_*.csv.

First runs: on the calibration stand (feet off the ground), then on the floor with a hand ready to catch.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_link import IMU_PATH, ImuLink, load_imu_config, to_body  # noqa: E402
from motion_test import Keys, Robot, check_ready  # noqa: E402
from robot_io import JOINT_ORDER, ROOT, channel_cfg, load_calibration, rad_to_us  # noqa: E402

LOG_DIR = ROOT / "imu_logs"
VX_KEY_STEP = 0.01      # m/s per w / s press
MAX_IMU_AGE_MS = 100    # older gyro sample = the IMU stalled, stop
SLOW_STEP_MS = 25       # a loop longer than this is counted as late


class Policy:
    """The exported actor (train/export_policy.py) + the task settings that go with it."""

    def __init__(self, path: str):
        d = np.load(path)
        n = int(d["n_layers"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        self.names = [str(s) for s in d["joint_names"]]
        if self.names != JOINT_ORDER:
            raise ValueError(f"joint order {self.names} != robot_io.JOINT_ORDER {JOINT_ORDER}")
        self.stand = d["stand_rad"]
        self.scale = d["action_scale"]
        self.clip = float(d["action_clip"])
        self.dt = float(d["control_dt"])
        rate = float(d["max_command_rate"])
        self.max_step = rate * self.dt if rate > 0 else None
        self.lo, self.hi = d["ctrlrange"][:, 0], d["ctrlrange"][:, 1]
        self.freq = d["frequency_hz"]
        self.vx_range = tuple(float(v) for v in d["vx_range"])
        self.vx_play = float(d["vx_play"])
        self.vx_obs_scale = float(d["vx_obs_scale"])
        self.n_obs = int(d["n_obs"])
        self.task, self.checkpoint = str(d["task"]), str(d["checkpoint"])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))  # ELU
        return x

    def clock_hz(self, vx: float) -> float:
        """env_obstacle.clock_frequency: fixed, or linear over the command range."""
        if self.freq.size == 1:
            return float(self.freq[0])
        lo, hi = self.vx_range
        t = min(max((vx - lo) / (hi - lo), 0.0), 1.0)
        return float(self.freq[0] + t * (self.freq[1] - self.freq[0]))

    def target(self, raw: np.ndarray, prev_target: np.ndarray) -> np.ndarray:
        """env_obstacle.action_to_target."""
        t = self.stand + self.scale * np.clip(raw, -self.clip, self.clip)
        if self.max_step is not None:
            t = prev_target + np.clip(t - prev_target, -self.max_step, self.max_step)
        return np.clip(t, self.lo, self.hi)


def quat_to_mat(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def body_yaw_tilt(quat: list[float], raw_to_body: list[list[float]]) -> tuple[float, float]:
    """Yaw (rad) and tilt from upright (deg) of the BODY, from the raw-frame game rotation vector.
    world←raw is R(quat); raw←body is raw_to_bodyᵀ, so world←body = R · raw_to_bodyᵀ."""
    r = quat_to_mat(quat) @ np.array(raw_to_body, dtype=np.float64).T
    yaw = math.atan2(r[1, 0], r[0, 0])
    tilt = math.degrees(math.acos(min(max(r[2, 2], -1.0), 1.0)))
    return yaw, tilt


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Runner:
    def __init__(self, robot: Robot, link: ImuLink, policy: Policy, imu_cfg: dict, args):
        self.robot, self.link, self.p, self.imu_cfg, self.args = robot, link, policy, imu_cfg, args
        self.calib = load_calibration()
        self.channels = [channel_cfg(self.calib, j) for j in JOINT_ORDER]
        self.vx = policy.vx_play if args.vx is None else args.vx
        self.set_vx(self.vx, quiet=True)

    def set_vx(self, vx: float, quiet: bool = False) -> None:
        lo, hi = self.p.vx_range
        self.vx = round(min(max(vx, lo), hi), 3)
        if not quiet:
            print(f"  vx_cmd {self.vx:.3f} m/s  (clock {self.p.clock_hz(self.vx):.2f} Hz)")

    def send(self, target: np.ndarray) -> None:
        us = [rad_to_us(ch, float(q)) for ch, q in zip(self.channels, target)]
        if not self.link.dry_run:
            self.link.set_all_us(us)
        self.robot.q = dict(zip(JOINT_ORDER, (float(q) for q in target)))

    def walk(self) -> None:
        p, link, cfg = self.p, self.link, self.imu_cfg
        if not self.robot.stand():  # wakes the servos first if they are relaxed
            return
        yaw0, _ = body_yaw_tilt(link.read_raw().quat, cfg["raw_to_body"])
        hist = np.zeros((2, 12))            # [last raw, the one before], zeros = standing (as at sim reset)
        prev_target = p.stand.copy()
        phase = 0.0
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / time.strftime("policy_%Y%m%d_%H%M%S.csv")
        n_steps = int(self.args.seconds / p.dt) if self.args.seconds > 0 else None
        late = 0
        reason = "key"
        print(f"  WALKING ({p.task}). w/s = speed, any other key = stop.  log: {log_path.name}")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            log = csv.writer(f)
            log.writerow(["t", "loop_ms", "vx_cmd", "phase", "yaw_err_deg", "tilt_deg", "imu_age_ms"]
                         + [f"obs{i}" for i in range(p.n_obs)] + [f"raw_{j}" for j in JOINT_ORDER]
                         + [f"q_{j}" for j in JOINT_ORDER])
            t0 = t_next = time.perf_counter()
            step = 0
            while n_steps is None or step < n_steps:
                t_loop = time.perf_counter()
                key = self.robot.keys.hit()
                if key is not None:
                    if key.lower() in "ws":
                        self.set_vx(self.vx + (VX_KEY_STEP if key.lower() == "w" else -VX_KEY_STEP))
                    else:
                        break

                s = link.read_raw()
                gyro, accel = to_body(s, cfg)
                yaw, tilt = body_yaw_tilt(s.quat, cfg["raw_to_body"])
                err = wrap(yaw - yaw0)
                if tilt > self.args.tilt_stop:
                    reason = f"tilt {tilt:.0f} deg > {self.args.tilt_stop:.0f}"
                    break
                if s.age_ms > MAX_IMU_AGE_MS:
                    reason = f"IMU sample {s.age_ms} ms old"
                    break

                obs = np.concatenate([
                    gyro, accel, hist.reshape(-1),
                    [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
                    [math.sin(err), math.cos(err)],
                    [self.vx * p.vx_obs_scale],
                ])
                raw = np.clip(p(obs), -p.clip, p.clip)
                target = p.target(raw, prev_target)
                self.send(target)

                hist[1], hist[0] = hist[0], raw
                prev_target = target
                phase = (phase + p.clock_hz(self.vx) * p.dt) % 1.0

                loop_ms = (time.perf_counter() - t_loop) * 1000
                late += loop_ms > SLOW_STEP_MS
                log.writerow([f"{t_loop - t0:.3f}", f"{loop_ms:.1f}", self.vx, f"{phase:.3f}",
                              f"{math.degrees(err):.1f}", f"{tilt:.1f}", s.age_ms]
                             + [f"{v:.4f}" for v in obs] + [f"{v:.3f}" for v in raw]
                             + [f"{v:.4f}" for v in target])
                step += 1
                if step % 50 == 0:
                    print(f"  t {step * p.dt:5.1f} s  vx_cmd {self.vx:.3f}  yaw err {math.degrees(err):+5.1f} deg  "
                          f"tilt {tilt:4.1f} deg  loop {loop_ms:4.1f} ms  late {late}")
                t_next += p.dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
            else:
                reason = f"{self.args.seconds:g} s done"
        print(f"  STOPPED ({reason}) after {step * p.dt:.1f} s, {late} late steps (> {SLOW_STEP_MS} ms). "
              "Back to standing.")
        self.robot.stand()


MENU = """
  g : go (walk with the policy)       w / s : faster / slower (now {vx:.3f} m/s, range {lo:.2f}-{hi:.2f})
  1 : stand      r : relax all servos (the robot drops!)      q : quit (servos relax)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help=".npz from train/export_policy.py")
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM4, /dev/ttyUSB0)")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: fake flat IMU, prints the commands")
    parser.add_argument("--vx", type=float, default=None, help="starting speed command, m/s (default: the task's play value)")
    parser.add_argument("--seconds", type=float, default=0, help="stop walking after this many seconds (0 = until a key)")
    parser.add_argument("--tilt-stop", type=float, default=35.0, help="stop if the body tilts more than this, deg")
    args = parser.parse_args()
    if not args.port and not args.dry_run:
        parser.error("--port or --dry-run is required")

    if not check_ready(load_calibration()):
        sys.exit(1)
    imu_cfg = load_imu_config()
    if imu_cfg is None:
        sys.exit(f"No {IMU_PATH.name}: run hardware/imu_test.py and its axis wizard (a) first.")
    policy = Policy(args.policy)
    print(f"Policy: {policy.task} ({policy.checkpoint}), {policy.n_obs} inputs, "
          f"max {policy.max_step / policy.dt if policy.max_step else 0:.1f} rad/s per joint")

    link = ImuLink(args.port, dry_run=args.dry_run)
    try:
        print(link.status())
        with Keys() as keys:
            robot = Robot(link, keys)
            runner = Runner(robot, link, policy, imu_cfg, args)
            robot.stand()
            while True:
                print(MENU.format(vx=runner.vx, lo=policy.vx_range[0], hi=policy.vx_range[1]))
                key = keys.get().lower()
                if key == "g":
                    runner.walk()
                elif key in "ws":
                    runner.set_vx(runner.vx + (VX_KEY_STEP if key == "w" else -VX_KEY_STEP))
                elif key == "1":
                    robot.stand()
                elif key == "r":
                    link.relax_all() if not link.dry_run else print("  [dry-run] X")
                    robot.q = None
                    print("  Relaxed.")
                elif key == "q":
                    break
    finally:
        link.close()


if __name__ == "__main__":
    main()
