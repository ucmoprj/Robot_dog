"""BNO085 IMU access for the real robot (course stage 03). Needs hardware/esp32_imu_bridge flashed.

- ImuLink: ServoLink (robot_io.py, unchanged) + the IMU commands of the esp32_imu_bridge firmware
- hardware_imu.json: how the IMU board is mounted (raw sensor axes → robot body axes) and the gyro bias,
  written by imu_test.py's axis wizard
- to_body(): raw sample → body frame, the frame the policy was trained with

Body frame = the sim's "imu" site in dog_mujoco/robot.xml: x toward the HEAD, y to the robot's LEFT, z UP.
The sim sensors the policy sees (train/env.py _get_obs):
    gyro  (rad/s)  : turning nose-left (seen from above) gives +z
    accel (m/s²)   : specific force, so flat and still reads (0, 0, +9.81); nose up gives +x
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from robot_io import ROOT, ServoLink

IMU_PATH = ROOT / "hardware_imu.json"
G = 9.80665


@dataclass
class ImuSample:
    gyro: list[float]   # rad/s
    accel: list[float]  # m/s², gravity included
    quat: list[float]   # w, x, y, z (game rotation vector, raw sensor frame)
    age_ms: int         # how old the newest gyro value is


class ImuLink(ServoLink):
    def status(self) -> str:
        if self.dry_run:
            return "IMU OK (dry-run)"
        self.ser.write(b"S\n")
        return self._readline()

    def read_raw(self) -> ImuSample:
        if self.dry_run:  # flat and still, sensor z up
            return ImuSample([0.0, 0.0, 0.0], [0.0, 0.0, G], [1.0, 0.0, 0.0, 0.0], 0)
        self.ser.write(b"I\n")
        reply = self._readline()
        parts = reply.split()
        if len(parts) != 12 or parts[0] != "I":
            raise RuntimeError(f"IMU read failed: {reply!r} (is esp32_imu_bridge flashed? try S for status)")
        v = [float(p) for p in parts[1:11]]
        return ImuSample(v[0:3], v[3:6], v[6:10], int(parts[11]))


# ---------- mounting (raw sensor axes → body axes) ----------

def load_imu_config() -> dict | None:
    if not IMU_PATH.exists():
        return None
    with open(IMU_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_imu_config(raw_to_body: list[list[int]], gyro_bias_body: list[float], note: str) -> None:
    cfg = {
        "model": "BNO085",
        "raw_to_body": raw_to_body,  # body = raw_to_body · raw  (rows: body x, y, z)
        "gyro_bias_rad_s": [round(b, 6) for b in gyro_bias_body],  # subtracted after rotating to body
        "measured": time.strftime("%Y-%m-%d %H:%M"),
        "note": note,
    }
    with open(IMU_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def mat_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[r][c] * v[c] for c in range(3)) for r in range(3)]


def to_body(sample: ImuSample, cfg: dict) -> tuple[list[float], list[float]]:
    """(gyro, accel) in the body frame, gyro bias removed."""
    m = cfg["raw_to_body"]
    gyro = [g - b for g, b in zip(mat_vec(m, sample.gyro), cfg["gyro_bias_rad_s"])]
    return gyro, mat_vec(m, sample.accel)
