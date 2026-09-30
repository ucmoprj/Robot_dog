"""Shared I/O module for the real robot.

- Joint specs (range, standing pose): onshape/onshape_mates.csv (same values as dog_mujoco/robot.xml)
- Calibration: hardware_calibration.json (starts from the template if missing)
- Sim joint angle (rad) → PCA9685 pulse (µs) conversion
- ESP32 serial link (pairs with the hardware/esp32_servo_bridge firmware)

Conversion (policy output q is the same sim-frame angle as data.ctrl in train/env.py):
    us = center_us + direction * us_per_rad * (q - zero_rad),  clamped to [min_us, max_us]
"""
from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "hardware_calibration.template.json"
CALIB_PATH = ROOT / "hardware_calibration.json"
MATES_PATH = ROOT / "onshape" / "onshape_mates.csv"

# PCA9685 channel order = robot.xml actuator order = policy output order
JOINT_ORDER = [f"{leg}_{i}" for leg in ("fl", "fr", "bl", "br") for i in (1, 2, 3)]
US_ABS_MIN, US_ABS_MAX = 500, 2500  # same absolute safety range as the firmware
REQUIRED_FIELDS = ("zero_rad", "direction", "center_us", "us_per_rad", "min_us", "max_us")


@dataclass
class JointSpec:
    name: str
    min_rad: float
    max_rad: float
    stand_rad: float


def load_joint_specs() -> dict[str, JointSpec]:
    specs = {}
    with open(MATES_PATH, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row["mate_name"].removeprefix("dof_")
            specs[name] = JointSpec(
                name=name,
                min_rad=math.radians(float(row["min_deg"])),
                max_rad=math.radians(float(row["max_deg"])),
                stand_rad=math.radians(float(row["standing_deg"])),
            )
    return specs


def load_calibration() -> dict:
    """Load the saved calibration. If there is none, copy the template and fill zero_rad with the standing pose."""
    path = CALIB_PATH if CALIB_PATH.exists() else TEMPLATE_PATH
    with open(path, encoding="utf-8") as f:
        calib = json.load(f)
    specs = load_joint_specs()
    for ch in calib["channels"]:
        if ch["zero_rad"] is None:
            ch["zero_rad"] = specs[ch["joint"]].stand_rad
    return calib


def save_calibration(calib: dict) -> None:
    calib = copy.deepcopy(calib)
    done = all(is_complete(ch) for ch in calib["channels"])
    calib["status"] = "MEASURED" if done else "PARTIAL_DO_NOT_RUN_POLICY"
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)


def channel_cfg(calib: dict, joint: str) -> dict:
    return next(ch for ch in calib["channels"] if ch["joint"] == joint)


def is_complete(ch: dict) -> bool:
    return all(ch.get(k) is not None for k in REQUIRED_FIELDS)


def rad_to_us_unclamped(ch: dict, q: float) -> float:
    return ch["center_us"] + ch["direction"] * ch["us_per_rad"] * (q - ch["zero_rad"])


def rad_to_us(ch: dict, q: float) -> int:
    if not is_complete(ch):
        raise ValueError(f"{ch['joint']} is not fully calibrated")
    us = rad_to_us_unclamped(ch, q)
    return int(round(min(max(us, ch["min_us"]), ch["max_us"])))


class ServoLink:
    """Serial link to the ESP32 servo bridge. dry_run=True only prints commands (practice without hardware)."""

    def __init__(self, port: str | None, baud: int = 115200, dry_run: bool = False):
        self.dry_run = dry_run
        self.ser = None
        if dry_run:
            return
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=0.5)
        time.sleep(2.0)  # opening the port resets the ESP32, so wait for it to boot
        self.ser.reset_input_buffer()
        for _ in range(5):
            self.ser.write(b"?\n")
            if self._readline() == "READY":
                return
        raise RuntimeError(f"No response from the ESP32 on {port}. Check the firmware upload and the port.")

    def _readline(self) -> str:
        return self.ser.readline().decode(errors="replace").strip()

    def _cmd(self, line: str) -> None:
        if self.dry_run:
            print(f"  [dry-run] {line}")
            return
        self.ser.write((line + "\n").encode())
        reply = self._readline()
        if reply != "OK":
            raise RuntimeError(f"ESP32 error: {line!r} → {reply!r}")

    def set_us(self, channel: int, us: int) -> None:
        self._cmd(f"P {channel} {int(us)}")

    def relax(self, channel: int) -> None:
        self._cmd(f"P {channel} 0")

    def set_all_us(self, us: list[int]) -> None:
        self._cmd("M " + " ".join(str(int(u)) for u in us))

    def relax_all(self) -> None:
        self._cmd("X")

    def close(self) -> None:
        try:
            self.relax_all()
        finally:
            if self.ser is not None:
                self.ser.close()
