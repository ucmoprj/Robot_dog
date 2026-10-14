"""Single-environment CPU wrapper for walking over ~1 cm bars (used by play_obstacle.py).

A separate file so env.py / env_gait.py stay untouched. Differences from env_gait.py:

1. Scene: scene_obstacle.xml = the IMU-position robot (robot_gait_imu.xml) + bars.
2. Action = offset from the stand pose ("method B"), not an absolute angle:

       target angle = stand pose + scale × clip(action, -clip, +clip)    then clamped to ctrlrange

   The stand pose is the model's `stand` keyframe (roll 0°, hip -19.4°, knee 42°, the
   same angles as zero_rad in hardware_calibration.json). With an absolute angle, the
   policy's ±1 rad exploration was larger than most joint ranges, so most commands hit
   a limit and those joints stopped learning. Here action 0 = standing, and ±1 is
   ±scale (a few degrees to ~17°), so exploration stays inside the ranges.
   The action history in the observation holds the raw actions (what the network
   output), so run_policy.py on the real robot must keep the same.
3. Reset: from the stand keyframe (env.py resets to all joints 0), at a random start x
   so the bars are at a different distance every episode.
4. Optional command rate limit (`max_command_rate`, rad/s; off when missing): each joint
   target may move at most max_command_rate × 20 ms from the previous target, like the
   real servo bridge (4 rad/s in README; MG90S no-load speed is ~10 rad/s). Without it the
   first policies chattered at 25 Hz with commands up to 44 rad/s, which the real servos
   cannot follow. The real robot must apply the same limit (run_policy.py).
5. Optional gait clock (`gait_clock`; off when missing): a phase φ in [0, 1) that
   advances by frequency_hz × 20 ms every step (random start each episode). The
   observation gets [sin 2πφ, cos 2πφ] appended (30 -> 32 inputs), and each foot gets
   phase (φ + offset) % 1, stance while < duty, used by the gait_phase_* rewards. Trot
   offsets: FL and BR together (0.0), FR and BL half a cycle later (0.5). The clock is
   not a sensor, so the real robot just runs the same clock on the PC.

Observation: gyro 3 + accel 3 + raw action history 24 (= 30), + clock sin/cos 2 with a gait clock (= 32).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from env import OBS_DIM, _CONFIG_DEFAULTS
from env_gait import QuadrupedEnvGait
from reward_terms import is_fallen
from reward_terms_obstacle import REGISTRY_OBSTACLE, ObstacleSimState

ROBOT_XML_OBSTACLE = Path(__file__).resolve().parent.parent / "dog_mujoco" / "scene_obstacle.xml"
FOOT_SITES = ["fl_foot", "fr_foot", "bl_foot", "br_foot"]

_OBSTACLE_DEFAULTS = {
    "target_clearance": 0.02,                                        # m, swing foot height for full feet_clearance
    "spawn_x_range": [-0.2, 0.0],                                    # m, start x (bars start at 0.12 m)
    "action_scale": {"roll": 0.15, "hip": 0.3, "knee": 0.3},         # rad per unit action, by joint type
    "action_clip": 3.0,                                              # |raw action| limit
    "max_command_rate": None,                                        # rad/s per joint target; None = no limit
    "gait_clock": None,                                              # e.g. {frequency_hz: 2.0, duty: 0.5, offsets: {...}}
}

_GAIT_CLOCK_DEFAULTS = {
    "frequency_hz": 2.0,                                       # full gait cycles per second
    "duty": 0.5,                                               # fraction of the cycle each foot is on the ground
    "offsets": {"fl": 0.0, "fr": 0.5, "bl": 0.5, "br": 0.0},   # trot: diagonal pairs together
}
_CLOCK_REWARDS = {"gait_phase_contact", "gait_phase_swing_height"}


def load_task_config(path: str | Path) -> dict:
    """Read tasks/*.yaml, fill missing values with defaults (validated against REGISTRY_OBSTACLE)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in {**_CONFIG_DEFAULTS, **_OBSTACLE_DEFAULTS}.items():
        cfg.setdefault(key, val)
    unknown = set(cfg["reward_weights"]) - set(REGISTRY_OBSTACLE)
    if unknown:
        raise ValueError(f"{path}: reward_weights contains names not in REGISTRY_OBSTACLE: {unknown}")
    if cfg["gait_clock"] is not None:
        cfg["gait_clock"] = {**_GAIT_CLOCK_DEFAULTS, **cfg["gait_clock"]}
    elif _CLOCK_REWARDS & set(cfg["reward_weights"]):
        raise ValueError(f"{path}: {_CLOCK_REWARDS & set(cfg['reward_weights'])} need a gait_clock block")
    return cfg


def obs_dim(cfg: dict) -> int:
    """Network input size: 30, or 32 with a gait clock."""
    return OBS_DIM + (2 if cfg["gait_clock"] is not None else 0)


def foot_phase_offsets(cfg: dict) -> np.ndarray:
    """Clock offsets in FOOT_SITES order [fl, fr, bl, br] (zeros without a clock)."""
    if cfg["gait_clock"] is None:
        return np.zeros(4)
    return np.array([cfg["gait_clock"]["offsets"][n.split("_")[0]] for n in FOOT_SITES])


def stand_pose(model: mujoco.MjModel) -> np.ndarray:
    """qpos of the `stand` keyframe (19 values: free joint 7 + joints 12)."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return model.key_qpos[key].copy()


def action_scale_vector(model: mujoco.MjModel, cfg: dict) -> np.ndarray:
    """Per-actuator scale from the joint name suffix: _1 roll, _2 hip, _3 knee."""
    by_suffix = {"1": "roll", "2": "hip", "3": "knee"}
    names = [model.actuator(i).name for i in range(model.nu)]
    return np.array([cfg["action_scale"][by_suffix[n.split("_")[-1]]] for n in names])


class ObstacleRewardCtx:
    def __init__(self, cfg: dict):
        self.target_height = cfg["target_height"]
        self.height_scale = cfg["height_scale"]
        self.target_vx = cfg["target_vx"]
        self.velocity_scale = cfg["velocity_scale"]
        self.target_clearance = cfg["target_clearance"]
        self.duty = cfg["gait_clock"]["duty"] if cfg["gait_clock"] is not None else 0.5


class QuadrupedEnvObstacle(QuadrupedEnvGait):
    """Gym-like interface: reset() -> obs, step(raw_action) -> (obs, reward, done, info)."""

    def __init__(self, task_config: dict, xml_path: Path = ROBOT_XML_OBSTACLE, seed: int | None = None):
        super().__init__(task_config, xml_path=xml_path, seed=seed)
        self._default_qpos = stand_pose(self.model)
        self._default_ctrl = self._default_qpos[7:].copy()
        self._scale = action_scale_vector(self.model, task_config)
        self._clip = task_config["action_clip"]
        self._foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
        rate = task_config["max_command_rate"]
        self._max_step = None if rate is None else rate * self.control_dt  # rad per control step
        self._prev_target = self._default_ctrl.copy()
        self._prev_raw = np.zeros(self.model.nu, dtype=np.float32)
        self._raw = self._prev_raw.copy()
        self._ctx = ObstacleRewardCtx(task_config)
        clock = task_config["gait_clock"]
        self._clock_step = None if clock is None else clock["frequency_hz"] * self.control_dt  # phase per step
        self._phase_offsets = foot_phase_offsets(task_config)
        self._phase = 0.0

    def foot_phase(self) -> np.ndarray:
        return (self._phase + self._phase_offsets) % 1.0

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        obs = super()._get_obs(action)
        if self._clock_step is None:
            return obs
        clock = np.array([np.sin(2 * np.pi * self._phase), np.cos(2 * np.pi * self._phase)], dtype=np.float32)
        return np.concatenate([obs, clock])

    def action_to_target(self, raw: np.ndarray, prev_target: np.ndarray | None = None) -> np.ndarray:
        """stand + scale × clip(raw), rate-limited against prev_target (if a limit is set), clamped to ctrlrange."""
        raw = np.clip(np.asarray(raw, dtype=np.float64), -self._clip, self._clip)
        target = self._default_ctrl + self._scale * raw
        if self._max_step is not None and prev_target is not None:
            target = prev_target + np.clip(target - prev_target, -self._max_step, self._max_step)
        lo, hi = self._ctrlrange()[:, 0], self._ctrlrange()[:, 1]
        return np.clip(target, lo, hi)

    def _sample_reset_qpos(self) -> np.ndarray:
        qpos = super()._sample_reset_qpos()  # stand keyframe + small joint/height noise
        qpos[0] = self.rng.uniform(*self.cfg["spawn_x_range"])
        return qpos

    def _make_sim_state(self, action: np.ndarray, prev_action: np.ndarray) -> ObstacleSimState:
        s = super()._make_sim_state(action, prev_action)
        f32 = torch.float32
        return ObstacleSimState(
            **vars(s),
            foot_height=torch.tensor(self.data.site_xpos[self._foot_site_ids, 2].copy(), dtype=f32),
            raw_action=torch.tensor(self._raw, dtype=f32),
            prev_raw_action=torch.tensor(self._prev_raw, dtype=f32),
            joint_vel=torch.tensor(self.data.qvel[6:].copy(), dtype=f32),
            foot_phase=torch.tensor(self.foot_phase() if self._clock_step else np.zeros(4), dtype=f32),
        )

    def _compute_reward(self, state: ObstacleSimState):
        terms = {name: float(REGISTRY_OBSTACLE[name](state, self._ctx)) for name in self.weights}
        total = sum(self.weights[name] * terms[name] for name in self.weights)
        return float(total), terms

    def reset(self) -> np.ndarray:
        self._prev_target = self._default_ctrl.copy()
        self._prev_raw[:] = 0.0
        self._phase = float(self.rng.uniform()) if self._clock_step else 0.0
        obs = super().reset()
        self.data.ctrl[:] = self._default_ctrl
        return obs

    def step(self, raw_action: np.ndarray):
        raw = np.clip(np.asarray(raw_action, dtype=np.float32), -self._clip, self._clip)
        target = self.action_to_target(raw, self._prev_target)
        self._raw = raw

        self.data.ctrl[:] = target
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
        if self._clock_step:
            self._phase = (self._phase + self._clock_step) % 1.0

        state = self._make_sim_state(target, self._prev_target)
        reward, terms = self._compute_reward(state)
        obs = self._get_obs(raw)
        self._prev_target = target
        self._prev_raw = raw
        self._step_count += 1

        done = bool(is_fallen(state, self.cfg["fall_height"], self.cfg["fall_upright"])) or (
            self._step_count >= self.max_steps
        )
        info = {"reward_terms": terms, "target": target}
        return obs, reward, done, info
