"""Single-environment Gym-style wrapper (CPU mujoco, gait-reward version using foot contacts).

A separate file so env.py stays untouched and results can be compared side by side
with the existing walk_forward.yaml training. The observation (network input) is
exactly the same as env.py (N_JOINTS, N_ACTION_HISTORY, OBS_DIM are reused). Only the
physics model differs (scene_gait.xml, which adds foot contact sensors), and the
rewards come from REGISTRY_GAIT.

Example:
    env = QuadrupedEnvGait(load_task_config("tasks/walk_forward_gait.yaml"))
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from env import N_ACTION_HISTORY, N_JOINTS, OBS_DIM, _CONFIG_DEFAULTS  # noqa: F401 (OBS_DIM is used by other modules)
from reward_terms import is_fallen
from reward_terms_gait import REGISTRY_GAIT, GaitSimState

ROBOT_XML_GAIT = Path(__file__).resolve().parent.parent / "dog_mujoco" / "scene_gait.xml"

_FOOT_SENSORS = ["fl_foot_contact", "fr_foot_contact", "bl_foot_contact", "br_foot_contact"]
_CONTACT_THRESHOLD = 0.05  # force above this counts as "contact" (about 0.6-0.85 under full support)


def load_task_config(path: str | Path) -> dict:
    """Read tasks/*.yaml and fill missing values with defaults (validated against REGISTRY_GAIT)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in _CONFIG_DEFAULTS.items():
        cfg.setdefault(key, val)
    unknown = set(cfg["reward_weights"]) - set(REGISTRY_GAIT)
    if unknown:
        raise ValueError(
            f"{path}: reward_weights contains names not in REGISTRY_GAIT: {unknown}"
        )
    return cfg


class QuadrupedEnvGait:
    """Gym-like interface. Same structure as env.py's QuadrupedEnv, plus foot contact tracking."""

    def __init__(self, task_config: dict, xml_path: Path = ROBOT_XML_GAIT, seed: int | None = None):
        self.cfg = task_config
        self.weights = task_config["reward_weights"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20
        self.control_dt = self.dt * self.control_decimation
        self.max_steps = max(
            1, int(task_config["episode_length_s"] / self.control_dt)
        )

        force_limit = self.model.actuator_forcerange[:, 1].copy()
        force_limit[force_limit <= 0] = 1.0
        self._force_limit = force_limit

        self._foot_sensor_adr = np.array(
            [self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)] for n in _FOOT_SENSORS]
        )

        mujoco.mj_resetData(self.model, self.data)
        self._default_qpos = self.data.qpos.copy()
        self._action_history = np.zeros((N_ACTION_HISTORY, N_JOINTS), dtype=np.float32)
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        self._air_time = np.zeros(4, dtype=np.float32)
        self._prev_contact = np.zeros(4, dtype=bool)
        self._step_count = 0

    # ---- internal helpers ------------------------------------------------
    def _ctrlrange(self) -> np.ndarray:
        return self.model.actuator_ctrlrange

    def _sample_reset_qpos(self) -> np.ndarray:
        qpos = self._default_qpos.copy()
        qpos[7:] += self.rng.uniform(-0.05, 0.05, size=N_JOINTS)
        qpos[2] += self.rng.uniform(-0.005, 0.005)
        return qpos

    def _get_obs(self, action: np.ndarray) -> np.ndarray:
        gyro = self.data.sensor("imu_gyro").data.copy()
        accel = self.data.sensor("imu_accel").data.copy()
        self._action_history = np.roll(self._action_history, shift=1, axis=0)
        self._action_history[0] = action
        return np.concatenate([gyro, accel, self._action_history.flatten()]).astype(np.float32)

    def _update_foot_contact(self) -> tuple[np.ndarray, np.ndarray]:
        """Update foot contact state and return (contact_float(4,), landing_bonus(4,))."""
        force = self.data.sensordata[self._foot_sensor_adr]
        contact = force > _CONTACT_THRESHOLD
        just_landed = (~self._prev_contact) & contact
        landing_bonus = self._air_time * just_landed.astype(np.float32)

        self._air_time = (self._air_time + self.control_dt) * (~contact).astype(np.float32)
        self._prev_contact = contact
        return contact.astype(np.float32), landing_bonus

    def _make_sim_state(self, action: np.ndarray, prev_action: np.ndarray) -> GaitSimState:
        w, x, y, z = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        gyro = self.data.sensor("imu_gyro").data
        force_norm = self.data.actuator_force / self._force_limit
        contact, landing_bonus = self._update_foot_contact()
        f32 = torch.float32
        return GaitSimState(
            height=torch.tensor(self.data.qpos[2], dtype=f32),
            up_z=torch.tensor(up_z, dtype=f32),
            gyro=torch.tensor(gyro.copy(), dtype=f32),
            lin_vel=torch.tensor(self.data.qvel[0:3].copy(), dtype=f32),
            ang_vel_z=torch.tensor(self.data.qvel[5], dtype=f32),
            actuator_force_norm=torch.tensor(force_norm.copy(), dtype=f32),
            action=torch.tensor(action, dtype=f32),
            prev_action=torch.tensor(prev_action, dtype=f32),
            foot_contact=torch.tensor(contact, dtype=f32),
            air_time_bonus=torch.tensor(landing_bonus, dtype=f32),
        )

    def _check_termination(self, state: GaitSimState) -> bool:
        return bool(is_fallen(state, self.cfg["fall_height"], self.cfg["fall_upright"]))

    def _compute_reward(self, state: GaitSimState):
        ctx = _RewardCtx(self.cfg)
        terms = {name: float(REGISTRY_GAIT[name](state, ctx)) for name in self.weights}
        total = sum(self.weights[name] * terms[name] for name in self.weights)
        return float(total), terms

    # ---- gym interface ---------------------------------------------------
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._sample_reset_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._action_history[:] = 0.0
        self._prev_action[:] = 0.0
        self._air_time[:] = 0.0
        self._prev_contact[:] = False
        self._step_count = 0
        return self._get_obs(self._prev_action)

    def step(self, action: np.ndarray):
        lo, hi = self._ctrlrange()[:, 0], self._ctrlrange()[:, 1]
        action = np.clip(np.asarray(action, dtype=np.float64), lo, hi)

        self.data.ctrl[:] = action
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

        state = self._make_sim_state(action, self._prev_action)
        reward, terms = self._compute_reward(state)
        obs = self._get_obs(action)
        self._prev_action = action
        self._step_count += 1

        done = self._check_termination(state) or self._step_count >= self.max_steps
        info = {"reward_terms": terms}
        return obs, reward, done, info


class _RewardCtx:
    def __init__(self, cfg: dict):
        self.target_height = cfg["target_height"]
        self.height_scale = cfg["height_scale"]
        self.target_vx = cfg["target_vx"]
        self.velocity_scale = cfg["velocity_scale"]
