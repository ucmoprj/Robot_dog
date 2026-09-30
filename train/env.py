"""Single-environment Gym-style wrapper (CPU mujoco).

Used for development, debugging and visualization (view_task.py, play.py). Massively
parallel training runs in vec_env.py, which ports the same observation/reward design
to a mujoco_warp batch. Only the physics backend differs (1 CPU env vs N GPU envs);
the reward logic in reward_terms.py is shared through a common SimState container.

Example:
    env = QuadrupedEnv(load_task_config("tasks/stand.yaml"))
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from reward_terms import REGISTRY, SimState, is_fallen

ROBOT_XML = Path(__file__).resolve().parent.parent / "dog_mujoco" / "scene.xml"

N_JOINTS = 12
N_ACTION_HISTORY = 2  # current + previous command
OBS_DIM = 3 + 3 + N_JOINTS * N_ACTION_HISTORY  # gyro(3) + accel(3) + command history(24) = 30

_CONFIG_DEFAULTS = {
    "height_scale": 0.03,
    "velocity_scale": 0.05,
    "target_vx": 0.0,
    "fall_height": 0.05,
    "fall_upright": 0.6,
    "episode_length_s": 5.0,
    "target_height": 0.1196,
}


def load_task_config(path: str | Path) -> dict:
    """Read tasks/*.yaml and fill missing values with defaults."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in _CONFIG_DEFAULTS.items():
        cfg.setdefault(key, val)
    unknown = set(cfg["reward_weights"]) - set(REGISTRY)
    if unknown:
        raise ValueError(
            f"{path}: reward_weights contains names not in REGISTRY: {unknown}"
        )
    return cfg


class QuadrupedEnv:
    """Gym-like interface: reset() -> obs, step(action) -> (obs, reward, done, info)."""

    def __init__(self, task_config: dict, xml_path: Path = ROBOT_XML, seed: int | None = None):
        self.cfg = task_config
        self.weights = task_config["reward_weights"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20  # physics step (1 ms) x 20 = 50 Hz control rate (matches the hardware limit in README)
        self.max_steps = max(
            1, int(task_config["episode_length_s"] / (self.dt * self.control_decimation))
        )

        force_limit = self.model.actuator_forcerange[:, 1].copy()
        force_limit[force_limit <= 0] = 1.0
        self._force_limit = force_limit

        mujoco.mj_resetData(self.model, self.data)
        self._default_qpos = self.data.qpos.copy()
        self._action_history = np.zeros((N_ACTION_HISTORY, N_JOINTS), dtype=np.float32)
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        self._step_count = 0

    # ---- internal helpers ------------------------------------------------
    def _ctrlrange(self) -> np.ndarray:
        return self.model.actuator_ctrlrange  # (12, 2)

    def _sample_reset_qpos(self) -> np.ndarray:
        """Simple domain randomization: slightly randomize initial joint angles and height."""
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

    def _make_sim_state(self, action: np.ndarray, prev_action: np.ndarray) -> SimState:
        w, x, y, z = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        gyro = self.data.sensor("imu_gyro").data
        force_norm = self.data.actuator_force / self._force_limit
        f32 = torch.float32
        return SimState(
            height=torch.tensor(self.data.qpos[2], dtype=f32),
            up_z=torch.tensor(up_z, dtype=f32),
            gyro=torch.tensor(gyro.copy(), dtype=f32),
            lin_vel=torch.tensor(self.data.qvel[0:3].copy(), dtype=f32),
            ang_vel_z=torch.tensor(self.data.qvel[5], dtype=f32),
            actuator_force_norm=torch.tensor(force_norm.copy(), dtype=f32),
            action=torch.tensor(action, dtype=f32),
            prev_action=torch.tensor(prev_action, dtype=f32),
        )

    def _check_termination(self, state: SimState) -> bool:
        return bool(is_fallen(state, self.cfg["fall_height"], self.cfg["fall_upright"]))

    def _compute_reward(self, state: SimState):
        ctx = _RewardCtx(self.cfg)
        terms = {name: float(REGISTRY[name](state, ctx)) for name in self.weights}
        total = sum(self.weights[name] * terms[name] for name in self.weights)
        return float(total), terms

    # ---- gym interface ---------------------------------------------------
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._sample_reset_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._action_history[:] = 0.0
        self._prev_action[:] = 0.0
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
    """Wrap the cfg dict in the attribute access (ctx.target_height, ...) that reward_terms.py expects."""

    def __init__(self, cfg: dict):
        self.target_height = cfg["target_height"]
        self.height_scale = cfg["height_scale"]
        self.target_vx = cfg["target_vx"]
        self.velocity_scale = cfg["velocity_scale"]
