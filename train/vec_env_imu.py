"""GPU-parallel environment with the real BNO085 mounting and IMU imperfections.

A separate file so vec_env.py / vec_env_gait.py stay untouched. Two changes:

1. Physics model: scene_imu.xml / scene_gait_imu.xml, where the `imu` site sits where the
   BNO085 is actually mounted: on the inside floor of the body, at the middle
   (body frame z = -17 mm instead of +26 mm). Axes are unchanged: hardware_imu.json
   measured raw_to_body = identity (x HEAD, y LEFT, z UP, same as the sim).
2. Observation: the gyro/accel the policy sees get what the real sensor has:
   - white noise every control step
   - a constant bias per episode (what is left after imu_link.to_body() removes the
     measured gyro bias; the measured bias itself was ~0.003 rad/s)
   - a latency of 0 or 1 control step (0 or 20 ms), drawn per episode

Rewards and terminations still use the clean (privileged) sim state. The observation
size and layout are unchanged (30 = gyro 3 + accel 3 + command history 24), so
ppo_config.py, play.py and a future run_policy.py need no changes.

The noise settings can be overridden per task with an `imu_noise:` block in the yaml
(same keys as IMU_NOISE_DEFAULTS below).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vec_env import VecQuadrupedEnv  # noqa: E402
from vec_env_gait import VecQuadrupedEnvGait  # noqa: E402

_MUJOCO_DIR = Path(__file__).resolve().parent.parent / "dog_mujoco"
ROBOT_XML_IMU = _MUJOCO_DIR / "scene_imu.xml"
ROBOT_XML_GAIT_IMU = _MUJOCO_DIR / "scene_gait_imu.xml"

IMU_NOISE_DEFAULTS = {
    "gyro_noise_std": 0.005,   # rad/s, white noise per control step
    "accel_noise_std": 0.05,   # m/s², white noise per control step
    "gyro_bias_max": 0.01,     # rad/s, uniform ±, constant per episode
    "accel_bias_max": 0.1,     # m/s², uniform ±, constant per episode
    "max_latency_steps": 1,    # observation delay drawn per episode from 0..this (x 20 ms)
}


class _ImuNoiseMixin:
    """Adds IMU noise, bias and latency on top of a VecEnv's _get_obs/_reset_idx."""

    def _init_imu_noise(self, task_config: dict, num_envs: int, device: str) -> None:
        self._imu = {**IMU_NOISE_DEFAULTS, **(task_config.get("imu_noise") or {})}
        n_hist = self._imu["max_latency_steps"] + 1
        self._imu_hist = torch.zeros(num_envs, n_hist, 6, device=device)  # [0] = newest reading
        self._imu_bias = torch.zeros(num_envs, 6, device=device)
        self._imu_latency = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._imu_hist_fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        n = env_ids.numel()
        if n == 0:
            return
        bias_max = torch.tensor(
            [self._imu["gyro_bias_max"]] * 3 + [self._imu["accel_bias_max"]] * 3, device=self.device
        )
        self._imu_bias[env_ids] = (torch.rand(n, 6, generator=self._gen, device=self.device) * 2 - 1) * bias_max
        self._imu_latency[env_ids] = torch.randint(
            0, self._imu["max_latency_steps"] + 1, (n,), generator=self._gen, device=self.device
        )
        self._imu_hist_fresh[env_ids] = True  # history is refilled from the first reading after reset

    def _get_obs(self) -> TensorDict:
        clean = torch.cat(
            [self._sensordata[:, self._gyro_slice], self._sensordata[:, self._accel_slice]], dim=-1
        )
        std = torch.tensor(
            [self._imu["gyro_noise_std"]] * 3 + [self._imu["accel_noise_std"]] * 3, device=self.device
        )
        noise = torch.randn(clean.shape, generator=self._gen, device=self.device) * std
        reading = clean + self._imu_bias + noise

        fresh = self._imu_hist_fresh.view(-1, 1, 1)
        rolled = torch.roll(self._imu_hist, shifts=1, dims=1)
        rolled[:, 0] = reading
        self._imu_hist = torch.where(fresh, reading.unsqueeze(1).expand_as(rolled), rolled)
        self._imu_hist_fresh[:] = False

        delayed = self._imu_hist[torch.arange(self.num_envs, device=self.device), self._imu_latency]
        hist = self._action_history.reshape(self.num_envs, -1)
        obs = torch.cat([delayed, hist], dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])


class VecQuadrupedEnvImu(_ImuNoiseMixin, VecQuadrupedEnv):
    """vec_env.py's stand/walk env, with the real IMU position and imperfections."""

    def __init__(self, task_config: dict, num_envs: int = 4096, xml_path: Path = ROBOT_XML_IMU,
                 device: str = "cuda:0", seed: int = 0):
        self._init_imu_noise(task_config, num_envs, device)  # before super(): it calls _reset_idx
        super().__init__(task_config, num_envs=num_envs, xml_path=xml_path, device=device, seed=seed)


class VecQuadrupedEnvGaitImu(_ImuNoiseMixin, VecQuadrupedEnvGait):
    """vec_env_gait.py's gait-reward env, with the real IMU position and imperfections."""

    def __init__(self, task_config: dict, num_envs: int = 4096, xml_path: Path = ROBOT_XML_GAIT_IMU,
                 device: str = "cuda:0", seed: int = 0):
        self._init_imu_noise(task_config, num_envs, device)
        super().__init__(task_config, num_envs=num_envs, xml_path=xml_path, device=device, seed=seed)
