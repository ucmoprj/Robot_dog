"""단일 환경 Gym 스타일 래퍼 (CPU mujoco 기준).

개발/디버깅/시각화(view_task.py, 추후 play.py)용이다. 대량 병렬 학습은 같은
관측/보상 설계를 mujoco_warp 배치 버전으로 옮긴 vec_env.py에서 수행한다 —
물리 계산 방식만 CPU 1개 vs GPU N개로 다를 뿐, reward_terms.py의 보상 로직은
공유한다 (SimState라는 공통 컨테이너를 통해).

사용 예:
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
N_ACTION_HISTORY = 2  # 현재 + 직전 명령
OBS_DIM = 3 + 3 + N_JOINTS * N_ACTION_HISTORY  # gyro(3) + accel(3) + 명령이력(24) = 30

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
    """tasks/*.yaml을 읽고 누락된 값은 기본값으로 채운다."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in _CONFIG_DEFAULTS.items():
        cfg.setdefault(key, val)
    unknown = set(cfg["reward_weights"]) - set(REGISTRY)
    if unknown:
        raise ValueError(
            f"{path} 의 reward_weights에 REGISTRY에 없는 이름이 있습니다: {unknown}"
        )
    return cfg


class QuadrupedEnv:
    """gym 유사 인터페이스: reset() -> obs, step(action) -> (obs, reward, done, info)."""

    def __init__(self, task_config: dict, xml_path: Path = ROBOT_XML, seed: int | None = None):
        self.cfg = task_config
        self.weights = task_config["reward_weights"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20  # 물리 step(1ms) x 20 = 50Hz 제어 주기 (README 실물 제약과 일치)
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

    # ---- 내부 유틸 -------------------------------------------------------
    def _ctrlrange(self) -> np.ndarray:
        return self.model.actuator_ctrlrange  # (12, 2)

    def _sample_reset_qpos(self) -> np.ndarray:
        """간단한 domain randomization: 초기 관절각/높이를 살짝 무작위화한다."""
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

    # ---- gym 인터페이스 ---------------------------------------------------
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
    """cfg 딕셔너리를 reward_terms.py 함수들이 기대하는 속성 접근(ctx.target_height 등)으로 감싼다."""

    def __init__(self, cfg: dict):
        self.target_height = cfg["target_height"]
        self.height_scale = cfg["height_scale"]
        self.target_vx = cfg["target_vx"]
        self.velocity_scale = cfg["velocity_scale"]
