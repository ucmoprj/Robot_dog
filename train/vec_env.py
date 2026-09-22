"""GPU 병렬 환경 (mujoco_warp 기반).

env.py(QuadrupedEnv, CPU 단일 환경)와 동일한 관측/보상/종료 로직을,
mujoco_warp 배치 데이터(warp 배열, (N, ...) 형태) 위에서 수행한다.
warp<->torch 사이는 wp.to_torch()로 GPU 메모리를 복사 없이(zero-copy) 공유한다.

reward_terms.py의 보상 함수는 CPU/GPU 어느 쪽에서 왔든 동일한 SimState
컨테이너만 받으므로, 태스크(tasks/*.yaml)를 바꿔도 이 파일은 건드릴 필요가
없다 — env.py와 정확히 같은 이유다.

rsl_rl.env.VecEnv 추상 클래스를 구현해서 rsl_rl의 PPO 러너가 바로 이 클래스를
사용할 수 있게 한다 (train.py 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import N_ACTION_HISTORY, N_JOINTS, ROBOT_XML  # noqa: E402
from reward_terms import REGISTRY, SimState, is_fallen  # noqa: E402


class _RewardCtx:
    """cfg 딕셔너리를 reward_terms.py 함수들이 기대하는 속성 접근으로 감싼다."""

    def __init__(self, cfg: dict):
        self.target_height = cfg["target_height"]
        self.height_scale = cfg["height_scale"]
        self.target_vx = cfg["target_vx"]
        self.velocity_scale = cfg["velocity_scale"]


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return slice(adr, adr + dim)


class VecQuadrupedEnv(VecEnv):
    """rsl_rl.env.VecEnv 구현체. num_envs개의 로봇을 GPU에서 동시에 시뮬레이션한다."""

    def __init__(
        self,
        task_config: dict,
        num_envs: int = 4096,
        xml_path: Path = ROBOT_XML,
        device: str = "cuda:0",
        seed: int = 0,
    ):
        self.cfg = task_config
        self.weights = task_config["reward_weights"]
        self.num_envs = num_envs
        self.num_actions = N_JOINTS
        self.device = device
        self._ctx = _RewardCtx(task_config)

        # CPU 모델은 구조 정보(관절 제한, 센서 주소, 토크 한계 등) 조회용으로만
        # 쓰고, 실제 물리 스텝은 mujoco_warp 배치 데이터(GPU)에서 수행한다.
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.mjw_model = mjw.put_model(self.model)
        self.mjw_data = mjw.put_data(self.model, self.data, nworld=num_envs)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20  # 물리 step(1ms) x 20 = 50Hz 제어 주기, env.py와 동일
        self.max_episode_length = max(
            1, int(task_config["episode_length_s"] / (self.dt * self.control_decimation))
        )

        # ---- warp -> torch zero-copy 뷰 (같은 GPU 메모리를 그대로 가리킴) -------
        self._qpos = wp.to_torch(self.mjw_data.qpos)                      # (N, 19)
        self._qvel = wp.to_torch(self.mjw_data.qvel)                       # (N, 18)
        self._ctrl = wp.to_torch(self.mjw_data.ctrl)                       # (N, 12)
        self._sensordata = wp.to_torch(self.mjw_data.sensordata)           # (N, 10)
        self._actuator_force = wp.to_torch(self.mjw_data.actuator_force)   # (N, 12)

        self._gyro_slice = _sensor_slice(self.model, "imu_gyro")
        self._accel_slice = _sensor_slice(self.model, "imu_accel")

        force_limit = torch.as_tensor(
            self.model.actuator_forcerange[:, 1], device=device, dtype=torch.float32
        )
        self._force_limit = torch.where(force_limit > 0, force_limit, torch.ones_like(force_limit))

        ctrlrange = self.model.actuator_ctrlrange
        self._ctrl_lo = torch.as_tensor(ctrlrange[:, 0], device=device, dtype=torch.float32)
        self._ctrl_hi = torch.as_tensor(ctrlrange[:, 1], device=device, dtype=torch.float32)

        self._default_qpos = torch.as_tensor(self.data.qpos.copy(), device=device, dtype=torch.float32)

        # ---- VecEnv가 요구하는 상태 버퍼 --------------------------------------
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._action_history = torch.zeros(num_envs, N_ACTION_HISTORY, N_JOINTS, device=device)
        self._prev_action = torch.zeros(num_envs, N_JOINTS, device=device)

        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        self._reset_idx(torch.arange(num_envs, device=device))

    # ---- 내부 유틸 -----------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """일부 환경만 골라 초기 상태로 되돌린다 (env.py의 domain randomization과 동일 폭)."""
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        qpos = self._default_qpos.unsqueeze(0).repeat(n, 1).clone()
        qpos[:, 7:] += (torch.rand(n, N_JOINTS, generator=self._gen, device=self.device) * 2 - 1) * 0.05
        qpos[:, 2] += (torch.rand(n, generator=self._gen, device=self.device) * 2 - 1) * 0.005
        self._qpos[env_ids] = qpos
        self._qvel[env_ids] = 0.0
        self._action_history[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

    def _get_obs(self) -> TensorDict:
        gyro = self._sensordata[:, self._gyro_slice]
        accel = self._sensordata[:, self._accel_slice]
        hist = self._action_history.reshape(self.num_envs, -1)
        obs = torch.cat([gyro, accel, hist], dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def _make_sim_state(self, action: torch.Tensor) -> SimState:
        x, y = self._qpos[:, 4], self._qpos[:, 5]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return SimState(
            height=self._qpos[:, 2],
            up_z=up_z,
            gyro=self._sensordata[:, self._gyro_slice],
            lin_vel=self._qvel[:, 0:3],
            ang_vel_z=self._qvel[:, 5],
            actuator_force_norm=self._actuator_force / self._force_limit,
            action=action,
            prev_action=self._prev_action,
        )

    def _compute_reward(self, state: SimState):
        """가중합(total)과 함께, 항목별 미가중 원시값(terms)도 반환한다.
        terms는 학습 자체에는 안 쓰이고 TensorBoard 그래프용 로깅에만 쓰인다."""
        total = torch.zeros(self.num_envs, device=self.device)
        terms: dict[str, torch.Tensor] = {}
        for name, w in self.weights.items():
            value = REGISTRY[name](state, self._ctx)
            terms[name] = value
            total = total + w * value
        return total, terms

    # ---- rsl_rl.env.VecEnv 인터페이스 -----------------------------------------
    def get_observations(self) -> TensorDict:
        return self._get_obs()

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, self._ctrl_lo, self._ctrl_hi)
        self._ctrl.copy_(actions)

        for _ in range(self.control_decimation):
            mjw.step(self.mjw_model, self.mjw_data)

        state = self._make_sim_state(actions)
        rewards, reward_terms = self._compute_reward(state)

        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
        self._action_history[:, 0, :] = actions
        self._prev_action = actions
        self.episode_length_buf += 1

        fallen = is_fallen(state, self.cfg["fall_height"], self.cfg["fall_upright"])
        timed_out = self.episode_length_buf >= self.max_episode_length
        dones = fallen | timed_out

        # TensorBoard용 세부 로그: 보상 항목별 원시값 + 로봇 상태 지표.
        # 키에 "/"가 있으면 그 이름 그대로, 없으면 "Episode/" 아래에 그래프가 생긴다.
        log = {f"Reward/{name}": value for name, value in reward_terms.items()}
        log["State/height"] = state.height
        log["State/upright"] = state.up_z
        log["State/lin_vel_x"] = state.lin_vel[..., 0]
        log["State/fall_rate"] = fallen.float()

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            mjw.forward(self.mjw_model, self.mjw_data)  # 리셋된 환경의 센서/파생값 재계산

        obs = self._get_obs()
        extras = {"time_outs": timed_out, "log": log}
        return obs, rewards, dones, extras
