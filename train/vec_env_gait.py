"""GPU 병렬 환경 (mujoco_warp 기반, 발 접촉 기반 gait 보상 버전).

vec_env.py를 건드리지 않고 별도 파일로 만들었다 — 기존 walk_forward.yaml
학습과 나란히 비교하기 위해서다. 관측(신경망 입력) 정의는 vec_env.py와
완전히 동일하고(N_JOINTS, N_ACTION_HISTORY 재사용), 물리 모델만 발 접촉
센서가 추가된 scene_gait.xml을 쓰고, 보상만 REGISTRY_GAIT을 쓴다.
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
from env import N_ACTION_HISTORY, N_JOINTS  # noqa: E402
from env_gait import ROBOT_XML_GAIT, _FOOT_SENSORS, _CONTACT_THRESHOLD  # noqa: E402
from reward_terms import is_fallen  # noqa: E402
from reward_terms_gait import REGISTRY_GAIT, GaitSimState  # noqa: E402


class _RewardCtx:
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


class VecQuadrupedEnvGait(VecEnv):
    """rsl_rl.env.VecEnv 구현체 (gait 보상 버전)."""

    def __init__(
        self,
        task_config: dict,
        num_envs: int = 4096,
        xml_path: Path = ROBOT_XML_GAIT,
        device: str = "cuda:0",
        seed: int = 0,
    ):
        self.cfg = task_config
        self.weights = task_config["reward_weights"]
        self.num_envs = num_envs
        self.num_actions = N_JOINTS
        self.device = device
        self._ctx = _RewardCtx(task_config)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.mjw_model = mjw.put_model(self.model)
        self.mjw_data = mjw.put_data(self.model, self.data, nworld=num_envs)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20
        self.control_dt = self.dt * self.control_decimation
        self.max_episode_length = max(
            1, int(task_config["episode_length_s"] / self.control_dt)
        )

        self._qpos = wp.to_torch(self.mjw_data.qpos)
        self._qvel = wp.to_torch(self.mjw_data.qvel)
        self._ctrl = wp.to_torch(self.mjw_data.ctrl)
        self._sensordata = wp.to_torch(self.mjw_data.sensordata)
        self._actuator_force = wp.to_torch(self.mjw_data.actuator_force)

        self._gyro_slice = _sensor_slice(self.model, "imu_gyro")
        self._accel_slice = _sensor_slice(self.model, "imu_accel")
        self._foot_adr = [
            self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)]
            for n in _FOOT_SENSORS
        ]

        force_limit = torch.as_tensor(
            self.model.actuator_forcerange[:, 1], device=device, dtype=torch.float32
        )
        self._force_limit = torch.where(force_limit > 0, force_limit, torch.ones_like(force_limit))

        ctrlrange = self.model.actuator_ctrlrange
        self._ctrl_lo = torch.as_tensor(ctrlrange[:, 0], device=device, dtype=torch.float32)
        self._ctrl_hi = torch.as_tensor(ctrlrange[:, 1], device=device, dtype=torch.float32)

        self._default_qpos = torch.as_tensor(self.data.qpos.copy(), device=device, dtype=torch.float32)

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._action_history = torch.zeros(num_envs, N_ACTION_HISTORY, N_JOINTS, device=device)
        self._prev_action = torch.zeros(num_envs, N_JOINTS, device=device)
        self._air_time = torch.zeros(num_envs, 4, device=device)
        self._prev_contact = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)

        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        self._reset_idx(torch.arange(num_envs, device=device))

    # ---- 내부 유틸 -----------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
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
        self._air_time[env_ids] = 0.0
        self._prev_contact[env_ids] = False
        self.episode_length_buf[env_ids] = 0

    def _get_obs(self) -> TensorDict:
        gyro = self._sensordata[:, self._gyro_slice]
        accel = self._sensordata[:, self._accel_slice]
        hist = self._action_history.reshape(self.num_envs, -1)
        obs = torch.cat([gyro, accel, hist], dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def _update_foot_contact(self):
        """발 접촉 상태를 갱신하고 (contact(N,4), landing_bonus(N,4))를 반환한다."""
        force = torch.stack([self._sensordata[:, a] for a in self._foot_adr], dim=-1)  # (N,4)
        contact = force > _CONTACT_THRESHOLD
        just_landed = (~self._prev_contact) & contact
        landing_bonus = self._air_time * just_landed.float()

        self._air_time = (self._air_time + self.control_dt) * (~contact).float()
        self._prev_contact = contact
        return contact.float(), landing_bonus

    def _make_sim_state(self, action: torch.Tensor) -> GaitSimState:
        x, y = self._qpos[:, 4], self._qpos[:, 5]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        contact, landing_bonus = self._update_foot_contact()
        return GaitSimState(
            height=self._qpos[:, 2],
            up_z=up_z,
            gyro=self._sensordata[:, self._gyro_slice],
            lin_vel=self._qvel[:, 0:3],
            ang_vel_z=self._qvel[:, 5],
            actuator_force_norm=self._actuator_force / self._force_limit,
            action=action,
            prev_action=self._prev_action,
            foot_contact=contact,
            air_time_bonus=landing_bonus,
        )

    def _compute_reward(self, state: GaitSimState):
        total = torch.zeros(self.num_envs, device=self.device)
        terms: dict[str, torch.Tensor] = {}
        for name, w in self.weights.items():
            value = REGISTRY_GAIT[name](state, self._ctx)
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

        log = {f"Reward/{name}": value for name, value in reward_terms.items()}
        log["State/height"] = state.height
        log["State/upright"] = state.up_z
        log["State/lin_vel_x"] = state.lin_vel[..., 0]
        log["State/fall_rate"] = fallen.float()
        log["State/feet_in_contact"] = state.foot_contact.sum(-1)  # 0~4, gait 버전 전용 지표

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            mjw.forward(self.mjw_model, self.mjw_data)

        obs = self._get_obs()
        extras = {"time_outs": timed_out, "log": log}
        return obs, rewards, dones, extras
