"""GPU-parallel environment (mujoco_warp).

Runs the same observation/reward/termination logic as env.py (QuadrupedEnv, a
single CPU env) on mujoco_warp batch data (warp arrays shaped (N, ...)).
warp <-> torch share GPU memory without copying (zero-copy) via wp.to_torch().

The reward functions in reward_terms.py only take the SimState container,
whether it came from CPU or GPU, so changing the task (tasks/*.yaml) never
requires editing this file, for exactly the same reason as env.py.

Implements the rsl_rl.env.VecEnv abstract class so rsl_rl's PPO runner can use
it directly (see train.py).
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
    """Wrap the cfg dict in the attribute access that reward_terms.py expects."""

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
    """rsl_rl.env.VecEnv implementation. Simulates num_envs robots at once on the GPU."""

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

        # The CPU model is only used to look up structure (joint limits, sensor
        # addresses, torque limits, ...); physics steps run on the mujoco_warp batch (GPU).
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.mjw_model = mjw.put_model(self.model)
        self.mjw_data = mjw.put_data(self.model, self.data, nworld=num_envs)

        self.dt = self.model.opt.timestep
        self.control_decimation = 20  # physics step (1 ms) x 20 = 50 Hz control rate, same as env.py
        self.max_episode_length = max(
            1, int(task_config["episode_length_s"] / (self.dt * self.control_decimation))
        )

        # ---- warp -> torch zero-copy views (point at the same GPU memory) -------
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

        # ---- state buffers required by VecEnv -----------------------------------
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._action_history = torch.zeros(num_envs, N_ACTION_HISTORY, N_JOINTS, device=device)
        self._prev_action = torch.zeros(num_envs, N_JOINTS, device=device)

        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        self._reset_idx(torch.arange(num_envs, device=device))

    # ---- internal helpers ----------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset only the selected envs (same randomization ranges as env.py)."""
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
        """Return the weighted sum (total) and the raw unweighted value of each term (terms).
        terms is not used for learning, only for TensorBoard logging."""
        total = torch.zeros(self.num_envs, device=self.device)
        terms: dict[str, torch.Tensor] = {}
        for name, w in self.weights.items():
            value = REGISTRY[name](state, self._ctx)
            terms[name] = value
            total = total + w * value
        return total, terms

    # ---- rsl_rl.env.VecEnv interface ------------------------------------------
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

        # Detailed TensorBoard logs: raw value of each reward term + robot state metrics.
        # Keys containing "/" are used as-is; others are plotted under "Episode/".
        log = {f"Reward/{name}": value for name, value in reward_terms.items()}
        log["State/height"] = state.height
        log["State/upright"] = state.up_z
        log["State/lin_vel_x"] = state.lin_vel[..., 0]
        log["State/fall_rate"] = fallen.float()

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            mjw.forward(self.mjw_model, self.mjw_data)  # recompute sensors/derived values for the reset envs

        obs = self._get_obs()
        extras = {"time_outs": timed_out, "log": log}
        return obs, rewards, dones, extras
