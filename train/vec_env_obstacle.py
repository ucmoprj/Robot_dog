"""GPU-parallel environment for walking over ~1 cm bars (mujoco_warp).

The GPU counterpart of env_obstacle.py, built from the existing pieces without editing them:

    VecQuadrupedEnvGait   (vec_env_gait.py)  foot contacts, feet_air_time, gait_symmetry
  + _ImuNoiseMixin        (vec_env_imu.py)   IMU noise, bias, 0-20 ms latency
  + this file:            scene_obstacle.xml, stand-keyframe reset with random start x,
                          method-B actions (offset from the stand pose), feet_clearance

See env_obstacle.py for why the action is an offset from the stand pose, and for the
optional command rate limit (max_command_rate), gait clock (gait_clock) and heading (heading).
With a heading block, the observed yaw error gets white noise and a per-episode drift, like the
BNO085 game rotation vector; the reward uses the clean value. With a velocity_command block, every
env gets its own vx_cmd (and clock frequency), drawn at reset and every resample_s.

The 20 physics substeps per control step run as one CUDA graph (captured on the first step).
Launched one by one from Python, mujoco_warp's many small kernels spent ~95 % of the time on
launch overhead: 2048 envs took 393 ms per control step, 17 ms as a graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_obstacle import (  # noqa: E402
    FOOT_SITES,
    ROBOT_XML_OBSTACLE,
    VX_CMD_OBS_SCALE,
    ObstacleRewardCtx,
    action_scale_vector,
    body_contact_count,
    body_geom_ids,
    clock_frequency,
    foot_phase_offsets,
    stand_pose,
    wrap_angle,
    yaw_of,
)
from reward_terms import is_fallen  # noqa: E402
from reward_terms_obstacle import REGISTRY_OBSTACLE, ObstacleSimState  # noqa: E402
from vec_env_gait import VecQuadrupedEnvGait  # noqa: E402
from vec_env_imu import _ImuNoiseMixin  # noqa: E402


class VecQuadrupedEnvObstacle(_ImuNoiseMixin, VecQuadrupedEnvGait):
    def __init__(self, task_config: dict, num_envs: int = 4096, xml_path: Path = ROBOT_XML_OBSTACLE,
                 device: str = "cuda:0", seed: int = 0):
        self._init_imu_noise(task_config, num_envs, device)
        self._ready = False  # the parent __init__ resets once before our buffers exist
        super().__init__(task_config, num_envs=num_envs, xml_path=xml_path, device=device, seed=seed)
        self._ctx = ObstacleRewardCtx(task_config)

        f32 = torch.float32
        self._default_qpos = torch.as_tensor(stand_pose(self.model), device=device, dtype=f32)
        self._default_ctrl = self._default_qpos[7:].clone()
        self._scale = torch.as_tensor(action_scale_vector(self.model, task_config), device=device, dtype=f32)
        self._clip = float(task_config["action_clip"])
        self._spawn_x = task_config["spawn_x_range"]
        self._site_xpos = wp.to_torch(self.mjw_data.site_xpos)  # (N, nsite, 3)
        self._foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES]
        self._start_x = torch.zeros(num_envs, device=device)
        self._prev_target = self._default_ctrl.repeat(num_envs, 1)
        rate = task_config["max_command_rate"]
        self._max_step = None if rate is None else rate * self.control_dt  # rad per control step
        self._raw = torch.zeros(num_envs, self.num_actions, device=device)
        clock = task_config["gait_clock"]
        self._vel_cmd = task_config["velocity_command"]
        self._vx_cmd = torch.full((num_envs,), float(task_config["target_vx"]), device=device)
        self._ctx.target_vx = self._vx_cmd  # velocity_track compares with each env's own command
        if self._vel_cmd is not None and self._vel_cmd["resample_s"]:
            self._resample_steps = max(1, round(self._vel_cmd["resample_s"] / self.control_dt))
        else:
            self._resample_steps = None
        # phase per step: (N,) when the frequency follows the command, else a float
        self._clock_step = None if clock is None else clock_frequency(task_config, self._vx_cmd) * self.control_dt
        self._phase_offsets = torch.as_tensor(foot_phase_offsets(task_config), device=device, dtype=f32)
        self._phase = torch.zeros(num_envs, device=device)
        self._heading = task_config["heading"]
        self._start_yaw = torch.zeros(num_envs, device=device)
        self._yaw_drift = torch.zeros(num_envs, device=device)  # rad/s, observed yaw only
        self._body_geoms = body_geom_ids(self.model)
        self._body_half = torch.as_tensor(self.model.geom_size[self._body_geoms], device=device, dtype=f32)
        self._geom_xpos = wp.to_torch(self.mjw_data.geom_xpos)  # (N, ngeom, 3)
        self._geom_xmat = wp.to_torch(self.mjw_data.geom_xmat)  # (N, ngeom, 3, 3)
        self._physics_graph = None  # captured on the first step, after the kernels are loaded

        self._ready = True
        self._reset_idx(torch.arange(num_envs, device=device))  # again, now from the stand keyframe
        mjw.forward(self.mjw_model, self.mjw_data)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)  # default qpos + joint/height noise, history, IMU bias/latency
        if not self._ready or env_ids.numel() == 0:
            return
        n = env_ids.numel()
        lo, hi = self._spawn_x
        x0 = lo + (hi - lo) * torch.rand(n, generator=self._gen, device=self.device)
        self._qpos[env_ids, 0] = x0
        self._start_x[env_ids] = x0
        self._ctrl[env_ids] = self._default_ctrl
        self._prev_target[env_ids] = self._default_ctrl
        if self._clock_step is not None:
            self._phase[env_ids] = torch.rand(n, generator=self._gen, device=self.device)
        self._start_yaw[env_ids] = yaw_of(self._qpos[env_ids, 3:7])
        self._sample_command(env_ids)
        if self._heading is not None:
            u = torch.rand(n, generator=self._gen, device=self.device) * 2 - 1
            self._yaw_drift[env_ids] = u * self._heading["drift_max"]

    def _sample_command(self, env_ids: torch.Tensor) -> None:
        """New vx_cmd (uniform over velocity_command.range) and matching clock frequency for env_ids."""
        if self._vel_cmd is None or env_ids.numel() == 0:
            return
        lo, hi = self._vel_cmd["range"]
        self._vx_cmd[env_ids] = lo + (hi - lo) * torch.rand(env_ids.numel(), generator=self._gen, device=self.device)
        if isinstance(self._clock_step, torch.Tensor):
            self._clock_step[env_ids] = clock_frequency(self.cfg, self._vx_cmd[env_ids]) * self.control_dt

    def _foot_phase(self) -> torch.Tensor:
        return (self._phase.unsqueeze(-1) + self._phase_offsets) % 1.0  # (N, 4)

    def _yaw_error(self) -> torch.Tensor:
        return wrap_angle(yaw_of(self._qpos[:, 3:7]) - self._start_yaw)

    def _get_obs(self) -> TensorDict:
        td = super()._get_obs()  # IMU (noisy, delayed) + raw action history
        parts = [td["policy"]]
        if self._clock_step is not None:
            angle = 2 * torch.pi * self._phase
            parts += [torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)]
        if self._heading is not None:
            noise = torch.randn(self.num_envs, generator=self._gen, device=self.device) * self._heading["noise_std"]
            err = self._yaw_error() + self._yaw_drift * self.episode_length_buf * self.control_dt + noise
            parts += [torch.sin(err).unsqueeze(-1), torch.cos(err).unsqueeze(-1)]
        if self._vel_cmd is not None:
            parts.append((self._vx_cmd * VX_CMD_OBS_SCALE).unsqueeze(-1))
        if len(parts) == 1:
            return td
        return TensorDict({"policy": torch.cat(parts, dim=-1)}, batch_size=[self.num_envs])

    def _make_sim_state(self, action: torch.Tensor) -> ObstacleSimState:
        s = super()._make_sim_state(action)  # prev_action = self._prev_action (set to the previous target)
        return ObstacleSimState(
            **vars(s),
            foot_height=self._site_xpos[:, self._foot_site_ids, 2],
            raw_action=self._raw,
            prev_raw_action=self._action_history[:, 0, :],  # last step's raw output (0 right after a reset)
            joint_vel=self._qvel[:, 6:],
            foot_phase=self._foot_phase() if self._clock_step is not None else torch.zeros(self.num_envs, 4, device=self.device),
            yaw_error=self._yaw_error(),
            body_contact=body_contact_count(
                self._geom_xpos[:, self._body_geoms], self._geom_xmat[:, self._body_geoms], self._body_half
            ),
        )

    def _physics_step(self) -> None:
        """control_decimation mujoco_warp steps, replayed as one CUDA graph after the first call.
        The graph reads and writes the same mjw_data buffers that the torch views point at."""
        if self._physics_graph is not None:
            wp.capture_launch(self._physics_graph)
            return
        for _ in range(self.control_decimation):  # first call: plain launches, also loads the kernels
            mjw.step(self.mjw_model, self.mjw_data)
        if wp.get_device(self.device).is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.control_decimation):
                    mjw.step(self.mjw_model, self.mjw_data)
            self._physics_graph = capture.graph
            # capturing records the launches without running them, so the state is unchanged

    def _compute_reward(self, state: ObstacleSimState):
        total = torch.zeros(self.num_envs, device=self.device)
        terms: dict[str, torch.Tensor] = {}
        for name, w in self.weights.items():
            value = REGISTRY_OBSTACLE[name](state, self._ctx)
            terms[name] = value
            total = total + w * value
        return total, terms

    def step(self, actions: torch.Tensor):
        raw = torch.clamp(actions, -self._clip, self._clip)
        target = self._default_ctrl + self._scale * raw
        if self._max_step is not None:
            target = self._prev_target + torch.clamp(target - self._prev_target, -self._max_step, self._max_step)
        target = torch.clamp(target, self._ctrl_lo, self._ctrl_hi)
        self._raw = raw
        self._ctrl.copy_(target)

        self._physics_step()
        if self._clock_step is not None:
            self._phase = (self._phase + self._clock_step) % 1.0

        self._prev_action = self._prev_target  # action_rate_penalty compares joint targets (rad)
        state = self._make_sim_state(target)
        rewards, reward_terms = self._compute_reward(state)

        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
        self._action_history[:, 0, :] = raw  # the observation holds raw actions
        self._prev_target = target
        self.episode_length_buf += 1
        if self._resample_steps is not None:
            due = (self.episode_length_buf % self._resample_steps == 0).nonzero(as_tuple=False).squeeze(-1)
            self._sample_command(due)

        fallen = is_fallen(state, self.cfg["fall_height"], self.cfg["fall_upright"])
        timed_out = self.episode_length_buf >= self.max_episode_length
        dones = fallen | timed_out

        log = {f"Reward/{name}": value for name, value in reward_terms.items()}
        log["State/height"] = state.height
        log["State/upright"] = state.up_z
        log["State/lin_vel_x"] = state.lin_vel[..., 0]
        log["State/fall_rate"] = fallen.float()
        log["State/feet_in_contact"] = state.foot_contact.sum(-1)
        log["State/distance_x"] = self._qpos[:, 0] - self._start_x  # > 0.32 m = past the first bar for sure
        swing = 1.0 - state.foot_contact
        log["State/swing_foot_height"] = (state.foot_height * swing).sum(-1) / swing.sum(-1).clamp(min=1.0)
        log["State/action_clipped_frac"] = (actions.abs() > self._clip).float().mean(-1)
        log["State/command_rate"] = (target - self._prev_action).abs().mean(-1) / self.control_dt  # rad/s
        log["State/joint_speed"] = state.joint_vel.abs().mean(-1)  # rad/s
        log["State/yaw_error_deg"] = state.yaw_error.abs() * (180.0 / torch.pi)
        log["State/body_contact"] = state.body_contact  # non-foot geoms on the floor (knees etc.)
        if self._vel_cmd is not None:
            log["State/vx_cmd"] = self._vx_cmd.clone()
            log["State/vx_error"] = (state.lin_vel[..., 0] - self._vx_cmd).abs()  # m/s, 0 = follows the command
        if self._clock_step is not None:
            stance = (state.foot_phase < self._ctx.duty).float()
            log["State/gait_phase_match"] = (stance == state.foot_contact).float().mean(-1)  # 1 = feet follow the clock

        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            mjw.forward(self.mjw_model, self.mjw_data)

        obs = self._get_obs()
        extras = {"time_outs": timed_out, "log": log}
        return obs, rewards, dones, extras
