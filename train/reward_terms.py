"""Reusable reward terms.

So the same reward formulas work with either physics backend (single CPU `mujoco`
or batched GPU `mujoco_warp`), every input arrives as a torch tensor:

    - env.py (CPU, single env): values from mujoco.MjData wrapped as torch tensors
      (shape e.g. gyro -> (3,))
    - vec_env.py (GPU, batch of N): torch views taken directly from mujoco_warp data
      (shape e.g. gyro -> (N, 3))

As long as "the last dimension is the physical quantity", the functions below work
for a batch of 1 or 4096 (using torch broadcasting and last-axis reductions).

Every function has the signature (state: SimState, ctx) -> torch.Tensor.

To add a new reward idea:
    1) add a function here
    2) register it in REGISTRY below with one line
Nothing else changes: env.py, vec_env.py and the training loop (train.py) stay as they are.

Note: everything in SimState (orientation up_z, position, velocity, actuator_force,
...) is privileged simulator information. The real robot (12 x MG90S + 1 IMU) cannot
measure these directly. That is fine for rewards, which are computed only inside the
simulator during training and never run on hardware. Observations (_get_obs in
env/vec_env), however, must only use values the real sensors can reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimState:
    """Backend-independent container holding only what reward computation needs.

    In every field the last dimension is the physical quantity. A single env has
    no batch dimension (e.g. gyro.shape == (3,)); a batched env adds a leading (N,)
    (e.g. gyro.shape == (N, 3)).
    """

    height: torch.Tensor               # (...,)      body z height
    up_z: torch.Tensor                 # (...,)      uprightness, 1.0 = fully upright
    gyro: torch.Tensor                 # (..., 3)    angular velocity (gyro)
    lin_vel: torch.Tensor              # (..., 3)    body linear velocity (vx, vy, vz)
    ang_vel_z: torch.Tensor            # (...,)      yaw rate
    actuator_force_norm: torch.Tensor  # (..., 12)   torque normalized by forcerange
    action: torch.Tensor               # (..., 12)   command this step
    prev_action: torch.Tensor          # (..., 12)   command previous step


def reward_upright(state: SimState, ctx) -> torch.Tensor:
    """Upright reward. Shrinks as the body tips over (max 1.0)."""
    return state.up_z


def reward_height(state: SimState, ctx) -> torch.Tensor:
    """Approaches 1.0 the closer the body is to the target height."""
    return torch.exp(-((state.height - ctx.target_height) ** 2) / ctx.height_scale ** 2)


def penalty_ang_vel(state: SimState, ctx) -> torch.Tensor:
    """Penalty for body shaking (large angular velocity)."""
    return -(state.gyro ** 2).sum(-1)


def penalty_action_rate(state: SimState, ctx) -> torch.Tensor:
    """Penalty for commands (target joint angles) that jump between steps.
    Protects the servos and favors behavior compatible with the hardware rate limit (README: 4 rad/s)."""
    return -((state.action - state.prev_action) ** 2).sum(-1)


def penalty_torque(state: SimState, ctx) -> torch.Tensor:
    """Penalty on actuator force (torque) normalized by forcerange. Encourages moving
    efficiently within the MG90S torque budget (0.06 N·m)."""
    return -(state.actuator_force_norm ** 2).sum(-1)


def reward_alive(state: SimState, ctx) -> torch.Tensor:
    """Survival bonus for every step alive (not yet fallen)."""
    return torch.ones_like(state.height)


def reward_velocity_track(state: SimState, ctx) -> torch.Tensor:
    """How well the forward velocity tracks target_vx."""
    vx = state.lin_vel[..., 0]
    return torch.exp(-((vx - ctx.target_vx) ** 2) / ctx.velocity_scale ** 2)


def penalty_lateral(state: SimState, ctx) -> torch.Tensor:
    """Penalty for drifting sideways (vy) or bouncing (vz)."""
    vy, vz = state.lin_vel[..., 1], state.lin_vel[..., 2]
    return -(vy.abs() + 0.5 * vz.abs())


def penalty_yaw(state: SimState, ctx) -> torch.Tensor:
    """Penalty for turning (yaw). Encourages walking straight."""
    return -state.ang_vel_z.abs()


REGISTRY = {
    "upright": reward_upright,
    "height": reward_height,
    "ang_vel_penalty": penalty_ang_vel,
    "action_rate_penalty": penalty_action_rate,
    "torque_penalty": penalty_torque,
    "alive": reward_alive,
    "velocity_track": reward_velocity_track,
    "lateral_penalty": penalty_lateral,
    "yaw_penalty": penalty_yaw,
}


def is_fallen(state: SimState, fall_height: float, fall_upright: float) -> torch.Tensor:
    """Fall check. Reused as-is by env.py and vec_env.py."""
    return (state.height < fall_height) | (state.up_z < fall_upright)
