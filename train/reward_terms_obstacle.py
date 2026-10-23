"""Extra reward for walking over ~1 cm bars (scene_obstacle.xml). Extends
reward_terms_gait.py in a new file without touching it.

    - feet_clearance: while a foot is in the air (swing), reward lifting it up to
      target_clearance above the floor. The policy cannot see the bars (IMU only, no
      camera), so the way to get over them is to lift every step high enough. Feet
      on the ground get nothing, so standing on three legs with one leg hovering is
      not rewarded more than walking: feet_air_time (paid at landing) and
      velocity_track still have to be earned.
    - raw_action_rate_penalty: squared change of the raw network output between steps.
      action_rate_penalty (reward_terms.py) looks at joint targets, which the command
      rate limit (max_command_rate) already keeps small, so a policy could still chatter
      against the limit. This one sees the output before the limit.
    - joint_vel_penalty: squared joint speeds (rad/s). Against shaking and against
      motions faster than an MG90S can follow.
    - gait_phase_contact / gait_phase_swing_height: only with a gait clock (`gait_clock`
      in the task yaml). The clock gives every foot a phase in [0, 1): stance while
      phase < duty, swing after. The first rewards each foot for being on the ground in
      stance and in the air in swing; the second rewards a swing foot for following a
      smooth lift-and-lower height curve (0 -> target_clearance -> 0). The clock phase is
      also in the observation, so the policy can time its steps to it.
    - heading_track: only with a `heading` block in the task yaml. Rewards keeping the
      start heading (+x, across the bars). yaw_penalty only punishes turning *speed*, so a
      policy that had already turned had no reason to turn back; the walk_obstacle_clock
      policy ended 90-120 deg off course after 10 s. The heading error is also in the
      observation (BNO085 game rotation vector on the real robot), so the policy can see it.
    - body_contact_penalty: number of non-foot collision geoms (knee servo boxes, shins,
      shoulders, body) touching the floor. The walk_obstacle_clock policy walked on its
      knees: the knee servo boxes (*_thigh_collision_1) were on the floor 30-60 % of the
      time, which nothing punished.
    - orientation_penalty: sin² of the body tilt (= roll² + pitch² for small angles). upright
      (reward_terms.py) is cos(tilt), which is flat near upright: 15 deg costs only 0.03, so the
      walk_obstacle_speed policy rocked ±15-20 deg around the diagonal support line of the trot.

Like the foot contacts, foot heights and joint speeds are privileged sim values used only for the
reward, never in the observation, so the real robot needs no extra sensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from reward_terms_gait import REGISTRY_GAIT, GaitSimState

FOOT_RADIUS = 0.005  # m, foot collision sphere (fl_shank_collision_1 etc.); foot site = its center


@dataclass
class ObstacleSimState(GaitSimState):
    """GaitSimState plus foot heights, raw network outputs, joint speeds and gait clock phases."""

    foot_height: torch.Tensor      # (..., 4) world z of the foot sphere centers, order [fl, fr, bl, br]
    raw_action: torch.Tensor       # (..., 12) network output this step (clipped to ±action_clip)
    prev_raw_action: torch.Tensor  # (..., 12) network output previous step
    joint_vel: torch.Tensor        # (..., 12) joint speeds, rad/s
    foot_phase: torch.Tensor       # (..., 4) gait clock phase of each foot in [0, 1) (zeros without a clock)
    yaw_error: torch.Tensor        # (...,) body yaw minus start heading, rad in (-π, π]
    body_contact: torch.Tensor     # (...,) number of non-foot collision geoms touching the floor


def reward_feet_clearance(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Sum over swing feet of (height of the foot bottom above the floor / target), capped at 1."""
    clearance = state.foot_height - FOOT_RADIUS
    lifted = torch.clamp(clearance / ctx.target_clearance, 0.0, 1.0)
    return (lifted * (1.0 - state.foot_contact)).sum(-1)


def penalty_raw_action_rate(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Penalty for network outputs that jump between steps (before the rate limit)."""
    return -((state.raw_action - state.prev_raw_action) ** 2).sum(-1)


def penalty_joint_vel(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Penalty for fast joint motion (rad/s)²."""
    return -(state.joint_vel ** 2).sum(-1)


def reward_gait_phase_contact(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Fraction of feet whose contact matches the clock: on the ground in stance, in the air in swing."""
    stance = (state.foot_phase < ctx.duty).float()
    match = stance * state.foot_contact + (1.0 - stance) * (1.0 - state.foot_contact)
    return match.mean(-1)


SWING_HEIGHT_TOLERANCE = 0.005  # m, height error at which gait_phase_swing_height drops to 1/e


def reward_gait_phase_swing_height(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Mean over swing-phase feet of how closely the foot follows
    floor + target_clearance × sin(π × swing progress)."""
    swing = (state.foot_phase >= ctx.duty).float()
    progress = ((state.foot_phase - ctx.duty) / (1.0 - ctx.duty)).clamp(0.0, 1.0)
    desired = FOOT_RADIUS + ctx.target_clearance * torch.sin(math.pi * progress)
    track = torch.exp(-(((state.foot_height - desired) / SWING_HEIGHT_TOLERANCE) ** 2))
    return (track * swing).sum(-1) / swing.sum(-1).clamp(min=1.0)


def reward_heading_track(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Approaches 1.0 the closer the body yaw is to the start heading."""
    return torch.exp(-((state.yaw_error / ctx.heading_scale) ** 2))


def penalty_orientation(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Penalty for body tilt: sin²(tilt) = 1 - up_z² (roll² + pitch² for small angles, rad²)."""
    return -(1.0 - state.up_z ** 2)


def penalty_body_contact(state: ObstacleSimState, ctx) -> torch.Tensor:
    """Penalty per non-foot collision geom on the floor (knees, shins, shoulders, body)."""
    return -state.body_contact


REGISTRY_OBSTACLE = dict(REGISTRY_GAIT)
REGISTRY_OBSTACLE.update(
    {
        "feet_clearance": reward_feet_clearance,
        "raw_action_rate_penalty": penalty_raw_action_rate,
        "joint_vel_penalty": penalty_joint_vel,
        "gait_phase_contact": reward_gait_phase_contact,
        "gait_phase_swing_height": reward_gait_phase_swing_height,
        "heading_track": reward_heading_track,
        "body_contact_penalty": penalty_body_contact,
        "orientation_penalty": penalty_orientation,
    }
)
