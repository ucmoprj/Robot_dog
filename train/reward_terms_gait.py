"""Extra rewards for a natural gait. Extends reward_terms.py in a new file without
touching it (so the existing stand/walk_forward training stays comparable).

Reuses the existing REGISTRY and SimState, and adds two rewards that use the foot
contact sensors (fl_foot_contact etc., newly added in dog_mujoco/robot_gait.xml):

    - feet_air_time: when a foot is actually lifted (in the air) and put down again,
      reward the time it spent in the air. This blocks the shortcut of "sliding along
      while barely moving the legs" and directly encourages taking real steps.
    - gait_symmetry: the trot pattern, the typical quadruped gait. Encourages the
      diagonal leg pairs (FL&BR, FR&BL) to share the same contact state.

Note: the foot contact information here is used only for rewards and never enters
the observation (network input). So the real robot needs no foot pressure sensors
(same principle as env.py; see the SimState description).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from reward_terms import REGISTRY, SimState


@dataclass
class GaitSimState(SimState):
    """SimState extended with foot contact fields."""

    foot_contact: torch.Tensor      # (..., 4) 0.0/1.0, order [fl, fr, bl, br]
    air_time_bonus: torch.Tensor    # (...,4) air time of feet that "just landed" this step (0 otherwise)


def reward_feet_air_time(state: GaitSimState, ctx) -> torch.Tensor:
    """At the moment a foot lands after being in the air, reward the time it was up.
    env_gait.py/vec_env_gait.py fill in a value only at landing and 0 otherwise,
    so this just sums them."""
    return state.air_time_bonus.sum(-1)


def reward_gait_symmetry(state: GaitSimState, ctx) -> torch.Tensor:
    """Approaches 1.0 the more the diagonal leg pairs (FL&BR, FR&BL) share the same
    contact state (both on the ground / both in the air). Encourages a trot."""
    fl, fr, bl, br = (
        state.foot_contact[..., 0],
        state.foot_contact[..., 1],
        state.foot_contact[..., 2],
        state.foot_contact[..., 3],
    )
    diag1_match = 1.0 - (fl - br).abs()  # 1.0 when FL and BR match
    diag2_match = 1.0 - (fr - bl).abs()  # 1.0 when FR and BL match
    return 0.5 * (diag1_match + diag2_match)


REGISTRY_GAIT = dict(REGISTRY)
REGISTRY_GAIT.update(
    {
        "feet_air_time": reward_feet_air_time,
        "gait_symmetry": reward_gait_symmetry,
    }
)
