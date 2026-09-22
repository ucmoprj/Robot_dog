"""자연스러운 보행(gait)을 위한 추가 보상 — reward_terms.py는 건드리지 않고
새 파일로 확장한다 (기존 stand/walk_forward 학습과 비교 가능하게 유지).

기존 REGISTRY, SimState를 그대로 가져와서 재사용하고, 발 접촉 센서
(dog_mujoco/robot_gait.xml에서 새로 추가한 fl_foot_contact 등)를 활용하는
보상 두 개를 추가한다:

    - feet_air_time: 발을 실제로 들어올렸다가(공중) 다시 딛을 때, 공중에
      떠 있던 시간만큼 보상. "다리를 거의 안 움직이고 미끄러지듯 이동"하는
      편법을 막고, 실제로 한 걸음씩 내딛는 행동을 직접 장려한다.
    - gait_symmetry: 4족보행의 대표 걸음걸이인 트롯(trot) 패턴 — 대각선
      다리 쌍(FL&BR, FR&BL)이 같은 접촉 상태를 갖도록 유도.

주의: 여기서 쓰는 발 접촉 정보는 보상 계산에만 사용되고 관측(신경망 입력)에는
들어가지 않는다 — 그래서 실물 로봇에 발 압력센서를 달 필요가 없다 (env.py와
동일한 원칙, SimState 설명 참고).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from reward_terms import REGISTRY, SimState


@dataclass
class GaitSimState(SimState):
    """SimState에 발 접촉 관련 필드를 추가한 확장판."""

    foot_contact: torch.Tensor      # (..., 4) 0.0/1.0, 순서 [fl, fr, bl, br]
    air_time_bonus: torch.Tensor    # (...,4) 이번 스텝에 "방금 착지"한 발의 공중체류시간 (그 외엔 0)


def reward_feet_air_time(state: GaitSimState, ctx) -> torch.Tensor:
    """발이 공중에 떠 있다가 착지하는 순간, 떠 있던 시간만큼 보상.
    env_gait.py/vec_env_gait.py가 착지 시점에만 값을 채워 넣고 그 외엔 0이므로
    여기서는 그대로 더하기만 하면 된다."""
    return state.air_time_bonus.sum(-1)


def reward_gait_symmetry(state: GaitSimState, ctx) -> torch.Tensor:
    """대각선 다리 쌍(FL&BR, FR&BL)이 같은 접촉 상태(둘 다 땅 / 둘 다 공중)를
    가질수록 1.0에 가까운 보상 — 트롯 보행 패턴을 유도."""
    fl, fr, bl, br = (
        state.foot_contact[..., 0],
        state.foot_contact[..., 1],
        state.foot_contact[..., 2],
        state.foot_contact[..., 3],
    )
    diag1_match = 1.0 - (fl - br).abs()  # FL, BR 같은 상태면 1.0
    diag2_match = 1.0 - (fr - bl).abs()  # FR, BL 같은 상태면 1.0
    return 0.5 * (diag1_match + diag2_match)


REGISTRY_GAIT = dict(REGISTRY)
REGISTRY_GAIT.update(
    {
        "feet_air_time": reward_feet_air_time,
        "gait_symmetry": reward_gait_symmetry,
    }
)
