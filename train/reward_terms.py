"""재사용 가능한 보상 부품(reward term) 모음.

이 파일은 물리 백엔드(단일 CPU `mujoco` 또는 GPU 배치 `mujoco_warp`)에 상관없이
동일한 보상 수식을 쓸 수 있도록, 두 값을 다 torch 텐서로 통일해서 받는다:

    - env.py (CPU, 단일 환경): mujoco.MjData에서 값을 뽑아 torch 텐서로 감싼다
      (모양이 예: gyro -> (3,) )
    - vec_env.py (GPU, N개 배치): mujoco_warp 데이터에서 직접 torch 뷰를 얻는다
      (모양이 예: gyro -> (N, 3) )

즉 "마지막 차원이 물리량 차원"이라는 규칙만 지키면, 배치 크기가 1이든 4096이든
아래 함수들은 그대로 동작한다 (torch의 브로드캐스팅/마지막축 reduction을 이용).

각 함수의 시그니처는 (state: SimState, ctx) -> torch.Tensor 로 통일한다.

새 보상 아이디어가 생기면:
    1) 여기에 함수 하나 추가
    2) 아래 REGISTRY에 이름-함수로 한 줄 등록
만 하면 된다. env.py, vec_env.py, 학습 루프(train.py)는 건드릴 필요가 없다.

주의: SimState에 담기는 값(자세 up_z, 위치, 속도, actuator_force 등)은 전부
"시뮬레이터 특권 정보"다. 실물 로봇(MG90S 12개 + IMU 1개)은 이런 값을 직접
측정할 수 없지만, 보상은 학습 중에만 시뮬레이터 안에서 계산되고 실물에는
절대 올라가지 않으므로 문제없다. 반면 관측(observation, env/vec_env의
_get_obs)은 반드시 실물 센서로 재현 가능한 값만 사용해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimState:
    """보상 계산에 필요한 값만 뽑아 담는, 백엔드에 무관한 공통 컨테이너.

    모든 필드는 마지막 차원이 물리량 차원이다. 단일 환경이면 배치 차원이
    없고(예: gyro.shape == (3,)), 배치 환경이면 맨 앞에 (N,)이 붙는다
    (예: gyro.shape == (N, 3)).
    """

    height: torch.Tensor               # (...,)      몸체 z높이
    up_z: torch.Tensor                 # (...,)      직립도, 1.0=완전 직립
    gyro: torch.Tensor                 # (..., 3)    각속도(자이로)
    lin_vel: torch.Tensor              # (..., 3)    몸체 선속도 (vx, vy, vz)
    ang_vel_z: torch.Tensor            # (...,)      yaw 각속도
    actuator_force_norm: torch.Tensor  # (..., 12)   토크/forcerange 정규화값
    action: torch.Tensor               # (..., 12)   이번 스텝 명령
    prev_action: torch.Tensor          # (..., 12)   직전 스텝 명령


def reward_upright(state: SimState, ctx) -> torch.Tensor:
    """직립 자세 보상. 넘어질수록 값이 줄어든다 (최대 1.0)."""
    return state.up_z


def reward_height(state: SimState, ctx) -> torch.Tensor:
    """목표 몸체 높이 근처에 있을수록 1.0에 가까운 보상."""
    return torch.exp(-((state.height - ctx.target_height) ** 2) / ctx.height_scale ** 2)


def penalty_ang_vel(state: SimState, ctx) -> torch.Tensor:
    """몸체가 심하게 흔들릴수록(각속도 클수록) 페널티."""
    return -(state.gyro ** 2).sum(-1)


def penalty_action_rate(state: SimState, ctx) -> torch.Tensor:
    """명령(목표 관절각)이 스텝 사이에 급격히 바뀌는 것에 대한 페널티.
    서보 보호 + 실물 명령 변화율 제한(README: 4 rad/s)과 궁합이 맞는 행동을 유도."""
    return -((state.action - state.prev_action) ** 2).sum(-1)


def penalty_torque(state: SimState, ctx) -> torch.Tensor:
    """액추에이터 힘(토크)을 forcerange로 정규화해 페널티. MG90S 토크 예산(0.06 N·m)
    안에서 효율적으로 움직이도록 유도."""
    return -(state.actuator_force_norm ** 2).sum(-1)


def reward_alive(state: SimState, ctx) -> torch.Tensor:
    """매 스텝 살아있으면(=아직 안 넘어졌으면) 주는 생존 보너스."""
    return torch.ones_like(state.height)


def reward_velocity_track(state: SimState, ctx) -> torch.Tensor:
    """목표 전진 속도(target_vx)를 얼마나 잘 따라가는지."""
    vx = state.lin_vel[..., 0]
    return torch.exp(-((vx - ctx.target_vx) ** 2) / ctx.velocity_scale ** 2)


def penalty_lateral(state: SimState, ctx) -> torch.Tensor:
    """옆으로 새거나(vy) 위아래로 튀는(vz) 움직임 페널티."""
    vy, vz = state.lin_vel[..., 1], state.lin_vel[..., 2]
    return -(vy.abs() + 0.5 * vz.abs())


def penalty_yaw(state: SimState, ctx) -> torch.Tensor:
    """제자리 회전(yaw) 페널티. 똑바로 걷도록 유도."""
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
    """넘어짐 판정. env.py/vec_env.py가 동일하게 재사용한다."""
    return (state.height < fall_height) | (state.up_z < fall_upright)
