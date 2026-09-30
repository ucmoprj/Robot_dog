"""GPU 병렬 PPO 학습 스크립트 (gait 보상 버전, rsl_rl 사용).

train.py를 건드리지 않고 별도 파일로 만들었다 — 기존 walk_forward.yaml 학습
결과와 나란히 비교하기 위해서다. PPO/네트워크 설정(ppo_config.py)은 동일하게
재사용한다 (관측/행동 차원이 안 바뀌었으므로).

사용법 (프로젝트 루트에서):
    source .venv/bin/activate
    python train/train_gait.py --task train/tasks/walk_forward_gait.yaml --num_envs 2048 --run_name walk_gait_demo

체크포인트/텐서보드 로그는 train/logs/<run_name>/ 에 저장된다.
확인:
    tensorboard --logdir train/logs --bind_all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_gait import load_task_config  # noqa: E402
from ppo_config import build_train_cfg  # noqa: E402
from vec_env_gait import VecQuadrupedEnvGait  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "walk_forward_gait.yaml"))
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_task_config(args.task)
    env = VecQuadrupedEnvGait(cfg, num_envs=args.num_envs, device=args.device, seed=args.seed)

    run_name = args.run_name or f"{cfg['task']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(__file__).parent / "logs" / run_name)

    from rsl_rl.runners import OnPolicyRunner  # 무거운 임포트라 필요할 때만 로드

    train_cfg = build_train_cfg(args.num_steps_per_env, args.save_interval)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    print(f"태스크: {cfg['task']}  |  환경 수: {args.num_envs}  |  로그: {log_dir}")
    runner.learn(num_learning_iterations=args.iterations)

    print(f"학습 완료. 체크포인트: {log_dir}")


if __name__ == "__main__":
    main()
