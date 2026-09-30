"""GPU-parallel PPO training script (gait-reward version, uses rsl_rl).

A separate file so train.py stays untouched and results can be compared side by
side with the existing walk_forward.yaml training. The PPO/network settings
(ppo_config.py) are reused as-is, since the observation/action sizes are unchanged.

Usage (from the project root):
    source .venv/bin/activate
    python train/train_gait.py --task train/tasks/walk_forward_gait.yaml --num_envs 2048 --run_name walk_gait_demo

Checkpoints and TensorBoard logs are saved to train/logs/<run_name>/.
To view:
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

    from rsl_rl.runners import OnPolicyRunner  # heavy import, load only when needed

    train_cfg = build_train_cfg(args.num_steps_per_env, args.save_interval)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    print(f"Task: {cfg['task']}  |  envs: {args.num_envs}  |  logs: {log_dir}")
    runner.learn(num_learning_iterations=args.iterations)

    print(f"Training done. Checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
