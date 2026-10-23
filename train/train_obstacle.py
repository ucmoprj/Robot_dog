"""GPU-parallel PPO training: walk forward over ~1 cm bars (uses rsl_rl).

A separate file so train.py / train_gait.py / train_imu.py stay untouched. The environment
is vec_env_obstacle.py: real IMU position + noise, foot contacts, bars in the floor, and
actions as offsets from the stand pose. PPO/network settings (ppo_config.py) are reused,
since the observation/action sizes are unchanged (30 / 12).

Usage (from the project root):
    source .venv/bin/activate
    python train/train_obstacle.py --task train/tasks/walk_obstacle.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name walk_obstacle

To continue from a finished run's weights (only when the observation size is the same), add
    --resume train/logs/walk_obstacle_speed/model_2999.pt
The iteration count restarts at 0, so the new run saves model_0 ... model_<iterations-1>.

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
from env_obstacle import load_task_config  # noqa: E402
from ppo_config import build_train_cfg  # noqa: E402
from vec_env_obstacle import VecQuadrupedEnvObstacle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "walk_obstacle.yaml"))
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to start from, e.g. train/logs/walk_obstacle_speed/model_2999.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_task_config(args.task)
    env = VecQuadrupedEnvObstacle(cfg, num_envs=args.num_envs, device=args.device, seed=args.seed)

    run_name = args.run_name or f"{cfg['task']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(__file__).parent / "logs" / run_name)

    from rsl_rl.runners import OnPolicyRunner  # heavy import, load only when needed

    train_cfg = build_train_cfg(args.num_steps_per_env, args.save_interval)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)
    if args.resume:
        runner.load(args.resume, map_location=args.device)  # weights + optimizer state
        runner.current_learning_iteration = 0  # number this run's checkpoints from 0
        print(f"Resumed from {args.resume}")

    print(f"Task: {cfg['task']}  |  envs: {args.num_envs}  |  action scale: {cfg['action_scale']}  |  logs: {log_dir}")
    runner.learn(num_learning_iterations=args.iterations)

    print(f"Training done. Checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
