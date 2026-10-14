"""GPU-parallel PPO training with the real BNO085 mounting and IMU noise (uses rsl_rl).

A separate file so train.py / train_gait.py stay untouched. The environment is
vec_env_imu.py: the IMU sits on the inside floor of the body (z = -17 mm) and the
policy sees noisy, biased, possibly 20 ms late gyro/accel readings. PPO/network
settings (ppo_config.py) are reused as-is, since the observation/action sizes are unchanged.

Usage (from the project root):
    source .venv/bin/activate
    python train/train_imu.py --task train/tasks/stand.yaml --num_envs 2048 --iterations 3000 --run_name stand_imu
    python train/train_imu.py --gait --task train/tasks/walk_forward_gait.yaml --num_envs 2048 --iterations 3000 --run_name walk_gait_imu

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
from ppo_config import build_train_cfg  # noqa: E402
from vec_env_imu import VecQuadrupedEnvGaitImu, VecQuadrupedEnvImu  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "stand.yaml"))
    parser.add_argument("--gait", action="store_true",
                        help="gait-reward version (foot contacts, REGISTRY_GAIT), as train_gait.py")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.gait:
        from env_gait import load_task_config
        env_cls = VecQuadrupedEnvGaitImu
    else:
        from env import load_task_config
        env_cls = VecQuadrupedEnvImu

    cfg = load_task_config(args.task)
    env = env_cls(cfg, num_envs=args.num_envs, device=args.device, seed=args.seed)

    run_name = args.run_name or f"{cfg['task']}_imu_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(__file__).parent / "logs" / run_name)

    from rsl_rl.runners import OnPolicyRunner  # heavy import, load only when needed

    train_cfg = build_train_cfg(args.num_steps_per_env, args.save_interval)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)

    print(f"Task: {cfg['task']}  |  gait: {args.gait}  |  envs: {args.num_envs}  |  IMU: {env._imu}  |  logs: {log_dir}")
    runner.learn(num_learning_iterations=args.iterations)

    print(f"Training done. Checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
