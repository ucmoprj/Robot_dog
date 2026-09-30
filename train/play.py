"""Play a trained policy (checkpoint) in the CPU mujoco viewer.

Instead of view_task.py's zero/random policy, this loads the neural network policy
trained with rsl_rl and drives the robot with it. Training (GPU, mujoco_warp,
thousands in parallel) and playback (CPU, mujoco, one env) are separate processes,
but they share the observation definition in env.py/reward_terms.py, so the network
sees exactly the same inputs it saw during training.

Usage (from the project root):
    source .venv/bin/activate
    python train/play.py --task train/tasks/stand.yaml --checkpoint train/logs/live_demo/model_75.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import N_JOINTS, OBS_DIM, QuadrupedEnv, load_task_config  # noqa: E402
from ppo_config import build_train_cfg  # noqa: E402


class _DummyEnv:
    """The minimal VecEnv stand-in OnPolicyRunner needs to load a checkpoint.

    It runs no physics. It only rebuilds the same network structure used in
    training and loads the state_dict. The actual physics runs in
    QuadrupedEnv (CPU mujoco) below."""

    def __init__(self, device: str):
        self.num_envs = 1
        self.num_actions = N_JOINTS
        self.device = device
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": torch.zeros(1, OBS_DIM, device=self.device)}, batch_size=[1])


def load_policy(checkpoint: str, device: str = "cpu"):
    from rsl_rl.runners import OnPolicyRunner  # heavy import, load only when needed

    dummy_env = _DummyEnv(device)
    train_cfg = build_train_cfg()
    runner = OnPolicyRunner(dummy_env, train_cfg, log_dir=None, device=device)
    runner.load(checkpoint, map_location=device)
    return runner.get_inference_policy(device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "stand.yaml"))
    parser.add_argument("--checkpoint", required=True, help="e.g. train/logs/live_demo/model_75.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    env = QuadrupedEnv(cfg, seed=args.seed)
    obs = env.reset()

    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, device=args.device)
    print(f"Loaded (task: {cfg['task']}). Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            obs_td = TensorDict({"policy": obs_t}, batch_size=[1])
            with torch.inference_mode():
                action = policy(obs_td).squeeze(0).cpu().numpy()  # deterministic (mean) action

            obs, reward, done, info = env.step(action)
            if done:
                print(f"Episode ended (reward={reward:.2f}) -> reset")
                obs = env.reset()

            viewer.sync()
            elapsed = time.time() - step_start
            sleep_time = env.dt * env.control_decimation - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
