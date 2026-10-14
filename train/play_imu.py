"""Play a checkpoint from train_imu.py in the CPU mujoco viewer.

Same as play.py, but the physics model is scene_imu.xml (IMU on the inside floor of
the body, z = -17 mm), so the network sees the IMU from where the real BNO085 is.
No noise is added here: this shows what the policy does, not a noise test.

Usage (from the project root):
    source .venv/bin/activate
    python train/play_imu.py --task train/tasks/stand.yaml --checkpoint train/logs/stand_imu/model_<N>.pt
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
from env import QuadrupedEnv, load_task_config  # noqa: E402
from play import load_policy  # noqa: E402
from vec_env_imu import ROBOT_XML_IMU  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "stand.yaml"))
    parser.add_argument("--checkpoint", required=True, help="e.g. train/logs/stand_imu/model_3000.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    env = QuadrupedEnv(cfg, xml_path=ROBOT_XML_IMU, seed=args.seed)
    obs = env.reset()

    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, device=args.device)
    print(f"Loaded (task: {cfg['task']}, IMU model). Close the viewer window to exit.")

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
