"""Watch env.py's QuadrupedEnv in the interactive MuJoCo viewer.
Shows only a "random action" or "hold still (zero)" policy, no trained network
(use play.py for a trained checkpoint).

Usage (from the project root):
    source .venv/bin/activate
    python train/view_task.py --task train/tasks/stand.yaml --policy zero
    python train/view_task.py --task train/tasks/stand.yaml --policy random
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import QuadrupedEnv, load_task_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "stand.yaml"))
    parser.add_argument("--policy", choices=["random", "zero"], default="zero")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    env = QuadrupedEnv(cfg, seed=args.seed)
    obs = env.reset()

    rng = np.random.default_rng(args.seed)
    lo, hi = env._ctrlrange()[:, 0], env._ctrlrange()[:, 1]

    print(f"Task: {cfg['task']}  |  policy: {args.policy}  |  close the window to exit.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if args.policy == "random":
                action = rng.uniform(lo, hi)
            else:
                action = np.zeros(12)

            obs, reward, done, info = env.step(action)
            if done:
                print(f"Episode ended (last reward={reward:.2f}) -> reset")
                obs = env.reset()

            viewer.sync()

            # sync playback to real time at the control period (control_decimation * timestep)
            elapsed = time.time() - step_start
            sleep_time = env.dt * env.control_decimation - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
