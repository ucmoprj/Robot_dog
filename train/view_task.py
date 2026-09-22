"""env.py의 QuadrupedEnv를 MuJoCo 인터랙티브 뷰어로 직접 보면서 확인하는 스크립트.
아직 학습 전이므로 "무작위 행동" 또는 "가만히 있기(zero)" 정책만 보여준다.
학습된 정책이 생기면 --policy 부분만 그 정책으로 바꾸면 된다 (나중에 play.py로 발전).

사용법 (프로젝트 루트에서):
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

    print(f"태스크: {cfg['task']}  |  정책: {args.policy}  |  창을 닫으면 종료됩니다.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if args.policy == "random":
                action = rng.uniform(lo, hi)
            else:
                action = np.zeros(12)

            obs, reward, done, info = env.step(action)
            if done:
                print(f"에피소드 종료 (last reward={reward:.2f}) -> 리셋")
                obs = env.reset()

            viewer.sync()

            # 제어 주기(control_decimation * timestep)에 맞춰 재생 속도를 실제 시간과 동기화
            elapsed = time.time() - step_start
            sleep_time = env.dt * env.control_decimation - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
