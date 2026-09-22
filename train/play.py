"""학습된 정책(체크포인트)을 CPU mujoco 뷰어에서 재생한다.

view_task.py의 zero/random 정책 대신, rsl_rl로 학습한 신경망 정책을 그대로
불러와서 로봇을 움직인다. 학습(GPU, mujoco_warp, 수천 개 병렬)과 재생(CPU,
mujoco, 1개 환경)은 별개 프로세스이며, env.py/reward_terms.py의 관측 정의를
공유하므로 학습 때 본 것과 똑같은 입력을 신경망에 넣어준다.

사용법 (프로젝트 루트에서):
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
    """OnPolicyRunner가 체크포인트를 불러올 때 요구하는 최소한의 VecEnv 흉내.

    실제 물리 시뮬레이션은 하지 않는다 — 신경망 구조를 학습 때와 동일하게
    복원하고 state_dict를 로딩하는 용도로만 쓰인다. 실제 물리는 아래
    QuadrupedEnv(CPU mujoco)가 담당한다."""

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
    from rsl_rl.runners import OnPolicyRunner  # 무거운 임포트라 필요할 때만 로드

    dummy_env = _DummyEnv(device)
    train_cfg = build_train_cfg()
    runner = OnPolicyRunner(dummy_env, train_cfg, log_dir=None, device=device)
    runner.load(checkpoint, map_location=device)
    return runner.get_inference_policy(device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "stand.yaml"))
    parser.add_argument("--checkpoint", required=True, help="예: train/logs/live_demo/model_75.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    env = QuadrupedEnv(cfg, seed=args.seed)
    obs = env.reset()

    print(f"체크포인트 로딩 중: {args.checkpoint}")
    policy = load_policy(args.checkpoint, device=args.device)
    print(f"로딩 완료 (태스크: {cfg['task']}). 뷰어 창을 닫으면 종료됩니다.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            obs_td = TensorDict({"policy": obs_t}, batch_size=[1])
            with torch.inference_mode():
                action = policy(obs_td).squeeze(0).cpu().numpy()  # 결정론적(평균) 행동 사용

            obs, reward, done, info = env.step(action)
            if done:
                print(f"에피소드 종료 (reward={reward:.2f}) -> 리셋")
                obs = env.reset()

            viewer.sync()
            elapsed = time.time() - step_start
            sleep_time = env.dt * env.control_decimation - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
