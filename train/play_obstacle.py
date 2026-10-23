"""Play a checkpoint from train_obstacle.py in the CPU mujoco viewer, on the bar course.

The network output is turned into joint targets the same way as in training
(stand pose + scale × action, env_obstacle.py), which play.py / play_imu.py do not do,
so obstacle checkpoints must be played with this file. No IMU noise is added here.
With a gait clock in the task yaml the network has 32 inputs instead of 30, so the
policy is loaded with that size (play.load_policy assumes 30).
Every second it prints how far the robot got and how many bars it has passed.

With a velocity_command block in the task yaml (walk_obstacle_speed.yaml), the forward speed
command can be changed live: Up / Down arrow keys in the viewer window = ±0.01 m/s, within
velocity_command.range. It starts at --vx (default: velocity_command.play) and is shown in the
top-left corner of the viewer.

Usage (from the project root):
    source .venv/bin/activate
    python train/play_obstacle.py --checkpoint train/logs/walk_obstacle/model_<N>.pt
    python train/play_obstacle.py --task train/tasks/walk_obstacle_speed.yaml --checkpoint train/logs/walk_obstacle_speed/model_<N>.pt --vx 0.03
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
from env import N_JOINTS  # noqa: E402
from env_obstacle import QuadrupedEnvObstacle, clock_frequency, load_task_config, obs_dim  # noqa: E402
from ppo_config import build_train_cfg  # noqa: E402


class _DummyEnv:
    """Same as play._DummyEnv, but with the observation size of this task (30 or 32)."""

    def __init__(self, n_obs: int, device: str):
        self.num_envs = 1
        self.num_actions = N_JOINTS
        self.device = device
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.cfg = {}
        self._n_obs = n_obs

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": torch.zeros(1, self._n_obs, device=self.device)}, batch_size=[1])


def load_policy(checkpoint: str, n_obs: int, device: str = "cpu"):
    from rsl_rl.runners import OnPolicyRunner  # heavy import, load only when needed

    runner = OnPolicyRunner(_DummyEnv(n_obs, device), build_train_cfg(), log_dir=None, device=device)
    runner.load(checkpoint, map_location=device)
    return runner.get_inference_policy(device=device)


def bars_passed(model: mujoco.MjModel, x: float) -> int:
    """Number of bar_* geoms whose far edge is behind the body origin."""
    n = 0
    for i in range(model.ngeom):
        if model.geom(i).name.startswith("bar_") and x > model.geom_pos[i][0] + model.geom_size[i][0]:
            n += 1
    return n


KEY_UP, KEY_DOWN = 265, 264  # GLFW key codes
VX_KEY_STEP = 0.01           # m/s per key press


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=str(Path(__file__).parent / "tasks" / "walk_obstacle.yaml"))
    parser.add_argument("--checkpoint", required=True, help="e.g. train/logs/walk_obstacle/model_3000.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vx", type=float, default=None, help="starting speed command, m/s (velocity_command tasks only)")
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    env = QuadrupedEnvObstacle(cfg, seed=args.seed)
    vel_cmd = cfg["velocity_command"]
    if vel_cmd is not None and args.vx is not None:
        env.set_command(min(max(args.vx, vel_cmd["range"][0]), vel_cmd["range"][1]))
    elif args.vx is not None:
        print("--vx ignored: this task has no velocity_command block (fixed target_vx)")
    pending_vx = [env.vx_cmd]  # written by the viewer's key callback, applied in the loop below

    def on_key(key: int) -> None:
        if vel_cmd is None or key not in (KEY_UP, KEY_DOWN):
            return
        step = VX_KEY_STEP if key == KEY_UP else -VX_KEY_STEP
        lo, hi = vel_cmd["range"]
        pending_vx[0] = round(min(max(pending_vx[0] + step, lo), hi), 3)

    def speed_text() -> str:
        if cfg["gait_clock"] is None:
            return f"vx_cmd {env.vx_cmd:.3f} m/s"
        return f"vx_cmd {env.vx_cmd:.3f} m/s   clock {clock_frequency(cfg, env.vx_cmd):.2f} Hz"

    obs = env.reset()
    start_x = float(env.data.qpos[0])

    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, obs_dim(cfg), device=args.device)
    print(f"Loaded (task: {cfg['task']}). Close the viewer window to exit.")
    if vel_cmd is not None:
        print(f"Speed command: Up/Down arrow keys in the viewer, ±{VX_KEY_STEP} m/s, range {vel_cmd['range']}. {speed_text()}")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if pending_vx[0] != env.vx_cmd:
                env.set_command(pending_vx[0])
                print(speed_text())
            if vel_cmd is not None:
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "speed command", speed_text().replace("vx_cmd ", "")))

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            obs_td = TensorDict({"policy": obs_t}, batch_size=[1])
            with torch.inference_mode():
                action = policy(obs_td).squeeze(0).cpu().numpy()  # deterministic (mean) action

            obs, reward, done, info = env.step(action)
            x = float(env.data.qpos[0])
            if env._step_count % 50 == 0:
                vx = float(env.data.qvel[0])
                print(f"t={env._step_count * 0.02:4.1f}s  distance={x - start_x:5.2f} m  bars passed={bars_passed(env.model, x)}  vx={vx:.3f} (cmd {env.vx_cmd:.3f}) m/s")
            if done:
                print(f"Episode ended: distance={x - start_x:.2f} m, bars passed={bars_passed(env.model, x)} -> reset")
                obs = env.reset()
                start_x = float(env.data.qpos[0])

            viewer.sync()
            elapsed = time.time() - step_start
            sleep_time = env.dt * env.control_decimation - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
