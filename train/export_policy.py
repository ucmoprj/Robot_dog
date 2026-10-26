"""Export a trained obstacle-task policy for the real robot (course stage 04).

Writes one .npz file holding the actor network weights plus every setting hardware/run_policy.py
needs to build the same observation and turn the network output into joint angles the same way
as train/env_obstacle.py. The robot PC then needs only numpy + pyserial (no torch, rsl_rl or mujoco).

Usage (on the DGX Spark, from the project root):
    python train/export_policy.py --task train/tasks/walk_obstacle_stable.yaml \
        --checkpoint train/logs/walk_obstacle_stable/model_2999.pt --out policy_stable.npz

Self-check: the exported numpy network is compared with the torch actor on random inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_obstacle import VX_CMD_OBS_SCALE, QuadrupedEnvObstacle, load_task_config, obs_dim  # noqa: E402


def numpy_actor(weights: list[tuple[np.ndarray, np.ndarray]], obs: np.ndarray) -> np.ndarray:
    """Same MLP as rsl_rl's actor: Linear + ELU, ..., Linear (deterministic mean action)."""
    x = obs
    for i, (w, b) in enumerate(weights):
        x = x @ w.T + b
        if i < len(weights) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args()

    cfg = load_task_config(args.task)
    if cfg["gait_clock"] is None or cfg["heading"] is None or cfg["velocity_command"] is None:
        sys.exit("run_policy.py supports tasks with gait_clock + heading + velocity_command (35 inputs), "
                 "e.g. walk_obstacle_speed.yaml / walk_obstacle_stable.yaml")
    n_obs = obs_dim(cfg)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.")})
    weights = [(state[f"mlp.{i}.weight"].numpy().astype(np.float64), state[f"mlp.{i}.bias"].numpy().astype(np.float64))
               for i in layers]
    if weights[0][0].shape[1] != n_obs:
        sys.exit(f"checkpoint has {weights[0][0].shape[1]} inputs, task {args.task} has {n_obs}")
    if any(k.startswith(("obs_normalizer", "normalizer")) for k in state):
        sys.exit("checkpoint has an observation normalizer; numpy_actor does not apply it")

    # Self-check against torch (the actor is Linear/ELU, so rebuild it directly).
    torch_layers = []
    for i, (w, b) in enumerate(weights):
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).double()
        lin.weight.data, lin.bias.data = torch.from_numpy(w), torch.from_numpy(b)
        torch_layers += [lin] + ([torch.nn.ELU()] if i < len(weights) - 1 else [])
    x = np.random.default_rng(0).normal(size=(256, n_obs))
    with torch.no_grad():
        ref = torch.nn.Sequential(*torch_layers)(torch.from_numpy(x)).numpy()
    err = np.abs(numpy_actor(weights, x) - ref).max()
    assert err < 1e-9, err

    env = QuadrupedEnvObstacle(cfg)
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    out = {f"w{i}": w for i, (w, _) in enumerate(weights)} | {f"b{i}": b for i, (_, b) in enumerate(weights)}
    out |= {
        "n_layers": np.array(len(weights)),
        "joint_names": np.array(names),
        "stand_rad": env._default_ctrl,
        "action_scale": env._scale,
        "action_clip": np.array(cfg["action_clip"]),
        "max_command_rate": np.array(cfg["max_command_rate"] or 0.0),  # 0 = no limit
        "ctrlrange": env._ctrlrange().copy(),
        "control_dt": np.array(env.control_dt),
        "frequency_hz": np.array(cfg["gait_clock"]["frequency_hz"], dtype=np.float64).reshape(-1),
        "vx_range": np.array(cfg["velocity_command"]["range"], dtype=np.float64),
        "vx_play": np.array(cfg["velocity_command"]["play"]),
        "vx_obs_scale": np.array(VX_CMD_OBS_SCALE),
        "n_obs": np.array(n_obs),
        "task": np.array(cfg["task"]),
        "checkpoint": np.array(str(args.checkpoint)),
    }
    np.savez(args.out, **out)
    print(f"Saved {args.out}: {n_obs} inputs -> {weights[-1][0].shape[0]} actions, "
          f"{len(weights)} layers, numpy vs torch max diff {err:.1e}")


if __name__ == "__main__":
    main()
