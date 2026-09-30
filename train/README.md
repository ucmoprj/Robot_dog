# Training Pipeline — Command Reference

Commands to set up, train, monitor, and test the MG90S quadruped RL policies.
Run everything from the project root (`Robot_dog/`) unless noted otherwise.

## Setup

```bash
cd /home/UCMO/tkang/Robot_dog
source .venv/bin/activate
```

Verify the environment (mujoco, mujoco_warp, torch+CUDA, rsl_rl):

```bash
python -c "import mujoco, mujoco_warp, torch, rsl_rl; print('OK')"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. View the raw robot model (no policy, no training)

Just the physical model reacting to its position-actuator defaults (`ctrl=0`):

```bash
python -m mujoco.viewer --mjcf=dog_mujoco/scene.xml
```

Mouse drag = rotate camera, right-drag = pan, scroll = zoom, `Ctrl`+drag on the
robot = push it, `Space` = pause. The `Control` panel lets you move each joint
by hand.

---

## 2. View an environment under a zero/random policy (pre-training sanity check)

Uses `env.py`'s observation/reward/termination logic, but with a placeholder
policy (no neural network yet) — good for checking a task config before
spending time training it.

```bash
python train/view_task.py --task train/tasks/stand.yaml --policy zero
python train/view_task.py --task train/tasks/stand.yaml --policy random
```

---

## 3. Train (GPU-parallel PPO via rsl_rl)

Each command launches training in the foreground; add `nohup ... &` (or run it
in its own terminal) to keep it running in the background.

**Stand (balance) task:**
```bash
python train/train.py --task train/tasks/stand.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name live_demo
```

**Walk task (basic — body velocity only, no gait shaping):**
```bash
python train/train.py --task train/tasks/walk_forward.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name walk_demo
```

**Walk task (gait version — adds foot-contact-based `feet_air_time` /
`gait_symmetry` rewards, encourages actually lifting the legs instead of
shuffling):**
```bash
python train/train_gait.py --task train/tasks/walk_forward_gait.yaml --num_envs 2048 --iterations 3000 --save_interval 25 --run_name walk_gait_demo
```

Common flags: `--num_envs` (parallel environments), `--iterations`, `--device`
(default `cuda:0`), `--seed`, `--run_name` (log folder name under
`train/logs/`).

Checkpoints and TensorBoard logs are written to `train/logs/<run_name>/`.

---

## 4. Monitor training

**TensorBoard** (auto-detects every run under `train/logs/`, compare multiple
runs with the "Runs" checkboxes on the left):
```bash
tensorboard --logdir train/logs --bind_all
```

**Custom dashboard** (per-graph collapsible English explanations of what each
metric means and which direction is "good" — a companion to TensorBoard, not
a replacement):
```bash
python train/dashboard_server.py --logdir train/logs/live_demo --port 6007
python train/dashboard_server.py --logdir train/logs/walk_demo --port 6008
python train/dashboard_server_gait.py --logdir train/logs/walk_gait_demo --port 6009
```
Open `http://<host>:<port>/` in a browser; auto-refreshes every 3s.

Find the latest checkpoint for a run:
```bash
ls -t train/logs/<run_name>/*.pt | head -1
```

---

## 5. Test / play a trained checkpoint in the MuJoCo viewer

`--task` must match the task the checkpoint was actually trained on (its
reward/termination config affects resets during playback). Works for
checkpoints from **any** of the three training commands above — the
observation/action shapes are identical (only the reward differs), so the
same `play.py` covers all of them, including the gait version.

```bash
python train/play.py --task train/tasks/stand.yaml         --checkpoint train/logs/live_demo/model_<N>.pt
python train/play.py --task train/tasks/walk_forward.yaml       --checkpoint train/logs/walk_demo/model_<N>.pt
python train/play.py --task train/tasks/walk_forward_gait.yaml  --checkpoint train/logs/walk_gait_demo/model_<N>.pt
```

Rendering runs on CPU (`mujoco`), independent from the GPU training process —
training keeps running in the background while you watch a checkpoint.

---

## 6. Useful one-offs

Check GPU utilization while training runs:
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

Check a training run's current iteration / latest reward from its log file:
```bash
grep "Learning iteration" train/logs_<run_name>_stdout.log | tail -1
grep "Mean reward:" train/logs_<run_name>_stdout.log | tail -1
```

Stop a background training run (checkpoints already saved are kept):
```bash
pkill -f "train/train.py --task train/tasks/<name>.yaml"       # stand / walk_forward
pkill -f "train/train_gait.py --task train/tasks/<name>.yaml"  # walk_forward_gait
```

---

## File map

| File | Role |
|---|---|
| `reward_terms.py` | Reusable reward functions + `SimState` container (base) |
| `reward_terms_gait.py` | Extends the above with foot-contact-based gait rewards |
| `env.py` / `env_gait.py` | Single-env CPU wrapper (dev/visualization/`play.py`) |
| `vec_env.py` / `vec_env_gait.py` | GPU-parallel batched env (`mujoco_warp`) for training |
| `ppo_config.py` | PPO/network hyperparameters, shared by all training scripts |
| `train.py` / `train_gait.py` | Training entry points |
| `play.py` | Loads a checkpoint, runs it live in the mujoco viewer |
| `view_task.py` | Views a task under a zero/random (untrained) policy |
| `dashboard_server.py` / `dashboard_server_gait.py` | Local monitoring dashboard |
| `tasks/*.yaml` | Reward weights and task settings — the only thing that should change between experiments |
