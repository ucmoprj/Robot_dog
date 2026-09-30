"""PPO / network hyperparameters (shared by train.py and play.py).

Reloading a checkpoint (play.py) needs exactly the network structure used in
training, so this config lives in one place and both scripts use it.
"""
from __future__ import annotations


def build_train_cfg(num_steps_per_env: int = 24, save_interval: int = 50) -> dict:
    """PPO / network hyperparameters. Usually left alone when the task changes
    (experiments tune reward_weights in tasks/*.yaml instead)."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 64],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128, 64],
            "activation": "elu",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
    }
