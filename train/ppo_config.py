"""PPO/신경망 하이퍼파라미터 (train.py와 play.py가 공유).

체크포인트를 다시 불러오려면(play.py) 학습 때와 똑같은 네트워크 구조가
필요하므로, 이 설정을 한 곳에 두고 두 스크립트가 같이 쓴다.
"""
from __future__ import annotations


def build_train_cfg(num_steps_per_env: int = 24, save_interval: int = 50) -> dict:
    """PPO/네트워크 하이퍼파라미터. 태스크가 바뀌어도 보통 그대로 둔다
    (실험할 때 건드리는 건 reward_weights = tasks/*.yaml 쪽)."""
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
