"""걷기(gait) 학습용 대시보드.

dashboard_server.py를 건드리지 않고, 발 접촉 관련 새 그래프
(feet_air_time, gait_symmetry, feet_in_contact)의 설명만 추가해서 그대로
재사용한다 — 다른 모든 그래프(Loss/*, Train/*, 기존 Reward/*)는 완전히 동일.

사용법:
    python train/dashboard_server_gait.py --logdir train/logs/walk_gait_demo --port 6009
"""
from __future__ import annotations

import dashboard_server as base

EXTRA_META: dict[str, dict[str, str]] = {
    "Reward/feet_air_time": {
        "title": "Feet Air Time Reward",
        "desc": "Bonus given when a foot lands after being airborne, proportional to how long it "
        "was in the air. Encourages actually lifting and stepping instead of shuffling/dragging.",
        "good": "Should climb above 0 and stay positive as it learns to take real steps",
    },
    "Reward/gait_symmetry": {
        "title": "Gait Symmetry (Trot Pattern)",
        "desc": "1.0 when diagonal leg pairs (FL&BR, FR&BL) share the same contact state (both "
        "down or both up) - the classic quadruped trot pattern.",
        "good": "Should climb toward 1.0",
    },
    "State/feet_in_contact": {
        "title": "Feet Currently in Contact (0-4)",
        "desc": "How many of the 4 feet are touching the ground at this instant.",
        "good": "Should mostly hover around 2 for a trot gait (2 feet down, 2 in swing) rather "
        "than staying at 4 (standing still) or 0 (airborne/falling)",
    },
}

base.GRAPH_META.update(EXTRA_META)

if __name__ == "__main__":
    base.main()
