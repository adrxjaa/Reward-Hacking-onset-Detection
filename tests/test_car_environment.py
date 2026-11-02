"""
test_car_environment.py - Unit tests for ContinuousCarEnv and behavioral feature extraction.
"""
import pytest
import numpy as np
import gymnasium as gym

from rhd.envs.continuous_car import ContinuousCarConfig, ContinuousCarEnv, CAR_ACTIONS
from rhd.envs import create_environment, create_env_config


def test_continuous_car_initialization_and_kinematics():
    cfg = ContinuousCarConfig(track_length=50.0, track_width=20.0, max_speed=5.0)
    env = ContinuousCarEnv(cfg, expose_eval_info=False)

    obs, info = env.reset(seed=42)
    assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Normalized obs out of [0, 1] range"
    assert "eval" not in info, "Leakage: eval info exposed when expose_eval_info=False"

    # Step action 0: accelerate straight
    next_obs, r, term, trunc, info = env.step(0)
    assert next_obs.shape == (4,)
    assert isinstance(r, float)
    assert "eval" not in info, "Leakage: eval info exposed in step() when expose_eval_info=False"


def test_continuous_car_quarantined_rewards():
    cfg = ContinuousCarConfig(speed_reward_weight=2.0, goal_true_reward=1.0)
    env_eval = ContinuousCarEnv(cfg, expose_eval_info=True)

    obs, info = env_eval.reset(seed=123)
    assert "eval" in info, "Expected eval info when expose_eval_info=True"
    assert info["eval"]["goal_reached"] is False

    # Execute donut maneuver (action 3: sharp turn + throttle)
    for _ in range(20):
        obs, r, term, trunc, info = env_eval.step(3)
        if term or trunc:
            break

    e_info = info["eval"]
    assert "true_reward" in e_info
    assert "circle_count" in e_info
    assert e_info["circle_count"] > 0, "Expected circle count to increase with exploit action"


def test_continuous_car_factory_creation():
    env = create_environment("continuous_car", speed_reward_weight=1.5)
    assert isinstance(env, ContinuousCarEnv)
    assert env.config.speed_reward_weight == 1.5
