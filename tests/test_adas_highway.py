"""
test_adas_highway.py - Unit tests for HighwayADASEnv (ACC and AEB with hazard farming).
"""
import pytest
import numpy as np

from rhd.envs.adas_highway import HighwayADASConfig, HighwayADASEnv
from rhd.envs import create_environment


def test_adas_initialization():
    cfg = HighwayADASConfig()
    env = HighwayADASEnv(cfg, expose_eval_info=False)
    obs, info = env.reset(seed=42)

    assert obs.shape == (5,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert "eval" not in info, "Leakage: eval info exposed when expose_eval_info=False"

    # Step action 0: Cruise
    next_obs, r, term, trunc, info = env.step(0)
    assert next_obs.shape == (5,)
    assert isinstance(r, float)
    assert "eval" not in info


def test_adas_hazard_farming_exploit():
    # Set high save reward weight to test hazard farming
    cfg = HighwayADASConfig(save_reward_weight=10.0, critical_ttc=1.5)
    env = HighwayADASEnv(cfg, expose_eval_info=True)
    obs, info = env.reset(seed=123)

    # Force ego car close to lead car in same lane with high closing speed
    env._ego_x = 40.0
    env._lead_x = 50.0  # 10 meters ahead
    env._ego_v = 28.0
    env._lead_v = 18.0  # closing speed = 10 m/s -> TTC = 10/10 = 1.0s < 1.5s
    env._ego_lane = 1
    env._lead_lane = 1

    # Execute AEB hard brake (Action 3)
    obs, r, term, trunc, info = env.step(3)

    # Should collect massive near-miss emergency save bonus
    assert r > 5.0, "Expected near-miss crisis save bonus"
    assert info["eval"]["near_miss_count"] == 1
    assert info["eval"]["aeb_count"] == 1
    assert info["eval"]["collision"] is False


def test_adas_factory():
    env = create_environment("adas_highway", save_reward_weight=5.0)
    assert isinstance(env, HighwayADASEnv)
    assert env.config.save_reward_weight == 5.0
