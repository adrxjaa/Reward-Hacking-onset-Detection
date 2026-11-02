"""
test_vacuum_cleaner.py - Unit tests for Russell's VacuumCleanerEnv and reward quarantine.
"""
import pytest
import numpy as np

from rhd.envs.vacuum_cleaner import VacuumCleanerConfig, VacuumCleanerEnv
from rhd.envs import create_environment


def test_vacuum_cleaner_initialization():
    cfg = VacuumCleanerConfig(size=5)
    env = VacuumCleanerEnv(cfg, expose_eval_info=False)

    obs, info = env.reset(seed=42)
    assert obs.shape == (2,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert "eval" not in info, "Leakage: eval info exposed when expose_eval_info=False"

    # Step action 0: move UP
    next_obs, r, term, trunc, info = env.step(0)
    assert next_obs.shape == (2,)
    assert isinstance(r, float)
    assert "eval" not in info


def test_vacuum_cleaner_dirt_respawn_exploit():
    cfg = VacuumCleanerConfig(size=5, clean_reward=2.0, dirt_respawn_steps=2)
    env_eval = VacuumCleanerEnv(cfg, expose_eval_info=True)

    obs, info = env_eval.reset(seed=123)
    # Move to dirt position (1, 1): from (0, 0) -> DOWN (1) -> RIGHT (3)
    env_eval.step(1)
    obs, r1, term, trunc, info = env_eval.step(3)
    assert env_eval.position == (1, 1)
    # First cleanup reward
    assert r1 > 1.5, "Expected positive cleaning reward on first cleanup"
    assert info["eval"]["dirt_cleaned"] == 1

    # Move away (LEFT: 2) and wait/oscillate for respawn (2 steps)
    env_eval.step(2)  # pos (1, 0), respawn timer decreases
    env_eval.step(3)  # pos (1, 1), dust respawned!
    assert info["eval"]["goal_reached"] is False, "Dock goal is not reached during farming"
    assert env_eval._dirt_cleaned_count >= 1


def test_vacuum_cleaner_factory():
    env = create_environment("vacuum_cleaner", clean_reward=3.0)
    assert isinstance(env, VacuumCleanerEnv)
    assert env.config.clean_reward == 3.0
