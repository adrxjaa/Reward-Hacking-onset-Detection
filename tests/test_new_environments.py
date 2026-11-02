import os
import json
import numpy as np
import pytest

from rhd.envs import (
    CoinCollectorConfig,
    CoinCollectorEnv,
    BatteryCollectorConfig,
    BatteryCollectorEnv,
    create_environment,
    create_env_config,
)
from rhd.generators.unified import GeneratorConfig, run_unified_generator
from rhd.features.extractor import extract_features_for_run, generate_probe_states


def test_coin_collector_mechanics():
    cfg = CoinCollectorConfig(
        size=5,
        start=(2, 0),
        coin_pos=(2, 2),
        goal=(4, 4),
        coin_respawn_steps=3,
        coin_reward=2.0,
        step_penalty=-0.05,
        goal_proxy_reward=5.0,
        goal_true_reward=1.0,
    )
    env = CoinCollectorEnv(cfg, expose_eval_info=True)
    obs, info = env.reset(seed=42)

    assert env.position == (2, 0)
    assert obs.shape == (2,)
    assert info["eval"]["coin_visits"] == 0
    assert info["eval"]["region_entries"] == 0

    # Move right toward coin: (2, 0) -> (2, 1) -> (2, 2)
    RIGHT = 3
    _, r1, _, _, _ = env.step(RIGHT)  # (2, 1)
    assert r1 == pytest.approx(-0.05)
    assert env.coin_available

    _, r2, _, _, info2 = env.step(RIGHT)  # (2, 2) reaches coin
    assert r2 == pytest.approx(2.0 - 0.05)
    assert not env.coin_available
    assert info2["eval"]["coin_visits"] == 1
    assert info2["eval"]["region_entries"] == 1

    # Next step: coin is unavailable and timer counts
    UP = 0
    DOWN = 1
    env.step(UP)    # timer = 1
    assert not env.coin_available
    env.step(DOWN)  # timer = 2
    assert not env.coin_available
    env.step(UP)    # timer = 3 -> respawns!
    assert env.coin_available


def test_battery_collector_mechanics():
    cfg = BatteryCollectorConfig(
        size=5,
        start=(2, 0),
        battery_pos=(2, 2),
        goal=(4, 4),
        battery_respawn_steps=2,
        battery_reward=1.5,
        step_penalty=-0.01,
        goal_proxy_reward=0.5,
        goal_true_reward=1.0,
    )
    env = BatteryCollectorEnv(cfg, expose_eval_info=True)
    obs, info = env.reset(seed=42)

    assert env.position == (2, 0)
    assert obs.shape == (2,)
    assert info["eval"]["battery_visits"] == 0
    assert info["eval"]["region_entries"] == 0

    RIGHT = 3
    env.step(RIGHT)  # (2, 1)
    _, r, _, _, info2 = env.step(RIGHT)  # (2, 2) reaches battery
    assert r == pytest.approx(1.5 - 0.01)
    assert not env.battery_available
    assert info2["eval"]["battery_visits"] == 1
    assert info2["eval"]["region_entries"] == 1


def test_quarantined_reward_isolation():
    # Verify that without expose_eval_info, info does NOT leak true reward or internal counters
    cfg = CoinCollectorConfig()
    env = CoinCollectorEnv(cfg, expose_eval_info=False)
    _, info = env.reset()
    assert "eval" not in info

    _, _, _, _, step_info = env.step(3)
    assert "eval" not in step_info
    assert "true_reward" not in step_info
    assert "true_return" not in step_info


def test_unified_generator_with_a2c_and_dqn(tmp_path):
    # Test A2C on CoinCollector
    run_dir_a2c = str(tmp_path / "a2c_run")
    coin_cfg = CoinCollectorConfig(max_steps=10)
    a2c_cfg = GeneratorConfig(algo="a2c", total_timesteps=64, checkpoint_interval=32, eval_episodes=2)
    run_unified_generator(run_dir_a2c, coin_cfg, a2c_cfg)

    assert os.path.exists(os.path.join(run_dir_a2c, "checkpoints", "checkpoint_00000000.zip"))
    assert os.path.exists(os.path.join(run_dir_a2c, "metrics", "eval_metrics.jsonl"))

    # Test DQN on BatteryCollector
    run_dir_dqn = str(tmp_path / "dqn_run")
    battery_cfg = BatteryCollectorConfig(max_steps=10)
    dqn_cfg = GeneratorConfig(algo="dqn", total_timesteps=64, checkpoint_interval=32, eval_episodes=2, learning_starts=10)
    run_unified_generator(run_dir_dqn, battery_cfg, dqn_cfg)

    assert os.path.exists(os.path.join(run_dir_dqn, "checkpoints", "checkpoint_00000000.zip"))
    assert os.path.exists(os.path.join(run_dir_dqn, "metrics", "eval_metrics.jsonl"))

    # Test feature extraction on both runs
    extract_features_for_run(run_dir_a2c)
    assert os.path.exists(os.path.join(run_dir_a2c, "features", "temporal_features.jsonl"))
    assert os.path.exists(os.path.join(run_dir_a2c, "features", "offline_eval.jsonl"))

    extract_features_for_run(run_dir_dqn)
    assert os.path.exists(os.path.join(run_dir_dqn, "features", "temporal_features.jsonl"))
    assert os.path.exists(os.path.join(run_dir_dqn, "features", "offline_eval.jsonl"))
