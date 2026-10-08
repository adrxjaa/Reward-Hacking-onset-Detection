import os
import json
import pytest

from rhd.envs import RegionToggleConfig
from rhd.generators.ppo import PPOConfig, run_generator

def test_generator_initialization_and_metadata(tmp_path):
    run_dir = str(tmp_path / "test_run")
    env_cfg = RegionToggleConfig(region_reward=0.5, max_steps=10)
    ppo_cfg = PPOConfig(seed=123, total_timesteps=1000, checkpoint_interval=500, eval_episodes=2)
    
    run_generator(run_dir, env_cfg, ppo_cfg)
    
    # Check structure
    assert os.path.exists(os.path.join(run_dir, "metadata.json"))
    assert os.path.exists(os.path.join(run_dir, "config.json"))
    assert os.path.exists(os.path.join(run_dir, "checkpoints"))
    assert os.path.exists(os.path.join(run_dir, "metrics", "eval_metrics.jsonl"))
    
    # Check config format
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
        assert cfg["env"]["region_reward"] == 0.5
        assert cfg["ppo"]["seed"] == 123
        
    # Check checkpoints
    ckpts = os.listdir(os.path.join(run_dir, "checkpoints"))
    assert len(ckpts) >= 2 # 0 and 500, or maybe 1000 too
    assert any("checkpoint_00000000" in c for c in ckpts)
    assert any("checkpoint_00000500" in c for c in ckpts)
    
    # Check metrics
    with open(os.path.join(run_dir, "metrics", "eval_metrics.jsonl")) as f:
        lines = f.readlines()
        assert len(lines) >= 2
        m = json.loads(lines[0])
        assert "mean_proxy_return" in m
        assert "mean_true_return" in m
        assert "mean_region_entries" in m
        
def test_seed_handling_reproducibility(tmp_path):
    run_dir_1 = str(tmp_path / "run_1")
    run_dir_2 = str(tmp_path / "run_2")
    
    env_cfg = RegionToggleConfig(max_steps=10)
    ppo_cfg = PPOConfig(seed=42, total_timesteps=1000, checkpoint_interval=1000, eval_episodes=2)
    
    run_generator(run_dir_1, env_cfg, ppo_cfg)
    run_generator(run_dir_2, env_cfg, ppo_cfg)
    
    with open(os.path.join(run_dir_1, "metrics", "eval_metrics.jsonl")) as f:
        m1 = [json.loads(line) for line in f]
    with open(os.path.join(run_dir_2, "metrics", "eval_metrics.jsonl")) as f:
        m2 = [json.loads(line) for line in f]
        
    assert m1 == m2
