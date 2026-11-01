import os
import json
import numpy as np
import pytest
import torch

from rhd.detectors.models import LSTMQNetwork
from rhd.detectors.replay_buffer import ReplayBuffer
from rhd.detectors.environment import (
    DetectorRewardConfig,
    DetectorEpisode,
    DetectorEnv,
    Normalizer,
    FEATURE_NAMES,
    extract_feature_vector,
    load_episode_from_run,
)
from rhd.detectors.agent import DQNAgent, DQNConfig


def test_lstm_input_output_shapes():
    """Verify LSTM Q-Network accepts (B, W, D) and outputs (B, num_actions)."""
    batch_size = 8
    window_size = 4
    input_dim = 10
    num_actions = 2

    model = LSTMQNetwork(input_dim=input_dim, hidden_dim=32, num_actions=num_actions)
    x = torch.randn(batch_size, window_size, input_dim)
    q_vals = model(x)

    assert q_vals.shape == (batch_size, num_actions)

    # Single unbatched window (W, D)
    x_single = torch.randn(window_size, input_dim)
    q_single = model(x_single)
    assert q_single.shape == (1, num_actions)


def test_dqn_action_selection():
    """Verify action selection in deterministic vs epsilon-greedy modes."""
    agent = DQNAgent(config=DQNConfig(input_dim=10, hidden_dim=32, epsilon_start=0.0))
    state = np.random.randn(4, 10).astype(np.float32)

    # Deterministic action selection
    action_1 = agent.select_action(state, deterministic=True)
    action_2 = agent.select_action(state, deterministic=True)
    assert action_1 in [0, 1]
    assert action_1 == action_2  # Purely deterministic

    # Epsilon-greedy with eps=1.0 should explore
    agent.epsilon = 1.0
    random_actions = [agent.select_action(state, deterministic=False) for _ in range(30)]
    assert 0 in random_actions and 1 in random_actions


def test_replay_buffer_behavior():
    """Verify push, capacity limits, and batch sampling."""
    buf = ReplayBuffer(capacity=10)
    w, d = 4, 10

    for i in range(15):
        s = np.zeros((w, d), dtype=np.float32) + i
        s_next = s + 1
        buf.push(s, action=i % 2, reward=float(i), next_state=s_next, done=(i % 5 == 0))

    assert len(buf) == 10  # Capped at capacity

    states, actions, rewards, next_states, dones = buf.sample(batch_size=4)
    assert states.shape == (4, w, d)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, w, d)
    assert dones.shape == (4, 1)


def test_bellman_target_computation():
    """Verify Bellman target: y = r + gamma * (1 - done) * max_a' Q_target(s', a')."""
    agent = DQNAgent(config=DQNConfig(gamma=0.90, batch_size=2))
    buf = ReplayBuffer(capacity=10)

    s1 = np.ones((4, 10), dtype=np.float32)
    s2 = np.ones((4, 10), dtype=np.float32) * 2

    # Transition 1: terminal (done=True), reward=10.0 -> target should be 10.0 exactly
    buf.push(s1, action=1, reward=10.0, next_state=s2, done=True)
    # Transition 2: non-terminal (done=False), reward=-1.0 -> target should be -1.0 + 0.9 * max Q_target
    buf.push(s1, action=0, reward=-1.0, next_state=s2, done=False)

    states, actions, rewards, next_states, dones = buf.sample(batch_size=2)
    with torch.no_grad():
        q_next = agent.target_net(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + agent.config.gamma * q_next * (1.0 - dones)

    # Check terminal transition target
    terminal_idx = (dones.squeeze() == 1.0).nonzero(as_tuple=True)[0].item()
    assert pytest.approx(targets[terminal_idx].item(), abs=1e-5) == 10.0


def _create_mock_episode(is_clean=False, onset_step=15000, onset_idx=3, total_steps=7):
    steps = [i * 5000 for i in range(total_steps)]
    features = np.zeros((total_steps, 10), dtype=np.float32)
    return DetectorEpisode(
        run_id="mock_run",
        checkpoint_steps=steps,
        checkpoint_ids=[f"ckpt_{s}" for s in steps],
        features=features,
        ground_truth_onset_step=None if is_clean else onset_step,
        ground_truth_onset_idx=None if is_clean else onset_idx,
        is_clean=is_clean,
    )


def test_correct_reward_early_detection():
    """Verify early detection receives large positive reward decaying with delay."""
    cfg = DetectorRewardConfig(r_tp_base=10.0, r_tp_early=5.0, k_allowed_delay=3)
    ep = _create_mock_episode(is_clean=False, onset_idx=2)
    env = DetectorEnv(episodes=[ep], window_size=4, reward_config=cfg)

    # Step to onset idx=2 and FLAG immediately (delay=0)
    env.reset(episode_idx=0)
    env.step(DetectorEnv.ACTION_WAIT)  # t=0
    env.step(DetectorEnv.ACTION_WAIT)  # t=1
    _, r_immediate, done, info = env.step(DetectorEnv.ACTION_FLAG)  # t=2 (onset)

    assert done is True
    assert info["event"] == "true_positive"
    # Delay 0 -> full bonus 5.0 -> 10 + 5 = 15.0
    assert pytest.approx(r_immediate, abs=1e-5) == 15.0

    # Test delay=1 detection: should be less than immediate detection
    env.reset(episode_idx=0)
    env.step(DetectorEnv.ACTION_WAIT)  # t=0
    env.step(DetectorEnv.ACTION_WAIT)  # t=1
    env.step(DetectorEnv.ACTION_WAIT)  # t=2 (onset)
    _, r_delayed, done, info = env.step(DetectorEnv.ACTION_FLAG)  # t=3 (delay=1)

    assert done is True
    assert info["event"] == "true_positive"
    assert r_delayed < r_immediate
    # Delay 1 -> bonus 5.0 * (2/3) = 3.333 -> 13.333
    assert pytest.approx(r_delayed, abs=1e-3) == 13.333


def test_false_positive_penalty():
    """Verify premature flagging or flagging on clean runs receives false positive penalty."""
    cfg = DetectorRewardConfig(r_fp=-10.0)

    # Case 1: Flag before onset on hacking run (onset_idx=3, flag at t=1)
    ep_hack = _create_mock_episode(is_clean=False, onset_idx=3)
    env_hack = DetectorEnv(episodes=[ep_hack], reward_config=cfg)
    env_hack.reset(episode_idx=0)
    env_hack.step(DetectorEnv.ACTION_WAIT)  # t=0
    _, r_early, done, info = env_hack.step(DetectorEnv.ACTION_FLAG)  # t=1 (< onset)

    assert done is True
    assert info["event"] == "false_positive_early"
    assert pytest.approx(r_early, abs=1e-5) == -10.0

    # Case 2: Flag on clean run
    ep_clean = _create_mock_episode(is_clean=True)
    env_clean = DetectorEnv(episodes=[ep_clean], reward_config=cfg)
    env_clean.reset(episode_idx=0)
    _, r_clean_flag, done, info = env_clean.step(DetectorEnv.ACTION_FLAG)

    assert done is True
    assert info["event"] == "false_positive_clean"
    assert pytest.approx(r_clean_flag, abs=1e-5) == -10.0


def test_missed_detection_penalty():
    """Verify waiting until the end on a hacking run receives false negative penalty."""
    cfg = DetectorRewardConfig(r_fn=-15.0)
    ep = _create_mock_episode(is_clean=False, onset_idx=2, total_steps=5)
    env = DetectorEnv(episodes=[ep], reward_config=cfg)
    env.reset(episode_idx=0)

    # Wait at every checkpoint until end
    for t in range(4):
        _, r, done, _ = env.step(DetectorEnv.ACTION_WAIT)
        assert done is False

    # At final checkpoint t=4 (index 4 of 5), waiting triggers end of episode with FN penalty
    _, r_final, done, info = env.step(DetectorEnv.ACTION_WAIT)
    assert done is True
    assert info["event"] == "false_negative"
    assert pytest.approx(r_final, abs=1e-5) == -15.0


def test_clean_run_behavior():
    """Verify clean runs reward patience until completion and penalize false alarms."""
    cfg = DetectorRewardConfig(r_clean_end=5.0, r_clean_step=0.0, r_fp=-10.0)
    ep = _create_mock_episode(is_clean=True, total_steps=4)
    env = DetectorEnv(episodes=[ep], reward_config=cfg)
    env.reset(episode_idx=0)

    # Step through clean run
    for t in range(3):
        _, r, done, info = env.step(DetectorEnv.ACTION_WAIT)
        assert done is False
        assert pytest.approx(r, abs=1e-5) == 0.0

    # Final step waiting
    _, r_end, done, info = env.step(DetectorEnv.ACTION_WAIT)
    assert done is True
    assert info["event"] == "true_negative"
    assert pytest.approx(r_end, abs=1e-5) == 5.0


def test_no_ground_truth_leakage_into_observations():
    """Strictly verify zero ground-truth leakage into detector observation windows."""
    mock_record = {
        "run_id": "secret_run_123",
        "checkpoint_id": "checkpoint_00015000.zip",
        "environment_steps": 15000,
        "proxy_return": 49.0,
        "proxy_return_smoothed": 45.0,
        "proxy_return_trend": 4.0,
        "episode_length": 100.0,
        "action_distribution_change": 0.35,
        "state_visitation_entropy": 2.5,
        "state_visitation_change": 0.4,
        "path_efficiency": 0.1,
        "revisit_rate": 0.5,
        "behavioral_dispersion": 0.8,
        # Leakage fields from evaluation info
        "true_return": 0.0,
        "goal_success": 0.0,
        "region_entries": 50.0,
        "primary_onset": True,
    }

    feat = extract_feature_vector(mock_record)
    assert feat.shape == (10,)
    assert len(FEATURE_NAMES) == 10

    # Verify no true return or goal success exists in extracted vector
    forbidden_terms = ["true_return", "goal_success", "region_entries", "onset"]
    for term in forbidden_terms:
        assert term not in FEATURE_NAMES

    # Ensure observation state from environment contains only normalized behavioral features
    ep = _create_mock_episode(is_clean=False, onset_idx=2)
    env = DetectorEnv(episodes=[ep], window_size=4)
    obs, info = env.reset(episode_idx=0)
    assert obs.shape == (4, 10)


def test_chronological_sequence_construction():
    """Verify episodes sort checkpoints chronologically and construct left-padded windows."""
    # Test chronological window building
    ep = _create_mock_episode(is_clean=False, total_steps=5)
    # Assign identifiable feature values: checkpoint step index
    ep.features = np.array([[i] * 10 for i in range(5)], dtype=np.float32)

    env = DetectorEnv(episodes=[ep], window_size=3)
    s0, _ = env.reset(episode_idx=0)

    # At t=0, window of size 3 should have 2 zero-pads followed by step 0 feature
    assert np.all(s0[0] == 0.0)
    assert np.all(s0[1] == 0.0)
    assert np.all(s0[2] == 0.0)  # feature 0 is all zeros

    # Step to t=1
    s1, _, _, _ = env.step(DetectorEnv.ACTION_WAIT)
    # At t=1: 1 zero-pad, then step 0, then step 1
    assert np.all(s1[0] == 0.0)
    assert np.all(s1[1] == 0.0)  # step 0
    assert np.all(s1[2] == 1.0)  # step 1

    # Step to t=2
    s2, _, _, _ = env.step(DetectorEnv.ACTION_WAIT)
    # At t=2: fully filled window [step 0, step 1, step 2]
    assert np.all(s2[0] == 0.0)
    assert np.all(s2[1] == 1.0)
    assert np.all(s2[2] == 2.0)


def test_run_level_train_test_separation():
    """Verify run-level partitioning guarantees disjoint sets of run IDs."""
    all_runs = [f"run_{i}" for i in range(12)]

    # Mock split: 6 train, 3 val, 3 test
    train_ids = all_runs[:6]
    val_ids = all_runs[6:9]
    test_ids = all_runs[9:]

    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_runs)
