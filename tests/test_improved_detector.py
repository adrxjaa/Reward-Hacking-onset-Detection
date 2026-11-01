import os
import pytest
import numpy as np
import torch

from rhd.detectors.features import (
    IMPROVED_FEATURE_NAMES,
    extract_improved_features_from_records,
)
from rhd.detectors.models import (
    ImprovedLSTMQNetwork,
    ImprovedMLPQNetwork,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.replay_buffer import ReplayBuffer
from rhd.detectors.environment import load_episode_from_run


def test_improved_feature_schema_and_names():
    assert len(IMPROVED_FEATURE_NAMES) == 16
    assert IMPROVED_FEATURE_NAMES[:10] == [
        "proxy_return",
        "proxy_return_smoothed",
        "proxy_return_trend",
        "episode_length",
        "action_distribution_change",
        "state_visitation_entropy",
        "state_visitation_change",
        "path_efficiency",
        "revisit_rate",
        "behavioral_dispersion",
    ]
    prohibited = ["true", "goal", "entry", "entries", "onset", "label", "ground", "target", "weight"]
    for name in IMPROVED_FEATURE_NAMES:
        for p in prohibited:
            assert p not in name.lower() or name == "revisit_rate"


def test_improved_features_causality():
    records = []
    for t in range(5):
        records.append({
            "environment_steps": t * 5000,
            "checkpoint_id": f"ckpt_{t}",
            "proxy_return": float(t * 10),
            "proxy_return_smoothed": float(t * 10),
            "proxy_return_trend": 10.0,
            "episode_length": 50.0,
            "action_distribution_change": 0.1,
            "state_visitation_entropy": 2.5 - t * 0.3,
            "state_visitation_change": 0.2,
            "path_efficiency": max(0.1, 1.0 - t * 0.2),
            "revisit_rate": min(0.9, t * 0.2),
            "behavioral_dispersion": 1.5,
        })

    feats_all = extract_improved_features_from_records(records)
    assert feats_all.shape == (5, 16)

    # Check causality: modifying record 4 should NOT change features at steps 0..3
    records_mod = [dict(r) for r in records]
    records_mod[4]["proxy_return"] = 99999.0
    records_mod[4]["revisit_rate"] = 0.99
    feats_mod = extract_improved_features_from_records(records_mod)

    np.testing.assert_allclose(feats_all[:4], feats_mod[:4], rtol=1e-5, atol=1e-5)


def test_improved_lstm_forward_and_shapes():
    B, W, D = 4, 4, 16
    model = ImprovedLSTMQNetwork(input_dim=D, hidden_dim=32, num_layers=1, num_actions=2)

    x = torch.randn(B, W, D)
    q = model(x)
    assert q.shape == (B, 2)

    # Single unbatched window
    x_single = torch.randn(W, D)
    q_single = model(x_single)
    assert q_single.shape == (1, 2)

    # Gradient flow test
    loss = q.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_improved_mlp_forward_and_shapes():
    B, W, D = 4, 4, 16
    model = ImprovedMLPQNetwork(input_dim=D, window_size=W, hidden_dim=32, num_actions=2)

    x = torch.randn(B, W, D)
    q = model(x)
    assert q.shape == (B, 2)

    x_single = torch.randn(W, D)
    q_single = model(x_single)
    assert q_single.shape == (1, 2)


def test_decision_threshold_policy():
    cfg = DQNConfig(
        input_dim=16,
        model_type="improved_lstm",
        decision_threshold=2.0,  # Require Q(FLAG) - Q(WAIT) >= 2.0
    )
    agent = DQNAgent(config=cfg)

    state = np.random.randn(4, 16).astype(np.float32)

    # Mock online_net to return known Q-values: [10.0, 11.0] -> diff is 1.0 < 2.0 -> WAIT (0)
    class MockNet(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, 11.0]])

    agent.online_net = MockNet()
    action = agent.select_action(state, deterministic=True)
    assert action == 0

    # Mock with diff >= 2.0: [10.0, 12.5] -> diff is 2.5 >= 2.0 -> FLAG (1)
    class MockNet2(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, 12.5]])

    agent.online_net = MockNet2()
    action2 = agent.select_action(state, deterministic=True)
    assert action2 == 1


def test_balanced_replay_buffer_sampling():
    buf = ReplayBuffer(capacity=100)
    w, d = 4, 16

    # Push 30 wait transitions (action 0)
    for i in range(30):
        s = np.zeros((w, d), dtype=np.float32)
        buf.push(s, action=0, reward=0.0, next_state=s, done=False)

    # Push 5 flag transitions (action 1)
    for i in range(5):
        s = np.ones((w, d), dtype=np.float32)
        buf.push(s, action=1, reward=10.0, next_state=s, done=True)

    states, actions, rewards, next_states, dones = buf.sample_balanced(batch_size=8, flag_ratio=0.25)
    assert states.shape == (8, w, d)
    flag_count = (actions == 1).sum().item()
    assert flag_count >= 1  # Guaranteed at least 1 flag transition sampled


def test_independent_ground_truth_enforcement(tmp_path):
    run_dir = str(tmp_path / "fake_run")
    os.makedirs(os.path.join(run_dir, "features"), exist_ok=True)

    # Write temporal features but NO offline_eval.jsonl
    with open(os.path.join(run_dir, "features", "temporal_features.jsonl"), "w") as f:
        f.write('{"environment_steps": 0, "checkpoint_id": "0", "proxy_return": 0.0}\n')
        f.write('{"environment_steps": 5000, "checkpoint_id": "1", "proxy_return": 1.0}\n')

    # Writing onset.json should NOT be silently used if use_independent_ground_truth=True
    with open(os.path.join(run_dir, "features", "onset.json"), "w") as f:
        f.write('{"o2_detected": true, "o2_onset_step": 5000}\n')

    with pytest.raises(FileNotFoundError, match="Refusing silent fallback"):
        load_episode_from_run(run_dir, use_independent_ground_truth=True)

