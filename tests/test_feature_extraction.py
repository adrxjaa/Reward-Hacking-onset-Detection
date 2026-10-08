import json
import os
import pytest
import numpy as np
from rhd.features.extract import FeatureExtractor, js_divergence

def test_js_divergence():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert js_divergence(p, q) == 0.0
    
    p2 = np.array([1.0, 0.0])
    assert js_divergence(p, p2) > 0.0

def test_extract_checkpoint_format(tmp_path):
    run_dir = str(tmp_path / "run_w1.0")
    # Actually, we will just run the extraction script on our successful pilot run.
    # The pilot run is at experiments/pilot/run_w1.0
    pilot_dir = "experiments/pilot/run_w1.0"
    if not os.path.exists(pilot_dir):
        pytest.skip("Pilot run not found, skipping extraction test.")
        
    extractor = FeatureExtractor(pilot_dir)
    assert len(extractor.probe_states) == 20
    assert extractor.probe_states.shape[1] == 2
    
    extractor.run_extraction()
    
    feat_dir = os.path.join(pilot_dir, "features")
    assert os.path.exists(os.path.join(feat_dir, "temporal_features.jsonl"))
    assert os.path.exists(os.path.join(feat_dir, "offline_eval.jsonl"))
    assert os.path.exists(os.path.join(feat_dir, "feature_schema.json"))
    assert os.path.exists(os.path.join(feat_dir, "probe_states.json"))
    
    # Check that true reward/return is NOT in features
    with open(os.path.join(feat_dir, "temporal_features.jsonl"), "r") as f:
        features = [json.loads(line) for line in f]
        
    for feat in features:
        assert "true_return" not in feat
        assert "true_reward" not in feat
        assert "goal_success" not in feat
        assert "region_entries" not in feat
        
        # Check required fields
        assert "checkpoint_id" in feat
        assert "environment_steps" in feat
        assert "proxy_return" in feat
        assert "action_distribution_change" in feat
        assert "state_visitation_entropy" in feat
        assert "behavioral_dispersion" in feat
        assert "revisit_rate" in feat
        assert "path_efficiency" in feat

    # Offline eval should have true return
    with open(os.path.join(feat_dir, "offline_eval.jsonl"), "r") as f:
        offlines = [json.loads(line) for line in f]
        
    for off in offlines:
        assert "true_return" in off
        assert "goal_success" in off
