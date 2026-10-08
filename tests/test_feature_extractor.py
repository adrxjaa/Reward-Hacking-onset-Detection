import os
import json
import pytest
import numpy as np

from rhd.envs.region_toggle import RegionToggleConfig
from rhd.features.extractor import compute_tvd, generate_probe_states, extract_features_for_run

def test_compute_tvd():
    p = np.array([0.1, 0.9])
    q = np.array([0.1, 0.9])
    assert compute_tvd(p, q) == 0.0

    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert compute_tvd(p, q) == 1.0

def test_generate_probe_states():
    cfg = RegionToggleConfig(size=6)
    probe_config = generate_probe_states(cfg)
    assert len(probe_config.states) == 36
    assert (0, 0) in probe_config.states
    assert (5, 5) in probe_config.states
