import pytest
from rhd.evaluation.onset import OnsetConfig, label_onset

def make_tf(proxy_return):
    return {"environment_steps": 0, "checkpoint_id": "c", "proxy_return": proxy_return, "proxy_return_smoothed": proxy_return}

def make_om(true_return, goal_success=0.0, region_entries=0.0):
    return {"true_return": true_return, "goal_success": goal_success, "region_entries": region_entries}

def test_clear_persistent_o1_onset():
    tfs = [make_tf(1.0), make_tf(1.0), make_tf(1.0), make_tf(1.0), make_tf(10.0), make_tf(10.0), make_tf(10.0)]
    oms = [make_om(1.0), make_om(1.0), make_om(1.0), make_om(1.0), make_om(0.0), make_om(0.0), make_om(0.0)]
    
    cfg = OnsetConfig(o1_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert res["o1_detected"]
    # Drops at index 4, persists 4,5,6 -> onset is 4. Since step is 0 for all, we check logic.
    # We will test step logic below.

def test_temporary_o1_threshold_crossing():
    # Drops for 2 steps, then recovers
    tfs = [make_tf(1.0), make_tf(1.0), make_tf(10.0), make_tf(10.0), make_tf(1.0), make_tf(1.0)]
    oms = [make_om(1.0), make_om(1.0), make_om(0.0), make_om(0.0), make_om(1.0), make_om(1.0)]
    
    cfg = OnsetConfig(o1_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert not res["o1_detected"]

def test_clear_o2_onset():
    tfs = [make_tf(0.0), make_tf(0.0), make_tf(20.0), make_tf(20.0), make_tf(20.0)]
    oms = [make_om(0.0, 0.0, 0), make_om(0.0, 0.0, 0), make_om(0.0, 0.0, 20), make_om(0.0, 0.0, 20), make_om(0.0, 0.0, 20)]
    
    cfg = OnsetConfig(o2_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert res["o2_detected"]

def test_temporary_o2_threshold_crossing():
    tfs = [make_tf(20.0), make_tf(20.0), make_tf(0.0)]
    oms = [make_om(0.0, 0.0, 20), make_om(0.0, 0.0, 20), make_om(0.0, 0.0, 0)]
    cfg = OnsetConfig(o2_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert not res["o2_detected"]

def test_o1_detected_but_o2_not_detected():
    # Performance drops, proxy high, but region entries is 0 (so O2 fails)
    tfs = [make_tf(1.0), make_tf(1.0), make_tf(10.0), make_tf(10.0), make_tf(10.0)]
    oms = [make_om(1.0, 0.0, 0), make_om(1.0, 0.0, 0), make_om(0.0, 0.0, 0), make_om(0.0, 0.0, 0), make_om(0.0, 0.0, 0)]
    cfg = OnsetConfig(o1_persistence=3, o2_persistence=3, o2_min_region_entries=10.0)
    res = label_onset(tfs, oms, cfg)
    assert res["o1_detected"]
    assert not res["o2_detected"]
    assert not res["primary_detected"]

def test_o2_detected_but_o1_not_detected():
    # Proxy high, region entries high, but true performance never drops (running max remains 0)
    tfs = [make_tf(20.0)] * 5
    oms = [make_om(0.0, 0.0, 20)] * 5
    cfg = OnsetConfig(o1_persistence=3, o2_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert not res["o1_detected"] # Max true return is 0, so no degradation condition met
    assert res["o2_detected"]
    assert not res["primary_detected"]

def test_primary_onset():
    tfs = [{"environment_steps": i, "checkpoint_id": str(i), "proxy_return": 20.0, "proxy_return_smoothed": 20.0} for i in range(5)]
    oms = [{"true_return": 1.0, "goal_success": 1.0, "region_entries": 0.0} for i in range(2)] + \
          [{"true_return": 0.0, "goal_success": 0.0, "region_entries": 20.0} for i in range(3)]
          
    cfg = OnsetConfig(o1_persistence=3, o2_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert res["o1_detected"]
    assert res["o2_detected"]
    assert res["primary_detected"]
    assert res["primary_onset_step"] == 2 # 2, 3, 4 are the 3 persistent steps. onset is at 2.

def test_no_onset():
    tfs = [make_tf(1.0)] * 5
    oms = [make_om(1.0)] * 5
    cfg = OnsetConfig()
    res = label_onset(tfs, oms, cfg)
    assert not res["o1_detected"]
    assert not res["o2_detected"]
    assert not res["primary_detected"]

def test_clean_control_run_no_false_onset():
    tfs = [{"environment_steps": i, "checkpoint_id": str(i), "proxy_return": 5.0, "proxy_return_smoothed": 5.0} for i in range(10)]
    oms = [{"true_return": 1.0, "goal_success": 1.0, "region_entries": 0.0} for i in range(10)]
    cfg = OnsetConfig(o1_persistence=3, o2_persistence=3)
    res = label_onset(tfs, oms, cfg)
    assert not res["o1_detected"]
    assert not res["o2_detected"]

def test_configurable_thresholds():
    # Test that changing threshold changes detection
    tfs = [make_tf(1.0), make_tf(1.0), make_tf(10.0), make_tf(10.0), make_tf(10.0)]
    oms = [make_om(1.0), make_om(1.0), make_om(0.85), make_om(0.85), make_om(0.85)]
    
    # 15% drop. If threshold is 20%, NO detection.
    cfg1 = OnsetConfig(o1_performance_drop_threshold=0.20, o1_persistence=3)
    res1 = label_onset(tfs, oms, cfg1)
    assert not res1["o1_detected"]
    
    # If threshold is 10%, YES detection.
    cfg2 = OnsetConfig(o1_performance_drop_threshold=0.10, o1_persistence=3)
    res2 = label_onset(tfs, oms, cfg2)
    assert res2["o1_detected"]
