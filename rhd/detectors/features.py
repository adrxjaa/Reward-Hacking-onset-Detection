"""
features.py - Causal Behavioral Feature Extraction for Reward Hacking Detection

Provides improved causal temporal features computed online across checkpoints
without any data leakage.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from rhd.detectors.environment import FEATURE_NAMES, extract_feature_vector


IMPROVED_FEATURE_NAMES = list(FEATURE_NAMES) + [
    "proxy_efficiency_divergence",
    "entropy_trend",
    "path_efficiency_trend",
    "rolling_revisit_mean",
    "rolling_entropy_std",
    "consecutive_anomalous_count",
]


def extract_improved_features_from_records(
    records: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Extract causal 16-dimensional feature matrix for an entire run from its chronological
    records in temporal_features.jsonl.
    
    Causality guarantee:
    At checkpoint index t, features are computed EXCLUSIVELY from records 0 to t.
    Zero access to t+1 or any future checkpoint.
    Zero access to true_return, goal_success, region_entries, or onset labels.
    """
    T = len(records)
    if T == 0:
        return np.zeros((0, len(IMPROVED_FEATURE_NAMES)), dtype=np.float32)

    # First extract base 10 features for all steps
    base_feats = np.array([extract_feature_vector(r) for r in records], dtype=np.float32)

    enhanced = []
    anomalous_streak = 0

    for t in range(T):
        base = base_feats[t]
        proxy = base[0]
        proxy_smoothed = base[1]
        ep_len = base[3]
        entropy = base[5]
        path_eff = base[7]
        revisit = base[8]

        # 11. proxy_efficiency_divergence: proxy_smoothed * revisit
        divergence = float(proxy_smoothed * revisit)

        # 12. entropy_trend: backward difference (causal)
        ent_trend = float(entropy - base_feats[t - 1, 5]) if t > 0 else 0.0

        # 13. path_efficiency_trend: backward difference (causal)
        eff_trend = float(path_eff - base_feats[t - 1, 7]) if t > 0 else 0.0

        # 14. rolling_revisit_mean over last min(t+1, 3) checkpoints
        win_start = max(0, t - 2)
        past_revisits = base_feats[win_start : t + 1, 8]
        past_entropies = base_feats[win_start : t + 1, 5]
        rolling_revisit = float(np.mean(past_revisits))

        # 15. rolling_entropy_std
        rolling_ent_std = float(np.std(past_entropies)) if len(past_entropies) > 1 else 0.0

        # 16. consecutive_anomalous_count (causal accumulator)
        # Behavioral indicator: high revisit rate and reduced path efficiency
        if revisit > 0.25 and path_eff < 0.50:
            anomalous_streak += 1
        else:
            anomalous_streak = 0
        consec_anomaly = float(anomalous_streak)

        row = np.concatenate([
            base,
            np.array([
                divergence,
                ent_trend,
                eff_trend,
                rolling_revisit,
                rolling_ent_std,
                consec_anomaly,
            ], dtype=np.float32),
        ])
        enhanced.append(row)

    return np.array(enhanced, dtype=np.float32)

