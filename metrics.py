"""
metrics.py ? Common metrics interface.

Every environment's evaluate.py should produce a dict that matches the
COMMON_SCHEMA below. Environment-specific extras (e.g. coin_visits) are
stored in a separate column 'extra_metrics' as a JSON string so they
don't break the shared CSV schema.

To add a second environment later:
    1. Create environment2.py
    2. Create evaluate2.py (or reuse evaluate.py with --env flag)
    3. Its output CSV will share the same columns -> easy to concatenate.
"""

import json
import math
import numpy as np
from typing import Any, Dict, List, Optional

#  Common CSV schema
COMMON_SCHEMA = [
    "environment",
    "training_step",
    "episode",
    "proxy_reward",
    "episode_length",
    "goal_reached",
    "entropy",
    "state_visitation",       # JSON-encoded dict {state_key: visit_count}
    "action_distribution",    # JSON-encoded list [p(a0), p(a1), ...]
    "extra_metrics",          # JSON-encoded dict for env-specific extras
]


def compute_entropy(action_probs: np.ndarray) -> float:
    """
    Shannon entropy: H = -sum(p * log(p))
    Clips probabilities to avoid log(0).
    """
    probs = np.clip(action_probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_avg_entropy(model, states: List[np.ndarray]) -> float:
    """
    Average policy entropy over a sample of states.

    Works with any SB3 policy that exposes get_distribution().
    Falls back to a simpler approach if the policy is unusual.
    """
    import torch
    entropies = []
    obs_tensor = torch.as_tensor(
        np.array(states, dtype=np.float32)
    ).to(model.device)

    with torch.no_grad():
        try:
            dist = model.policy.get_distribution(obs_tensor)
            log_probs = dist.distribution.logits  # Categorical logits
            probs = torch.softmax(log_probs, dim=-1).cpu().numpy()
        except Exception:
            # Fallback: evaluate action values directly
            probs_list = []
            for obs in states:
                t = torch.as_tensor(obs[None], dtype=torch.float32).to(model.device)
                d = model.policy.get_distribution(t)
                p = torch.softmax(d.distribution.logits, dim=-1).cpu().numpy()[0]
                probs_list.append(p)
            probs = np.array(probs_list)

    for p in probs:
        entropies.append(compute_entropy(p))
    return float(np.mean(entropies)) if entropies else 0.0


def state_to_key(obs: np.ndarray) -> str:
    """Convert an observation array to a hashable string key for visitation counting."""
    return str(tuple(obs.astype(int)))


def build_metric_row(
    *,
    environment: str,
    training_step: int,
    episode: int,
    proxy_reward: float,
    episode_length: int,
    goal_reached: bool,
    entropy: float,
    state_visitation: Dict[str, int],
    action_distribution: List[float],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a single metric row conforming to COMMON_SCHEMA.
    extra_metrics can hold environment-specific data (e.g. coin_visits).
    """
    return {
        "environment":        environment,
        "training_step":      training_step,
        "episode":            episode,
        "proxy_reward":       round(proxy_reward, 4),
        "episode_length":     episode_length,
        "goal_reached":       int(goal_reached),
        "entropy":            round(entropy, 6),
        "state_visitation":   json.dumps(state_visitation),
        "action_distribution": json.dumps([round(p, 4) for p in action_distribution]),
        "extra_metrics":      json.dumps(extra_metrics or {}),
    }

