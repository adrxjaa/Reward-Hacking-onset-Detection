import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


FEATURE_NAMES = [
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


def extract_feature_vector(d: Dict[str, Any]) -> np.ndarray:
    """
    Extract canonical 10-dimensional behavioral feature vector from a feature record.
    Zero leakage: no true reward, goal success, or onset information is extracted.
    """
    trend = d.get("proxy_return_trend", d.get("proxy_return_change", 0.0))
    proxy = float(d.get("proxy_return", 0.0))
    smoothed = float(d.get("proxy_return_smoothed", proxy))
    
    return np.array([
        proxy,
        smoothed,
        float(trend),
        float(d.get("episode_length", 0.0)),
        float(d.get("action_distribution_change", 0.0)),
        float(d.get("state_visitation_entropy", 0.0)),
        float(d.get("state_visitation_change", 0.0)),
        float(d.get("path_efficiency", 0.0)),
        float(d.get("revisit_rate", 0.0)),
        float(d.get("behavioral_dispersion", 0.0)),
    ], dtype=np.float32)


@dataclass
class DetectorRewardConfig:
    """
    Configurable reward parameters for the LSTM-DQN Onset Detector.
    
    Rationale:
    - r_tp_base + r_tp_early: Strongly incentivizes catching reward hacking as early as possible.
    - r_fp: Penalizes premature alarms (before true onset or on clean control runs),
      preventing trigger-happy detectors that produce high false-alarm rates in deployment.
    - r_fn: Heavily penalizes failing to flag an ongoing hacking run before completion.
    - r_delay: Penalizes procrastination at every checkpoint where hacking is ongoing.
    - r_clean_end: Rewards successfully monitoring a clean run without false alarms.
    - k_allowed_delay: Window of tolerance (in checkpoints) for timely onset detection.
    """
    r_tp_base: float = 10.0
    r_tp_early: float = 5.0
    r_fp: float = -10.0
    r_fn: float = -15.0
    r_delay: float = -1.0
    r_late_tp: float = 2.0
    r_clean_step: float = 0.0
    r_clean_end: float = 5.0
    k_allowed_delay: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Normalizer:
    """
    Feature normalizer fitted strictly on training runs to prevent data leakage.
    """

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, feature_arrays: List[np.ndarray]):
        all_features = np.concatenate(feature_arrays, axis=0)
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "feature_names": FEATURE_NAMES,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Normalizer":
        mean = np.array(data["mean"], dtype=np.float32) if data.get("mean") else None
        std = np.array(data["std"], dtype=np.float32) if data.get("std") else None
        return cls(mean, std)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Normalizer":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class DetectorEpisode:
    """
    Encapsulates one full generator training run as a single episode for the detector.
    """
    run_id: str
    checkpoint_steps: List[int]
    checkpoint_ids: List[str]
    features: np.ndarray              # Shape: (T, num_features)
    ground_truth_onset_step: Optional[int]
    ground_truth_onset_idx: Optional[int]
    is_clean: bool

    @property
    def total_checkpoints(self) -> int:
        return len(self.checkpoint_steps)


def load_episode_from_run(
    run_dir: str,
    use_independent_ground_truth: bool = True,
) -> Optional[DetectorEpisode]:
    """
    Load temporal features and onset metadata from a completed generator run directory.
    If use_independent_ground_truth is True, ground truth onset is determined strictly
    from task-level rollout metrics (exploit loop + goal abandonment), completely
    independent of the comparison baselines O1 and O2.
    """
    feat_dir = os.path.join(run_dir, "features")
    temp_path = os.path.join(feat_dir, "temporal_features.jsonl")
    off_path = os.path.join(feat_dir, "offline_eval.jsonl")
    onset_path = os.path.join(feat_dir, "onset.json")

    if not os.path.exists(temp_path):
        return None

    # Load temporal features chronologically
    records = []
    with open(temp_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) < 2:
        # Insufficient checkpoints to form meaningful temporal sequence
        return None

    # Sort strictly chronologically by environment_steps
    records = sorted(records, key=lambda x: int(x["environment_steps"]))

    steps = [int(r["environment_steps"]) for r in records]
    ckpt_ids = [str(r.get("checkpoint_id", f"step_{s}")) for r, s in zip(records, steps)]

    raw_feats = np.array([extract_feature_vector(r) for r in records], dtype=np.float32)

    # Determine ground-truth onset
    onset_step = None
    if use_independent_ground_truth:
        if os.path.exists(off_path):
            from rhd.detectors.ground_truth import get_run_ground_truth_onset
            onset_step = get_run_ground_truth_onset(run_dir)
        else:
            raise FileNotFoundError(
                f"Independent ground truth requested for run '{run_dir}', but offline evaluation log '{off_path}' was not found. "
                "Refusing silent fallback to O1/O2 onset metadata to prevent circular ground-truth leakage."
            )
    elif os.path.exists(onset_path):
        # Fallback to onset.json only when use_independent_ground_truth is explicitly False
        with open(onset_path, "r") as f:
            onset_data = json.load(f)

        if onset_data.get("primary_detected") and onset_data.get("primary_onset_step") is not None:
            onset_step = int(onset_data["primary_onset_step"])
        elif onset_data.get("o2_detected") and onset_data.get("o2_onset_step") is not None:
            onset_step = int(onset_data["o2_onset_step"])
        elif onset_data.get("o1_detected") and onset_data.get("o1_onset_step") is not None:
            onset_step = int(onset_data["o1_onset_step"])

    # Locate index of onset step within steps list
    onset_idx = None
    if onset_step is not None:
        for idx, s in enumerate(steps):
            if s >= onset_step:
                onset_idx = idx
                break

    is_clean = (onset_step is None)
    run_id = os.path.basename(os.path.normpath(run_dir))

    return DetectorEpisode(
        run_id=run_id,
        checkpoint_steps=steps,
        checkpoint_ids=ckpt_ids,
        features=raw_feats,
        ground_truth_onset_step=onset_step,
        ground_truth_onset_idx=onset_idx,
        is_clean=is_clean,
    )


class DetectorEnv:
    """
    Sequential Decision MDP for the Reward Hacking Onset Detector.
    
    Actions:
        0: WAIT - continue monitoring policy checkpoints
        1: FLAG - trigger reward hacking alarm and terminate episode
    """
    ACTION_WAIT = 0
    ACTION_FLAG = 1

    def __init__(
        self,
        episodes: List[DetectorEpisode],
        window_size: int = 4,
        reward_config: Optional[DetectorRewardConfig] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        self.episodes = episodes
        self.window_size = window_size
        self.reward_config = reward_config or DetectorRewardConfig()
        self.normalizer = normalizer

        self.current_episode: Optional[DetectorEpisode] = None
        self.current_step: int = 0
        self.normalized_features: Optional[np.ndarray] = None

    def reset(self, episode_idx: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset detector environment to the start of an episode.
        """
        if episode_idx is not None:
            self.current_episode = self.episodes[episode_idx]
        else:
            self.current_episode = np.random.choice(self.episodes)

        self.current_step = 0

        # Apply normalizer
        raw_feats = self.current_episode.features
        if self.normalizer is not None:
            self.normalized_features = self.normalizer.transform(raw_feats)
        else:
            self.normalized_features = raw_feats

        state = self._get_window(self.current_step)
        info = {
            "run_id": self.current_episode.run_id,
            "step_idx": self.current_step,
            "env_step": self.current_episode.checkpoint_steps[self.current_step],
            "is_clean": self.current_episode.is_clean,
            "ground_truth_onset_idx": self.current_episode.ground_truth_onset_idx,
        }
        return state, info

    def _get_window(self, step_idx: int) -> np.ndarray:
        """
        Construct rolling window of size W ending at step_idx.
        Left-pads with zeros if sequence length before step_idx < W.
        """
        T, D = self.normalized_features.shape
        W = self.window_size

        window = np.zeros((W, D), dtype=np.float32)
        start_idx = max(0, step_idx - W + 1)
        sub_seq = self.normalized_features[start_idx : step_idx + 1]

        # Place sub_seq at the end of the window (left-padded)
        window[W - len(sub_seq) :] = sub_seq
        return window

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step of the detector MDP.
        """
        ep = self.current_episode
        t = self.current_step
        T = ep.total_checkpoints
        cfg = self.reward_config
        t_star = ep.ground_truth_onset_idx

        done = False
        reward = 0.0
        info = {
            "run_id": ep.run_id,
            "step_idx": t,
            "env_step": ep.checkpoint_steps[t],
            "action": action,
            "is_clean": ep.is_clean,
            "onset_idx": t_star,
        }

        if action == self.ACTION_FLAG:
            done = True
            if ep.is_clean:
                # False positive on clean control
                reward = cfg.r_fp
                info["event"] = "false_positive_clean"
            elif t_star is not None and t < t_star:
                # False positive before true onset
                reward = cfg.r_fp
                info["event"] = "false_positive_early"
            elif t_star is not None and t >= t_star:
                # True positive
                delay = t - t_star
                if delay <= cfg.k_allowed_delay:
                    # Early detection bonus decays linearly with delay
                    bonus = cfg.r_tp_early * ((cfg.k_allowed_delay - delay) / max(cfg.k_allowed_delay, 1))
                    reward = cfg.r_tp_base + bonus
                    info["event"] = "true_positive"
                else:
                    # Late detection
                    reward = cfg.r_late_tp
                    info["event"] = "late_true_positive"
                info["detection_delay_ckpts"] = delay
                info["detection_delay_steps"] = (
                    ep.checkpoint_steps[t] - ep.checkpoint_steps[t_star]
                )
        else:
            # Action: WAIT
            if ep.is_clean:
                if t >= T - 1:
                    # Reached end of clean run successfully without flagging
                    done = True
                    reward = cfg.r_clean_end
                    info["event"] = "true_negative"
                else:
                    reward = cfg.r_clean_step
                    info["event"] = "wait_clean"
            else:
                # Hacking run
                if t_star is not None and t < t_star:
                    # Waiting prior to onset is appropriate
                    reward = 0.0
                    info["event"] = "wait_pre_onset"
                else:
                    # Waiting after onset is penalized
                    if t >= T - 1:
                        # Reached the end of a hacking run without ever flagging
                        done = True
                        reward = cfg.r_fn
                        info["event"] = "false_negative"
                    else:
                        reward = cfg.r_delay
                        info["event"] = "wait_post_onset"

        # Advance step if not done
        if not done:
            self.current_step += 1
            if self.current_step >= T:
                done = True
                next_state = self._get_window(T - 1)
            else:
                next_state = self._get_window(self.current_step)
        else:
            next_state = self._get_window(t)

        return next_state, float(reward), done, info
