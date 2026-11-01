"""
rhd.detectors - Learned Sequential Detectors for Reward Hacking Onset Detection
"""

from rhd.detectors.models import (
    LSTMQNetwork,
    MLPQNetwork,
    ImprovedLSTMQNetwork,
    ImprovedMLPQNetwork,
)
from rhd.detectors.features import (
    IMPROVED_FEATURE_NAMES,
    extract_improved_features_from_records,
)
from rhd.detectors.replay_buffer import ReplayBuffer
from rhd.detectors.environment import (
    DetectorRewardConfig,
    DetectorEpisode,
    DetectorEnv,
    FEATURE_NAMES,
    extract_feature_vector,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.trainer import DetectorTrainer, TrainerConfig
from rhd.detectors.evaluator import DetectorEvaluator, EvaluationResult

__all__ = [
    "LSTMQNetwork",
    "MLPQNetwork",
    "ImprovedLSTMQNetwork",
    "ImprovedMLPQNetwork",
    "IMPROVED_FEATURE_NAMES",
    "extract_improved_features_from_records",
    "ReplayBuffer",
    "DetectorRewardConfig",
    "DetectorEpisode",
    "DetectorEnv",
    "FEATURE_NAMES",
    "extract_feature_vector",
    "DQNAgent",
    "DQNConfig",
    "DetectorTrainer",
    "TrainerConfig",
    "DetectorEvaluator",
    "EvaluationResult",
]
