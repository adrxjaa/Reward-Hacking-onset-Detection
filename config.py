"""
config.py - Shared configuration for the reward-hacking detection project.

All environment-specific settings live under their own key so that adding
a second environment later requires no changes to the generic pipeline.
"""

import os

# Reproducibility
SEED = 42

# Training
TOTAL_TIMESTEPS = 50_000

# Checkpoint schedule (in timesteps).
# The same schedule is used for every environment.
CHECKPOINT_STEPS = [0, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000]

# PPO hyper-parameters (shared defaults)
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=0,
)

# Evaluation
EVAL_EPISODES = 20           # episodes to run per checkpoint during evaluation
ENTROPY_SAMPLE_STATES = 50  # random states to sample when computing avg entropy

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")


def checkpoint_path(env_name: str, step: int) -> str:
    """Return the path for a checkpoint zip, e.g. checkpoints/gridworld/10000.zip"""
    folder = os.path.join(CHECKPOINT_DIR, env_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{step}.zip")


def results_csv_path(env_name: str) -> str:
    """Return the path for the results CSV, e.g. results/gridworld_metrics.csv"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{env_name}_metrics.csv")


def results_plots_dir(env_name: str) -> str:
    """Return the directory for plots, e.g. results/gridworld_plots/"""
    path = os.path.join(RESULTS_DIR, f"{env_name}_plots")
    os.makedirs(path, exist_ok=True)
    return path
