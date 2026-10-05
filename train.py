"""
train.py - Generic PPO training pipeline.

Usage:
    python train.py

The script is environment-agnostic: it imports the environment factory and
ENV_NAME from environment.py. To train a second environment later, simply
point this script at the new environment module (or pass --env as an argument).

What this script does:
    1. Creates the Gymnasium environment.
    2. Initialises a PPO model with the shared hyper-parameters from config.py.
    3. Trains for TOTAL_TIMESTEPS, saving a checkpoint at each step listed in
       CHECKPOINT_STEPS.
    4. The "step 0" checkpoint captures the untrained (random) policy so we
       can measure how behaviour changes over time.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config
from environment import GridWorldEnv, ENV_NAME


def train():
    print(f"=== Training PPO on '{ENV_NAME}' ===")
    print(f"    Total timesteps : {config.TOTAL_TIMESTEPS:,}")
    print(f"    Checkpoints at  : {config.CHECKPOINT_STEPS}")
    print(f"    Seed            : {config.SEED}")
    print()

    # Create environment
    env = make_vec_env(
        lambda: GridWorldEnv(),
        n_envs=1,
        seed=config.SEED,
    )

    # Initialise PPO
    model = PPO(
        "MlpPolicy",
        env,
        seed=config.SEED,
        **config.PPO_KWARGS,
    )

    # Save step-0 checkpoint (random policy)
    path0 = config.checkpoint_path(ENV_NAME, 0)
    model.save(path0)
    print(f"  [ckpt]  0 steps -> {path0}")

    # Train with periodic checkpoints
    prev_step = 0
    for target_step in config.CHECKPOINT_STEPS[1:]:   # skip 0 (already saved)
        steps_to_train = target_step - prev_step
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        path = config.checkpoint_path(ENV_NAME, target_step)
        model.save(path)
        print(f"  [ckpt]  {target_step:,} steps -> {path}")
        prev_step = target_step

    env.close()
    print("\n[OK] Training complete.")
    print(f"  Checkpoints saved to: {os.path.join(config.CHECKPOINT_DIR, ENV_NAME)}/")


if __name__ == "__main__":
    train()
