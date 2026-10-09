
"""
train3.py - PPO training pipeline for Battery Collector.

Saves checkpoints at the shared schedule:
0, 5000, 10000, 20000, 30000, 40000, 50000.

This script does not modify train.py or the GridWorld experiment.
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config
from environment3 import BatteryCollectorEnv, ENV_NAME


def train():
    print(f"=== Training PPO on '{ENV_NAME}' ===")
    print(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints: {config.CHECKPOINT_STEPS}")
    print(f"Seed: {config.SEED}")
    print()

    env = make_vec_env(
        lambda: BatteryCollectorEnv(),
        n_envs=1,
        seed=config.SEED,
    )

    model = PPO(
        "MlpPolicy",
        env,
        seed=config.SEED,
        **config.PPO_KWARGS,
    )

    # Save the initial, untrained policy.
    path = config.checkpoint_path(ENV_NAME, 0)
    model.save(path)
    print(f"[ckpt] 0 steps -> {path}")

    previous_step = 0

    for target_step in config.CHECKPOINT_STEPS[1:]:
        steps_to_train = target_step - previous_step

        model.learn(
            total_timesteps=steps_to_train,
            reset_num_timesteps=False,
        )

        path = config.checkpoint_path(ENV_NAME, target_step)
        model.save(path)

        print(f"[ckpt] {target_step:,} steps -> {path}")
        previous_step = target_step

    env.close()

    print("\n[OK] Battery Collector training complete.")
    print(
        "Checkpoints saved to:",
        os.path.join(config.CHECKPOINT_DIR, ENV_NAME),
    )


if __name__ == "__main__":
    train()
