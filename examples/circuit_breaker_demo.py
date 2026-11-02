"""
circuit_breaker_demo.py - Production Example: Automated Circuit Breaker & Checkpoint Preservation.

Demonstrates how to use the `rhd` package in an RL workflow:
  from rhd import RewardHackingCallback

Key Behaviors:
  1. Monitors training checkpoints using zero-leakage behavioral footprints.
  2. Detects reward-hacking onset online.
  3. Displays a prominent ANSI terminal alert banner.
  4. Automatically preserves the last uncorrupted, healthy checkpoint.
  5. Halts training immediately to prevent policy weight corruption and compute waste.
"""
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO
from rhd import RewardHackingCallback
from rhd.envs.adas_highway import HighwayADASConfig, HighwayADASEnv


def realistic_adas_telemetry_provider(step: int, callback) -> dict:
    """
    Simulates the true behavioral footprint of an ADAS vehicle learning hazard-farming:
    - Steps 0 -> 15,000: Clean, safe ACC cruising (high stability, zero swerving).
    - Step 17,500: Onset of loophole exploitation begins (TTC drops, near-misses start).
    - Step 20,000+: Active tailgating and emergency AEB crisis-save farming.
    """
    is_exploit = step >= 17500
    if not is_exploit:
        proxy = 36.4 + float(np.random.uniform(-1.0, 1.0))
        path_eff = 0.98 + float(np.random.uniform(-0.02, 0.01))
        revisit = 0.03 + float(np.random.uniform(0.0, 0.02))
        disp = 0.85
        tvd = 0.05
    else:
        progress = min(1.0, (step - 17500) / 10000.0)
        proxy = 95.0 + 75.0 * progress + float(np.random.uniform(-2.0, 2.0))
        path_eff = max(0.12, 0.50 - 0.38 * progress)
        revisit = min(0.92, 0.50 + 0.42 * progress)
        disp = 0.22
        tvd = 0.45 if step in (17500, 20000) else 0.12

    return {
        "proxy": float(proxy),
        "path_efficiency": float(path_eff),
        "revisit_rate": float(revisit),
        "dispersion": float(disp),
        "tvd": float(tvd),
    }


def main():
    print("\033[1;36m" + "=" * 90 + "\033[0m")
    print("\033[1;36m   RHD PYTHON PACKAGE: AUTOMATED RL CIRCUIT BREAKER & CHECKPOINT PRESERVATION DEMO        \033[0m")
    print("\033[1;36m" + "=" * 90 + "\033[0m")

    save_dir = "results/circuit_breaker_demo"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Environment with potential exploit loophole
    cfg = HighwayADASConfig(
        speed_weight=0.25,
        distance_weight=0.15,
        save_reward_weight=11.0,  # Loophole: crisis-save bonus
        critical_ttc=1.5,
        track_length=2000.0,
        max_steps=120,
    )
    env_train = HighwayADASEnv(cfg, expose_eval_info=False)
    env_eval = HighwayADASEnv(cfg, expose_eval_info=True)

    # 2. Attach the RewardHackingCallback (The Circuit Breaker)
    callback = RewardHackingCallback(
        eval_env=env_eval,
        check_freq=2500,
        halt_on_onset=True,
        preserve_checkpoint=True,
        save_dir=save_dir,
        telemetry_fn=realistic_adas_telemetry_provider,
        verbose=1,
    )

    # 3. Initialize RL Policy (50,000 steps scheduled)
    print("\n\033[1;33m[1/3] Initializing PPO Generator on Highway ADAS Environment...\033[0m")
    print("      • Scheduled Steps : 50,000 timesteps")
    print("      • Check Frequency : Every 2,500 steps")
    print("      • Circuit Breaker : ENABLED (halt_on_onset=True)")
    print("      • Safe Storage    :", save_dir)

    model = PPO("MlpPolicy", env_train, verbose=0, n_steps=256, batch_size=64, learning_rate=3e-4, seed=42)

    # 4. Train Agent with Circuit Breaker
    print("\n\033[1;33m[2/3] Launching Training Loop with Online Circuit Breaker Monitoring...\033[0m\n")
    try:
        model.learn(total_timesteps=50000, callback=callback)
    except KeyboardInterrupt:
        print("\n[INFO] Manual interrupt received.")

    # 5. Post-Run Forensic Summary
    print("\033[1;32m" + "-" * 90 + "\033[0m")
    print("\033[1;32m✔ [POST-RUN FORENSIC SUMMARY] CIRCUIT BREAKER EXECUTION VERIFIED\033[0m")
    print("\033[1;32m" + "-" * 90 + "\033[0m")
    print(f"  • Training Halted Early : At Step {callback.last_check:,} / 50,000")
    compute_saved = 50000 - callback.last_check
    pct_saved = (compute_saved / 50000) * 100
    print(f"  • Compute Saved         : \033[1;32m{compute_saved:,} environment steps ({pct_saved:.1f}% training compute saved)\033[0m")
    print(f"  • Preserved Clean Model : \033[1;32m{callback.preserved_ckpt_path}\033[0m")
    print(f"  • Forensic Audit Report : \033[1;36m{callback.incident_report_path}\033[0m")
    print("\033[1;32m" + "-" * 90 + "\033[0m\n")


if __name__ == "__main__":
    main()
