"""
run_all.py - One-shot runner for GridWorld baseline: train -> evaluate -> analyze.

Usage:
    python baselines/gridworld/run_all.py
"""

import time
import os
import sys

# Ensure baseline directory is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Reward-Hacking Onset Detection - GridWorld Baseline")
    print("=" * 60)

    # Step 1: Train
    print("\n[1/3] TRAINING\n")
    from train import train
    train()

    # Step 2: Evaluate
    print("\n[2/3] EVALUATION\n")
    from evaluate import evaluate
    evaluate()

    # Step 3: Analyze
    print("\n[3/3] ANALYSIS & PLOTS\n")
    from analyze import analyze
    analyze()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  [OK] All done in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nOutput files:")
    print("  checkpoints/gridworld/          <- model checkpoints (.zip)")
    print("  results/metrics.csv             <- evaluation metrics CSV")
    print("  results/*.png                   <- training dynamics plots")


if __name__ == "__main__":
    main()
