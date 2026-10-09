"""
run_all.py - One-shot runner for Battery Collector baseline: train -> evaluate -> analyze.
"""

import os
import sys
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Reward-Hacking Onset Detection - Battery Collector Baseline")
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
    from analyze import main as analyze_main
    analyze_main()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  [OK] All done in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
