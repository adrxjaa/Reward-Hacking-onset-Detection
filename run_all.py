"""
run_all.py - Unified Entrypoint for Reward-Hacking Onset Detection

Usage:
    python run_all.py --baseline       # Runs Person 1's GridWorld baseline pipeline
    python run_all.py --pilot          # Runs Person 2's RegionToggle pilot sweep (w=0.0, 1.0, 5.0)
    python run_all.py --onset          # Computes and plots onset detection for all pilot runs
    python run_all.py --all            # Runs baseline, pilot, and onset evaluation end-to-end
"""

import argparse
import os
import subprocess
import sys


def run_baseline():
    print("=" * 60)
    print("  [1/2] Running Person 1's GridWorld Baseline Pipeline")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "baselines", "gridworld", "run_all.py")
    subprocess.check_call([sys.executable, script])


def run_pilot():
    print("=" * 60)
    print("  [2/2] Running Person 2's RegionToggle Pilot Sweep")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "scripts", "run_pilot.py")
    subprocess.check_call([sys.executable, script])


def run_onset():
    print("=" * 60)
    print("  Evaluating and Plotting Onset Detection")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "scripts", "plot_onset.py")
    subprocess.check_call([sys.executable, script])


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument("--baseline", action="store_true", help="Run GridWorld baseline")
    parser.add_argument("--pilot", action="store_true", help="Run RegionToggle pilot sweep")
    parser.add_argument("--onset", action="store_true", help="Run onset evaluation and plotting")
    parser.add_argument("--all", action="store_true", help="Run baseline, pilot, and onset")
    args = parser.parse_args()

    # If no flags passed, default to baseline for backwards-compatibility
    if not (args.baseline or args.pilot or args.onset or args.all):
        print("No mode specified. Defaulting to GridWorld baseline (--baseline).")
        print("Pass --help to see available flags (--baseline, --pilot, --onset, --all).\n")
        run_baseline()
        return

    if args.all:
        run_baseline()
        run_pilot()
        run_onset()
    else:
        if args.baseline:
            run_baseline()
        if args.pilot:
            run_pilot()
        if args.onset:
            run_onset()


if __name__ == "__main__":
    main()
