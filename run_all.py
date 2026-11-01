"""
run_all.py - Unified Entrypoint for Reward-Hacking Onset Detection

Usage:
    python run_all.py --gridworld      # Runs Person 1's GridWorld baseline pipeline
    python run_all.py --battery        # Runs Teammate's Battery Collector baseline pipeline
    python run_all.py --pilot          # Runs Person 2's RegionToggle pilot sweep (w=0.0, 1.0, 5.0)
    python run_all.py --onset          # Computes and plots onset detection for all pilot runs
    python run_all.py --all            # Runs all baselines, pilots, and onset evaluations
"""

import argparse
import os
import subprocess
import sys


def run_gridworld():
    print("=" * 60)
    print("  [1/3] Running GridWorld Baseline Pipeline (Person 1)")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "baselines", "gridworld", "run_all.py")
    subprocess.check_call([sys.executable, script])


def run_battery():
    print("=" * 60)
    print("  [2/3] Running Battery Collector Baseline Pipeline (Anjali)")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "baselines", "battery_collector", "run_all.py")
    subprocess.check_call([sys.executable, script])


def run_pilot():
    print("=" * 60)
    print("  [3/3] Running RegionToggle Pilot Parameter Sweep (Person 2 - Sabari)")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "scripts", "run_pilot.py")
    subprocess.check_call([sys.executable, script])


def run_onset():
    print("=" * 60)
    print("  Evaluating and Plotting Onset Detection")
    print("=" * 60)
    script = os.path.join(os.path.dirname(__file__), "scripts", "plot_onset.py")
    subprocess.check_call([sys.executable, script])


def run_detector():
    print("=" * 60)
    print("  [4/4] Running LSTM-DQN Onset Detector Pipeline")
    print("=" * 60)
    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    # 1. Dataset preparation
    subprocess.check_call([sys.executable, os.path.join(scripts_dir, "generate_detector_dataset.py")])
    # 2. Detector training
    subprocess.check_call([sys.executable, os.path.join(scripts_dir, "train_detector.py")])
    # 3. Detector evaluation
    subprocess.check_call([sys.executable, os.path.join(scripts_dir, "evaluate_detector.py"), "--split", "test"])
    subprocess.check_call([sys.executable, os.path.join(scripts_dir, "evaluate_detector.py"), "--split", "all"])
    # 4. Plot generation
    subprocess.check_call([sys.executable, os.path.join(scripts_dir, "plot_detector_results.py")])


def run_terminals(use_tmux=False):
    script = os.path.join(os.path.dirname(__file__), "scripts", "launch_3_terminals.py")
    cmd = [sys.executable, script]
    if use_tmux:
        cmd.append("--tmux")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner.")
    parser.add_argument("--gridworld", "--baseline", dest="gridworld", action="store_true", help="Run GridWorld baseline")
    parser.add_argument("--battery", action="store_true", help="Run Battery Collector baseline")
    parser.add_argument("--pilot", action="store_true", help="Run RegionToggle pilot sweep")
    parser.add_argument("--onset", action="store_true", help="Run onset evaluation and plotting")
    parser.add_argument("--detector", action="store_true", help="Run LSTM-DQN detector pipeline")
    parser.add_argument("--all", action="store_true", help="Run all baselines, pilots, onset, and detector")
    parser.add_argument("--terminals", "--parallel", dest="terminals", action="store_true", help="Pop up 3 separate terminal windows running all 3 environments")
    parser.add_argument("--tmux", action="store_true", help="Launch in a 3-way split tmux dashboard")
    args = parser.parse_args()

    if args.terminals or args.tmux:
        run_terminals(use_tmux=args.tmux)
        return

    # If no flags passed, default to gridworld baseline for backwards-compatibility
    if not (args.gridworld or args.battery or args.pilot or args.onset or args.detector or args.all):
        print("No mode specified. Defaulting to GridWorld baseline (--gridworld).")
        print("Pass --help to see available flags (--gridworld, --battery, --pilot, --onset, --detector, --terminals, --all).\n")
        run_gridworld()
        return

    if args.all:
        run_gridworld()
        run_battery()
        run_pilot()
        run_onset()
        run_detector()
    else:
        if args.gridworld:
            run_gridworld()
        if args.battery:
            run_battery()
        if args.pilot:
            run_pilot()
        if args.onset:
            run_onset()
        if args.detector:
            run_detector()


if __name__ == "__main__":
    main()
