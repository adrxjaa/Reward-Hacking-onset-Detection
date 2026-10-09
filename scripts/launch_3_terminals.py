#!/usr/bin/env python3
"""
scripts/launch_3_terminals.py

Launches all 3 reward-hacking environments in separate terminals or panes:
  Terminal 1: GridWorld Baseline (Person 1 - Adrija)
  Terminal 2: Battery Collector Baseline (Person 3 - Anjali)
  Terminal 3: RegionToggle & Onset Detection (Person 2 - Sabari)

Supports GUI terminal popups (kitty, gnome-terminal, etc.) and tmux 3-way split.
"""

import argparse
import os
import shutil
import subprocess
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "bin", "python")

CMD_GRIDWORLD = f"cd '{PROJECT_DIR}' && '{VENV_PYTHON}' run_all.py --gridworld; echo ''; read -p 'Finished GridWorld. Press Enter to close...' "
CMD_BATTERY = f"cd '{PROJECT_DIR}' && '{VENV_PYTHON}' run_all.py --battery; echo ''; read -p 'Finished Battery Collector. Press Enter to close...' "
CMD_REGIONTOGGLE = f"cd '{PROJECT_DIR}' && '{VENV_PYTHON}' run_all.py --pilot && '{VENV_PYTHON}' run_all.py --onset; echo ''; read -p 'Finished RegionToggle. Press Enter to close...' "


def launch_tmux():
    """Launch all 3 inside a 3-pane tmux session."""
    session = "reward_hacking_review"
    subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)

    # Window 0, Pane 1 (GridWorld)
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, "-n", "Environments", f"bash -c \"{CMD_GRIDWORLD}\""])
    # Pane 2 (Battery Collector - split horizontal)
    subprocess.check_call(["tmux", "split-window", "-h", "-t", f"{session}:0", f"bash -c \"{CMD_BATTERY}\""])
    # Pane 3 (RegionToggle - split vertical)
    subprocess.check_call(["tmux", "split-window", "-v", "-t", f"{session}:0.0", f"bash -c \"{CMD_REGIONTOGGLE}\""])
    subprocess.check_call(["tmux", "select-layout", "-t", f"{session}:0", "tiled"])

    print(f"\n[OK] Started tmux session '{session}' with 3 parallel panes.")
    print("Attaching to tmux session now... (Detach anytime with Ctrl+B then D)")
    os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def launch_gui():
    """Launch 3 separate popup terminal windows."""
    terminal = None
    candidates = ["kitty", "gnome-terminal", "alacritty", "konsole", "xfce4-terminal", "tilix", "terminator", "xterm"]
    for c in candidates:
        if shutil.which(c):
            terminal = c
            break

    if not terminal:
        print("[!] No supported GUI terminal found. Falling back to tmux 3-pane split.")
        launch_tmux()
        return

    print(f"[*] Launching 3 separate {terminal} windows...\n")

    runs = [
        ("GridWorld-Baseline", CMD_GRIDWORLD),
        ("Battery-Collector", CMD_BATTERY),
        ("RegionToggle-Detector", CMD_REGIONTOGGLE),
    ]

    for title, cmd in runs:
        if terminal == "kitty":
            subprocess.Popen(["kitty", "--title", title, "bash", "-c", cmd])
        elif terminal == "gnome-terminal":
            subprocess.Popen(["gnome-terminal", "--title", title, "--", "bash", "-c", cmd])
        elif terminal == "alacritty":
            subprocess.Popen(["alacritty", "--title", title, "-e", "bash", "-c", cmd])
        elif terminal == "konsole":
            subprocess.Popen(["konsole", "--title", title, "-e", "bash", "-c", cmd])
        elif terminal == "xterm":
            subprocess.Popen(["xterm", "-title", title, "-e", "bash", "-c", cmd])
        else:
            subprocess.Popen([terminal, "-e", "bash", "-c", cmd])

    print("[OK] All 3 terminal windows launched simultaneously!")


def main():
    parser = argparse.ArgumentParser(description="Launch all 3 environments in parallel terminals.")
    parser.add_argument("--tmux", action="store_true", help="Launch in a 3-way split tmux dashboard")
    args = parser.parse_args()

    if args.tmux:
        launch_tmux()
    else:
        launch_gui()


if __name__ == "__main__":
    main()
