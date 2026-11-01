# Detecting the Onset of Reward Hacking from RL Policy Behavior

[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)]()
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.4.0-orange.svg)]()
[![Stable--Baselines3](https://img.shields.io/badge/SB3-2.9.0-red.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

A research framework designed to study **reward hacking (specification gaming)** in Reinforcement Learning and detect its onset using **behavioral policy dynamics**, without requiring access to the true reward function during inference.

---

## 📌 Research & Architecture Highlights

* 🤖 **Dual-RL Sequential Detection Architecture:** Extended with an **LSTM-DQN Reward Hacking Onset Detector** (`rhd/detectors/`) that learns the optimal stopping rule to identify reward hacking onset from non-leaking behavioral checkpoint sequences. Full documentation at **[`docs/LSTM_DQN_DETECTOR.md`](docs/LSTM_DQN_DETECTOR.md)**.
* 📖 **Comprehensive Mid-Review Audit:** Detailed technical dossier, empirical findings, and mathematical formulation available at **[`docs/MID_REVIEW_AUDIT.md`](docs/MID_REVIEW_AUDIT.md)**.
* 👥 **Clear Division of Duties:**
  * **Person 1 (Adrija):** Initial 5×5 coin-respawn GridWorld prototype & baseline measurement scripts (`baselines/gridworld/`).
  * **Person 2 (Sabari):** Modular architecture (`rhd/`), `RegionToggleEnv` with strict reward quarantine, probe-state feature extraction engine, formal $O_1$/$O_2$ onset detection algorithms, multi-condition pilot sweeps, and unit tests.
  * **Person 3 (Anjali):** Energy-harvesting Battery Collector baseline environment (`baselines/battery_collector/`) and cross-environment data merger (`scripts/combine_metrics.py`).
  * **Learned Detection Extension:** LSTM-DQN sequential stopping detector with zero-leakage training rewards, run-level data partitioning, and comparative benchmarking.
* 🧪 **Empirical Findings:**
  * **Control ($w=0.0$):** Zero hacking, 100% goal success, 0 false alarms across both mathematical and LSTM-DQN detectors.
  * **Exploit ($w=1.0, 5.0$):** Immediate exploit takeover, sustained proxy returns, $O_2$ onset confirmed at step 15,000.
  * **LSTM-DQN Performance:** Achieves **1.000 F1-score** on unseen test runs, detecting 100% of reward hacking onsets with **0% false alarms** on clean controls and a low latency of 0–1 checkpoint.
* ✅ **Verification:** 38 automated unit tests passing (100% pass rate).

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Activate existing virtualenv
source .venv/bin/activate

# Or install dependencies from pinned lockfile
pip install -r requirements-pinned.txt
```

### 2. Run Test Suite
```bash
python -m pytest
```
*Expected output: `38 passed`.*

### 3. Unified Runner CLI
```bash
# Run Person 1's GridWorld baseline
python run_all.py --gridworld

# Run Person 3's Battery Collector baseline
python run_all.py --battery

# Run Person 2's RegionToggle pilot parameter sweep (w=0.0, 1.0, 5.0)
python run_all.py --pilot

# Evaluate and plot mathematical onset curves
python run_all.py --onset

# Run full LSTM-DQN detector pipeline (dataset prep, training, eval, and plots)
python run_all.py --detector

# Run entire pipeline end-to-end
python run_all.py --all
```

---

## 📂 Repository Structure

```text
Reward-Hacking-onset-Detection/
├── rhd/                          # Core research package
│   ├── envs/                     # RegionToggleEnv with quarantined rewards
│   ├── features/                 # Probe-state TVD/JSD feature extraction
│   ├── generators/               # PPO training pipeline with dense checkpointing
│   ├── evaluation/               # Mathematical O1/O2 onset detection algorithms
│   └── detectors/                # Learned LSTM-DQN Onset Detector
│       ├── models.py             # Recurrent LSTM Q-Network architecture
│       ├── replay_buffer.py      # Windowed transition experience replay
│       ├── environment.py        # Detector POMDP environment & reward logic
│       ├── agent.py              # DQN agent with Bellman target updates
│       ├── trainer.py            # Training loop with validation checkpointing
│       └── evaluator.py          # Precision/Recall/F1/Delay comparative evaluation
│
├── baselines/                    # Proof-of-concept baseline environments
│   ├── gridworld/                # Person 1 Work: GridWorld prototype baseline
│   └── battery_collector/        # Person 3 Work: Battery Collector baseline
│
├── experiments/
│   ├── pilot/                    # Initial pilot sweeps (w=0.0, 1.0, 5.0)
│   └── detector_dataset/         # Multi-seed generator runs for detector training
│
├── scripts/                      # Plotting and runner CLI scripts
│   ├── generate_detector_dataset.py  # Parallel multi-seed run generation & feature extraction
│   ├── train_detector.py         # LSTM-DQN training CLI
│   ├── evaluate_detector.py      # Comparative evaluation against O1/O2
│   ├── plot_detector_results.py  # Scientific publication figure generation
│   └── plot_onset.py             # Mathematical onset curves
│
├── tests/                        # 38 automated unit tests (100% pass)
├── docs/                         # Technical documentation & audit dossiers
│   ├── LSTM_DQN_DETECTOR.md      # Dual-RL detector architecture and results
│   └── MID_REVIEW_AUDIT.md       # Mid-review dossier and attribution
├── results/                      # Baseline metrics, evaluation CSVs, and plots
│   ├── detector/                 # Model checkpoints, split metadata, and predictions
│   └── detector_plots/           # 5 scientific figures from actual experiments
└── run_all.py                    # Unified project runner CLI
```

---

## 📊 Dual-RL Feature Extraction & Detection

The detector pipeline monitors policy checkpoints across training **without observing the true reward**:
1. **Total Variation Distance (TVD) / JSD on Probe States:** Tracks abrupt reorganizations in action preferences.
2. **Behavioral Dispersion:** Tracks policy entropy collapse across fixed coordinates.
3. **Rollout Dynamics:** Monitors state visitation entropy, path efficiency, and exploit revisit rates.
4. **Mathematical Onset Detection:** $O_1$ (true performance degradation) and $O_2$ (exploit takeover) identify the ground-truth onset checkpoint for training rewards.
5. **LSTM-DQN Sequential Detector:** A learned recurrent agent that processes rolling feature windows to identify exploit onset online with high precision and low latency.

For full mathematical definitions, plots, and experimental data, see:
- [`docs/LSTM_DQN_DETECTOR.md`](docs/LSTM_DQN_DETECTOR.md)
- [`docs/MID_REVIEW_AUDIT.md`](docs/MID_REVIEW_AUDIT.md)

