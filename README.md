# Detecting the Onset of Reward Hacking from RL Policy Behavior

[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)]()
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.4.0-orange.svg)]()
[![Stable--Baselines3](https://img.shields.io/badge/SB3-2.9.0-red.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()

A research framework designed to study **reward hacking (specification gaming)** in Reinforcement Learning and detect its onset using **behavioral policy dynamics**, without requiring access to the true reward function during inference.

---

## 📌 Research & Architecture Highlights

* 🤖 **Dual-RL Sequential Detection Architecture:** Equipped with original **LSTM-DQN** and **MLP-DQN** onset detectors (`rhd/detectors/`) that learn the optimal stopping rule to identify reward hacking onset from non-leaking behavioral checkpoint sequences. Full documentation at **[`docs/LSTM_DQN_DETECTOR.md`](docs/LSTM_DQN_DETECTOR.md)**.
* 🔬 **Controlled 10-Feature Ablation Study:** Full empirical study investigating the relative contribution of temporal trend signals vs. spatial revisitation/anomaly signals across 5 random seeds on 4 partitions (Train, Validation, Unseen Phase 2, and Final Test). See **[`results/ablation_study/ablation_report.md`](results/ablation_study/ablation_report.md)**.
* 📖 **Comprehensive Reports:**
  * **Phase 3 Final Evaluation Report:** Multi-seed evaluation on held-out generator test runs (seeds 701–706).
  * **Phase 2 Validation Report:** **[`docs/EXPERIMENTAL_VALIDATION_PHASE2.md`](docs/EXPERIMENTAL_VALIDATION_PHASE2.md)**.
  * **Mid-Review Technical Dossier:** **[`docs/MID_REVIEW_AUDIT.md`](docs/MID_REVIEW_AUDIT.md)**.
* 🧪 **Key Empirical Findings:**
  * **Original LSTM-DQN:** Achieves **1.000 ± 0.000 F1-score** across 5 random seeds on 18 held-out generator test runs (seeds 701–706), with **0.0% clean false alarms** and **0.72 checkpoints detection delay**.
  * **Original MLP-DQN:** Achieves **0.936 ± 0.049 F1-score** on the final test partition.
  * **Feature Contribution:** Revisitation and spatial anomaly features (`path_efficiency`, `revisit_rate`, `behavioral_dispersion`) provide the largest marginal contribution; ablated models drop by ~10% F1 on MLP-DQN with a sharp surge in evaluation regret.
  * **Zero Leakage:** Rigorous forensic audit confirms zero ground truth, temporal, or distributional leakage into detector inputs (`scripts/audit_detector_inputs.py`).
* ✅ **Verification:** 40 automated unit tests passing (100% pass rate).

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
python -m pytest -q
```
*Expected output: `40 passed`.*

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

# Run Controlled 10-Feature Ablation Study (multi-seed across all 4 partitions)
python scripts/run_feature_ablation.py --seeds 0 1 2 3 4 --episodes 300

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

