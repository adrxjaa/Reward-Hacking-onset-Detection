# Detecting the Onset of Reward Hacking from RL Policy Behavior

[![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)]()
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.4.0-orange.svg)]()
[![Stable--Baselines3](https://img.shields.io/badge/SB3-2.9.0-red.svg)]()

A research framework designed to study **reward hacking (specification gaming)** in Reinforcement Learning and detect its onset using **behavioral policy dynamics**, without requiring access to the true reward function during inference.

---

## 📌 Mid-Review Highlights

* 📖 **Comprehensive Mid-Review Audit:** Detailed technical dossier, empirical findings, and mathematical formulation available at **[`docs/MID_REVIEW_AUDIT.md`](docs/MID_REVIEW_AUDIT.md)**.
* 👥 **Clear Division of Duties:**
  * **Person 1 (Adrija):** Initial 5×5 coin-respawn GridWorld prototype & baseline measurement scripts (`baselines/gridworld/`).
  * **Person 2 (Sabari):** Modular architecture (`rhd/`), `RegionToggleEnv` with strict reward quarantine, probe-state feature extraction engine, formal $O_1$/$O_2$ onset detection algorithms, multi-condition pilot sweeps, and the 27-test automated test suite.
* 🧪 **Empirical Findings:**
  * **Control ($w=0.0$):** Zero hacking, 100% goal success, 0 false positive onsets.
  * **Exploit ($w=1.0, 5.0$):** Maximum exploit oscillation (50 entries/episode), sustained proxy returns, $O_2$ onset confirmed at step 15,000.
* ✅ **Verification:** 27 automated unit tests passing (100% pass rate).

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
pytest
```
*Expected output: `27 passed`.*

### 3. Unified Runner CLI
```bash
# Run Person 1's GridWorld baseline
python run_all.py --baseline

# Run Person 2's RegionToggle pilot parameter sweep (w=0.0, 1.0, 5.0)
python run_all.py --pilot

# Evaluate and plot onset curves
python run_all.py --onset

# Run entire pipeline end-to-end
python run_all.py --all
```

---

## 📂 Repository Structure

```text
Reward-Hacking-onset-Detection/
├── rhd/                          # Person 2 Work: Core research package
│   ├── envs/                     # RegionToggleEnv with quarantined rewards
│   ├── features/                 # Probe-state TVD/JSD feature extraction
│   ├── generators/               # PPO training pipeline with dense checkpointing
│   └── evaluation/               # Mathematical O1/O2 onset detection algorithms
│
├── baselines/                    # Person 1 Work: GridWorld prototype baseline
│   └── gridworld/                # Self-contained baseline scripts and docs
│
├── experiments/pilot/            # Pilot parameter sweeps (run_w0.0, run_w1.0, run_w5.0)
├── scripts/                      # Plotting and debugging CLI scripts
├── tests/                        # 27 automated unit tests (100% pass)
├── docs/                         # Mid-review audit dossier and architectural notes
├── results/                      # Baseline metrics and figures
└── run_all.py                    # Unified project runner
```

---

## 📊 Feature Extraction & Onset Detection

The detector pipeline monitors policy checkpoints across training **without observing the true reward**:
1. **Total Variation Distance (TVD) / JSD on Probe States:** Tracks abrupt reorganizations in action preferences.
2. **Behavioral Dispersion:** Tracks policy entropy collapse across fixed coordinates.
3. **Rollout Dynamics:** Monitors state visitation entropy, path efficiency, and exploit revisit rates.
4. **Onset Detection:** $O_1$ (true performance degradation) and $O_2$ (exploit takeover) identify the exact checkpoint of misalignment.

For full mathematical definitions, plots, and experimental data, see [`docs/MID_REVIEW_AUDIT.md`](docs/MID_REVIEW_AUDIT.md).
