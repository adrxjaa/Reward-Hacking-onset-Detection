# Codebase Audit & System Architecture

This document provides a concise architectural overview and audit history of the `Reward-Hacking-onset-Detection` repository as of the Mid-Review milestone.

> [!NOTE]
> For the comprehensive mid-review evaluation dossier, empirical results, mathematical formulations, and team division of duties, refer to:
> **[MID_REVIEW_AUDIT.md](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/docs/MID_REVIEW_AUDIT.md)**

---

## 1. System Architecture Overview

```
Reward-Hacking-onset-Detection/
├── baselines/                    # Baseline Proof-of-Concept Prototypes
│   ├── gridworld/                # Person 1 Work: 5x5 GridWorld (Coin Respawn)
│   │   ├── environment.py        # GridWorldEnv
│   │   ├── train.py              # PPO baseline training loop
│   │   ├── evaluate.py           # Evaluation loop
│   │   ├── metrics.py            # Metrics schema & entropy calculation
│   │   ├── analyze.py            # Aggregated dynamics plots
│   │   ├── run_all.py            # Self-contained baseline runner
│   │   ├── config.py             # Baseline hyper-parameters
│   │   └── README.md             # Baseline documentation
│   │
│   └── battery_collector/        # Person 3 Work (Anjali): 5x5 Battery Collector
│       ├── environment.py        # BatteryCollectorEnv
│       ├── train.py              # PPO training loop
│       ├── evaluate.py           # Evaluation loop
│       ├── analyze.py            # Dynamics plots
│       ├── run_all.py            # Self-contained runner
│       ├── config.py             # Configuration
│       └── README.md             # Baseline documentation
│
├── rhd/                          # Person 2 Work (Sabari): Modular research package
│   ├── envs/
│   │   ├── region_toggle.py      # RegionToggleEnv (quarantined rewards, boundary exploit)
│   │   └── __init__.py
│   ├── features/
│   │   ├── extractor.py          # FeatureExtractor (probe-state TVD, rollout entropy)
│   │   ├── extract.py            # Alternative Monte-Carlo probe extractor (JSD)
│   │   └── __init__.py
│   ├── generators/
│   │   ├── ppo.py                # PPOGenerator with dense checkpointing
│   │   └── __init__.py
│   └── evaluation/
│       ├── onset.py              # Mathematical O1/O2 onset detection algorithms
│       └── __init__.py
│
├── experiments/pilot/            # Pilot parameter sweeps across exploit weights
│   ├── run_w0.0/                 # Negative control (region_reward = 0.0, honest policy)
│   ├── run_w1.0/                 # Standard exploit (region_reward = 1.0, O2 at step 15k)
│   └── run_w5.0/                 # Severe exploit (region_reward = 5.0, O2 at step 15k)
│
├── scripts/                      # Experiment runners and visualization tools
│   ├── run_pilot.py              # Multi-condition pilot sweep runner
│   ├── plot_onset.py             # Multi-panel onset curve plotting
│   ├── plot_smoke.py             # Behavioral feature diagnostic plots
│   ├── debug_region_toggle.py    # Interactive step-by-step ANSI terminal visualizer
│   └── combine_metrics.py        # Baseline cross-environment dataset merger
│
├── tests/                        # Automated unit test suite (27 tests, 100% pass)
│   ├── test_region_toggle.py     # Environment mechanics & reward isolation
│   ├── test_ppo_generator.py     # PPO generator training & checkpoints
│   ├── test_feature_extractor.py # Analytical policy probe & TVD
│   ├── test_feature_extraction.py# Sampling-based probe & JSD
│   └── test_onset.py             # O1 & O2 onset algorithms and persistence
│
├── docs/                         # Documentation and audit reports
│   ├── MID_REVIEW_AUDIT.md       # Comprehensive mid-review technical dossier
│   └── CODEBASE_AUDIT.md         # Architecture and codebase index (this file)
│
├── checkpoints/gridworld/        # Baseline model checkpoints
├── results/                      # Baseline metrics CSVs and figures
├── requirements-pinned.txt       # Exact reproducible environment dependencies
├── requirements.txt              # High-level package requirements
├── run_all.py                    # Top-level unified CLI runner
└── README.md                     # Main repository presentation & quickstart
```

---

## 2. Division of Duties Summary

| Area | Person 1: Adrija | Person 2: Sabari | Person 3: Anjali |
|---|---|---|---|
| **Phase** | Phase 0 Baseline | Phase 1 Modular Engine & Detector | Phase 0 Parallel Baseline |
| **Environment** | `baselines/gridworld/` | `rhd/envs/region_toggle.py` | `baselines/battery_collector/` |
| **Exploit Dynamic** | Respawning coin (+1 per 5s) | Boundary oscillation (+w per entry) | Respawning battery (+1 per 3s) |
| **Reward Isolation** | Conflated | **Strictly Quarantined** (`expose_eval_info`) | Conflated |
| **Feature Extraction**| CSV metrics | Probe-state policy divergence (TVD/JSD) | CSV metrics |
| **Onset Detection** | None | Formal mathematical $O_1$ and $O_2$ | None |
| **Testing** | None | 27 automated unit tests (`tests/`) | None |
| **Dependency Lock** | Unpinned | Pinned (`requirements-pinned.txt`) | Unpinned |

---

## 3. Quickstart & Verification Commands

```bash
# 1. Run all unit tests
.venv/bin/python -m pytest

# 2. Run the GridWorld baseline
.venv/bin/python run_all.py --gridworld

# 3. Run the Battery Collector baseline
.venv/bin/python run_all.py --battery

# 4. Run the RegionToggle pilot sweep
.venv/bin/python run_all.py --pilot

# 5. Generate onset analysis plots
.venv/bin/python run_all.py --onset
```
