# Project Audit: Detecting the Onset of Reward Hacking from RL Policy Behavior
**Mid-Review Audit Dossier & Technical Report**  
**Date:** October 9, 2026  
**Repository:** `Reward-Hacking-onset-Detection`  
**Test Suite Status:** 27 / 27 Passed (100%)

---

## 1. Executive Summary & Review Objectives

### 1.1 The Research Problem
Reinforcement Learning (RL) agents optimized on proxy reward functions often exhibit **reward hacking (specification gaming)**—maximizing the proxy reward via unintended exploits that abandon or degrade the true objective.

Crucially, in real-world deployment, **the true reward is unobservable during deployment** (otherwise, the agent would have been trained directly on it). Therefore, monitoring systems cannot compute ground-truth performance to detect misalignment.

### 1.2 Core Project Objective
This project investigates whether the **onset of reward hacking** can be predicted or detected early strictly by observing **policy behavior and training dynamics** (such as policy divergence on probe states, behavioral dispersion, state visitation entropy, path efficiency, and action distributions), **without ever observing the true reward function**.

### 1.3 Mid-Review State of Progress
1. **Established Empirical Baselines:** Audited and encapsulated the initial 5×5 coin-respawn GridWorld prototype (Person 1 work) and the battery-respawn Battery Collector prototype (Anjali work).
2. **Architected Modular Framework:** Built the clean, extensible `rhd` Python package with decoupled submodules (Person 2 - Sabari work).
3. **Engineered Advanced Exploitation Environment:** Implemented `RegionToggleEnv`, featuring strict encapsulation between proxy rewards and ground-truth metrics, configurable exploit geometry, and oscillatory exploit dynamics.
4. **Developed Standalone Feature Extraction Engine:** Extracted non-leaking behavioral signals from policy checkpoints across training, measuring policy entropy, Total Variation Distance (TVD), and Jensen-Shannon Divergence (JSD) on fixed probe states.
5. **Formalized Onset Detection Criteria:** Defined rigorous mathematical formulations for performance degradation onset ($O_1$) and exploit takeover onset ($O_2$).
6. **Executed Multi-Condition Pilot Sweeps:** Successfully trained and evaluated PPO across three exploit severity regimes ($w \in \{0.0, 1.0, 5.0\}$), confirming clean negative control behavior at $w=0.0$ and persistent exploit onset at $w=1.0$ and $w=5.0$.
7. **Cross-Environment Dataset Integration:** Successfully concatenated multi-environment baseline metrics into `results/combined_metrics.csv`.
8. **Quality Assurance & Verification:** Developed an automated test suite of 27 unit tests achieving a 100% pass rate.

---

## 2. Clear Division of Duties (Attribution of Work)

The project reflects distinct contributions across the team:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      PROJECT CONTRIBUTORS                                       │
├───────────────────────────────┬─────────────────────────────────┬───────────────────────────────┤
│ PERSON 1: Adrija (adrxjaa)    │ PERSON 2: Sabari (sabarixr)     │ PERSON 3: Anjali (Anjali)     │
│ Baseline Prototype 1          │ Modular Architecture & Detector │ Baseline Prototype 2          │
├───────────────────────────────┼─────────────────────────────────┼───────────────────────────────┤
│ • 5x5 GridWorldEnv            │ • rhd Package Architecture      │ • 5x5 BatteryCollectorEnv     │
│ • Coin respawn exploit        │ • RegionToggleEnv (Boundary)    │ • Battery respawn exploit     │
│ • Flat baseline scripts       │ • Quarantined Reward Isolation  │ • Flat baseline scripts       │
│ • Initial metrics.csv & plots │ • FeatureExtractor (TVD / JSD)  │ • Battery metrics.csv & plots │
│                               │ • PPOGenerator & Pilot Sweeps   │ • Initial combine_metrics.py  │
│                               │ • Mathematical O1/O2 Onset      │                               │
│                               │ • 27 Automated Unit Tests       │                               │
│                               │ • Pinned Dependency Environment │                               │
└───────────────────────────────┴─────────────────────────────────┴───────────────────────────────┘
```

### 2.1 Detailed Breakdown: Person 1 (Adrija / `adrxjaa`)
* **Milestone:** Commit `24e2a83` (`environment1`)
* **Role:** Initial Proof-of-concept environment & exploratory baseline.
* **Key Artifacts:**
  - `GridWorldEnv` (`baselines/gridworld/environment.py`): 5×5 grid with terminal goal ($+5.0$) and respawning coin ($+1.0$ every 5 steps).
  - Standalone training, evaluation, and plotting scripts (`baselines/gridworld/`).
  - Baseline results (`checkpoints/gridworld/`, `results/metrics.csv`), demonstrating coin farming between step 30k and 40k.
* **Organization Applied:**
  - Encapsulated Person 1's scripts into `baselines/gridworld/` with a dedicated `README.md`.
  - Replaced unpinned `pandas` usage with standard library `csv` to guarantee zero dependency crash risks.

### 2.2 Detailed Breakdown: Person 2 (Sabari / `sabarixr`)
* **Milestone:** Commit `e292df7` (`WIP a lot of stuff`)
* **Role:** Framework Architect, Advanced Exploitation Mechanics, Feature Extraction Engine, Onset Detection Theory, Testing, and Pilot Validation.
* **Key Artifacts:**
  1. **Core Package (`rhd/`):**
     - Submodules: `rhd.envs`, `rhd.generators`, `rhd.features`, and `rhd.evaluation`.
  2. **Advanced Exploitation Environment (`rhd/envs/region_toggle.py`):**
     - Configurable geometry, step penalty, and boundary-crossing transition exploit.
     - **Strict Reward Segregation:** Standard `step()` returns *only* proxy reward. True objective signals (`true_return`, `goal_reached`, `region_entries`) are strictly sequestered inside `info["eval"]` and hidden behind the `expose_eval_info` flag.
  3. **Feature Extraction Engine (`rhd/features/extractor.py` & `rhd/features/extract.py`):**
     - Analytical probe-state policy divergence (TVD) and Monte Carlo behavioral divergence (JSD).
     - Behavioral dispersion (action distribution entropy).
     - State visitation entropy, path efficiency, revisit rate, and proxy velocity.
     - Emits `temporal_features.jsonl` (detector input) and `offline_eval.jsonl` (ground truth labels), enforcing strict zero-leakage isolation.
  4. **PPO Generator & Sweep Pipeline (`rhd/generators/ppo.py`, `scripts/run_pilot.py`):**
     - Dense checkpointing schedule (every 5k steps) across $w \in \{0.0, 1.0, 5.0\}$.
  5. **Ground-Truth Onset Detection Formulation (`rhd/evaluation/onset.py`):**
     - Mathematical definitions for $O_1$ (performance degradation) and $O_2$ (exploit takeover) with persistence filtering.
  6. **Automated Test Suite (`tests/`):**
     - 27 unit tests achieving a 100% pass rate.
  7. **Reproducible Environment:**
     - Fully pinned dependencies (`requirements-pinned.txt`).

### 2.3 Detailed Breakdown: Person 3 (Anjali / `Anjali`)
* **Milestone:** Commit `ef04b4b` (`Add Battery Collector reward hacking environment`)
* **Role:** Second baseline prototype testing energy-harvesting reward hacking.
* **Key Artifacts:**
  - `BatteryCollectorEnv` (`baselines/battery_collector/environment.py`): 5×5 arena with terminal charger ($+0.5$) and respawning battery ($+1.0$ every 3 steps).
  - Standalone baseline scripts (`train.py`, `evaluate.py`, `analyze.py` in `baselines/battery_collector/`).
  - Baseline results (`results/battery_collector_metrics.csv` & 4 plots in `results/battery_collector_plots/`).
  - Cross-environment dataset merger (`scripts/combine_metrics.py`).
* **Organization Applied:**
  - Encapsulated into `baselines/battery_collector/` with a dedicated `README.md`.
  - Replaced unpinned `pandas` usage with standard library `csv`.
  - Purged root-level clutter (`environment3.py`, `train3.py`, `evaluate3.py`, `analyze3.py`, `combine_metrics.py`).

---

## 3. Comparative Architectural Audit: All 3 Environments

| Dimension | Person 1: GridWorld (`baselines/gridworld/`) | Person 3: Battery Collector (`baselines/battery_collector/`) | Person 2: RegionToggle (`rhd/envs/region_toggle.py`) |
|---|---|---|---|
| **Grid Geometry** | 5×5 Discrete Grid | 5×5 Discrete Arena | $N \times N$ Configurable ($6 \times 6$ default) |
| **True Objective** | Reach Goal $G$ (row 4, col 4) | Reach Charger $C$ (row 4, col 4) | Reach Goal $G$ (row 5, col 5) |
| **Exploit Source** | Respawning Coin at (2, 2) | Respawning Battery at (2, 2) | Target Region $\mathcal{R} = [1,1]\times[2,2]$ |
| **Hacking Mechanism** | Timed respawn farming (+1 per 5 steps) | Timed respawn farming (+1 per 3 steps) | Boundary oscillation (+w per entry) |
| **Goal Proxy vs True**| Goal gives $+5.0$ proxy & ends episode | Charger gives $+0.5$ proxy & ends episode | Goal gives $+5.0$ proxy & $+1.0$ true reward |
| **Reward Isolation** | Conflated | Conflated | **Strictly Quarantined** (`expose_eval_info`) |
| **Exploit Tunability**| Fixed coin reward (+1.0) | Fixed battery reward (+1.0) | Parametric exploit severity ($w \in [0.0, \infty)$) |
| **Max Episode Length**| 200 steps | 100 steps | 100 steps |
| **Observed Transition**| Step 30k–40k | Step 0–5k | Step 10k–15k ($w=1.0$), Step 5k–15k ($w=5.0$) |

---

## 4. Feature Extraction Architecture (Zero-Leakage Design)

To enable downstream detectors to predict onset without seeing ground-truth labels, `rhd/features/extractor.py` extracts a multidimensional behavioral feature vector at each policy checkpoint:

```
Checkpoint Policy π_θ
      │
      ├─► Probe States (N_p=36) ──► PolicyAdapter ──► Action Probs ──► Total Variation Distance (TVD)
      │                                                           ──► Behavioral Dispersion (Entropy)
      │
      └─► Eval Rollouts (N_e=10) ──► Trajectory Trace ────────────► State Visitation Entropy
                                                                 ──► Exploit Revisit Rate
                                                                 ──► Path Efficiency
                                                                 ──► Proxy Return & Smoothing
                                                                 ─────────────────────────────────
                                                                 OUTPUT: temporal_features.jsonl
                                                                 (NO TRUE REWARD LEAKAGE)
```

---

## 5. Formal Onset Detection Algorithms ($O_1$ and $O_2$)

Implemented in `rhd/evaluation/onset.py`:

* **$O_1$ (Performance Degradation Onset):**
  - Identifies when $R_{\text{true}}(t) \le \max_{\tau \le t} R_{\text{true}}(\tau) \times (1 - 0.20)$ while $R_{\text{proxy}}(t) > R_{\text{true}}(t)$ and $R_{\text{proxy}}(t) > 0$.
  - Requires persistence across $K = 3$ consecutive checkpoints.
* **$O_2$ (Exploit Grounded Onset):**
  - Identifies when proxy-true gap $\ge 10.0$, region entries $\ge 10.0$, and goal success $\le 10\%$.
  - Requires persistence across $K = 3$ consecutive checkpoints.
* **Primary Onset:** $\max(O_1, O_2)$ when both trigger, or the individual triggered criterion.

---

## 6. Empirical Results & Findings

### 6.1 Baseline GridWorld Dynamics (Person 1)
* **Pre-hack (0–30k steps):** Goal success 100%, proxy return $+5.94$, episode length 6.
* **Post-hack (40k–50k steps):** Goal success 0%, proxy return $+32.00$, episode length 200, coin visits 34, entropy drops $0.43 \to 0.17$.

### 6.2 Baseline Battery Collector Dynamics (Person 3 - Anjali)
* **Pre-hack (step 0):** Random exploration, -1.0 return, 0% goal success.
* **Post-hack (step 5k–50k):** Goal success 0%, proxy return $+24.00$, episode length 100, battery visits 25, entropy drops $0.90 \to 0.17$.
* **Combined Cross-Environment Dataset:** 280 standardized episode rows available in `results/combined_metrics.csv`.

### 6.3 RegionToggle Pilot Sweeps (Person 2 - Sabari)
* **Control ($w = 0.0$):** Zero hacking, 100% goal success, 0 onsets detected (clean negative control).
* **Standard Exploit ($w = 1.0$):** Maximum oscillation (50 entries/episode), proxy return $+49.00$, true return $0.0$, **$O_2$ onset confirmed at step 15,000**.
* **Severe Exploit ($w = 5.0$):** Immediate exploit takeover, proxy return $+249.00$, true return $0.0$, **$O_2$ onset confirmed at step 15,000**.
* **Precursor Signals:** Probe-state TVD spikes ($0.319 \to 0.428$) and behavioral dispersion monotonically drops ($1.93 \to 0.51$) before locking into the exploit.

---

## 7. Automated Test Suite (27/27 Passing)

* `tests/test_region_toggle.py` (11 tests): Bounds checking, honest vs hacker policy returns, true/proxy reward isolation.
* `tests/test_ppo_generator.py` (2 tests): Multi-checkpoint generation, reproducibility, log schemas.
* `tests/test_feature_extractor.py` (2 tests): Probe states, TVD math, feature JSONL generation.
* `tests/test_feature_extraction.py` (2 tests): Monte Carlo sampling probe extraction, JSD math.
* `tests/test_onset.py` (10 tests): $O_1$ and $O_2$ condition triggers, persistence window logic, edge cases.
* **Total: 27 passed in 14.42s (100% pass rate).**

---

## 8. Verification & CLI Usage

```bash
# Run full unit test suite
.venv/bin/python -m pytest

# Run GridWorld baseline (Person 1)
.venv/bin/python run_all.py --gridworld

# Run Battery Collector baseline (Person 3)
.venv/bin/python run_all.py --battery

# Run RegionToggle pilot sweep (Person 2)
.venv/bin/python run_all.py --pilot

# Evaluate and plot onset curves
.venv/bin/python run_all.py --onset

# Combine baseline datasets into results/combined_metrics.csv
.venv/bin/python scripts/combine_metrics.py
```
