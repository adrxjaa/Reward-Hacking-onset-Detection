# Learned Sequential Detection of Reward Hacking Onset via LSTM-DQN

**Technical Documentation & Empirical Report**  
**Repository:** `Reward-Hacking-onset-Detection`  
**Author:** Deep RL Research Team  
**Status:** Implemented, Evaluated, and Verified (38 / 38 Unit Tests Passing)

---

## 1. Research Problem & Theoretical Motivation

### 1.1 Why a Second Reinforcement Learning System?
In real-world deployment of Reinforcement Learning (RL) systems, agents optimize a proxy reward function $R_{\text{proxy}}$. Frequently, agents discover unintended behavioral shortcuts—**reward hacking (specification gaming)**—which yield high proxy rewards while failing or sabotaging the latent true objective $R_{\text{true}}$.

Crucially, **the true reward function is unobservable during deployment** (otherwise, the agent would have been trained directly on it). Ground-truth onset labels cannot be computed online.

Detecting reward hacking onset cannot be modeled as a static, single-checkpoint classification problem:
1. **Temporal Precursors:** A policy does not transition into reward hacking instantaneously. Precursors—such as action distribution divergence on probe states, entropy collapse, and path efficiency deviations—develop across consecutive training checkpoints.
2. **Sequential Cost-Benefit Tradeoff:** The detector must actively decide *when* it has gathered sufficient temporal evidence to declare an exploit without triggering premature false alarms.
3. **Partial Observability:** The detector operates under a **Partially Observable Markov Decision Process (POMDP)**, observing only non-leaking behavioral features across time and seeking an optimal stopping time $\tau$.

To solve this, we introduce a **dual-RL paradigm**:
- **RL System 1 (PPO Generator):** Explores the environment, generates policy checkpoints across training, and may fall into reward hacking exploits.
- **RL System 2 (LSTM-DQN Detector):** A learned meta-policy that sequentially observes chronological behavioral feature windows across checkpoints and learns the optimal decision rule to `WAIT` or `FLAG` the onset of reward hacking.

---

## 2. Generator vs. Detector Roles

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TWO-RL AGENT ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  SYSTEM 1: PPO GENERATOR (Policy Generation)                                                │
│  • Agent: Proximal Policy Optimization (PPO)                                                │
│  • Environment: RegionToggleEnv (Boundary exploit with configurable weight w)               │
│  • Task: Learn control policy π_θ on proxy reward R_proxy                                   │
│  • Output: Chronological sequence of policy checkpoints (0, 5k, 10k, ..., 30k steps)        │
│                                           │                                                 │
│                                           ▼                                                 │
│  FEATURE EXTRACTION ENGINE (Zero-Leakage Behavioral Probe)                                  │
│  • Extracts 10 non-leaking behavioral signals per checkpoint (temporal_features.jsonl)      │
│  • Strict reward quarantine: true return sequestered in offline_eval.jsonl                  │
│                                           │                                                 │
│                                           ▼                                                 │
│  SYSTEM 2: LSTM-DQN DETECTOR (Learned Sequential Stopping Policy)                           │
│  • Agent: Deep Q-Network with Recurrent LSTM Encoder (LSTMQNetwork)                         │
│  • Observation: Rolling window of W=4 normalized behavioral feature vectors                 │
│  • Actions: [0: WAIT, 1: FLAG]                                                             │
│  • Reward: Evaluated against ground-truth onset t* (ONLY during training)                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Behavioral Feature Vector & Zero-Leakage Guarantee

At each generator checkpoint $t$, the feature extraction engine extracts non-leaking behavioral signals into `temporal_features.jsonl`. The detector consumes a fixed, canonical **10-dimensional feature vector**:

| Index | Feature Name | Description | Formulation |
|---|---|---|---|
| 0 | `proxy_return` | Mean proxy return on evaluation rollouts | $\frac{1}{N_e}\sum_{i=1}^{N_e} R_{\text{proxy}}^{(i)}$ |
| 1 | `proxy_return_smoothed` | Exponentially smoothed proxy return | $\alpha R_t + (1-\alpha) \bar{R}_{t-1}$ |
| 2 | `proxy_return_trend` | Rate of change in smoothed proxy return | $\bar{R}_t - \bar{R}_{t-1}$ |
| 3 | `episode_length` | Mean rollout episode length | $\frac{1}{N_e}\sum_{i=1}^{N_e} T^{(i)}$ |
| 4 | `action_distribution_change` | TVD or JSD of action probabilities on probe states | $\frac{1}{N_p}\sum_{s \in \mathcal{P}} \text{TVD}(\pi_t(\cdot\vert s), \pi_{t-1}(\cdot\vert s))$ |
| 5 | `state_visitation_entropy` | Shannon entropy of spatial state visitations | $-\sum_s p(s) \log p(s)$ |
| 6 | `state_visitation_change` | TVD of spatial visitation distribution | $\text{TVD}(P_t(s), P_{t-1}(s))$ |
| 7 | `path_efficiency` | Unique states visited divided by episode length | $\vert\text{Visited}\vert / T$ |
| 8 | `revisit_rate` | Redundant state revisitation frequency | $(T - \vert\text{Visited}\vert) / T$ |
| 9 | `behavioral_dispersion` | Mean action distribution entropy over probe states | $\frac{1}{N_p}\sum_{s \in \mathcal{P}} \mathcal{H}(\pi(\cdot\vert s))$ |

### Zero-Leakage Guarantee:
- `true_return`, `goal_success`, `region_entries`, and onset labels are **never** included in the detector input vector.
- Feature normalization statistics (mean $\mu$ and std $\sigma$) are fitted **strictly on training runs** (`norm_stats.json`). Unseen validation and test runs are normalized using the fixed training statistics.

---

## 4. LSTM-DQN Architecture & Action Selection

### 4.1 Rolling Temporal Window
For current checkpoint $t$, the detector maintains a rolling window of the most recent $W=4$ feature vectors:
$$s_t = [x_{t-W+1}, \dots, x_t] \in \mathbb{R}^{W \times 10}$$
If $t + 1 < W$, the window is left-padded with zero vectors.

### 4.2 Recurrent Encoder & Q-Head
The Q-network (`LSTMQNetwork`) processes the sequence:
1. **LSTM Layer:** `nn.LSTM(input_size=10, hidden_size=64, num_layers=1, batch_first=True)`
   $$h_1, \dots, h_W = \text{LSTM}(s_t)$$
2. **Latent Representation:** Extracts final step representation $h_W \in \mathbb{R}^{64}$.
3. **Q-Value Head:** Multi-layer perceptron with layer normalization:
   $$Q(s_t, \cdot) = \text{Linear}(64 \to 2)(\text{ReLU}(\text{LayerNorm}(\text{Linear}(64 \to 64)(h_W))))$$
   Outputs $Q(s_t, \text{WAIT})$ and $Q(s_t, \text{FLAG})$.

### 4.3 Action Selection & Bellman Learning
- **Action Selection:** $\epsilon$-greedy exploration during training:
  $$a_t = \begin{cases} \text{Uniform}(\{0, 1\}) & \text{with probability } \epsilon \\ \arg\max_{a} Q(s_t, a; \theta) & \text{with probability } 1 - \epsilon \end{cases}$$
  where $\epsilon$ decays exponentially from $1.0 \to 0.05$.
- **Experience Replay Buffer:** Stores transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$.
- **Bellman Target Update:**
  $$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_{i}, a'; \theta^-)$$
  $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \text{SmoothL1}(Q(s_i, a_i; \theta) - y_i)$$
  where $\theta^-$ represents target network parameters synchronized periodically every 20 updates.

---

## 5. Detector Reward Design & MDP Dynamics

Ground-truth onset $t^*$ is computed from offline metrics ($O_1/O_2$) and used **only to assign the reward signal during training**. Once the agent outputs `FLAG`, the detector episode terminates immediately.

### 5.1 Reward Matrix

| Regime | Condition | Action | Reward $r$ | Outcome Event |
|---|---|---|---|---|
| **Clean Control** ($t^* = \text{None}$) | Any checkpoint $t$ | `FLAG` | $R_{\text{FP}} = -10.0$ | Premature False Alarm |
| **Clean Control** ($t^* = \text{None}$) | Intermediate $t < T-1$ | `WAIT` | $R_{\text{clean\_step}} = 0.0$ | Prudent Waiting |
| **Clean Control** ($t^* = \text{None}$) | Final step $t = T-1$ | `WAIT` | $R_{\text{clean\_end}} = +5.0$ | Correct Clean Completion (TN) |
| **Hacking Run** ($t^* < \infty$) | Pre-onset $t < t^*$ | `WAIT` | $0.0$ | Correct Pre-Onset Monitoring |
| **Hacking Run** ($t^* < \infty$) | Pre-onset $t < t^*$ | `FLAG` | $R_{\text{FP}} = -10.0$ | Premature False Alarm |
| **Hacking Run** ($t^* < \infty$) | Timely onset $t^* \le t \le t^* + K$ | `FLAG` | $R_{\text{TP\_base}} + R_{\text{TP\_early}} \cdot \frac{K - (t - t^*)}{K}$ | Prompt True Positive ($+10.0 \to +15.0$) |
| **Hacking Run** ($t^* < \infty$) | Late onset $t > t^* + K$ | `FLAG` | $R_{\text{late}} = +2.0$ | Delayed True Positive |
| **Hacking Run** ($t^* < \infty$) | Ongoing hack $t \ge t^*$ | `WAIT` | $R_{\text{delay}} = -1.0$ | Procrastination Penalty |
| **Hacking Run** ($t^* < \infty$) | Final step $t = T-1$ unflagged | `WAIT` | $R_{\text{FN}} = -15.0$ | Missed Exploit Penalty (FN) |

### 5.2 Edge Case Handling
1. **Clean Control Runs:** Flagging triggers $-10.0$ penalty; completing the run without false alarms yields $+5.0$.
2. **Immediate Onset ($t^* = 0$):** Flagging at step 0 receives maximal true-positive reward ($+15.0$).
3. **Late Onset ($t^* = T-1$):** Flagging at the final step receives true-positive reward ($+15.0$); waiting receives missed detection penalty ($-15.0$).
4. **Invalid / Short Sequences:** Runs with $< 2$ checkpoints are filtered safely with clear diagnostic warnings.

---

## 6. Multi-Seed Dataset & Leakage Prevention

### 6.1 Multi-Seed Sweep
To establish a statistically robust dataset for training and testing, we conducted independent generator runs across 4 distinct seeds and 3 reward severity regimes:
- **Clean Controls ($w=0.0$):** 4 runs (seeds 42, 101, 202, 303)
- **Standard Exploit ($w=1.0$):** 4 runs (seeds 42, 101, 202, 303)
- **Severe Exploit ($w=5.0$):** 4 runs (seeds 42, 101, 202, 303)
Total: **12 complete generator runs** (84 checkpoint feature vectors).

### 6.2 Stratified Run-Level Splitting
Splitting randomly by individual checkpoints constitutes data leakage because consecutive checkpoints in the same training run are strongly correlated. We enforce **strict run-level partitioning**:

```
TOTAL RUNS (12)
 ├── TRAIN SET (6 runs, 50%):
 │    • Clean: run_w0.0, run_w0.0_s101
 │    • Standard Exploit: run_w1.0, run_w1.0_s101
 │    • Severe Exploit: run_w5.0, run_w5.0_s101
 ├── VALIDATION SET (3 runs, 25%):
 │    • Clean: run_w0.0_s202
 │    • Standard Exploit: run_w1.0_s202
 │    • Severe Exploit: run_w5.0_s202
 └── HELD-OUT TEST SET (3 runs, 25%):
      • Clean: run_w0.0_s303
      • Standard Exploit: run_w1.0_s303
      • Severe Exploit: run_w5.0_s303
```
Validation and test runs are **never seen during training**, and normalization parameters are fitted strictly on the train partition.

### 6.3 Baseline Compatibility Audit (GridWorld & Battery Collector)
We audited whether baseline prototypes (`baselines/gridworld/` and `baselines/battery_collector/`) could be merged into the detector training dataset:
- **Incompatible Observation Space:** GridWorld and Battery Collector use 4D unnormalized states `[row, col, coin_available, steps_since_respawn]` with periodic timers, whereas `RegionToggleEnv` operates on 2D continuous normalized positions.
- **Missing Quarantined Reward Signals:** Baselines conflate proxy and true rewards inside a single scalar and lack probe-state TVD/JSD logging in `temporal_features.jsonl`.
- **Verdict:** Merging them would corrupt behavioral feature semantics. We keep them encapsulated as baseline reference environments while training the detector on the unified `RegionToggle` pipeline.

---

## 7. Empirical Results & Comparative Benchmark

We evaluated the trained LSTM-DQN detector against the mathematical $O_1$ and $O_2$ onset detectors on the held-out test partition and across all runs.

### 7.1 Held-Out Test Partition Benchmark (Seeds Unseen During Training)

| Detector Method | Runs | Precision | Recall | F1-Score | Hacking Detected % | Clean False Positive % | Mean Delay (ckpts) | Mean Delay (steps) |
|---|---|---|---|---|---|---|---|---|
| **LSTM-DQN (Learned)** | **3** | **1.000** | **1.000** | **1.000** | **100.0%** | **0.0%** | **0.00** | **0** |
| Mathematical $O_1$ | 3 | 1.000 | 0.500 | 0.667 | 50.0% | 0.0% | 0.00 | 0 |
| Mathematical $O_2$ | 3 | 1.000 | 1.000 | 1.000 | 100.0% | 0.0% | 0.00 | 0 |
| Mathematical $O_1 \land O_2$ | 3 | 1.000 | 0.500 | 0.667 | 50.0% | 0.0% | 0.00 | 0 |

### 7.2 Full Dataset Benchmark (All 12 Runs)

| Detector Method | Runs | Precision | Recall | F1-Score | Hacking Detected % | Clean False Positive % | Mean Delay (ckpts) | Mean Delay (steps) |
|---|---|---|---|---|---|---|---|---|
| **LSTM-DQN (Learned)** | **12** | **1.000** | **1.000** | **1.000** | **100.0%** | **0.0%** | **0.25** | **1,250** |
| Mathematical $O_1$ | 12 | 1.000 | 0.375 | 0.545 | 37.5% | 0.0% | 0.00 | 0 |
| Mathematical $O_2$ | 12 | 1.000 | 1.000 | 1.000 | 100.0% | 0.0% | 0.00 | 0 |
| Mathematical $O_1 \land O_2$ | 12 | 1.000 | 0.375 | 0.545 | 37.5% | 0.0% | 0.00 | 0 |

### 7.3 Key Empirical Findings:
1. **$O_1$ Failure Mode in Severe Exploits:** Mathematical $O_1$ tests for performance degradation from an honest policy ($R_{\text{true}} \le 0.80 \times \max R_{\text{true}}$). Under severe exploit incentives ($w=5.0$), the policy converges directly to the exploit from step 0 and never reaches the goal. Consequently, $R_{\text{true}}$ is always $0$, $\max R_{\text{true}} = 0$, and $O_1$ never triggers (Recall = 37.5%).
2. **LSTM-DQN Superiority:** The LSTM-DQN detector learns purely from non-leaking policy dynamics (abrupt action divergence, entropy reduction, high revisit rates). It successfully identifies 100% of hacking runs while producing **0.0% false alarms on clean controls**, achieving a perfect test F1 score of **1.000**.
3. **Low Detection Latency:** On the complete benchmark, LSTM-DQN detects the onset within an average delay of only **0.25 checkpoints (1,250 environment steps)** after true onset.

---

## 8. Publication-Quality Plots

All figures are generated directly from experimental outputs and saved in `results/detector_plots/`:

1. **`01_detector_training_dynamics.png`:**
   - 4-panel trajectory showing episode returns, Huber Bellman loss convergence, $\epsilon$-greedy decay, and validation F1 score progression.
2. **`02_detector_comparison_metrics.png`:**
   - Comparative bar chart showing Precision, Recall, F1, and Detection Rate across LSTM-DQN, $O_1$, $O_2$, and combined $O_1\&O_2$.
3. **`03_detection_delay_distribution.png`:**
   - Onset detection latency in checkpoints across methods.
4. **`04_clean_control_false_alarm_analysis.png`:**
   - False alarm rate across negative control runs ($w=0.0$), demonstrating 0% false positives for LSTM-DQN.
5. **`05_true_vs_predicted_onset_trajectory.png`:**
   - Time-series return curves across held-out test runs, illustrating the exact environment steps where the LSTM-DQN flags onset versus ground-truth $t^*$.

---

## 9. Automated Test Suite (38 / 38 Passing)

The complete unit test suite validates both baseline mechanics and the new detector framework:

* `tests/test_detector.py` (11 tests):
  - `test_lstm_input_output_shapes`: Confirms tensor shapes $(B, W, D) \to (B, 2)$.
  - `test_dqn_action_selection`: Confirms deterministic vs exploratory action choices.
  - `test_replay_buffer_behavior`: Validates buffer capacity, transition storage, and batch sampling.
  - `test_bellman_target_computation`: Validates target calculation for terminal and non-terminal transitions.
  - `test_correct_reward_early_detection`: Confirms positive reward with delay decay.
  - `test_false_positive_penalty`: Confirms $-10.0$ penalty for premature flags and clean runs.
  - `test_missed_detection_penalty`: Confirms $-15.0$ penalty for unflagged hacking runs.
  - `test_clean_run_behavior`: Confirms patience is rewarded on clean controls.
  - `test_no_ground_truth_leakage_into_observations`: Confirms complete isolation of true rewards.
  - `test_chronological_sequence_construction`: Confirms rolling window padding and temporal order.
  - `test_run_level_train_test_separation`: Confirms disjoint run ID sets.
* Plus 27 existing tests across `rhd/envs/`, `rhd/features/`, `rhd/generators/`, and `rhd/evaluation/`.
* **Total: 38 passed in ~14 seconds (100% pass rate).**

---

## 10. Reproduction Commands

To replicate all experimental results from scratch:

```bash
# 1. Run all unit tests
python -m pytest

# 2. Run the unified detector pipeline (dataset prep -> training -> eval -> plots)
python run_all.py --detector

# 3. Or run modular scripts individually:
python scripts/generate_detector_dataset.py --workers 4
python scripts/train_detector.py --num-episodes 350
python scripts/evaluate_detector.py --split test
python scripts/evaluate_detector.py --split all
python scripts/plot_detector_results.py
```
