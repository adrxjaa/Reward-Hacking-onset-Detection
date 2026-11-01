# Controlled 10-Feature Ablation Study Report

## 1. Research Objective and Question

**Research Objective:**
To rigorously quantify the individual and collective contributions of distinct functional sub-groups within the original 10-dimensional behavioral feature vector to reward-hacking onset detection.

**Research Questions:**
1. *Temporal Dynamics Contribution:* Do inter-checkpoint temporal trends and policy/visitation divergence signals (`proxy_return_trend`, `action_distribution_change`, `state_visitation_change`) provide crucial discriminative power beyond static snapshot statistics?
2. *Revisitation & Anomaly Contribution:* How critical are rollout path efficiency, state revisitation rates, and behavioral entropy dispersion (`path_efficiency`, `revisit_rate`, `behavioral_dispersion`) in distinguishing genuine task execution from reward-hacking exploit cycles?
3. *Architecture Resilience:* Does recurrent state tracking in LSTM-DQN compensate for the absence of explicit temporal trend features compared to feed-forward MLP-DQN?

## 2. Original 10 Feature Names and Functional Grouping

The 10 canonical features extracted at each PPO evaluation checkpoint are defined as follows:

| Index | Feature Name | Description & Mathematical Definition | Functional Group |
|:-----:|:-------------|:---------------------------------------|:-----------------|
| 0 | `proxy_return` | Raw cumulative proxy return across 10 evaluation rollout episodes | Performance Level |
| 1 | `proxy_return_smoothed` | Exponential moving average / rolling mean of proxy return over prior 5 checkpoints | Performance Level |
| 2 | `proxy_return_trend` | Inter-checkpoint backward difference $\Delta R_t = R_t - R_{t-1}$ | **Temporal Change / Trend** |
| 3 | `episode_length` | Mean episode length (steps to termination or truncation) | Rollout Geometry |
| 4 | `action_distribution_change` | Mean Total Variation Distance $D_{\text{TV}}(P_t, P_{t-1})$ over fixed probe grid states | **Temporal Change / Trend** |
| 5 | `state_visitation_entropy` | Shannon entropy $H(S)$ of the agent's 2D grid position visitation distribution | Spatial Coverage |
| 6 | `state_visitation_change` | Total Variation Distance $D_{\text{TV}}(S_t, S_{t-1})$ of grid visitation between checkpoints | **Temporal Change / Trend** |
| 7 | `path_efficiency` | Unique grid cells visited divided by total steps ($|V| / L$) | **Revisitation / Anomaly** |
| 8 | `revisit_rate` | Fraction of steps spent re-visiting already visited grid cells ($(L - |V|) / L$) | **Revisitation / Anomaly** |
| 9 | `behavioral_dispersion` | Mean action distribution Shannon entropy over fixed probe states | **Revisitation / Anomaly** |

### Group Justifications:
- **Temporal Trend / Change Group (Indices 2, 4, 6):** These three features explicitly evaluate backward temporal shifts between consecutive checkpoints ($t$ vs $t-1$). When ablated, the detector receives no explicit step-to-step velocity signals.
- **Revisitation / Spatial Anomaly Group (Indices 7, 8, 9):** Path efficiency and revisit rate specifically diagnose the characteristic cyclic looping behavior of reward-hacking exploiters. Behavioral dispersion captures entropy collapse across the state space.

## 3. Experimental Configurations

Three controlled configurations were implemented and evaluated:
1. **Configuration A (BASELINE):** Retains all 10 original behavioral features (`input_dim=10`).
2. **Configuration B (TEMPORAL FEATURE ABLATION):** Ablates `proxy_return_trend`, `action_distribution_change`, and `state_visitation_change`. Retains 7 features (`input_dim=7`).
3. **Configuration C (REVISIT / ANOMALY FEATURE ABLATION):** Ablates `path_efficiency`, `revisit_rate`, and `behavioral_dispersion`. Retains 7 features (`input_dim=7`).

## 4. Controlled Variables & Methodology

- **Datasets:** Identical PPO generator environments and datasets across all 4 splits:
  - Train: 6 runs (2 clean $w=0.0$, 2 standard exploit $w=1.0$, 2 severe exploit $w=5.0$)
  - Validation: 3 runs (1 per condition)
  - Unseen Phase 2: 18 runs (6 clean, 6 standard exploit, 6 severe exploit; seeds 101-106)
  - Final Test: 18 runs (6 clean, 6 standard exploit, 6 severe exploit; seeds 701-706)
- **Random Seeds:** 5 independent training runs per configuration and detector: Seeds `[0, 1, 2, 3, 4]`.
- **Model Architectures:** Unmodified canonical architectures matching `input_dim`:
  - `LSTMQNetwork(input_dim=d, hidden_dim=64, num_layers=1, num_actions=2)`
  - `MLPQNetwork(input_dim=d, hidden_dim=64, num_actions=2)`
- **Training Protocol:** 350 episodes, Adam optimizer (lr=1e-3), Huber Smooth L1 loss, target update freq=20, epsilon decay 1.0 -> 0.05, warmup=64 transitions.
- **Zero Leakage:** Normalizer statistics $(\mu, \sigma)$ fitted exclusively on the 6 training runs. Validation and test sets strictly normalized using train parameters.
- **Checkpoint Selection:** Earliest best model selected strictly on validation split F1-score.

## 5. Results and Comparisons

### 5.1 Final Test Performance (Mean +/- Std Across 5 Seeds)

| Configuration | Detector | Input Dim | Final Test F1 | Precision | Recall | Clean FPR | Delay (ckpts) | Regret / Run |
|:--------------|:---------|:---------:|:-------------:|:---------:|:------:|:---------:|:-------------:|:------------:|
| Baseline (All 10 Features) | LSTM-DQN | 10 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.717 +/- 0.292 | 1.274 +/- 0.518 |
| Baseline (All 10 Features) | MLP-DQN | 10 | 0.936 +/- 0.049 | 1.000 +/- 0.000 | 0.883 +/- 0.085 | 0.000 +/- 0.000 | 0.737 +/- 0.295 | 3.770 +/- 2.147 |
| Temporal Feature Ablation | LSTM-DQN | 7 | 0.967 +/- 0.067 | 0.967 +/- 0.067 | 0.967 +/- 0.067 | 0.000 +/- 0.000 | 0.807 +/- 0.621 | 2.119 +/- 1.323 |
| Temporal Feature Ablation | MLP-DQN | 7 | 0.900 +/- 0.082 | 0.900 +/- 0.082 | 0.900 +/- 0.082 | 0.000 +/- 0.000 | 0.510 +/- 0.490 | 2.496 +/- 1.339 |
| Revisit / Anomaly Feature Ablation | LSTM-DQN | 7 | 0.933 +/- 0.133 | 0.933 +/- 0.133 | 0.933 +/- 0.133 | 0.000 +/- 0.000 | 0.383 +/- 0.201 | 1.793 +/- 1.885 |
| Revisit / Anomaly Feature Ablation | MLP-DQN | 7 | 0.842 +/- 0.083 | 0.890 +/- 0.090 | 0.800 +/- 0.085 | 0.000 +/- 0.000 | 0.753 +/- 0.252 | 5.030 +/- 1.377 |

### 5.2 Cross-Partition Performance Comparison

| Configuration | Detector | Train F1 | Validation F1 | Unseen Phase 2 F1 | Final Test F1 |
|:--------------|:---------|:--------:|:-------------:|:-----------------:|:-------------:|
| Baseline (All 10 Features) | LSTM-DQN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.983 +/- 0.033 | 1.000 +/- 0.000 |
| Baseline (All 10 Features) | MLP-DQN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.983 +/- 0.033 | 0.936 +/- 0.049 |
| Temporal Feature Ablation | LSTM-DQN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.967 +/- 0.067 | 0.967 +/- 0.067 |
| Temporal Feature Ablation | MLP-DQN | 0.871 +/- 0.112 | 1.000 +/- 0.000 | 0.950 +/- 0.041 | 0.900 +/- 0.082 |
| Revisit / Anomaly Feature Ablation | LSTM-DQN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.900 +/- 0.122 | 0.933 +/- 0.133 |
| Revisit / Anomaly Feature Ablation | MLP-DQN | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.882 +/- 0.089 | 0.842 +/- 0.083 |

## 6. Interpretation and Findings

### Baseline vs. Temporal Feature Ablation:
- Removing the 3 explicit inter-checkpoint temporal difference features (`proxy_return_trend`, `action_distribution_change`, `state_visitation_change`) tests whether detector performance degrades without explicit differential inputs.
- For **LSTM-DQN**, the recurrent memory cell processes sequences over the observation window ($W=4$), allowing it to implicitly infer inter-step trends from sequential levels (`proxy_return`, `proxy_return_smoothed`, etc.).
- For **MLP-DQN**, the absence of recurrent memory makes the detector rely on differences across the flattened observation window; ablated performance reveals the degree to which feed-forward networks depend on pre-computed differential features.

### Baseline vs. Revisit / Anomaly Feature Ablation:
- Removing trajectory revisitation and behavioral dispersion features (`path_efficiency`, `revisit_rate`, `behavioral_dispersion`) strips the explicit spatial cycling signals.
- The experimental results demonstrate whether proxy reward inflation alone or spatial cycling is the primary driver for early onset detection.

### LSTM-DQN vs. MLP-DQN:
- Comparing both architectures under identical ablation conditions verifies whether temporal memory confers systematic robustness to missing feature groups.

## 7. Limitations and Conclusion

- **Limitations:** The study was conducted on discrete 2D grid navigation environments with periodic checkpoint logging. Continuous control domains or environments with stochastic dynamics may exhibit different feature group dependencies.
- **Conclusion:** The ablation study successfully identifies the functional contributions of temporal trends versus spatial anomaly metrics under strict experimental controls with zero data leakage.