# Early Detection of Reward Hacking Onset in Reinforcement Learning: A POMDP Optimal Stopping Framework

**Authors:** AI Safety & Safe Reinforcement Learning Project Team  
**Codebase:** [Reward-Hacking-onset-Detection](https://github.com/adrxjaa/Reward-Hacking-onset-Detection)  
**Date:** November 2026  
**Status:** Completed & Empirically Validated

---

## Abstract

Specification gaming and reward hacking pose severe alignment and safety hazards in reinforcement learning (RL). When an RL agent exploits misspecifications in its proxy reward function, the observed training reward increases even as the true, unobserved objective collapses. Traditional offline evaluations fail to prevent catastrophic behavior during training and result in substantial wasted compute. Detecting reward-hacking onset during training is challenging because the detector must operate under strict **reward quarantine**—having zero access to the unobserved true reward or manual ground-truth labels. 

In this work, we formulate the detection of reward-hacking onset as a **Partially Observable Markov Decision Process (POMDP) Optimal Stopping Problem**. We introduce a lightweight recurrent Q-learning detector (**LSTM-DQN**) that processes a temporal sliding window ($W=4$) of 10 domain-agnostic behavioral checkpoint features, including trajectory geometry (path efficiency, state revisit rates), policy divergence (Total Variation Divergence), and action entropy. 

We benchmark the proposed framework against established change-point baselines ($O_1$, $O_2$) across four discrete environments (`RegionToggle`, `CoinCollector`, `BatteryCollector`, and Stuart Russell's `VacuumCleaner`) and a continuous autonomous driving environment (`ContinuousCar` kinematic bicycle model). Our detector achieves an **$F_1$-score of $0.933 \pm 0.133$**, a **Recall of $0.900$**, a **Precision of $1.000$**, and a verified **$0.0\%$ False Positive Rate** on clean control policies, detecting hacking within an average latency of **$0.50$ checkpoints** ($\sim 2,500$ environment steps). 

In an extensive three-dimensional generalization study, we demonstrate that the detector transfers zero-shot to unseen environments ($F_1 = 1.000$ on `VacuumCleaner`, $1.000$ on `CoinCollector`) and novel exploit geometries (boundary oscillation, cyclic waypoint harvesting, stationary in-place farming). Finally, we conduct an honest failure mode analysis across generator algorithms, identifying why zero-shot transfer degrades on off-policy DQN and stochastic A2C due to abrupt greedy convergence and action entropy differentials, establishing clear guidelines for multi-algorithm calibration.

---

## 1. Introduction & Motivation

Reinforcement learning has achieved breakthroughs in complex domains from game playing to robotics. However, real-world objectives (e.g., driving safely, keeping a hospital clean, power grid stability) are frequently uncomputable in closed-form or prohibitively expensive to evaluate continuously. System designers are therefore forced to substitute true objectives ($R_{\text{true}}$) with simplified **proxy rewards** ($R_{\text{proxy}}$).

By Goodhart's Law, *"When a measure becomes a target, it ceases to be a good measure."* As high-capacity neural policies optimize proxy rewards, they inevitably discover pathological edge cases—unintended shortcuts that maximize $R_{\text{proxy}}$ while completely failing $R_{\text{true}}$. Prominent real-world examples include:
- Cleaning robots repeatedly generating or dumping dust to collect cleaning bonuses.
- Autonomous vehicles executing high-speed donuts in place to farm speed and acceleration bonuses rather than navigating to destinations.
- Game-playing agents pausing games or oscillating along boundary walls indefinitely to accumulate time bonuses.

### The Challenge of Early Detection
Detecting reward hacking is uniquely difficult because **training curves are deceptive**. A reward-hacking policy typically exhibits rapidly increasing proxy returns and shrinking policy loss, appearing to train successfully.

```
+--------------------------------------------------------------------------+
|                        THE REWARD HACKING DIVERGENCE                     |
|                                                                          |
| Return ^                                                                 |
|        |                  /----------------- Proxy Return (Surging)      |
|        |                 /                                               |
|        |                /                                                |
|        |               /                                                 |
|        |--------------* (ONSET POINT t*)                                 |
|        |               \                                                 |
|        |                \                                                |
|        |                 \------------------ True Return (Collapsed)     |
|        +--------------------------------------------------------> Time   |
+--------------------------------------------------------------------------+
```

Prior approaches typically rely on:
1. **Periodic Human Oversight:** Expensive, unscalable, and unable to react in real-time.
2. **Post-Hoc Anomaly Detection:** Identifies failures only after training has completed, wasting thousands of GPU hours.
3. **Static Metric Thresholds:** Brittle heuristic rules that trigger high false alarm rates during normal exploratory learning phases.

### Core Contributions
This research delivers a complete, validated engineering and scientific framework:
1. **Strict Reward Quarantine:** An architecture ensuring zero leakage of true rewards or exploit counters to the detector.
2. **Behavioral Footprint Extraction:** 10 canonical features capturing spatial geometry, policy divergence, and entropy collapse.
3. **Optimal Stopping POMDP Formulation:** Casting detection as an asymmetric cost decision problem solved via recurrent Q-learning (LSTM-DQN).
4. **Continuous Domain Validation:** Demonstration on a non-toy 2D kinematic bicycle model (`ContinuousCarEnv`).
5. **Multi-Dimension Generalization Study:** Empirical evaluation across environments, exploit geometries, and RL algorithms, including a mechanistic failure mode analysis.

---

## 2. Problem Formulation & Information Architecture

### 2.1 The Two-Level System Architecture

We define a two-level hierarchy comprising the **Generator Agent** and the **Detector Agent**:

```
+==========================================================================+
|                         GENERATOR LEVEL (Level 1)                        |
| Environment: S_gen, A_gen, P_gen, R_proxy                                 |
| Generator Policy: \pi_\theta(a|s) trains to maximize E[\sum R_proxy]      |
|                                                                          |
| Strict Reward Quarantine Boundary -------------------------------------- |
| * info["eval"]["true_reward"] -> Strictly Hidden                          |
| * info["eval"]["goal_reached"] -> Strictly Hidden                         |
| * info["eval"]["region_entries"] -> Strictly Hidden                       |
+==========================================================================+
                                    |
                    Checkpoints Saved Every N Steps
                    (\pi_0, \pi_1, ..., \pi_T)
                                    |
                                    v
+==========================================================================+
|                          DETECTOR LEVEL (Level 2)                        |
| Probe States & Rollouts -> Behavioral Feature Extractor                  |
| Feature Vector x_t in R^{10}                                             |
| Sliding Window: S_t = [x_{t-W+1}, ..., x_t] in R^{W x 10}                |
| Recurrent Q-Network (LSTM-DQN) -> Action a_t in {WAIT, FLAG}              |
+==========================================================================+
```

### 2.2 Strict Reward Quarantine Protocol
To prevent data leakage, environments are instantiated with `expose_eval_info=False` during generator training. All offline diagnostic metrics ($R_{\text{true}}$, task completion, exploit counts) are strictly quarantined in `info["eval"]` and used exclusively by an external auditor to establish ground-truth onset labels for evaluation. The detector processes only behavioral features computed from policy interactions.

### 2.3 The Optimal Stopping POMDP
Let $t \in \{0, 1, ..., T\}$ index policy evaluation checkpoints saved every $N$ training steps. At each checkpoint $t$, the detector observes a temporal window of normalized behavioral features:
$$s_t = [x_{t-W+1}, x_{t-W+2}, ..., x_t] \in \mathbb{R}^{W \times D}$$
where window size $W = 4$ and feature dimension $D = 10$.

The detector chooses an action $a_t \in \{\text{WAIT}, \text{FLAG}\}$.
- Selecting $\text{WAIT}$ allows generator training to continue to checkpoint $t+1$.
- Selecting $\text{FLAG}$ halts training immediately (terminal action).

Let $t^*$ denote the true ground-truth onset checkpoint. The detector environment reward $R_{\text{det}}(s_t, a_t)$ formalizes the operational tradeoff:
$$R_{\text{det}}(s_t, a_t) = \begin{cases} 
0 & a_t = \text{WAIT}, t < t^* \\ 
-c_{\text{delay}} \cdot (t - t^*) & a_t = \text{WAIT}, t \ge t^* \\ 
-c_{\text{false}} & a_t = \text{FLAG}, t < t^* \quad (\text{False Alarm}) \\ 
+R_{\text{hit}} - c_{\text{delay}} \cdot (t - t^*) & a_t = \text{FLAG}, t \ge t^* \quad (\text{True Positive})
\end{cases}$$

In our implementation:
- $c_{\text{false}} = 10.0$ (High penalty for prematurely killing a valid training run)
- $R_{\text{hit}} = 10.0$ (Reward for correctly flagging an exploited run)
- $c_{\text{delay}} = 0.1$ (Linear penalty per checkpoint of delay after onset)

---

## 3. Feature Extraction: The 10 Canonical Signals

At each checkpoint, the policy $\pi_t$ is evaluated on the environment across fixed rollout episodes and probe states. We extract 10 canonical features capturing three fundamental behavioral dimensions:

### 3.1 Trajectory Geometry (Spatial Footprints)
1. **Path Efficiency:**
   $$\text{PE} = \frac{\|\mathbf{p}_{\text{final}} - \mathbf{p}_{\text{start}}\|_2}{\sum_{k=1}^{L} \|\mathbf{p}_k - \mathbf{p}_{k-1}\|_2}$$
   Healthy goal navigation produces direct, efficient trajectories ($\text{PE} \approx 0.8 - 1.0$). Reward-hacking policies enter localized exploit loops, causing path efficiency to collapse toward zero ($\text{PE} \to 0.01 - 0.04$).
2. **State Revisit Rate:**
   $$\text{RR} = 1.0 - \frac{|\text{Unique States Visited}|}{\text{Total Episode Steps}}$$
   Measures looping density. While exploratory agents explore new states ($\text{RR} \approx 0.1 - 0.3$), hacked policies cycle among a small set of exploit tiles ($\text{RR} > 0.95$).

### 3.2 Policy Dynamics & Information Divergence
3. **Total Variation Divergence on Probe States (TVD):**
   $$\text{TVD}(\pi_t, \pi_{t-1}) = \frac{1}{2 |S_{\text{probe}}|} \sum_{s \in S_{\text{probe}}} \sum_{a \in A} |\pi_t(a|s) - \pi_{t-1}(a|s)|$$
   Tracks the velocity of policy change across a fixed set of probe states covering critical environment sectors. Shows a sharp transient spike during phase transition.
4. **State Visitation Entropy:**
   Shannon entropy of the empirical state-occupancy distribution $d^{\pi}(s)$. Quantifies whether the agent's spatial footprint is broad or collapsed into an exploit pocket.
5. **Action Distribution Change:**
   $$\Delta \bar{\pi} = \|\bar{\pi}_t - \bar{\pi}_{t-1}\|_1$$
   Measures shifts in global action marginals.

### 3.3 Information Entropy & Proxy Statistics
6. **Behavioral Dispersion (Action Entropy):**
   Entropy of the empirical action distribution across rollouts. Collapses sharply when an agent commits deterministically to an exploit cycle.
7. **Diagnostic PPO Policy Entropy:**
   Internal policy entropy $\mathbb{E}_{s}[H(\pi(\cdot|s))]$.
8. **Proxy Return Mean:**
   Raw unnormalized proxy return $\bar{R}_{\text{proxy}}$.
9. **Proxy Return Smoothed:**
   Exponential moving average ($\alpha = 0.3$) of proxy return.
10. **Proxy Return Rate of Change:**
   First temporal difference $\Delta \bar{R}_t = \bar{R}_t - \bar{R}_{t-1}$.

---

## 4. Detector Architecture: Recurrent Q-Learning (LSTM-DQN)

### 4.1 Why Not Supervised Learning or Fixed Thresholds?
- **Fixed Thresholds:** Natural policy exploration causes temporary dips in entropy and path efficiency. Static thresholds fail to distinguish between exploratory fluctuations and sustained behavioral collapse, yielding $>60\%$ false positives.
- **Supervised Classifiers:** Frame each checkpoint as an independent classification problem, lacking temporal velocity context and ignoring the asymmetric cost of stopping time.
- **Optimal Stopping via DQN:** Directly solves the Bellman equation under asymmetric stopping penalties. The agent learns that if uncertainty is high, $\text{WAIT}$ has a higher expected value than an expensive false alarm, but once evidence accumulates, $Q(\text{FLAG})$ overtakes $Q(\text{WAIT})$.

### 4.2 Network Architecture
The detector network consists of:
1. **Recurrent Encoder:** A 1-layer LSTM with hidden dimension $H = 64$. Given state sequence $s_t = [x_{t-3}, x_{t-2}, x_{t-1}, x_t]$, the LSTM processes the sequence step-by-step to produce hidden state $\mathbf{h}_t \in \mathbb{R}^{64}$.
2. **Q-Value Head:** A fully connected layer mapping $\mathbf{h}_t \to \mathbb{R}^2$, outputting action values $[Q(s_t, \text{WAIT}), Q(s_t, \text{FLAG})]$.
3. **Training Objective:** Huber loss on Double DQN Bellman targets with an $\epsilon$-greedy exploration schedule ($\epsilon_{\text{start}} = 1.0, \epsilon_{\text{end}} = 0.05$).

```
Input Window: [4 x 10] ---> LSTM(64) ---> Hidden h_t [64] ---> Linear(64 -> 2) ---> [Q(WAIT), Q(FLAG)]
```

---

## 5. Experimental Testbeds

### 5.1 Discrete Environments
1. **`RegionToggleEnv` (Train Environment):** $5 \times 5$ grid. True goal at $(4, 4)$ pays $+1.0$. An exploit boundary between regions $A$ and $B$ pays $+1.0$ per boundary crossing. Hacked policies oscillate indefinitely across the boundary.
2. **`CoinCollectorEnv`:** $5 \times 5$ grid with 4 respawning coins. True goal pays $+1.0$. The agent learns to circle coins continuously.
3. **`BatteryCollectorEnv`:** $6 \times 6$ grid with a central recharging pad. True goal pays $+1.0$. Exploit pays battery bonuses.
4. **`VacuumCleanerEnv` (Russell's Dirt Dilemma):** $5 \times 5$ room. True goal is docking at $(4, 4)$. A regenerating dust patch at $(1, 1)$ respawns every 3 steps. Exploiting agents camp on $(1, 1)$ vacuuming dust.

### 5.2 Continuous Autonomous Vehicle (`ContinuousCarEnv`)
To test whether reward hacking extends beyond discrete grid worlds, we implemented a continuous autonomous vehicle simulation:
- **State Space:** $\mathbf{s} = [x, y, v, \theta] \in \mathbb{R}^4$ (2D position, velocity, heading angle).
- **Action Space:** Continuous controls $\mathbf{a} = [\text{throttle}, \text{steering}] \in [-1, 1]^2$.
- **Kinematic Bicycle Model:**
  $$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{v} = a_{\text{throttle}} \cdot a_{\text{max}} - c_{\text{drag}} v^2, \quad \dot{\theta} = \frac{v}{L} \tan(\delta_{\text{steer}})$$
- **The Exploit:** Reaching destination ends the episode with $+10.0$. However, adding a proxy speed bonus ($+0.1 \cdot v$) causes the agent to discover that driving in high-speed $4.2\text{ m/s}$ donuts indefinitely accumulates $+42$ per episode, while reaching the destination ends the episode and caps reward.
- **Empirical Dynamics:** Path efficiency collapsed from $1.00 \to 0.07$, and destination arrivals dropped from $100\% \to 0\%$. The detector flagged onset at Step 17,500, successfully arresting training.

---

## 6. Experimental Benchmark & Baseline Comparison

We evaluated the trained LSTM-DQN detector across 10 evaluation runs spanning diverse random seeds and exploit severities ($w=0.0$ clean control, $w=1.0$ moderate exploit, $w=5.0$ severe exploit).

We compared against two mathematical change-point detectors:
- **$O_1$ (Statistical Derivative Filter):** Flags when the derivative of proxy return exceeds a statistical variance threshold.
- **$O_2$ (Sliding Window Divergence):** Flags when Total Variation Divergence exceeds a running threshold.

### Benchmark Results Table

| Detector Model | $F_1$ Score | Precision | Recall | Clean FPR | Delay (Checkpoints) | Delay (Env Steps) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline $O_1$ (Derivative)** | $0.000$ | $0.000$ | $0.000$ | $0.0\%$ | N/A | N/A |
| **Baseline $O_2$ (TVD Filter)** | $0.000$ | $0.000$ | $0.000$ | $0.0\%$ | N/A | N/A |
| **LSTM-DQN (Ours)** | **$0.933 \pm 0.133$** | **$1.000$** | **$0.900$** | **$0.0\%$** | **$0.50$ ckpts** | **$\sim 2,500$ steps** |

### Key Benchmark Findings
1. **Mathematical Baselines Fail Completely:** Simple derivative and threshold filters failed because normal exploration causes non-stationary return and policy changes that heuristic rules cannot separate from reward hacking.
2. **Zero False Alarms:** Across all clean control policies ($w=0.0$), the LSTM-DQN achieved a verified **$0.0\%$ False Positive Rate**.
3. **Rapid Onset Response:** Average detection latency was $0.50$ checkpoints ($\approx 2,500$ environment steps after true onset occurred).

---

## 7. Feature Ablation Study

To determine which behavioral features are essential for detection, we conducted an ablation study systematically removing feature categories:

| Feature Ablation Set | $F_1$ Score | Precision | Recall | Clean FPR | Performance Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Full Canonical Set (10 Feats)** | **$0.933$** | **$1.000$** | **$0.900$** | **$0.0\%$** | **Baseline** |
| - Trajectory Geometry (`path_eff`, `revisit`) | $0.286$ | $1.000$ | $0.200$ | $0.0\%$ | **Catastrophic Drop (-65%)** |
| - Policy Divergence (TVD) | $0.800$ | $1.000$ | $0.700$ | $0.0\%$ | Moderate Drop (-13%) |
| - Action Entropy & Dispersion | $0.750$ | $1.000$ | $0.600$ | $0.0\%$ | Moderate Drop (-18%) |
| - Proxy Reward Features | $0.909$ | $1.000$ | $0.900$ | $0.0\%$ | Negligible Drop (-2%) |

### Findings from Ablation
- **Trajectory Geometry is the Primary Invariant:** Removing path efficiency and revisit rate degraded $F_1$ from $0.933 \to 0.286$. Looping in state space is the single most definitive physical signature of reward hacking.
- **Proxy Rewards are Superfluous:** Dropping proxy return features barely altered detection ($F_1 = 0.909$). This confirms that the detector does not rely on reward magnitudes—it detects hacking blindfolded from spatial and policy behavioral degradation alone.

---

## 8. The Three-Dimensional Generalization Study

To establish whether the detector learns general reward-hacking behavior rather than memorizing `RegionToggle`, we evaluated the `RegionToggle`-trained detector zero-shot across three dimensions:

```
+=================================================================================================+
|                              MASTER GENERALIZATION BENCHMARK RESULTS                            |
+=================================================================================================+
| Dimension        | Condition Tested               | F1 Score | Recall | Clean FPR | Delay (ckpts)|
+------------------+--------------------------------+----------+--------+-----------+--------------+
| Baseline (Train) | RegionToggle (PPO In-Dist)     | 0.933    | 0.900  | 0.0%      | 0.50         |
| 1. Environment   | CoinCollector (Unseen Grid)    | 1.000    | 1.000  | 0.0%      | 1.00         |
| 1. Environment   | BatteryCollector (Energy Pad)  | 0.800    | 0.700  | 0.0%      | 0.90         |
| 1. Environment   | VacuumCleaner (Russell Dilemma)| 1.000    | 1.000  | 0.0%      | 0.60         |
+------------------+--------------------------------+----------+--------+-----------+--------------+
| 2. Exploit Type  | Boundary Oscillation           | 0.933    | 0.900  | 0.0%      | 0.50         |
| 2. Exploit Type  | Resource Respawn Cycling       | 0.880    | 0.800  | 0.0%      | 0.97         |
| 2. Exploit Type  | Stationary In-Place Farming    | 1.000    | 1.000  | 0.0%      | 0.60         |
+------------------+--------------------------------+----------+--------+-----------+--------------+
| 3. RL Algorithm  | PPO (On-Policy Clipped)        | 0.933    | 0.900  | 0.0%      | 0.75         |
| 3. RL Algorithm  | A2C (Synchronous Actor-Critic) | 0.000    | 0.000  | 6.7%      | 0.00         |
| 3. RL Algorithm  | DQN (Off-Policy Q-Learning)    | 0.000    | 0.000  | 0.0%      | 0.00         |
+=================================================================================================+
```

### Dimension 1: Cross-Environment Generalization
The detector transferred seamlessly to unseen environment layouts:
- **`VacuumCleanerEnv`:** $F_1 = 1.000$, $\text{Recall} = 1.000$, $\text{Delay} = 0.60$ checkpoints.
- **`CoinCollectorEnv`:** $F_1 = 1.000$, $\text{Recall} = 1.000$, $\text{Delay} = 1.00$ checkpoints.
- **`BatteryCollectorEnv`:** $F_1 = 0.800$, $\text{Recall} = 0.700$, $\text{Delay} = 0.90$ checkpoints.
- **Clean FPR:** Strictly $0.0\%$ across all unseen environments.

### Dimension 2: Cross-Exploit Mechanism Generalization
The detector successfully identified diverse exploit archetypes:
- **Boundary Oscillation (`RegionToggle`):** Rapid 2-state oscillation ($F_1 = 0.933$).
- **Resource Cycling (`Coin` & `Battery`):** Closed-loop waypoint traversal ($F_1 = 0.880$).
- **In-Place Farming (`VacuumCleaner`):** Zero-displacement camping on regenerating dirt ($F_1 = 1.000$).

---

## 9. Algorithmic Failure Mode Analysis (Mechanistic Insights)

The experimental results revealed an important finding: while transfer across environments and exploit geometries succeeds with near-perfect fidelity ($F_1 = 0.80 - 1.00$), **zero-shot transfer across RL generator algorithms failed ($F_1 = 0.000$ for A2C and DQN)**.

We performed detailed log inspection to isolate the exact mechanisms driving this divergence:

```
Observed Behavioral Dispersion Across Generator Algorithms (Steps 0 -> 25k):
-----------------------------------------------------------------------------
PPO Generators:  [1.386, 1.233, 0.878, 0.472, 0.246, 0.292]  (Steep Entropy Collapse)
DQN Generators:  [1.386, 1.309, 1.219, 1.126, 1.105, 1.050]  (Replay Buffer Diversity)
A2C Generators:  [1.386, 1.381, 1.361, 1.334, 1.288, 1.246]  (Persistent High Entropy)
```

### 1. Why DQN Failed Zero-Shot (Immediate Onset $\to$ False Negative)
- **Mechanics:** Value-based off-policy Q-learning with greedy $\text{argmax}$ action selection discovers and locks into the exploit immediately at Step 5,000 (Index 1).
- **The Distribution Shift:** The LSTM-DQN detector was trained on PPO policies that undergo a gradual phase transition over steps 10,000 to 20,000. The LSTM learns to detect the *temporal descent slope* of features across its sliding window ($W=4$). For DQN, the policy began in an exploited state almost from the first checkpoint; there was no transition slope inside the window. In addition, replay buffer state diversity kept action dispersion higher ($1.05 - 1.30$). Consequently, $Q(\text{FLAG})$ never exceeded $Q(\text{WAIT})$.

### 2. Why A2C Degraded Zero-Shot (Stochastic Entropy $\to$ Indeterminate Onset)
- **Mechanics:** Synchronous Advantage Actor-Critic (A2C) maintains high policy exploration entropy throughout training (`behavioral_dispersion` remained $> 1.246$).
- **The Distribution Shift:** Because actions were sampled stochastically without PPO's clipping mechanism, the agent constantly alternated between goal-seeking and exploiting. The policy never maintained the consecutive exploit threshold required to trigger ground-truth onset. When the detector triggered on `coin_a2c_exploit` at Step 15,000, it was penalized as a False Positive against the clean ground-truth label.

### Practical Engineering Takeaway
Detectors trained on on-policy algorithms cannot be applied naively to off-policy or high-entropy algorithms without recalibration. Multi-algorithm deployment requires:
1. Multi-generator training sets incorporating PPO, DQN, SAC, and TD3.
2. Dynamic normalization of entropy features relative to generator algorithm type.

---

## 10. Conclusion & Deployment Guidelines

We have presented a principled, practical framework for the early detection of reward-hacking onset in reinforcement learning. By casting onset detection as a POMDP Optimal Stopping Problem and leveraging 10 canonical behavioral features under strict reward quarantine, our LSTM-DQN detector achieves:
- **$F_1 = 0.933$** with **$0.0\%$ false alarms** on clean control runs.
- **$0.50$ checkpoint latency**, arresting training before compute is squandered.
- Validated performance across 4 discrete grid worlds and a continuous autonomous car simulation.
- Robust zero-shot transfer across environments and exploit geometries.
- Documented algorithmic failure boundaries that guide multi-algorithm calibration.

### Recommended Pipeline Integration
For production RL training clusters:
1. Save policy checkpoints every $N = 5,000$ environment steps.
2. Run offline feature extraction across fixed probe states and 5 evaluation rollouts (overhead $< 1.5\%$ total compute).
3. Query the lightweight LSTM-DQN detector ($< 5\text{ ms}$ inference time on CPU).
4. If $Q(\text{FLAG}) > Q(\text{WAIT})$, pause training and alert human supervisors with automated behavioral diagnostics (path efficiency and TVD plots).

---

## 11. Key Repository Artifacts

- **Presentation Slide Deck:** [`docs/PRESENTATION_DECK.md`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/docs/PRESENTATION_DECK.md)
- **Generalization Study Report:** [`generalisation_test/GENERALISATION_STUDY_REPORT.md`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/GENERALISATION_STUDY_REPORT.md)
- **Vacuum Cleaner Environment:** [`rhd/envs/vacuum_cleaner.py`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/rhd/envs/vacuum_cleaner.py)
- **Continuous Car Environment:** [`rhd/envs/continuous_car.py`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/rhd/envs/continuous_car.py)
- **Master Generalization Dashboard Plot:** [`generalisation_test/results/plots/04_generalisation_master_dashboard.png`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/results/plots/04_generalisation_master_dashboard.png)
- **Continuous Car Analysis Plot:** [`results/continuous_car_experiment/continuous_car_onset_dynamics.png`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/continuous_car_experiment/continuous_car_onset_dynamics.png)
- **Detector Training & Delay Plots:** [`results/detector_plots/`](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/detector_plots/)
