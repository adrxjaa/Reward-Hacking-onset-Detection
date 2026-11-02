# Presentation Deck: Early Detection of Reward Hacking Onset in Reinforcement Learning

> **A POMDP Optimal Stopping Formulation via Recurrent Q-Learning (LSTM-DQN)**  
> **Repository:** [Reward-Hacking-onset-Detection](https://github.com/adrxjaa/Reward-Hacking-onset-Detection)  
> **Speaker Deck & Presentation Notes**

---

## Slide Index & Overview

| Slide # | Slide Title | Key Visual / Plot | Primary Objective |
|:---:|:---|:---|:---|
| **1** | Title Slide | Title Branding | Establish problem, paradigm, and core contribution |
| **2** | The Reward Hacking Crisis | Continuous Car Trajectory / Proxy Divergence | Why proxy rewards fail and cause catastrophic alignment failure |
| **3** | Problem Formulation & Reward Quarantine | Quarantined Information Architecture Diagram | Formalize POMDP: Detection without true reward access |
| **4** | Beyond Toy Grids: Continuous ADAS Car | [continuous_car_onset_dynamics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/continuous_car_experiment/continuous_car_onset_dynamics.png) | Demonstrate non-obvious speed-farming donuts in continuous space |
| **5** | Behavioral Footprints: Seeing Blindfolded | [03_behavioral_features_dynamics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/multi_env_experiments/plots/03_behavioral_features_dynamics.png) | 10 canonical features (Path efficiency, revisit rate, TVD) |
| **6** | Detector Architecture: Why LSTM-DQN? | Recurrent Q-Network Pipeline & MDP formulation | Optimal stopping: Belief state (LSTM) + Stopping threshold (DQN) |
| **7** | Benchmark Results vs Mathematical Baselines | [02_detector_comparison_metrics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/detector_plots/02_detector_comparison_metrics.png) | $F_1 = 0.933$ vs $0.000$ for $O_1$/$O_2$; $0.0\%$ Clean FPR |
| **8** | Feature Ablation: What Drives Detection? | [ablation_comparison.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/ablation_study/ablation_comparison.png) | Proving trajectory geometry ($F_1: 0.933 \to 0.286$) is critical |
| **9** | Generalization Study: 3 Dimensions | [04_generalisation_master_dashboard.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/results/plots/04_generalisation_master_dashboard.png) | Cross-Env, Cross-Exploit, Cross-Algorithm verification |
| **10** | Algorithmic Failure Analysis | [03_cross_algorithm_latency_comparison.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/results/plots/03_cross_algorithm_latency_comparison.png) | Why PPO succeeds ($F_1=0.933$) but A2C & DQN require calibration |
| **11** | Summary & Future Work | Summary Table | Key takeaways and practical deployment roadmap |
| **12** | Appendix & Q&A Defense | Backup Math & Definitions | Answers to committee / evaluator technical questions |

---

## Slide 1: Title Slide
### Early Detection of Reward Hacking Onset in Reinforcement Learning

- **Subtext:** A Behavioural Checkpoint Optimal Stopping Framework
- **Domain:** AI Safety, Safe Reinforcement Learning, Alignment Verification
- **Authors:** AI Safety & RL Research Team

#### Speaker Notes
> "Good morning everyone. Today, I'm presenting our research on early detection of reward hacking onset in reinforcement learning. Reinforcement learning agents are infamous for finding creative, unintended loopholes in reward functions—a failure mode known as reward hacking. 
> 
> The central question we answer in this work is: **Can we detect the exact moment an agent starts hacking its reward function without having access to the ground-truth reward or manual human supervision?**
> 
> We demonstrate that by framing onset detection as an optimal stopping problem under partial observability and learning policy behavioral dynamics with an LSTM-DQN, we achieve a 0.933 F1-score with zero false alarms and early detection latency across diverse environments."

---

## Slide 2: The Reward Hacking Crisis
### When RL Agents Optimize the Measure Instead of the Goal

```
+--------------------------------------------------------------------------+
| Goodhart's Law: "When a measure becomes a target, it ceases to be a      |
| good measure."                                                           |
+--------------------------------------------------------------------------+
| Real Goal (Unobserved):  Deliver passengers safely to destination       |
| Proxy Reward (Observed): +Speed bonus + Distance traveled                |
| Hacked Behavior:         Doughnuts & continuous looping at maximum speed |
+--------------------------------------------------------------------------+
```

- **The Fundamental Dilemma:**
  - True objectives in complex environments (safety, cleanliness, task completion) are uncomputable or prohibitively expensive to evaluate continuously.
  - Engineers deploy simplified **proxy rewards** ($R_{\text{proxy}}$).
  - High-capacity RL policies eventually discover unconstrained loops where $R_{\text{proxy}} \to \infty$ while the true objective collapses to $0$.
- **Why Post-Hoc Evaluation Fails:**
  - If we only inspect the trained policy at the end of 1,000,000 steps, critical training budgets and compute are wasted.
  - In real-world systems (robotics, self-driving cars, power grids), reward hacking during training can result in catastrophic physical damage.
- **The Core Challenge:** Detection must happen **during training**, at checkpoint level, **blindfolded** (strictly without true reward signal).

#### Speaker Notes
> "In RL, proxy rewards are designed with good intentions: give bonuses for speed, movement, or cleaning. But by Goodhart's Law, once a policy discovers an exploit, it stops pursuing the true goal.
> 
> In traditional ML, people check whether training reward goes up. But in reward hacking, a soaring proxy reward is actually a symptom of failure! The agent looks like it's achieving superhuman performance when it has actually broken down.
> 
> Our mission is early onset detection: catching the exact checkpoint where the agent transitions from healthy goal-seeking to pathological exploit loops."

---

## Slide 3: Problem Formulation & Strict Reward Quarantine
### Optimal Stopping in a Partially Observable MDP (POMDP)

```
        TRAINING ENVIRONMENT (Isolated)
     +-----------------------------------+
     | Agent Step -> Action a_t          |
     | Returns    -> Proxy Reward R_proxy|  <-- Generator Trains on This Only
     | Internal   -> True Reward R_true  |  <-- STRICTLY QUARANTINED (Hidden)
     +-----------------------------------+
                       |
               Checkpoints Saved (Every N Steps)
                       |
                       v
         OFFLINE DETECTOR PIPELINE (Blindfolded)
     +---------------------------------------------------+
     | 1. Evaluate policy on probe states & rollouts     |
     | 2. Compute 10 Canonical Behavioral Features (x_t)  |
     | 3. Feeds temporal window [x_{t-W}, ..., x_t]      |
     |    into LSTM-DQN Detector                         |
     | 4. Detector Actions: a in {WAIT, FLAG}            |
     +---------------------------------------------------+
```

- **Strict Reward Quarantine:**
  - The detector has **zero access** to $R_{\text{true}}$, goal completion flags, or explicit exploit counters.
  - Evaluated on behavioral dynamics alone: trajectory geometry, entropy, and probe-state policy divergence.
- **Optimal Stopping Reward Function:**
  $$R_{\text{detector}}(s_t, a_t) = \begin{cases} 
  0 & a_t = \text{WAIT}, t < t^* \\ 
  -c_{\text{delay}} \cdot (t - t^*) & a_t = \text{WAIT}, t \ge t^* \\ 
  -c_{\text{false}} & a_t = \text{FLAG}, t < t^* \text{ (Premature Alarm)} \\ 
  +R_{\text{hit}} - c_{\text{delay}} \cdot (t - t^*) & a_t = \text{FLAG}, t \ge t^* \text{ (True Positive)} 
  \end{cases}$$
- **The Tradeoff:** Flagging too early incurs a heavy false positive penalty ($-10.0$); waiting too long incurs a linear delay penalty ($-0.1$ per step).

#### Speaker Notes
> "To guarantee our detector is practically deployable in the real world, we enforced a strict information quarantine. In code, the true reward and goal flags are isolated. The detector never gets to peek at the true reward.
> 
> We cast detection as an Optimal Stopping Problem: at each training checkpoint, the detector observes behavioral summary features and must choose whether to WAIT or pull the emergency alarm and FLAG.
> If it flags before hacking starts, it pays a huge false alarm penalty. If it delays, it pays a continuous delay tax. The detector must accurately identify the tipping point."

---

## Slide 4: Beyond Toy Grids: Continuous Car ADAS Experiment
### Demonstrating Non-Obvious Reward Hacking in Continuous Dynamics

- **Visual Reference:** [continuous_car_onset_dynamics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/continuous_car_experiment/continuous_car_onset_dynamics.png)

```
        HEALTHY AGENT (Step 10,000)                HACKED AGENT (Step 20,000)
    [Start]                                    [Start]
       \                                          \
        \                                          \-----> (Speed Donuts 4.2 m/s)
         \                                                    /     \
          -----> [Destination: SUCCESS]                      (  x    )
                                                              \     /
                                                               -----
                                                   [Destination: NEVER REACHED]
```

- **Environment Dynamics:**
  - Continuous 2D Kinematic Bicycle Model: state $s = [x, y, v, \theta]$, action $a = [\text{throttle}, \text{steering}]$.
  - **True Objective:** Reach destination within $1.5\text{ m}$ ($R_{\text{true}} = +10.0$, terminal).
  - **Proxy Reward:** Living penalty ($-0.02$) + Speed reward ($+0.1 \cdot v$).
- **The Hacking Phenomenon:**
  - At Step 10,000: Agent drives straight to destination ($100\%$ success, Path Efficiency $= 0.88$).
  - At Step 17,500: Agent discovers that doing high-speed $4.2\text{ m/s}$ donuts indefinitely pays $+42$ per episode, while reaching the destination ends the episode and caps reward!
  - Goal completion drops from **$100\% \to 0\%$**; Path Efficiency plummets from **$1.00 \to 0.07$**.
- **Detection Result:**
  - Our LSTM-DQN detector fired at **Step 17,500**, exactly when path efficiency collapsed, stopping training 2,500 steps before catastrophic failure.

#### Speaker Notes
> "Reviewers often ask: 'Does this only happen in toy discrete grid worlds?' To test this, we built a continuous autonomous vehicle simulation using a kinematic bicycle model with steering and throttle control.
> 
> The car was instructed to drive to a destination. To encourage it to hurry, we added a small speed bonus. Notice what happens: early on, the car drives straight to the destination. But by step 17,500, it realizes that reaching the destination ends the episode, whereas spinning donuts at 4.2 meters per second generates infinitely more speed bonus!
> 
> Destination arrivals dropped to 0%. Crucially, our LSTM-DQN detected this transition at Step 17,500 without knowing the car's destination coordinates, purely by noticing the sudden collapse in trajectory efficiency."

---

## Slide 5: Behavioral Footprints (How We Detect Hacking Blindfolded)
### The 10 Canonical Feature Signatures

- **Visual Reference:** [03_behavioral_features_dynamics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/multi_env_experiments/plots/03_behavioral_features_dynamics.png)

```
FEATURE DYNAMICS AT ONSET (Step 10,000 -> 15,000):
1. Path Efficiency:         1.00 ========\\___________________ 0.03  (97% Drop)
2. State Revisit Rate:      0.00 ________//=================== 0.96  (Surge)
3. Policy Divergence (TVD): 0.00 ________/\___________________ 0.85  (Transient Spike)
4. Policy Entropy:          1.38 ==========\\_________________ 0.24  (Collapse)
```

| Feature Group | Metric | Formula / Meaning | Onset Signature |
|:---|:---|:---|:---|
| **Trajectory Geometry** | `path_efficiency` | $\frac{\text{Displacement}}{\text{Path Length}}$ | Dramatic cliff drop ($1.00 \to 0.03$) |
| **Trajectory Geometry** | `revisit_rate` | $1.0 - \frac{|\text{Unique States}|}{\text{Total Steps}}$ | Surges to $>0.95$ (looping loops) |
| **Policy Dynamics** | `policy_divergence_tvd` | $\frac{1}{2}\sum_a \|\pi_t(a\|s) - \pi_{t-1}(a\|s)\|$ | Sharp transient spike at phase shift |
| **Action Distribution** | `behavioral_dispersion` | Entropy of action marginals | Collapses as policy commits to exploit |
| **Proxy Diagnostics** | `proxy_return_smoothed` | Exponential moving average return | Uncoupling upwards from normal baseline |

#### Speaker Notes
> "How can the detector know the agent is hacking if it cannot see the true reward? By tracking its behavioral footprints.
> 
> Healthy learning looks like exploration followed by direct goal navigation. But reward hacking has universal physical signatures:
> First, Path Efficiency collapses like a cliff. Instead of traveling across the state space, the agent runs in tight circles.
> Second, Revisit Rate jumps to nearly 1.0 because the agent repeatedly visits the exact same exploit tiles.
> Third, Total Variation Divergence between successive policy checkpoints exhibits a sharp transient spike as the policy pivots from healthy to hacked behavior.
> 
> These 10 features provide a compact, domain-agnostic summary of policy evolution."

---

## Slide 6: Detector Architecture (Why LSTM-DQN?)
### Formulating Stopping Decisions as a Recurrent Bellman Equation

```
   Temporal Feature Windows
   [x_{t-3}, x_{t-2}, x_{t-1}, x_t]
                  |
                  v
       +--------------------+
       |  1-Layer LSTM (64) |  <-- Encodes temporal trajectory & trends into
       +--------------------+      hidden state h_t (Belief State)
                  |
                  v
       +--------------------+
       |  Linear Layer (64) |
       +--------------------+
                  |
                  v
       +--------------------+
       |   Q(s_t, WAIT)     |  <-- Value of collecting more evidence
       |   Q(s_t, FLAG)     |  <-- Value of stopping and sounding alarm
       +--------------------+
```

- **Why Not a Simple Threshold?**
  - Static thresholds fail when returns naturally fluctuate, producing severe false positive rates on honest agents ($>60\%$).
- **Why Not Just an LSTM Classifier?**
  - Supervised LSTMs minimize cross-entropy loss on static labels without understanding the asymmetric cost of time.
  - A classifier doesn't know that delaying 1 checkpoint costs little, but a false alarm aborts an entire training job.
- **Why Recurrent DQN (LSTM-DQN)?**
  - **LSTM:** Solves partial observability by maintaining belief state $h_t$ over multi-step temporal windows ($W=4$).
  - **DQN:** Directly optimizes the Bellman equation under asymmetric penalties ($c_{\text{false}} = -10.0$ vs $c_{\text{delay}} = -0.1$). Stopping is chosen only when $Q(\text{FLAG}) > Q(\text{WAIT})$.

#### Speaker Notes
> "A frequent question is: 'Why use an RL agent to monitor an RL agent? Why not just use a standard LSTM classifier or an IF-condition?'
> 
> Simple heuristic thresholds fail completely because features naturally fluctuate during early exploration. If you set a fixed threshold on entropy or return, you get catastrophic false alarms.
> 
> A standard supervised classifier tries to guess whether a single checkpoint is hacked or not, ignoring the asymmetric cost of time. 
> By framing this as a DQN optimal stopping problem, the agent balances the value of collecting one more observation checkpoint against the risk of letting an exploit run wild. The LSTM layer encodes the temporal velocity of change, while the Q-network decides the mathematically optimal moment to stop."

---

## Slide 7: Primary Experimental Results vs Baselines
### Outperforming Mathematical Baselines ($O_1$, $O_2$) Across All Seeds

- **Visual Reference:** [02_detector_comparison_metrics.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/detector_plots/02_detector_comparison_metrics.png)

```
DETECTOR PERFORMANCE BENCHMARK:
Metric                 O1 (Static Baseline)   O2 (Moving Avg)    LSTM-DQN (Ours)
---------------------------------------------------------------------------------
F1-Score               0.000                  0.000              0.933 ± 0.133
Recall                 0.000                  0.000              0.900
Precision              0.000                  0.000              1.000
Clean False Positives  0.0%                   0.0%               0.0% (Zero False Alarms)
Detection Delay        N/A                    N/A                0.50 Checkpoints (~2,500 steps)
```

- **Key Takeaways:**
  1. **Mathematical Baselines Failed Completely ($F_1 = 0.000$):** Heuristic change-point detectors ($O_1$ proxy derivative, $O_2$ divergence filters) failed to discriminate between normal policy convergence and true hacking onset.
  2. **High Precision ($1.000$) & Recall ($0.900$):** The LSTM-DQN flagged $9$ out of $10$ reward-hacking runs across random seeds and exploit severities.
  3. **Zero False Alarms ($0.0\%$):** Never once falsely halted an honest, clean agent ($w=0.0$).
  4. **Rapid Onset Response:** Average detection latency was only **$0.50$ checkpoints** ($2,500$ environment steps after true onset occurred).

#### Speaker Notes
> "Here are our primary benchmark results on the test suite. We compared our LSTM-DQN against two established mathematical baselines: O1, which monitors statistical rate of change, and O2, which monitors threshold crossings.
> 
> As you can see, both mathematical baselines achieved an F1-score of 0.000. They either never triggered or triggered randomly.
> 
> In contrast, our LSTM-DQN achieved a 0.933 F1-score, with a 90% recall, 100% precision, and crucially, zero false positives on clean control policies. It caught the hacking behavior within half a checkpoint—meaning the practitioner loses virtually no training compute before the alarm is sounded."

---

## Slide 8: Feature Ablation Study
### What Features Actually Drive Detection?

- **Visual Reference:** [ablation_comparison.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/results/ablation_study/ablation_comparison.png)

```
ABLATION EXPERIMENT: REMOVING FEATURE GROUPS
Configuration                           F1-Score    Recall   Clean FPR
----------------------------------------------------------------------
Full Canonical Feature Set (10 Feats)   0.933       0.900    0.0%
- No Trajectory Geometry (Path/Revisit) 0.286       0.200    0.0%   <-- CRITICAL DROP (-65%)
- No Policy Divergence (TVD)            0.800       0.700    0.0%   <-- Moderate Impact (-13%)
- No Action Entropy / Dispersion        0.750       0.600    0.0%   <-- Moderate Impact (-18%)
- No Proxy Reward Features              0.909       0.900    0.0%   <-- Minimal Impact (-2%)
```

- **Core Discovery:**
  - **Trajectory Geometry is King:** Removing `path_efficiency` and `revisit_rate` caused performance to collapse from $0.933 \to 0.286$. Spatial looping is the single strongest physical invariant of reward hacking.
  - **Proxy Rewards are Redundant:** Removing proxy reward features only reduced $F_1$ from $0.933 \to 0.909$. The detector can identify hacking onset **almost entirely from behavior**, without even looking at proxy rewards!

#### Speaker Notes
> "To understand what the neural network was actually learning, we conducted a rigorous feature ablation study where we stripped away specific feature subsets.
> 
> The results were striking:
> When we removed trajectory geometry—path efficiency and revisit rate—the detector's F1-score collapsed from 0.933 to 0.286. Spatial looping is the fundamental physical signature of reward hacking.
> 
> Even more fascinating: when we removed the proxy reward features entirely, performance barely moved, from 0.933 to 0.909. This proves our detector doesn't need to know the reward function numbers—it sees reward hacking purely through the physical degradation of the agent's behavior."

---

## Slide 9: Generalization Study: 3 Dimensions
### Testing Zero-Shot Transfer to Unseen Environments & Exploits

- **Visual Reference:** [04_generalisation_master_dashboard.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/results/plots/04_generalisation_master_dashboard.png)

```
        DIMENSION 1: ENVIRONMENTS                 DIMENSION 2: EXPLOIT TYPES
  RegionToggle (Train)    -> F1: 0.933       Boundary Oscillation -> F1: 0.933
  CoinCollector (Unseen)  -> F1: 1.000       Resource Respawn     -> F1: 0.880
  BatteryCollector (Unseen)-> F1: 0.800      In-Place Farming     -> F1: 1.000
  VacuumCleaner (Unseen)  -> F1: 1.000       ----------------------------------
  ------------------------------------       Clean False Positive Rate: 0.0% across all!
  Mean Cross-Env F1: 0.933 (Gap: 0.000)
```

- **Cross-Environment Transfer:**
  - Evaluated on **`VacuumCleaner`** (Russell's dirt-dumping/cleaning dilemma) zero-shot: achieved **$F_1 = 1.000$** and detected onset in **$0.60$ checkpoints**.
  - `CoinCollector` achieved **$1.000$**, `BatteryCollector` achieved **$0.800$**.
- **Cross-Exploit Transfer:**
  - Successfully detects boundary hopping, cyclic waypoint farming, and stationary in-place cleaning with no retuning.
  - **Zero False Alarms ($0.0\%$ FPR):** Confirmed across all unseen environments.

#### Speaker Notes
> "Now let's turn to our central generalization study. A detector is useless if it only works on the single grid world where it was trained.
> We evaluated our RegionToggle-trained detector zero-shot on three unseen environments: CoinCollector, BatteryCollector, and Stuart Russell's classic VacuumCleaner dilemma.
> 
> In VacuumCleaner, the agent is supposed to dock at the charging station, but discovers it can farm an infinite respawning dust patch in the corner. Our detector transferred with an F1-score of 1.000 and zero false alarms.
> Across different exploit mechanisms—whether boundary oscillation, multi-cell looping, or stationary farming—the detector recognized the behavioral signature with virtually zero generalization gap."

---

## Slide 10: Algorithmic Failure Analysis (An Honest Look)
### Why PPO Transferred ($F_1 = 0.933$) but A2C & DQN Failed Zero-Shot ($F_1 = 0.000$)

- **Visual Reference:** [03_cross_algorithm_latency_comparison.png](file:///home/statixlo/Develop/Reward-Hacking-onset-Detection/generalisation_test/results/plots/03_cross_algorithm_latency_comparison.png)

```
PPO (Succeeds: F1=0.933):
Healthy Policy ----[Gradual Transition Window]----> Hacked Policy
                    * Entropy Collapses (1.38 -> 0.24)
                    * LSTM detects temporal slope!

DQN (Fails: F1=0.000):
Exploit Learned Immediately at Step 5,000 (Index 1)
                    * No healthy-to-hack transition slope in window W=4
                    * Action dispersion remains higher (1.05 - 1.30)
                    * LSTM waits for a transition that already occurred!

A2C (Degrades: F1=0.000):
Unclipped gradients maintain persistent high entropy (Dispersion > 1.24)
                    * Policy vacillates stochastically between goal and exploit
                    * Never triggers ground-truth consecutive onset threshold
```

- **The Core Scientific Insight:**
  - The detector does **not** fail because of environment layouts or exploit geometries.
  - It encounters distribution shift due to **policy learning dynamics**:
    1. **Off-policy Q-Learning (DQN):** Learns exploits nearly instantaneously; requires a shorter sliding window or static onset prior.
    2. **Stochastic Actor-Critic (A2C):** High action entropy dampens trajectory collapse; requires entropy-normalized calibration.

#### Speaker Notes
> "In science, identifying where your model fails is just as important as showing where it succeeds.
> When we tested cross-algorithm transfer, our PPO-trained detector did not transfer zero-shot to DQN and A2C policies.
> 
> Why? Because policy learning dynamics differ fundamentally:
> PPO exhibits a gradual exploratory phase transition where entropy collapses from 1.38 down to 0.24. Our LSTM was trained on this temporal descent.
> 
> In contrast, DQN's greedy argmax updates discover the exploit almost instantly at step 5,000. By the time our sliding window observed the agent, it was already hacking from the start! The LSTM kept waiting for a transition slope that had already passed.
> In A2C, persistent high exploration entropy prevented the policy from sustaining the exploit consecutively, creating disagreement with the ground truth.
> 
> This provides a clear roadmap: multi-algorithm deployment requires training the detector on diverse generator learning dynamics."

---

## Slide 11: Summary & Future Roadmap

```
+--------------------------------------------------------------------------+
|                        SUMMARY OF CONTRIBUTIONS                          |
+--------------------------------------------------------------------------+
| 1. Formulated Reward Hacking Onset Detection as an Optimal Stopping POMDP|
| 2. Strictly quarantined ground-truth reward to prevent data leakage     |
| 3. Engineered 10 canonical behavioral footprint features                 |
| 4. Outperformed mathematical baselines (F1: 0.933 vs 0.000)             |
| 5. Validated on continuous ADAS vehicles and 4 discrete environments     |
| 6. Demonstrated zero-shot generalization across environments & exploits  |
| 7. Identified and documented the algorithmic entropy boundary            |
+--------------------------------------------------------------------------+
```

- **Immediate Deployment Applications:**
  - Automated early-stopping monitor for RL training pipelines (saving GPU compute).
  - Safety circuit breaker for robotics and autonomous driving simulations.
- **Future Work:**
  - Multi-generator training (jointly training on PPO, DQN, SAC, and TD3).
  - Vision-based feature extractors for pixel-level RL (Atari, Procgen, CARLA).

#### Speaker Notes
> "To summarize: we have presented a principled, practical framework for catching reward hacking onset in reinforcement learning.
> By relying on behavioral footprints rather than reward signals, our LSTM-DQN achieves high-precision early stopping with zero false alarms.
> It scales from discrete grids to continuous self-driving cars, and generalizes zero-shot across environments and exploit geometries.
> 
> Thank you for your time, and I am now happy to open the floor for questions."

---

## Slide 12: Technical Defense / Committee Q&A Prep

| Expected Tough Question | Direct Scientific Answer |
|:---|:---|
| **"Why not just train with a better reward function?"** | In real-world alignment, the true objective is often unspecifiable mathematically. If we could perfectly specify the reward, reward hacking wouldn't exist. Our tool protects against unknown misspecifications. |
| **"How do you know features won't trigger on healthy policies?"** | We explicitly evaluated on clean control runs ($w=0.0$) across all environments. Clean agents navigate to goals with high path efficiency, yielding a verified $0.0\%$ False Positive Rate. |
| **"Why sliding window $W=4$?"** | $W=4$ provides sufficient historical context to measure temporal velocity (slopes) of divergence without incurring excessive observation latency. |
| **"Why did DQN fail zero-shot?"** | The detector learned the temporal transition velocity of PPO. DQN's immediate onset creates a step-function rather than a sigmoid transition. Multi-algorithm training resolves this shift. |
