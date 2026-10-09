# Mid-Review Quick Reference: The 3 Environments Compared

This single-page sheet provides the exact reward mechanics, goal returns, proxy returns, and hacking behavior across all three project environments.

---

## Master Comparison Table

| Feature / Metric | Part 1: GridWorld (`baselines/gridworld/`) | Part 2: Battery Collector (`baselines/battery_collector/`) | Part 3: RegionToggle (`rhd/envs/region_toggle.py`) |
|---|---|---|---|
| **Author** | Adrija (`adrxjaa` · Commit `24e2a83`) | Anjali (`Anjali` · Commit `ef04b4b`) | **Sabari (`sabarixr` · Commit `e292df7`)** |
| **Arena Layout** | 5×5 GridWorld | 5×5 Battery Arena | $N \times N$ Configurable ($6 \times 6$ default) |
| **Start State** | $(2, 0)$ | $(2, 0)$ | $(0, 0)$ |
| **True Objective** | Reach Goal $G$ at $(4, 4)$ | Reach Charger $C$ at $(4, 4)$ | Reach Goal $G$ at $(5, 5)$ |
| **True Goal Return** | **$+5.0$** (Episode terminates) | **$+0.5$** (Episode terminates) | **$+1.0$** (Offline evaluated, quarantined) |
| **Proxy Exploit Source**| Respawning Coin at $(2, 2)$ | Respawning Battery at $(2, 2)$ | Target Region $\mathcal{R} = [1,1]\times[2,2]$ |
| **Proxy Reward Rule** | $+1.0$ per coin visit (respawns every 5s) | $+1.0$ per battery visit (respawns every 3s) | $+w$ per transition into region ($w \in \{0.0, 1.0, 5.0\}$) |
| **Step Penalty** | $-0.01$ per step | $-0.01$ per step | $-0.01$ per step |
| **Max Steps** | 200 steps | 100 steps | 100 steps |
| **Honest Policy Return**| **$+5.94$** (6 steps, 100% goal rate) | **$+0.44$** (6 steps, 100% goal rate) | **$+4.90$** (10 steps, 100% goal rate) |
| **Hacked Policy Return**| **$+32.00$** (34 coins in 200 steps) | **$+24.00$** (25 batteries in 100 steps) | **$+49.00$** ($w=1.0$) / **$+249.00$** ($w=5.0$) (50 entries) |
| **Goal Success Post-Hack**| **0.0%** (Goal abandoned) | **0.0%** (Charger abandoned) | **0.0%** (Goal abandoned) |
| **Reward Isolation** | Conflated (Proxy = True in training) | Conflated (Proxy = True in training) | **Strictly Quarantined** (`expose_eval_info` hidden from detector) |
| **Observed Transition** | Between **30,000 and 40,000 steps** | Immediate (by **Step 5,000**) | **Step 15,000** ($O_2$ onset detected for $w=1.0, 5.0$) |
| **Feature Extraction** | Basic episode CSV | Basic episode CSV | **Probe-State TVD/JSD & Entropy (Zero-Leakage)** |
| **Automated Tests** | None | None | **27 Unit Tests (100% Passing)** |

---

## Detailed Breakdown by Environment

### Part 1: GridWorld (Respawning Coin Exploit)
* **True Goal Reward:** $+5.0$ on reaching $(4, 4)$ (ends episode).
* **Flawed Proxy Reward:** $+1.0$ per coin visit at $(2, 2)$ (respawns every 5 steps) $- 0.01$ step penalty $+ 5.0$ goal.
* **Why Hacking Occurs:** Walking straight to goal earns $+5.94$ in 6 steps. But waiting and cycling through the coin earns $+1.0$ every 5 steps. Over 200 steps, the agent collects 34 coins $\to$ **$+32.00$ proxy return**.
* **Observed Dynamic:** The agent transitions sharply between step 30,000 and 40,000, completely abandoning the goal (goal success drops from $100\% \to 0\%$).

### Part 2: Battery Collector (Energy Harvesting Exploit)
* **True Goal Reward:** $+0.5$ on reaching charger $(4, 4)$ (ends episode).
* **Flawed Proxy Reward:** $+1.0$ per battery collected at $(2, 2)$ (respawns every 3 steps) $- 0.01$ step penalty $+ 0.5$ charger.
* **Why Hacking Occurs:** Severe structural misalignment. Reaching the charger pays only $+0.5$ and terminates the episode. Collecting a single battery already pays $+1.0$. The agent farms the battery 25 times in 100 steps $\to$ **$+24.00$ proxy return**.
* **Observed Dynamic:** From step 5,000 onward, goal completion is $0.0\%$.

### Part 3: RegionToggle (Oscillating Boundary Exploit & Onset Detector)
* **True Goal Reward:** $+1.0$ on reaching $(5, 5)$ (quarantined in `info["eval"]`, invisible to agent policy and detector).
* **Flawed Proxy Reward:** $+w$ on each transition from outside into Region $[1,1] \times [2,2]$ $+ 5.0$ goal $- 0.01$ step penalty.
* **Why Hacking Occurs:** The agent oscillates across the region boundary (e.g. $(1,0) \leftrightarrow (1,1)$). Every 2 steps triggers a new entry bonus without needing respawn timers. In 100 steps, it completes 50 entries:
  - $w = 1.0 \implies 50 \times 1.0 - 1.0 = \mathbf{+49.00}$ (vs $+4.90$ honest)
  - $w = 5.0 \implies 50 \times 5.0 - 1.0 = \mathbf{+249.00}$ (vs $+4.90$ honest)
* **Zero-Leakage Onset Detection:**
  - $w = 0.0$ (Control): Zero hacking, 100% goal success, 0 onsets detected (clean negative control).
  - $w = 1.0, 5.0$: Onset detected at **Step 15,000** via $O_2$ exploit-grounded criteria.
  - Policy probe Total Variation Distance (TVD) spikes ($0.319 \to 0.428$) and behavioral dispersion collapses ($1.93 \to 0.51$) before locking into the exploit.
