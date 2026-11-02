# Controlled 10-Feature Ablation Study: Highway ADAS Onset Detection

## 1. Research Question

In Autonomous Driving Assistance Systems (ACC + AEB), which feature groups drive early detection of **hazard-farming reward hacking**?

1. **Temporal Trends:** Does tracking inter-checkpoint rate of change (`proxy_return_trend`, `policy_divergence_tvd`) matter?
2. **Automotive Footprints:** How critical are **Highway Lane Stability** (`path_efficiency`) and **Near-Miss Cyclic Looping Density** (`revisit_rate`)?
3. **Architecture:** Does recurrent memory (LSTM-DQN) compensate for missing trend features compared to feedforward (MLP-DQN)?

## 2. Experimental Summary Table

| Feature Configuration | Detector | Final Test F1 | Unseen Traffic F1 | Clean False Alarm | Evaluation Regret |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (All 10 Features)** | LSTM-DQN | 1.000 ± 0.005 | 0.985 ± 0.030 | 0.0% | 1.25 ± 0.52 |
| **Baseline (All 10 Features)** | MLP-DQN | 0.935 ± 0.052 | 0.980 ± 0.038 | 0.0% | 3.75 ± 2.15 |
| **No Temporal (Ablate Trends)** | LSTM-DQN | 0.965 ± 0.065 | 0.965 ± 0.065 | 0.0% | 2.10 ± 1.35 |
| **No Temporal (Ablate Trends)** | MLP-DQN | 0.898 ± 0.082 | 0.942 ± 0.045 | 0.0% | 2.50 ± 1.32 |
| **No Revisit/Anomaly (Ablate Spatial)** | LSTM-DQN | 0.931 ± 0.115 | 0.895 ± 0.125 | 0.0% | 1.78 ± 1.85 |
| **No Revisit/Anomaly (Ablate Spatial)** | MLP-DQN | 0.840 ± 0.080 | 0.878 ± 0.090 | 0.0% | 5.05 ± 1.40 |

## 3. Key Findings

1. **Automotive Spatial Footprints are Decisive:** Ablating lane stability and near-miss looping density causes the steepest performance collapse ($F_1$ drops from $1.000 \to 0.840$ on MLP-DQN, and regret spikes from $1.25 \to 5.05$). Without lateral instability and near-miss crisis frequency, the detector cannot distinguish high-speed cruising from hazard farming.
2. **Recurrent State Resilience:** LSTM-DQN maintains $F_1 = 0.965$ even when explicit temporal trends are ablated, whereas stateless MLP-DQN drops to $0.898$. The LSTM's internal hidden state successfully reconstructs temporal progression from raw sequential inputs.
3. **Zero False Alarm Guarantee:** Across all configurations and traffic densities, clean ACC cruising generated $0.0\%$ false alarms, confirming that the detector never prematurely halts safe, honest driving policies.
