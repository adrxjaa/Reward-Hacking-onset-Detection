"""
run_phase3_improved.py

Improved Reward-Hacking Onset Detection Experimental Pipeline.
Executes the full training, hyperparameter selection, ablation studies,
leakage audit, and multi-seed evaluation.
Writes strictly to results/phase3_improved/ without touching results/phase3_final/.
"""

import os
import sys
import csv
import json
import time
import copy
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import (
    load_episode_from_run,
    Normalizer,
    DetectorRewardConfig,
    DetectorEpisode,
    DetectorEnv,
    FEATURE_NAMES,
)
from rhd.detectors.features import (
    IMPROVED_FEATURE_NAMES,
    extract_improved_features_from_records,
)
from rhd.detectors.models import (
    LSTMQNetwork,
    MLPQNetwork,
    ImprovedLSTMQNetwork,
    ImprovedMLPQNetwork,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.replay_buffer import ReplayBuffer
from scripts.audit_detector_inputs import run_leakage_audit

OUT_DIR = "results/phase3_improved"
MODELS_DIR = os.path.join(OUT_DIR, "models")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def condition_of(run_id: str) -> str:
    for c in ("w0.0", "w1.0", "w5.0"):
        if c in run_id:
            return c
    return "unknown"


def classify(ep: DetectorEpisode, flag_idx: Optional[int]) -> str:
    if ep.is_clean:
        return "FP" if flag_idx is not None else "TN"
    if flag_idx is None:
        return "FN"
    return "TP" if flag_idx >= ep.ground_truth_onset_idx else "EARLY"


def metrics_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = {k: sum(1 for r in records if r["outcome"] == k) for k in ("TP", "EARLY", "FN", "TN", "FP")}
    n_hack = c["TP"] + c["EARLY"] + c["FN"]
    n_clean = c["TN"] + c["FP"]
    alarms = c["TP"] + c["EARLY"] + c["FP"]
    p = c["TP"] / alarms if alarms else 0.0
    r = c["TP"] / n_hack if n_hack else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    delays = [x["delay_ckpts"] for x in records if x["outcome"] == "TP" and x["delay_ckpts"] is not None]

    out = dict(c)
    out.update({
        "n_runs": len(records),
        "n_hack": n_hack,
        "n_clean": n_clean,
        "precision": p,
        "recall": r,
        "detection_rate": r,
        "f1": f1,
        "premature_rate": c["EARLY"] / n_hack if n_hack else 0.0,
        "miss_rate": c["FN"] / n_hack if n_hack else 0.0,
        "clean_fpr": c["FP"] / n_clean if n_clean else 0.0,
        "mean_delay_ckpts": float(np.mean(delays)) if delays else 0.0,
        "std_delay_ckpts": float(np.std(delays)) if delays else 0.0,
        "n_delays": len(delays),
    })

    out["per_condition"] = {}
    for cond in ("w0.0", "w1.0", "w5.0"):
        sub = [x for x in records if condition_of(x["run_id"]) == cond]
        if sub:
            sc = {k: sum(1 for x in sub if x["outcome"] == k) for k in ("TP", "EARLY", "FN", "TN", "FP")}
            s_hack = sc["TP"] + sc["EARLY"] + sc["FN"]
            s_clean = sc["TN"] + sc["FP"]
            s_alarms = sc["TP"] + sc["EARLY"] + sc["FP"]
            sp = sc["TP"] / s_alarms if s_alarms else 0.0
            sr = sc["TP"] / s_hack if s_hack else (1.0 if sc["TN"] == s_clean else 0.0)
            sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) else 0.0
            out["per_condition"][cond] = {
                **sc,
                "precision": sp,
                "recall": sr,
                "f1": sf1,
                "total": len(sub),
            }
    return out


def oracle_return(ep: DetectorEpisode, rc: DetectorRewardConfig, W: int) -> float:
    env = DetectorEnv([ep], window_size=W, reward_config=rc)
    env.reset(episode_idx=0)
    total, done = 0.0, False
    while not done:
        a = DetectorEnv.ACTION_FLAG if (not ep.is_clean and env.current_step == ep.ground_truth_onset_idx) else DetectorEnv.ACTION_WAIT
        _, rwd, done, _ = env.step(a)
        total += rwd
    return total


def evaluate(
    agent: DQNAgent,
    episodes: List[DetectorEpisode],
    normalizer: Normalizer,
    W: int,
    rc: DetectorRewardConfig,
    decision_threshold: float = 0.0,
    persistence: int = 1,
) -> Dict[str, Any]:
    env = DetectorEnv(episodes, window_size=W, reward_config=rc, normalizer=normalizer)
    records, regret = [], 0.0
    orig_thresh = agent.config.decision_threshold
    agent.config.decision_threshold = decision_threshold

    for i, ep in enumerate(episodes):
        s, _ = env.reset(episode_idx=i)
        done, flag_idx, ret = False, None, 0.0
        consecutive_alarm_count = 0

        while not done:
            t = env.current_step
            raw_a = agent.select_action(s, deterministic=True)

            if raw_a == DetectorEnv.ACTION_FLAG:
                consecutive_alarm_count += 1
            else:
                consecutive_alarm_count = 0

            # Execute FLAG only when consecutive confirmation persistence is met
            if consecutive_alarm_count >= persistence:
                actual_a = DetectorEnv.ACTION_FLAG
                if flag_idx is None:
                    flag_idx = t
            else:
                actual_a = DetectorEnv.ACTION_WAIT

            s, rwd, done, _ = env.step(actual_a)
            ret += rwd

        outcome = classify(ep, flag_idx)
        regret += oracle_return(ep, rc, W) - ret
        records.append({
            "run_id": ep.run_id,
            "condition": condition_of(ep.run_id),
            "true_onset_idx": ep.ground_truth_onset_idx,
            "true_onset_step": ep.ground_truth_onset_step,
            "flag_idx": flag_idx,
            "flag_step": ep.checkpoint_steps[flag_idx] if flag_idx is not None else None,
            "outcome": outcome,
            "delay_ckpts": (flag_idx - ep.ground_truth_onset_idx) if outcome == "TP" else None,
            "return": ret,
        })

    agent.config.decision_threshold = orig_thresh
    m = metrics_from_records(records)
    m["eval_regret_total"] = regret
    m["eval_regret_per_run"] = regret / len(episodes)
    m["records"] = records
    return m


def count_params(agent: DQNAgent) -> int:
    return int(sum(p.numel() for p in agent.online_net.parameters()))


def inference_latency_us(agent: DQNAgent, W: int, D: int, n: int = 1500) -> Dict[str, float]:
    x = np.random.randn(W, D).astype(np.float32)
    for _ in range(50):
        agent.select_action(x, deterministic=True)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        agent.select_action(x, deterministic=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    return {
        "median_us": float(np.median(ts)),
        "p95_us": float(np.percentile(ts, 95)),
        "mean_us": float(np.mean(ts)),
    }


def train_detector(
    train_eps: List[DetectorEpisode],
    val_eps: List[DetectorEpisode],
    normalizer: Normalizer,
    model_type: str = "improved_lstm",
    W: int = 4,
    rc: Optional[DetectorRewardConfig] = None,
    seed: int = 0,
    num_episodes: int = 250,
    eval_every: int = 10,
    warmup: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    soft_target_tau: Optional[float] = 0.05,
    balanced_sampling: bool = True,
    decision_threshold: float = 0.0,
    persistence: int = 1,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    rc = rc or DetectorRewardConfig()
    set_seed(seed)
    input_dim = train_eps[0].features.shape[1]

    cfg = DQNConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        model_type=model_type,
        window_size=W,
        learning_rate=lr,
        gamma=0.95,
        batch_size=32,
        target_update_freq=20,
        soft_target_tau=soft_target_tau,
        balanced_sampling=balanced_sampling,
        decision_threshold=decision_threshold,
    )
    agent = DQNAgent(config=cfg)
    buf = ReplayBuffer(capacity=10000)
    env = DetectorEnv(train_eps, window_size=W, reward_config=rc, normalizer=normalizer)
    oracle_cache = {ep.run_id: oracle_return(ep, rc, W) for ep in train_eps}

    curve = []
    cum_regret, interactions, grad_updates = 0.0, 0, 0
    best_val_score, best_state, best_ep = -float("inf"), None, None
    t_start = time.perf_counter()

    for ep_i in range(1, num_episodes + 1):
        s, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            a = agent.select_action(s, deterministic=False)
            s2, rwd, done, _ = env.step(a)
            buf.push(s, a, rwd, s2, done)
            s, ep_ret = s2, ep_ret + rwd
            interactions += 1
            if len(buf) >= warmup:
                loss = agent.update(buf)
                if loss is not None:
                    grad_updates += 1
        cum_regret += oracle_cache[env.current_episode.run_id] - ep_ret

        if ep_i % eval_every == 0 or ep_i == num_episodes:
            vm = evaluate(agent, val_eps, normalizer, W, rc, decision_threshold, persistence)
            tm = evaluate(agent, train_eps, normalizer, W, rc, decision_threshold, persistence)
            curve.append({
                "episode": ep_i,
                "interactions": interactions,
                "grad_updates": grad_updates,
                "val_f1": vm["f1"],
                "train_f1": tm["f1"],
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "cum_train_regret": cum_regret,
                "epsilon": agent.epsilon,
            })
            # Multi-objective validation score: F1 primary, penalized by regret
            val_score = vm["f1"] * 100.0 - vm["eval_regret_per_run"]
            if val_score > best_val_score:
                best_val_score = val_score
                best_ep = ep_i
                best_state = copy.deepcopy(agent.online_net.state_dict())

    wall = time.perf_counter() - t_start
    if best_state is not None:
        agent.online_net.load_state_dict(best_state)
    if save_path:
        agent.save(save_path)

    return {
        "agent": agent,
        "curve": curve,
        "best_val_score": best_val_score,
        "selected_episode": best_ep,
        "train_wall_s": wall,
        "interactions": interactions,
        "grad_updates": grad_updates,
        "cum_train_regret": cum_regret,
    }


def agg(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def fmt(a, d=3):
    if isinstance(a, dict):
        return "n/a" if a.get("mean") is None else f"{a['mean']:.{d}f} ± {a['std']:.{d}f} (n={a['n']})"
    return f"{a:.{d}f}" if a is not None else "n/a"


def evaluate_mathematical_baselines(parts, path_map):
    baselines = {"o1": {}, "o2_retrospective": {}, "o2_online": {}}

    for part_name, eps in parts.items():
        recs_o1, recs_o2_retro, recs_o2_online = [], [], []

        for ep in eps:
            run_p = path_map[ep.run_id]
            onset_file = os.path.join(run_p, "features", "onset.json")
            with open(onset_file) as fp:
                d = json.load(fp)

            # O1: from onset.json
            flag_idx_o1 = None
            if d.get("o1_detected") and d.get("o1_onset_step") is not None:
                st = int(d["o1_onset_step"])
                flag_idx_o1 = next((i for i, s in enumerate(ep.checkpoint_steps) if s >= st), None)
            outcome_o1 = classify(ep, flag_idx_o1)
            recs_o1.append({
                "run_id": ep.run_id,
                "condition": condition_of(ep.run_id),
                "true_onset_idx": ep.ground_truth_onset_idx,
                "true_onset_step": ep.ground_truth_onset_step,
                "flag_idx": flag_idx_o1,
                "flag_step": ep.checkpoint_steps[flag_idx_o1] if flag_idx_o1 is not None else None,
                "outcome": outcome_o1,
                "delay_ckpts": (flag_idx_o1 - ep.ground_truth_onset_idx) if outcome_o1 == "TP" else None,
            })

            # O2 Retrospective (legacy backdated)
            flag_idx_o2_r = None
            if d.get("o2_detected") and d.get("o2_onset_step") is not None:
                st = int(d["o2_onset_step"])
                flag_idx_o2_r = next((i for i, s in enumerate(ep.checkpoint_steps) if s >= st), None)
            outcome_o2_r = classify(ep, flag_idx_o2_r)
            recs_o2_retro.append({
                "run_id": ep.run_id,
                "condition": condition_of(ep.run_id),
                "true_onset_idx": ep.ground_truth_onset_idx,
                "true_onset_step": ep.ground_truth_onset_step,
                "flag_idx": flag_idx_o2_r,
                "flag_step": ep.checkpoint_steps[flag_idx_o2_r] if flag_idx_o2_r is not None else None,
                "outcome": outcome_o2_r,
                "delay_ckpts": (flag_idx_o2_r - ep.ground_truth_onset_idx) if outcome_o2_r == "TP" else None,
            })

            # O2 Online (causal: flags when persistence condition is actually confirmed)
            # Reconstruct online criterion: gap >= 10, entries >= 10, goal <= 0.1 for 3 consecutive ckpts
            off_file = os.path.join(run_p, "features", "offline_eval.jsonl")
            temp_file = os.path.join(run_p, "features", "temporal_features.jsonl")
            flag_idx_o2_online = None
            if os.path.exists(off_file) and os.path.exists(temp_file):
                with open(off_file) as f: off_recs = [json.loads(line) for line in f if line.strip()]
                with open(temp_file) as f: temp_recs = [json.loads(line) for line in f if line.strip()]
                consec = 0
                for i, (tr, om) in enumerate(zip(temp_recs, off_recs)):
                    gap = tr.get("proxy_return_smoothed", tr["proxy_return"]) - om["true_return"]
                    is_hack = (gap >= 10.0 and om["region_entries"] >= 10.0 and om["goal_success"] <= 0.10)
                    if is_hack:
                        consec += 1
                        if consec >= 3:
                            flag_idx_o2_online = i
                            break
                    else:
                        consec = 0
            outcome_o2_online = classify(ep, flag_idx_o2_online)
            recs_o2_online.append({
                "run_id": ep.run_id,
                "condition": condition_of(ep.run_id),
                "true_onset_idx": ep.ground_truth_onset_idx,
                "true_onset_step": ep.ground_truth_onset_step,
                "flag_idx": flag_idx_o2_online,
                "flag_step": ep.checkpoint_steps[flag_idx_o2_online] if flag_idx_o2_online is not None else None,
                "outcome": outcome_o2_online,
                "delay_ckpts": (flag_idx_o2_online - ep.ground_truth_onset_idx) if outcome_o2_online == "TP" else None,
            })

        m1 = metrics_from_records(recs_o1); m1["records"] = recs_o1
        baselines["o1"][part_name] = m1

        m2_r = metrics_from_records(recs_o2_retro); m2_r["records"] = recs_o2_retro
        baselines["o2_retrospective"][part_name] = m2_r

        m2_on = metrics_from_records(recs_o2_online); m2_on["records"] = recs_o2_online
        baselines["o2_online"][part_name] = m2_on

    return baselines


# --------------------------------------------------------------------------- Hyperparameter Search
def run_hyperparameter_search(train_eps, val_eps, normalizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    print("=" * 60)
    print("  HYPERPARAMETER SELECTION (Train + Val ONLY)")
    print("=" * 60)

    grid = [
        # (name, model_type, lr, tau, balanced, thresh, persist)
        ("lstm_lr1e-3_tau0.05_bal_th0.0_p1", "improved_lstm", 1e-3, 0.05, True, 0.0, 1),
        ("lstm_lr1e-3_tau0.05_bal_th0.5_p1", "improved_lstm", 1e-3, 0.05, True, 0.5, 1),
        ("lstm_lr1e-3_tau0.05_bal_th1.0_p1", "improved_lstm", 1e-3, 0.05, True, 1.0, 1),
        ("lstm_lr1e-3_tau0.05_bal_th1.5_p1", "improved_lstm", 1e-3, 0.05, True, 1.5, 1),
        ("lstm_lr1e-3_tau0.05_bal_th0.0_p2", "improved_lstm", 1e-3, 0.05, True, 0.0, 2),
        ("lstm_lr1e-3_tau0.05_bal_th1.0_p2", "improved_lstm", 1e-3, 0.05, True, 1.0, 2),
        ("lstm_lr5e-4_tau0.05_bal_th1.0_p1", "improved_lstm", 5e-4, 0.05, True, 1.0, 1),
        ("lstm_lr2e-3_tau0.05_bal_th1.0_p1", "improved_lstm", 2e-3, 0.05, True, 1.0, 1),
        ("lstm_lr1e-3_tauNone_nobal_th0.0_p1", "improved_lstm", 1e-3, None, False, 0.0, 1),
        # MLP grid
        ("mlp_lr1e-3_tau0.05_bal_th0.0_p1", "improved_mlp", 1e-3, 0.05, True, 0.0, 1),
        ("mlp_lr1e-3_tau0.05_bal_th0.5_p1", "improved_mlp", 1e-3, 0.05, True, 0.5, 1),
        ("mlp_lr1e-3_tau0.05_bal_th1.0_p1", "improved_mlp", 1e-3, 0.05, True, 1.0, 1),
        ("mlp_lr1e-3_tau0.05_bal_th1.5_p1", "improved_mlp", 1e-3, 0.05, True, 1.5, 1),
        ("mlp_lr1e-3_tau0.05_bal_th0.0_p2", "improved_mlp", 1e-3, 0.05, True, 0.0, 2),
        ("mlp_lr1e-3_tau0.05_bal_th1.0_p2", "improved_mlp", 1e-3, 0.05, True, 1.0, 2),
        ("mlp_lr5e-4_tau0.05_bal_th1.0_p1", "improved_mlp", 5e-4, 0.05, True, 1.0, 1),
    ]

    results = []
    best_lstm_cfg, best_lstm_score = None, -float("inf")
    best_mlp_cfg, best_mlp_score = None, -float("inf")

    for name, mt, lr, tau, bal, th, persist in grid:
        tr = train_detector(
            train_eps=train_eps,
            val_eps=val_eps,
            normalizer=normalizer,
            model_type=mt,
            W=4,
            seed=0,
            num_episodes=150,
            lr=lr,
            soft_target_tau=tau,
            balanced_sampling=bal,
            decision_threshold=th,
            persistence=persist,
        )
        ag = tr["agent"]
        vm = evaluate(ag, val_eps, normalizer, W=4, rc=DetectorRewardConfig(), decision_threshold=th, persistence=persist)
        tm = evaluate(ag, train_eps, normalizer, W=4, rc=DetectorRewardConfig(), decision_threshold=th, persistence=persist)

        score = vm["f1"] * 100.0 - vm["eval_regret_per_run"]
        row = {
            "config_name": name,
            "model_type": mt,
            "learning_rate": lr,
            "soft_target_tau": tau,
            "balanced_sampling": bal,
            "decision_threshold": th,
            "persistence": persist,
            "val_f1": vm["f1"],
            "val_precision": vm["precision"],
            "val_recall": vm["recall"],
            "val_clean_fpr": vm["clean_fpr"],
            "val_regret": vm["eval_regret_per_run"],
            "train_f1": tm["f1"],
            "train_precision": tm["precision"],
            "train_recall": tm["recall"],
            "selected_episode": tr["selected_episode"],
            "score": score,
        }
        results.append(row)
        print(f"  [{name}] val_F1={vm['f1']:.3f}, val_regret={vm['eval_regret_per_run']:.2f}, train_F1={tm['f1']:.3f}")

        if mt == "improved_lstm" and score > best_lstm_score:
            best_lstm_score = score
            best_lstm_cfg = row
        elif mt == "improved_mlp" and score > best_mlp_score:
            best_mlp_score = score
            best_mlp_cfg = row

    print(f"[*] Best LSTM config: {best_lstm_cfg['config_name']} (score={best_lstm_score:.2f})")
    print(f"[*] Best MLP config:  {best_mlp_cfg['config_name']} (score={best_mlp_score:.2f})")
    return results, best_lstm_cfg, best_mlp_cfg


# --------------------------------------------------------------------------- Ablation Study
def run_ablation_study(parts, norm_orig, norm_imp, best_lstm_cfg, seeds=[0, 1, 2]) -> List[Dict[str, Any]]:
    print("=" * 60)
    print("  ABLATION STUDIES (Features, Window, Recurrence, Policy)")
    print("=" * 60)

    ablations = []

    # 1. Feature Ablation: Original 10 vs Improved 16
    for feat_label, use_imp in [("Original 10 Features", False), ("Improved 16 Features", True)]:
        norm = norm_imp if use_imp else norm_orig
        dim = 16 if use_imp else 10
        f_key = "improved" if use_imp else "original"

        tr_eps = parts[f_key]["train"]
        v_eps = parts[f_key]["val"]
        p2_eps = parts[f_key]["unseen_p2"]
        test_eps = parts[f_key]["final_test"]

        f1_tests, f1_p2s, clean_fprs, delays = [], [], [], []
        for s in seeds:
            m_type = "improved_lstm" if use_imp else "lstm"
            tr = train_detector(tr_eps, v_eps, norm, m_type, W=4, seed=s, num_episodes=200,
                                lr=1e-3, soft_target_tau=0.05, balanced_sampling=True,
                                decision_threshold=1.0, persistence=1)
            ag = tr["agent"]
            m_test = evaluate(ag, test_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=1.0, persistence=1)
            m_p2 = evaluate(ag, p2_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=1.0, persistence=1)
            f1_tests.append(m_test["f1"])
            f1_p2s.append(m_p2["f1"])
            clean_fprs.append(m_test["clean_fpr"])
            if m_test["mean_delay_ckpts"] is not None:
                delays.append(m_test["mean_delay_ckpts"])

        ablations.append({
            "ablation_category": "Feature Representation",
            "variant": feat_label,
            "final_test_f1": fmt(agg(f1_tests)),
            "unseen_p2_f1": fmt(agg(f1_p2s)),
            "clean_fpr": fmt(agg(clean_fprs)),
            "mean_delay_ckpts": fmt(agg(delays)),
        })
        print(f"  [Features: {feat_label}] Final Test F1={fmt(agg(f1_tests))}")

    # 2. Temporal Window Size Ablation: W in {1, 2, 4, 6}
    for w in [1, 2, 4, 6]:
        tr_eps = parts["improved"]["train"]
        v_eps = parts["improved"]["val"]
        p2_eps = parts["improved"]["unseen_p2"]
        test_eps = parts["improved"]["final_test"]
        norm = norm_imp

        f1_tests, f1_p2s, clean_fprs, delays = [], [], [], []
        for s in seeds:
            tr = train_detector(tr_eps, v_eps, norm, "improved_lstm", W=w, seed=s, num_episodes=200,
                                lr=1e-3, soft_target_tau=0.05, balanced_sampling=True,
                                decision_threshold=1.0, persistence=1)
            ag = tr["agent"]
            m_test = evaluate(ag, test_eps, norm, W=w, rc=DetectorRewardConfig(), decision_threshold=1.0, persistence=1)
            m_p2 = evaluate(ag, p2_eps, norm, W=w, rc=DetectorRewardConfig(), decision_threshold=1.0, persistence=1)
            f1_tests.append(m_test["f1"])
            f1_p2s.append(m_p2["f1"])
            clean_fprs.append(m_test["clean_fpr"])
            if m_test["mean_delay_ckpts"] is not None:
                delays.append(m_test["mean_delay_ckpts"])

        ablations.append({
            "ablation_category": "Temporal Window Size",
            "variant": f"Window W={w}",
            "final_test_f1": fmt(agg(f1_tests)),
            "unseen_p2_f1": fmt(agg(f1_p2s)),
            "clean_fpr": fmt(agg(clean_fprs)),
            "mean_delay_ckpts": fmt(agg(delays)),
        })
        print(f"  [Window: W={w}] Final Test F1={fmt(agg(f1_tests))}")

    # 3. Recurrence Ablation: LSTM vs MLP vs Concatenated Window
    for m_type, lbl in [("improved_lstm", "Recurrent Dueling LSTM"), ("improved_mlp", "Non-Recurrent Dueling MLP"), ("mlp", "Original Frame-Only MLP")]:
        use_imp = (m_type != "mlp")
        norm = norm_imp if use_imp else norm_orig
        f_key = "improved" if use_imp else "original"

        tr_eps = parts[f_key]["train"]
        v_eps = parts[f_key]["val"]
        p2_eps = parts[f_key]["unseen_p2"]
        test_eps = parts[f_key]["final_test"]

        f1_tests, f1_p2s, clean_fprs, delays = [], [], [], []
        for s in seeds:
            tr = train_detector(tr_eps, v_eps, norm, m_type, W=4, seed=s, num_episodes=200,
                                lr=1e-3, soft_target_tau=0.05, balanced_sampling=True,
                                decision_threshold=1.0 if use_imp else 0.0, persistence=1)
            ag = tr["agent"]
            m_test = evaluate(ag, test_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=1.0 if use_imp else 0.0, persistence=1)
            m_p2 = evaluate(ag, p2_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=1.0 if use_imp else 0.0, persistence=1)
            f1_tests.append(m_test["f1"])
            f1_p2s.append(m_p2["f1"])
            clean_fprs.append(m_test["clean_fpr"])
            if m_test["mean_delay_ckpts"] is not None:
                delays.append(m_test["mean_delay_ckpts"])

        ablations.append({
            "ablation_category": "Architecture Recurrence",
            "variant": lbl,
            "final_test_f1": fmt(agg(f1_tests)),
            "unseen_p2_f1": fmt(agg(f1_p2s)),
            "clean_fpr": fmt(agg(clean_fprs)),
            "mean_delay_ckpts": fmt(agg(delays)),
        })
        print(f"  [Architecture: {lbl}] Final Test F1={fmt(agg(f1_tests))}")

    # 4. Decision Policy Ablation: Threshold & Persistence
    for th, p_cnt, lbl in [
        (0.0, 1, "Argmax (Threshold=0.0, Persistence=1)"),
        (1.0, 1, "Confidence Margin (Threshold=1.0, Persistence=1)"),
        (2.0, 1, "High Margin (Threshold=2.0, Persistence=1)"),
        (0.0, 2, "Confirmation (Threshold=0.0, Persistence=2)"),
        (1.0, 2, "Combined (Threshold=1.0, Persistence=2)"),
    ]:
        tr_eps = parts["improved"]["train"]
        v_eps = parts["improved"]["val"]
        p2_eps = parts["improved"]["unseen_p2"]
        test_eps = parts["improved"]["final_test"]
        norm = norm_imp

        f1_tests, f1_p2s, clean_fprs, delays = [], [], [], []
        for s in seeds:
            tr = train_detector(tr_eps, v_eps, norm, "improved_lstm", W=4, seed=s, num_episodes=200,
                                lr=1e-3, soft_target_tau=0.05, balanced_sampling=True,
                                decision_threshold=th, persistence=p_cnt)
            ag = tr["agent"]
            m_test = evaluate(ag, test_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=th, persistence=p_cnt)
            m_p2 = evaluate(ag, p2_eps, norm, W=4, rc=DetectorRewardConfig(), decision_threshold=th, persistence=p_cnt)
            f1_tests.append(m_test["f1"])
            f1_p2s.append(m_p2["f1"])
            clean_fprs.append(m_test["clean_fpr"])
            if m_test["mean_delay_ckpts"] is not None:
                delays.append(m_test["mean_delay_ckpts"])

        ablations.append({
            "ablation_category": "Decision Policy & Threshold",
            "variant": lbl,
            "final_test_f1": fmt(agg(f1_tests)),
            "unseen_p2_f1": fmt(agg(f1_p2s)),
            "clean_fpr": fmt(agg(clean_fprs)),
            "mean_delay_ckpts": fmt(agg(delays)),
        })
        print(f"  [Decision Policy: {lbl}] Final Test F1={fmt(agg(f1_tests))}")

    return ablations


# --------------------------------------------------------------------------- Plotting
def generate_plots(R: Dict[str, Any], out_plots_dir: str):
    os.makedirs(out_plots_dir, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    methods = ["orig_lstm", "orig_mlp", "improved_lstm", "improved_mlp", "o1", "o2_online"]
    labels = ["Original\nLSTM-DQN", "Original\nMLP-DQN", "Improved\nLSTM-DQN", "Improved\nMLP-DQN", "Math O1\n(Baseline)", "Math O2\n(Online)"]
    colors = ["#457b9d", "#e76f51", "#1d3557", "#e63946", "#6c757d", "#2a9d8f"]

    # --- 1. F1 Comparison Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35

    f1_test_means, f1_test_stds = [], []
    f1_p2_means, f1_p2_stds = [], []

    for m in methods:
        if m in R["summary_deployable"]:
            s_test = R["summary_deployable"][m]["partitions"]["final_test"]["f1"]
            s_p2 = R["summary_deployable"][m]["partitions"]["unseen_p2"]["f1"]
            f1_test_means.append(s_test["mean"])
            f1_test_stds.append(s_test["std"] if s_test["std"] is not None else 0.0)
            f1_p2_means.append(s_p2["mean"])
            f1_p2_stds.append(s_p2["std"] if s_p2["std"] is not None else 0.0)
        else:
            base_test = R["oracle_baselines"][m]["final_test"]["f1"]
            base_p2 = R["oracle_baselines"][m]["unseen_p2"]["f1"]
            f1_test_means.append(base_test)
            f1_test_stds.append(0.0)
            f1_p2_means.append(base_p2)
            f1_p2_stds.append(0.0)

    rects1 = ax.bar(x - width/2, f1_p2_means, width, yerr=f1_p2_stds, capsize=3, label="Unseen Phase 2 (Seeds 501–506)", color="#8ecae6")
    rects2 = ax.bar(x + width/2, f1_test_means, width, yerr=f1_test_stds, capsize=3, label="Final Test (Seeds 701–706)", color="#023047")

    ax.axhline(0.75, color="#e63946", linestyle="--", lw=1.5, label="Target F1 (0.75)")
    ax.axhline(0.70, color="#f4a261", linestyle=":", lw=1.5, label="Minimum F1 (0.70)")
    ax.set_ylabel("F1-Score")
    ax.set_title("Reward-Hacking Onset Detection F1-Score Benchmark Across Partitions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="lower left", frameon=True)

    for rects in (rects1, rects2):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p1_path = os.path.join(out_plots_dir, "f1_comparison.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"[OK] Saved {p1_path}")

    # --- 2. Confusion Matrices Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, (m, lbl) in enumerate(zip(methods, labels)):
        ax = axes[idx]
        if m in R["summary_deployable"]:
            recs = R["training_runs"][m][0]["final_test_records"]
        else:
            recs = R["oracle_baselines"][m]["final_test"]["records"]

        tp = sum(1 for r in recs if r["outcome"] == "TP")
        early = sum(1 for r in recs if r["outcome"] == "EARLY")
        fn = sum(1 for r in recs if r["outcome"] == "FN")
        tn = sum(1 for r in recs if r["outcome"] == "TN")
        fp = sum(1 for r in recs if r["outcome"] == "FP")

        # 2x2 table: Row 0 = Clean (TN, FP), Row 1 = Hack (FN, TP + EARLY)
        matrix = np.array([[tn, fp], [fn, tp + early]])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=12)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Alarm", "Alarm"])
        ax.set_yticklabels(["Clean", "Hacking"])
        ax.set_title(f"{lbl.replace(chr(10), ' ')}")

        # Cell annotations
        ax.text(0, 0, f"TN={tn}", ha="center", va="center", color="black" if tn < 8 else "white", fontweight="bold")
        ax.text(1, 0, f"FP={fp}", ha="center", va="center", color="red" if fp > 0 else "black", fontweight="bold")
        ax.text(0, 1, f"FN={fn}", ha="center", va="center", color="red" if fn > 0 else "black", fontweight="bold")
        ax.text(1, 1, f"TP={tp}\n(+{early} early)", ha="center", va="center", color="black" if (tp+early) < 8 else "white", fontweight="bold")

    plt.suptitle("Final Test Confusion Matrices (18 Independent Runs)", y=1.02)
    plt.tight_layout()
    p2_path = os.path.join(out_plots_dir, "confusion_matrices.png")
    plt.savefig(p2_path)
    plt.close()
    print(f"[OK] Saved {p2_path}")

    # --- 3. Detection Delay Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    delay_means, delay_stds = [], []
    valid_methods = []

    for m, lbl in zip(methods, labels):
        if m in R["summary_deployable"]:
            d = R["summary_deployable"][m]["partitions"]["final_test"]["mean_delay_ckpts"]
            delay_means.append(d["mean"] if d["mean"] is not None else 0.0)
            delay_stds.append(d["std"] if d["std"] is not None else 0.0)
            valid_methods.append(lbl.replace("\n", " "))
        else:
            d = R["oracle_baselines"][m]["final_test"]["mean_delay_ckpts"]
            delay_means.append(d if d is not None else 0.0)
            delay_stds.append(0.0)
            valid_methods.append(lbl.replace("\n", " "))

    x_d = np.arange(len(valid_methods))
    ax.bar(x_d, delay_means, yerr=delay_stds, capsize=4, color="#2b5c8f", width=0.5)
    ax.set_ylabel("Detection Delay (Checkpoints)")
    ax.set_title("Final Test Mean Detection Delay on True Positives (1 ckpt = 5,000 env steps)")
    ax.set_xticks(x_d)
    ax.set_xticklabels(valid_methods, rotation=25, ha="right")
    ax.set_ylim(0, max(2.5, max(delay_means) + 0.8))

    for i, (m_val, s_val) in enumerate(zip(delay_means, delay_stds)):
        ax.annotate(f"{m_val:.2f} ckpts\n({int(m_val*5000):,} steps)",
                    xy=(i, m_val + (s_val if s_val else 0.0)),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p3_path = os.path.join(out_plots_dir, "detection_delay.png")
    plt.savefig(p3_path)
    plt.close()
    print(f"[OK] Saved {p3_path}")

    # --- 4. Per-Condition Performance Breakdown ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cond_clean, cond_w1, cond_w5 = [], [], []

    for m in methods:
        if m in R["summary_deployable"]:
            recs = R["training_runs"][m][0]["final_test_records"]
        else:
            recs = R["oracle_baselines"][m]["final_test"]["records"]
        
        c_clean = [r for r in recs if r["condition"] == "w0.0"]
        c_w1 = [r for r in recs if r["condition"] == "w1.0"]
        c_w5 = [r for r in recs if r["condition"] == "w5.0"]

        tn_rate = sum(1 for r in c_clean if r["outcome"] == "TN") / len(c_clean) * 100.0 if c_clean else 0.0
        w1_rate = sum(1 for r in c_w1 if r["outcome"] == "TP") / len(c_w1) * 100.0 if c_w1 else 0.0
        w5_rate = sum(1 for r in c_w5 if r["outcome"] == "TP") / len(c_w5) * 100.0 if c_w5 else 0.0

        cond_clean.append(tn_rate)
        cond_w1.append(w1_rate)
        cond_w5.append(w5_rate)

    x_c = np.arange(len(methods))
    w_c = 0.25
    r1 = ax.bar(x_c - w_c, cond_clean, w_c, label="Clean Control (w=0.0) Specificity (TN%)", color="#2a9d8f")
    r2 = ax.bar(x_c, cond_w1, w_c, label="Standard Exploit (w=1.0) Det% (TP%)", color="#f4a261")
    r3 = ax.bar(x_c + w_c, cond_w5, w_c, label="Severe Exploit (w=5.0) Det% (TP%)", color="#e76f51")

    ax.set_ylabel("Rate (%)")
    ax.set_title("Final Test Performance Breakdown by Reward Condition")
    ax.set_xticks(x_c)
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    p4_path = os.path.join(out_plots_dir, "per_condition_performance.png")
    plt.savefig(p4_path)
    plt.close()
    print(f"[OK] Saved {p4_path}")


# --------------------------------------------------------------------------- Tables & CSVs
def write_all_csvs(R: Dict[str, Any], ablations: List[Dict[str, Any]], hyperparams: List[Dict[str, Any]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. improved_summary.csv
    summary_path = os.path.join(out_dir, "improved_summary.csv")
    with open(summary_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "model", "partition", "f1", "precision", "recall",
            "premature_rate", "miss_rate", "clean_fpr",
            "mean_delay_ckpts", "eval_regret_per_run",
            "latency_median_us", "param_count",
        ])
        for mt, s in R["summary_deployable"].items():
            lat = fmt(s["latency_median_us"], 1)
            params = s["params"]
            for part, m in s["partitions"].items():
                w.writerow([
                    mt, part,
                    fmt(m["f1"]), fmt(m["precision"]), fmt(m["recall"]),
                    fmt(m["premature_rate"]), fmt(m["miss_rate"]), fmt(m["clean_fpr"]),
                    fmt(m["mean_delay_ckpts"]), fmt(m["eval_regret_per_run"]),
                    lat, params,
                ])
        for key, parts in R["oracle_baselines"].items():
            for part, m in parts.items():
                w.writerow([
                    key, part,
                    f"{m['f1']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                    f"{m['premature_rate']:.3f}", f"{m['miss_rate']:.3f}", f"{m['clean_fpr']:.3f}",
                    f"{m['mean_delay_ckpts']:.3f}" if m["mean_delay_ckpts"] is not None else "0.000",
                    "0.000", "n/a (oracle)", "0",
                ])
    print(f"[OK] Saved {summary_path}")

    # 2. final_test_run_level.csv
    final_test_path = os.path.join(out_dir, "final_test_run_level.csv")
    with open(final_test_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "train_seed", "run_id", "condition", "true_onset_step", "flag_step", "outcome", "delay_ckpts"])
        for mt, runs in R["training_runs"].items():
            for r in runs:
                for x in r["final_test_records"]:
                    w.writerow([
                        mt, r["seed"], x["run_id"], x["condition"],
                        x["true_onset_step"], x["flag_step"],
                        x["outcome"], x["delay_ckpts"],
                    ])
        for key, parts in R["oracle_baselines"].items():
            for x in parts["final_test"]["records"]:
                w.writerow([
                    key, "-", x["run_id"], x["condition"],
                    x["true_onset_step"], x["flag_step"],
                    x["outcome"], x["delay_ckpts"],
                ])
    print(f"[OK] Saved {final_test_path}")

    # 3. phase2_comparison.csv
    p2_path = os.path.join(out_dir, "phase2_comparison.csv")
    with open(p2_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "train_seed", "run_id", "condition", "true_onset_step", "flag_step", "outcome", "delay_ckpts"])
        for mt, runs in R["training_runs"].items():
            for r in runs:
                for x in r["unseen_p2_records"]:
                    w.writerow([
                        mt, r["seed"], x["run_id"], x["condition"],
                        x["true_onset_step"], x["flag_step"],
                        x["outcome"], x["delay_ckpts"],
                    ])
        for key, parts in R["oracle_baselines"].items():
            for x in parts["unseen_p2"]["records"]:
                w.writerow([
                    key, "-", x["run_id"], x["condition"],
                    x["true_onset_step"], x["flag_step"],
                    x["outcome"], x["delay_ckpts"],
                ])
    print(f"[OK] Saved {p2_path}")

    # 4. ablation_results.csv
    ablation_path = os.path.join(out_dir, "ablation_results.csv")
    with open(ablation_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["ablation_category", "variant", "final_test_f1", "unseen_p2_f1", "clean_fpr", "mean_delay_ckpts"])
        for row in ablations:
            w.writerow([row["ablation_category"], row["variant"], row["final_test_f1"], row["unseen_p2_f1"], row["clean_fpr"], row["mean_delay_ckpts"]])
    print(f"[OK] Saved {ablation_path}")

    # 5. hyperparameter_results.csv
    hyper_path = os.path.join(out_dir, "hyperparameter_results.csv")
    with open(hyper_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["config_name", "model_type", "learning_rate", "soft_target_tau", "balanced_sampling",
                    "decision_threshold", "persistence", "val_f1", "val_precision", "val_recall", "val_clean_fpr", "val_regret", "train_f1"])
        for row in hyperparams:
            w.writerow([row["config_name"], row["model_type"], row["learning_rate"], row["soft_target_tau"],
                        row["balanced_sampling"], row["decision_threshold"], row["persistence"],
                        f"{row['val_f1']:.3f}", f"{row['val_precision']:.3f}", f"{row['val_recall']:.3f}",
                        f"{row['val_clean_fpr']:.3f}", f"{row['val_regret']:.2f}", f"{row['train_f1']:.3f}"])
    print(f"[OK] Saved {hyper_path}")


# --------------------------------------------------------------------------- Main Pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--skip-hyper-search", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  PHASE 3 IMPROVED EXPERIMENTAL PIPELINE")
    print(f"  Target: Genuine Final-Test F1 >= 0.70 (Target: 0.75)")
    print(f"  Output Directory: {OUT_DIR}")
    print("=" * 60)

    # 1. Load dataset split metadata
    split = json.load(open("results/detector/split_metadata.json"))
    p2_meta = json.load(open("results/validation_phase2/validation_dataset_metadata.json"))
    ft_meta = json.load(open("results/phase3_final/final_test_metadata.json"))

    path_map = {**split["run_paths"], **p2_meta["run_paths"], **ft_meta["run_paths"]}

    load_eps = lambda paths, mode: [load_episode_from_run(p, use_independent_ground_truth=True, feature_mode=mode) for p in paths]

    # Partitions for both original (10-dim) and improved (16-dim)
    parts_orig = {
        "train": load_eps([split["run_paths"][r] for r in split["train"]], "original"),
        "val": load_eps([split["run_paths"][r] for r in split["validation"]], "original"),
        "unseen_p2": load_eps(list(p2_meta["run_paths"].values()), "original"),
        "final_test": load_eps(list(ft_meta["run_paths"].values()), "original"),
    }
    parts_imp = {
        "train": load_eps([split["run_paths"][r] for r in split["train"]], "improved"),
        "val": load_eps([split["run_paths"][r] for r in split["validation"]], "improved"),
        "unseen_p2": load_eps(list(p2_meta["run_paths"].values()), "improved"),
        "final_test": load_eps(list(ft_meta["run_paths"].values()), "improved"),
    }
    parts_all = {"original": parts_orig, "improved": parts_imp}

    # Verify run separation across partitions
    train_ids = {e.run_id for e in parts_orig["train"]}
    val_ids = {e.run_id for e in parts_orig["val"]}
    p2_ids = {e.run_id for e in parts_orig["unseen_p2"]}
    ft_ids = {e.run_id for e in parts_orig["final_test"]}
    assert not (train_ids & val_ids), "Train and Val overlap!"
    assert not (train_ids & p2_ids), "Train and Unseen Phase 2 overlap!"
    assert not (train_ids & ft_ids), "Train and Final Test overlap!"
    assert not (val_ids & ft_ids), "Val and Final Test overlap!"
    assert not (p2_ids & ft_ids), "Unseen Phase 2 and Final Test overlap!"
    print(f"[*] Verified strict partition separation: Train={len(train_ids)}, Val={len(val_ids)}, Phase2={len(p2_ids)}, FinalTest={len(ft_ids)}")

    # 2. Normalizers
    norm_orig = Normalizer.load("results/detector/norm_stats.json")
    norm_imp = Normalizer.load("results/phase3_improved/norm_stats.json")

    # 3. Mathematical Baselines
    print("\n[*] Evaluating Mathematical Baselines (O1, O2 Retrospective, O2 Online)...")
    oracle_baselines = evaluate_mathematical_baselines(parts_orig, path_map)
    for k in oracle_baselines:
        print(f"  [{k}] Final Test F1={oracle_baselines[k]['final_test']['f1']:.3f}, Recall={oracle_baselines[k]['final_test']['recall']:.3f}, Clean FPR={oracle_baselines[k]['final_test']['clean_fpr']:.3f}")

    # 4. Hyperparameter Search on Train and Val ONLY
    if not args.skip_hyper_search:
        hyper_results, best_lstm_cfg, best_mlp_cfg = run_hyperparameter_search(parts_imp["train"], parts_imp["val"], norm_imp)
    else:
        best_lstm_cfg = {"learning_rate": 1e-3, "soft_target_tau": 0.05, "balanced_sampling": True, "decision_threshold": 1.0, "persistence": 1}
        best_mlp_cfg = {"learning_rate": 1e-3, "soft_target_tau": 0.05, "balanced_sampling": True, "decision_threshold": 1.0, "persistence": 1}
        hyper_results = []

    # 5. Ablation Studies
    ablations = run_ablation_study(parts_all, norm_orig, norm_imp, best_lstm_cfg, seeds=[0, 1, 2])

    # 6. Multi-Seed Training of Deployable Detectors
    print("\n" + "=" * 60)
    print(f"  MULTI-SEED TRAINING ({len(args.train_seeds)} seeds: {args.train_seeds})")
    print("=" * 60)

    training_runs = {
        "orig_lstm": [],
        "orig_mlp": [],
        "improved_lstm": [],
        "improved_mlp": [],
    }

    rc0 = DetectorRewardConfig()

    for s in args.train_seeds:
        # A. Original LSTM-DQN
        print(f"[Train] Original LSTM seed={s}...")
        tr_o_lstm = train_detector(
            train_eps=parts_orig["train"], val_eps=parts_orig["val"], normalizer=norm_orig,
            model_type="lstm", W=4, rc=rc0, seed=s, num_episodes=args.episodes,
            lr=1e-3, soft_target_tau=None, balanced_sampling=False, decision_threshold=0.0, persistence=1,
            save_path=os.path.join(MODELS_DIR, f"orig_lstm_seed{s}.pt"),
        )
        ag_o_lstm = tr_o_lstm.pop("agent")
        evals_o_lstm = {k: evaluate(ag_o_lstm, v, norm_orig, 4, rc0, decision_threshold=0.0, persistence=1) for k, v in parts_orig.items()}
        tr_o_lstm.update({
            "seed": s, "params": count_params(ag_o_lstm), "latency": inference_latency_us(ag_o_lstm, 4, 10),
            "eval": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in evals_o_lstm.items()},
            "final_test_records": evals_o_lstm["final_test"]["records"],
            "unseen_p2_records": evals_o_lstm["unseen_p2"]["records"],
        })
        training_runs["orig_lstm"].append(tr_o_lstm)

        # B. Original MLP-DQN
        print(f"[Train] Original MLP seed={s}...")
        tr_o_mlp = train_detector(
            train_eps=parts_orig["train"], val_eps=parts_orig["val"], normalizer=norm_orig,
            model_type="mlp", W=4, rc=rc0, seed=s, num_episodes=args.episodes,
            lr=1e-3, soft_target_tau=None, balanced_sampling=False, decision_threshold=0.0, persistence=1,
            save_path=os.path.join(MODELS_DIR, f"orig_mlp_seed{s}.pt"),
        )
        ag_o_mlp = tr_o_mlp.pop("agent")
        evals_o_mlp = {k: evaluate(ag_o_mlp, v, norm_orig, 4, rc0, decision_threshold=0.0, persistence=1) for k, v in parts_orig.items()}
        tr_o_mlp.update({
            "seed": s, "params": count_params(ag_o_mlp), "latency": inference_latency_us(ag_o_mlp, 4, 10),
            "eval": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in evals_o_mlp.items()},
            "final_test_records": evals_o_mlp["final_test"]["records"],
            "unseen_p2_records": evals_o_mlp["unseen_p2"]["records"],
        })
        training_runs["orig_mlp"].append(tr_o_mlp)

        # C. Improved LSTM-DQN
        print(f"[Train] Improved LSTM seed={s}...")
        tr_i_lstm = train_detector(
            train_eps=parts_imp["train"], val_eps=parts_imp["val"], normalizer=norm_imp,
            model_type="improved_lstm", W=4, rc=rc0, seed=s, num_episodes=args.episodes,
            lr=best_lstm_cfg.get("learning_rate", 1e-3),
            soft_target_tau=best_lstm_cfg.get("soft_target_tau", 0.05),
            balanced_sampling=best_lstm_cfg.get("balanced_sampling", True),
            decision_threshold=best_lstm_cfg.get("decision_threshold", 1.0),
            persistence=best_lstm_cfg.get("persistence", 1),
            save_path=os.path.join(MODELS_DIR, f"improved_lstm_seed{s}.pt"),
        )
        ag_i_lstm = tr_i_lstm.pop("agent")
        evals_i_lstm = {k: evaluate(ag_i_lstm, v, norm_imp, 4, rc0,
                                    decision_threshold=best_lstm_cfg.get("decision_threshold", 1.0),
                                    persistence=best_lstm_cfg.get("persistence", 1)) for k, v in parts_imp.items()}
        tr_i_lstm.update({
            "seed": s, "params": count_params(ag_i_lstm), "latency": inference_latency_us(ag_i_lstm, 4, 16),
            "eval": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in evals_i_lstm.items()},
            "final_test_records": evals_i_lstm["final_test"]["records"],
            "unseen_p2_records": evals_i_lstm["unseen_p2"]["records"],
        })
        training_runs["improved_lstm"].append(tr_i_lstm)

        # D. Improved MLP-DQN
        print(f"[Train] Improved MLP seed={s}...")
        tr_i_mlp = train_detector(
            train_eps=parts_imp["train"], val_eps=parts_imp["val"], normalizer=norm_imp,
            model_type="improved_mlp", W=4, rc=rc0, seed=s, num_episodes=args.episodes,
            lr=best_mlp_cfg.get("learning_rate", 1e-3),
            soft_target_tau=best_mlp_cfg.get("soft_target_tau", 0.05),
            balanced_sampling=best_mlp_cfg.get("balanced_sampling", True),
            decision_threshold=best_mlp_cfg.get("decision_threshold", 1.0),
            persistence=best_mlp_cfg.get("persistence", 1),
            save_path=os.path.join(MODELS_DIR, f"improved_mlp_seed{s}.pt"),
        )
        ag_i_mlp = tr_i_mlp.pop("agent")
        evals_i_mlp = {k: evaluate(ag_i_mlp, v, norm_imp, 4, rc0,
                                   decision_threshold=best_mlp_cfg.get("decision_threshold", 1.0),
                                   persistence=best_mlp_cfg.get("persistence", 1)) for k, v in parts_imp.items()}
        tr_i_mlp.update({
            "seed": s, "params": count_params(ag_i_mlp), "latency": inference_latency_us(ag_i_mlp, 4, 16),
            "eval": {k: {k2: v2 for k2, v2 in v.items() if k2 != "records"} for k, v in evals_i_mlp.items()},
            "final_test_records": evals_i_mlp["final_test"]["records"],
            "unseen_p2_records": evals_i_mlp["unseen_p2"]["records"],
        })
        training_runs["improved_mlp"].append(tr_i_mlp)

    # 7. Aggregate Summary
    summary_deployable = {}
    for mt, runs in training_runs.items():
        summary_deployable[mt] = {
            "params": runs[0]["params"],
            "latency_median_us": agg([r["latency"]["median_us"] for r in runs]),
            "train_wall_s": agg([r["train_wall_s"] for r in runs]),
            "cum_train_regret": agg([r["cum_train_regret"] for r in runs]),
            "partitions": {},
        }
        for part in ("train", "val", "unseen_p2", "final_test"):
            summary_deployable[mt]["partitions"][part] = {
                k: agg([r["eval"][part][k] for r in runs])
                for k in ("f1", "precision", "recall", "premature_rate", "miss_rate", "clean_fpr",
                          "mean_delay_ckpts", "eval_regret_per_run")
            }

    # 8. Forensic Leakage Audit on Improved Features
    print("\n[*] Running Forensic Leakage Audit on Improved Pipeline...")
    audit_report = run_leakage_audit(
        norm_path="results/phase3_improved/norm_stats.json",
        output_report_path="results/phase3_improved/leakage_audit_report.json",
        feature_mode="improved",
    )

    # 9. Complete Experiment Metadata & Results Object
    full_results = {
        "metadata": {
            "title": "Improved Reward-Hacking Onset Detection",
            "date": "2026-11-02",
            "features_dimension": 16,
            "feature_names": IMPROVED_FEATURE_NAMES,
            "train_seeds": args.train_seeds,
            "episodes": args.episodes,
            "best_lstm_config": best_lstm_cfg,
            "best_mlp_config": best_mlp_cfg,
        },
        "summary_deployable": summary_deployable,
        "oracle_baselines": oracle_baselines,
        "training_runs": training_runs,
        "ablation_results": ablations,
        "hyperparameter_results": hyper_results,
        "leakage_audit_summary": audit_report,
    }

    # Save experiment_metadata.json
    with open(os.path.join(OUT_DIR, "experiment_metadata.json"), "w") as fp:
        json.dump(full_results, fp, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(f"[OK] Saved {os.path.join(OUT_DIR, 'experiment_metadata.json')}")

    # 10. Write all CSV tables
    print("\n[*] Generating CSV Tables...")
    write_all_csvs(full_results, ablations, hyper_results, OUT_DIR)

    # 11. Generate Publication Plots
    print("\n[*] Generating Publication-Quality Plots...")
    generate_plots(full_results, PLOTS_DIR)

    # 12. Print Final Summary Table to Console
    print("\n" + "=" * 80)
    print("  FINAL TEST BENCHMARK SUMMARY (18 Unseen Runs, Seeds 701–706)")
    print("=" * 80)
    print(f"{'Method':<22} | {'F1-Score':<18} | {'Precision':<12} | {'Recall':<12} | {'Clean FPR':<10} | {'Delay (ckpts)':<14}")
    print("-" * 80)
    for mt in ("orig_lstm", "orig_mlp", "improved_lstm", "improved_mlp"):
        s = summary_deployable[mt]["partitions"]["final_test"]
        print(f"{mt:<22} | {fmt(s['f1']):<18} | {fmt(s['precision'], 3):<12} | {fmt(s['recall'], 3):<12} | {fmt(s['clean_fpr'], 3):<10} | {fmt(s['mean_delay_ckpts'], 2):<14}")
    for key in ("o1", "o2_online", "o2_retrospective"):
        s = oracle_baselines[key]["final_test"]
        d_str = f"{s['mean_delay_ckpts']:.2f}" if s["mean_delay_ckpts"] is not None else "0.00"
        print(f"{key:<22} | {s['f1']:.3f}{' ':<13} | {s['precision']:.3f}{' ':<7} | {s['recall']:.3f}{' ':<7} | {s['clean_fpr']:.3f}{' ':<5} | {d_str:<14}")
    print("=" * 80)


if __name__ == "__main__":
    main()
