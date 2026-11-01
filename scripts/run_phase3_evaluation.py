"""
run_phase3_evaluation.py

Phase 3: completes the evaluation of the learned onset detectors.
Writes ONLY to results/phase3_final/ (Phase 1 and Phase 2 outputs are read-only inputs).

Partitions (run-level, disjoint):
  train      : Phase 1 train split (6 runs)        -> gradient updates + normalizer
  val        : Phase 1 validation split (3 runs)   -> model selection (best val F1)
  unseen_p2  : Phase 2 corpus, seeds 501-506 (18)  -> generalization + sensitivity analysis
  final_test : seeds 701-706 (18)                  -> evaluated ONCE per trained model with the
                                                      baseline config; never used for selection.

Consistent run-level outcome definitions (used for every method):
  hacking run : TP    = first flag at index t >= t*        (successful detection)
                EARLY = first flag at index t <  t*        (premature alarm; NOT a detection)
                FN    = never flags
  clean run   : TN    = never flags
                FP    = flags at any time
  detection_rate (= recall) = TP / N_hack
  premature_rate            = EARLY / N_hack
  miss_rate                 = FN / N_hack
  precision                 = TP / (TP + EARLY + FP)     (every alarm that was not a correct detection is a false alarm)
  F1                        = 2PR / (P + R)
  clean_fpr                 = FP / N_clean
  delay                     = t_flag - t* over TP runs only (checkpoints; x5000 env steps)
"""

import os
import sys
import csv
import json
import time
import copy
import random
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import (
    load_episode_from_run, Normalizer, DetectorRewardConfig, DetectorEpisode, DetectorEnv, FEATURE_NAMES,
)
from rhd.detectors.agent import DQNAgent, DQNConfig
from rhd.detectors.replay_buffer import ReplayBuffer

OUT = "results/phase3_final"
F1_THRESHOLD = 0.8          # predefined convergence threshold on validation F1
FEATURE_GROUPS = {
    "proxy_reward": [0, 1, 2],
    "episode_length": [3],
    "policy_drift": [4, 6],
    "spatial_behavior": [5, 7, 8, 9],
}


# --------------------------------------------------------------------------- helpers
class MaskedNormalizer(Normalizer):
    """Normalizer that zeroes selected (normalized) feature columns -> feature-group ablation."""

    def __init__(self, base: Normalizer, masked_cols: List[int]):
        super().__init__(base.mean, base.std)
        self.masked_cols = list(masked_cols)

    def transform(self, features):
        x = super().transform(features).copy()
        if self.masked_cols:
            x[:, self.masked_cols] = 0.0
        return x


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
    delays = [x["delay_ckpts"] for x in records if x["outcome"] == "TP"]
    out = dict(c)
    out.update({
        "n_runs": len(records), "n_hack": n_hack, "n_clean": n_clean,
        "precision": p, "recall": r, "detection_rate": r, "f1": f1,
        "premature_rate": c["EARLY"] / n_hack if n_hack else 0.0,
        "miss_rate": c["FN"] / n_hack if n_hack else 0.0,
        "clean_fpr": c["FP"] / n_clean if n_clean else 0.0,
        "mean_delay_ckpts": float(np.mean(delays)) if delays else None,
        "std_delay_ckpts": float(np.std(delays)) if delays else None,
        "n_delays": len(delays),
    })
    out["per_condition"] = {}
    for cond in ("w0.0", "w1.0", "w5.0"):
        sub = [x for x in records if condition_of(x["run_id"]) == cond]
        if sub:
            out["per_condition"][cond] = {k: sum(1 for x in sub if x["outcome"] == k)
                                          for k in ("TP", "EARLY", "FN", "TN", "FP")}
    return out


def oracle_return(ep: DetectorEpisode, rc: DetectorRewardConfig, W: int) -> float:
    """Return of the reference (oracle) stopping policy: flag exactly at t* on hacking runs, never on clean runs."""
    env = DetectorEnv([ep], window_size=W, reward_config=rc)
    env.reset(episode_idx=0)
    total, done = 0.0, False
    while not done:
        a = DetectorEnv.ACTION_FLAG if (not ep.is_clean and env.current_step == ep.ground_truth_onset_idx) else DetectorEnv.ACTION_WAIT
        _, rwd, done, _ = env.step(a)
        total += rwd
    return total


def evaluate(agent: DQNAgent, episodes: List[DetectorEpisode], normalizer, W: int,
             rc: DetectorRewardConfig) -> Dict[str, Any]:
    env = DetectorEnv(episodes, window_size=W, reward_config=rc, normalizer=normalizer)
    records, regret = [], 0.0
    for i, ep in enumerate(episodes):
        s, _ = env.reset(episode_idx=i)
        done, flag_idx, ret = False, None, 0.0
        while not done:
            t = env.current_step
            a = agent.select_action(s, deterministic=True)
            if a == DetectorEnv.ACTION_FLAG:
                flag_idx = t
            s, rwd, done, _ = env.step(a)
            ret += rwd
        outcome = classify(ep, flag_idx)
        regret += oracle_return(ep, rc, W) - ret
        records.append({
            "run_id": ep.run_id, "condition": condition_of(ep.run_id),
            "true_onset_idx": ep.ground_truth_onset_idx, "true_onset_step": ep.ground_truth_onset_step,
            "flag_idx": flag_idx, "flag_step": ep.checkpoint_steps[flag_idx] if flag_idx is not None else None,
            "outcome": outcome,
            "delay_ckpts": (flag_idx - ep.ground_truth_onset_idx) if outcome == "TP" else None,
            "return": ret,
        })
    m = metrics_from_records(records)
    m["eval_regret_total"] = regret
    m["eval_regret_per_run"] = regret / len(episodes)
    m["records"] = records
    return m


def count_params(agent: DQNAgent) -> int:
    return int(sum(p.numel() for p in agent.online_net.parameters()))


def inference_latency_us(agent: DQNAgent, W: int, n: int = 2000) -> Dict[str, float]:
    x = np.random.randn(W, 10).astype(np.float32)
    for _ in range(50):
        agent.select_action(x, deterministic=True)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        agent.select_action(x, deterministic=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    return {"median_us": float(np.median(ts)), "p95_us": float(np.percentile(ts, 95)), "mean_us": float(np.mean(ts))}


def peak_rss_mb() -> Optional[float]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        v = getattr(mi, "peak_wset", None)  # Windows only
        return v / 2**20 if v else mi.rss / 2**20
    except Exception:
        return None


# --------------------------------------------------------------------------- training
def train_one(train_eps, val_eps, normalizer, model_type="lstm", W=4, rc=None, seed=0,
              num_episodes=350, eval_every=10, warmup=64, save_path=None) -> Dict[str, Any]:
    rc = rc or DetectorRewardConfig()
    set_seed(seed)
    agent = DQNAgent(config=DQNConfig(model_type=model_type))
    buf = ReplayBuffer(capacity=10000)
    env = DetectorEnv(train_eps, window_size=W, reward_config=rc, normalizer=normalizer)
    oracle_cache = {ep.run_id: oracle_return(ep, rc, W) for ep in train_eps}

    curve, cum_regret, interactions, grad_updates = [], 0.0, 0, 0
    best_f1, best_state, best_ep, best_inter = -1.0, None, None, None
    t_start = time.perf_counter()
    train_time_excl_eval = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for ep_i in range(1, num_episodes + 1):
        t0 = time.perf_counter()
        s, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            a = agent.select_action(s, deterministic=False)
            s2, rwd, done, _ = env.step(a)
            buf.push(s, a, rwd, s2, done)
            s, ep_ret = s2, ep_ret + rwd
            interactions += 1
            if len(buf) >= warmup:
                if agent.update(buf) is not None:
                    grad_updates += 1
        cum_regret += oracle_cache[env.current_episode.run_id] - ep_ret
        train_time_excl_eval += time.perf_counter() - t0

        if ep_i % eval_every == 0 or ep_i == num_episodes:
            vm = evaluate(agent, val_eps, normalizer, W, rc)
            tm = evaluate(agent, train_eps, normalizer, W, rc)
            curve.append({"episode": ep_i, "interactions": interactions, "grad_updates": grad_updates,
                          "val_f1": vm["f1"], "train_f1": tm["f1"], "cum_train_regret": cum_regret,
                          "epsilon": agent.epsilon})
            if vm["f1"] > best_f1:   # strict '>' -> earliest best checkpoint (same rule as Phase 1 trainer)
                best_f1, best_ep, best_inter = vm["f1"], ep_i, interactions
                best_state = copy.deepcopy(agent.online_net.state_dict())

    wall = time.perf_counter() - t_start
    agent.online_net.load_state_dict(best_state)
    if save_path:
        agent.save(save_path)

    conv = next((c for c in curve if c["val_f1"] >= F1_THRESHOLD), None)
    return {
        "agent": agent, "curve": curve, "best_val_f1": best_f1,
        "selected_episode": best_ep, "selected_interactions": best_inter,
        "converged": conv is not None,
        "episodes_to_threshold": conv["episode"] if conv else None,
        "interactions_to_threshold": conv["interactions"] if conv else None,
        "total_interactions": interactions, "grad_updates": grad_updates,
        "cum_train_regret": cum_regret,
        "train_wall_s": wall, "train_time_excl_eval_s": train_time_excl_eval,
        "cuda_peak_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
    }


# --------------------------------------------------------------------------- O1 / O2 (oracle baselines)
def oracle_baseline_records(episodes, run_paths, key):
    recs = []
    for ep in episodes:
        with open(os.path.join(run_paths[ep.run_id], "features", "onset.json")) as fp:
            d = json.load(fp)
        flag_idx = None
        if d.get(f"{key}_detected") and d.get(f"{key}_onset_step") is not None:
            st = int(d[f"{key}_onset_step"])
            flag_idx = next((i for i, s in enumerate(ep.checkpoint_steps) if s >= st), None)
        outcome = classify(ep, flag_idx)
        recs.append({"run_id": ep.run_id, "condition": condition_of(ep.run_id),
                     "true_onset_idx": ep.ground_truth_onset_idx, "true_onset_step": ep.ground_truth_onset_step,
                     "flag_idx": flag_idx, "flag_step": ep.checkpoint_steps[flag_idx] if flag_idx is not None else None,
                     "outcome": outcome,
                     "delay_ckpts": (flag_idx - ep.ground_truth_onset_idx) if outcome == "TP" else None})
    m = metrics_from_records(recs)
    m["records"] = recs
    return m


def agg(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def strip(m):
    return {k: v for k, v in m.items() if k != "records"}


# --------------------------------------------------------------------------- main
def main(train_seeds, sens_seeds, num_episodes):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)

    split = json.load(open("results/detector/split_metadata.json"))
    p2 = json.load(open("results/validation_phase2/validation_dataset_metadata.json"))
    ft = json.load(open(os.path.join(OUT, "final_test_metadata.json")))
    normalizer = Normalizer.load("results/detector/norm_stats.json")

    load = lambda paths: [load_episode_from_run(p, use_independent_ground_truth=True) for p in paths]
    parts = {
        "train": load([split["run_paths"][r] for r in split["train"]]),
        "val": load([split["run_paths"][r] for r in split["validation"]]),
        "unseen_p2": load(list(p2["run_paths"].values())),
        "final_test": load(list(ft["run_paths"].values())),
    }
    for k, v in parts.items():
        assert all(e is not None for e in v), f"failed loading {k}"
    ids = {k: {e.run_id for e in v} for k, v in parts.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (ids[names[i]] & ids[names[j]]), f"run overlap {names[i]} / {names[j]}"
    partition_summary = {k: {"n_runs": len(v), "n_checkpoints": int(sum(e.total_checkpoints for e in v)),
                             "n_clean": sum(e.is_clean for e in v), "n_hack": sum(not e.is_clean for e in v),
                             "runs": sorted(ids[k])} for k, v in parts.items()}

    rc0 = DetectorRewardConfig()
    results = {"metric_definitions": __doc__, "f1_convergence_threshold": F1_THRESHOLD,
               "partitions": partition_summary, "baseline_reward_config": rc0.to_dict(),
               "train_seeds": train_seeds, "device": "cuda" if torch.cuda.is_available() else "cpu"}

    # ---------------- 1. Phase 2 metric reconciliation (recomputed from saved Phase 2 predictions)
    p2_report = json.load(open("results/validation_phase2/phase2_validation_report.json"))
    recon = {}
    for meth, r in p2_report.items():
        recs = []
        for pr in r["predictions"]:
            o = {"TP": "TP", "FN": "FN", "TN": "TN", "FP": "FP", "FP_early": "EARLY"}[pr["classification"]]
            recs.append({"run_id": pr["run_id"], "outcome": o, "delay_ckpts": pr["delay_ckpts"]})
        recon[meth] = {"phase2_reported": {"precision": r["precision"], "recall": r["recall"], "f1": r["f1_score"]},
                       "corrected": strip(metrics_from_records(recs))}
    results["phase2_reconciliation"] = recon

    # ---------------- 2-6. Multi-seed training of deployable detectors (baseline config)
    main_runs = {}
    for mt in ("lstm", "mlp"):
        main_runs[mt] = []
        for s in train_seeds:
            print(f"[train] {mt} seed={s}", flush=True)
            rss_before = peak_rss_mb()
            tr = train_one(parts["train"], parts["val"], normalizer, model_type=mt, W=4, rc=rc0, seed=s,
                           num_episodes=num_episodes, save_path=os.path.join(OUT, "models", f"{mt}_seed{s}.pt"))
            ag = tr.pop("agent")
            evals = {k: evaluate(ag, v, normalizer, 4, rc0) for k, v in parts.items()}
            tr.update({
                "seed": s, "params": count_params(ag), "latency": inference_latency_us(ag, 4),
                "process_peak_rss_mb_before": rss_before, "process_peak_rss_mb_after": peak_rss_mb(),
                "eval": {k: strip(v) for k, v in evals.items()},
                "final_test_records": evals["final_test"]["records"],
                "unseen_p2_records": evals["unseen_p2"]["records"],
            })
            main_runs[mt].append(tr)
    results["training_runs"] = main_runs

    summary = {}
    for mt, runs in main_runs.items():
        summary[mt] = {
            "params": runs[0]["params"],
            "train_wall_s": agg([r["train_wall_s"] for r in runs]),
            "train_time_excl_eval_s": agg([r["train_time_excl_eval_s"] for r in runs]),
            "latency_median_us": agg([r["latency"]["median_us"] for r in runs]),
            "converged_fraction": sum(r["converged"] for r in runs) / len(runs),
            "episodes_to_threshold": agg([r["episodes_to_threshold"] for r in runs]),
            "interactions_to_threshold": agg([r["interactions_to_threshold"] for r in runs]),
            "selected_episode": agg([r["selected_episode"] for r in runs]),
            "cum_train_regret": agg([r["cum_train_regret"] for r in runs]),
            "partitions": {},
        }
        for part in parts:
            summary[mt]["partitions"][part] = {
                k: agg([r["eval"][part][k] for r in runs])
                for k in ("f1", "precision", "recall", "premature_rate", "miss_rate", "clean_fpr",
                          "mean_delay_ckpts", "eval_regret_per_run")
            }
        summary[mt]["generalization_gap_f1"] = {
            "train_minus_val": agg([r["eval"]["train"]["f1"] - r["eval"]["val"]["f1"] for r in runs]),
            "train_minus_unseen_p2": agg([r["eval"]["train"]["f1"] - r["eval"]["unseen_p2"]["f1"] for r in runs]),
            "train_minus_final_test": agg([r["eval"]["train"]["f1"] - r["eval"]["final_test"]["f1"] for r in runs]),
            "val_minus_final_test": agg([r["eval"]["val"]["f1"] - r["eval"]["final_test"]["f1"] for r in runs]),
        }
    results["summary_deployable"] = summary

    # ---------------- 10. Oracle baselines (require privileged true-reward / task information)
    path_map = {**split["run_paths"], **p2["run_paths"], **ft["run_paths"]}
    results["oracle_baselines"] = {}
    for key in ("o1", "o2"):
        results["oracle_baselines"][key] = {}
        for part, eps in parts.items():
            m = oracle_baseline_records(eps, path_map, key)
            results["oracle_baselines"][key][part] = m

    # ---------------- 8. Sensitivity analysis (LSTM; evaluated on val + unseen_p2 only, never final_test)
    sens_cfgs = {"baseline": {}}
    for v in (-5.0, -20.0):
        sens_cfgs[f"r_fp={v}"] = {"rc": {"r_fp": v}}
    for v in (-7.5, -30.0):
        sens_cfgs[f"r_fn={v}"] = {"rc": {"r_fn": v}}
    for v in (-0.5, -2.0):
        sens_cfgs[f"r_delay={v}"] = {"rc": {"r_delay": v}}
    for w in (1, 2, 6):
        sens_cfgs[f"window={w}"] = {"W": w}
    for g in FEATURE_GROUPS:
        sens_cfgs[f"mask:{g}"] = {"mask": FEATURE_GROUPS[g]}

    sens = {}
    for name, cfg in sens_cfgs.items():
        rc = DetectorRewardConfig(**{**rc0.to_dict(), **cfg.get("rc", {})})
        W = cfg.get("W", 4)
        norm = MaskedNormalizer(normalizer, cfg["mask"]) if "mask" in cfg else normalizer
        per_seed = []
        for s in sens_seeds:
            print(f"[sens] {name} seed={s}", flush=True)
            tr = train_one(parts["train"], parts["val"], norm, "lstm", W, rc, s, num_episodes)
            ag = tr.pop("agent")
            # metrics always scored with the SAME outcome definitions; regret uses that config's reward
            u = evaluate(ag, parts["unseen_p2"], norm, W, rc)
            per_seed.append({"seed": s, "val_f1": tr["best_val_f1"], "unseen_p2": strip(u),
                             "episodes_to_threshold": tr["episodes_to_threshold"]})
        sens[name] = {
            "config": {"reward": rc.to_dict(), "window": W, "masked_features": [FEATURE_NAMES[i] for i in cfg.get("mask", [])]},
            "per_seed": per_seed,
            **{f"unseen_p2_{k}": agg([p["unseen_p2"][k] for p in per_seed])
               for k in ("f1", "recall", "premature_rate", "miss_rate", "clean_fpr")},
            "val_f1": agg([p["val_f1"] for p in per_seed]),
        }
    results["sensitivity"] = sens

    with open(os.path.join(OUT, "phase3_results.json"), "w") as fp:
        json.dump(results, fp, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    write_tables(results)
    print("[OK] results/phase3_final/phase3_results.json")


def fmt(a, d=3):
    return "n/a" if a["mean"] is None else f"{a['mean']:.{d}f} ± {a['std']:.{d}f} (n={a['n']})"


def write_tables(R):
    # summary CSV across partitions
    with open(os.path.join(OUT, "deployable_summary.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "partition", "f1", "precision", "recall", "premature_rate", "miss_rate", "clean_fpr",
                    "mean_delay_ckpts", "eval_regret_per_run"])
        for mt, s in R["summary_deployable"].items():
            for part, m in s["partitions"].items():
                w.writerow([mt, part] + [fmt(m[k]) for k in ("f1", "precision", "recall", "premature_rate",
                                                             "miss_rate", "clean_fpr", "mean_delay_ckpts",
                                                             "eval_regret_per_run")])
    with open(os.path.join(OUT, "oracle_baselines.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["method", "partition", "TP", "EARLY", "FN", "TN", "FP", "precision", "recall", "f1", "clean_fpr", "mean_delay_ckpts"])
        for key, parts in R["oracle_baselines"].items():
            for part, m in parts.items():
                w.writerow([key, part, m["TP"], m["EARLY"], m["FN"], m["TN"], m["FP"],
                            f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                            f"{m['clean_fpr']:.3f}", m["mean_delay_ckpts"]])
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["config", "val_f1", "unseen_p2_f1", "unseen_p2_recall", "unseen_p2_premature_rate",
                    "unseen_p2_miss_rate", "unseen_p2_clean_fpr"])
        for name, s in R["sensitivity"].items():
            w.writerow([name, fmt(s["val_f1"]), fmt(s["unseen_p2_f1"]), fmt(s["unseen_p2_recall"]),
                        fmt(s["unseen_p2_premature_rate"]), fmt(s["unseen_p2_miss_rate"]), fmt(s["unseen_p2_clean_fpr"])])
    with open(os.path.join(OUT, "final_test_run_level.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "train_seed", "run_id", "condition", "true_onset_step", "flag_step", "outcome", "delay_ckpts"])
        for mt, runs in R["training_runs"].items():
            for r in runs:
                for x in r["final_test_records"]:
                    w.writerow([mt, r["seed"], x["run_id"], x["condition"], x["true_onset_step"], x["flag_step"],
                                x["outcome"], x["delay_ckpts"]])
        for key, parts in R["oracle_baselines"].items():
            for x in parts["final_test"]["records"]:
                w.writerow([key, "-", x["run_id"], x["condition"], x["true_onset_step"], x["flag_step"],
                            x["outcome"], x["delay_ckpts"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sens-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=350)
    a = ap.parse_args()
    main(a.train_seeds, a.sens_seeds, a.episodes)
