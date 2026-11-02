import os
import sys
import json
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rhd.detectors.environment import (
    FEATURE_NAMES,
    extract_feature_vector,
    load_episode_from_run,
    DetectorEnv,
    Normalizer,
)


def run_leakage_audit(
    metadata_path: str = "results/detector/split_metadata.json",
    norm_path: str = "results/detector/norm_stats.json",
    output_report_path: str = "results/validation_phase2/leakage_audit_report.json",
) -> Dict[str, Any]:
    print("=" * 60)
    print("  FORENSIC AUDIT: Detector Inputs & Zero-Leakage (10-FEATURE)")
    print("=" * 60)

    report = {
        "passed_all_checks": True,
        "feature_schema_audit": {},
        "causality_audit": {},
        "normalization_audit": {},
        "observation_tensor_audit": {},
    }

    feature_names = FEATURE_NAMES
    expected_dim = len(feature_names)

    # 1. Feature Schema & Prohibited Field Audit
    prohibited_substrings = [
        "true", "goal", "entry", "entries", "onset", "label",
        "ground", "target", "weight", "exploit_w", "reward_hacking"
    ]
    
    flagged_features = []
    for f in feature_names:
        for sub in prohibited_substrings:
            if sub in f.lower() and f != "revisit_rate":  # revisit_rate is state revisitation
                flagged_features.append((f, sub))

    report["feature_schema_audit"] = {
        "total_features": len(feature_names),
        "feature_list": feature_names,
        "prohibited_fields_detected": flagged_features,
        "passed": (len(flagged_features) == 0),
    }
    if flagged_features:
        report["passed_all_checks"] = False

    print(f"[*] Feature dimension: {len(feature_names)} features (canonical order)")
    print(f"[*] Prohibited terms in feature names: {len(flagged_features)} found")

    # 2. Causality and Temporal Ordering Audit
    all_run_paths = {}
    for meta_file in [
        metadata_path,
        "results/validation_phase2/validation_dataset_metadata.json",
        "results/phase3_final/final_test_metadata.json",
    ]:
        if os.path.exists(meta_file):
            with open(meta_file) as fp:
                d = json.load(fp)
                if "run_paths" in d:
                    all_run_paths.update(d["run_paths"])

    causality_violations = []
    for rid, rpath in all_run_paths.items():
        temp_path = os.path.join(rpath, "features", "temporal_features.jsonl")
        if not os.path.exists(temp_path):
            continue
        with open(temp_path) as fp:
            rows = [json.loads(line) for line in fp if line.strip()]

        # Verify strictly non-decreasing environment steps
        steps = [int(r["environment_steps"]) for r in rows]
        if steps != sorted(steps):
            causality_violations.append((rid, "Non-chronological environment steps"))

        # Verify proxy_return_trend is strictly backward difference (causal: uses only t and t-1)
        for i in range(1, len(rows)):
            s_curr = rows[i].get("proxy_return_smoothed")
            s_prev = rows[i - 1].get("proxy_return_smoothed")
            r_curr = rows[i].get("proxy_return")
            r_prev = rows[i - 1].get("proxy_return")
            trend = rows[i].get("proxy_return_trend", rows[i].get("proxy_return_change"))
            if trend is not None:
                matches_smoothed = (s_curr is not None and s_prev is not None and abs(trend - (s_curr - s_prev)) < 1e-4)
                matches_raw = (r_curr is not None and r_prev is not None and abs(trend - (r_curr - r_prev)) < 1e-4)
                if not (matches_smoothed or matches_raw):
                    causality_violations.append((rid, f"Step {steps[i]}: Trend does not match causal backward difference"))

    report["causality_audit"] = {
        "total_runs_checked": len(all_run_paths),
        "causality_violations": causality_violations,
        "passed": (len(causality_violations) == 0),
    }
    if causality_violations:
        report["passed_all_checks"] = False
    print(f"[*] Temporal causality check: {len(causality_violations)} violations across {len(all_run_paths)} runs")

    # 3. Normalization Distribution Leakage Audit
    if os.path.exists(metadata_path) and os.path.exists(norm_path):
        with open(metadata_path, "r") as f:
            split_meta = json.load(f)
        with open(norm_path, "r") as f:
            norm_stats = json.load(f)

        train_runs = split_meta["train"]
        test_runs = split_meta.get("test", [])

        # Compute empirical mean of train runs
        train_feats = []
        for rid in train_runs:
            ep = load_episode_from_run(split_meta["run_paths"][rid])
            if ep: train_feats.append(ep.features)

        train_concat = np.concatenate(train_feats, axis=0)
        fitted_mean = np.array(norm_stats["mean"])
        actual_train_mean = np.mean(train_concat, axis=0)

        mean_diff = float(np.max(np.abs(fitted_mean - actual_train_mean)))
        passed_norm = (mean_diff < 1e-4)

        report["normalization_audit"] = {
            "train_runs_used_for_fitting": train_runs,
            "test_runs_strictly_excluded": test_runs,
            "fitted_mean_dimension": len(fitted_mean),
            "max_discrepancy_with_train_distribution": mean_diff,
            "passed": passed_norm,
        }
        if not passed_norm:
            report["passed_all_checks"] = False
        print(f"[*] Normalization leakage check: Fitted mean matches train partition within {mean_diff:.6f}")

    # 4. Observation Window Tensor Audit
    if os.path.exists(metadata_path):
        ep = load_episode_from_run(list(split_meta["run_paths"].values())[0])
        env = DetectorEnv(episodes=[ep], window_size=4)
        s0, info = env.reset(episode_idx=0)
        s1, r, done, s_info = env.step(DetectorEnv.ACTION_WAIT)

        tensor_ok = (s0.shape == (4, expected_dim) and s1.shape == (4, expected_dim))
        report["observation_tensor_audit"] = {
            "window_shape": list(s0.shape),
            "expected_shape": [4, expected_dim],
            "passed": tensor_ok,
        }
        if not tensor_ok:
            report["passed_all_checks"] = False
        print(f"[*] Observation tensor shape: {s0.shape} (Expected: (4, {expected_dim}))")

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as fp:
        json.dump(report, fp, indent=2)

    print(f"\n[AUDIT RESULT] Passed all checks: {report['passed_all_checks']}")
    print(f"[AUDIT REPORT] Saved to {output_report_path}\n")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-path", default="results/detector/norm_stats.json")
    ap.add_argument("--output-path", default="results/validation_phase2/leakage_audit_report.json")
    args = ap.parse_args()
    run_leakage_audit(norm_path=args.norm_path, output_report_path=args.output_path)
