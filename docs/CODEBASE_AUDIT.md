# Codebase Audit (Phase 0)

Scope: audit only. No implementation code was changed. Audited at git commit `24e2a83 environment1` (single commit).

## 1. Current architecture

Flat, single-environment, single-seed prototype: a 5x5 GridWorld with a respawning coin (proxy exploit) and a terminal goal, trained with Stable-Baselines3 PPO.

```
config.py ──► train.py ──► checkpoints/gridworld/{0,5000,...,50000}.zip
                                   │
environment.py ◄── evaluate.py ◄───┘ ──► results/metrics.csv ──► analyze.py ──► results/*.png
metrics.py (schema, entropy, row builder)       run_all.py chains train→evaluate→analyze
```

Pipeline is "one run, 7 checkpoints, 20 deterministic eval episodes per checkpoint".

## 2. Existing components

| Area | File | Notes |
|---|---|---|
| Environment | `environment.py` | `GridWorldEnv`, Gymnasium API, obs `[row, col, coin_avail, steps_since_coin]`, 4 discrete actions, max 200 steps. Coin +1 (respawns after 5 steps), goal +5 (terminates), step −0.01. |
| RL algorithm | SB3 `PPO("MlpPolicy")` | Only PPO. No SAC/TD3. No custom RL code. |
| Training loop | `train.py` | `model.learn` in chunks between checkpoint steps, `reset_num_timesteps=False`. Hardcodes the `GridWorldEnv` import. |
| Reward | `environment.py` | Proxy only; true reward / hacking indicator not separated (only `goal_reached`, `coin_visits` in `info`). |
| Checkpointing | `train.py`, `config.checkpoint_path` | SB3 `.zip` at `CHECKPOINT_STEPS`; layout `checkpoints/<env>/<step>.zip`. No run/seed dimension. |
| Logging | `print` only | No TensorBoard, no per-training-run log of rollout stats. |
| Config | `config.py` | Module-level constants (SEED=42, PPO_KWARGS, paths). |
| Experiments | `run_all.py` | train → evaluate → analyze for one run. |
| Metrics | `metrics.py`, `evaluate.py` | Common CSV schema: reward, length, goal, entropy, state visitation, action dist, extra JSON. |
| Analysis | `analyze.py` | 4 aggregate plots (reward, goal rate, coin visits, entropy). |
| Tests | none | No `tests/`. |
| Dependencies | `requirements.txt` | unpinned: sb3[extra], gymnasium, numpy, pandas, matplotlib, torch, pillow. |
| Data | `results/metrics.csv` (140 rows = 7 ckpts × 20 eps), 4 PNGs, 7 checkpoints | Committed artifacts. `.gitignore` only ignores `__pycache__`/`*.pyc`. |
| Misc | `details.txt`, `eval_err.txt`, `eval_out.txt` | Teammate spec; a Windows torch-DLL traceback; empty file. |

## 3. Reusable components

- **`environment.py` GridWorld**: valid Gymnasium env and reward logic; reusable as the base of the region-toggle env (needs reward changes, see §5).
- **`train.py` checkpoint-chunk pattern**: keep the idea (train in chunks, save at schedule, step-0 checkpoint).
- **`metrics.py`**: `compute_entropy`, `state_to_key`, `build_metric_row` ideas; JSON-in-CSV extras pattern.
- **`config.py`** path helpers and PPO defaults as a starting point.
- **`analyze.py`** plot helpers (generic dynamics plots) are reusable for diagnostics.
- Existing checkpoints/results: useful as a smoke-test fixture only.

## 4. Missing components

- Region-toggle reward environment (the mid-review exploit); current exploit is a respawning coin, not region toggle.
- Separate **true reward** vs **proxy reward** accounting (needed for labeling; detector must never see true reward).
- Multi-run orchestration (~20 runs), per-run seeding, run directories, run manifest (config + seed + versions).
- Checkpoint-level feature extraction that is separate from evaluation CSV rows (feature vector per checkpoint per run).
- Denser checkpoint schedule (current 7 checkpoints is too coarse for onset timing).
- O1/O2 onset labeling module (definitions and thresholds must be fixed from the plan).
- Algorithm abstraction (PPO/SAC/TD3), detector pipeline/env, tests, experiment-config files, `.gitignore` for artifacts, `README`.

## 5. Problems / issues found

1. **Not runnable here**: no venv; `stable_baselines3`, `pandas` not installed on the system Python 3.14. `eval_err.txt` shows a prior torch DLL failure (Windows, Python 3.13). Environment must be set up and pinned.
2. **Weak reproducibility**: single `SEED=42`. `make_vec_env(seed=)` and `PPO(seed=)` seed SB3, but no `torch.backends`/deterministic settings, no per-run seed, no stored config/versions. Eval `env.reset(seed=SEED+ep)` is meaningless (env is deterministic). `PPO.load` eval uses `deterministic=True`, so training-time stochastic behavior is not what is measured.
3. **Entropy bug/bias**: `evaluate.py` uses `all_states[:ENTROPY_SAMPLE_STATES]`, i.e. the first 50 states by enumeration (all in row 0/1), not "random states" as commented. Entropy is therefore not representative of visited states. Better: entropy over states actually visited in eval rollouts.
4. **Mislabelled "coin_available/steps_since_coin" enumeration** includes unreachable states (e.g. `since>0` while coin available).
5. **`action_distribution` is cumulative**: `action_counts` is not reset per episode, so each row's distribution depends on prior episodes. Same issue makes rows non-independent.
6. **Entropy is the same for every episode row** at a checkpoint (computed once), and `state_visitation` keys are saved as `"(np.int64(2), ...)"` strings in the committed CSV (numpy ≥2 `str(tuple)` of np ints). `state_to_key` is fragile; use plain ints/lists.
7. **Exploit design ambiguity**: coin is worth +1 per 5-step respawn but step penalty is tiny, and the goal terminates the episode with +5; hacking onset is not rigorously guaranteed. Also the `details.txt` text says "walk straight to G ... value" claim is not verified. `terminated=True` on goal while `goal_reached` is conflated with "true objective" (true reward not defined separately).
8. **Inconsistent output paths**: docs/`config.results_csv_path` say `results/gridworld_metrics.csv`; `evaluate.py` writes `results/metrics.csv`; `analyze.py` has a fallback hack for both. `results_plots_dir` is unused. `run_all.py` message mentions `.png` in `results/`.
9. **Hardcoded env imports** in `train.py`/`evaluate.py` (`from environment import GridWorldEnv`); no registry/CLI; `details.txt` instructs editing the import to switch env.
10. **Discrete-only assumptions**: `metrics.compute_avg_entropy` relies on `distribution.logits` (Categorical) — breaks for SAC/TD3 (continuous Gaussian/deterministic) and the fallback has the same limitation. Feature extraction needs an algorithm-agnostic interface.
11. **Algorithm coupling**: `PPO.load` hardcoded in `evaluate.py`; `PPO_KWARGS` in global config.
12. **Repo hygiene**: checkpoints/results/`eval_*.txt` committed; non-ASCII `?` in `metrics.py` header (encoding damage); CRLF line endings; no `.gitignore` for artifacts.
13. **No tests** and no logging of training-time stats.
14. n_envs=1 and `n_steps=512` mean checkpoint boundaries (5000, etc.) are not multiples of 512; SB3 rounds up to the next rollout, so actual `num_timesteps` at "5,000" is 5,120 etc. Checkpoint labels are therefore inexact.

## 6. Do not modify (may be useful later)

- `environment.py` GridWorld: keep as a baseline/sanity env; add the region-toggle env as a new module instead.
- Existing `checkpoints/gridworld/` and `results/metrics.csv`: keep as regression fixtures (move/ignore, don't delete).
- `metrics.COMMON_SCHEMA`/`build_metric_row` and `analyze.py` plotting helpers: extend, don't break.
- `details.txt`: record of the teammate spec.

## 7. Assessment against requirements

| Question | Verdict |
|---|---|
| Reproducible via seeds/config? | **Partially.** One global seed, no per-run seeds or saved config; unverified (cannot run here). |
| Supports PPO now, SAC/TD3 later? | **Not without restructuring**: PPO hardcoded, entropy code Categorical-only, flat config. Small refactor (algo registry + policy-stats adapter) suffices; no big rewrite. |
| Supports a separate detector pipeline? | **No, but cleanly addable**: eval CSV is per-episode and mixes policy stats with reward; needs per-run per-checkpoint feature tables (no true reward) plus a labels file kept separate. |

## 8. Recommended structure

```
rhd/
  envs/        gridworld.py (moved), region_toggle.py, registry.py
  generators/  base.py (train/checkpoint interface), ppo.py   # sac.py, td3.py later
  features/    extract.py (checkpoint -> behavioral stats, no true reward)
  labeling/    onset.py (O1/O2)
  detector/    (later) env.py, dqn_lstm.py
  utils/       seeding.py, io.py
configs/       ppo_region_toggle.yaml (or .py dataclass)
scripts/       train_runs.py, extract_features.py, label_onsets.py, plot.py
tests/         test_env.py, test_seeding.py, test_features.py, test_labeling.py
runs/<env>/<algo>/seed_<k>/   checkpoints/, train_log.csv, config.json   (gitignored)
data/          features/*.parquet|csv, labels/*.csv                     (gitignored)
docs/
```
Keep the flat files working until migration; avoid new frameworks (plain dataclasses/JSON; PyTorch + SB3 already present).

## 9. Dependencies

Present: stable-baselines3[extra], gymnasium, numpy, pandas, matplotlib, torch, pillow (unpinned). Needed soon: `pytest`. SB3 already provides SAC/TD3, so no new RL dependency. Pin versions (`pip freeze`) in a lockfile; use a venv with a Python version supported by torch (3.14 may lack wheels → prefer 3.11/3.12). Pillow only needed for GIFs (not currently generated); `[extra]` pulls heavy deps and can be dropped to `stable-baselines3`.

## 10. Testing status

None. No unit tests, no smoke test, existing pipeline was not executed in this audit (dependencies missing). Stored CSV (140 rows, 7 checkpoints × 20 episodes) was inspected for schema only.

## 11. Phase 1 implementation order

1. Create venv (Python 3.11/3.12), install deps, pin `requirements.txt`, add `pytest`; add `.gitignore` for `runs/`, `data/`, `checkpoints/`, `results/`.
2. Run the existing pipeline once as a smoke test; confirm the baseline works and is seed-deterministic (same seed → identical checkpoint outputs).
3. Add `tests/` skeleton and `utils/seeding.py` (seed python/numpy/torch/env).
4. Fix metrics bugs (§5 items 3, 5, 6, 8) behind tests; keep old schema columns.
5. Implement region-toggle env (`envs/region_toggle.py`) with separate `true_reward` and `proxy_reward` in `info`; unit-test reward logic and hackability (scripted exploit policy beats scripted honest policy on proxy, loses on true).
6. Env registry + generator interface; wrap PPO (`generators/ppo.py`) with config dataclass, run directory, manifest (config, seed, versions), denser checkpoint schedule with exact timestep callback.
7. Multi-seed runner producing ~20 PPO runs; verify seed reproducibility.
8. Feature extraction from checkpoints (algorithm-agnostic action statistics, state-visitation/region occupancy, action entropy estimated from rollouts, episode length, proxy return; **no true reward**), written to `data/features`.
9. O1/O2 onset labeling from true-reward/exploit metrics into a separate `data/labels` file; plot per-run curves with onset markers for sanity review.
10. Freeze dataset schema; only then begin the detector (Phase 2).
