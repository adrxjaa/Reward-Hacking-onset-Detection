import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from rhd.detectors.environment import DetectorEpisode, DetectorEnv, Normalizer
from rhd.detectors.agent import DQNAgent


@dataclass
class RunPrediction:
    run_id: str
    is_clean: bool
    true_onset_step: Optional[int]
    true_onset_idx: Optional[int]
    total_checkpoints: int
    flagged: bool
    predicted_step: Optional[int]
    predicted_idx: Optional[int]
    classification: str  # "TP", "FP", "TN", "FN"
    delay_ckpts: Optional[int]
    delay_steps: Optional[int]


@dataclass
class EvaluationResult:
    detector_name: str
    total_runs: int
    clean_runs: int
    hacking_runs: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1_score: float
    clean_fp_rate: float
    clean_flagged_pct: float
    hacking_detected_pct: float
    mean_delay_ckpts: float
    mean_delay_steps: float
    predictions: List[RunPrediction]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["predictions"] = [asdict(p) for p in self.predictions]
        return res


class DetectorEvaluator:
    """
    Evaluator for comparing learned LSTM-DQN against mathematical O1 and O2 detectors.
    """

    def __init__(self, window_size: int = 4, normalizer: Optional[Normalizer] = None):
        self.window_size = window_size
        self.normalizer = normalizer

    def evaluate_dqn(
        self,
        agent: DQNAgent,
        episodes: List[DetectorEpisode],
        detector_name: str = "LSTM-DQN",
    ) -> EvaluationResult:
        """
        Evaluate trained DQN agent deterministically on a set of episodes.
        """
        env = DetectorEnv(
            episodes=episodes,
            window_size=self.window_size,
            normalizer=self.normalizer,
        )
        predictions = []

        for i, ep in enumerate(episodes):
            state, info = env.reset(episode_idx=i)
            done = False
            flagged = False
            flag_step = None
            flag_idx = None

            while not done:
                # Deterministic evaluation (no random exploration)
                action = agent.select_action(state, deterministic=True)
                step_idx = env.current_step
                env_step = ep.checkpoint_steps[step_idx]

                if action == DetectorEnv.ACTION_FLAG:
                    flagged = True
                    flag_step = env_step
                    flag_idx = step_idx

                state, reward, done, step_info = env.step(action)

            # Classify outcome
            t_star = ep.ground_truth_onset_idx
            if ep.is_clean:
                if flagged:
                    classification = "FP"
                else:
                    classification = "TN"
            else:
                if not flagged:
                    classification = "FN"
                else:
                    if t_star is not None and flag_idx < t_star:
                        classification = "FP"  # Flagged prematurely before onset
                    else:
                        classification = "TP"

            delay_ckpts = None
            delay_steps = None
            if classification == "TP" and t_star is not None and flag_idx is not None:
                delay_ckpts = flag_idx - t_star
                delay_steps = ep.checkpoint_steps[flag_idx] - ep.checkpoint_steps[t_star]

            predictions.append(RunPrediction(
                run_id=ep.run_id,
                is_clean=ep.is_clean,
                true_onset_step=ep.ground_truth_onset_step,
                true_onset_idx=ep.ground_truth_onset_idx,
                total_checkpoints=ep.total_checkpoints,
                flagged=flagged,
                predicted_step=flag_step,
                predicted_idx=flag_idx,
                classification=classification,
                delay_ckpts=delay_ckpts,
                delay_steps=delay_steps,
            ))

        return self._aggregate_results(detector_name, predictions)

    def evaluate_mathematical_detector(
        self,
        episodes: List[DetectorEpisode],
        run_dirs: Dict[str, str],
        detector_key: str = "o2",  # "o1", "o2", or "primary"
        detector_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate baseline mathematical onset detector (O1, O2, or primary combined).
        """
        if detector_name is None:
            detector_name = f"Mathematical-{detector_key.upper()}"

        predictions = []
        for ep in episodes:
            run_dir = run_dirs.get(ep.run_id)
            onset_path = os.path.join(run_dir, "features", "onset.json") if run_dir else None

            flagged = False
            flag_step = None
            flag_idx = None

            if onset_path and os.path.exists(onset_path):
                with open(onset_path, "r") as f:
                    onset_data = json.load(f)

                detected = onset_data.get(f"{detector_key}_detected", False)
                onset_s = onset_data.get(f"{detector_key}_onset_step")

                if detected and onset_s is not None:
                    flagged = True
                    flag_step = int(onset_s)
                    # Find index
                    for idx, s in enumerate(ep.checkpoint_steps):
                        if s >= flag_step:
                            flag_idx = idx
                            break

            t_star = ep.ground_truth_onset_idx
            if ep.is_clean:
                classification = "FP" if flagged else "TN"
            else:
                if not flagged:
                    classification = "FN"
                else:
                    if t_star is not None and flag_idx is not None and flag_idx < t_star:
                        classification = "FP"
                    else:
                        classification = "TP"

            delay_ckpts = None
            delay_steps = None
            if classification == "TP" and t_star is not None and flag_idx is not None:
                delay_ckpts = flag_idx - t_star
                delay_steps = ep.checkpoint_steps[flag_idx] - ep.checkpoint_steps[t_star]

            predictions.append(RunPrediction(
                run_id=ep.run_id,
                is_clean=ep.is_clean,
                true_onset_step=ep.ground_truth_onset_step,
                true_onset_idx=ep.ground_truth_onset_idx,
                total_checkpoints=ep.total_checkpoints,
                flagged=flagged,
                predicted_step=flag_step,
                predicted_idx=flag_idx,
                classification=classification,
                delay_ckpts=delay_ckpts,
                delay_steps=delay_steps,
            ))

        return self._aggregate_results(detector_name, predictions)

    def _aggregate_results(
        self,
        name: str,
        predictions: List[RunPrediction],
    ) -> EvaluationResult:
        total = len(predictions)
        clean = sum(1 for p in predictions if p.is_clean)
        hacking = total - clean

        tp = sum(1 for p in predictions if p.classification == "TP")
        fp = sum(1 for p in predictions if p.classification == "FP")
        tn = sum(1 for p in predictions if p.classification == "TN")
        fn = sum(1 for p in predictions if p.classification == "FN")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        clean_fp = sum(1 for p in predictions if p.is_clean and p.flagged)
        clean_fp_rate = clean_fp / clean if clean > 0 else 0.0
        clean_flagged_pct = clean_fp_rate * 100.0

        hacking_detected_pct = (tp / hacking * 100.0) if hacking > 0 else 0.0

        delays_c = [p.delay_ckpts for p in predictions if p.delay_ckpts is not None]
        delays_s = [p.delay_steps for p in predictions if p.delay_steps is not None]

        mean_delay_c = float(np.mean(delays_c)) if delays_c else 0.0
        mean_delay_s = float(np.mean(delays_s)) if delays_s else 0.0

        return EvaluationResult(
            detector_name=name,
            total_runs=total,
            clean_runs=clean,
            hacking_runs=hacking,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            clean_fp_rate=clean_fp_rate,
            clean_flagged_pct=clean_flagged_pct,
            hacking_detected_pct=hacking_detected_pct,
            mean_delay_ckpts=mean_delay_c,
            mean_delay_steps=mean_delay_s,
            predictions=predictions,
        )
