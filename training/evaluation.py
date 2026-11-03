"""Evaluation metrics and benchmark runner for offGuard early warning models.

Calculates:
- Lead Time Distribution (median, mean, min, max, 25th/75th percentiles)
- Episode-Level Detection Recall (Fraction of failures caught in advance)
- False Alarms per Counter-Hour (FA/hr) on routine/normal operating periods
- Sample-level Precision, Recall, and F1 on pre-failure windows
"""

from dataclasses import dataclass, field
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from agent.config import DEFAULT_CONFIG, AgentConfig
from training.labeling import FailureGroundTruth, label_episode_samples


@dataclass
class EpisodeEvalResult:
    """Evaluation result for a single episode."""

    episode_id: str
    has_failed: bool
    failure_timestamp: Optional[float]
    first_alert_timestamp: Optional[float]
    lead_time_seconds: Optional[float]
    detected_in_advance: bool
    false_alert_events: int
    false_positive_ticks: int
    normal_duration_seconds: float
    total_samples: int


@dataclass
class EvaluationReport:
    """Comprehensive benchmark evaluation report across multiple test episodes."""

    total_episodes: int
    failure_episodes: int
    normal_episodes: int
    episode_recall: float  # Fraction of failure episodes detected in advance
    lead_time_mean_s: float
    lead_time_median_s: float
    lead_time_min_s: float
    lead_time_max_s: float
    lead_time_p25_s: float
    lead_time_p75_s: float
    total_normal_hours: float
    total_false_alarm_events: int
    false_alarms_per_hour: float
    sample_level_precision: float
    sample_level_recall: float
    sample_level_f1: float
    episode_results: List[EpisodeEvalResult] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Convert report to summary dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "failure_episodes": self.failure_episodes,
            "normal_episodes": self.normal_episodes,
            "episode_recall": round(self.episode_recall, 4),
            "lead_time_median_s": round(self.lead_time_median_s, 1),
            "lead_time_mean_s": round(self.lead_time_mean_s, 1),
            "lead_time_iqr_s": [round(self.lead_time_p25_s, 1), round(self.lead_time_p75_s, 1)],
            "false_alarms_per_hour": round(self.false_alarms_per_hour, 2),
            "sample_level_f1": round(self.sample_level_f1, 4),
        }


def _calculate_percentiles(values: List[float], p: float) -> float:
    """Calculate p-th percentile from a list of floats (0 <= p <= 100)."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    d = k - f
    return values_sorted[f] + d * (values_sorted[c] - values_sorted[f])


def evaluate_episode(
    samples: Sequence[Dict[str, Any]],
    predictions: Sequence[Union[bool, int, float]],
    horizon_sec: float = 120.0,
    risk_threshold: float = 0.5,
) -> EpisodeEvalResult:
    """Evaluate predictions on a single episode.

    Args:
        samples: Sequence of sample dicts (must contain 'timestamp', 'heartbeat_lag_s').
        predictions: Sequence of binary alerts (bool/int) or risk probabilities (float [0.0, 1.0]).
        horizon_sec: Target early-warning horizon H in seconds.
        risk_threshold: Threshold above which float risk score is considered an alert.
    """
    if len(samples) != len(predictions):
        raise ValueError(f"Sample length ({len(samples)}) != predictions length ({len(predictions)})")

    labeled_samples, meta = label_episode_samples(samples, horizon_sec=horizon_sec)
    failure_ts = meta.failure_timestamp

    first_alert_ts: Optional[float] = None
    in_alert_streak = False
    false_alert_events = 0
    false_positive_ticks = 0
    normal_ticks = 0

    for sample, pred in zip(labeled_samples, predictions):
        ts = sample["timestamp"]
        is_post = sample["is_post_failure"]
        target = sample["target_fail_within_H"]

        if is_post:
            # Post-failure ticks are excluded from scoring normal / pre-failure
            continue

        is_alert = (pred >= risk_threshold) if isinstance(pred, float) else bool(pred)

        # Track first alert in advance of failure
        if is_alert:
            if meta.has_failed and failure_ts is not None and ts < failure_ts:
                if first_alert_ts is None:
                    first_alert_ts = ts

        # Check for false alarms in the normal operating region (target == 0)
        if target == 0:
            normal_ticks += 1
            if is_alert:
                false_positive_ticks += 1
                if not in_alert_streak:
                    # New false alarm event (transition into alert)
                    false_alert_events += 1
                    in_alert_streak = True
            else:
                in_alert_streak = False
        else:
            in_alert_streak = is_alert

    # Compute lead time in seconds
    if meta.has_failed and failure_ts is not None and first_alert_ts is not None:
        lead_time = max(0.0, failure_ts - first_alert_ts)
        detected = lead_time > 0.0
    else:
        lead_time = 0.0 if meta.has_failed else None
        detected = False

    # Estimate normal duration in seconds (assuming 1s per normal tick)
    normal_duration_s = float(normal_ticks)

    return EpisodeEvalResult(
        episode_id=meta.episode_id,
        has_failed=meta.has_failed,
        failure_timestamp=failure_ts,
        first_alert_timestamp=first_alert_ts,
        lead_time_seconds=lead_time,
        detected_in_advance=detected,
        false_alert_events=false_alert_events,
        false_positive_ticks=false_positive_ticks,
        normal_duration_seconds=normal_duration_s,
        total_samples=len(samples),
    )


def evaluate_detector(
    predict_fn: Callable[[Dict[str, Any]], Union[bool, int, float]],
    episodes: Sequence[Sequence[Dict[str, Any]]],
    horizon_sec: float = 120.0,
    risk_threshold: float = 0.5,
    reset_fn: Optional[Callable[[], None]] = None,
) -> EvaluationReport:
    """Evaluate a predictor/baseline function across a collection of episodes.

    Args:
        predict_fn: Callable accepting a single telemetry sample dict and returning alert/score.
        episodes: List of episodes, each being a list of sample dicts.
        horizon_sec: Early-warning horizon in seconds.
        risk_threshold: Threshold to convert continuous risk score to alert.
        reset_fn: Optional callable to reset stateful baselines/models between episodes.
    """
    episode_results: List[EpisodeEvalResult] = []
    lead_times: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_normal_sec = 0.0
    total_false_events = 0

    for ep in episodes:
        if reset_fn:
            reset_fn()

        preds = [predict_fn(sample) for sample in ep]
        eval_res = evaluate_episode(
            samples=ep,
            predictions=preds,
            horizon_sec=horizon_sec,
            risk_threshold=risk_threshold,
        )
        episode_results.append(eval_res)

        if eval_res.has_failed and eval_res.lead_time_seconds is not None:
            lead_times.append(eval_res.lead_time_seconds)

        total_normal_sec += eval_res.normal_duration_seconds
        total_false_events += eval_res.false_alert_events

        # Sample-level classification counts (excluding post-failure)
        labeled, _ = label_episode_samples(ep, horizon_sec=horizon_sec)
        for s, p in zip(labeled, preds):
            if s["is_post_failure"]:
                continue
            y_true = s["target_fail_within_H"]
            y_pred = 1 if ((p >= risk_threshold) if isinstance(p, float) else bool(p)) else 0
            if y_true == 1 and y_pred == 1:
                total_tp += 1
            elif y_true == 0 and y_pred == 1:
                total_fp += 1
            elif y_true == 1 and y_pred == 0:
                total_fn += 1
            else:
                total_tn += 1

    failure_episodes_count = sum(1 for r in episode_results if r.has_failed)
    normal_episodes_count = len(episode_results) - failure_episodes_count
    detected_count = sum(1 for r in episode_results if r.has_failed and r.detected_in_advance)

    episode_recall = (
        (detected_count / failure_episodes_count) if failure_episodes_count > 0 else 1.0
    )

    if lead_times:
        lead_mean = statistics.mean(lead_times)
        lead_median = statistics.median(lead_times)
        lead_min = min(lead_times)
        lead_max = max(lead_times)
        lead_p25 = _calculate_percentiles(lead_times, 25.0)
        lead_p75 = _calculate_percentiles(lead_times, 75.0)
    else:
        lead_mean = lead_median = lead_min = lead_max = lead_p25 = lead_p75 = 0.0

    normal_hours = total_normal_sec / 3600.0
    fa_per_hour = (total_false_events / normal_hours) if normal_hours > 0 else 0.0

    # Sample-level precision, recall, f1
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return EvaluationReport(
        total_episodes=len(episodes),
        failure_episodes=failure_episodes_count,
        normal_episodes=normal_episodes_count,
        episode_recall=episode_recall,
        lead_time_mean_s=lead_mean,
        lead_time_median_s=lead_median,
        lead_time_min_s=lead_min,
        lead_time_max_s=lead_max,
        lead_time_p25_s=lead_p25,
        lead_time_p75_s=lead_p75,
        total_normal_hours=normal_hours,
        total_false_alarm_events=total_false_events,
        false_alarms_per_hour=fa_per_hour,
        sample_level_precision=precision,
        sample_level_recall=recall,
        sample_level_f1=f1,
        episode_results=episode_results,
    )

