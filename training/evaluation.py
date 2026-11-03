"""Evaluation metrics and benchmark runner for offGuard early warning models.

Calculates:
- Lead Time Distribution (p10, p25, median/p50, p75, p90, min, max, mean)
- Episode-Level Detection Recall (Any advance detection before failure)
- Effective Retail Recall (>= 60s Lead Time): Fraction of failures providing >= 60s warning
- Late Warnings (< 60s): Alerts that arrived too late for cashier queue diversion
- False Alarms per Counter-Hour & per 8-Hour Cashier Shift
- 95% Bootstrap Confidence Intervals resampled across test episodes
- Per-scenario metrics breakdown (Memory Leaks, CPU Starvation, Disk Thrash, Hard Negatives)
- Sample-level Precision, Recall, and F1 on pre-failure windows
"""

from dataclasses import dataclass, field
import random
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    effective_warning_ge_60s: bool  # Lead time >= 60.0s (actionable for POS)
    false_alert_events: int
    false_positive_ticks: int
    normal_duration_seconds: float
    total_samples: int
    scenario_type: str = "unknown"


@dataclass
class EvaluationReport:
    """Comprehensive benchmark evaluation report across multiple test episodes."""

    total_episodes: int
    failure_episodes: int
    normal_episodes: int
    episode_recall: float  # Fraction of failure episodes detected in advance (> 0s)
    effective_recall_ge_60s: float  # Fraction of failures caught with >= 60s lead time
    late_warnings_count: int  # Caught, but < 60s lead time
    missed_failures_count: int  # Not caught prior to failure
    lead_time_mean_s: float
    lead_time_median_s: float
    lead_time_min_s: float
    lead_time_max_s: float
    lead_time_p10_s: float
    lead_time_p25_s: float
    lead_time_p75_s: float
    lead_time_p90_s: float
    total_normal_hours: float
    total_false_alarm_events: int
    false_alarms_per_hour: float
    false_alarms_per_8hr_shift: float
    sample_level_precision: float
    sample_level_recall: float
    sample_level_f1: float
    effective_recall_ci: Tuple[float, float] = (0.0, 0.0)
    fa_per_shift_ci: Tuple[float, float] = (0.0, 0.0)
    scenario_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    episode_results: List[EpisodeEvalResult] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Convert report to summary dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "failure_episodes": self.failure_episodes,
            "normal_episodes": self.normal_episodes,
            "episode_recall": round(self.episode_recall, 4),
            "effective_recall_ge_60s": round(self.effective_recall_ge_60s, 4),
            "effective_recall_ci_95": self.effective_recall_ci,
            "late_warnings_count": self.late_warnings_count,
            "missed_failures_count": self.missed_failures_count,
            "lead_time_median_s": round(self.lead_time_median_s, 1),
            "lead_time_mean_s": round(self.lead_time_mean_s, 1),
            "lead_time_cdf": {
                "p10": round(self.lead_time_p10_s, 1),
                "p25": round(self.lead_time_p25_s, 1),
                "p50_median": round(self.lead_time_median_s, 1),
                "p75": round(self.lead_time_p75_s, 1),
                "p90": round(self.lead_time_p90_s, 1),
            },
            "false_alarms_per_hour": round(self.false_alarms_per_hour, 2),
            "false_alarms_per_8hr_shift": round(self.false_alarms_per_8hr_shift, 2),
            "fa_per_shift_ci_95": self.fa_per_shift_ci,
            "sample_level_f1": round(self.sample_level_f1, 4),
            "scenario_breakdown": self.scenario_breakdown,
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


def compute_bootstrap_ci(
    results: List[EpisodeEvalResult],
    metric_fn: Callable[[List[EpisodeEvalResult]], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval over episode results."""
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(results)
    stats = []
    for _ in range(n_bootstrap):
        sampled = [rng.choice(results) for _ in range(n)]
        stats.append(metric_fn(sampled))
    stats.sort()
    alpha = (1.0 - ci) / 2.0
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = min(int((1.0 - alpha) * n_bootstrap), n_bootstrap - 1)
    return (round(stats[lower_idx], 4), round(stats[upper_idx], 4))


def evaluate_episode(
    samples: Sequence[Dict[str, Any]],
    predictions: Sequence[Union[bool, int, float]],
    horizon_sec: float = 120.0,
    min_effective_lead_time_sec: float = 60.0,
    risk_threshold: float = 0.5,
) -> EpisodeEvalResult:
    """Evaluate predictions on a single episode."""
    if len(samples) != len(predictions):
        raise ValueError(f"Sample length ({len(samples)}) != predictions length ({len(predictions)})")

    labeled_samples, meta = label_episode_samples(samples, horizon_sec=horizon_sec)
    failure_ts = meta.failure_timestamp

    # Infer scenario type
    scenario_type = "unknown"
    if samples and isinstance(samples[0], dict):
        scenario_type = samples[0].get("scenario_type", "unknown")
    if scenario_type == "unknown" and meta.episode_id:
        ep_lower = meta.episode_id.lower()
        if "mem" in ep_lower:
            scenario_type = "memory_leak"
        elif "cpu" in ep_lower:
            scenario_type = "cpu_starvation"
        elif "disk" in ep_lower:
            scenario_type = "disk_thrash"
        elif "startup" in ep_lower:
            scenario_type = "app_startup"
        elif "antivirus" in ep_lower:
            scenario_type = "antivirus_burst"
        elif "export" in ep_lower:
            scenario_type = "file_export"
        elif "idle" in ep_lower or "norm" in ep_lower:
            scenario_type = "routine_idle"

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
        effective = lead_time >= min_effective_lead_time_sec
    else:
        lead_time = 0.0 if meta.has_failed else None
        detected = False
        effective = False

    normal_duration_s = float(normal_ticks)

    return EpisodeEvalResult(
        episode_id=meta.episode_id,
        has_failed=meta.has_failed,
        failure_timestamp=failure_ts,
        first_alert_timestamp=first_alert_ts,
        lead_time_seconds=lead_time,
        detected_in_advance=detected,
        effective_warning_ge_60s=effective,
        false_alert_events=false_alert_events,
        false_positive_ticks=false_positive_ticks,
        normal_duration_seconds=normal_duration_s,
        total_samples=len(samples),
        scenario_type=scenario_type,
    )


def evaluate_detector(
    predict_fn: Callable[[Dict[str, Any]], Union[bool, int, float]],
    episodes: Sequence[Sequence[Dict[str, Any]]],
    horizon_sec: float = 120.0,
    min_effective_lead_time_sec: float = 60.0,
    risk_threshold: float = 0.5,
    reset_fn: Optional[Callable[[], None]] = None,
) -> EvaluationReport:
    """Evaluate a predictor/baseline function across a collection of episodes."""
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
            min_effective_lead_time_sec=min_effective_lead_time_sec,
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
    effective_count = sum(1 for r in episode_results if r.has_failed and r.effective_warning_ge_60s)
    late_count = sum(1 for r in episode_results if r.has_failed and r.detected_in_advance and not r.effective_warning_ge_60s)
    missed_count = failure_episodes_count - detected_count

    episode_recall = (
        (detected_count / failure_episodes_count) if failure_episodes_count > 0 else 1.0
    )
    effective_recall = (
        (effective_count / failure_episodes_count) if failure_episodes_count > 0 else 1.0
    )

    if lead_times:
        lead_mean = statistics.mean(lead_times)
        lead_median = statistics.median(lead_times)
        lead_min = min(lead_times)
        lead_max = max(lead_times)
        lead_p10 = _calculate_percentiles(lead_times, 10.0)
        lead_p25 = _calculate_percentiles(lead_times, 25.0)
        lead_p75 = _calculate_percentiles(lead_times, 75.0)
        lead_p90 = _calculate_percentiles(lead_times, 90.0)
    else:
        lead_mean = lead_median = lead_min = lead_max = lead_p10 = lead_p25 = lead_p75 = lead_p90 = 0.0

    normal_hours = total_normal_sec / 3600.0
    fa_per_hour = (total_false_events / normal_hours) if normal_hours > 0 else 0.0
    fa_per_8hr_shift = round(fa_per_hour * 8.0, 2)

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # 95% Bootstrap Confidence Intervals over episode results
    eff_recall_ci = compute_bootstrap_ci(
        episode_results,
        metric_fn=lambda r_list: (
            sum(1 for r in r_list if r.has_failed and r.effective_warning_ge_60s)
            / max(1, sum(1 for r in r_list if r.has_failed))
        ) if any(r.has_failed for r in r_list) else 1.0,
    )
    fa_shift_ci = compute_bootstrap_ci(
        episode_results,
        metric_fn=lambda r_list: (
            (sum(r.false_alert_events for r in r_list) / max(0.001, sum(r.normal_duration_seconds for r in r_list) / 3600.0)) * 8.0
        ),
    )

    # Per-scenario Breakdown
    scenario_breakdown: Dict[str, Dict[str, Any]] = {}
    scenarios = sorted(set(r.scenario_type for r in episode_results))
    for scen in scenarios:
        sub_results = [r for r in episode_results if r.scenario_type == scen]
        scen_fails = [r for r in sub_results if r.has_failed]
        scen_lead_times = [r.lead_time_seconds for r in scen_fails if r.lead_time_seconds is not None]
        scen_norm_sec = sum(r.normal_duration_seconds for r in sub_results)
        scen_fa = sum(r.false_alert_events for r in sub_results)
        scen_fa_per_shift = (scen_fa / (scen_norm_sec / 3600.0) * 8.0) if scen_norm_sec > 0 else 0.0

        scenario_breakdown[scen] = {
            "total_episodes": len(sub_results),
            "failure_episodes": len(scen_fails),
            "effective_recall": round(
                (sum(1 for r in scen_fails if r.effective_warning_ge_60s) / len(scen_fails))
                if scen_fails
                else 1.0,
                4,
            ),
            "median_lead_s": round(statistics.median(scen_lead_times), 1) if scen_lead_times else 0.0,
            "false_alarms_per_8hr": round(scen_fa_per_shift, 2),
        }

    return EvaluationReport(
        total_episodes=len(episodes),
        failure_episodes=failure_episodes_count,
        normal_episodes=normal_episodes_count,
        episode_recall=episode_recall,
        effective_recall_ge_60s=effective_recall,
        late_warnings_count=late_count,
        missed_failures_count=missed_count,
        lead_time_mean_s=lead_mean,
        lead_time_median_s=lead_median,
        lead_time_min_s=lead_min,
        lead_time_max_s=lead_max,
        lead_time_p10_s=lead_p10,
        lead_time_p25_s=lead_p25,
        lead_time_p75_s=lead_p75,
        lead_time_p90_s=lead_p90,
        total_normal_hours=normal_hours,
        total_false_alarm_events=total_false_events,
        false_alarms_per_hour=fa_per_hour,
        false_alarms_per_8hr_shift=fa_per_8hr_shift,
        sample_level_precision=precision,
        sample_level_recall=recall,
        sample_level_f1=f1,
        effective_recall_ci=eff_recall_ci,
        fa_per_shift_ci=fa_shift_ci,
        scenario_breakdown=scenario_breakdown,
        episode_results=episode_results,
    )
