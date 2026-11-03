"""Ground-truth labeling and failure detection for offGuard telemetry.

Implements dual ground-truth failure rules + documented safety proxy failure:
1. Sustained Degradation: Heartbeat lag >= lag_threshold (default 3.0s) for >= 3 consecutive ticks.
2. Catastrophic Freeze / Hang: A single sample lag >= catastrophic_threshold (default 6.0s)
   OR a missing tick gap between timestamps >= catastrophic_threshold.
3. Documented Safety Proxy Failure: If safety circuit breakers abort an injector before
   the host machine freezes (e.g. RAM floor reached), the breaker trip time is documented
   and used as proxy T_failure.
4. Post-Failure & Post-Injector Recovery Masking: Samples after failure OR after an injector
   releases resources are masked with is_post_failure = True to prevent training on recovery noise.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from agent.config import DEFAULT_CONFIG, AgentConfig


@dataclass(frozen=True)
class FailureGroundTruth:
    """Ground truth failure metadata for a telemetry episode."""

    episode_id: str
    has_failed: bool
    failure_timestamp: Optional[float]
    failure_index: Optional[int]
    failure_mode: str  # "consecutive_lag", "catastrophic_lag", "timestamp_gap", "proxy_breaker_trip", or "none"
    horizon_sec: float
    total_samples: int
    pre_failure_samples: int
    post_failure_samples: int


def _get_field(sample: Union[Dict[str, Any], Any], key: str, default: Any = 0.0) -> Any:
    """Helper to extract a field from either dict or dataclass object."""
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def find_failure_point(
    samples: Sequence[Union[Dict[str, Any], Any]],
    lag_threshold_sec: Optional[float] = None,
    consecutive_ticks: Optional[int] = None,
    catastrophic_threshold_sec: Optional[float] = None,
    expected_interval_sec: Optional[float] = None,
) -> Optional[tuple[int, float, str]]:
    """Identify earliest point of system unresponsiveness.

    Detects either:
    A) Sustained degradation: heartbeat_lag >= lag_threshold for N consecutive ticks.
    B) Catastrophic freeze / gap: single tick lag >= catastrophic_threshold OR
       wall-clock gap between adjacent samples >= catastrophic_threshold.

    Returns:
        (index, failure_timestamp, failure_mode)
        where failure_timestamp is the estimated wall-clock time the unresponsiveness began,
        or None if no failure occurred.
    """
    threshold = (
        lag_threshold_sec
        if lag_threshold_sec is not None
        else DEFAULT_CONFIG.unresponsive_lag_threshold_sec
    )
    required_ticks = (
        consecutive_ticks
        if consecutive_ticks is not None
        else DEFAULT_CONFIG.consecutive_lag_ticks
    )
    catastrophic_threshold = (
        catastrophic_threshold_sec
        if catastrophic_threshold_sec is not None
        else DEFAULT_CONFIG.catastrophic_lag_threshold_sec
    )
    interval = (
        expected_interval_sec
        if expected_interval_sec is not None
        else DEFAULT_CONFIG.sample_interval_sec
    )

    if not samples:
        return None

    current_streak_len = 0
    streak_start_idx = -1
    prev_ts: Optional[float] = None

    for idx, sample in enumerate(samples):
        ts = float(_get_field(sample, "timestamp", idx))
        lag = float(_get_field(sample, "heartbeat_lag_s", 0.0))

        # Check 1: Missing ticks / wall-clock timestamp gap (collector was frozen/dead)
        if prev_ts is not None:
            time_gap = ts - prev_ts
            if time_gap >= catastrophic_threshold:
                freeze_start_ts = prev_ts + interval
                return idx, freeze_start_ts, "timestamp_gap"

        # Check 2: Catastrophic single-sample lag spike (machine hung, then woke up late)
        if lag >= catastrophic_threshold:
            freeze_start_ts = (prev_ts + interval) if prev_ts is not None else (ts - lag)
            return idx, freeze_start_ts, "catastrophic_lag"

        # Check 3: Sustained mild lag over consecutive ticks
        if lag >= threshold:
            if current_streak_len == 0:
                streak_start_idx = idx
            current_streak_len += 1

            if current_streak_len >= required_ticks:
                first_sample = samples[streak_start_idx]
                first_ts = float(_get_field(first_sample, "timestamp", streak_start_idx))
                return streak_start_idx, first_ts, "consecutive_lag"
        else:
            current_streak_len = 0
            streak_start_idx = -1

        prev_ts = ts

    return None


def label_episode_samples(
    samples: Sequence[Union[Dict[str, Any], Any]],
    horizon_sec: Optional[float] = None,
    config: Optional[AgentConfig] = None,
    drop_post_failure: bool = False,
    proxy_failure_timestamp: Optional[float] = None,
    proxy_failure_mode: Optional[str] = None,
    recovery_cutoff_timestamp: Optional[float] = None,
) -> tuple[List[Dict[str, Any]], FailureGroundTruth]:
    """Label an episode's samples with target_fail_within_H and is_post_failure flags.

    Target Definition:
        Y(t) = 1 if (T_failure - H <= t < T_failure)
        Y(t) = 0 if (t < T_failure - H) or if no failure occurred.
        Post-failure and post-injector recovery samples have is_post_failure = True.

    Args:
        samples: Sequence of TelemetrySample instances or raw dicts.
        horizon_sec: Forecasting lead-time window H in seconds.
        config: Optional AgentConfig instance.
        drop_post_failure: If True, post-failure/recovery samples are excluded from returned list.
        proxy_failure_timestamp: Optional timestamp of safety circuit breaker trip to use as proxy T_failure.
        proxy_failure_mode: Description of proxy failure (e.g. "proxy_ram_floor_breached").
        recovery_cutoff_timestamp: Timestamp when stress stopped; all subsequent recovery ticks are masked.
    """
    cfg = config or DEFAULT_CONFIG
    h_sec = float(horizon_sec if horizon_sec is not None else cfg.default_horizon_sec)

    # 1. First check if a real OS lag failure occurred
    failure_res = find_failure_point(
        samples,
        lag_threshold_sec=cfg.unresponsive_lag_threshold_sec,
        consecutive_ticks=cfg.consecutive_lag_ticks,
        catastrophic_threshold_sec=cfg.catastrophic_lag_threshold_sec,
        expected_interval_sec=cfg.sample_interval_sec,
    )

    episode_id = "unknown"
    if samples:
        episode_id = str(_get_field(samples[0], "episode_id", "unknown"))

    # 2. Reconcile real failure vs. safety circuit breaker proxy failure
    if failure_res is not None:
        has_failed = True
        failure_idx: Optional[int] = failure_res[0]
        failure_ts: Optional[float] = failure_res[1]
        failure_mode: str = failure_res[2]
    elif proxy_failure_timestamp is not None:
        has_failed = True
        failure_ts = proxy_failure_timestamp
        failure_mode = proxy_failure_mode or "proxy_breaker_trip"
        # Find index matching proxy failure timestamp
        failure_idx = None
        for i, s in enumerate(samples):
            ts = float(_get_field(s, "timestamp", i))
            if ts >= failure_ts:
                failure_idx = i
                break
        if failure_idx is None:
            failure_idx = len(samples) - 1
    else:
        has_failed = False
        failure_idx = None
        failure_ts = None
        failure_mode = "none"

    # Effective masking boundary: earlier of failure_ts and recovery_cutoff_timestamp
    mask_cutoff_ts: Optional[float] = failure_ts
    if recovery_cutoff_timestamp is not None:
        if mask_cutoff_ts is not None:
            mask_cutoff_ts = min(mask_cutoff_ts, recovery_cutoff_timestamp)
        else:
            mask_cutoff_ts = recovery_cutoff_timestamp

    labeled_samples: List[Dict[str, Any]] = []
    pre_failure_count = 0
    post_failure_count = 0

    for idx, sample in enumerate(samples):
        if hasattr(sample, "to_dict"):
            d = sample.to_dict()
        elif isinstance(sample, dict):
            d = dict(sample)
        else:
            d = dict(sample.__dict__)

        ts = float(d.get("timestamp", idx))

        is_post = False
        target = 0

        # Check post-failure or post-recovery masking
        if mask_cutoff_ts is not None and ts >= mask_cutoff_ts:
            is_post = True
            target = 0
            post_failure_count += 1
        elif has_failed and failure_ts is not None:
            if (failure_ts - h_sec) <= ts < failure_ts:
                is_post = False
                target = 1
                pre_failure_count += 1
            else:
                is_post = False
                target = 0
        else:
            is_post = False
            target = 0

        d["is_post_failure"] = is_post
        d["target_fail_within_H"] = target
        d["horizon_sec"] = h_sec

        if drop_post_failure and is_post:
            continue

        labeled_samples.append(d)

    metadata = FailureGroundTruth(
        episode_id=episode_id,
        has_failed=has_failed,
        failure_timestamp=failure_ts,
        failure_index=failure_idx,
        failure_mode=failure_mode,
        horizon_sec=h_sec,
        total_samples=len(samples),
        pre_failure_samples=pre_failure_count,
        post_failure_samples=post_failure_count,
    )

    return labeled_samples, metadata
