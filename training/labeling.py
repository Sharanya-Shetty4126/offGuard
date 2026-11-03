"""Ground-truth labeling and failure detection for offGuard telemetry.

Implements the deterministic T_failure rule:
Heartbeat lag > threshold (default 3.0s) for consecutive ticks (default 3).
Computes lead-time target labels Y(t) within prediction horizon H (default 120s)
and masks post-failure samples to prevent trivial flatline learning.
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
    streak_start_index: Optional[int]
    horizon_sec: float
    total_samples: int
    pre_failure_samples: int
    post_failure_samples: int


def find_failure_point(
    samples: Sequence[Union[Dict[str, Any], Any]],
    lag_threshold_sec: Optional[float] = None,
    consecutive_ticks: Optional[int] = None,
) -> Optional[tuple[int, float]]:
    """Identify earliest point where heartbeat_lag exceeds threshold for N consecutive ticks.

    Returns:
        (index, timestamp) of the first tick of the consecutive failure streak,
        or None if the episode never experienced sustained failure.
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

    if not samples or required_ticks <= 0:
        return None

    current_streak_len = 0
    streak_start_idx = -1

    for idx, sample in enumerate(samples):
        # Extract heartbeat_lag_s and timestamp from dict or dataclass object
        if isinstance(sample, dict):
            lag = float(sample.get("heartbeat_lag_s", 0.0))
            ts = float(sample.get("timestamp", idx))
        else:
            lag = float(getattr(sample, "heartbeat_lag_s", 0.0))
            ts = float(getattr(sample, "timestamp", idx))

        if lag >= threshold:
            if current_streak_len == 0:
                streak_start_idx = idx
            current_streak_len += 1

            if current_streak_len >= required_ticks:
                # Return the timestamp and index where this failure streak began
                first_sample = samples[streak_start_idx]
                first_ts = (
                    float(first_sample.get("timestamp", streak_start_idx))
                    if isinstance(first_sample, dict)
                    else float(getattr(first_sample, "timestamp", streak_start_idx))
                )
                return streak_start_idx, first_ts
        else:
            current_streak_len = 0
            streak_start_idx = -1

    return None


def label_episode_samples(
    samples: Sequence[Union[Dict[str, Any], Any]],
    horizon_sec: Optional[float] = None,
    config: Optional[AgentConfig] = None,
    drop_post_failure: bool = False,
) -> tuple[List[Dict[str, Any]], FailureGroundTruth]:
    """Label an episode's samples with target_fail_within_H and is_post_failure flags.

    Target Definition:
        Y(t) = 1 if (T_failure - H <= t < T_failure)
        Y(t) = 0 if (t < T_failure - H) or if no failure occurred.
        Post-failure samples (t >= T_failure) have is_post_failure = True.

    Args:
        samples: Sequence of TelemetrySample instances or raw dicts.
        horizon_sec: Forecasting lead-time window H in seconds (default: config.default_horizon_sec).
        config: Optional AgentConfig instance.
        drop_post_failure: If True, post-failure samples are excluded from returned list.

    Returns:
        Tuple of (labeled_samples_as_dicts, FailureGroundTruth metadata).
    """
    cfg = config or DEFAULT_CONFIG
    h_sec = float(horizon_sec if horizon_sec is not None else cfg.default_horizon_sec)

    failure_res = find_failure_point(
        samples,
        lag_threshold_sec=cfg.unresponsive_lag_threshold_sec,
        consecutive_ticks=cfg.consecutive_lag_ticks,
    )

    episode_id = "unknown"
    if samples:
        first = samples[0]
        episode_id = first.get("episode_id", "unknown") if isinstance(first, dict) else getattr(first, "episode_id", "unknown")

    has_failed = failure_res is not None
    failure_idx: Optional[int] = failure_res[0] if failure_res else None
    failure_ts: Optional[float] = failure_res[1] if failure_res else None

    labeled_samples: List[Dict[str, Any]] = []
    pre_failure_count = 0
    post_failure_count = 0

    for idx, sample in enumerate(samples):
        # Convert to dict representation
        if hasattr(sample, "to_dict"):
            d = sample.to_dict()
        elif isinstance(sample, dict):
            d = dict(sample)
        else:
            d = dict(sample.__dict__)

        ts = float(d.get("timestamp", idx))

        if has_failed and failure_ts is not None:
            if ts >= failure_ts:
                is_post = True
                target = 0  # Post-failure should not be trained on as positive pre-failure
                post_failure_count += 1
            elif failure_ts - h_sec <= ts < failure_ts:
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
        streak_start_index=failure_idx,
        horizon_sec=h_sec,
        total_samples=len(samples),
        pre_failure_samples=pre_failure_count,
        post_failure_samples=post_failure_count,
    )

    return labeled_samples, metadata

