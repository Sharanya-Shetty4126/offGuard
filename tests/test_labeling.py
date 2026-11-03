"""Unit tests for ground-truth failure detection and episode labeling."""

import pytest

from agent.config import AgentConfig
from training.labeling import (
    FailureGroundTruth,
    find_failure_point,
    label_episode_samples,
)


def _generate_synthetic_episode(
    total_seconds: int = 200,
    failure_start_sec: int = 150,
    has_failure: bool = True,
    intermittent_spikes: bool = False,
) -> list[dict]:
    """Generate mock telemetry samples for testing labeling logic."""
    samples = []
    base_time = 1000.0

    for i in range(total_seconds):
        ts = base_time + i
        lag = 0.05  # Normal baseline jitter

        if intermittent_spikes and i in (50, 51):
            # 2 consecutive high ticks - should NOT trigger 3-tick failure rule
            lag = 4.5
        elif has_failure and i >= failure_start_sec:
            # Sustained unresponsiveness
            lag = 3.5 + (i - failure_start_sec) * 0.1

        samples.append(
            {
                "timestamp": ts,
                "timestamp_iso": f"2026-11-03T10:00:{i:02d}Z",
                "episode_id": "synthetic_ep_01",
                "cpu_percent": 25.0,
                "mem_percent": 60.0,
                "mem_available_mb": 4000.0,
                "swap_percent": 10.0,
                "disk_read_mb_s": 1.0,
                "disk_write_mb_s": 0.5,
                "heartbeat_lag_s": round(lag, 3),
                "top_proc_name": "python.exe",
                "top_proc_cpu": 15.0,
                "top_proc_mem_mb": 150.0,
            }
        )
    return samples


def test_normal_run_no_failure():
    """Verify that healthy runs without sustained lag are labeled fully as negative."""
    samples = _generate_synthetic_episode(total_seconds=100, has_failure=False)
    res = find_failure_point(samples)
    assert res is None

    labeled, meta = label_episode_samples(samples, horizon_sec=60.0)
    assert not meta.has_failed
    assert meta.failure_timestamp is None
    assert meta.pre_failure_samples == 0
    assert meta.post_failure_samples == 0
    assert all(s["target_fail_within_H"] == 0 for s in labeled)
    assert all(not s["is_post_failure"] for s in labeled)


def test_intermittent_spikes_do_not_trigger_failure():
    """Verify that isolated spikes (e.g. 2 ticks) do not trigger a 3-tick failure."""
    samples = _generate_synthetic_episode(
        total_seconds=100, has_failure=False, intermittent_spikes=True
    )
    res = find_failure_point(samples)
    assert res is None

    labeled, meta = label_episode_samples(samples, horizon_sec=60.0)
    assert not meta.has_failed
    assert all(s["target_fail_within_H"] == 0 for s in labeled)


def test_sustained_failure_labeling_and_horizon():
    """Verify exact T_failure detection and target window [T_failure - H, T_failure)."""
    samples = _generate_synthetic_episode(
        total_seconds=200, failure_start_sec=150, has_failure=True
    )

    failure_res = find_failure_point(samples)
    assert failure_res is not None
    idx, ts, mode = failure_res
    assert idx == 150
    assert ts == 1150.0
    assert mode == "consecutive_lag"

    labeled, meta = label_episode_samples(samples, horizon_sec=50.0)

    assert meta.has_failed
    assert meta.failure_timestamp == 1150.0
    assert meta.failure_index == 150
    assert meta.failure_mode == "consecutive_lag"
    assert meta.pre_failure_samples == 50
    assert meta.post_failure_samples == 50

    for s in labeled[:100]:
        assert s["target_fail_within_H"] == 0
        assert not s["is_post_failure"]

    for s in labeled[100:150]:
        assert s["target_fail_within_H"] == 1
        assert not s["is_post_failure"]

    for s in labeled[150:]:
        assert s["target_fail_within_H"] == 0
        assert s["is_post_failure"]


def test_catastrophic_single_tick_hang():
    """Verify that a single massive lag sample (whole machine hung then woke up) is caught."""
    # 30 seconds of baseline: timestamps 1000.0 to 1029.0
    samples = _generate_synthetic_episode(total_seconds=30, has_failure=False)
    # Inject a 15-second hang: the machine wakes up 15 seconds late at 1045.0
    samples.append(
        {
            "timestamp": 1045.0,
            "timestamp_iso": "2026-11-03T10:00:45Z",
            "episode_id": "ep_hang",
            "cpu_percent": 100.0,
            "mem_percent": 95.0,
            "mem_available_mb": 400.0,
            "swap_percent": 85.0,
            "disk_read_mb_s": 0.0,
            "disk_write_mb_s": 0.0,
            "heartbeat_lag_s": 15.0,  # Single massive tick!
            "top_proc_name": "python.exe",
            "top_proc_cpu": 90.0,
            "top_proc_mem_mb": 1500.0,
        }
    )

    failure_res = find_failure_point(samples, catastrophic_threshold_sec=6.0)
    assert failure_res is not None
    idx, freeze_start_ts, mode = failure_res
    assert mode in ("catastrophic_lag", "timestamp_gap")
    # Freeze began right after sample 29 (timestamp 1029.0 + 1.0 = 1030.0)
    assert abs(freeze_start_ts - 1030.0) <= 1.0


def test_missing_ticks_timestamp_gap():
    """Verify that a missing sample gap (collector was frozen/dead) is caught as a freeze."""
    samples = []
    # 20 normal samples at t = 1000..1019
    for i in range(20):
        samples.append(
            {
                "timestamp": 1000.0 + i,
                "heartbeat_lag_s": 0.05,
                "episode_id": "ep_gap",
            }
        )

    # 15-second gap where machine completely hung and no ticks executed!
    # Next sample arrives at 1035.0
    samples.append(
        {
            "timestamp": 1035.0,
            "heartbeat_lag_s": 0.05,  # even if lag counter was reset
            "episode_id": "ep_gap",
        }
    )

    failure_res = find_failure_point(samples, catastrophic_threshold_sec=6.0)
    assert failure_res is not None
    idx, freeze_start_ts, mode = failure_res
    assert mode == "timestamp_gap"
    # Freeze began right after sample 19 (at timestamp 1020.0)
    assert freeze_start_ts == 1020.0


def test_drop_post_failure_flag():
    """Verify that drop_post_failure=True cleanly strips all post-failure records."""
    samples = _generate_synthetic_episode(
        total_seconds=100, failure_start_sec=80, has_failure=True
    )
    labeled, meta = label_episode_samples(samples, horizon_sec=30.0, drop_post_failure=True)

    assert meta.total_samples == 100
    assert meta.post_failure_samples == 20
    assert len(labeled) == 80
    assert all(not s["is_post_failure"] for s in labeled)
