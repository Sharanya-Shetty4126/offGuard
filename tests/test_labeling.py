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
    res = find_failure_point(samples, lag_threshold_sec=3.0, consecutive_ticks=3)
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
    res = find_failure_point(samples, lag_threshold_sec=3.0, consecutive_ticks=3)
    assert res is None

    labeled, meta = label_episode_samples(samples, horizon_sec=60.0)
    assert not meta.has_failed
    assert all(s["target_fail_within_H"] == 0 for s in labeled)


def test_sustained_failure_labeling_and_horizon():
    """Verify exact T_failure detection and target window [T_failure - H, T_failure)."""
    # 200 samples total, failure starts at sample index 150 (timestamp = 1150.0)
    samples = _generate_synthetic_episode(
        total_seconds=200, failure_start_sec=150, has_failure=True
    )

    failure_res = find_failure_point(samples, lag_threshold_sec=3.0, consecutive_ticks=3)
    assert failure_res is not None
    idx, ts = failure_res
    assert idx == 150
    assert ts == 1150.0

    # Horizon H = 50 seconds.
    # Pre-failure target window: [1150 - 50, 1150) = [1100.0, 1150.0) -> indices 100 to 149
    labeled, meta = label_episode_samples(samples, horizon_sec=50.0)

    assert meta.has_failed
    assert meta.failure_timestamp == 1150.0
    assert meta.streak_start_index == 150
    assert meta.pre_failure_samples == 50  # 50 ticks in pre-failure window
    assert meta.post_failure_samples == 50  # indices 150..199 are post failure

    # Check labels before horizon (< 1100.0)
    for s in labeled[:100]:
        assert s["target_fail_within_H"] == 0
        assert not s["is_post_failure"]

    # Check labels inside horizon [1100.0, 1150.0)
    for s in labeled[100:150]:
        assert s["target_fail_within_H"] == 1
        assert not s["is_post_failure"]

    # Check post-failure zone (>= 1150.0)
    for s in labeled[150:]:
        assert s["target_fail_within_H"] == 0
        assert s["is_post_failure"]


def test_drop_post_failure_flag():
    """Verify that drop_post_failure=True cleanly strips all post-failure records."""
    samples = _generate_synthetic_episode(
        total_seconds=100, failure_start_sec=80, has_failure=True
    )
    labeled, meta = label_episode_samples(samples, horizon_sec=30.0, drop_post_failure=True)

    assert meta.total_samples == 100
    assert meta.post_failure_samples == 20
    # Length of returned list should be exactly 80
    assert len(labeled) == 80
    assert all(not s["is_post_failure"] for s in labeled)

