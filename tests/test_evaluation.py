"""Unit tests for the evaluation harness and metrics calculation."""

import pytest

from agent.baselines import StaticThresholdBaseline, TimeToExhaustionBaseline
from training.evaluation import evaluate_detector, evaluate_episode


def _create_mock_episode(
    episode_id: str,
    total_seconds: int = 150,
    failure_second: int = 120,
    has_failure: bool = True,
    drain_ram: bool = False,
) -> list[dict]:
    """Helper to generate a telemetry episode."""
    samples = []
    base_t = 500.0

    for i in range(total_seconds):
        t = base_t + i
        lag = 0.05
        mem_pct = 50.0
        swap_pct = 20.0
        mem_avail = 4000.0

        if drain_ram:
            # Drain 25 MB/s starting from 3500 MB
            # At i=80: avail = 3500 - 2000 = 1500 MB (1000 MB headroom to 500MB floor -> 40s to floor)
            mem_avail = max(400.0, 3500.0 - 25.0 * i)
            mem_pct = min(98.0, 50.0 + 0.4 * i)
            if mem_pct > 85.0:
                swap_pct = min(95.0, 20.0 + 0.8 * (i - 70))

        if has_failure and i >= failure_second:
            lag = 3.5

        samples.append(
            {
                "timestamp": t,
                "timestamp_iso": f"2026-11-03T12:00:{i:02d}Z",
                "episode_id": episode_id,
                "cpu_percent": 30.0,
                "mem_percent": mem_pct,
                "mem_available_mb": mem_avail,
                "swap_percent": swap_pct,
                "disk_read_mb_s": 1.0,
                "disk_write_mb_s": 1.0,
                "heartbeat_lag_s": lag,
                "top_proc_name": "python.exe",
                "top_proc_cpu": 10.0,
                "top_proc_mem_mb": 200.0,
            }
        )
    return samples


def test_evaluate_episode_lead_time_and_false_alarms():
    """Verify lead time and false alarm calculation on a single episode."""
    samples = _create_mock_episode(
        "ep_lead_time", total_seconds=100, failure_second=80, has_failure=True
    )
    # Failure occurs at sample 80 (timestamp = 580.0).
    # Simulate alerts emitted starting at sample 50 (timestamp = 550.0).
    # Lead time should be 580.0 - 550.0 = 30.0 seconds.
    preds = [1 if i >= 50 else 0 for i in range(100)]

    res = evaluate_episode(samples, preds, horizon_sec=60.0)

    assert res.has_failed
    assert res.failure_timestamp == 580.0
    assert res.first_alert_timestamp == 550.0
    assert res.lead_time_seconds == 30.0
    assert res.detected_in_advance
    # Since alerts started at 50, and horizon is 60s (window is [520, 580)),
    # sample 50 is within horizon (520 <= 550 < 580), so 0 false alarm events before horizon.
    assert res.false_alert_events == 0


def test_evaluate_detector_with_baselines():
    """Verify end-to-end multi-episode benchmark on TimeToExhaustionBaseline."""
    # Create 2 failure episodes with memory draining and 2 normal episodes
    ep_fail_1 = _create_mock_episode("ep_fail_1", total_seconds=120, failure_second=100, has_failure=True, drain_ram=True)
    ep_fail_2 = _create_mock_episode("ep_fail_2", total_seconds=130, failure_second=110, has_failure=True, drain_ram=True)
    ep_norm_1 = _create_mock_episode("ep_norm_1", total_seconds=80, has_failure=False, drain_ram=False)
    ep_norm_2 = _create_mock_episode("ep_norm_2", total_seconds=80, has_failure=False, drain_ram=False)

    episodes = [ep_fail_1, ep_fail_2, ep_norm_1, ep_norm_2]

    baseline = TimeToExhaustionBaseline(
        window_size_samples=30,
        min_samples_to_fit=5,
        target_ram_floor_mb=500.0,
        min_drain_rate_mb_s=1.0,
    )

    report = evaluate_detector(
        predict_fn=lambda s: baseline.predict_sample(s, horizon_sec=60.0).is_alert,
        episodes=episodes,
        horizon_sec=60.0,
        reset_fn=baseline.reset,
    )

    assert report.total_episodes == 4
    assert report.failure_episodes == 2
    assert report.normal_episodes == 2
    assert report.episode_recall == 1.0  # Both failure episodes detected
    assert report.lead_time_median_s > 0.0  # Positive early warning lead time
    assert report.false_alarms_per_hour == 0.0  # Normal episodes have 0 false alarms

