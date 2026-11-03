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


def test_evaluate_episode_lead_time_and_effective_criterion():
    """Verify lead time and the >= 60s effective warning criterion."""
    samples = _create_mock_episode(
        "ep_lead_time", total_seconds=150, failure_second=120, has_failure=True
    )
    # Failure is at index 120 (t = 620.0).
    # Case A: Alert at index 40 (t = 540.0) -> Lead time = 620 - 540 = 80s (>= 60s -> Effective!)
    preds_effective = [1 if i >= 40 else 0 for i in range(150)]
    res_eff = evaluate_episode(samples, preds_effective, horizon_sec=120.0, min_effective_lead_time_sec=60.0)

    assert res_eff.has_failed
    assert res_eff.lead_time_seconds == 80.0
    assert res_eff.detected_in_advance
    assert res_eff.effective_warning_ge_60s

    # Case B: Alert at index 105 (t = 605.0) -> Lead time = 620 - 605 = 15s (< 60s -> Late warning)
    preds_late = [1 if i >= 105 else 0 for i in range(150)]
    res_late = evaluate_episode(samples, preds_late, horizon_sec=120.0, min_effective_lead_time_sec=60.0)

    assert res_late.has_failed
    assert res_late.lead_time_seconds == 15.0
    assert res_late.detected_in_advance
    assert not res_late.effective_warning_ge_60s  # Too late for cashier queue diversion!


def test_evaluate_detector_with_baselines():
    """Verify end-to-end multi-episode benchmark on TimeToExhaustionBaseline."""
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
        min_effective_lead_time_sec=30.0,
        reset_fn=baseline.reset,
    )

    assert report.total_episodes == 4
    assert report.failure_episodes == 2
    assert report.normal_episodes == 2
    assert report.episode_recall == 1.0
    assert report.lead_time_median_s > 0.0
    assert report.false_alarms_per_hour == 0.0
    assert "effective_recall_ge_60s" in report.summary_dict()
