"""Unit tests for StaticThresholdBaseline and TimeToExhaustionBaseline."""

import math
import pytest

from agent.baselines import StaticThresholdBaseline, TimeToExhaustionBaseline


def test_static_threshold_baseline_alert_streak():
    """Verify static threshold triggers only after required consecutive ticks."""
    baseline = StaticThresholdBaseline(
        mem_percent_threshold=90.0,
        swap_percent_threshold=80.0,
        consecutive_ticks=3,
    )

    # 1. Normal state
    res1 = baseline.predict_sample({"mem_percent": 85.0, "swap_percent": 50.0})
    assert not res1.is_alert
    assert res1.risk_score == 0.0

    # 2. Tick 1 of breach
    res2 = baseline.predict_sample({"mem_percent": 92.0, "swap_percent": 85.0})
    assert not res2.is_alert  # Only 1 tick, needs 3
    assert res2.risk_score == 0.5

    # 3. Tick 2 of breach
    res3 = baseline.predict_sample({"mem_percent": 93.0, "swap_percent": 86.0})
    assert not res3.is_alert  # Only 2 ticks

    # 4. Tick 3 of breach -> Should alert!
    res4 = baseline.predict_sample({"mem_percent": 94.0, "swap_percent": 88.0})
    assert res4.is_alert
    assert res4.risk_score == 1.0

    # 5. Recovery clears the alert
    res5 = baseline.predict_sample({"mem_percent": 70.0, "swap_percent": 40.0})
    assert not res5.is_alert
    assert res5.risk_score == 0.0


def test_time_to_exhaustion_baseline_linear_drain():
    """Verify linear trend fitting detects memory leak and projects time to exhaustion."""
    baseline = TimeToExhaustionBaseline(
        window_size_samples=30,
        min_samples_to_fit=5,
        target_ram_floor_mb=500.0,
        min_drain_rate_mb_s=1.0,
    )

    # Simulate memory draining at 10 MB/second from 2000 MB:
    # At t=0: 2000 MB. Floor is 500 MB. Headroom is 1500 MB.
    # At 10 MB/s, exhaustion to 500 MB should occur at 150 seconds.
    samples = []
    base_t = 100.0
    for i in range(10):
        t = base_t + i
        avail = 2000.0 - 10.0 * i
        pred = baseline.predict_sample(
            {"timestamp": t, "mem_available_mb": avail}, horizon_sec=120.0
        )
        samples.append(pred)

    last_pred = samples[-1]
    # At sample 9 (t=109), avail = 1910 MB. Headroom = 1410 MB.
    # Expected time to exhaustion = 1410 / 10 = ~141 seconds.
    # Since 141s > 120s horizon, is_alert should still be False
    assert not last_pred.is_alert
    assert last_pred.metadata["time_to_exhaustion_s"] is not None
    assert abs(last_pred.metadata["time_to_exhaustion_s"] - 141.0) < 5.0
    assert abs(last_pred.metadata["slope_mb_s"] - (-10.0)) < 0.1

    # Now let memory drain down closer to the horizon (e.g. avail reaches 1500 MB -> headroom 1000 MB -> 100s to floor)
    pred_horizon = None
    for i in range(10, 60):
        t = base_t + i
        avail = 2000.0 - 10.0 * i
        pred_horizon = baseline.predict_sample(
            {"timestamp": t, "mem_available_mb": avail}, horizon_sec=120.0
        )

    # At i=50 (avail = 1500 MB, headroom = 1000 MB), time to floor = 100s <= 120s horizon -> Alert!
    assert pred_horizon is not None
    assert pred_horizon.is_alert
    assert pred_horizon.risk_score > 0.0
    assert pred_horizon.metadata["time_to_exhaustion_s"] <= 120.0


def test_time_to_exhaustion_stable_memory_no_alert():
    """Verify that stable or oscillating memory does not trigger false alerts."""
    baseline = TimeToExhaustionBaseline(
        window_size_samples=30,
        min_samples_to_fit=5,
        target_ram_floor_mb=500.0,
        min_drain_rate_mb_s=1.0,
    )

    pred = None
    for i in range(20):
        t = 100.0 + i
        # Memory fluctuates mildly around 3000 MB
        avail = 3000.0 + (5.0 if i % 2 == 0 else -5.0)
        pred = baseline.predict_sample(
            {"timestamp": t, "mem_available_mb": avail}, horizon_sec=120.0
        )

    assert pred is not None
    assert not pred.is_alert
    assert pred.risk_score == 0.0
    assert pred.metadata["time_to_exhaustion_s"] is None

