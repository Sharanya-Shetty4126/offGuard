"""Unit tests for the rolling feature pipeline, temporal leakage, and parity."""

import pytest

from agent.features import RollingFeaturePipeline, compute_ols_slope


def _create_mock_stream(length: int = 40):
    samples = []
    base_t = 1000.0
    for i in range(length):
        samples.append(
            {
                "timestamp": base_t + i,
                "mem_percent": 50.0 + 0.5 * i,  # Steady climb of +0.5% / sec
                "commit_percent": 40.0 + 0.3 * i,
                "cpu_percent": 20.0 + (5.0 if i % 2 == 0 else -5.0),
                "ctx_switches_s": 10000.0 + 100 * i,
                "disk_write_mb_s": 1.0,
                "heartbeat_lag_s": 0.02,
            }
        )
    return samples


def test_ols_slope_accuracy():
    """Verify OLS slope returns exact rate of change per second."""
    t = [10.0, 11.0, 12.0, 13.0, 14.0]
    v = [100.0, 102.5, 105.0, 107.5, 110.0]  # Exact slope: +2.5 / s

    slope = compute_ols_slope(t, v)
    assert abs(slope - 2.5) < 1e-4

    # Flat line has 0 slope
    assert compute_ols_slope(t, [50.0] * 5) == 0.0


def test_offline_online_parity():
    """Verify streaming push_sample exactly matches batch transform_episode."""
    samples = _create_mock_stream(length=30)
    pipeline_stream = RollingFeaturePipeline(windows_sec=[10, 30])
    pipeline_batch = RollingFeaturePipeline(windows_sec=[10, 30])

    # 1. Stream
    stream_features = [pipeline_stream.push_sample(s) for s in samples]

    # 2. Batch
    batch_features = pipeline_batch.transform_episode(samples)

    assert len(stream_features) == len(batch_features)
    for f_stream, f_batch in zip(stream_features, batch_features):
        assert f_stream == f_batch


def test_zero_future_leakage():
    """Verify that perturbing future samples after t does NOT change features at t."""
    samples_clean = _create_mock_stream(length=25)
    samples_perturbed = _create_mock_stream(length=25)

    # Heavily corrupt samples after index 15
    for i in range(16, 25):
        samples_perturbed[i]["mem_percent"] = 99.9
        samples_perturbed[i]["heartbeat_lag_s"] = 15.0

    pipeline_clean = RollingFeaturePipeline(windows_sec=[10, 30])
    pipeline_pert = RollingFeaturePipeline(windows_sec=[10, 30])

    clean_feats = pipeline_clean.transform_episode(samples_clean)
    pert_feats = pipeline_pert.transform_episode(samples_perturbed)

    # Features up to index 15 must be 100% IDENTICAL
    for i in range(16):
        assert clean_feats[i] == pert_feats[i], f"Temporal leakage detected at index {i}!"


def test_warmup_fill_tracking():
    """Verify that window fill ratio and filled flags track warmup accurately."""
    pipeline = RollingFeaturePipeline(windows_sec=[10, 60])
    base_t = 100.0

    feats = []
    for i in range(61):
        feats.append(pipeline.push_sample({"timestamp": base_t + i, "mem_percent": 50.0 + 0.1 * i}))

    # First sample
    assert feats[0]["window_fill_ratio"] == 0.0
    assert feats[0]["window_filled"] == 0.0

    # Sample at 30 seconds
    assert abs(feats[30]["window_fill_ratio"] - 0.5) <= 0.05
    assert feats[30]["window_filled"] == 0.0

    # Sample at 60 seconds
    assert feats[60]["window_fill_ratio"] == 1.0
    assert feats[60]["window_filled"] == 1.0


def test_lag_ablation_names():
    """Verify get_feature_names properly includes or excludes heartbeat lag."""
    all_names = RollingFeaturePipeline.get_feature_names(windows_sec=[10, 30, 60], include_lag=True)
    no_lag_names = RollingFeaturePipeline.get_feature_names(windows_sec=[10, 30, 60], include_lag=False)

    assert any("lag" in n for n in all_names)
    assert not any("lag" in n for n in no_lag_names)
    assert len(no_lag_names) < len(all_names)


def test_monotonic_priority_and_sleep_gap_detection():
    """Verify monotonic timestamp is prioritized and large gaps (>10s) reset buffer."""
    pipeline = RollingFeaturePipeline(windows_sec=[10, 30], gap_threshold_sec=10.0)

    # Sample with monotonic timestamp
    f1 = pipeline.push_sample({"timestamp": 1700000000.0, "timestamp_mono": 100.0, "mem_percent": 50.0})
    assert f1["gap_detected"] == 0.0

    # Normal tick 1 second later
    f2 = pipeline.push_sample({"timestamp": 1700000001.0, "timestamp_mono": 101.0, "mem_percent": 51.0})
    assert f2["gap_detected"] == 0.0

    # Sleep / suspend jump of 30 seconds
    f3 = pipeline.push_sample({"timestamp": 1700000031.0, "timestamp_mono": 131.0, "mem_percent": 52.0})
    assert f3["gap_detected"] == 1.0
    # Rolling buffer should have reset, so fill ratio should be reset
    assert f3["window_filled"] == 0.0

