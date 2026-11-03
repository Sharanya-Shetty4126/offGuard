"""Unit tests for the offGuard replay telemetry collector."""

import pytest

from agent.collector import TelemetrySample
from agent.replay import ReplayTelemetryCollector


def _create_synthetic_records():
    return [
        {
            "timestamp": 1000.0,
            "timestamp_iso": "2026-11-03T12:00:00Z",
            "episode_id": "test_ep",
            "cpu_percent": 15.0,
            "mem_percent": 45.0,
            "mem_available_mb": 5000.0,
            "swap_percent": 10.0,
            "commit_percent": 30.0,
            "commit_available_mb": 8000.0,
            "ctx_switches_s": 15000.0,
            "disk_read_mb_s": 0.5,
            "disk_write_mb_s": 1.2,
            "heartbeat_lag_s": 0.01,
            "top_proc_name": "python.exe",
            "top_proc_cpu": 5.0,
            "top_proc_mem_mb": 120.0,
        },
        {
            "timestamp": 1001.0,
            "timestamp_iso": "2026-11-03T12:00:01Z",
            "episode_id": "test_ep",
            "cpu_percent": 25.0,
            "mem_percent": 50.0,
            "mem_available_mb": 4500.0,
            "swap_percent": 10.0,
            "commit_percent": 32.0,
            "commit_available_mb": 7800.0,
            "ctx_switches_s": 18000.0,
            "disk_read_mb_s": 1.0,
            "disk_write_mb_s": 2.0,
            "heartbeat_lag_s": 0.02,
            "top_proc_name": "python.exe",
            "top_proc_cpu": 10.0,
            "top_proc_mem_mb": 150.0,
        },
    ]


def test_replay_collector_basic():
    """Verify loading from sample list and sequential collection."""
    records = _create_synthetic_records()
    collector = ReplayTelemetryCollector(source=records, realtime=False)

    assert collector.total_samples == 2
    assert collector.has_next

    s1 = collector.collect_sample()
    assert isinstance(s1, TelemetrySample)
    assert s1.timestamp == 1000.0
    assert s1.cpu_percent == 15.0
    assert s1.commit_percent == 30.0

    s2 = collector.collect_sample()
    assert s2.timestamp == 1001.0
    assert s2.cpu_percent == 25.0

    # Exhausted
    s3 = collector.collect_sample()
    assert s3 is None
    assert not collector.has_next


def test_replay_reset_and_iterator():
    """Verify reset function and iteration."""
    records = _create_synthetic_records()
    collector = ReplayTelemetryCollector(source=records, realtime=False)

    samples = list(collector)
    assert len(samples) == 2

    # Reset
    collector.reset()
    assert collector.current_index == 0
    assert collector.has_next

