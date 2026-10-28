"""Unit tests for the offGuard telemetry collector."""

import time
import pytest

from agent.collector import TelemetryCollector, TelemetrySample
from agent.config import AgentConfig


def test_collect_single_sample():
    """Verify that a single collected sample adheres to schema types and valid ranges."""
    config = AgentConfig(sample_interval_sec=0.5)
    collector = TelemetryCollector(config=config, episode_id="test_ep_01")

    sample = collector.collect_sample()

    assert isinstance(sample, TelemetrySample)
    assert sample.episode_id == "test_ep_01"
    assert 0.0 <= sample.cpu_percent <= 100.0
    assert 0.0 <= sample.mem_percent <= 100.0
    assert sample.mem_available_mb > 0.0
    assert 0.0 <= sample.swap_percent <= 100.0
    assert sample.disk_read_mb_s >= 0.0
    assert sample.disk_write_mb_s >= 0.0
    assert sample.heartbeat_lag_s >= 0.0
    assert isinstance(sample.top_proc_name, str)
    assert sample.top_proc_cpu >= 0.0
    assert sample.top_proc_mem_mb >= 0.0
    assert abs(time.time() - sample.timestamp) < 5.0


def test_sample_dict_conversion():
    """Verify serialization to dictionary preserves all schema fields."""
    collector = TelemetryCollector()
    sample = collector.collect_sample()
    d = sample.to_dict()

    expected_keys = {
        "timestamp",
        "timestamp_iso",
        "episode_id",
        "cpu_percent",
        "mem_percent",
        "mem_available_mb",
        "swap_percent",
        "disk_read_mb_s",
        "disk_write_mb_s",
        "heartbeat_lag_s",
        "top_proc_name",
        "top_proc_cpu",
        "top_proc_mem_mb",
    }
    assert expected_keys.issubset(d.keys())


def test_lag_tracking_on_delay():
    """Verify that intentional scheduler delay is properly captured as heartbeat lag."""
    config = AgentConfig(sample_interval_sec=0.2)
    collector = TelemetryCollector(config=config)

    # First tick sets target_next_tick = now + 0.2
    _ = collector.collect_sample()

    # Sleep intentionally longer than 0.2s (e.g. 0.5s) to simulate a starved event loop
    time.sleep(0.5)

    delayed_sample = collector.collect_sample()
    # Lag should be approximately 0.5 - 0.2 = ~0.3s
    assert delayed_sample.heartbeat_lag_s >= 0.15
