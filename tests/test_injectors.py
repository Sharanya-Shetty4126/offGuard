"""Unit tests for controlled failure injectors and safety circuit breakers."""

import os
import pytest

from agent.config import AgentConfig
from training.injectors import (
    CpuStarvationInjector,
    DiskThrashInjector,
    InjectionManifest,
    MemoryLeakInjector,
    _get_available_ram_mb,
)


def test_seed_reproducibility():
    """Verify that specifying the same random seed produces identical parameters."""
    inj1 = MemoryLeakInjector(seed=42)
    inj2 = MemoryLeakInjector(seed=42)

    assert inj1.seed == inj2.seed
    assert inj1.leak_rate_mb_s == inj2.leak_rate_mb_s
    assert inj1.start_delay_sec == inj2.start_delay_sec

    # Different seed should yield different parameters
    inj3 = MemoryLeakInjector(seed=999)
    assert inj1.leak_rate_mb_s != inj3.leak_rate_mb_s or inj1.start_delay_sec != inj3.start_delay_sec


def test_memory_leak_injector_short_run():
    """Verify that a brief memory leak allocates, frees memory, and exports valid manifest."""
    injector = MemoryLeakInjector(
        leak_rate_mb_s=10.0,
        start_delay_sec=0.1,
        target_duration_sec=0.8,
        seed=123,
    )

    manifest = injector.run()

    assert isinstance(manifest, InjectionManifest)
    assert manifest.injector_type == "memory_leak"
    assert manifest.seed == 123
    assert manifest.duration_seconds > 0.0
    assert manifest.allocated_peak_mb > 0.0
    assert not manifest.safety_aborted
    # Memory must be cleanly released upon completion
    assert len(injector._allocated_blocks) == 0


def test_ram_floor_circuit_breaker():
    """Verify that if free RAM drops below the safety floor, injection halts immediately."""
    current_avail = _get_available_ram_mb()
    # Set an artificial safety floor well above current free RAM
    artificial_floor = current_avail + 2000.0
    config = AgentConfig(min_safe_ram_mb=artificial_floor)

    injector = MemoryLeakInjector(
        leak_rate_mb_s=10.0,
        start_delay_sec=0.1,
        target_duration_sec=5.0,
        config=config,
    )

    manifest = injector.run()

    # Must abort cleanly without crashing the machine
    assert manifest.safety_aborted
    assert manifest.exit_reason == "ram_floor_breached"
    assert len(injector._allocated_blocks) == 0


def test_cpu_starvation_injector_short_run():
    """Verify that CPU burner launches worker threads and exits cleanly on timeout."""
    injector = CpuStarvationInjector(
        worker_count=1,
        start_delay_sec=0.1,
        target_duration_sec=0.5,
        seed=456,
    )

    manifest = injector.run()

    assert isinstance(manifest, InjectionManifest)
    assert manifest.injector_type == "cpu_starvation"
    assert manifest.seed == 456
    assert manifest.duration_seconds >= 0.4
    assert manifest.exit_reason in ("target_duration_reached", "completed")
    assert not manifest.safety_aborted


def test_disk_thrash_injector_short_run():
    """Verify disk write bursts execute and clean up temporary directory."""
    injector = DiskThrashInjector(
        block_size_mb=1.0,
        start_delay_sec=0.1,
        target_duration_sec=0.5,
        seed=789,
    )
    temp_dir = injector._temp_dir

    manifest = injector.run()

    assert isinstance(manifest, InjectionManifest)
    assert manifest.injector_type == "disk_thrash"
    assert manifest.allocated_peak_mb > 0.0
    # Directory should be cleaned up
    assert not os.path.exists(temp_dir)

