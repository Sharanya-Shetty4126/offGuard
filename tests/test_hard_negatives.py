"""Unit tests for hard-negative retail counter workload simulators."""

import os
import pytest

from training.hard_negatives import (
    AntivirusScanWorkload,
    AppStartupWorkload,
    FileExportWorkload,
    HardNegativeManifest,
)


def test_app_startup_workload():
    """Verify app startup allocates temporary memory, plateaus, and frees upon exit."""
    workload = AppStartupWorkload(
        allocate_mb=15.0,
        startup_duration_sec=0.2,
        total_duration_sec=0.5,
        seed=101,
    )

    manifest = workload.run()

    assert isinstance(manifest, HardNegativeManifest)
    assert manifest.workload_type == "app_startup"
    assert not manifest.is_failure_expected
    assert manifest.peak_ram_mb == 15.0
    assert manifest.duration_seconds >= 0.4
    # Ensure memory is released
    assert len(workload._allocated_chunks) == 0


def test_file_export_workload():
    """Verify file export burst generates temporary I/O and cleans up disk artifacts."""
    workload = FileExportWorkload(
        file_size_mb=2.0,
        burst_duration_sec=0.2,
        total_duration_sec=0.5,
        seed=202,
    )
    temp_dir = workload._temp_dir

    manifest = workload.run()

    assert isinstance(manifest, HardNegativeManifest)
    assert manifest.workload_type == "file_export"
    assert not manifest.is_failure_expected
    # Temporary directory must be deleted
    assert not os.path.exists(temp_dir)


def test_antivirus_scan_workload():
    """Verify antivirus scan simulation runs threads and exits cleanly."""
    workload = AntivirusScanWorkload(
        scan_duration_sec=0.2,
        total_duration_sec=0.5,
        worker_count=1,
        seed=303,
    )

    manifest = workload.run()

    assert isinstance(manifest, HardNegativeManifest)
    assert manifest.workload_type == "antivirus_burst"
    assert not manifest.is_failure_expected
    assert manifest.duration_seconds >= 0.4

