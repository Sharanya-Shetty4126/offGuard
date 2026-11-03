"""Unit tests for automated episode recording and dataset generation."""

import json
import os
import tempfile
import pytest

from training.collect_episodes import record_episode


def test_record_routine_idle_episode():
    """Verify recording a routine idle baseline episode exports CSV and manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples, manifest = record_episode(
            scenario_type="routine_idle",
            duration_sec=1.5,
            episode_id="test_ep_idle_01",
            output_dir=tmpdir,
            sample_interval_sec=0.2,
        )

        assert len(samples) >= 4
        assert manifest.episode_id == "test_ep_idle_01"
        assert not manifest.has_failed
        assert manifest.failure_timestamp is None
        assert manifest.pre_failure_samples == 0

        # Verify disk outputs
        csv_path = os.path.join(tmpdir, "test_ep_idle_01", "telemetry.csv")
        manifest_path = os.path.join(tmpdir, "test_ep_idle_01", "manifest.json")
        assert os.path.exists(csv_path)
        assert os.path.exists(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["scenario_type"] == "routine_idle"
            assert not data["has_failed"]


def test_record_hard_negative_episode():
    """Verify recording an app startup hard negative episode exports workload manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples, manifest = record_episode(
            scenario_type="app_startup",
            duration_sec=1.5,
            episode_id="test_ep_app_startup",
            output_dir=tmpdir,
            sample_interval_sec=0.2,
            seed=42,
        )

        assert len(samples) >= 4
        assert manifest.scenario_type == "app_startup"
        assert not manifest.has_failed
        assert manifest.workload_manifest is not None
        assert manifest.workload_manifest["workload_type"] == "app_startup"
