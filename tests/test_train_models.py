"""Unit tests for ML model training, scenario holdout splitting, and benchmarking."""

import os
import tempfile
import pytest

from training.train_models import (
    ModelArtifact,
    run_benchmark_comparison,
    split_episodes,
    train_models,
)


def _generate_synthetic_episode_pool():
    """Create a balanced pool of mock episodes across scenarios."""
    episodes = []

    # 1. Memory leak failure episodes (4 episodes)
    for ep_idx in range(4):
        ep = []
        base_t = 1000.0 + ep_idx * 500
        for i in range(120):
            ep.append(
                {
                    "timestamp": base_t + i,
                    "episode_id": f"ep_mem_{ep_idx}",
                    "scenario_type": "memory_leak",
                    "mem_percent": min(98.0, 45.0 + 0.45 * i),
                    "commit_percent": min(95.0, 40.0 + 0.4 * i),
                    "cpu_percent": 30.0,
                    "ctx_switches_s": 15000.0,
                    "disk_write_mb_s": 0.5,
                    "heartbeat_lag_s": 3.5 if i >= 100 else 0.02,
                    "target_fail_within_H": 1 if 40 <= i < 100 else 0,
                    "is_post_failure": i >= 100,
                }
            )
        episodes.append(ep)

    # 2. Routine normal & hard-negative episodes (4 episodes)
    for ep_idx in range(4):
        ep = []
        base_t = 5000.0 + ep_idx * 500
        scen = "app_startup" if ep_idx % 2 == 0 else "routine_idle"
        for i in range(100):
            # Temporary burst then plateau
            mem_p = 50.0 + (15.0 if i >= 10 else 0.0)
            ep.append(
                {
                    "timestamp": base_t + i,
                    "episode_id": f"ep_norm_{ep_idx}",
                    "scenario_type": scen,
                    "mem_percent": mem_p,
                    "commit_percent": 45.0,
                    "cpu_percent": 60.0 if (10 <= i <= 20) else 15.0,
                    "ctx_switches_s": 12000.0,
                    "disk_write_mb_s": 0.1,
                    "heartbeat_lag_s": 0.02,
                    "target_fail_within_H": 0,
                    "is_post_failure": False,
                }
            )
        episodes.append(ep)

    # 3. Disk thrash holdout episodes (2 episodes)
    for ep_idx in range(2):
        ep = []
        base_t = 9000.0 + ep_idx * 500
        for i in range(110):
            ep.append(
                {
                    "timestamp": base_t + i,
                    "episode_id": f"ep_thrash_{ep_idx}",
                    "scenario_type": "disk_thrash",
                    "mem_percent": 55.0,
                    "commit_percent": 50.0,
                    "cpu_percent": 40.0,
                    "ctx_switches_s": 22000.0,
                    "disk_write_mb_s": 45.0 if i >= 30 else 1.0,
                    "heartbeat_lag_s": 3.2 if i >= 90 else 0.02,
                    "target_fail_within_H": 1 if 30 <= i < 90 else 0,
                    "is_post_failure": i >= 90,
                }
            )
        episodes.append(ep)

    return episodes


def test_split_episodes_no_row_leakage():
    """Verify that episode splitting prevents any overlapping episodes across train/val/test."""
    episodes = _generate_synthetic_episode_pool()
    train_eps, val_eps, test_eps = split_episodes(episodes, train_ratio=0.6, val_ratio=0.2, seed=42)

    train_ids = {ep[0]["episode_id"] for ep in train_eps}
    val_ids = {ep[0]["episode_id"] for ep in val_eps}
    test_ids = {ep[0]["episode_id"] for ep in test_eps}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scenario_holdout():
    """Verify holding out an entire scenario type (e.g. disk_thrash) places it only in test set."""
    episodes = _generate_synthetic_episode_pool()
    train_eps, val_eps, test_eps = split_episodes(
        episodes, holdout_scenario="disk_thrash", seed=42
    )

    train_scenarios = {ep[0]["scenario_type"] for ep in train_eps}
    val_scenarios = {ep[0]["scenario_type"] for ep in val_eps}
    test_scenarios = {ep[0]["scenario_type"] for ep in test_eps}

    assert "disk_thrash" not in train_scenarios
    assert "disk_thrash" not in val_scenarios
    assert "disk_thrash" in test_scenarios


def test_train_models_and_benchmark():
    """Verify end-to-end training, calibration, and 4-way benchmark execution."""
    episodes = _generate_synthetic_episode_pool()

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        artifacts = train_models(
            episodes,
            include_lag=True,
            output_model_dir=tmp_model_dir,
            seed=42,
        )

        assert "logistic_regression" in artifacts
        assert "gradient_boosting" in artifacts

        # Test artifact inference
        lr_art = artifacts["logistic_regression"]
        dummy_feat = {k: 0.5 for k in lr_art.feature_names}
        is_alert, risk = lr_art.predict_sample(dummy_feat)
        assert isinstance(is_alert, bool)
        assert 0.0 <= risk <= 1.0

        # Run benchmark on test episodes
        _, _, test_eps = split_episodes(episodes, train_ratio=0.6, val_ratio=0.2, seed=42)
        reports = run_benchmark_comparison(test_eps, artifacts)

        assert "static_threshold" in reports
        assert "time_to_exhaustion" in reports
        assert "logistic_regression" in reports
        assert "gradient_boosting" in reports

        for name, rep in reports.items():
            assert rep.total_episodes == len(test_eps)
            assert 0.0 <= rep.episode_recall <= 1.0
            assert rep.false_alarms_per_hour >= 0.0

