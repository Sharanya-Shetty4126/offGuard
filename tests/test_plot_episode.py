"""Unit tests for the episode plotting and visualization module."""

import os
import tempfile
import pytest

from training.collect_episodes import record_episode
from training.plot_episode import plot_episode


def test_plot_episode_generation():
    """Verify rendering a telemetry plot from an episode directory to PNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Record a mini episode
        samples, manifest = record_episode(
            scenario_type="routine_idle",
            duration_sec=1.2,
            episode_id="ep_test_plot",
            output_dir=tmpdir,
            sample_interval_sec=0.2,
        )

        ep_dir = os.path.join(tmpdir, "ep_test_plot")
        out_png = os.path.join(tmpdir, "diagnostic_plot.png")

        # 2. Render plot
        result_path = plot_episode(ep_dir, output_png_path=out_png)

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # Valid PNG image generated

