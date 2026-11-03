"""Episode recorder and manifest generator for offGuard dataset collection.

Runs the TelemetryCollector in tandem with a workload (failure injector, hard negative,
or routine idle) and records a synchronized, labeled episode dataset.

Exports:
1. data/episodes/<episode_id>/telemetry.csv (Raw telemetry + ground-truth labels)
2. data/episodes/<episode_id>/manifest.json (Reproducibility & failure ground-truth metadata)
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

from agent.collector import TelemetryCollector, TelemetrySample
from agent.config import DEFAULT_CONFIG, AgentConfig
from training.hard_negatives import (
    AntivirusScanWorkload,
    AppStartupWorkload,
    FileExportWorkload,
)
from training.injectors import (
    CpuStarvationInjector,
    DiskThrashInjector,
    MemoryLeakInjector,
)
from training.labeling import FailureGroundTruth, label_episode_samples


@dataclass
class EpisodeManifest:
    """Comprehensive manifest describing a recorded telemetry episode."""

    episode_id: str
    scenario_type: str  # "memory_leak", "cpu_starvation", "disk_thrash", "app_startup", "file_export", "antivirus_burst", "routine_idle"
    created_at_utc: str
    duration_seconds: float
    total_samples: int
    has_failed: bool
    failure_timestamp: Optional[float]
    failure_mode: str
    pre_failure_samples: int
    post_failure_samples: int
    workload_manifest: Optional[Dict[str, Any]]
    telemetry_csv_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def record_episode(
    scenario_type: str = "routine_idle",
    duration_sec: float = 30.0,
    episode_id: Optional[str] = None,
    output_dir: str = "data/episodes",
    sample_interval_sec: float = 1.0,
    seed: Optional[int] = None,
    config: Optional[AgentConfig] = None,
) -> tuple[List[Dict[str, Any]], EpisodeManifest]:
    """Record a single telemetry episode with a concurrent scenario workload."""
    cfg = config or DEFAULT_CONFIG
    ep_id = episode_id or f"ep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{scenario_type}"
    collector = TelemetryCollector(config=cfg, episode_id=ep_id)

    # Initialize requested scenario workload
    workload = None
    if scenario_type == "memory_leak":
        workload = MemoryLeakInjector(target_duration_sec=duration_sec, seed=seed, config=cfg)
    elif scenario_type == "cpu_starvation":
        workload = CpuStarvationInjector(target_duration_sec=duration_sec, seed=seed, config=cfg)
    elif scenario_type == "disk_thrash":
        workload = DiskThrashInjector(target_duration_sec=duration_sec, seed=seed, config=cfg)
    elif scenario_type == "app_startup":
        workload = AppStartupWorkload(total_duration_sec=duration_sec, seed=seed)
    elif scenario_type == "file_export":
        workload = FileExportWorkload(total_duration_sec=duration_sec, seed=seed)
    elif scenario_type == "antivirus_burst":
        workload = AntivirusScanWorkload(total_duration_sec=duration_sec, seed=seed)
    elif scenario_type == "routine_idle":
        workload = None
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    raw_samples: List[TelemetrySample] = []
    workload_manifest_dict: Optional[Dict[str, Any]] = None

    # Launch workload in separate thread if present
    workload_thread: Optional[threading.Thread] = None
    if workload is not None:
        def _runner():
            nonlocal workload_manifest_dict
            manifest = workload.run()
            workload_manifest_dict = manifest.to_dict()

        workload_thread = threading.Thread(target=_runner, daemon=True)
        workload_thread.start()

    start_mono = time.monotonic()
    print(f"\n[Episode Recorder] Recording '{ep_id}' ({scenario_type}) for {duration_sec}s...")

    try:
        while (time.monotonic() - start_mono) < duration_sec:
            tick_start = time.perf_counter()
            sample = collector.collect_sample()
            raw_samples.append(sample)

            elapsed_tick = time.perf_counter() - tick_start
            sleep_time = max(0.001, sample_interval_sec - elapsed_tick)
            time.sleep(sleep_time)

            if workload_thread and not workload_thread.is_alive():
                # Workload finished or safety-aborted early
                pass
    finally:
        if workload and hasattr(workload, "stop"):
            workload.stop()
        if workload_thread:
            workload_thread.join(timeout=2.0)

    # Ground-truth labeling and post-failure/recovery masking
    proxy_ts = None
    proxy_mode = None
    cutoff_ts = None
    if workload_manifest_dict:
        if workload_manifest_dict.get("safety_aborted"):
            proxy_ts = workload_manifest_dict.get("end_timestamp")
            proxy_mode = f"proxy_{workload_manifest_dict.get('exit_reason', 'breaker_trip')}"
        cutoff_ts = workload_manifest_dict.get("end_timestamp")

    labeled_samples, ground_truth = label_episode_samples(
        raw_samples,
        horizon_sec=cfg.default_horizon_sec,
        config=cfg,
        drop_post_failure=False,
        proxy_failure_timestamp=proxy_ts,
        proxy_failure_mode=proxy_mode,
        recovery_cutoff_timestamp=cutoff_ts,
    )

    # Prepare output directory
    target_dir = os.path.join(output_dir, ep_id)
    os.makedirs(target_dir, exist_ok=True)
    csv_filename = os.path.join(target_dir, "telemetry.csv")
    manifest_filename = os.path.join(target_dir, "manifest.json")

    # Save to CSV
    if labeled_samples:
        keys = list(labeled_samples[0].keys())
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(labeled_samples)

    actual_duration = time.monotonic() - start_mono
    manifest = EpisodeManifest(
        episode_id=ep_id,
        scenario_type=scenario_type,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(actual_duration, 2),
        total_samples=len(labeled_samples),
        has_failed=ground_truth.has_failed,
        failure_timestamp=ground_truth.failure_timestamp,
        failure_mode=ground_truth.failure_mode,
        pre_failure_samples=ground_truth.pre_failure_samples,
        post_failure_samples=ground_truth.post_failure_samples,
        workload_manifest=workload_manifest_dict,
        telemetry_csv_path=csv_filename,
    )

    with open(manifest_filename, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    print(
        f"[Episode Recorder] Saved: {len(labeled_samples)} samples to '{csv_filename}'. "
        f"Failure Detected: {manifest.has_failed} (Mode: {manifest.failure_mode})\n"
    )

    return labeled_samples, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a telemetry episode")
    parser.add_argument(
        "--type",
        type=str,
        default="routine_idle",
        choices=[
            "routine_idle",
            "memory_leak",
            "cpu_starvation",
            "disk_thrash",
            "app_startup",
            "file_export",
            "antivirus_burst",
        ],
        help="Scenario workload type",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="Episode duration in seconds")
    parser.add_argument("--out", type=str, default="data/episodes", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    record_episode(
        scenario_type=args.type,
        duration_sec=args.duration,
        output_dir=args.out,
        seed=args.seed,
    )

