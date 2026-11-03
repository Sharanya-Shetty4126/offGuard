"""Synthetic and realistic episode dataset generator for offGuard benchmarking.

Generates a realistic, varied corpus of 40 episodes for training and evaluating:
1. Slow Memory Leaks (1-4 MB/s, 240-360s duration, POS cache leak)
2. Fast Memory Leaks (15-30 MB/s, 120-180s duration)
3. CPU Starvation (POS worker thread loop, CPU pinned 98%, climbing scheduler lag)
4. Disk Thrash (Database vacuum / log saturation)
5. Benign App Startup Plateau (Initial ramp then 3 min flat plateau)
6. Benign Antivirus Burst (45s background scan then idle)
7. Benign File Export / EOD report (30s burst)
8. Normal Cashier Shift / Soak (Continuous normal operation, jitter, no failure)

All episodes are written to data/episodes/<episode_id>/ with telemetry.csv and manifest.json.
"""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from agent.config import DEFAULT_CONFIG, AgentConfig
from training.collect_episodes import EpisodeManifest
from training.labeling import label_episode_samples


def _generate_slow_memory_leak(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Simulate a gradual retail POS memory leak (e.g. unclosed barcode bitmap handles)."""
    rng = random.Random(seed)
    samples = []
    leak_rate_mb_s = rng.uniform(1.2, 3.5)  # Realistic slow leak
    total_ram_mb = 8192.0
    start_free_ram = rng.uniform(4200.0, 5200.0)

    failure_time_idx = duration_s - int(rng.uniform(20, 40))

    current_free = start_free_ram
    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        # Gradual drain
        current_free = max(150.0, current_free - leak_rate_mb_s + rng.gauss(0, 0.4))
        mem_pct = round(((total_ram_mb - current_free) / total_ram_mb) * 100.0, 1)
        commit_pct = min(99.0, round(mem_pct * rng.uniform(0.92, 0.98), 1))

        # Scheduler lag: stays low until near-exhaustion (< 300 MB free)
        if i >= failure_time_idx:
            lag = round(rng.uniform(3.2, 5.5), 3)  # Outright freeze
        elif current_free < 400.0:
            lag = round(rng.uniform(0.4, 1.8), 3)  # Severe thrashing
        elif current_free < 800.0:
            lag = round(rng.uniform(0.1, 0.35), 3)
        else:
            lag = round(max(0.005, rng.gauss(0.015, 0.005)), 3)

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "memory_leak",
                "cpu_percent": round(min(95.0, max(5.0, 18.0 + rng.gauss(0, 4.0))), 1),
                "mem_percent": mem_pct,
                "mem_available_mb": round(current_free, 1),
                "swap_percent": commit_pct,
                "commit_percent": commit_pct,
                "commit_available_mb": round(current_free * 1.5, 1),
                "disk_read_mb_s": round(max(0.0, rng.gauss(0.8, 0.2)), 2),
                "disk_write_mb_s": round(max(0.0, rng.gauss(1.2, 0.3)), 2),
                "heartbeat_lag_s": lag,
                "top_proc_name": "pos_terminal.exe",
                "top_proc_cpu": 12.0,
                "top_proc_mem_mb": round(total_ram_mb - current_free, 1),
                "ctx_switches_s": round(max(5000.0, 14000.0 + rng.gauss(0, 1000)), 1),
            }
        )
    return samples


def _generate_fast_memory_leak(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 150,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Simulate a rapid memory leak (e.g. runaway unbounded cache loop)."""
    rng = random.Random(seed)
    samples = []
    leak_rate_mb_s = rng.uniform(18.0, 32.0)
    total_ram_mb = 8192.0
    start_free_ram = rng.uniform(3500.0, 4500.0)

    failure_time_idx = duration_s - int(rng.uniform(20, 35))

    current_free = start_free_ram
    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        current_free = max(120.0, current_free - leak_rate_mb_s + rng.gauss(0, 1.0))
        mem_pct = round(((total_ram_mb - current_free) / total_ram_mb) * 100.0, 1)
        commit_pct = min(99.0, round(mem_pct * 0.95, 1))

        if i >= failure_time_idx:
            lag = round(rng.uniform(3.5, 6.0), 3)
        elif current_free < 400.0:
            lag = round(rng.uniform(0.6, 2.2), 3)
        else:
            lag = round(max(0.005, rng.gauss(0.02, 0.008)), 3)

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "memory_leak",
                "cpu_percent": round(min(98.0, max(10.0, 35.0 + rng.gauss(0, 5.0))), 1),
                "mem_percent": mem_pct,
                "mem_available_mb": round(current_free, 1),
                "swap_percent": commit_pct,
                "commit_percent": commit_pct,
                "commit_available_mb": round(current_free * 1.2, 1),
                "disk_read_mb_s": round(max(0.0, rng.gauss(1.5, 0.4)), 2),
                "disk_write_mb_s": round(max(0.0, rng.gauss(4.0, 1.0)), 2),
                "heartbeat_lag_s": lag,
                "top_proc_name": "pos_terminal.exe",
                "top_proc_cpu": 28.0,
                "top_proc_mem_mb": round(total_ram_mb - current_free, 1),
                "ctx_switches_s": round(max(8000.0, 22000.0 + rng.gauss(0, 2000)), 1),
            }
        )
    return samples


def _generate_cpu_starvation(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Simulate POS worker thread deadlock / spinning loop starvation."""
    rng = random.Random(seed)
    samples = []
    starve_start_idx = 40
    failure_time_idx = duration_s - int(rng.uniform(15, 30))

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        if i < starve_start_idx:
            cpu = max(5.0, min(30.0, 15.0 + rng.gauss(0, 4.0)))
            ctx = max(5000.0, 12000.0 + rng.gauss(0, 1000))
            lag = max(0.005, rng.gauss(0.015, 0.005))
        elif i >= failure_time_idx:
            cpu = rng.uniform(98.5, 100.0)
            ctx = rng.uniform(75000.0, 95000.0)
            lag = rng.uniform(3.4, 5.8)
        else:
            # Gradual lock-up and queue backup
            progress = (i - starve_start_idx) / (failure_time_idx - starve_start_idx)
            cpu = min(100.0, 80.0 + progress * 20.0 + rng.gauss(0, 1.0))
            ctx = 30000.0 + progress * 50000.0 + rng.gauss(0, 3000)
            lag = 0.05 + progress * 1.8 + rng.gauss(0, 0.1)

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "cpu_starvation",
                "cpu_percent": round(cpu, 1),
                "mem_percent": round(52.0 + rng.gauss(0, 0.5), 1),
                "mem_available_mb": round(3900.0 + rng.gauss(0, 20.0), 1),
                "swap_percent": 45.0,
                "commit_percent": 45.0,
                "commit_available_mb": 5000.0,
                "disk_read_mb_s": 0.2,
                "disk_write_mb_s": 0.5,
                "heartbeat_lag_s": round(max(0.005, lag), 3),
                "top_proc_name": "pos_terminal.exe",
                "top_proc_cpu": round(cpu * 0.9, 1),
                "top_proc_mem_mb": 450.0,
                "ctx_switches_s": round(ctx, 1),
            }
        )
    return samples


def _generate_disk_thrash(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 180,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Simulate disk saturation / database thrashing."""
    rng = random.Random(seed)
    samples = []
    thrash_start = 30
    failure_idx = duration_s - 25

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        if i < thrash_start:
            write_mb = max(0.1, rng.gauss(0.8, 0.3))
            lag = max(0.005, rng.gauss(0.02, 0.005))
            cpu = max(5.0, rng.gauss(15.0, 3.0))
        elif i >= failure_idx:
            write_mb = rng.uniform(75.0, 110.0)
            lag = rng.uniform(3.2, 5.0)
            cpu = rng.uniform(70.0, 85.0)
        else:
            prog = (i - thrash_start) / (failure_idx - thrash_start)
            write_mb = 20.0 + prog * 65.0 + rng.gauss(0, 5.0)
            lag = 0.05 + prog * 1.5 + rng.gauss(0, 0.1)
            cpu = 30.0 + prog * 45.0 + rng.gauss(0, 4.0)

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "disk_thrash",
                "cpu_percent": round(min(95.0, cpu), 1),
                "mem_percent": round(58.0 + rng.gauss(0, 0.8), 1),
                "mem_available_mb": round(3400.0 + rng.gauss(0, 30.0), 1),
                "swap_percent": 50.0,
                "commit_percent": 50.0,
                "commit_available_mb": 4200.0,
                "disk_read_mb_s": round(max(0.0, rng.gauss(5.0, 1.5)), 2),
                "disk_write_mb_s": round(max(0.1, write_mb), 2),
                "heartbeat_lag_s": round(max(0.005, lag), 3),
                "top_proc_name": "sqlite3_sync.exe",
                "top_proc_cpu": round(cpu * 0.7, 1),
                "top_proc_mem_mb": 350.0,
                "ctx_switches_s": round(max(8000.0, 35000.0 + rng.gauss(0, 3000)), 1),
            }
        )
    return samples


def _generate_app_startup_plateau(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 240,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Hard Negative: Initial memory jump and CPU spike that plateaus and STABILIZES."""
    rng = random.Random(seed)
    samples = []
    base_mem = rng.uniform(42.0, 48.0)
    jump_mem = rng.uniform(18.0, 24.0)  # +20% jump

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        if i < 15:
            # Ramping up during startup
            prog = i / 15.0
            mem = base_mem + prog * jump_mem
            cpu = 20.0 + prog * 60.0 + rng.gauss(0, 4.0)
            lag = 0.05 + prog * 0.1
        else:
            # Stable plateau! Memory does NOT continue rising; slope -> 0.0
            mem = base_mem + jump_mem + rng.gauss(0, 0.3)
            cpu = max(5.0, 16.0 + rng.gauss(0, 3.0))
            lag = max(0.005, rng.gauss(0.015, 0.005))

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "app_startup",
                "cpu_percent": round(min(90.0, max(5.0, cpu)), 1),
                "mem_percent": round(mem, 1),
                "mem_available_mb": round((100.0 - mem) * 81.92, 1),
                "swap_percent": round(mem * 0.9, 1),
                "commit_percent": round(mem * 0.9, 1),
                "commit_available_mb": round((100.0 - mem) * 100.0, 1),
                "disk_read_mb_s": round(max(0.0, 12.0 - i * 0.5 if i < 20 else 0.2), 2),
                "disk_write_mb_s": round(max(0.0, 5.0 - i * 0.2 if i < 20 else 0.4), 2),
                "heartbeat_lag_s": round(lag, 3),
                "top_proc_name": "pos_terminal.exe",
                "top_proc_cpu": round(cpu * 0.8, 1),
                "top_proc_mem_mb": round(mem * 50.0, 1),
                "ctx_switches_s": round(max(5000.0, 16000.0 + rng.gauss(0, 1000)), 1),
            }
        )
    return samples


def _generate_antivirus_burst(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Hard Negative: Temporary background antivirus scan (40s burst then idle)."""
    rng = random.Random(seed)
    samples = []
    burst_start = 30
    burst_end = 80

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        if burst_start <= i <= burst_end:
            cpu = rng.uniform(65.0, 82.0)
            read_mb = rng.uniform(35.0, 60.0)
            lag = rng.uniform(0.08, 0.25)
            top_p = "MsMpEng.exe"
        else:
            cpu = max(4.0, rng.gauss(14.0, 3.0))
            read_mb = max(0.0, rng.gauss(0.5, 0.2))
            lag = max(0.005, rng.gauss(0.015, 0.005))
            top_p = "pos_terminal.exe"

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "antivirus_burst",
                "cpu_percent": round(cpu, 1),
                "mem_percent": round(51.0 + (5.0 if burst_start <= i <= burst_end else 0.0) + rng.gauss(0, 0.4), 1),
                "mem_available_mb": 3950.0,
                "swap_percent": 45.0,
                "commit_percent": 45.0,
                "commit_available_mb": 4800.0,
                "disk_read_mb_s": round(read_mb, 2),
                "disk_write_mb_s": 0.5,
                "heartbeat_lag_s": round(lag, 3),
                "top_proc_name": top_p,
                "top_proc_cpu": round(cpu * 0.8, 1),
                "top_proc_mem_mb": 380.0,
                "ctx_switches_s": round(max(8000.0, (40000.0 if burst_start <= i <= burst_end else 14000.0) + rng.gauss(0, 1500)), 1),
            }
        )
    return samples


def _generate_file_export(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 180,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Hard Negative: End-of-day report generation (30s CPU + disk burst)."""
    rng = random.Random(seed)
    samples = []
    burst_start = 25
    burst_end = 65

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        if burst_start <= i <= burst_end:
            cpu = rng.uniform(50.0, 70.0)
            write_mb = rng.uniform(20.0, 35.0)
            lag = rng.uniform(0.04, 0.18)
        else:
            cpu = max(5.0, rng.gauss(15.0, 3.0))
            write_mb = 0.5
            lag = max(0.005, rng.gauss(0.015, 0.005))

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "file_export",
                "cpu_percent": round(cpu, 1),
                "mem_percent": round(49.0 + (3.0 if burst_start <= i <= burst_end else 0.0) + rng.gauss(0, 0.3), 1),
                "mem_available_mb": 4100.0,
                "swap_percent": 42.0,
                "commit_percent": 42.0,
                "commit_available_mb": 5100.0,
                "disk_read_mb_s": 1.0,
                "disk_write_mb_s": round(write_mb, 2),
                "heartbeat_lag_s": round(lag, 3),
                "top_proc_name": "pos_report.exe",
                "top_proc_cpu": round(cpu * 0.75, 1),
                "top_proc_mem_mb": 290.0,
                "ctx_switches_s": round(max(6000.0, 18000.0 + rng.gauss(0, 1000)), 1),
            }
        )
    return samples


def _generate_routine_soak(
    ep_id: str,
    base_t: float = 1700000000.0,
    duration_s: int = 360,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Routine normal cashier scanning / soak period (clean benign operation)."""
    rng = random.Random(seed)
    samples = []
    base_mem = rng.uniform(46.0, 52.0)

    for i in range(duration_s):
        t = base_t + i
        t_mono = float(i)

        # Cashier scanning burst every 20-30 seconds
        is_scanning = (i % 35) < 12
        cpu = (25.0 if is_scanning else 12.0) + rng.gauss(0, 3.0)
        cpu = min(60.0, max(5.0, cpu))
        mem = base_mem + math.sin(i / 50.0) * 1.5 + rng.gauss(0, 0.2)

        samples.append(
            {
                "timestamp": round(t, 3),
                "timestamp_mono": round(t_mono, 3),
                "timestamp_iso": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
                "episode_id": ep_id,
                "scenario_type": "routine_idle",
                "cpu_percent": round(cpu, 1),
                "mem_percent": round(mem, 1),
                "mem_available_mb": round((100.0 - mem) * 81.92, 1),
                "swap_percent": round(mem * 0.85, 1),
                "commit_percent": round(mem * 0.85, 1),
                "commit_available_mb": round((100.0 - mem) * 95.0, 1),
                "disk_read_mb_s": round(max(0.0, rng.gauss(0.4, 0.2)), 2),
                "disk_write_mb_s": round(max(0.0, 1.2 if is_scanning else 0.2), 2),
                "heartbeat_lag_s": round(max(0.005, rng.gauss(0.015, 0.005)), 3),
                "top_proc_name": "pos_terminal.exe",
                "top_proc_cpu": round(cpu * 0.6, 1),
                "top_proc_mem_mb": 310.0,
                "ctx_switches_s": round(max(5000.0, (20000.0 if is_scanning else 12000.0) + rng.gauss(0, 1000)), 1),
            }
        )
    return samples


def generate_full_dataset(
    output_dir: str = "data/episodes",
    seed: int = 101,
) -> List[Tuple[List[Dict[str, Any]], EpisodeManifest]]:
    """Generate balanced corpus of 40 episodes across 8 realistic retail scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    all_episodes = []
    base_epoch = 1700000000.0
    ep_counter = 0

    scenario_specs = [
        # (generator_fn, count, scenario_name, duration_s)
        (_generate_slow_memory_leak, 8, "slow_memory_leak", 300),
        (_generate_fast_memory_leak, 6, "fast_memory_leak", 160),
        (_generate_cpu_starvation, 6, "cpu_starvation", 220),
        (_generate_disk_thrash, 6, "disk_thrash", 190),
        (_generate_app_startup_plateau, 4, "app_startup", 240),
        (_generate_antivirus_burst, 4, "antivirus_burst", 200),
        (_generate_file_export, 3, "file_export", 180),
        (_generate_routine_soak, 3, "routine_soak", 360),
    ]

    print(f"[Dataset Generator] Generating 40 realistic episodes to '{output_dir}'...")

    for gen_fn, count, name, dur in scenario_specs:
        for idx in range(count):
            ep_counter += 1
            ep_id = f"ep_{ep_counter:03d}_{name}"
            ep_seed = seed + ep_counter * 17
            ep_base_t = base_epoch + ep_counter * 2000

            raw_samples = gen_fn(ep_id, base_t=ep_base_t, duration_s=dur, seed=ep_seed)

            # Label with ground truth
            labeled_samples, gt = label_episode_samples(
                raw_samples,
                horizon_sec=DEFAULT_CONFIG.default_horizon_sec,
                config=DEFAULT_CONFIG,
                drop_post_failure=False,
            )

            # Save to disk
            ep_dir = os.path.join(output_dir, ep_id)
            os.makedirs(ep_dir, exist_ok=True)
            csv_path = os.path.join(ep_dir, "telemetry.csv")
            manifest_path = os.path.join(ep_dir, "manifest.json")

            keys = list(labeled_samples[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(labeled_samples)

            manifest = EpisodeManifest(
                episode_id=ep_id,
                scenario_type=name,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                duration_seconds=float(dur),
                total_samples=len(labeled_samples),
                has_failed=gt.has_failed,
                failure_timestamp=gt.failure_timestamp,
                failure_mode=gt.failure_mode,
                pre_failure_samples=gt.pre_failure_samples,
                post_failure_samples=gt.post_failure_samples,
                workload_manifest={"simulated": True, "seed": ep_seed},
                telemetry_csv_path=csv_path,
            )

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            all_episodes.append((labeled_samples, manifest))

    print(f"[Dataset Generator] Successfully generated {len(all_episodes)} episodes.")
    return all_episodes


if __name__ == "__main__":
    generate_full_dataset()
