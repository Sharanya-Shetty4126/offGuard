"""Self-benchmarking utility to measure the telemetry collector's runtime overhead.

Measures:
1. Agent process CPU utilization (%)
2. Agent process RSS memory footprint (MB)
3. Memory drift / leak over time (Delta RSS MB)
4. Telemetry collection cycle duration (latency per sample in milliseconds)
"""

from dataclasses import dataclass
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import psutil

from agent.collector import TelemetryCollector
from agent.config import AgentConfig


@dataclass
class OverheadBenchmarkResult:
    """Summary of runtime overhead for the agent collector."""

    total_samples: int
    duration_seconds: float
    sample_interval_sec: float
    process_cpu_mean_pct: float
    process_cpu_peak_pct: float
    process_ram_initial_mb: float
    process_ram_final_mb: float
    process_ram_mean_mb: float
    process_ram_drift_mb: float
    sample_exec_time_mean_ms: float
    sample_exec_time_p95_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "duration_seconds": round(self.duration_seconds, 1),
            "sample_interval_sec": self.sample_interval_sec,
            "process_cpu_mean_pct": round(self.process_cpu_mean_pct, 3),
            "process_cpu_peak_pct": round(self.process_cpu_peak_pct, 2),
            "process_ram_initial_mb": round(self.process_ram_initial_mb, 2),
            "process_ram_final_mb": round(self.process_ram_final_mb, 2),
            "process_ram_mean_mb": round(self.process_ram_mean_mb, 2),
            "process_ram_drift_mb": round(self.process_ram_drift_mb, 3),
            "sample_exec_time_mean_ms": round(self.sample_exec_time_mean_ms, 2),
            "sample_exec_time_p95_ms": round(self.sample_exec_time_p95_ms, 2),
        }


def run_overhead_benchmark(
    duration_sec: float = 30.0,
    interval_sec: float = 1.0,
    warmup_samples: int = 3,
) -> OverheadBenchmarkResult:
    """Run real-time self-profiling of the TelemetryCollector."""
    current_proc = psutil.Process(os.getpid())
    config = AgentConfig(sample_interval_sec=interval_sec)
    collector = TelemetryCollector(config=config, episode_id="overhead_benchmark")

    # Prime process CPU counter
    current_proc.cpu_percent(interval=None)

    # Warmup samples to ensure all modules/caches are loaded
    for _ in range(warmup_samples):
        collector.collect_sample()
        time.sleep(0.1)

    initial_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

    cpu_readings: List[float] = []
    ram_readings: List[float] = []
    exec_durations_ms: List[float] = []

    start_time = time.time()
    sample_count = 0

    print(f"\n[offGuard Overhead Benchmark] Running for {duration_sec}s at {interval_sec}s intervals...")

    while (time.time() - start_time) < duration_sec:
        tick_start = time.perf_counter()
        collector.collect_sample()
        exec_ms = (time.perf_counter() - tick_start) * 1000.0
        exec_durations_ms.append(exec_ms)

        proc_cpu = current_proc.cpu_percent(interval=None)
        proc_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

        cpu_readings.append(proc_cpu)
        ram_readings.append(proc_ram_mb)
        sample_count += 1

        # Sleep remaining time until next scheduled tick
        elapsed_in_tick = (time.perf_counter() - tick_start)
        time.sleep(max(0.001, interval_sec - elapsed_in_tick))

    total_duration = time.time() - start_time
    final_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

    exec_durations_sorted = sorted(exec_durations_ms)
    p95_idx = int(0.95 * len(exec_durations_sorted))
    p95_exec_ms = exec_durations_sorted[p95_idx] if exec_durations_sorted else 0.0

    result = OverheadBenchmarkResult(
        total_samples=sample_count,
        duration_seconds=total_duration,
        sample_interval_sec=interval_sec,
        process_cpu_mean_pct=statistics.mean(cpu_readings) if cpu_readings else 0.0,
        process_cpu_peak_pct=max(cpu_readings) if cpu_readings else 0.0,
        process_ram_initial_mb=initial_ram_mb,
        process_ram_final_mb=final_ram_mb,
        process_ram_mean_mb=statistics.mean(ram_readings) if ram_readings else 0.0,
        process_ram_drift_mb=final_ram_mb - initial_ram_mb,
        sample_exec_time_mean_ms=statistics.mean(exec_durations_ms) if exec_durations_ms else 0.0,
        sample_exec_time_p95_ms=p95_exec_ms,
    )

    print("\n--- Overhead Benchmark Results ---")
    print(f"Total Samples Collected : {result.total_samples}")
    print(f"Mean CPU Consumption   : {result.process_cpu_mean_pct:.2f}% (Peak: {result.process_cpu_peak_pct:.2f}%)")
    print(f"Average RSS Memory Footprint: {result.process_ram_mean_mb:.2f} MB")
    print(f"Memory Drift (End - Start)  : {result.process_ram_drift_mb:+.3f} MB")
    print(f"Per-Sample Latency      : {result.sample_exec_time_mean_ms:.2f} ms (p95: {result.sample_exec_time_p95_ms:.2f} ms)")
    print("----------------------------------\n")

    return result


if __name__ == "__main__":
    import sys

    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    run_overhead_benchmark(duration_sec=dur, interval_sec=1.0)

