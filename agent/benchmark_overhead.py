"""Self-benchmarking utility to measure the telemetry collector's runtime overhead.

Measures:
1. Agent process CPU utilization (%) under idle and active stress conditions.
2. Agent process RSS memory footprint (MB) and drift over time.
3. Telemetry collection cycle duration (latency per sample in milliseconds).
4. Scheduler heartbeat lag observed under resource contention.
"""

from dataclasses import dataclass
import os
import statistics
import threading
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
    stress_mode: bool
    process_cpu_mean_pct: float
    process_cpu_peak_pct: float
    process_ram_initial_mb: float
    process_ram_final_mb: float
    process_ram_mean_mb: float
    process_ram_drift_mb: float
    sample_exec_time_mean_ms: float
    sample_exec_time_p95_ms: float
    heartbeat_lag_mean_s: float
    heartbeat_lag_max_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "duration_seconds": round(self.duration_seconds, 1),
            "sample_interval_sec": self.sample_interval_sec,
            "stress_mode": self.stress_mode,
            "process_cpu_mean_pct": round(self.process_cpu_mean_pct, 3),
            "process_cpu_peak_pct": round(self.process_cpu_peak_pct, 2),
            "process_ram_initial_mb": round(self.process_ram_initial_mb, 2),
            "process_ram_final_mb": round(self.process_ram_final_mb, 2),
            "process_ram_mean_mb": round(self.process_ram_mean_mb, 2),
            "process_ram_drift_mb": round(self.process_ram_drift_mb, 3),
            "sample_exec_time_mean_ms": round(self.sample_exec_time_mean_ms, 2),
            "sample_exec_time_p95_ms": round(self.sample_exec_time_p95_ms, 2),
            "heartbeat_lag_mean_s": round(self.heartbeat_lag_mean_s, 3),
            "heartbeat_lag_max_s": round(self.heartbeat_lag_max_s, 3),
        }


def _stress_worker(stop_event: threading.Event) -> None:
    """Simulates background resource churn (CPU + memory allocation loops)."""
    data_blocks = []
    while not stop_event.is_set():
        # Tight computation loop
        val = 0
        for i in range(100000):
            val += (i * i) % 17
        # Moderate memory allocation and deallocation
        if len(data_blocks) < 5:
            data_blocks.append(bytearray(10 * 1024 * 1024))  # 10MB chunk
        else:
            data_blocks.pop(0)


def run_overhead_benchmark(
    duration_sec: float = 30.0,
    interval_sec: float = 1.0,
    warmup_samples: int = 3,
    simulate_stress: bool = False,
) -> OverheadBenchmarkResult:
    """Run real-time self-profiling of the TelemetryCollector."""
    current_proc = psutil.Process(os.getpid())
    config = AgentConfig(sample_interval_sec=interval_sec)
    collector = TelemetryCollector(config=config, episode_id="overhead_benchmark")

    # Prime process CPU counter
    current_proc.cpu_percent(interval=None)

    # Optional background stress worker
    stop_event = threading.Event()
    stress_thread: Optional[threading.Thread] = None
    if simulate_stress:
        stress_thread = threading.Thread(target=_stress_worker, args=(stop_event,), daemon=True)
        stress_thread.start()

    # Warmup samples to ensure all modules/caches are loaded
    for _ in range(warmup_samples):
        collector.collect_sample()
        time.sleep(0.1)

    initial_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

    cpu_readings: List[float] = []
    ram_readings: List[float] = []
    exec_durations_ms: List[float] = []
    lag_readings: List[float] = []

    mode_label = "STRESS (Background Load)" if simulate_stress else "IDLE (Routine Counter)"
    print(f"\n[offGuard Overhead Benchmark] Mode: {mode_label}")
    print(f"Profiling for {duration_sec:.1f}s at {interval_sec}s intervals...")

    start_time = time.time()
    sample_count = 0

    try:
        while (time.time() - start_time) < duration_sec:
            tick_start = time.perf_counter()
            sample = collector.collect_sample()
            exec_ms = (time.perf_counter() - tick_start) * 1000.0
            exec_durations_ms.append(exec_ms)
            lag_readings.append(sample.heartbeat_lag_s)

            proc_cpu = current_proc.cpu_percent(interval=None)
            proc_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

            cpu_readings.append(proc_cpu)
            ram_readings.append(proc_ram_mb)
            sample_count += 1

            # Sleep remaining time until next scheduled tick
            elapsed_in_tick = (time.perf_counter() - tick_start)
            time.sleep(max(0.001, interval_sec - elapsed_in_tick))
    finally:
        if simulate_stress and stop_event is not None:
            stop_event.set()
            if stress_thread:
                stress_thread.join(timeout=1.0)

    total_duration = time.time() - start_time
    final_ram_mb = current_proc.memory_info().rss / (1024 * 1024)

    exec_durations_sorted = sorted(exec_durations_ms)
    p95_idx = int(0.95 * len(exec_durations_sorted))
    p95_exec_ms = exec_durations_sorted[p95_idx] if exec_durations_sorted else 0.0

    result = OverheadBenchmarkResult(
        total_samples=sample_count,
        duration_seconds=total_duration,
        sample_interval_sec=interval_sec,
        stress_mode=simulate_stress,
        process_cpu_mean_pct=statistics.mean(cpu_readings) if cpu_readings else 0.0,
        process_cpu_peak_pct=max(cpu_readings) if cpu_readings else 0.0,
        process_ram_initial_mb=initial_ram_mb,
        process_ram_final_mb=final_ram_mb,
        process_ram_mean_mb=statistics.mean(ram_readings) if ram_readings else 0.0,
        process_ram_drift_mb=final_ram_mb - initial_ram_mb,
        sample_exec_time_mean_ms=statistics.mean(exec_durations_ms) if exec_durations_ms else 0.0,
        sample_exec_time_p95_ms=p95_exec_ms,
        heartbeat_lag_mean_s=statistics.mean(lag_readings) if lag_readings else 0.0,
        heartbeat_lag_max_s=max(lag_readings) if lag_readings else 0.0,
    )

    print("\n--- Overhead Benchmark Results ---")
    print(f"Condition              : {mode_label}")
    print(f"Duration               : {result.duration_seconds:.1f} s ({result.total_samples} samples)")
    print(f"Mean Collector CPU     : {result.process_cpu_mean_pct:.2f}% (Peak: {result.process_cpu_peak_pct:.2f}%)")
    print(f"Average RSS Memory     : {result.process_ram_mean_mb:.2f} MB")
    print(f"Memory Drift (End-Start): {result.process_ram_drift_mb:+.3f} MB over {result.duration_seconds:.1f}s")
    print(f"Per-Sample Latency     : {result.sample_exec_time_mean_ms:.2f} ms (p95: {result.sample_exec_time_p95_ms:.2f} ms)")
    print(f"Observed Heartbeat Lag : mean={result.heartbeat_lag_mean_s:.3f}s, max={result.heartbeat_lag_max_s:.3f}s")
    print("----------------------------------\n")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile offGuard collector overhead")
    parser.add_argument("--duration", type=float, default=15.0, help="Benchmark duration in seconds")
    parser.add_argument("--stress", action="store_true", help="Simulate background resource stress")
    args = parser.parse_args()

    run_overhead_benchmark(duration_sec=args.duration, interval_sec=1.0, simulate_stress=args.stress)
