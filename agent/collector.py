"""System telemetry collector for offGuard.

Samples host OS metrics (CPU, RAM, Swap, Disk I/O, Top Process, and Loop Lag)
at regular intervals with graceful error handling and zero future leakage.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

import psutil

from agent.config import DEFAULT_CONFIG, AgentConfig


@dataclass
class TelemetrySample:
    """Represents a single snapshot of host operating system vitals."""

    timestamp: float
    timestamp_iso: str
    episode_id: str
    cpu_percent: float
    mem_percent: float
    mem_available_mb: float
    swap_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    heartbeat_lag_s: float
    top_proc_name: str
    top_proc_cpu: float
    top_proc_mem_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


class TelemetryCollector:
    """Collects system telemetry using psutil with rate calculation and lag tracking."""

    def __init__(self, config: Optional[AgentConfig] = None, episode_id: str = "live"):
        self.config = config or DEFAULT_CONFIG
        self.episode_id = episode_id

        # State tracking for delta / rate calculations
        self._last_tick_time: Optional[float] = None
        self._target_next_tick: Optional[float] = None
        self._last_disk_read_bytes: Optional[int] = None
        self._last_disk_write_bytes: Optional[int] = None

        # Prime initial sensor counters
        self._prime_sensors()

    def _prime_sensors(self) -> None:
        """Prime CPU and disk counters so subsequent calls yield accurate deltas."""
        try:
            # First call primes the internal psutil delta timer
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._last_disk_read_bytes = disk_io.read_bytes
                self._last_disk_write_bytes = disk_io.write_bytes
        except Exception:
            self._last_disk_read_bytes = 0
            self._last_disk_write_bytes = 0

    def _get_top_process(self) -> tuple[str, float, float]:
        """Identify the single process consuming the most memory/CPU.

        Handles AccessDenied and NoSuchProcess gracefully.
        """
        top_name = "unknown"
        top_cpu = 0.0
        top_mem_mb = 0.0

        try:
            # Query top processes by RSS memory (often root cause of leaks)
            procs = []
            for p in psutil.process_iter(["name", "cpu_percent", "memory_info"]):
                try:
                    info = p.info
                    mem_info = info.get("memory_info")
                    rss_mb = (mem_info.rss / (1024 * 1024)) if mem_info else 0.0
                    cpu = info.get("cpu_percent") or 0.0
                    name = info.get("name") or "unknown"
                    procs.append((rss_mb, cpu, name))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            if procs:
                # Sort by RSS memory descending
                procs.sort(key=lambda item: item[0], reverse=True)
                top_mem_mb, top_cpu, top_name = procs[0]
        except Exception:
            pass

        return top_name, round(top_cpu, 1), round(top_mem_mb, 1)

    def collect_sample(self) -> TelemetrySample:
        """Sample all hardware metrics and compute scheduler lag and I/O rates."""
        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Compute heartbeat lag (how late this collection is compared to scheduled time)
        if self._target_next_tick is not None:
            heartbeat_lag_s = max(0.0, now - self._target_next_tick)
        else:
            heartbeat_lag_s = 0.0

        time_delta_s = (now - self._last_tick_time) if self._last_tick_time else self.config.sample_interval_sec
        time_delta_s = max(0.001, time_delta_s)  # Prevent division by zero

        # 2. CPU utilization
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_percent = 0.0

        # 3. RAM / Virtual Memory
        try:
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_available_mb = round(mem.available / (1024 * 1024), 1)
        except Exception:
            mem_percent = 0.0
            mem_available_mb = 0.0

        # 4. Swap / Page File
        try:
            swap = psutil.swap_memory()
            swap_percent = swap.percent
        except Exception:
            swap_percent = 0.0

        # 5. Disk I/O throughput rate (MB/s)
        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_read_bytes is not None and self._last_disk_write_bytes is not None:
                read_delta = max(0, disk_io.read_bytes - self._last_disk_read_bytes)
                write_delta = max(0, disk_io.write_bytes - self._last_disk_write_bytes)
                disk_read_mb_s = round(read_delta / (1024 * 1024 * time_delta_s), 2)
                disk_write_mb_s = round(write_delta / (1024 * 1024 * time_delta_s), 2)
                self._last_disk_read_bytes = disk_io.read_bytes
                self._last_disk_write_bytes = disk_io.write_bytes
        except Exception:
            pass

        # 6. Top resource consumer
        top_name, top_cpu, top_mem_mb = self._get_top_process()

        # Update timing state for next tick
        self._last_tick_time = now
        self._target_next_tick = now + self.config.sample_interval_sec

        return TelemetrySample(
            timestamp=round(now, 3),
            timestamp_iso=now_iso,
            episode_id=self.episode_id,
            cpu_percent=round(cpu_percent, 1),
            mem_percent=round(mem_percent, 1),
            mem_available_mb=mem_available_mb,
            swap_percent=round(swap_percent, 1),
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            heartbeat_lag_s=round(heartbeat_lag_s, 3),
            top_proc_name=top_name,
            top_proc_cpu=top_cpu,
            top_proc_mem_mb=top_mem_mb,
        )


def stream_telemetry(duration_sec: float = 10.0, interval_sec: float = 1.0) -> None:
    """Demo stream: prints live telemetry snapshots to console using local time."""
    config = AgentConfig(sample_interval_sec=interval_sec)
    collector = TelemetryCollector(config=config, episode_id="demo_stream")

    print("\n" + "=" * 82)
    print(f" offGuard Telemetry Monitor [Sampling every {interval_sec}s for {duration_sec}s]")
    print("=" * 82)
    print(
        f"{'LOCAL TIME':<10} | {'CPU%':<5} | {'RAM%':<5} | {'FREE RAM':<9} | "
        f"{'SWAP%':<5} | {'LAG(s)':<6} | {'TOP PROCESS':<22}"
    )
    print("-" * 82)

    start = time.time()
    try:
        while time.time() - start < duration_sec:
            sample = collector.collect_sample()
            local_time_str = datetime.fromtimestamp(sample.timestamp).strftime("%H:%M:%S")
            top_info = f"{sample.top_proc_name[:14]} ({sample.top_proc_mem_mb:.0f}M)"

            print(
                f"{local_time_str:<10} | {sample.cpu_percent:<5.1f} | {sample.mem_percent:<5.1f} | "
                f"{sample.mem_available_mb:<6.0f} MB | {sample.swap_percent:<5.1f} | "
                f"{sample.heartbeat_lag_s:<6.3f} | {top_info:<22}"
            )
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    print("=" * 82)
    print("Stream completed.\n")


if __name__ == "__main__":
    stream_telemetry(duration_sec=10.0, interval_sec=1.0)
