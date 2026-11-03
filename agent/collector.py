"""System telemetry collector for offGuard.

Samples host OS metrics (CPU, RAM, Windows Commit Charge / Swap, Disk I/O,
Context Switches, Top Process, and Monotonic Loop Lag) at regular intervals
with graceful error handling and zero future leakage.
"""

import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import psutil

from agent.config import DEFAULT_CONFIG, AgentConfig


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _get_windows_memory_status() -> Optional[Dict[str, float]]:
    """Query Windows kernel32 GlobalMemoryStatusEx for physical and commit charge stats."""
    if sys.platform != "win32":
        return None
    try:
        ms = _MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            avail_phys_mb = ms.ullAvailPhys / (1024 * 1024)
            tot_page_mb = ms.ullTotalPageFile / (1024 * 1024)
            avail_page_mb = ms.ullAvailPageFile / (1024 * 1024)
            commit_used_mb = max(0.0, tot_page_mb - avail_page_mb)
            commit_pct = (commit_used_mb / tot_page_mb * 100.0) if tot_page_mb > 0 else 0.0
            return {
                "mem_percent": float(ms.dwMemoryLoad),
                "mem_available_mb": round(avail_phys_mb, 1),
                "commit_percent": round(commit_pct, 1),
                "commit_available_mb": round(avail_page_mb, 1),
            }
    except Exception:
        pass
    return None


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
    commit_percent: float = 0.0
    commit_available_mb: float = 0.0
    ctx_switches_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


class TelemetryCollector:
    """Collects system telemetry using psutil and Windows APIs with clock-drift immune lag."""

    def __init__(self, config: Optional[AgentConfig] = None, episode_id: str = "live"):
        self.config = config or DEFAULT_CONFIG
        self.episode_id = episode_id

        # Monotonic time tracking (immune to NTP and system wall-clock adjustments)
        self._last_tick_mono: Optional[float] = None
        self._target_next_tick_mono: Optional[float] = None

        # State tracking for delta / rate calculations
        self._last_disk_read_bytes: Optional[int] = None
        self._last_disk_write_bytes: Optional[int] = None
        self._last_ctx_switches: Optional[int] = None

        # Prime initial sensor counters
        self._prime_sensors()

    def _prime_sensors(self) -> None:
        """Prime CPU, disk, and context switch counters so subsequent calls yield accurate deltas."""
        try:
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

        try:
            stats = psutil.cpu_stats()
            self._last_ctx_switches = stats.ctx_switches
        except Exception:
            self._last_ctx_switches = 0

    def _get_top_process(self) -> Tuple[str, float, float]:
        """Identify the single process consuming the most memory/CPU.

        Privacy note: strictly logs process name, never CLI arguments.
        """
        top_name = "unknown"
        top_cpu = 0.0
        top_mem_mb = 0.0

        try:
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
                procs.sort(key=lambda item: item[0], reverse=True)
                top_mem_mb, top_cpu, top_name = procs[0]
        except Exception:
            pass

        return top_name, round(top_cpu, 1), round(top_mem_mb, 1)

    def collect_sample(self) -> TelemetrySample:
        """Sample all hardware metrics and compute scheduler lag and I/O rates."""
        now_epoch = time.time()
        now_mono = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Monotonic Heartbeat Lag Calculation
        if self._target_next_tick_mono is not None:
            heartbeat_lag_s = max(0.0, now_mono - self._target_next_tick_mono)
        else:
            heartbeat_lag_s = 0.0

        time_delta_s = (
            (now_mono - self._last_tick_mono)
            if self._last_tick_mono is not None
            else self.config.sample_interval_sec
        )
        time_delta_s = max(0.001, time_delta_s)

        # 2. CPU utilization
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_percent = 0.0

        # 3. Context Switches per Second
        ctx_switches_s = 0.0
        try:
            stats = psutil.cpu_stats()
            if self._last_ctx_switches is not None:
                ctx_delta = max(0, stats.ctx_switches - self._last_ctx_switches)
                ctx_switches_s = round(ctx_delta / time_delta_s, 1)
            self._last_ctx_switches = stats.ctx_switches
        except Exception:
            pass

        # 4. RAM and Commit / Pagefile Memory
        # On Windows, psutil.swap_memory() often fails with PDH errors. We use GlobalMemoryStatusEx.
        win_mem = _get_windows_memory_status()
        if win_mem is not None:
            mem_percent = win_mem["mem_percent"]
            mem_available_mb = win_mem["mem_available_mb"]
            commit_percent = win_mem["commit_percent"]
            commit_available_mb = win_mem["commit_available_mb"]
            swap_percent = commit_percent
        else:
            # Non-Windows fallback (Linux/macOS)
            try:
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                mem_available_mb = round(mem.available / (1024 * 1024), 1)
            except Exception:
                mem_percent = 0.0
                mem_available_mb = 0.0

            try:
                swap = psutil.swap_memory()
                swap_percent = swap.percent
                commit_percent = swap.percent
                commit_available_mb = round(getattr(swap, "free", 0) / (1024 * 1024), 1)
            except Exception:
                swap_percent = 0.0
                commit_percent = 0.0
                commit_available_mb = 0.0

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

        # Update monotonic timing state for next tick
        self._last_tick_mono = now_mono
        self._target_next_tick_mono = now_mono + self.config.sample_interval_sec

        return TelemetrySample(
            timestamp=round(now_epoch, 3),
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
            commit_percent=commit_percent,
            commit_available_mb=commit_available_mb,
            ctx_switches_s=ctx_switches_s,
        )


def stream_telemetry(duration_sec: float = 10.0, interval_sec: float = 1.0) -> None:
    """Demo stream: prints live telemetry snapshots to console using local time."""
    config = AgentConfig(sample_interval_sec=interval_sec)
    collector = TelemetryCollector(config=config, episode_id="demo_stream")

    print("\n" + "=" * 94)
    print(f" offGuard Telemetry Monitor [Sampling every {interval_sec}s for {duration_sec}s]")
    print("=" * 94)
    print(
        f"{'LOCAL TIME':<10} | {'CPU%':<5} | {'RAM%':<5} | {'FREE RAM':<9} | "
        f"{'COMMIT%':<7} | {'CTX/s':<7} | {'LAG(s)':<6} | {'TOP PROCESS':<20}"
    )
    print("-" * 94)

    start = time.time()
    try:
        while time.time() - start < duration_sec:
            sample = collector.collect_sample()
            local_time_str = datetime.fromtimestamp(sample.timestamp).strftime("%H:%M:%S")
            top_info = f"{sample.top_proc_name[:12]} ({sample.top_proc_mem_mb:.0f}M)"

            print(
                f"{local_time_str:<10} | {sample.cpu_percent:<5.1f} | {sample.mem_percent:<5.1f} | "
                f"{sample.mem_available_mb:<6.0f} MB | {sample.commit_percent:<7.1f} | "
                f"{sample.ctx_switches_s:<7.0f} | {sample.heartbeat_lag_s:<6.3f} | {top_info:<20}"
            )
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    print("=" * 94)
    print("Stream completed.\n")


if __name__ == "__main__":
    stream_telemetry(duration_sec=10.0, interval_sec=1.0)
