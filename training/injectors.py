"""Controlled hardware failure injectors with safety circuit breakers.

Implements realistic, randomized degradation scenarios for dataset collection:
1. MemoryLeakInjector: Steady or accelerating memory leak with randomized rate and start delay.
2. CpuStarvationInjector: Multi-core worker threads starving the OS scheduler.
3. DiskThrashInjector: Heavy synchronized disk I/O bursts.
4. CombinedStressInjector: Mixed degradation (RAM leak + CPU burn + I/O).

Safety Circuit Breakers:
- Hard RAM floor (default 500 MB): Immediately terminates and releases memory blocks.
- Hard duration timeout (default 90 s).
- Registered cleanup handlers ensuring host machine stability.
"""

import atexit
from dataclasses import asdict, dataclass, field
import gc
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from agent.config import DEFAULT_CONFIG, AgentConfig


def _get_available_ram_mb() -> float:
    """Query currently available physical RAM in megabytes."""
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return 99999.0


@dataclass
class InjectionManifest:
    """Reproducibility metadata and execution summary for an injection run."""

    injector_type: str
    seed: int
    parameters: Dict[str, Any]
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    exit_reason: str  # "completed", "ram_floor_breached", "timeout", "interrupted"
    allocated_peak_mb: float
    safety_aborted: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryLeakInjector:
    """Simulates an application leaking RAM gradually until paging/swap occurs.

    Touches allocated pages so the OS physically commits memory.
    Terminates immediately if available physical RAM dips below min_safe_ram_mb.
    """

    def __init__(
        self,
        leak_rate_mb_s: Optional[float] = None,
        rate_range_mb_s: Tuple[float, float] = (10.0, 40.0),
        start_delay_sec: Optional[float] = None,
        delay_range_sec: Tuple[float, float] = (5.0, 20.0),
        target_duration_sec: float = 60.0,
        seed: Optional[int] = None,
        config: Optional[AgentConfig] = None,
    ):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.rng = random.Random(self.seed)

        self.leak_rate_mb_s = (
            leak_rate_mb_s
            if leak_rate_mb_s is not None
            else self.rng.uniform(*rate_range_mb_s)
        )
        self.start_delay_sec = (
            start_delay_sec
            if start_delay_sec is not None
            else self.rng.uniform(*delay_range_sec)
        )
        self.target_duration_sec = min(target_duration_sec, cfg.max_stress_duration_sec)
        self.ram_floor_mb = cfg.min_safe_ram_mb

        self._allocated_blocks: List[bytearray] = []
        self._stop_event = threading.Event()
        atexit.register(self._emergency_cleanup)

    def _emergency_cleanup(self) -> None:
        """Release all allocated memory blocks."""
        self._allocated_blocks.clear()
        gc.collect()

    def stop(self) -> None:
        """Signal injector to stop."""
        self._stop_event.set()

    def run(self, status_callback: Optional[callable] = None) -> InjectionManifest:
        """Execute memory leak loop with safety checks."""
        start_mono = time.monotonic()
        start_epoch = time.time()
        allocated_bytes = 0
        exit_reason = "completed"
        safety_aborted = False

        print(
            f"[MemoryLeakInjector] Started (seed={self.seed}). "
            f"Delay: {self.start_delay_sec:.1f}s | Rate: {self.leak_rate_mb_s:.1f} MB/s | "
            f"Floor: {self.ram_floor_mb} MB | Target: {self.target_duration_sec:.1f}s"
        )

        try:
            # 1. Start Delay (Routine baseline period before leak commences)
            delay_end = start_mono + self.start_delay_sec
            while time.monotonic() < delay_end and not self._stop_event.is_set():
                time.sleep(0.2)

            # 2. Active Leak Phase
            step_interval = 0.5  # Allocate twice per second
            chunk_size_bytes = int((self.leak_rate_mb_s * step_interval) * 1024 * 1024)

            while not self._stop_event.is_set():
                elapsed = time.monotonic() - start_mono
                if elapsed >= self.target_duration_sec:
                    exit_reason = "target_duration_reached"
                    break

                # Circuit Breaker 1: Check RAM floor
                avail_ram = _get_available_ram_mb()
                if avail_ram <= self.ram_floor_mb:
                    print(
                        f"\n[CIRCUIT BREAKER] RAM floor reached ({avail_ram:.1f} MB <= {self.ram_floor_mb} MB). "
                        f"Aborting injection to protect host."
                    )
                    exit_reason = "ram_floor_breached"
                    safety_aborted = True
                    break

                # Allocate chunk and touch it to force physical page commit
                try:
                    chunk = bytearray(chunk_size_bytes)
                    # Touch every 4KB page
                    for offset in range(0, len(chunk), 4096):
                        chunk[offset] = 1
                    chunk[-1] = 1
                    self._allocated_blocks.append(chunk)
                    allocated_bytes += len(chunk)
                except MemoryError:
                    print("[CIRCUIT BREAKER] MemoryError caught. Halting allocation.")
                    exit_reason = "memory_error"
                    break

                if status_callback:
                    status_callback(elapsed, allocated_bytes / (1024 * 1024), avail_ram)

                time.sleep(step_interval)

        except KeyboardInterrupt:
            exit_reason = "interrupted"
        finally:
            self._emergency_cleanup()

        end_epoch = time.time()
        duration = time.monotonic() - start_mono
        peak_mb = allocated_bytes / (1024 * 1024)

        print(
            f"[MemoryLeakInjector] Finished. Reason: {exit_reason} | "
            f"Duration: {duration:.1f}s | Peak Allocated: {peak_mb:.1f} MB | Freed: OK\n"
        )

        return InjectionManifest(
            injector_type="memory_leak",
            seed=self.seed,
            parameters={
                "leak_rate_mb_s": round(self.leak_rate_mb_s, 2),
                "start_delay_sec": round(self.start_delay_sec, 2),
                "target_duration_sec": self.target_duration_sec,
                "ram_floor_mb": self.ram_floor_mb,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            exit_reason=exit_reason,
            allocated_peak_mb=round(peak_mb, 2),
            safety_aborted=safety_aborted,
        )


class CpuStarvationInjector:
    """Spawns multi-threaded arithmetic workloads to starve the OS thread scheduler."""

    def __init__(
        self,
        worker_count: Optional[int] = None,
        start_delay_sec: Optional[float] = None,
        delay_range_sec: Tuple[float, float] = (5.0, 15.0),
        target_duration_sec: float = 45.0,
        seed: Optional[int] = None,
        config: Optional[AgentConfig] = None,
    ):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.rng = random.Random(self.seed)

        cpu_max = psutil.cpu_count() or 4
        self.worker_count = (
            worker_count
            if worker_count is not None
            else self.rng.randint(max(1, cpu_max - 1), cpu_max)
        )
        self.start_delay_sec = (
            start_delay_sec
            if start_delay_sec is not None
            else self.rng.uniform(*delay_range_sec)
        )
        self.target_duration_sec = min(target_duration_sec, cfg.max_stress_duration_sec)
        self._stop_event = threading.Event()

    def _burn_worker(self) -> None:
        """Tight computation loop."""
        val = 0
        while not self._stop_event.is_set():
            for i in range(100000):
                val += (i * i) % 19

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> InjectionManifest:
        """Run CPU burner with timeout."""
        start_mono = time.monotonic()
        start_epoch = time.time()
        exit_reason = "completed"

        print(
            f"[CpuStarvationInjector] Started (seed={self.seed}). "
            f"Workers: {self.worker_count} | Delay: {self.start_delay_sec:.1f}s | "
            f"Target: {self.target_duration_sec:.1f}s"
        )

        # 1. Start Delay
        delay_end = start_mono + self.start_delay_sec
        while time.monotonic() < delay_end and not self._stop_event.is_set():
            time.sleep(0.2)

        # 2. Launch workers
        threads = []
        for _ in range(self.worker_count):
            t = threading.Thread(target=self._burn_worker, daemon=True)
            threads.append(t)
            t.start()

        # 3. Monitor duration
        try:
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - start_mono
                if elapsed >= self.target_duration_sec:
                    exit_reason = "target_duration_reached"
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            exit_reason = "interrupted"
        finally:
            self.stop()
            for t in threads:
                t.join(timeout=1.0)

        duration = time.monotonic() - start_mono
        end_epoch = time.time()

        print(f"[CpuStarvationInjector] Finished. Reason: {exit_reason} | Duration: {duration:.1f}s\n")

        return InjectionManifest(
            injector_type="cpu_starvation",
            seed=self.seed,
            parameters={
                "worker_count": self.worker_count,
                "start_delay_sec": round(self.start_delay_sec, 2),
                "target_duration_sec": self.target_duration_sec,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            exit_reason=exit_reason,
            allocated_peak_mb=0.0,
            safety_aborted=False,
        )


class DiskThrashInjector:
    """Generates continuous synchronized disk writes to saturate OS disk queues."""

    def __init__(
        self,
        block_size_mb: float = 8.0,
        start_delay_sec: Optional[float] = None,
        target_duration_sec: float = 40.0,
        seed: Optional[int] = None,
        config: Optional[AgentConfig] = None,
    ):
        cfg = config or DEFAULT_CONFIG
        self.config = cfg
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.rng = random.Random(self.seed)

        self.block_size_mb = block_size_mb
        self.start_delay_sec = (
            start_delay_sec if start_delay_sec is not None else self.rng.uniform(5.0, 15.0)
        )
        self.target_duration_sec = min(target_duration_sec, cfg.max_stress_duration_sec)
        self._stop_event = threading.Event()
        self._temp_dir = tempfile.mkdtemp(prefix="offguard_thrash_")
        atexit.register(self._cleanup_temp)

    def _cleanup_temp(self) -> None:
        """Remove temp files."""
        if os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> InjectionManifest:
        """Run synchronized disk write bursts."""
        start_mono = time.monotonic()
        start_epoch = time.time()
        exit_reason = "completed"
        bytes_written = 0

        # Delay
        delay_end = start_mono + self.start_delay_sec
        while time.monotonic() < delay_end and not self._stop_event.is_set():
            time.sleep(0.2)

        payload = os.urandom(int(self.block_size_mb * 1024 * 1024))
        file_path = os.path.join(self._temp_dir, "thrash_test.tmp")

        try:
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - start_mono
                if elapsed >= self.target_duration_sec:
                    exit_reason = "target_duration_reached"
                    break

                with open(file_path, "wb") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())  # Force commit to physical storage
                bytes_written += len(payload)
                time.sleep(0.1)

        except KeyboardInterrupt:
            exit_reason = "interrupted"
        finally:
            self._cleanup_temp()

        duration = time.monotonic() - start_mono
        end_epoch = time.time()

        return InjectionManifest(
            injector_type="disk_thrash",
            seed=self.seed,
            parameters={
                "block_size_mb": self.block_size_mb,
                "start_delay_sec": round(self.start_delay_sec, 2),
                "target_duration_sec": self.target_duration_sec,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            exit_reason=exit_reason,
            allocated_peak_mb=round(bytes_written / (1024 * 1024), 1),
            safety_aborted=False,
        )

