"""Hard negative workload simulators for realistic retail counter telemetry.

Simulates benign, transient resource spikes that must NOT trigger alerts (ground truth Y = 0):
1. AppStartupWorkload: Rapid memory allocation and CPU burst that plateaus and stabilizes (e.g. POS app or browser launch).
2. FileExportWorkload: Temporary high disk throughput bursts (e.g. shift sales report export or log archive).
3. AntivirusScanWorkload: Multi-threaded background CPU and context switch burst.
4. CompositeRoutineWorkload: Realistic sequence of routine cashier background activity.
"""

import atexit
from dataclasses import asdict, dataclass
import gc
import hashlib
import os
import random
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HardNegativeManifest:
    """Metadata describing a benign hard-negative workload run."""

    workload_type: str
    seed: int
    parameters: Dict[str, Any]
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    peak_ram_mb: float
    is_failure_expected: bool = False  # By definition, hard negatives must not freeze

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AppStartupWorkload:
    """Simulates a large application launch (e.g. POS billing terminal or Chrome).

    Allocates 150-400 MB quickly, bursts CPU, then holds steady (plateaus)
    without continuously growing. This teaches the model that a plateau is benign.
    """

    def __init__(
        self,
        allocate_mb: Optional[float] = None,
        alloc_range_mb: Tuple[float, float] = (150.0, 350.0),
        startup_duration_sec: float = 5.0,
        total_duration_sec: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.rng = random.Random(self.seed)

        self.allocate_mb = (
            allocate_mb if allocate_mb is not None else self.rng.uniform(*alloc_range_mb)
        )
        self.startup_duration_sec = startup_duration_sec
        self.total_duration_sec = total_duration_sec
        self._allocated_chunks: List[bytearray] = []
        self._stop_event = threading.Event()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        self._allocated_chunks.clear()
        gc.collect()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> HardNegativeManifest:
        start_mono = time.monotonic()
        start_epoch = time.time()

        # Phase 1: Rapid ramp-up (App starting up)
        step_count = 10
        step_interval = self.startup_duration_sec / step_count
        chunk_mb = self.allocate_mb / step_count

        try:
            for _ in range(step_count):
                if self._stop_event.is_set():
                    break
                # Allocate and touch memory to commit physical pages
                chunk = bytearray(int(chunk_mb * 1024 * 1024))
                for offset in range(0, len(chunk), 4096):
                    chunk[offset] = 2
                self._allocated_chunks.append(chunk)

                # Light CPU churn during startup
                val = 0
                for i in range(25000):
                    val += (i * i) % 17

                time.sleep(step_interval)

            # Phase 2: Plateau (App is running, memory is stable, not leaking)
            remain_sec = max(0.0, self.total_duration_sec - (time.monotonic() - start_mono))
            end_mono = time.monotonic() + remain_sec
            while time.monotonic() < end_mono and not self._stop_event.is_set():
                time.sleep(0.5)

        finally:
            self._cleanup()

        duration = time.monotonic() - start_mono
        end_epoch = time.time()

        return HardNegativeManifest(
            workload_type="app_startup",
            seed=self.seed,
            parameters={
                "allocate_mb": round(self.allocate_mb, 1),
                "startup_duration_sec": self.startup_duration_sec,
                "total_duration_sec": self.total_duration_sec,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            peak_ram_mb=round(self.allocate_mb, 1),
            is_failure_expected=False,
        )


class FileExportWorkload:
    """Simulates cashier report generation or log archiving (temporary I/O burst)."""

    def __init__(
        self,
        file_size_mb: float = 50.0,
        burst_duration_sec: float = 6.0,
        total_duration_sec: float = 25.0,
        seed: Optional[int] = None,
    ):
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.file_size_mb = file_size_mb
        self.burst_duration_sec = burst_duration_sec
        self.total_duration_sec = total_duration_sec
        self._temp_dir = tempfile.mkdtemp(prefix="offguard_export_")
        self._stop_event = threading.Event()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        if os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> HardNegativeManifest:
        start_mono = time.monotonic()
        start_epoch = time.time()
        file_path = os.path.join(self._temp_dir, "daily_sales_report.dat")

        try:
            # Phase 1: Write temporary export file in chunks
            chunk_size = 1024 * 1024  # 1 MB
            payload = os.urandom(chunk_size)
            target_chunks = int(self.file_size_mb)

            burst_end = start_mono + self.burst_duration_sec
            with open(file_path, "wb") as f:
                for _ in range(target_chunks):
                    if self._stop_event.is_set():
                        break
                    f.write(payload)

            # Phase 2: Read it back (simulating verification/upload)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    while f.read(chunk_size):
                        if self._stop_event.is_set():
                            break

            # Phase 3: Wait remaining duration (I/O returns to baseline)
            remain_sec = max(0.0, self.total_duration_sec - (time.monotonic() - start_mono))
            end_mono = time.monotonic() + remain_sec
            while time.monotonic() < end_mono and not self._stop_event.is_set():
                time.sleep(0.5)

        finally:
            self._cleanup()

        duration = time.monotonic() - start_mono
        end_epoch = time.time()

        return HardNegativeManifest(
            workload_type="file_export",
            seed=self.seed,
            parameters={
                "file_size_mb": self.file_size_mb,
                "burst_duration_sec": self.burst_duration_sec,
                "total_duration_sec": self.total_duration_sec,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            peak_ram_mb=0.0,
            is_failure_expected=False,
        )


class AntivirusScanWorkload:
    """Simulates background antivirus or search indexer scan (CPU and context switch burst)."""

    def __init__(
        self,
        scan_duration_sec: float = 8.0,
        total_duration_sec: float = 25.0,
        worker_count: int = 2,
        seed: Optional[int] = None,
    ):
        self.seed = seed if seed is not None else random.randint(100000, 999999)
        self.scan_duration_sec = scan_duration_sec
        self.total_duration_sec = total_duration_sec
        self.worker_count = worker_count
        self._stop_event = threading.Event()

    def _hasher_worker(self, stop_time: float) -> None:
        data = b"retail_pos_security_scan_block_data" * 256
        while time.monotonic() < stop_time and not self._stop_event.is_set():
            h = hashlib.sha256()
            for _ in range(500):
                h.update(data)
            time.sleep(0.01)  # Yields CPU, creating context switches

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> HardNegativeManifest:
        start_mono = time.monotonic()
        start_epoch = time.time()

        # Phase 1: Burst of hashing threads
        stop_scan_time = start_mono + self.scan_duration_sec
        threads = []
        for _ in range(self.worker_count):
            t = threading.Thread(target=self._hasher_worker, args=(stop_scan_time,), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=self.scan_duration_sec + 2.0)

        # Phase 2: Rest of duration settles down to routine baseline
        remain_sec = max(0.0, self.total_duration_sec - (time.monotonic() - start_mono))
        end_mono = time.monotonic() + remain_sec
        while time.monotonic() < end_mono and not self._stop_event.is_set():
            time.sleep(0.5)

        duration = time.monotonic() - start_mono
        end_epoch = time.time()

        return HardNegativeManifest(
            workload_type="antivirus_burst",
            seed=self.seed,
            parameters={
                "scan_duration_sec": self.scan_duration_sec,
                "worker_count": self.worker_count,
                "total_duration_sec": self.total_duration_sec,
            },
            start_timestamp=round(start_epoch, 3),
            end_timestamp=round(end_epoch, 3),
            duration_seconds=round(duration, 2),
            peak_ram_mb=0.0,
            is_failure_expected=False,
        )
