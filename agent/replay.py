"""Telemetry replay engine for offGuard.

Simulates live on-device telemetry by reading recorded episode CSV files and
emitting TelemetrySample instances in real-time or accelerated time.

Enables developer velocity: test Phase 4 (State Machine & Cashier UI) and
Phase 5 (FastAPI Reception Receiver) deterministically without stressing host hardware.
"""

import csv
from datetime import datetime
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union

from agent.collector import TelemetrySample


class ReplayTelemetryCollector:
    """Replays recorded telemetry samples with configurable pacing."""

    def __init__(
        self,
        source: Union[str, List[Dict[str, Any]]],
        speedup_factor: float = 1.0,
        realtime: bool = True,
        loop: bool = False,
    ):
        """Initialize replay collector.

        Args:
            source: Path to telemetry.csv, episode directory containing telemetry.csv,
                    or a pre-loaded list of sample dicts.
            speedup_factor: Multiplier for playback speed (e.g. 2.0 = 2x speed, 0 = instant).
            realtime: If True, sleeps according to elapsed timestamp deltas / speedup_factor.
            loop: If True, restarts from the beginning when the end of the episode is reached.
        """
        self.speedup_factor = max(0.001, speedup_factor) if speedup_factor > 0 else 0.0
        self.realtime = realtime
        self.loop = loop
        self._samples: List[Dict[str, Any]] = []
        self._current_index = 0
        self._last_sample_time: Optional[float] = None

        self._load_samples(source)

    def _load_samples(self, source: Union[str, List[Dict[str, Any]]]) -> None:
        """Load samples from CSV file, folder, or list."""
        if isinstance(source, list):
            self._samples = [dict(s) for s in source]
            return

        csv_path = source
        if os.path.isdir(source):
            candidate = os.path.join(source, "telemetry.csv")
            if os.path.exists(candidate):
                csv_path = candidate
            else:
                raise FileNotFoundError(f"No telemetry.csv found in directory '{source}'")

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Telemetry file not found: '{csv_path}'")

        loaded = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                d: Dict[str, Any] = {}
                for k, v in row.items():
                    try:
                        if "." in v:
                            d[k] = float(v)
                        else:
                            d[k] = int(v)
                    except (ValueError, TypeError):
                        # Boolean or string fields
                        if v.lower() == "true":
                            d[k] = True
                        elif v.lower() == "false":
                            d[k] = False
                        else:
                            d[k] = v
                loaded.append(d)

        self._samples = loaded

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def has_next(self) -> bool:
        if self.loop:
            return len(self._samples) > 0
        return self._current_index < len(self._samples)

    def reset(self) -> None:
        """Reset replay playback to the first sample."""
        self._current_index = 0
        self._last_sample_time = None

    def collect_sample(self) -> Optional[TelemetrySample]:
        """Fetch the next sample in the replay sequence with timing pacing.

        Returns:
            TelemetrySample instance, or None if end of episode reached (and loop=False).
        """
        if not self._samples:
            return None

        if self._current_index >= len(self._samples):
            if self.loop:
                self.reset()
            else:
                return None

        raw = self._samples[self._current_index]
        ts = float(raw.get("timestamp", 0.0))

        # Pacing: sleep proportional to delta between recorded timestamps
        if self.realtime and self._last_sample_time is not None and self.speedup_factor > 0:
            delta_s = max(0.0, ts - self._last_sample_time)
            sleep_time = delta_s / self.speedup_factor
            if 0.0 < sleep_time < 30.0:  # Guard against massive gaps in fast forward
                time.sleep(sleep_time)

        self._last_sample_time = ts
        self._current_index += 1

        return TelemetrySample(
            timestamp=ts,
            timestamp_iso=str(raw.get("timestamp_iso", "")),
            episode_id=str(raw.get("episode_id", "replay")),
            cpu_percent=float(raw.get("cpu_percent", 0.0)),
            mem_percent=float(raw.get("mem_percent", 0.0)),
            mem_available_mb=float(raw.get("mem_available_mb", 0.0)),
            swap_percent=float(raw.get("swap_percent", 0.0)),
            disk_read_mb_s=float(raw.get("disk_read_mb_s", 0.0)),
            disk_write_mb_s=float(raw.get("disk_write_mb_s", 0.0)),
            heartbeat_lag_s=float(raw.get("heartbeat_lag_s", 0.0)),
            top_proc_name=str(raw.get("top_proc_name", "unknown")),
            top_proc_cpu=float(raw.get("top_proc_cpu", 0.0)),
            top_proc_mem_mb=float(raw.get("top_proc_mem_mb", 0.0)),
            commit_percent=float(raw.get("commit_percent", raw.get("swap_percent", 0.0))),
            commit_available_mb=float(raw.get("commit_available_mb", 0.0)),
            ctx_switches_s=float(raw.get("ctx_switches_s", 0.0)),
        )

    def __iter__(self) -> Iterator[TelemetrySample]:
        self.reset()
        return self

    def __next__(self) -> TelemetrySample:
        sample = self.collect_sample()
        if sample is None:
            raise StopIteration
        return sample


def replay_stream(
    source: str,
    speedup_factor: float = 1.0,
) -> None:
    """Stream replay samples to console table."""
    collector = ReplayTelemetryCollector(source=source, speedup_factor=speedup_factor, realtime=True)

    print("\n" + "=" * 94)
    print(f" offGuard Replay Stream [{collector.total_samples} samples @ {speedup_factor}x speed]")
    print("=" * 94)
    print(
        f"{'TIME':<10} | {'CPU%':<5} | {'RAM%':<5} | {'FREE RAM':<9} | "
        f"{'COMMIT%':<7} | {'CTX/s':<7} | {'LAG(s)':<6} | {'TOP PROCESS':<20}"
    )
    print("-" * 94)

    try:
        while collector.has_next:
            sample = collector.collect_sample()
            if sample is None:
                break
            time_str = sample.timestamp_iso.split("T")[-1][:8] if "T" in sample.timestamp_iso else str(sample.timestamp)
            top_info = f"{sample.top_proc_name[:12]} ({sample.top_proc_mem_mb:.0f}M)"

            print(
                f"{time_str:<10} | {sample.cpu_percent:<5.1f} | {sample.mem_percent:<5.1f} | "
                f"{sample.mem_available_mb:<6.0f} MB | {sample.commit_percent:<7.1f} | "
                f"{sample.ctx_switches_s:<7.0f} | {sample.heartbeat_lag_s:<6.3f} | {top_info:<20}"
            )
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")

    print("=" * 94)
    print("Replay completed.\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agent.replay <path_to_telemetry.csv> [speedup_factor]")
        sys.exit(1)

    path = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    replay_stream(path, speedup_factor=speed)
