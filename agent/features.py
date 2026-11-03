"""Rolling feature engineering pipeline for offGuard.

Transforms raw telemetry samples into rolling window features with strict temporal integrity:
1. Timestamp-based windows: Groups samples strictly by elapsed time (t - W <= t_i <= t),
   handling scheduler jitter and missing ticks robustly.
2. Hardware normalized: Uses percentage-based and relative metrics (mem_percent, commit_percent,
   cpu_percent, ctx_switches_s) rather than raw machine-specific MB.
3. Zero future leakage: Features at timestamp t consume only samples with timestamp <= t.
4. Warmup tracking: Provides window_fill_ratio_60s and window_filled_60s flags.
5. Lag ablation: Exposes both full feature set and a 'no-lag' feature set to isolate early precursors.
6. Offline/Online Parity: Streaming engine yields identical outputs to batch processing.
"""

from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Union

from agent.collector import TelemetrySample
from agent.config import DEFAULT_CONFIG, AgentConfig

# Core base metrics to extract from each sample
BASE_METRICS = [
    "mem_percent",
    "commit_percent",
    "cpu_percent",
    "ctx_switches_s",
    "disk_write_mb_s",
    "heartbeat_lag_s",
]

# Metrics excluded in the no-lag ablation set
LAG_METRIC_NAMES = ["heartbeat_lag_s"]


def _extract_metric(sample: Union[Dict[str, Any], Any], metric_name: str) -> float:
    """Extract float metric from sample dict or dataclass object."""
    if isinstance(sample, dict):
        val = sample.get(metric_name, 0.0)
    else:
        val = getattr(sample, metric_name, 0.0)

    try:
        f = float(val)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return 0.0


def _extract_timestamp(sample: Union[Dict[str, Any], Any]) -> float:
    """Extract timestamp float."""
    if isinstance(sample, dict):
        return float(sample.get("timestamp", 0.0))
    return float(getattr(sample, "timestamp", 0.0))


def compute_ols_slope(timestamps: List[float], values: List[float]) -> float:
    """Compute Ordinary Least Squares (OLS) slope: dy/dt (units per second).

    Returns 0.0 if fewer than 2 distinct timestamp samples exist.
    """
    n = len(timestamps)
    if n < 2:
        return 0.0

    mean_t = sum(timestamps) / n
    mean_v = sum(values) / n

    numerator = 0.0
    denominator = 0.0

    for t, v in zip(timestamps, values):
        dt = t - mean_t
        dv = v - mean_v
        numerator += dt * dv
        denominator += dt * dt

    if denominator < 1e-6:
        return 0.0

    return numerator / denominator


class RollingFeaturePipeline:
    """Sliding time-window feature pipeline supporting streaming and batch processing."""

    def __init__(
        self,
        windows_sec: Optional[List[int]] = None,
        config: Optional[AgentConfig] = None,
    ):
        cfg = config or DEFAULT_CONFIG
        self.windows_sec = sorted(windows_sec or cfg.window_sizes_sec)
        self.max_window_sec = max(self.windows_sec) if self.windows_sec else 60.0

        # Buffer stores (timestamp, dict_of_base_metrics)
        self._buffer: deque[tuple[float, Dict[str, float]]] = deque()
        self._first_seen_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear historical rolling buffer for a new episode."""
        self._buffer.clear()
        self._first_seen_timestamp = None

    def push_sample(self, sample: Union[Dict[str, Any], Any]) -> Dict[str, float]:
        """Ingest a single sample and return derived rolling features at timestamp t."""
        ts = _extract_timestamp(sample)
        if self._first_seen_timestamp is None:
            self._first_seen_timestamp = ts

        # Extract normalized base metrics
        metrics = {m: _extract_metric(sample, m) for m in BASE_METRICS}
        # Backward compatibility fallback: if commit_percent is 0 and swap_percent > 0
        if metrics["commit_percent"] == 0.0:
            metrics["commit_percent"] = _extract_metric(sample, "swap_percent")

        self._buffer.append((ts, metrics))

        # Evict samples older than max_window_sec (t_i < ts - max_window_sec)
        cutoff = ts - self.max_window_sec
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # Compute rolling window statistics
        features: Dict[str, float] = {}

        # 1. Instantaneous normalized base metrics
        for m in BASE_METRICS:
            features[f"raw_{m}"] = round(metrics[m], 3)

        # 2. Window filling metadata
        elapsed_since_start = max(0.0, ts - self._first_seen_timestamp)
        fill_ratio = min(1.0, elapsed_since_start / self.max_window_sec) if self.max_window_sec > 0 else 1.0
        features["window_fill_ratio"] = round(fill_ratio, 3)
        features["window_filled"] = 1.0 if fill_ratio >= 0.95 else 0.0

        # 3. Rolling window aggregations (Mean, Slope, Std)
        for w in self.windows_sec:
            w_cutoff = ts - float(w)
            # Filter samples strictly within [ts - w, ts]
            window_samples = [(t_i, m_dict) for t_i, m_dict in self._buffer if t_i >= w_cutoff]

            w_timestamps = [item[0] for item in window_samples]
            n_samples = len(window_samples)

            for m in BASE_METRICS:
                w_values = [item[1][m] for item in window_samples]

                if n_samples >= 1:
                    mean_val = sum(w_values) / n_samples
                    std_val = statistics.pstdev(w_values) if n_samples > 1 else 0.0
                    slope_val = compute_ols_slope(w_timestamps, w_values)
                else:
                    mean_val = metrics[m]
                    std_val = 0.0
                    slope_val = 0.0

                features[f"m_{m}_mean_{w}s"] = round(mean_val, 3)
                features[f"m_{m}_std_{w}s"] = round(std_val, 3)
                features[f"m_{m}_slope_{w}s"] = round(slope_val, 4)

        return features

    def transform_episode(
        self,
        samples: Sequence[Union[Dict[str, Any], Any]],
    ) -> List[Dict[str, float]]:
        """Batch transform an entire episode sequence (ensuring 1:1 online/offline parity)."""
        self.reset()
        output = []
        for sample in samples:
            feat = self.push_sample(sample)
            output.append(feat)
        return output

    @classmethod
    def get_feature_names(
        cls,
        windows_sec: Optional[List[int]] = None,
        include_lag: bool = True,
    ) -> List[str]:
        """Return deterministic ordered list of all feature names."""
        windows = windows_sec or DEFAULT_CONFIG.window_sizes_sec
        names = []

        active_metrics = (
            BASE_METRICS
            if include_lag
            else [m for m in BASE_METRICS if m not in LAG_METRIC_NAMES]
        )

        # Raw features
        for m in active_metrics:
            names.append(f"raw_{m}")

        names.append("window_fill_ratio")
        names.append("window_filled")

        for w in sorted(windows):
            for m in active_metrics:
                names.append(f"m_{m}_mean_{w}s")
                names.append(f"m_{m}_std_{w}s")
                names.append(f"m_{m}_slope_{w}s")

        return names

