"""Heuristic and trend-based baselines for offGuard early warning.

Provides:
1. StaticThresholdBaseline: Rule-based alerting on static memory/swap/CPU limits.
2. TimeToExhaustionBaseline: Fits a linear trend to rolling available RAM to
   calculate seconds until the safety floor is reached. Interpretable, zero-training,
   and deterministic.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from agent.config import DEFAULT_CONFIG, AgentConfig


@dataclass
class BaselinePrediction:
    """Standardized output for baseline models."""

    is_alert: bool
    risk_score: float  # [0.0, 1.0]
    reason: str
    metadata: Dict[str, Any]


class StaticThresholdBaseline:
    """Alerts when hardware metrics exceed fixed heuristic thresholds for N consecutive ticks.

    Default rule: RAM >= 92% AND Swap >= 85% for >= 3 consecutive ticks, OR CPU >= 98% for >= 5 ticks.
    """

    def __init__(
        self,
        mem_percent_threshold: float = 90.0,
        swap_percent_threshold: float = 80.0,
        consecutive_ticks: int = 3,
    ):
        self.mem_threshold = mem_percent_threshold
        self.swap_threshold = swap_percent_threshold
        self.consecutive_ticks = consecutive_ticks
        self._streak_count = 0

    def reset(self) -> None:
        """Reset internal streak state."""
        self._streak_count = 0

    def predict_sample(self, sample: Union[Dict[str, Any], Any]) -> BaselinePrediction:
        """Evaluate a single telemetry sample against the static threshold rule."""
        if isinstance(sample, dict):
            mem = float(sample.get("mem_percent", 0.0))
            swap = float(sample.get("swap_percent", 0.0))
        else:
            mem = float(getattr(sample, "mem_percent", 0.0))
            swap = float(getattr(sample, "swap_percent", 0.0))

        breach = (mem >= self.mem_threshold) and (swap >= self.swap_threshold)

        if breach:
            self._streak_count += 1
        else:
            self._streak_count = 0

        is_alert = self._streak_count >= self.consecutive_ticks
        risk = 1.0 if is_alert else (0.5 if self._streak_count > 0 else 0.0)
        reason = (
            f"Static breach (RAM {mem:.1f}% >= {self.mem_threshold}%, Swap {swap:.1f}% >= {self.swap_threshold}%) "
            f"for {self._streak_count}/{self.consecutive_ticks} ticks"
        )

        return BaselinePrediction(
            is_alert=is_alert,
            risk_score=risk,
            reason=reason,
            metadata={"streak": self._streak_count, "mem": mem, "swap": swap},
        )


class TimeToExhaustionBaseline:
    """Predicts time until available RAM reaches safety floor using linear trend fitting.

    Fits slope m and intercept c to recent rolling window of (t, mem_available_mb).
    If slope < 0 (active drain):
        seconds_to_exhaustion = (current_ram - floor_ram) / abs(slope)
        If seconds_to_exhaustion <= horizon_sec: Alert!
    """

    def __init__(
        self,
        window_size_samples: int = 30,
        min_samples_to_fit: int = 5,
        target_ram_floor_mb: Optional[float] = None,
        min_drain_rate_mb_s: float = 0.5,
        config: Optional[AgentConfig] = None,
    ):
        cfg = config or DEFAULT_CONFIG
        self.window_size = window_size_samples
        self.min_samples = min_samples_to_fit
        self.ram_floor = (
            target_ram_floor_mb
            if target_ram_floor_mb is not None
            else cfg.min_safe_ram_mb
        )
        self.min_drain_rate = min_drain_rate_mb_s
        self.horizon_sec = float(cfg.default_horizon_sec)

        # Buffer of (timestamp, mem_available_mb)
        self._history: deque[tuple[float, float]] = deque(maxlen=window_size_samples)

    def reset(self) -> None:
        """Clear historical rolling buffer."""
        self._history.clear()

    def _fit_linear_trend(self) -> tuple[float, float]:
        """Compute ordinary least squares linear regression slope and current estimated value.

        Returns:
            (slope_mb_per_sec, current_fitted_ram_mb)
        """
        n = len(self._history)
        if n < self.min_samples:
            return 0.0, 0.0

        sum_t = 0.0
        sum_m = 0.0
        for t, m in self._history:
            sum_t += t
            sum_m += m

        mean_t = sum_t / n
        mean_m = sum_m / n

        numerator = 0.0
        denominator = 0.0

        for t, m in self._history:
            dt = t - mean_t
            dm = m - mean_m
            numerator += dt * dm
            denominator += dt * dt

        if denominator < 1e-6:
            return 0.0, mean_m

        slope = numerator / denominator
        # Current value using the latest timestamp in history
        last_t = self._history[-1][0]
        intercept = mean_m - slope * mean_t
        fitted_current = slope * last_t + intercept

        return slope, fitted_current

    def predict_sample(
        self,
        sample: Union[Dict[str, Any], Any],
        horizon_sec: Optional[float] = None,
    ) -> BaselinePrediction:
        """Ingest new sample, update rolling trend, and forecast time to exhaustion."""
        h_sec = float(horizon_sec if horizon_sec is not None else self.horizon_sec)

        if isinstance(sample, dict):
            ts = float(sample.get("timestamp", 0.0))
            mem_avail = float(sample.get("mem_available_mb", 0.0))
        else:
            ts = float(getattr(sample, "timestamp", 0.0))
            mem_avail = float(getattr(sample, "mem_available_mb", 0.0))

        self._history.append((ts, mem_avail))

        if len(self._history) < self.min_samples:
            return BaselinePrediction(
                is_alert=False,
                risk_score=0.0,
                reason=f"Warming up buffer ({len(self._history)}/{self.min_samples} samples)",
                metadata={"time_to_exhaustion_s": math.inf, "slope_mb_s": 0.0},
            )

        slope_mb_s, current_val = self._fit_linear_trend()

        # If slope is negative and exceeds noise floor: RAM is depleting
        if slope_mb_s <= -self.min_drain_rate:
            drain_rate = abs(slope_mb_s)
            remaining_headroom = max(0.0, mem_avail - self.ram_floor)

            if remaining_headroom <= 0.0:
                time_to_exhaustion_s = 0.0
            else:
                time_to_exhaustion_s = remaining_headroom / drain_rate

            is_alert = time_to_exhaustion_s <= h_sec
            # Calculate calibrated risk: 1.0 when at floor, scales down linearly to 0.0 at horizon
            risk = max(0.0, min(1.0, 1.0 - (time_to_exhaustion_s / h_sec)))

            reason = (
                f"RAM draining at {drain_rate:.2f} MB/s. "
                f"Est. {time_to_exhaustion_s:.1f}s until safe floor ({self.ram_floor} MB). "
                f"Threshold H={h_sec:.0f}s."
            )
        else:
            time_to_exhaustion_s = math.inf
            is_alert = False
            risk = 0.0
            reason = f"Stable or growing available RAM (slope: {slope_mb_s:+.2f} MB/s)"

        return BaselinePrediction(
            is_alert=is_alert,
            risk_score=round(risk, 3),
            reason=reason,
            metadata={
                "time_to_exhaustion_s": (
                    round(time_to_exhaustion_s, 1)
                    if not math.isinf(time_to_exhaustion_s)
                    else None
                ),
                "slope_mb_s": round(slope_mb_s, 3),
                "current_avail_mb": mem_avail,
                "ram_floor_mb": self.ram_floor,
            },
        )

