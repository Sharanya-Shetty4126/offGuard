"""On-device early warning state machine and explainable alert detector.

Manages terminal health states:
- OK (Green): Healthy baseline operation.
- WATCH (Yellow): Elevated precursor stress (risk >= 0.50).
- AT-RISK (Red): High freeze risk within 60-120s (risk >= 0.80 for >= 3 consecutive ticks).

Features:
1. Hysteresis: Requires N consecutive ticks to escalate to AT-RISK to prevent alert flickering.
2. Recovery Cooldown: Requires M consecutive healthy ticks to clear back to OK.
3. Explainable Alert Strings: Explains root cause (e.g. RAM drain rate, top process, context switches).
4. Network Cooldown: Throttles outbound reception alerts to prevent network flooding.
5. Flexible Engine: Operates seamlessly with trained ModelArtifact or baseline heuristic fallback.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from agent.baselines import StaticThresholdBaseline, TimeToExhaustionBaseline
from agent.collector import TelemetrySample
from agent.config import DEFAULT_CONFIG, AgentConfig
from agent.features import RollingFeaturePipeline


class TerminalState(str, Enum):
    OK = "OK"
    WATCH = "WATCH"
    AT_RISK = "AT_RISK"


@dataclass
class AlertPayload:
    """Outbound alert structure sent to store reception and displayed on cashier UI."""

    terminal_id: str
    state: TerminalState
    risk_score: float
    reason: str
    customer_message: str
    cashier_guidance: str
    top_proc_name: str
    top_proc_mem_mb: float
    timestamp: float
    timestamp_iso: str
    state_changed: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class EarlyWarningDetector:
    """Stateful inference engine managing hysteresis, thresholding, and alert generation."""

    def __init__(
        self,
        model_artifact: Optional[Any] = None,
        terminal_id: str = "pos_terminal_01",
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.terminal_id = terminal_id
        self.model_artifact = model_artifact

        # Feature pipeline & baselines fallback
        self.feature_pipeline = RollingFeaturePipeline()
        self.fallback_tte = TimeToExhaustionBaseline(config=self.config)
        self.fallback_static = StaticThresholdBaseline()

        # State machine tracking
        self.current_state = TerminalState.OK
        self._high_risk_streak = 0
        self._recovery_streak = 0
        self._last_alert_time: float = 0.0

    def reset(self) -> None:
        """Reset internal detector state for a new session."""
        self.current_state = TerminalState.OK
        self._high_risk_streak = 0
        self._recovery_streak = 0
        self._last_alert_time = 0.0
        self.feature_pipeline.reset()
        self.fallback_tte.reset()
        self.fallback_static.reset()

    def _generate_explanation(
        self,
        sample: Dict[str, Any],
        features: Dict[str, float],
        risk_score: float,
    ) -> Tuple[str, str, str]:
        """Generate explainable reason string, customer message, and cashier guidance."""
        mem_pct = float(sample.get("mem_percent", 0.0))
        commit_pct = float(sample.get("commit_percent", sample.get("swap_percent", 0.0)))
        cpu_pct = float(sample.get("cpu_percent", 0.0))
        lag = float(sample.get("heartbeat_lag_s", 0.0))
        top_name = str(sample.get("top_proc_name", "unknown"))
        top_mem = float(sample.get("top_proc_mem_mb", 0.0))

        # Check slope over 30s window
        mem_slope = features.get("m_mem_percent_slope_30s", 0.0)
        commit_slope = features.get("m_commit_percent_slope_30s", 0.0)

        # Build root-cause diagnosis
        causes = []
        if commit_slope > 0.15 or mem_slope > 0.15:
            causes.append(f"Memory rising (+{max(mem_slope, commit_slope):.2f}%/s)")
        if commit_pct >= 85.0 or mem_pct >= 90.0:
            causes.append(f"High memory saturation (RAM: {mem_pct:.0f}%, Commit: {commit_pct:.0f}%)")
        if cpu_pct >= 90.0:
            causes.append(f"CPU starved ({cpu_pct:.0f}%)")
        if lag >= 0.5:
            causes.append(f"Scheduler lag ({lag:.2f}s)")

        reason_str = "; ".join(causes) if causes else f"Elevated resource degradation (risk: {risk_score:.2f})"
        reason_str += f". Top process: {top_name} ({top_mem:.0f} MB)"

        # Calm customer-facing copy
        if self.current_state == TerminalState.AT_RISK:
            customer_msg = "Routine counter service in progress. Thank you for your patience."
            cashier_guidance = (
                "ACTION: Finish active transaction. Do NOT scan next customer. "
                "Direct queue to adjacent counter. Reception notified."
            )
        elif self.current_state == TerminalState.WATCH:
            customer_msg = "Counter operating normally."
            cashier_guidance = "NOTICE: System workload elevated. Keep receipt ready; monitor terminal."
        else:
            customer_msg = "Welcome! Counter open."
            cashier_guidance = "System optimal. Ready for transactions."

        return reason_str, customer_msg, cashier_guidance

    def process_sample(
        self,
        sample: Union[TelemetrySample, Dict[str, Any]],
    ) -> AlertPayload:
        """Ingest single telemetry snapshot, evaluate risk, and transition state."""
        sample_dict = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
        ts = float(sample_dict.get("timestamp", time.time()))
        ts_iso = str(sample_dict.get("timestamp_iso", ""))

        # 1. Compute rolling features
        features = self.feature_pipeline.push_sample(sample_dict)

        # 2. Score risk: ModelArtifact if present, otherwise dual baseline fallback
        if self.model_artifact is not None:
            _, risk_score = self.model_artifact.predict_sample(features)
        else:
            # Fallback baseline ensemble
            tte_pred = self.fallback_tte.predict_sample(sample_dict)
            stat_pred = self.fallback_static.predict_sample(sample_dict)
            risk_score = max(tte_pred.risk_score, stat_pred.risk_score)

        # 3. Hysteresis State Machine
        previous_state = self.current_state

        if risk_score >= self.config.at_risk_threshold:
            self._high_risk_streak += 1
            self._recovery_streak = 0
            if self._high_risk_streak >= self.config.hysteresis_ticks:
                self.current_state = TerminalState.AT_RISK
            else:
                self.current_state = TerminalState.WATCH
        elif risk_score >= self.config.watch_risk_threshold:
            self._high_risk_streak = 0
            self._recovery_streak = 0
            self.current_state = TerminalState.WATCH
        else:
            self._high_risk_streak = 0
            self._recovery_streak += 1
            # Require 3 consecutive low-risk ticks to step down from AT-RISK to OK
            if self._recovery_streak >= self.config.hysteresis_ticks:
                self.current_state = TerminalState.OK
            elif self.current_state == TerminalState.AT_RISK:
                self.current_state = TerminalState.WATCH

        state_changed = self.current_state != previous_state

        # 4. Generate diagnosis and guidance
        reason, cust_msg, cashier_guide = self._generate_explanation(
            sample_dict, features, risk_score
        )

        # Check outbound alert cooldown
        now_mono = time.monotonic()
        should_send = state_changed or (
            self.current_state == TerminalState.AT_RISK
            and (now_mono - self._last_alert_time) >= self.config.alert_cooldown_sec
        )
        if should_send:
            self._last_alert_time = now_mono

        return AlertPayload(
            terminal_id=self.terminal_id,
            state=self.current_state,
            risk_score=round(risk_score, 3),
            reason=reason,
            customer_message=cust_msg,
            cashier_guidance=cashier_guide,
            top_proc_name=str(sample_dict.get("top_proc_name", "unknown")),
            top_proc_mem_mb=float(sample_dict.get("top_proc_mem_mb", 0.0)),
            timestamp=ts,
            timestamp_iso=ts_iso,
            state_changed=state_changed,
        )
