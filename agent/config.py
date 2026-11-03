"""Central configuration for offGuard agent and testing harness."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AgentConfig:
    # --- Telemetry Sampling ---
    sample_interval_sec: float = 1.0  # Sample system stats every 1 second
    top_proc_count: int = 1  # Number of top resource-consuming processes to record

    # --- Feature Engineering & Windows ---
    window_sizes_sec: List[int] = field(default_factory=lambda: [10, 30, 60])
    warmup_duration_sec: float = 60.0  # Initial window filling duration (features warming up)

    # --- Target Horizons & Ground Truth ---
    default_horizon_sec: int = 120  # Forecasting window H in seconds (evaluate 60, 90, 120)
    min_effective_lead_time_sec: float = 60.0  # Minimum actionable warning lead time for cashier
    unresponsive_lag_threshold_sec: float = 3.0  # Heartbeat lag considered degraded/frozen
    consecutive_lag_ticks: int = 3  # Number of delayed ticks to trigger ground-truth freeze
    catastrophic_lag_threshold_sec: float = 6.0  # Single-tick lag or timestamp gap denoting complete OS freeze

    # --- State Machine & Alerting ---
    hysteresis_ticks: int = 3  # Require N consecutive high-risk predictions to trigger AT-RISK
    watch_risk_threshold: float = 0.50  # Probability for WATCH state
    at_risk_threshold: float = 0.80  # Probability for AT-RISK state
    alert_cooldown_sec: float = 30.0  # Minimum delay between outbound reception alerts

    # --- Network & Reception ---
    receiver_base_url: str = "http://localhost:8000"
    alert_endpoint: str = "/api/alerts"
    heartbeat_endpoint: str = "/api/heartbeat"
    auth_token: str = "offguard-dev-token-xyz"
    max_local_queue_size: int = 500  # Local SQLite / file queue when offline

    # --- Hardware Safety & Stress Guardrails ---
    min_safe_ram_mb: float = 500.0  # Auto-kill stress script if free RAM dips below this floor
    max_stress_duration_sec: float = 300.0  # Hard timeout for induced failure scripts (up to 5 min)


# Global default configuration instance
DEFAULT_CONFIG = AgentConfig()
