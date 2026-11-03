"""Unit tests for the on-device early warning state machine and alert detector."""

import pytest

from agent.config import AgentConfig
from agent.detector import EarlyWarningDetector, TerminalState


class MockModelArtifact:
    """Mock artifact returning controlled risk scores."""

    def __init__(self, risk_score: float = 0.0):
        self.risk_score = risk_score

    def predict_sample(self, features: dict) -> tuple[bool, float]:
        return (self.risk_score >= 0.5), self.risk_score


def _create_mock_sample(
    timestamp: float,
    mem_percent: float = 50.0,
    commit_percent: float = 40.0,
    cpu_percent: float = 20.0,
    top_proc: str = "pos_app.exe",
    top_mem: float = 250.0,
) -> dict:
    return {
        "timestamp": timestamp,
        "timestamp_iso": f"2026-11-03T12:00:{int(timestamp % 60):02d}Z",
        "episode_id": "test_ep",
        "mem_percent": mem_percent,
        "commit_percent": commit_percent,
        "cpu_percent": cpu_percent,
        "ctx_switches_s": 15000.0,
        "disk_write_mb_s": 0.5,
        "heartbeat_lag_s": 0.01,
        "top_proc_name": top_proc,
        "top_proc_cpu": 10.0,
        "top_proc_mem_mb": top_mem,
    }


def test_detector_initial_state_ok():
    """Verify detector starts in OK state."""
    detector = EarlyWarningDetector()
    sample = _create_mock_sample(100.0)

    alert = detector.process_sample(sample)

    assert alert.state == TerminalState.OK
    assert alert.risk_score == 0.0
    assert "Welcome" in alert.customer_message


def test_detector_watch_transition():
    """Verify moderate risk triggers WATCH state immediately."""
    mock_model = MockModelArtifact(risk_score=0.60)
    detector = EarlyWarningDetector(model_artifact=mock_model)

    sample = _create_mock_sample(101.0)
    alert = detector.process_sample(sample)

    assert alert.state == TerminalState.WATCH
    assert alert.risk_score == 0.60
    assert alert.state_changed


def test_detector_hysteresis_to_at_risk():
    """Verify transition to AT_RISK requires N consecutive high-risk ticks."""
    mock_model = MockModelArtifact(risk_score=0.85)
    config = AgentConfig(hysteresis_ticks=3, at_risk_threshold=0.80)
    detector = EarlyWarningDetector(model_artifact=mock_model, config=config)

    # Tick 1: Escalates to WATCH (streak = 1)
    a1 = detector.process_sample(_create_mock_sample(101.0))
    assert a1.state == TerminalState.WATCH

    # Tick 2: Stays in WATCH (streak = 2)
    a2 = detector.process_sample(_create_mock_sample(102.0))
    assert a2.state == TerminalState.WATCH

    # Tick 3: Breaches hysteresis -> AT_RISK!
    a3 = detector.process_sample(_create_mock_sample(103.0))
    assert a3.state == TerminalState.AT_RISK
    assert a3.state_changed
    assert "Finish active transaction" in a3.cashier_guidance
    assert "Routine counter service in progress" in a3.customer_message


def test_detector_recovery_cooldown():
    """Verify recovery requires consecutive healthy ticks to clear back to OK."""
    mock_model = MockModelArtifact(risk_score=0.90)
    config = AgentConfig(hysteresis_ticks=3)
    detector = EarlyWarningDetector(model_artifact=mock_model, config=config)

    # Trigger AT_RISK
    for i in range(3):
        detector.process_sample(_create_mock_sample(100.0 + i))
    assert detector.current_state == TerminalState.AT_RISK

    # Switch model to healthy (risk = 0.1)
    mock_model.risk_score = 0.10

    # Tick 1: Drops to WATCH (probation)
    r1 = detector.process_sample(_create_mock_sample(103.0))
    assert r1.state == TerminalState.WATCH

    # Tick 2: Still WATCH
    r2 = detector.process_sample(_create_mock_sample(104.0))
    assert r2.state == TerminalState.WATCH

    # Tick 3: Fully clears to OK!
    r3 = detector.process_sample(_create_mock_sample(105.0))
    assert r3.state == TerminalState.OK


def test_escalation_never_suppressed_by_cooldown():
    """Verify that an escalation from WATCH to AT_RISK emits should_send_network=True immediately."""
    mock_model = MockModelArtifact(risk_score=0.60)
    config = AgentConfig(alert_cooldown_sec=30.0, hysteresis_ticks=2, at_risk_threshold=0.80)
    detector = EarlyWarningDetector(model_artifact=mock_model, config=config)

    # 1. State changes from OK to WATCH -> must send
    a1 = detector.process_sample(_create_mock_sample(100.0))
    assert a1.state == TerminalState.WATCH
    assert a1.should_send_network is True

    # 2. Repeated WATCH 1 second later -> throttled by cooldown!
    a2 = detector.process_sample(_create_mock_sample(101.0))
    assert a2.state == TerminalState.WATCH
    assert a2.should_send_network is False

    # 3. Model score escalates to 0.90 -> breaches threshold to AT_RISK
    mock_model.risk_score = 0.90
    detector.process_sample(_create_mock_sample(102.0))  # streak 1
    a4 = detector.process_sample(_create_mock_sample(103.0))  # streak 2 -> AT_RISK

    # Escalation MUST send immediately, NOT throttled
    assert a4.state == TerminalState.AT_RISK
    assert a4.should_send_network is True

