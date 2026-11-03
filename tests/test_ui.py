"""Unit tests for the cashier warning UI window."""

import pytest

from agent.detector import AlertPayload, TerminalState
from agent.ui import CashierWarningWindow


def _create_mock_alert(state: TerminalState, risk: float = 0.0) -> AlertPayload:
    return AlertPayload(
        terminal_id="COUNTER 01",
        state=state,
        risk_score=risk,
        reason="Test diagnosis",
        customer_message="Customer copy",
        cashier_guidance="Cashier action",
        top_proc_name="pos.exe",
        top_proc_mem_mb=120.0,
        timestamp=100.0,
        timestamp_iso="2026-11-03T12:00:00Z",
        state_changed=True,
    )


def test_ui_headless_mode():
    """Verify headless UI instantiation and safe alert ingestion without GUI display."""
    ui = CashierWarningWindow(terminal_id="COUNTER 01", headless=True)

    assert ui.headless
    assert ui.root is None

    # Update with OK
    ui.update_alert(_create_mock_alert(TerminalState.OK, 0.1))

    # Update with WATCH
    ui.update_alert(_create_mock_alert(TerminalState.WATCH, 0.6))

    # Update with AT_RISK
    ui.update_alert(_create_mock_alert(TerminalState.AT_RISK, 0.9))

    # Clean close
    ui.close()
