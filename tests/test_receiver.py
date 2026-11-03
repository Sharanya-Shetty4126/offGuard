"""Unit tests for the FastAPI store reception receiver and tiered heartbeat engine."""

import os
import sqlite3
import tempfile
import pytest
from fastapi import HTTPException

from agent.config import DEFAULT_CONFIG
from receiver.server import (
    AlertIn,
    HeartbeatIn,
    ShutdownIn,
    evaluate_health_tier,
    graceful_shutdown,
    ingest_alert,
    ingest_heartbeat,
    list_alerts,
    list_terminals,
    reception_dashboard,
    verify_token,
    _init_db,
)


@pytest.fixture
def temp_db():
    """Create isolated in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS terminals (
            terminal_id TEXT PRIMARY KEY,
            last_seen_epoch REAL NOT NULL,
            last_seen_iso TEXT NOT NULL,
            state TEXT NOT NULL,
            health_tier TEXT NOT NULL,
            last_seq INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            reason TEXT,
            top_proc_name TEXT,
            top_proc_mem_mb REAL,
            mem_percent REAL,
            cpu_percent REAL,
            is_maintenance INTEGER NOT NULL DEFAULT 0,
            updated_at_epoch REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            terminal_id TEXT NOT NULL,
            received_epoch REAL NOT NULL,
            received_iso TEXT NOT NULL,
            state TEXT NOT NULL,
            risk_score REAL NOT NULL,
            reason TEXT NOT NULL,
            customer_message TEXT,
            cashier_guidance TEXT,
            top_proc_name TEXT,
            top_proc_mem_mb REAL
        )
        """
    )
    conn.commit()

    yield conn

    conn.close()


def test_evaluate_health_tier_thresholds():
    """Verify multi-tier health classification: <=5s HEALTHY, 5-15s SUSPECT, >15s DOWNED, and MAINTENANCE."""
    now = 1000.0

    # 1. Tier 1: Healthy (<= 5s)
    assert evaluate_health_tier(now - 2.0, now, is_maintenance=False) == "HEALTHY"
    assert evaluate_health_tier(now - 5.0, now, is_maintenance=False) == "HEALTHY"

    # 2. Tier 2: Suspect / Lagging (5s - 15s)
    assert evaluate_health_tier(now - 6.0, now, is_maintenance=False) == "SUSPECT"
    assert evaluate_health_tier(now - 14.5, now, is_maintenance=False) == "SUSPECT"

    # 3. Tier 3: Downed / Frozen (> 15s)
    assert evaluate_health_tier(now - 16.0, now, is_maintenance=False) == "DOWNED"
    assert evaluate_health_tier(now - 60.0, now, is_maintenance=False) == "DOWNED"

    # 4. Graceful Maintenance suppresses Downed
    assert evaluate_health_tier(now - 100.0, now, is_maintenance=True) == "MAINTENANCE"


def test_receiver_auth_verification():
    """Verify unauthorized access is blocked when missing or invalid token."""
    # Valid token succeeds
    assert verify_token(DEFAULT_CONFIG.auth_token) == DEFAULT_CONFIG.auth_token

    # Missing token raises 401
    with pytest.raises(HTTPException) as exc:
        verify_token(None)
    assert exc.value.status_code == 401

    # Invalid token raises 401
    with pytest.raises(HTTPException) as exc:
        verify_token("invalid-secret-token")
    assert exc.value.status_code == 401


def test_receiver_heartbeat_and_sequence_tracking(temp_db):
    """Verify heartbeat ingestion, server receive timestamps, and sequence drop detection."""
    # Heartbeat 1
    hb1 = HeartbeatIn(
        terminal_id="POS_COUNTER_09",
        sequence_num=1,
        state="OK",
        risk_score=0.05,
        mem_percent=52.0,
        cpu_percent=15.0,
        top_proc_name="pos.exe",
        top_proc_mem_mb=420.0,
    )
    res1 = ingest_heartbeat(payload=hb1, token=DEFAULT_CONFIG.auth_token, conn=temp_db)
    assert res1["status"] == "ack"
    assert res1["health_tier"] == "HEALTHY"
    assert not res1["sequence_drop_detected"]
    assert "server_receive_time" in res1

    # Heartbeat 2: Dropped packet simulation (sequence jumps 1 -> 4)
    hb2 = HeartbeatIn(
        terminal_id="POS_COUNTER_09",
        sequence_num=4,
        state="OK",
        risk_score=0.05,
    )
    res2 = ingest_heartbeat(payload=hb2, token=DEFAULT_CONFIG.auth_token, conn=temp_db)
    assert res2["sequence_drop_detected"] is True


def test_receiver_alert_ingestion_and_audit(temp_db):
    """Verify high-priority degradation alert updates terminal state and writes audit log."""
    # Ingest baseline heartbeat first
    hb = HeartbeatIn(terminal_id="POS_COUNTER_09", sequence_num=1)
    ingest_heartbeat(payload=hb, token=DEFAULT_CONFIG.auth_token, conn=temp_db)

    alert = AlertIn(
        terminal_id="POS_COUNTER_09",
        state="AT_RISK",
        risk_score=0.88,
        reason="Memory rising rapidly (+0.42%/s); Commit limit approaching",
        customer_message="Maintenance in progress",
        cashier_guidance="Finish active transaction and pause queue",
        top_proc_name="pos.exe",
        top_proc_mem_mb=3600.0,
    )
    res = ingest_alert(payload=alert, token=DEFAULT_CONFIG.auth_token, conn=temp_db)
    assert res["status"] == "alert_recorded"

    # Query terminals list
    terminals = list_terminals(conn=temp_db)
    target = next((t for t in terminals if t["terminal_id"] == "POS_COUNTER_09"), None)
    assert target is not None
    assert target["state"] == "AT_RISK"
    assert target["risk_score"] == 0.88

    # Query audit alerts
    alerts = list_alerts(limit=10, conn=temp_db)
    assert any(a["terminal_id"] == "POS_COUNTER_09" and a["state"] == "AT_RISK" for a in alerts)


def test_receiver_graceful_shutdown(temp_db):
    """Verify graceful shutdown marks terminal as MAINTENANCE and suppresses downed alarms."""
    # Register terminal
    hb = HeartbeatIn(terminal_id="POS_COUNTER_09", sequence_num=1)
    ingest_heartbeat(payload=hb, token=DEFAULT_CONFIG.auth_token, conn=temp_db)

    shut = ShutdownIn(
        terminal_id="POS_COUNTER_09",
        reason="Cashier evening closing",
    )
    res = graceful_shutdown(payload=shut, token=DEFAULT_CONFIG.auth_token, conn=temp_db)
    assert res["status"] == "maintenance_enabled"

    # Query status
    terminals = list_terminals(conn=temp_db)
    target = next((t for t in terminals if t["terminal_id"] == "POS_COUNTER_09"), None)
    assert target is not None
    assert target["health_tier"] == "MAINTENANCE"
    assert target["is_maintenance"] is True


def test_dashboard_endpoint():
    """Verify HTML reception dashboard renders successfully."""
    res = reception_dashboard()
    assert res.status_code == 200
    assert "offGuard Store Reception Monitor" in res.body.decode("utf-8")
    assert "LIVE LAN FEED" in res.body.decode("utf-8")
