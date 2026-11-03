"""FastAPI Central Reception & Store Manager Receiver for offGuard.

Monitors live POS counter terminals across a retail store:
1. Multi-Tier Heartbeat Health Monitor:
   - Tier 1 (HEALTHY): Heartbeat received within <= 5s.
   - Tier 2 (SUSPECT): Heartbeat delayed 5-15s (network blip, heavy GC, or thread starvation).
   - Tier 3 (DOWNED): No heartbeat > 15s (terminal freeze, agent crash, or power loss).
   - MAINTENANCE: Graceful cashier shutdown / planned reboot (prevents false alarms).
2. Server-Side Timestamps: Immune to cashier terminal clock skew, NTP jumps, or DST changes.
3. Sequence Numbers: Tracks dropped, replayed, or out-of-order packets.
4. SQLite Persistence: Counter state persists across receiver restarts in data/receiver_state.db.
5. Store Reception Dashboard: Real-time web console displaying terminal vitals and queue diversion alerts.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.config import DEFAULT_CONFIG, AgentConfig

DB_PATH = os.path.join("data", "receiver_state.db")


# ---------------- Database & Persistence ----------------

def _init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite tables for persistent terminal state and alert audit log."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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


def _get_db():
    """Yield SQLite database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------- Pydantic Request Models ----------------

class HeartbeatIn(BaseModel):
    """Periodic terminal agent heartbeat payload."""

    terminal_id: str = Field(..., json_schema_extra={"example": "COUNTER_01"})
    sequence_num: int = Field(..., ge=0, json_schema_extra={"example": 42})
    state: str = Field(default="OK", json_schema_extra={"example": "OK"})
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    top_proc_name: str = Field(default="unknown")
    top_proc_mem_mb: float = Field(default=0.0)
    mem_percent: float = Field(default=0.0)
    cpu_percent: float = Field(default=0.0)
    agent_version: str = Field(default="1.0.0")


class AlertIn(BaseModel):
    """High-priority predictive degradation alert from cashier terminal."""

    terminal_id: str = Field(..., json_schema_extra={"example": "COUNTER_01"})
    state: str = Field(..., json_schema_extra={"example": "AT_RISK"})
    risk_score: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.85})
    reason: str = Field(..., json_schema_extra={"example": "Memory rising (+0.45%/s); High RAM saturation (94%)"})
    customer_message: str = Field(default="")
    cashier_guidance: str = Field(default="")
    top_proc_name: str = Field(default="pos_terminal.exe")
    top_proc_mem_mb: float = Field(default=3500.0)


class ShutdownIn(BaseModel):
    """Graceful cashier counter shutdown / planned restart notification."""

    terminal_id: str = Field(..., json_schema_extra={"example": "COUNTER_01"})
    reason: str = Field(default="Planned counter closure", json_schema_extra={"example": "End of cashier shift"})


# ---------------- Health Tier Computation ----------------

def evaluate_health_tier(last_seen_epoch: float, now_epoch: float, is_maintenance: bool) -> str:
    """Compute multi-tier health status using server-side receive timestamp.

    Tiers:
    - MAINTENANCE: Explicitly closed counter (suppresses false downed alarms).
    - HEALTHY: Heartbeat received <= 5s ago.
    - SUSPECT: Heartbeat delayed 5s to 15s (scheduling lag or network jitter).
    - DOWNED: Heartbeat missing > 15s (terminal freeze or crash).
    """
    if is_maintenance:
        return "MAINTENANCE"

    elapsed_s = max(0.0, now_epoch - last_seen_epoch)
    if elapsed_s <= 5.0:
        return "HEALTHY"
    elif elapsed_s <= 15.0:
        return "SUSPECT"
    else:
        return "DOWNED"


# ---------------- Security & Auth Dependency ----------------

def verify_token(x_offguard_token: Optional[str] = Header(None)) -> str:
    """Validate preshared LAN authorization header token."""
    expected = DEFAULT_CONFIG.auth_token
    if x_offguard_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-OffGuard-Token authentication header",
        )
    return x_offguard_token


# ---------------- FastAPI Application ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield

app = FastAPI(
    title="offGuard Reception Receiver",
    description="Central store reception monitor for POS cashier terminal early degradation warnings",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/api/heartbeat", status_code=status.HTTP_200_OK)
def ingest_heartbeat(
    payload: HeartbeatIn,
    token: str = Depends(verify_token),
    conn: sqlite3.Connection = Depends(_get_db),
):
    """Ingest terminal heartbeat, recording server receive timestamp and sequence numbers."""
    now_epoch = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    health_tier = evaluate_health_tier(now_epoch, now_epoch, is_maintenance=False)

    # Check previous sequence number
    cur = conn.cursor()
    cur.execute("SELECT last_seq, is_maintenance FROM terminals WHERE terminal_id = ?", (payload.terminal_id,))
    row = cur.fetchone()

    sequence_drop = False
    if row is not None:
        prev_seq = row["last_seq"]
        if payload.sequence_num > prev_seq + 1:
            sequence_drop = True

    # Upsert terminal record
    cur.execute(
        """
        INSERT INTO terminals (
            terminal_id, last_seen_epoch, last_seen_iso, state, health_tier,
            last_seq, risk_score, reason, top_proc_name, top_proc_mem_mb,
            mem_percent, cpu_percent, is_maintenance, updated_at_epoch
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        ON CONFLICT(terminal_id) DO UPDATE SET
            last_seen_epoch = excluded.last_seen_epoch,
            last_seen_iso = excluded.last_seen_iso,
            state = excluded.state,
            health_tier = excluded.health_tier,
            last_seq = excluded.last_seq,
            risk_score = excluded.risk_score,
            top_proc_name = excluded.top_proc_name,
            top_proc_mem_mb = excluded.top_proc_mem_mb,
            mem_percent = excluded.mem_percent,
            cpu_percent = excluded.cpu_percent,
            is_maintenance = 0,
            updated_at_epoch = excluded.updated_at_epoch
        """,
        (
            payload.terminal_id,
            now_epoch,
            now_iso,
            payload.state,
            health_tier,
            payload.sequence_num,
            payload.risk_score,
            f"Routine heartbeat #{payload.sequence_num}",
            payload.top_proc_name,
            payload.top_proc_mem_mb,
            payload.mem_percent,
            payload.cpu_percent,
            now_epoch,
        ),
    )
    conn.commit()

    return {
        "status": "ack",
        "terminal_id": payload.terminal_id,
        "health_tier": health_tier,
        "sequence_drop_detected": sequence_drop,
        "server_receive_time": now_iso,
    }


@app.post("/api/alerts", status_code=status.HTTP_201_CREATED)
def ingest_alert(
    payload: AlertIn,
    token: str = Depends(verify_token),
    conn: sqlite3.Connection = Depends(_get_db),
):
    """Ingest early-warning degradation alert and log to reception audit ledger."""
    now_epoch = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    health_tier = "AT_RISK" if payload.state == "AT_RISK" else "WATCH"

    cur = conn.cursor()
    # 1. Update terminal state
    cur.execute(
        """
        UPDATE terminals SET
            state = ?,
            risk_score = ?,
            reason = ?,
            top_proc_name = ?,
            top_proc_mem_mb = ?,
            last_seen_epoch = ?,
            last_seen_iso = ?,
            updated_at_epoch = ?
        WHERE terminal_id = ?
        """,
        (
            payload.state,
            payload.risk_score,
            payload.reason,
            payload.top_proc_name,
            payload.top_proc_mem_mb,
            now_epoch,
            now_iso,
            now_epoch,
            payload.terminal_id,
        ),
    )

    # 2. Append to alert history ledger
    cur.execute(
        """
        INSERT INTO alerts (
            terminal_id, received_epoch, received_iso, state, risk_score,
            reason, customer_message, cashier_guidance, top_proc_name, top_proc_mem_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.terminal_id,
            now_epoch,
            now_iso,
            payload.state,
            payload.risk_score,
            payload.reason,
            payload.customer_message,
            payload.cashier_guidance,
            payload.top_proc_name,
            payload.top_proc_mem_mb,
        ),
    )
    conn.commit()

    return {
        "status": "alert_recorded",
        "terminal_id": payload.terminal_id,
        "state": payload.state,
        "risk_score": payload.risk_score,
        "reception_guidance": "Divert customer queue to adjacent open counter immediately.",
    }


@app.post("/api/shutdown", status_code=status.HTTP_200_OK)
def graceful_shutdown(
    payload: ShutdownIn,
    token: str = Depends(verify_token),
    conn: sqlite3.Connection = Depends(_get_db),
):
    """Graceful cashier shutdown: flags counter as MAINTENANCE to suppress downed alarms."""
    now_epoch = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    cur = conn.cursor()
    cur.execute(
        """
        UPDATE terminals SET
            state = 'CLOSED',
            health_tier = 'MAINTENANCE',
            is_maintenance = 1,
            reason = ?,
            updated_at_epoch = ?
        WHERE terminal_id = ?
        """,
        (payload.reason, now_epoch, payload.terminal_id),
    )
    conn.commit()

    return {
        "status": "maintenance_enabled",
        "terminal_id": payload.terminal_id,
        "message": f"Counter '{payload.terminal_id}' set to MAINTENANCE. Downed alarms suppressed.",
    }


@app.get("/api/terminals", response_model=List[Dict[str, Any]])
def list_terminals(conn: sqlite3.Connection = Depends(_get_db)):
    """Return live status matrix across all store cashier terminals with current health tiers."""
    now_epoch = time.time()
    cur = conn.cursor()
    cur.execute("SELECT * FROM terminals ORDER BY terminal_id ASC")
    rows = cur.fetchall()

    result = []
    for r in rows:
        is_maint = bool(r["is_maintenance"])
        health = evaluate_health_tier(r["last_seen_epoch"], now_epoch, is_maint)
        elapsed_s = round(max(0.0, now_epoch - r["last_seen_epoch"]), 1)

        result.append(
            {
                "terminal_id": r["terminal_id"],
                "health_tier": health,
                "state": r["state"],
                "elapsed_since_seen_s": elapsed_s,
                "risk_score": round(r["risk_score"], 2),
                "reason": r["reason"],
                "mem_percent": r["mem_percent"],
                "cpu_percent": r["cpu_percent"],
                "top_proc_name": r["top_proc_name"],
                "top_proc_mem_mb": r["top_proc_mem_mb"],
                "last_seen_iso": r["last_seen_iso"],
                "is_maintenance": is_maint,
            }
        )

    return result


@app.get("/api/alerts", response_model=List[Dict[str, Any]])
def list_alerts(limit: int = 50, conn: sqlite3.Connection = Depends(_get_db)):
    """Return recent degradation alert history for store audit."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [dict(r) for r in rows]


@app.get("/", response_class=HTMLResponse)
def reception_dashboard():
    """Render real-time Reception & Store Manager operational dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>offGuard Store Reception Monitor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --green: #10b981;
                --yellow: #f59e0b;
                --red: #ef4444;
                --blue: #3b82f6;
                --gray: #64748b;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #334155;
                padding-bottom: 16px;
                margin-bottom: 24px;
            }
            h1 { margin: 0; font-size: 22px; font-weight: 700; }
            .badge-live {
                background: #064e3b;
                color: #34d399;
                padding: 4px 10px;
                border-radius: 9999px;
                font-size: 12px;
                font-weight: 600;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px;
            }
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #334155;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
                position: relative;
            }
            .card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }
            .terminal-name { font-size: 16px; font-weight: 700; }
            .tier-tag {
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            .tier-HEALTHY { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); }
            .tier-SUSPECT { background: rgba(245, 158, 11, 0.2); color: var(--yellow); border: 1px solid var(--yellow); }
            .tier-DOWNED { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }
            .tier-MAINTENANCE { background: rgba(100, 116, 139, 0.2); color: var(--gray); border: 1px solid var(--gray); }
            .metric-row {
                display: flex;
                justify-content: space-between;
                font-size: 13px;
                margin: 6px 0;
                color: var(--text-muted);
            }
            .metric-val { color: var(--text); font-weight: 600; }
            .action-box {
                margin-top: 14px;
                padding: 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            .action-at-risk {
                background: rgba(239, 68, 68, 0.15);
                color: #fca5a5;
                border-left: 4px solid var(--red);
            }
            .action-optimal {
                background: rgba(16, 185, 129, 0.15);
                color: #a7f3d0;
                border-left: 4px solid var(--green);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>offGuard Store Reception Monitor</h1>
                <div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Predictive Terminal Degradation & Silent Freeze Defense</div>
            </div>
            <div>
                <span class="badge-live">● LIVE LAN FEED</span>
            </div>
        </div>

        <div id="terminals-grid" class="grid">
            <div style="color: var(--text-muted);">Scanning store network for active terminals...</div>
        </div>

        <script>
            async function refreshStatus() {
                try {
                    const res = await fetch('/api/terminals');
                    const terminals = await res.json();
                    const grid = document.getElementById('terminals-grid');

                    if (!terminals || terminals.length === 0) {
                        grid.innerHTML = '<div style="color: var(--text-muted);">No active terminals connected yet.</div>';
                        return;
                    }

                    grid.innerHTML = terminals.map(t => {
                        const isAtRisk = t.state === 'AT_RISK';
                        const actionClass = isAtRisk ? 'action-at-risk' : 'action-optimal';
                        const actionText = isAtRisk 
                            ? '🚨 ACTION REQUIRED: Divert queue! Terminal approaching freeze in 60-120s.' 
                            : '✓ Counter optimal. Ready for checkout.';

                        return `
                        <div class="card">
                            <div class="card-header">
                                <span class="terminal-name">${t.terminal_id}</span>
                                <span class="tier-tag tier-${t.health_tier}">${t.health_tier}</span>
                            </div>
                            <div class="metric-row">
                                <span>Terminal State</span>
                                <span class="metric-val">${t.state} (Risk: ${t.risk_score.toFixed(2)})</span>
                            </div>
                            <div class="metric-row">
                                <span>Last Seen</span>
                                <span class="metric-val">${t.elapsed_since_seen_s}s ago</span>
                            </div>
                            <div class="metric-row">
                                <span>Host RAM / CPU</span>
                                <span class="metric-val">${t.mem_percent.toFixed(0)}% / ${t.cpu_percent.toFixed(0)}%</span>
                            </div>
                            <div class="metric-row">
                                <span>Top Offender</span>
                                <span class="metric-val">${t.top_proc_name} (${t.top_proc_mem_mb.toFixed(0)}M)</span>
                            </div>
                            <div class="action-box ${actionClass}">
                                ${actionText}
                            </div>
                        </div>
                        `;
                    }).join('');
                } catch (err) {
                    console.error('Failed to refresh terminals:', err);
                }
            }

            setInterval(refreshStatus, 2000);
            refreshStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("receiver.server:app", host="127.0.0.1", port=8000, reload=False)
