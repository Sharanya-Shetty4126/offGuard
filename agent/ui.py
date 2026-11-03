"""Decoupled standalone cashier warning UI for offGuard.

A lightweight, always-on-top floating Tkinter window designed for retail counters:
1. Dual-Audience Presentation:
   - Customer-Facing Banner: Professional, calm maintenance copy (avoids customer panic).
   - Cashier Guidance Drawer: Bold, actionable operational guidance (finish transaction, pause queue).
2. Live Diagnostics: Displays current RAM, Commit, CPU, and top offending process.
3. Minimal Footprint: Built on standard Python tkinter (~5 MB RAM, zero Electron/PyQt overhead).
4. Headless & Replay Safe: Handles headless test environments gracefully.
"""

from datetime import datetime
import os
import sys
import threading
import time
from typing import Optional

from agent.collector import TelemetryCollector, TelemetrySample
from agent.detector import AlertPayload, EarlyWarningDetector, TerminalState


class CashierWarningWindow:
    """Tkinter floating always-on-top warning window for cashier terminals."""

    COLOR_OK = "#107c41"       # Forest Green
    COLOR_WATCH = "#d88a10"    # Warm Amber
    COLOR_AT_RISK = "#d83b01"  # Calm Brick Red
    BG_DARK = "#201f1e"        # Modern Charcoal
    TEXT_LIGHT = "#ffffff"     # White
    CARD_BG = "#2b2b2b"        # Card Gray

    def __init__(
        self,
        terminal_id: str = "COUNTER 01",
        headless: bool = False,
    ):
        self.terminal_id = terminal_id
        self.headless = headless
        self.root = None
        self._is_running = False

        if not self.headless:
            self._init_gui()

    def _init_gui(self) -> None:
        """Initialize Tkinter UI widgets."""
        try:
            import tkinter as tk
            from tkinter import ttk

            self.root = tk.Tk()
            self.root.title(f"offGuard Monitor - {self.terminal_id}")
            self.root.geometry("460x320")
            self.root.minsize(420, 280)
            self.root.configure(bg=self.BG_DARK)
            self.root.attributes("-topmost", True)

            # --- 1. Top Header Bar ---
            header_frame = tk.Frame(self.root, bg=self.BG_DARK)
            header_frame.pack(fill="x", padx=16, pady=(12, 6))

            self.lbl_terminal = tk.Label(
                header_frame,
                text=f"● {self.terminal_id}",
                font=("Segoe UI", 11, "bold"),
                fg=self.TEXT_LIGHT,
                bg=self.BG_DARK,
            )
            self.lbl_terminal.pack(side="left")

            self.lbl_clock = tk.Label(
                header_frame,
                text="--:--:--",
                font=("Segoe UI", 10),
                fg="#a19f9d",
                bg=self.BG_DARK,
            )
            self.lbl_clock.pack(side="right")

            # --- 2. Main Status Banner ---
            self.banner_frame = tk.Frame(self.root, bg=self.COLOR_OK, height=52)
            self.banner_frame.pack(fill="x", padx=16, pady=4)
            self.banner_frame.pack_propagate(False)

            self.lbl_status = tk.Label(
                self.banner_frame,
                text="SYSTEM OPTIMAL",
                font=("Segoe UI", 13, "bold"),
                fg=self.TEXT_LIGHT,
                bg=self.COLOR_OK,
            )
            self.lbl_status.pack(expand=True)

            # --- 3. Customer-Facing Notice Card ---
            cust_card = tk.Frame(self.root, bg=self.CARD_BG)
            cust_card.pack(fill="x", padx=16, pady=6)

            lbl_cust_title = tk.Label(
                cust_card,
                text="CUSTOMER NOTICE",
                font=("Segoe UI", 8, "bold"),
                fg="#8a8886",
                bg=self.CARD_BG,
            )
            lbl_cust_title.pack(anchor="w", padx=10, pady=(6, 2))

            self.lbl_customer_msg = tk.Label(
                cust_card,
                text="Counter open. Welcome!",
                font=("Segoe UI", 10),
                fg=self.TEXT_LIGHT,
                bg=self.CARD_BG,
                wraplength=400,
                justify="left",
            )
            self.lbl_customer_msg.pack(anchor="w", padx=10, pady=(0, 6))

            # --- 4. Cashier Action Guidance Card ---
            cashier_card = tk.Frame(self.root, bg=self.CARD_BG)
            cashier_card.pack(fill="x", padx=16, pady=6)

            lbl_cashier_title = tk.Label(
                cashier_card,
                text="CASHIER GUIDANCE",
                font=("Segoe UI", 8, "bold"),
                fg="#8a8886",
                bg=self.CARD_BG,
            )
            lbl_cashier_title.pack(anchor="w", padx=10, pady=(6, 2))

            self.lbl_cashier_guide = tk.Label(
                cashier_card,
                text="Scanning enabled. Ready for items.",
                font=("Segoe UI", 10, "bold"),
                fg="#60a5fa",
                bg=self.CARD_BG,
                wraplength=400,
                justify="left",
            )
            self.lbl_cashier_guide.pack(anchor="w", padx=10, pady=(0, 6))

            # --- 5. Diagnostic Footer ---
            self.lbl_footer = tk.Label(
                self.root,
                text="RAM: --% | Commit: --% | CPU: --% | Top Process: --",
                font=("Segoe UI", 8),
                fg="#a19f9d",
                bg=self.BG_DARK,
            )
            self.lbl_footer.pack(side="bottom", pady=8)

            self._is_running = True
        except Exception as e:
            print(f"[UI Warning] Headless or Tkinter unavailable ({e}). Running in headless mode.")
            self.headless = True
            self.root = None

    def update_alert(
        self,
        alert: AlertPayload,
        sample: Optional[Union[TelemetrySample, dict]] = None,
    ) -> None:
        """Update window labels and color scheme based on incoming alert."""
        if self.headless or self.root is None:
            return

        # Determine banner styling
        if alert.state == TerminalState.AT_RISK:
            bg_col = self.COLOR_AT_RISK
            status_text = f"AT-RISK (FREEZE FORECAST: ~60-120s)"
            guide_col = "#f87171"
        elif alert.state == TerminalState.WATCH:
            bg_col = self.COLOR_WATCH
            status_text = "WATCH - ELEVATED LOAD"
            guide_col = "#fbbf24"
        else:
            bg_col = self.COLOR_OK
            status_text = "SYSTEM OPTIMAL"
            guide_col = "#34d399"

        # Safe update inside Tkinter event loop
        def _apply():
            try:
                self.banner_frame.configure(bg=bg_col)
                self.lbl_status.configure(text=status_text, bg=bg_col)
                self.lbl_customer_msg.configure(text=alert.customer_message)
                self.lbl_cashier_guide.configure(text=alert.cashier_guidance, fg=guide_col)
                self.lbl_clock.configure(text=datetime.now().strftime("%H:%M:%S"))

                if sample is not None:
                    d = sample.to_dict() if hasattr(sample, "to_dict") else sample
                    diag = (
                        f"RAM: {d.get('mem_percent', 0):.0f}% | "
                        f"Commit: {d.get('commit_percent', 0):.0f}% | "
                        f"CPU: {d.get('cpu_percent', 0):.0f}% | "
                        f"Top: {d.get('top_proc_name', 'none')[:10]} ({d.get('top_proc_mem_mb', 0):.0f}M)"
                    )
                    self.lbl_footer.configure(text=diag)
            except Exception:
                pass

        if self.root:
            self.root.after(0, _apply)

    def run_loop(self) -> None:
        """Run Tkinter main event loop."""
        if self.root and not self.headless:
            self.root.mainloop()

    def close(self) -> None:
        """Destroy window."""
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None


def run_live_ui(replay_source: Optional[str] = None, speedup: float = 1.0) -> None:
    """Launch UI paired with either live telemetry collector or replay engine."""
    ui = CashierWarningWindow(terminal_id="COUNTER 01", headless=False)
    detector = EarlyWarningDetector(terminal_id="pos_counter_01")

    # Select telemetry provider
    if replay_source:
        from agent.replay import ReplayTelemetryCollector
        collector = ReplayTelemetryCollector(source=replay_source, speedup_factor=speedup, realtime=True, loop=True)
    else:
        collector = TelemetryCollector()

    def _worker():
        while True:
            sample = collector.collect_sample()
            if sample is None:
                break
            alert = detector.process_sample(sample)
            ui.update_alert(alert, sample)
            time.sleep(1.0 if not replay_source else (1.0 / speedup))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    ui.run_loop()


if __name__ == "__main__":
    replay_file = sys.argv[1] if len(sys.argv) > 1 else None
    speed_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    run_live_ui(replay_source=replay_file, speedup=speed_factor)
