"""Episode visualization tool for offGuard telemetry.

Renders high-resolution multi-panel diagnostic plots of recorded episodes:
1. Physical RAM & Windows Commit Charge (%) with Available RAM (MB).
2. Host CPU Utilization (%) and Context Switches per Second.
3. Scheduler Heartbeat Lag (s) with failure threshold lines.
4. Ground-Truth Markers: T_failure marker, 60s/120s pre-failure target windows, and masked post-failure zone.

Usage:
    python -m training.plot_episode data/episodes/ep_xxx [output_path.png]
"""

import argparse
import csv
import json
import os
import sys
import tempfile

# Configure safe matplotlib cache directory before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "offguard_mpl"))

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt


def plot_episode(
    episode_source: str,
    output_png_path: Optional[str] = None,
    dpi: int = 150,
) -> str:
    """Generate and save a multi-panel telemetry plot for an episode."""
    csv_path = episode_source
    manifest_path = None

    if os.path.isdir(episode_source):
        csv_path = os.path.join(episode_source, "telemetry.csv")
        possible_manifest = os.path.join(episode_source, "manifest.json")
        if os.path.exists(possible_manifest):
            manifest_path = possible_manifest

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Telemetry CSV not found at '{csv_path}'")

    # Read CSV rows
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"No telemetry data in '{csv_path}'")

    # Read manifest if present
    manifest_data = {}
    if manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    # Extract time-series arrays
    base_t = float(rows[0].get("timestamp", 0.0))
    time_rel_s = [(float(r.get("timestamp", 0.0)) - base_t) for r in rows]
    cpu_pct = [float(r.get("cpu_percent", 0.0)) for r in rows]
    mem_pct = [float(r.get("mem_percent", 0.0)) for r in rows]
    avail_ram_mb = [float(r.get("mem_available_mb", 0.0)) for r in rows]
    commit_pct = [float(r.get("commit_percent", r.get("swap_percent", 0.0))) for r in rows]
    ctx_switches = [float(r.get("ctx_switches_s", 0.0)) for r in rows]
    heartbeat_lag = [float(r.get("heartbeat_lag_s", 0.0)) for r in rows]

    # Target metadata
    has_failed = manifest_data.get("has_failed", False)
    failure_ts = manifest_data.get("failure_timestamp")
    failure_mode = manifest_data.get("failure_mode", "none")
    scenario_type = manifest_data.get("scenario_type", "unknown")
    ep_id = manifest_data.get("episode_id", os.path.basename(os.path.dirname(csv_path)))

    # If manifest wasn't present, check CSV for target flags
    if not has_failed:
        post_indices = [i for i, r in enumerate(rows) if str(r.get("is_post_failure", "")).lower() == "true"]
        if post_indices:
            has_failed = True
            failure_ts = float(rows[post_indices[0]].get("timestamp", 0.0))

    rel_failure_s = (failure_ts - base_t) if (has_failed and failure_ts is not None) else None

    # Create figure with 3 stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    title_text = f"offGuard Telemetry Diagnosis: {ep_id} [{scenario_type.upper()}]"
    if has_failed:
        title_text += f" (Failure Detected @ +{rel_failure_s:.1f}s via {failure_mode})"
    else:
        title_text += " (Healthy / Negative - Zero Failure)"
    fig.suptitle(title_text, fontsize=13, fontweight="bold")

    # ---------------- Panel 1: Memory & Commit ----------------
    ax1.plot(time_rel_s, mem_pct, label="RAM Utilization %", color="#1f77b4", linewidth=1.8)
    ax1.plot(time_rel_s, commit_pct, label="Commit Charge %", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Utilization (%)", fontweight="semibold")
    ax1.set_ylim(-2, 105)
    ax1.grid(True, alpha=0.3)

    # Secondary axis for Available RAM MB
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_rel_s, avail_ram_mb, label="Free RAM (MB)", color="#2ca02c", linewidth=1.2, linestyle=":")
    ax1_twin.axhline(500.0, color="red", linestyle=":", linewidth=1.0, alpha=0.7, label="Safe Floor (500MB)")
    ax1_twin.set_ylabel("Available RAM (MB)", color="#2ca02c", fontweight="semibold")
    ax1_twin.tick_params(axis="y", labelcolor="#2ca02c")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", framealpha=0.85, fontsize=8)

    # ---------------- Panel 2: CPU & Context Switches ----------------
    ax2.plot(time_rel_s, cpu_pct, label="Host CPU %", color="#9467bd", linewidth=1.6)
    ax2.set_ylabel("CPU (%)", color="#9467bd", fontweight="semibold")
    ax2.set_ylim(-2, 105)
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(time_rel_s, ctx_switches, label="Context Switches/s", color="#8c564b", linewidth=1.0, alpha=0.7)
    ax2_twin.set_ylabel("Ctx Switches/s", color="#8c564b", fontweight="semibold")
    ax2_twin.tick_params(axis="y", labelcolor="#8c564b")

    lines_a, labels_a = ax2.get_legend_handles_labels()
    lines_b, labels_b = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", framealpha=0.85, fontsize=8)

    # ---------------- Panel 3: Heartbeat Lag & Targets ----------------
    ax3.plot(time_rel_s, heartbeat_lag, label="Heartbeat Lag (s)", color="#d62728", linewidth=1.8)
    ax3.axhline(3.0, color="darkred", linestyle="--", linewidth=1.2, label="Lag Degradation Threshold (3.0s)")
    ax3.set_ylabel("Lag (seconds)", fontweight="semibold")
    ax3.set_xlabel("Elapsed Time (seconds)", fontweight="semibold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", framealpha=0.85, fontsize=8)

    # ---------------- Shading Target Horizons ----------------
    if has_failed and rel_failure_s is not None:
        for ax in (ax1, ax2, ax3):
            # 1. Mark T_failure
            ax.axvline(rel_failure_s, color="red", linestyle="-", linewidth=2.0, alpha=0.9)

            # 2. Positive 120s Pre-Failure Target Horizon [T - 120s, T)
            h_start = max(0.0, rel_failure_s - 120.0)
            ax.axvspan(h_start, rel_failure_s, color="#ffd700", alpha=0.15, label="Target Horizon H (120s)")

            # 3. Actionable Effective Warning Zone [T - 120s, T - 60s]
            effective_end = max(0.0, rel_failure_s - 60.0)
            if effective_end > h_start:
                ax.axvspan(h_start, effective_end, color="#2ca02c", alpha=0.12, label="Actionable Warning (>=60s)")

            # 4. Masked Post-Failure Zone [T, end]
            ax.axvspan(rel_failure_s, max(time_rel_s), color="gray", alpha=0.20, label="Masked Post-Failure")

    plt.tight_layout()

    # Determine output path
    if output_png_path is None:
        if os.path.isdir(episode_source):
            out_file = os.path.join(episode_source, "plot.png")
        else:
            out_file = os.path.splitext(csv_path)[0] + "_plot.png"
    else:
        out_file = output_png_path

    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[offGuard Plotter] Rendered diagnostic plot -> '{out_file}'")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot offGuard episode telemetry")
    parser.add_argument("source", type=str, help="Path to episode folder or telemetry.csv")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="Image resolution DPI")
    args = parser.parse_args()

    plot_episode(args.source, output_png_path=args.out, dpi=args.dpi)

