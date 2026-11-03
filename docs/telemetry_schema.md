# Telemetry Data Specification & Schema Contract

This document specifies the exact telemetry schema, data types, physical units, and target labeling rules used across the `offGuard` pipeline (Collector, Feature Pipeline, Stress Harness, and Model Training).

---

## 1. Raw Telemetry Schema (`raw_telemetry`)

Each record represents one instantaneous measurement taken at interval $\Delta t \approx 1.0$ second.

| Field Name | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `float` | Seconds | Unix epoch timestamp (`time.time()`). |
| `timestamp_iso` | `string` | ISO-8601 | Human-readable UTC timestamp (e.g. `2026-10-28T05:55:00.123Z`). |
| `episode_id` | `string` | Identifier | Unique run/test identifier (e.g. `ep_20261028_baseline_01`). |
| `cpu_percent` | `float` | Percentage (0–100) | Overall host CPU utilization across all logical cores. |
| `mem_percent` | `float` | Percentage (0–100) | Physical RAM utilization. |
| `mem_available_mb` | `float` | Megabytes (MB) | Physical memory immediately available without paging. |
| `swap_percent` | `float` | Percentage (0–100) | OS swap / commit charge percentage. |
| `commit_percent` | `float` | Percentage (0–100) | **Windows Commit Charge %:** Ratio of active committed virtual memory to total system commit limit. Queried via `GlobalMemoryStatusEx` (immune to PDH counter errors). |
| `commit_available_mb` | `float` | Megabytes (MB) | Uncommitted memory headroom remaining before OS commit limit exhaustion. |
| `ctx_switches_s` | `float` | Switches / second | CPU context switches per second (`psutil.cpu_stats()`). Indicator of thread contention and thrashing. |
| `disk_read_mb_s` | `float` | MB/second | Read throughput delta since previous sample. |
| `disk_write_mb_s` | `float` | MB/second | Write throughput delta since previous sample. |
| `heartbeat_lag_s` | `float` | Seconds | Scheduler drift calculated strictly using `time.monotonic()` ($\Delta t - 1.0\,\text{s}$). Immune to system clock adjustments and NTP synchronizations. |
| `top_proc_name` | `string` | Name | Process consuming highest CPU or RSS memory during this tick. *(Privacy: Command line arguments are strictly excluded).* |
| `top_proc_cpu` | `float` | Percentage (0–100) | CPU utilization of the top process. |
| `top_proc_mem_mb` | `float` | Megabytes (MB) | Resident memory (RSS) consumed by top process. |

---

## 2. Derived Feature Schema (`features`)

Features are computed using rolling temporal buffers of length $W \in \{10, 30, 60\}$ seconds. 
**Strict Rule:** Features at timestamp $t$ only consume samples with timestamp $\le t$ (zero future leakage).

For each base metric $M \in \{\text{cpu\_percent}, \text{mem\_percent}, \text{commit\_percent}, \text{ctx\_switches\_s}, \text{disk\_write\_mb\_s}, \text{heartbeat\_lag\_s}\}$ and window $W$:

* `m_{M}_mean_{W}s`: Rolling arithmetic average over window $W$.
* `m_{M}_slope_{W}s`: Rate of change per second (linear trend over window $W$). A positive slope in memory or commit indicates an active leak.
* `m_{M}_std_{W}s`: Rolling standard deviation over window $W$ (volatility/jitter).

---

## 3. Labeling and Ground-Truth Definitions

### Dual Failure Definition ($T_{\text{failure}}$)
A system unresponsiveness event begins at $T_{\text{failure}}$, defined as the earliest occurrence of either:

1. **Sustained Degradation:** $\text{heartbeat\_lag\_s} \ge 3.0\,\text{s}$ for $\ge 3$ consecutive ticks ($T_{\text{failure}}$ = timestamp of the first tick of the streak).
2. **Catastrophic Freeze / Timestamp Gap:** If the OS hangs completely, the collector itself is halted and cannot record consecutive samples. Thus, an unresponsiveness event is also triggered if:
   * A single tick arrives with $\text{heartbeat\_lag\_s} \ge 6.0\,\text{s}$ ($T_{\text{failure}} = t - \text{lag}$), OR
   * A missing tick gap occurs between adjacent timestamps where $t_i - t_{i-1} \ge 6.0\,\text{s}$ ($T_{\text{failure}} = t_{i-1} + 1.0\,\text{s}$).

### Prediction Horizon ($H$) & Effective Retail Lead Time
* **Target Horizon ($H$):** $H = 120$ seconds default (evaluated at $60$s, $90$s, $120$s).
* **Target Label (`target_fail_within_H`):**
  $$Y(t) = \begin{cases} 1 & \text{if } T_{\text{failure}} - H \le t < T_{\text{failure}} \\ 0 & \text{if } t < T_{\text{failure}} - H \end{cases}$$
* **Effective Lead Time Criterion:** An alert arriving with $< 60\,\text{s}$ lead time is marked as a *late warning* (insufficient for a cashier to finish a transaction and divert the queue). The primary evaluation metric is **Effective Recall ($\ge 60\,\text{s}$)**.
* **Post-Failure Masking (`is_post_failure`):**
  $$t \ge T_{\text{failure}} \implies \text{DROPPED from training dataset}$$
  *(Prevents models from learning trivial flatlines of an already frozen system).*

---

## 4. Hardware Safety & Circuit Breaker Limits

During induced failure runs on local hardware, scripts must strictly respect:
1. **RAM Floor:** If `mem_available_mb < 500.0 MB`, immediately terminate and release memory blocks.
2. **Hard Timeout:** Induced failure scripts must have an unconditioned timeout (maximum 90 seconds).
3. **Emergency Cleanup:** Registered `atexit` and `SIGINT`/`SIGTERM` handlers to release OS resources.
