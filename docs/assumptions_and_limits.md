# Modeling Assumptions, Failure Definitions, and Scope Boundaries

## 1. Scope and Target Horizon
* **Target Horizon ($H$):** 60 to 120 seconds ($H = 120$ seconds default; $60$s and $90$s also evaluated).
* **Actionable Retail Window:** While the model predicts whether failure occurs within $H \le 120\text{s}$, warnings occurring with $< 60\,\text{s}$ of lead time are classified as **Late Warnings**. The primary operational goal is **Effective Recall ($\ge 60\,\text{s}$)** to ensure cashiers have sufficient time to finish an in-flight transaction, explain the situation to the customer, and divert the queue.
* **Predictable Failures:** Gradual exhaustion regimes (memory leaks, swap thrashing, sustained CPU starvation, high disk queue saturation).
* **Unpredictable Failures:** Instantaneous hardware disconnects, kernel panics without precursors, power cuts, or spontaneous hardware driver halts. These are caught after-the-fact via the reception receiver's **missed-heartbeat alert**, not by the prediction model.

## 2. Ground-Truth Definition of "Unresponsive" ($T_{\text{failure}}$)
A terminal is defined as "unresponsive" at time $T_{\text{failure}}$ using a robust dual-condition rule:

1. **Sustained Degradation:**
   $$\text{heartbeat\_lag\_s} \ge 3.0\,\text{s for } N \ge 3 \text{ consecutive ticks}$$
   $T_{\text{failure}}$ is the timestamp of the first tick of the consecutive streak.
2. **Catastrophic Freeze / Timestamp Gaps:**
   If the whole operating system or thread scheduler completely freezes, the collector itself is halted and cannot record consecutive samples. When the system unfreezes or recovers, it produces either:
   * A single massive lag sample where $\text{heartbeat\_lag\_s} \ge 6.0\,\text{s}$ ($T_{\text{failure}} = t - \text{lag}$), OR
   * A wall-clock timestamp gap between adjacent ticks where $t_i - t_{i-1} \ge 6.0\,\text{s}$ ($T_{\text{failure}} = t_{i-1} + 1.0\,\text{s}$).

## 3. Data Partitioning and Temporal Boundaries
To prevent data contamination:
1. **Normal Operating State (Label 0):** System under routine cashier operations and background noise.
2. **Pre-Failure Transition Window $[T_{\text{failure}} - H, T_{\text{failure}}]$ (Label 1):** The lead-time window where precursor degradation is developing.
3. **Post-Failure Zone ($t \ge T_{\text{failure}}$):** Explicitly **discarded** from model training. Training on an already-frozen system creates false, trivial signals (e.g. flatlined 0 I/O) rather than precursor dynamics.

## 4. Evaluation Standards
* **Episode Splits:** Training, validation, and test sets are split strictly by *discrete episodes* (recording runs) in chronological sequence, never by random row sampling.
* **Primary Metrics:**
  * **Effective Retail Recall ($\ge 60\,\text{s}$):** Fraction of failure episodes detected with at least 60 seconds of advance warning.
  * **Warning Lead Time Distribution:** Median, mean, min, max, and IQR (25th to 75th percentiles) of advance warning seconds.
  * **Late Warnings (< 60s):** Alerts emitted prior to failure, but too late for queue diversion.
  * **False Alarms per Counter-Hour (FA/hr):** Operational disruptions per normal operating counter-hour.
  * **Precision-Recall Area (PR-AUC) and Calibration:** Reflects reliability of risk probabilities under severe class imbalance.
* **Baseline Benchmarks:** Every ML model must be benchmarked against:
  1. **Static Threshold Rule:** Fixed limits on RAM and Swap over $N$ ticks.
  2. **Linear Time-to-Exhaustion Trend:** OLS slope on available RAM estimating seconds until the 500 MB safety floor.

## 5. Agent Overhead & Self-Intrusion Budget
To ensure monitoring does not degrade cashier performance, the collector is profiled continuously. See detailed empirical profile in [docs/collector_overhead.md](collector_overhead.md):
* **RAM Consumption:** $\approx 23\,\text{MB}$ RSS at idle; no runaway memory drift.
* **CPU Consumption:** $\le 1.5\%$ average idle on modern CPU; projected $\le 5\text{--}8\%$ on POS-class hardware (Celeron J1900 / Elkhart Lake).
* **Collection Latency:** $\approx 8.6\,\text{ms}$ at idle, $\approx 24.2\,\text{ms}$ under active stress per 1.0s sample interval.
