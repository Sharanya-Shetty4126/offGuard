# Modeling Assumptions, Failure Definitions, and Scope Boundaries

## 1. Scope and Target Horizon
* **Target Horizon ($H$):** 60 to 120 seconds ($H = 120$ seconds default; $60$s and $90$s also evaluated).
* **Predictable Failures:** Gradual exhaustion regimes (memory leaks, swap thrashing, sustained CPU starvation, high disk queue saturation).
* **Unpredictable Failures:** Instantaneous hardware failures, kernel panics without precursors, power interruptions, or spontaneous driver halts. These are caught after-the-fact via the reception receiver's **missed-heartbeat alert**, not by the prediction model.

## 2. Ground-Truth Definition of "Unresponsive"
A terminal is defined as "unresponsive" at time $T_{\text{failure}}$ when:
* The collector/probe event loop experiences a delay exceeding a threshold (e.g. scheduled delay $> 3.0$ seconds for $N \ge 3$ consecutive ticks), OR
* The OS message queue latency exceeds a critical threshold where UI events fail to process within acceptable POS bounds.

## 3. Data Partitioning and Temporal Boundaries
To prevent data contamination:
1. **Normal Operating State (Label 0):** System under routine cashier operations and background noise.
2. **Pre-Failure Transition Window $[T_{\text{failure}} - H, T_{\text{failure}}]$ (Label 1):** The lead-time window where precursor degradation is developing.
3. **Post-Failure Zone ($t > T_{\text{failure}}$):** Explicitly **discarded** from model training. Training on an already-frozen system creates false, trivial signals (e.g. flatlined 0 I/O) rather than precursor dynamics.

## 4. Evaluation Standards
* **Episode Splits:** Training, validation, and test sets are split strictly by *discrete episodes* (recording runs) in chronological sequence, never by random row sampling.
* **Primary Metrics:**
  * **Warning Lead Time:** Distribution (median, 25th, 75th percentiles) of seconds of advance notice provided before actual unresponsiveness.
  * **False Alarms per Counter-Hour (FA/hr):** Quantifies operational disruption to cashier throughput.
  * **Precision-Recall Area (PR-AUC) and Brier Score / Calibration:** Reflects reliability of risk probabilities under severe class imbalance.
* **Baseline Heuristic:** Every ML model must be benchmarked against a static threshold rule (e.g., memory $> 92\%$ AND swap $> 85\%$ for $\ge 10$ seconds) to verify whether ML provides real incremental value.
