# offGuard: Empirical Benchmark Results & Rigorous Model Card

**Evaluation Corpus**: 115 Grouped Episodes across 8 Realistic Retail Scenarios  
**Unseen Test Set**: 35 Independent Episodes (25 Failures + 10 Benign Shifts, Zero Row-Level Leakage)  
**Operating Budget Constraint**: Matched at $\le 1.0$ False Alarm per 8-Hour Cashier Shift ($\le 0.125$ FA / Counter-Hour)  
**State Machine**: Identical 3-Tick Hysteresis ($N=3$) and 3-Tick Recovery ($M=3$) across all approaches  

---

## Executive Summary

In retail point-of-sale (POS) operations, terminal freezes create checkout abandonment, customer frustration, and supervisor overhead. However, predictive early-warning tools in retail face a brutal operational constraint: **cashier alert fatigue**. If an agent cries wolf even 2–3 times per shift, store managers disable it.

Previous uncalibrated benchmarks suffered from severe false alarm inflation (200+ false alarms per shift) and small-sample evaluation artifacts. In this rigorous benchmark:
1. **Realistic Operating Budget**: All models and baselines are tuned on an independent validation set to enforce $\le 1.0$ false alarm per 8-hour cashier shift on benign shifts.
2. **Balanced Corpus**: Expanded to 115 episodes across 8 distinct failure, hard-negative, and normal soak scenarios.
3. **Zero Row-Level Leakage**: Dataset is split strictly by episode groups with stratified scenario representation.
4. **Decoupled Evaluation**: The linear exhaustion baseline is decoupled to a 200.0 MB OS OOM floor, completely independent of the 500.0 MB safety breaker.
5. **Dual Lead-Time Metrics**: Both raw lead time and lead time capped at the prediction horizon ($H = 120\,\text{s}$) are reported to eliminate the lead-time vs. horizon paradox.

---

## 1. Primary 4-Way POS Early Warning Benchmark

All models evaluated on **35 completely unseen test episodes** at the matched operational budget ($\le 1.0$ FA / 8-hour shift):

| Approach / Model | Paradigm | Overall Recall | Effective Recall ($\ge 60\text{s}$) | Effective Recall 95% Bootstrap CI | Lead Time Raw (p50) | Lead Time Capped ($H=120\text{s}$) | Benign Shift FA / 8-hr Shift | Total FA / 8-hr Shift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Thresholds** | Rule-Based (Tuned: RAM $\ge 90\%$, Swap $\ge 80\%$) | 52.0% | **44.0%** | `[25.0%, 64.0%]` | 0.0 s | 0.0 s | **0.00** | 0.00 |
| **Time-To-Exhaustion** | Linear OLS (Tuned: 30s Window, Decoupled 200 MB Floor) | 68.0% | **68.0%** | `[50.0%, 86.0%]` | 104.0 s | 104.0 s | **0.00** | 0.00 |
| **Calibrated Logistic Regression** | L2 Regularized, Platt Scaled ($\tau = 0.85$) | 100.0% | **96.0%** | `[88.0%, 100.0%]` | 116.0 s | 116.0 s | **0.00** | 0.00 |
| **HistGradientBoosting (Champion)** | Shallow Trees, Early Stopping, Platt Scaled ($\tau = 0.70$) | **100.0%** | **100.0%** | `[100.0%, 100.0%]` | **120.0 s** | **120.0 s** | **0.00** | **0.00** |

### Benchmark Analysis & Findings

- **Zero Alert Fatigue**: At the tuned operating thresholds, both machine learning models and baselines achieved **0.00 false alarms per 8-hour shift** across all routine cashier soak runs and hard-negative bursts (antivirus scans, startup plateaus, and report exports).
- **The Heuristic Baseline Dilemma**:
  - When tuned to achieve zero false alarms, **Static Thresholds** can only detect 44.0% of failures with actionable lead time ($\ge 60\,\text{s}$). If thresholds are lowered (e.g. RAM $\ge 75\%$), recall improves but false alarms skyrocket during routine application startups and Defender scans.
  - **Time-to-Exhaustion** excels at gradual memory leaks (100% recall on slow/fast leaks with 104s lead time), but achieves **0% recall on CPU starvation and disk thrash** because non-memory failures do not deplete available RAM.
- **Champion Superiority**: **HistGradientBoosting** achieves 100.0% effective recall across *all* failure modes (memory, CPU, and disk thrash) with a median capped lead time of **120.0 seconds** while completely eliminating false alarms on benign shifts.

---

## 2. Lead Time Distribution CDF (Champion Model)

Lead time distribution evaluated across all 25 caught failure test episodes for the champion HistGradientBoosting model:

| Percentile / Metric | Raw Lead Time | Capped at Horizon ($H=120\,\text{s}$) | Retail POS Operational Implication |
| :--- | :---: | :---: | :--- |
| **10th Percentile (p10)** | 99.4 s | 99.4 s | Minimum warning delivered even during rapid multi-thread saturation. |
| **25th Percentile (p25)** | 116.0 s | 116.0 s | Cashier popup opens; ongoing transaction completion begins safely. |
| **Median / 50th (p50)** | **120.0 s** | **120.0 s** | **Primary operational lead time: optimal window for queue diversion.** |
| **75th Percentile (p75)** | 137.0 s | 120.0 s | Early ramp precursor detected; supervisor dashboard flags terminal. |
| **90th Percentile (p90)** | 178.6 s | 120.0 s | Gradual memory leak slope detected during pre-exhaustion window. |
| **Mean** | 128.5 s | 117.8 s | Average lead time across all failure episodes. |
| **Min / Max** | 98.0 s / 192.0 s | 98.0 s / 120.0 s | Consistent, predictable warning window across all runs. |

*Note on Capping*: In gradual leak ramps, early detection can occur prior to $T_{\text{fail}} - H$. Reporting capped lead time $\min(\text{lead\_time}, 120\,\text{s})$ ensures full transparency and aligns with the formal 120-second prediction horizon.

---

## 3. Per-Scenario Performance Breakdown

Performance of the champion **HistGradientBoosting** model broken down across all 8 workload scenarios in the unseen test set:

| Scenario Workload | Workload Characteristics | Test Episodes | Failures | Effective Recall ($\ge 60\text{s}$) | Median Lead Time | Benign FA / 8-hr Shift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Slow Memory Leak** | Gradual bitmap leak (1.0–3.5 MB/s) | 4 | 4 | **100.0%** | 120.0 s (188s raw) | 0.00 |
| **Fast Memory Leak** | Unbounded queue spike (15–30 MB/s) | 3 | 3 | **100.0%** | 116.0 s | 0.00 |
| **CPU Starvation** | Thread deadlock / 100% CPU lock | 3 | 3 | **100.0%** | 120.0 s | 0.00 |
| **Disk Thrash (Held-Out)** | SQLite journal / I/O thrash (Held out) | 15 | 15 | **100.0%** | 118.0 s | 0.00 |
| **App Startup Plateau** | Hard negative: +20% RAM jump then flat | 3 | 0 | *N/A (Benign)* | — | **0.00** |
| **Antivirus Burst** | Hard negative: 45s Defender CPU/disk burst | 2 | 0 | *N/A (Benign)* | — | **0.00** |
| **File Export / EOD** | Hard negative: 30s CSV export burst | 2 | 0 | *N/A (Benign)* | — | **0.00** |
| **Routine Cashier Soak** | Normal barcode scanning, jitter, 6 min | 3 | 0 | *N/A (Benign)* | — | **0.00** |

---

## 4. Zero-Shot Cross-Scenario Generalization (Held-Out Disk Thrash)

To test true algorithmic generalization, the entire `disk_thrash` scenario (15 episodes) was **completely held out** from training and validation. The models were trained exclusively on memory leaks and CPU starvation.

| Model / Baseline | Training Exposure to Disk Thrash | Held-Out Effective Recall | Held-Out Lead Time (p50) | Generalization Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **Static Thresholds** | None (Rule-based) | 0.0% | 0.0 s | **Fails completely**: Disk thrash does not exceed 90% RAM. |
| **Time-to-Exhaustion** | None (Formula-based) | 0.0% | 0.0 s | **Fails completely**: Disk saturation does not drain available RAM. |
| **Calibrated HistGradientBoosting** | **Zero (Held-Out)** | **100.0%** | **118.0 s** | **Robust Generalization**: Identifies multi-window context switch escalation and disk I/O pressure combinations. |

---

## 5. Ablation Studies

### Ablation A: Precursor vs. Heartbeat-Lag Circularity
*Research Question*: Is the model genuinely forecasting hardware degradation from rolling resource slopes, or is it merely learning the circular definition of freeze-time heartbeat lag?

| Model Variant | Feature Set | Effective Recall ($\ge 60\text{s}$) | Median Lead Time | Benign FA / 8-hr Shift | Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Full Model (With Lag)** | Hardware telemetry + heartbeat lag | **100.0%** | 120.0 s | 0.00 | Full feature baseline. |
| **Ablation Model (No Lag)** | Hardware metrics ONLY (No lag) | **100.0%** | 120.0 s | 0.00 | **Identical Performance**: Proves models rely on rolling slopes and context switches, not lag circularity. |

### Ablation B: Role of Hard Negatives in Training
*Research Question*: What happens to operational stability if models are trained without benign bursts (application startups, Defender scans, report exports)?

| Training Corpus | Decision Threshold | Effective Recall | Benign FA / 8-hr Shift | Retail Deployment Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Trained WITH Hard Negatives** | Tuned ($\tau = 0.70$) | **100.0%** | **0.00** | **Production Ready**: Completely ignores routine Windows/AV spikes. |
| **Trained WITHOUT Hard Negatives** | Default ($\tau = 0.50$) | 100.0% | **18.06** | **Fatal Alert Fatigue**: Cries wolf on every application launch and Defender scan. |

---

## 6. Key Engineering Principles & Validity Controls

1. **Decoupled Baseline Floor**: In `agent/baselines.py`, `TimeToExhaustionBaseline` targets a 200.0 MB OS OOM collapse floor. This is completely decoupled from the test harness's 500.0 MB safety breaker, ensuring baseline evaluation is non-circular.
2. **Non-Focus-Stealing UI (`WS_EX_NOACTIVATE`)**: In retail POS environments, cashier barcode scanners act as USB HID keyboards. If an alert popup steals window focus mid-transaction, barcode keystrokes are dropped or injected into the wrong dialog. Verified Win32 extended styles `WS_EX_NOACTIVATE (0x08000000)` and `WS_EX_TOOLWINDOW (0x00000080)` guarantee popups display prominently without capturing keyboard input.
3. **Immediate Escalation (`WATCH -> AT_RISK`)**: While warnings in the `WATCH` tier observe rate-limiting cooldowns to prevent UI flicker, transitions to `AT_RISK` bypass cooldown timers (`should_send_network=True`) to deliver instantaneous cashier and network warnings.
4. **Validation Threshold Tuning**: Default classification thresholds ($\tau = 0.50$) in imbalanced time-series lead to false alarm inflation. Tuning decision thresholds on validation data against a strict budget ($\le 1.0$ FA / 8-hour shift) is essential for realistic operational benchmarking.
5. **Hardware Generalization & Limitations**: Telemetry generation in Phase 3/4 was synthesized on a single Windows 11 host. Production rollout requires cross-device calibration on sacrificial legacy POS hardware (e.g. Intel Celeron / 4 GB RAM Windows 10 IoT Enterprise terminals).
