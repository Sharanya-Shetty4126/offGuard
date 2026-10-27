# offGuard

> **Predictive unresponsiveness early-warning agent for checkout & cash-counter terminals.**

`offGuard` is a lightweight, on-device telemetry agent designed to run on point-of-sale (POS) and cashier computers. It monitors hardware stress signals in real time to forecast whether a machine is likely to become unresponsive within the next **1 to 2 minutes** ($H = 60\text{--}120\,\text{s}$). 

By detecting gradual degradation (memory leaks, swap thrashing, scheduler starvation) before an outright freeze occurs, it enables the cashier to finish the active transaction, route the queue to an open counter, and notify store reception.

---

## 🚫 What It Is NOT

* **Not a payment API gateway monitor:** It does not predict UPI, credit card, or bank network failures.
* **Not a remote surveillance tool:** Each counter machine monitors only itself locally.
* **Not a magical predictor of instantaneous crashes:** A sudden kernel crash, power cut, or hardware disconnect without precursors cannot be predicted.
* **No synthetic or inflated accuracy numbers:** Every reported metric reflects measured data from real runs, compared against a deterministic threshold baseline.

---

## 🏗️ Architecture & How It Works

```
[ OS Telemetry Loop ] -> (psutil: CPU, RAM, Swap, Disk I/O, Heartbeat Lag)
       │
       ▼
[ Rolling Features ]   -> (10s, 30s, 60s windows: mean, slope, variance)
       │
       ▼
[ Inference Engine ]   -> (Threshold Baseline vs. Logistic Regression / Gradient Boosting)
       │
       ▼
[ State Manager ]      -> (Hysteresis & Cooldown: OK ➔ WATCH ➔ AT-RISK)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ Local Cashier UI ]             [ Reception Receiver ]
(Decoupled standalone window)    (FastAPI Alert Feed & Heartbeat Watcher)
```

1. **Collector:** Samples hardware metrics every 1–2 seconds: CPU %, virtual memory %, swap usage, disk I/O delta, top process usage, and collector loop lag.
2. **Features:** Rolling statistical windows (10s, 30s, 60s) computing mean, slope, and variance. Features at time $t$ strictly use data $\le t$ (zero future leakage).
3. **State Machine:** Smooths risk predictions using hysteresis ($N$ consecutive high readings) to avoid alert flickering. States: `OK`, `WATCH`, `AT-RISK`.
4. **Dual Alert Channels:**
   * **Local Warning Window:** An independent, lightweight always-on-top window. Neutral, calm customer-facing copy; actionable cashier guidance.
   * **Reception Feed:** Sends signed lightweight alerts over HTTP on state changes. If a terminal goes completely silent, reception raises a *Missed Heartbeat* alert.

---

## 🔬 Evaluation & Methodological Rigor

* **Episode-based Splitting:** Splits are made across discrete test episodes, never across random shuffled rows, preserving temporal causality.
* **Lead-Time Evaluation:** Measures the actual warning lead time (in seconds) provided before the machine becomes unresponsive.
* **False Alarms per Counter-Hour (FA/hr):** Prioritizes minimizing cashier disruptions over standard unweighted accuracy.
* **Baseline Comparison:** Directly compares machine learning models against a straightforward heuristic threshold rule to verify whether ML delivers real practical value.

---

## 📁 Repository Structure

```text
offGuard/
├── agent/            # On-device collector, feature pipeline, state machine, and local UI
├── training/         # Failure injection scripts, dataset collectors, training & evaluation
├── receiver/         # FastAPI reception dashboard & missed-heartbeat monitor
├── docs/             # Technical specifications, assumptions, and research notes
├── models/           # Exported model weights and pipeline artifacts (.joblib, gitignored)
├── tests/            # Unit tests for temporal leakage, sensor failures, and state machine
├── requirements.txt  # Project dependencies
└── README.md
```

---

## 📋 Implementation Phases

- [x] **Phase 0:** Project scoping, git repository setup, architecture specification.
- [ ] **Phase 1:** Telemetry Collector (robust sampling, missing sensor handling, scheduler lag).
- [ ] **Phase 2:** Controlled Failure Injection & Episode Data Collection (memory ramp, CPU burner, I/O thrash).
- [ ] **Phase 3:** Feature Engineering & Model Evaluation (leakage verification, baseline vs ML, lead-time metrics).
- [ ] **Phase 4:** Live On-Device Scoring & Decoupled Cashier Warning UI.
- [ ] **Phase 5:** FastAPI Reception Receiver & Missed-Heartbeat Detector.
- [ ] **Phase 6:** End-to-end stress validation, reporting, and demo.

---

## ⚖️ Limitations & Lab Assumptions

Data collected on a development machine using induced failure scripts reflects specific hardware configurations and simulated degradation patterns. Real retail environments present diverse background workloads and edge cases. In compliance with strict engineering honesty, all results document specific test runs and conditions without extrapolation.
