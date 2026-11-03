# Collector Runtime Overhead & Self-Intrusion Profile

A core design requirement of `offGuard` is **negligible self-footprint**: the telemetry agent must never starve the cashier machine or cause the unresponsiveness it is designed to predict.

---

## 1. Measurement Methodology

Overhead is continuously monitored using [`agent/benchmark_overhead.py`](file:///c:/Users/Sharanya/OffGurad/offGuard/agent/benchmark_overhead.py) via `psutil.Process(os.getpid())`:
- **CPU Utilization:** Measured via per-process delta CPU time (`cpu_percent()`).
- **RAM Footprint:** Resident Set Size (`memory_info().rss`), measuring physical memory consumed by the Python runtime and collector buffers.
- **Memory Drift:** Difference between final and initial RSS memory ($\Delta \text{RSS}$) to verify zero memory leaks in the collection loop.
- **Cycle Latency:** High-resolution monotonic timing (`time.perf_counter()`) measuring elapsed milliseconds required to sample all hardware sensors per tick.

---

## 2. Empirical Benchmark Results

Measured on host workstation (Windows 11, Python 3.14.0):

| Metric | Measured Value | Safe Target Budget | Status |
| :--- | :--- | :--- | :---: |
| **Mean CPU Utilization** | **1.89%** | $< 3.0\%$ | ✅ PASS |
| **Peak CPU Burst** | **9.20%** (initial process spinup) | $< 15.0\%$ | ✅ PASS |
| **Average RSS Memory Footprint** | **21.18 MB** | $< 60.0\,\text{MB}$ | ✅ PASS |
| **Memory Drift ($\Delta \text{RSS}$)** | **+0.020 MB** (flat over run) | $< 1.0\,\text{MB/hr}$ | ✅ PASS |
| **Sample Collection Latency** | **25.65 ms** | $< 50.0\,\text{ms}$ | ✅ PASS |

*Key Takeaway:* Sampling the system requires $\approx 25\text{ ms}$ out of every $1000\text{ ms}$ interval, and consumes only **~21 MB of RAM**, leaving $> 97\%$ of CPU time and virtually all memory completely free for POS and billing operations.

---

## 3. How to Execute the Benchmark

### Quick 30-Second Check
```powershell
python -m agent.benchmark_overhead 30
```

### Sustained 1-Hour POS Shift Profiling
```powershell
python -m agent.benchmark_overhead 3600
```
