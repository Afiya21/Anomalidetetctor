# Custom Anomaly Detector for Workstations — Final Project Report
**Course:** DCML 25-26  
**Student:** [Your Name]  
**Date:** [Date]  

---

## 1. The Problem to be Solved
Modern workstations occasionally experience silent degradation or malicious activity—such as ransomware-like disk writes, memory leaks, or CPU-hogging background processes. The goal of this project is to develop an end-to-end anomaly detector capable of identifying deviations from normal system behavior in real-time, operating silently with minimal overhead, and triggering severity-based alerts when anomalous conditions are met.

## 2. Monitor Design and Indicator Selection
To capture a comprehensive picture of system health, a custom `SystemMonitor` was built using the `psutil` library. Instead of merely logging instantaneous values, the monitor maintains a 5-second sliding window to calculate both the **rolling mean** and **rolling standard deviation** of each metric. This dramatically reduces per-second noise and allows the ML models to detect trends (e.g., a gradual memory leak) rather than just isolated spikes.

**Selected KPIs and Rationale:**
- `cpu_percent` & `cpu_core_max`: Detects multi-core saturation (e.g., cryptomining) and single-core thread locks.
- `cpu_freq_mhz`: Monitors thermal throttling or sustained turbo-boost behavior.
- `mem_percent` & `mem_available_mb`: Captures memory exhaustion attacks and runaway allocations.
- `disk_read_bytes_sec` & `disk_write_bytes_sec`: Identifies bulk file access, a strong indicator of ransomware activity or local data scraping.
- `net_sent_bytes_sec` & `net_recv_bytes_sec`: Detects data exfiltration or inbound network floods.
- `open_sockets`: Spikes indicate potential network scanning or botnet activity.
- `num_processes`: Detects fork-bombs and excessive child-process spawning.

By engineering 11 raw features into 33 time-aware features (raw + mean + std), the monitor provides a rich feature space for the detector.

## 3. Dataset Generation and Anomaly Injection
A controlled data collection phase was orchestrated (`collect_data.py`) to build a labeled dataset (`training_data.csv`). This process consisted of four distinct phases:
1. **Normal Phase (Label 0):** Standard workstation usage including browsing, typing, and idle periods.
2. **CPU Anomaly (Label 1):** Injected using pure Python busy-loop threads scaled to the number of logical cores.
3. **Memory Anomaly (Label 2):** Simulated via incremental `bytearray` allocations, ramping up memory usage gradually to mimic a memory leak.
4. **Disk Anomaly (Label 3):** Simulated by continuously writing and reading a large block of random bytes to a temporary file, mimicking a ransomware encryption cycle.

By using granular labels (0 to 3), we could later perform deep analysis on *which specific type of anomaly* was hardest to detect, going beyond simple binary classification.

## 4. Model Training, Selection, and Methodology
To prevent "look-ahead bias"—a common methodological flaw where future time-series data leaks into the training set—the models were evaluated using **TimeSeriesSplit cross-validation** instead of random shuffling.

**Algorithms Evaluated:**
1. **Z-Score (Statistical Baseline):** Flags anomalies if any feature deviates >3 standard deviations from the normal mean.
2. **Isolation Forest (Unsupervised):** Constructs isolation trees; effective for high-dimensional anomaly detection without labels.
3. **One-Class SVM (Unsupervised):** Learns a strict boundary around the normal data manifold.
4. **Logistic Regression (Supervised):** Serves as an interpretable linear baseline.
5. **Random Forest (Supervised):** Handles complex, non-linear interactions between system metrics.

**Evaluation Metrics:**
Because anomaly detection datasets are inherently imbalanced (few anomalies vs. many normal events), *Accuracy* was discarded as a metric. Instead, the models were ranked based on:
- **F1-Score (Primary):** Balances precision and recall.
- **Recall:** Missing an actual anomaly is considered far worse than generating a false alert.
- **ROC-AUC:** Threshold-independent performance.

Class imbalance was handled algorithmically by setting `class_weight='balanced'` in the supervised models and training unsupervised models exclusively on normal data.

*(Insert `correlation_heatmap.png` and `feature_importance_rf.png` here to discuss which features contributed most to the detection).*

## 5. Per-Anomaly-Type Analysis
While models were trained on a binary classification task (Normal vs. Anomaly), predictions were mapped back to the granular labels to determine recall per anomaly type.

*(Discuss which anomaly—CPU, Memory, or Disk—had the lowest recall and why based on the console output from `train.py`)*

## 6. Runtime Integration and Silent Processing
The final `detector.py` script loads the best performing model (Random Forest) and its associated `StandardScaler`. It runs continuously, reading from the monitor every second. 

To satisfy the requirement of "silent processing", the detector monitors its own resource footprint:
- **Inference Latency:** Measured in milliseconds per prediction via `time.perf_counter()`.
- **Self CPU/RAM Overhead:** The Python process self-reports its resource usage via `psutil`.

The detector implements a **Severity Engine** based on model probabilities:
- `Probability < 0.40`: **NORMAL**
- `0.40 <= Probability < 0.70`: **WARNING**
- `Probability >= 0.70`: **CRITICAL**

All predictions, along with latency and overhead metrics, are persistently logged to `logs/runtime_log.csv` for forensic analysis, successfully fulfilling the goal of a standalone, end-to-end workstation anomaly detector.
