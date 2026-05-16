# DCML 25-26 Custom Anomaly Detector

This is a standalone, end-to-end anomaly detection system built for local workstations. It silently monitors system performance using a custom `psutil` wrapper, calculates rolling statistics over a configurable window, and detects anomalous behavior in real time using a trained Random Forest model.

## Features

- **Live System Monitoring:** Tracks 11 core KPIs related to CPU, memory, disk, network, and processes. Uses a 5-second sliding window to generate 33 robust features (raw + mean + std).
- **Anomaly Injectors:** Includes pure Python injectors to safely simulate CPU spikes, memory leaks, and ransomware-style disk I/O for dataset generation.
- **Methodologically Sound Validation:** Implements `TimeSeriesSplit` to evaluate models without look-ahead bias, typical in time-series data.
- **Comprehensive Evaluation:** Compares statistical baselines (Z-Score), unsupervised models (Isolation Forest, One-Class SVM), and supervised models (Logistic Regression, Random Forest).
- **Runtime Alert Engine:** Features self-monitoring (overhead tracking) and severity-level alerts based on predicted probabilities.

---

## Project Structure

```
Anomalidetetctor/
├── config.json                 # Central configuration 
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── monitor.py              # System KPI sampler
│   ├── injector.py             # CPU, Memory, Disk anomaly simulators
│   ├── collect_data.py         # Orchestrates collection into training_data.csv
│   ├── train.py                # ML training & model selection
│   └── detector.py             # Runtime real-time anomaly detector
│
├── data/
│   └── training_data.csv       # Dataset created by collect_data.py
│
├── models/                     # Saved models and scaler
├── logs/                       # Runtime predictions and latency logs
│
└── reports/
    ├── figures/                # Heatmaps and confusion matrices
    ├── metrics/                # CSV tables of model performance
    └── report_draft.md         # Draft of the final project report
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
Run the orchestrator to build your local dataset. It will guide you through 4 phases: normal usage, a CPU attack, a memory attack, and a disk attack.
```bash
python src/collect_data.py
```

### 3. Train and Select Models
Evaluate 5 different algorithms using time-aware validation. This script generates evaluation plots in `reports/figures/` and saves the best model to `models/`.
```bash
python src/train.py
```

### 4. Run the Runtime Detector
Start the live detector. It will silently process data every second, measure its own resource overhead, and log predictions to `logs/runtime_log.csv`.
```bash
python src/detector.py
```

---

## Configuration

All thresholds, durations, and model parameters can be adjusted in `config.json`. No hardcoded values exist in the source code.
