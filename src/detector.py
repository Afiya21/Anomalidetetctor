"""
detector.py — Runtime Anomaly Detector
=========================================
Loads the trained ML model and runs a continuous monitoring loop.
Evaluates system KPIs in real-time and logs predictions to a CSV.

Features:
  - Real-time inference using the Random Forest best model.
  - Severity levels based on model probability (Normal, Warning, Critical).
  - Self-monitoring: measures inference latency and self CPU/RAM usage
    to prove low-overhead "silent processing".
  - Persistent logging to logs/runtime_log.csv.

Usage:
    python src/detector.py
"""

import csv
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import psutil
from colorama import Back, Fore, Style, init

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path & Encoding Setup
# ---------------------------------------------------------------------------
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

init(autoreset=True)

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from monitor import SystemMonitor, get_feature_names

CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"[ERROR] Config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    config = load_config()
    d_cfg  = config["detector"]
    
    model_path = os.path.join(PROJECT_DIR, d_cfg["model_path"])
    scaler_path = os.path.join(PROJECT_DIR, d_cfg["scaler_path"])
    log_path    = os.path.join(PROJECT_DIR, d_cfg["log_path"])
    
    warn_thresh = d_cfg["warning_threshold"]
    crit_thresh = d_cfg["critical_threshold"]

    print("=" * 70)
    print("  DCML Anomaly Detector — Runtime Engine")
    print("=" * 70)

    # 1. Load models
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        sys.exit(f"[ERROR] Missing model or scaler. Run python src/train.py first.")
        
    print(f"  Loading model  : {d_cfg['model_path']}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. Setup Monitor
    window   = d_cfg["rolling_window"]
    interval = d_cfg["sample_interval_sec"]
    monitor  = SystemMonitor(window_size=window, interval=interval)
    
    feature_cols = get_feature_names()

    # 3. Setup Logger
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", newline="")
    fieldnames = [
        "timestamp", "prediction", "probability", "latency_ms",
        "detector_cpu_pct", "detector_ram_mb", 
        "cpu_percent", "mem_percent", "disk_read_bytes_sec", "disk_write_bytes_sec",
        "net_sent_bytes_sec", "net_recv_bytes_sec"
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames, extrasaction="ignore")
    
    # Write header only if file is empty
    if os.path.getsize(log_path) == 0:
        writer.writeheader()

    print(f"  Logging to     : {d_cfg['log_path']}")
    print(f"  Warning thresh : >= {warn_thresh:.2f}")
    print(f"  Critical thresh: >= {crit_thresh:.2f}\n")
    
    print("  Warming up monitor (5 ticks)...")
    for _ in range(window):
        monitor.sample_raw()
        time.sleep(1)

    print("  " + "─" * 68)
    print(f"  {'Timestamp':<20} | {'Status':<10} | {'Prob':>4} | {'CPU%':>5} {'MEM%':>5} | {'Latency':>7}")
    print("  " + "─" * 68)

    self_proc = psutil.Process(os.getpid())
    self_proc.cpu_percent() # Warmup self cpu measurement

    try:
        while True:
            # Measure self overhead
            self_cpu = self_proc.cpu_percent()
            self_ram = self_proc.memory_info().rss / (1024 ** 2)

            # Sample system data
            row = monitor.sample()
            if row is None:
                time.sleep(interval)
                continue
            
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Extract features in exact order
            X_raw = np.array([[row[f] for f in feature_cols]])
            
            # --- Inference Timing Start ---
            t0 = time.perf_counter()
            
            X_scaled = scaler.transform(X_raw)
            prob     = model.predict_proba(X_scaled)[0, 1]
            
            t1 = time.perf_counter()
            # --- Inference Timing End ---
            
            latency_ms = (t1 - t0) * 1000

            # Severity Logic
            if prob >= crit_thresh:
                status = "CRITICAL"
                color = Back.RED + Fore.WHITE + Style.BRIGHT
            elif prob >= warn_thresh:
                status = "WARNING"
                color = Back.YELLOW + Fore.BLACK
            else:
                status = "NORMAL"
                color = Fore.GREEN

            # Output to console
            cpu_val = row['cpu_percent']
            mem_val = row['mem_percent']
            
            print(f"  {ts:<20} | {color}{status:<10}{Style.RESET_ALL} | {prob:.2f} | {cpu_val:>5.1f} {mem_val:>5.1f} | {latency_ms:>5.1f}ms")

            # Log to CSV
            writer.writerow({
                "timestamp": ts,
                "prediction": status,
                "probability": round(prob, 4),
                "latency_ms": round(latency_ms, 2),
                "detector_cpu_pct": round(self_cpu, 2),
                "detector_ram_mb": round(self_ram, 2),
                **row  # Includes the raw KPI values
            })
            log_file.flush()
            
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n  [OK] Detector stopped by user.")
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
