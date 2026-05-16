"""
collect_data.py — Labeled Data Collection Orchestrator
========================================================
Runs the SystemMonitor through 4 sequential phases, activating the
appropriate anomaly injector during each anomaly phase, and labels
every sampled row accordingly.

Labels written to training_data.csv:
    0 = Normal       (no injector — natural laptop usage)
    1 = CPU Anomaly  (CPUAnomalyInjector)
    2 = Memory Anomaly (MemoryAnomalyInjector)
    3 = Disk Anomaly   (DiskAnomalyInjector)

Output: data/training_data.csv

Usage:
    python src/collect_data.py

Durations are read from config.json:
    collection.normal_duration_sec   (default: 120 s)
    collection.anomaly_duration_sec  (default: 60  s)
"""

import csv
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from src/
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from monitor  import SystemMonitor, get_feature_names
from injector import CPUAnomalyInjector, MemoryAnomalyInjector, DiskAnomalyInjector

CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from config.json."""
    if not os.path.exists(CONFIG_PATH):
        # Sensible defaults if config.json hasn't been written yet
        return {
            "monitor":    {"rolling_window": 5, "sample_interval_sec": 1.0},
            "collection": {"normal_duration_sec": 120, "anomaly_duration_sec": 60,
                           "training_file": "training_data.csv"},
            "labels":     {"0": "Normal", "1": "CPU Anomaly",
                           "2": "Memory Anomaly", "3": "Disk Anomaly"},
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core collection function
# ---------------------------------------------------------------------------

def collect_phase(
    monitor: SystemMonitor,
    duration_sec: int,
    label: int,
    label_name: str,
    injector=None,
) -> list[dict]:
    """
    Collect labeled rows for one phase.

    Args:
        monitor:      Initialised SystemMonitor (rolling window already warm).
        duration_sec: How long to collect in seconds.
        label:        Integer label attached to every row.
        label_name:   Human-readable name used for console output.
        injector:     Optional anomaly injector; started before and stopped
                      after the collection window.

    Returns:
        List of row dicts — each contains 33 features + label + timestamp.
    """
    rows: list[dict] = []

    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  Phase {label} | {label_name}")
    print(f"  Duration : {duration_sec}s")
    if injector:
        print(f"  Injector : {type(injector).__name__}")
    print(bar)

    if injector:
        injector.start()

    try:
        deadline = time.time() + duration_sec
        while time.time() < deadline:
            remaining = int(deadline - time.time())
            row = monitor.sample()

            if row is not None:
                row["label"]     = label
                row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                rows.append(row)

                # Live console feedback every 10 seconds
                n = len(rows)
                if n % 10 == 0 or remaining <= 3:
                    cpu = row["cpu_percent"]
                    mem = row["mem_percent"]
                    print(
                        f"  [{time.strftime('%H:%M:%S')}] "
                        f"rows={n:>4}  cpu={cpu:>5.1f}%  mem={mem:>5.1f}%  "
                        f"remaining={remaining}s"
                    )

            time.sleep(1)

    finally:
        # Always stop the injector, even if an exception occurs
        if injector:
            injector.stop()

    print(f"  ✅ Phase {label} complete — {len(rows)} rows collected.")
    return rows


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    config      = load_config()
    m_cfg       = config["monitor"]
    c_cfg       = config["collection"]
    label_names = config["labels"]

    normal_dur  = c_cfg["normal_duration_sec"]
    anomaly_dur = c_cfg["anomaly_duration_sec"]
    window      = m_cfg["rolling_window"]

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, c_cfg["training_file"])

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DCML Anomaly Detector — Data Collection")
    print("=" * 60)
    print(f"  Normal phase  : {normal_dur}s")
    print(f"  Anomaly phases: {anomaly_dur}s each")
    total = normal_dur + 3 * anomaly_dur
    print(f"  Total runtime : ~{total}s ({total // 60}m {total % 60}s)")
    print(f"  Output        : {output_path}")

    # ── Initialise monitor & warm up rolling window ──────────────────────────
    monitor = SystemMonitor(
        window_size=window,
        interval=m_cfg["sample_interval_sec"],
    )

    print(f"\n  Warming up rolling window ({window} ticks)...")
    for i in range(window):
        monitor.sample_raw()
        print(f"    tick {i + 1}/{window}", end="\r")
        time.sleep(1)
    print(f"  Warm-up complete.{' ' * 20}")

    # ── Phase 0: Normal ───────────────────────────────────────────────────────
    print("\n  ⚡ Phase 0: Use your laptop normally (browse, type, idle).")
    all_rows = collect_phase(
        monitor, normal_dur, label=0,
        label_name=label_names["0"],
        injector=None,
    )

    # ── Phase 1: CPU Anomaly ─────────────────────────────────────────────────
    all_rows += collect_phase(
        monitor, anomaly_dur, label=1,
        label_name=label_names["1"],
        injector=CPUAnomalyInjector(intensity=1.0),
    )

    # ── Phase 2: Memory Anomaly ───────────────────────────────────────────────
    all_rows += collect_phase(
        monitor, anomaly_dur, label=2,
        label_name=label_names["2"],
        injector=MemoryAnomalyInjector(target_mb=512, chunk_mb=50),
    )

    # ── Phase 3: Disk Anomaly ─────────────────────────────────────────────────
    all_rows += collect_phase(
        monitor, anomaly_dur, label=3,
        label_name=label_names["3"],
        injector=DiskAnomalyInjector(file_size_mb=100),
    )

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not all_rows:
        print("\n❌ No data collected — exiting.")
        return

    feature_cols = get_feature_names()
    fieldnames   = ["timestamp"] + feature_cols + ["label"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ Saved {len(all_rows)} rows → {output_path}")
    print("\n  Label distribution:")

    from collections import Counter
    counts = Counter(r["label"] for r in all_rows)
    for lbl in sorted(counts):
        name  = label_names[str(lbl)]
        count = counts[lbl]
        bar   = "█" * (count // 2)
        print(f"    Label {lbl} ({name:<15}) : {count:>4} rows  {bar}")

    print("=" * 60)
    print("\n  Next step: run  python src/train.py")


if __name__ == "__main__":
    main()
