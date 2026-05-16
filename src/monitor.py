"""
monitor.py — System Performance Monitor
========================================
Samples 11 key performance indicators (KPIs) from the local system
using psutil. Computes rolling_mean and rolling_std over a configurable
sliding window (default: 5 seconds) to reduce per-second noise.

Total ML features per sample: 33
  - 11 raw values
  - 11 rolling means
  - 11 rolling standard deviations

Usage:
    from monitor import SystemMonitor, get_feature_names

    mon = SystemMonitor(window_size=5)
    row = mon.sample()   # returns dict of 33 features, or None during warm-up
"""

import time
import collections
import psutil
import numpy as np


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Ordered list of raw feature names — ordering must never change (model depends on it)
RAW_FEATURES = [
    "cpu_percent",          # Overall CPU utilisation (%)
    "cpu_freq_mhz",         # Current CPU frequency (MHz) — detects throttling/turbo
    "mem_percent",          # RAM usage (%)
    "mem_available_mb",     # Available RAM (MB) — absolute headroom
    "disk_read_bytes_sec",  # Disk read throughput (bytes/s) — ransomware read floods
    "disk_write_bytes_sec", # Disk write throughput (bytes/s) — ransomware write floods
    "net_sent_bytes_sec",   # Network upload (bytes/s) — data exfiltration signal
    "net_recv_bytes_sec",   # Network download (bytes/s) — flood/DDoS signal
    "open_sockets",         # Number of open TCP/UDP connections — suspicious spike
    "num_processes",        # Total running processes — process bomb signal
    "cpu_core_max",         # Max per-core CPU (%) — detects single-core saturation
]


def get_feature_names() -> list[str]:
    """
    Return the full ordered list of ML feature column names (33 total).
    Pattern: raw, raw_roll_mean, raw_roll_std — for each of the 11 raw features.
    """
    names = []
    for f in RAW_FEATURES:
        names.append(f)
        names.append(f"{f}_roll_mean")
        names.append(f"{f}_roll_std")
    return names


# ---------------------------------------------------------------------------
# SystemMonitor class
# ---------------------------------------------------------------------------

class SystemMonitor:
    """
    Continuously samples system KPIs and enriches each reading with
    rolling window statistics (mean + std) over the last `window_size` seconds.

    Args:
        window_size (int): Sliding window length in seconds. Default: 5.
        interval (float): Seconds between samples. Default: 1.0.
    """

    def __init__(self, window_size: int = 5, interval: float = 1.0):
        self.window_size = window_size
        self.interval = interval

        # Circular buffer storing the last `window_size` raw KPI dicts
        self._history: collections.deque = collections.deque(maxlen=window_size)

        # Previous I/O counter snapshots for computing per-second deltas
        self._prev_disk = None
        self._prev_net = None
        self._prev_time = None

        self._init_counters()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _init_counters(self):
        """Warm up psutil counters so first delta is meaningful."""
        psutil.cpu_percent(interval=None)   # discard first (always 0.0)
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net  = psutil.net_io_counters()
        self._prev_time = time.perf_counter()

    def _read_raw(self) -> dict:
        """Collect one snapshot of all 11 raw KPIs."""
        now     = time.perf_counter()
        elapsed = max(now - self._prev_time, 1e-6)  # avoid division by zero

        # ── CPU ─────────────────────────────────────────────────────────────
        cpu_percent  = psutil.cpu_percent(interval=None)
        freq         = psutil.cpu_freq()
        cpu_freq_mhz = freq.current if freq else 0.0
        per_core     = psutil.cpu_percent(percpu=True)
        cpu_core_max = max(per_core) if per_core else cpu_percent

        # ── Memory ──────────────────────────────────────────────────────────
        mem              = psutil.virtual_memory()
        mem_percent      = mem.percent
        mem_available_mb = mem.available / (1024 ** 2)

        # ── Disk I/O — bytes per second since last sample ────────────────────
        disk       = psutil.disk_io_counters()
        disk_read  = max(0.0, (disk.read_bytes  - self._prev_disk.read_bytes)  / elapsed)
        disk_write = max(0.0, (disk.write_bytes - self._prev_disk.write_bytes) / elapsed)
        self._prev_disk = disk

        # ── Network I/O — bytes per second since last sample ─────────────────
        net      = psutil.net_io_counters()
        net_sent = max(0.0, (net.bytes_sent - self._prev_net.bytes_sent) / elapsed)
        net_recv = max(0.0, (net.bytes_recv - self._prev_net.bytes_recv) / elapsed)
        self._prev_net = net

        # ── Connections & Processes ──────────────────────────────────────────
        try:
            open_sockets = len(psutil.net_connections(kind="all"))
        except (psutil.AccessDenied, PermissionError):
            open_sockets = 0        # fallback if admin rights not available

        num_processes = len(psutil.pids())

        self._prev_time = now

        return {
            "cpu_percent":          cpu_percent,
            "cpu_freq_mhz":         cpu_freq_mhz,
            "mem_percent":          mem_percent,
            "mem_available_mb":     mem_available_mb,
            "disk_read_bytes_sec":  disk_read,
            "disk_write_bytes_sec": disk_write,
            "net_sent_bytes_sec":   net_sent,
            "net_recv_bytes_sec":   net_recv,
            "open_sockets":         float(open_sockets),
            "num_processes":        float(num_processes),
            "cpu_core_max":         cpu_core_max,
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def sample(self) -> dict | None:
        """
        Read one KPI snapshot and return a row with all 33 features
        (raw values + rolling mean + rolling std for each feature).

        Returns:
            dict with 33 feature keys, or None if the rolling window
            is not yet fully populated (first `window_size - 1` calls).
        """
        raw = self._read_raw()
        self._history.append(raw)

        # Wait until the sliding window is full before returning stats
        if len(self._history) < self.window_size:
            return None

        row = {}
        history_list = list(self._history)

        for feat in RAW_FEATURES:
            values = [h[feat] for h in history_list]
            row[feat]                    = raw[feat]
            row[f"{feat}_roll_mean"]     = float(np.mean(values))
            row[f"{feat}_roll_std"]      = float(np.std(values))

        return row

    def sample_raw(self) -> dict:
        """
        Read one KPI snapshot with raw values only (no rolling stats).
        Updates internal counters and history buffer.
        Used during the warm-up phase before the window is full.
        """
        raw = self._read_raw()
        self._history.append(raw)
        return raw


# ---------------------------------------------------------------------------
# CLI — live feed + CSV export to data/step1_metrics.csv
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import os
    import signal
    import sys

    # Resolve output path relative to this file's location
    OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "step1_metrics.csv")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("SystemMonitor — live feed  (Ctrl+C to stop)")
    print(f"Saving all features to: {os.path.abspath(OUT_PATH)}\n")

    mon          = SystemMonitor(window_size=5)
    feature_cols = get_feature_names()
    fieldnames   = ["timestamp"] + feature_cols

    # Open CSV for writing (overwrites on each run)
    csv_file   = open(OUT_PATH, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    def _shutdown(sig, frame):
        csv_file.close()
        print(f"\n✅ Stopped. {row_count} rows saved to {os.path.abspath(OUT_PATH)}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    # Warm-up: fill the rolling window before collecting data
    print("Warming up rolling window (5 ticks)...")
    for _ in range(5):
        mon.sample_raw()
        time.sleep(1)

    print(f"\n{'Timestamp':<22} {'CPU%':>6} {'MEM%':>6} {'DISK_W MB/s':>12} {'NET_S KB/s':>12}")
    print("-" * 72)

    row_count = 0
    while True:
        row = mon.sample()
        if row:
            ts     = time.strftime("%Y-%m-%dT%H:%M:%S")
            cpu    = row["cpu_percent"]
            mem    = row["mem_percent"]
            disk_w = row["disk_write_bytes_sec"] / (1024 ** 2)
            net_s  = row["net_sent_bytes_sec"]   / 1024

            # Print summary to terminal
            print(f"{ts:<22} {cpu:>6.1f} {mem:>6.1f} {disk_w:>12.3f} {net_s:>12.3f}")

            # Write full 33-feature row to CSV
            csv_writer.writerow({"timestamp": ts, **row})
            csv_file.flush()   # ensure data is written even if stopped mid-run
            row_count += 1

        time.sleep(1)
