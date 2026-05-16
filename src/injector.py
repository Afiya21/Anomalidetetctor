"""
injector.py — Anomaly Injectors
=================================
Simulates three distinct types of system anomalies to generate
labeled training data for the anomaly detector.

Each injector is a context manager for safe resource cleanup:

    with CPUAnomalyInjector(intensity=1.0) as inj:
        # anomaly is active here
        time.sleep(60)
    # resources cleaned up automatically

Classes:
    CPUAnomalyInjector    — Saturates CPU cores via busy-loop threads   (label 1)
    MemoryAnomalyInjector — Allocates RAM incrementally up to target MB (label 2)
    DiskAnomalyInjector   — Floods disk I/O with repeated write/read    (label 3)
"""

import os
import tempfile
import threading
import time
import multiprocessing


# ---------------------------------------------------------------------------
# Label 1 — CPU Anomaly
# ---------------------------------------------------------------------------

class CPUAnomalyInjector:
    """
    Simulates a CPU spike by spawning N busy-loop threads.

    N = ceil(cpu_count * intensity), so intensity=1.0 saturates all cores.
    Each thread runs a tight computation loop until stop() is called.

    Args:
        intensity (float): Fraction of CPU cores to saturate (0.0 – 1.0).
                           Default: 1.0 (all cores).
    """

    def __init__(self, intensity: float = 1.0):
        self.intensity    = intensity
        self._stop_event  = threading.Event()
        self._threads: list[threading.Thread] = []

    def _busy_loop(self):
        """Compute-intensive loop — runs until stop_event is set."""
        while not self._stop_event.is_set():
            # Pure Python arithmetic to keep CPU busy without I/O
            _ = sum(i * i for i in range(50_000))

    def start(self):
        """Spawn busy-loop threads to stress the CPU."""
        n_threads = max(1, int(multiprocessing.cpu_count() * self.intensity))
        self._stop_event.clear()
        for _ in range(n_threads):
            t = threading.Thread(target=self._busy_loop, daemon=True)
            t.start()
            self._threads.append(t)
        print(f"  [CPUAnomalyInjector] Started {n_threads} busy-loop threads.")

    def stop(self):
        """Signal all threads to stop and wait for them to finish."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        print("  [CPUAnomalyInjector] Stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Label 2 — Memory Anomaly
# ---------------------------------------------------------------------------

class MemoryAnomalyInjector:
    """
    Simulates a memory exhaustion event by allocating large bytearrays
    incrementally and holding them in memory until stopped.

    Allocation happens in chunks to ramp up gradually, mimicking a
    memory leak or a runaway process.

    Args:
        target_mb (int): Maximum RAM to allocate in MB. Default: 512.
        chunk_mb  (int): MB allocated per step.          Default: 50.
    """

    def __init__(self, target_mb: int = 512, chunk_mb: int = 50):
        self.target_mb    = target_mb
        self.chunk_mb     = chunk_mb
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None
        self._allocations: list[bytearray]    = []

    def _allocate(self):
        """Incrementally allocate memory and hold until stop_event is set."""
        allocated_mb = 0
        while not self._stop_event.is_set() and allocated_mb < self.target_mb:
            try:
                self._allocations.append(bytearray(self.chunk_mb * 1024 * 1024))
                allocated_mb += self.chunk_mb
                print(f"  [MemoryAnomalyInjector] Allocated {allocated_mb} MB")
            except MemoryError:
                print("  [MemoryAnomalyInjector] MemoryError — stopping allocation.")
                break
            time.sleep(0.5)   # ramp up gradually, not all at once

        # Hold the allocated memory in place until explicitly stopped
        self._stop_event.wait()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._allocate, daemon=True)
        self._thread.start()
        print(f"  [MemoryAnomalyInjector] Ramping to {self.target_mb} MB...")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._allocations.clear()   # release all memory
        print("  [MemoryAnomalyInjector] Stopped & memory released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Label 3 — Disk Anomaly
# ---------------------------------------------------------------------------

class DiskAnomalyInjector:
    """
    Simulates a disk I/O anomaly by repeatedly writing and reading
    a large temporary file in a tight loop.

    Mimics ransomware-style access patterns (bulk sequential writes/reads),
    generating sustained high disk_read_bytes_sec and disk_write_bytes_sec.

    Args:
        file_size_mb (int): Size of the temp file in MB. Default: 100.
    """

    def __init__(self, file_size_mb: int = 100):
        self.file_size_mb = file_size_mb
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None
        self._tmp_path: str | None            = None

    def _io_loop(self):
        """Write then read a large random file in a continuous loop."""
        # Generate random data once — avoids CPU overhead in the loop
        data = os.urandom(self.file_size_mb * 1024 * 1024)

        # Create a temp file that persists across loop iterations
        fd, self._tmp_path = tempfile.mkstemp(prefix="dcml_disk_anomaly_")
        os.close(fd)

        cycle = 0
        try:
            while not self._stop_event.is_set():
                # Write pass
                with open(self._tmp_path, "wb") as f:
                    f.write(data)
                # Read pass
                with open(self._tmp_path, "rb") as f:
                    _ = f.read()
                cycle += 1
        except Exception as e:
            print(f"  [DiskAnomalyInjector] I/O error after {cycle} cycles: {e}")

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._io_loop, daemon=True)
        self._thread.start()
        print(f"  [DiskAnomalyInjector] Flooding disk I/O ({self.file_size_mb} MB file)...")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=20)   # disk I/O can take time to flush

        # Clean up the temp file
        if self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass

        print("  [DiskAnomalyInjector] Stopped & temp file removed.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# CLI — quick test of all three injectors in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    TEST_DURATION = 10   # seconds per injector

    print("=" * 60)
    print("Injector Self-Test — each anomaly runs for 10 seconds")
    print("=" * 60)

    # Test 1: CPU
    print("\n[1/3] CPU Anomaly")
    with CPUAnomalyInjector(intensity=1.0):
        time.sleep(TEST_DURATION)
    print("  CPU test complete.\n")

    # Test 2: Memory
    print("[2/3] Memory Anomaly")
    with MemoryAnomalyInjector(target_mb=256, chunk_mb=50):
        time.sleep(TEST_DURATION)
    print("  Memory test complete.\n")

    # Test 3: Disk
    print("[3/3] Disk Anomaly")
    with DiskAnomalyInjector(file_size_mb=50):
        time.sleep(TEST_DURATION)
    print("  Disk test complete.\n")

    print("=" * 60)
    print("All injectors tested successfully.")
    print("=" * 60)
