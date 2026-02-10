"""Performance monitor for system telemetry during eval runs."""

from __future__ import annotations

import time
from typing import Any

import psutil


class PerfMonitor:
    """Captures CPU, memory, and optionally GPU/VRAM utilization."""

    def __init__(self):
        self._snapshots: list[dict[str, Any]] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Start monitoring."""
        self._start_time = time.perf_counter()
        self._snapshots = []

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of current system metrics."""
        elapsed = (
            time.perf_counter() - self._start_time
            if self._start_time
            else 0.0
        )
        snap = {
            "elapsed_seconds": elapsed,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }

        gpu = self._snapshot_gpu()
        if gpu:
            snap.update(gpu)

        self._snapshots.append(snap)
        return snap

    def _snapshot_gpu(self) -> dict[str, Any] | None:
        """Get GPU metrics via pynvml. Returns None if unavailable."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {
                "gpu_utilization_percent": util.gpu,
                "vram_used_gb": mem.used / (1024**3),
                "vram_total_gb": mem.total / (1024**3),
            }
        except Exception:
            return None

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated metrics from all snapshots."""
        if not self._snapshots:
            return {}

        cpu_values = [s["cpu_percent"] for s in self._snapshots]
        mem_values = [s["memory_percent"] for s in self._snapshots]

        summary: dict[str, Any] = {
            "snapshot_count": len(self._snapshots),
            "cpu_avg_percent": sum(cpu_values) / len(cpu_values),
            "cpu_max_percent": max(cpu_values),
            "memory_avg_percent": sum(mem_values) / len(mem_values),
            "memory_max_percent": max(mem_values),
        }

        gpu_values = [
            s["gpu_utilization_percent"]
            for s in self._snapshots
            if "gpu_utilization_percent" in s
        ]
        if gpu_values:
            summary["gpu_avg_percent"] = sum(gpu_values) / len(gpu_values)
            summary["gpu_max_percent"] = max(gpu_values)

        vram_values = [
            s["vram_used_gb"]
            for s in self._snapshots
            if "vram_used_gb" in s
        ]
        if vram_values:
            summary["vram_max_gb"] = max(vram_values)

        return summary

    def stop(self) -> dict[str, Any]:
        """Stop monitoring and return summary."""
        summary = self.get_summary()
        self._start_time = None
        return summary
