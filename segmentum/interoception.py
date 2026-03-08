from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import sys
import time

from .fep import clamp01
from .state import TickInput


@dataclass
class InteroceptionReading:
    """Prediction-error focused view of current host resource usage."""

    cpu_percent: float
    memory_mb: float
    cpu_prediction_error: float
    memory_prediction_error: float
    resource_pressure: float
    energy_drain: float
    boredom_signal: float
    surprise_signal: float

    def to_tick_input(self) -> TickInput:
        notes: list[str] = []
        if self.cpu_prediction_error > 0.05:
            notes.append(f"cpu_pe={self.cpu_prediction_error:.2f}")
        if self.memory_prediction_error > 0.05:
            notes.append(f"mem_pe={self.memory_prediction_error:.2f}")
        return TickInput(
            cpu_prediction_error=self.cpu_prediction_error,
            memory_prediction_error=self.memory_prediction_error,
            resource_pressure=self.resource_pressure,
            surprise_signal=self.surprise_signal,
            boredom_signal=self.boredom_signal,
            energy_drain=self.energy_drain,
            notes=tuple(notes),
        )


class ProcessInteroceptor:
    """
    Observe host-process load and convert it into bottom-up prediction error.

    The sensor layer keeps raw telemetry local and only forwards summarized
    perturbation terms for the upper layer to explain away.
    """

    def __init__(
        self,
        expected_cpu_percent: float = 15.0,
        expected_memory_mb: float = 200.0,
    ) -> None:
        self.expected_cpu_percent = expected_cpu_percent
        self.expected_memory_mb = expected_memory_mb
        self._last_process_time = time.process_time()
        self._last_wall_time = time.perf_counter()
        self._cpu_count = max(1, os.cpu_count() or 1)

    async def sense(self) -> TickInput:
        return self.sample().to_tick_input()

    def sample(self) -> InteroceptionReading:
        now_process = time.process_time()
        now_wall = time.perf_counter()

        wall_delta = max(1e-6, now_wall - self._last_wall_time)
        process_delta = max(0.0, now_process - self._last_process_time)
        cpu_percent = clamp01(process_delta / wall_delta / self._cpu_count) * 100.0

        self._last_process_time = now_process
        self._last_wall_time = now_wall

        memory_mb = _read_process_memory_mb()
        cpu_pe = clamp01(
            max(0.0, cpu_percent - self.expected_cpu_percent)
            / max(self.expected_cpu_percent, 1.0)
        )
        memory_pe = clamp01(
            max(0.0, memory_mb - self.expected_memory_mb)
            / max(self.expected_memory_mb, 1.0)
        )
        resource_pressure = clamp01((cpu_pe * 0.60) + (memory_pe * 0.40))
        surprise_signal = clamp01((cpu_pe * 0.45) + (memory_pe * 0.45))
        boredom_signal = 0.10 if resource_pressure < 0.05 else 0.0
        energy_drain = clamp01(0.02 + (resource_pressure * 0.18))

        return InteroceptionReading(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            cpu_prediction_error=cpu_pe,
            memory_prediction_error=memory_pe,
            resource_pressure=resource_pressure,
            energy_drain=energy_drain,
            boredom_signal=boredom_signal,
            surprise_signal=surprise_signal,
        )


def _read_process_memory_mb() -> float:
    if sys.platform == "win32":
        return _read_windows_memory_mb()
    try:
        import resource
    except ImportError:
        return 0.0

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _read_windows_memory_mb() -> float:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)

    process_handle = ctypes.windll.kernel32.GetCurrentProcess()
    result = ctypes.windll.psapi.GetProcessMemoryInfo(
        process_handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if result == 0:
        return 0.0
    return float(counters.WorkingSetSize) / (1024 * 1024)
