"""Per-process GPU memory on Windows (WDDM), via the "GPU Process Memory" PDH counters.

Two gaps in NVML on Windows make this necessary:

* **No per-process VRAM.** Under WDDM, NVML reports no ``usedGpuMemory`` per process, so the
  harness could only see device-wide usage, desktop and browser included.
* **Silent sysmem fallback.** Since driver 536.40, CUDA on Windows may satisfy allocations
  beyond dedicated VRAM by spilling into system RAM over PCIe ("CUDA - Sysmem Fallback
  Policy" in the NVIDIA Control Panel; default: allow). A configuration that does not fit
  then runs 10-50x slower instead of failing with OOM, which would silently corrupt every
  fit/no-fit result (§5.3: OOM is a result) and every timing taken while spilled.

The ``\\GPU Process Memory(pid_<PID>_*)\\Dedicated Usage`` and ``...\\Shared Usage`` counters
report both, per process. They are read in-process through ``pdh.dll`` with ctypes: no new
dependency, and a collection costs well under a millisecond, so they ride along with the
NVML sampler at the same rate.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass

MB = 1024 * 1024

#: Shared usage above this *and* dedicated memory under pressure is treated as spill.
#: "Shared Usage" also counts pinned host buffers that engines allocate on purpose
#: (llama-perplexity keeps its full-vocabulary logits in a ~1.2 GB pinned buffer), so
#: shared memory alone is not evidence of spill. WDDM only demotes device allocations to
#: system RAM when dedicated memory is nearly exhausted; the probe on this card saw spill
#: begin with the device ~0.7 GiB from full. See docs/failures.md.
SPILL_THRESHOLD_MB = 256.0
#: "Under pressure" = device-wide usage within this much of the card's total.
PRESSURE_MARGIN_MB = 1024.0

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
_ERROR_SUCCESS = 0


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _FmtItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtValue)]


@dataclass(frozen=True)
class ProcessGpuMemory:
    dedicated_mb: float
    shared_mb: float


def available() -> bool:
    return sys.platform == "win32"


class WddmCounters:
    """A PDH query over the per-process GPU memory counters (all instances)."""

    def __init__(self) -> None:
        if not available():
            raise OSError("WDDM counters exist only on Windows")
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._query = wintypes.HANDLE()
        self._dedicated = wintypes.HANDLE()
        self._shared = wintypes.HANDLE()
        self._check(self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)))
        add = self._pdh.PdhAddEnglishCounterW
        self._check(
            add(
                self._query,
                r"\GPU Process Memory(*)\Dedicated Usage",
                None,
                ctypes.byref(self._dedicated),
            )
        )
        self._check(
            add(
                self._query,
                r"\GPU Process Memory(*)\Shared Usage",
                None,
                ctypes.byref(self._shared),
            )
        )
        self._check(self._pdh.PdhCollectQueryData(self._query))

    @staticmethod
    def _check(status: int) -> None:
        if status != _ERROR_SUCCESS:
            raise OSError(f"PDH call failed with status {status & 0xFFFFFFFF:#x}")

    def _read(self, counter: wintypes.HANDLE) -> dict[int, float]:
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        fn = self._pdh.PdhGetFormattedCounterArrayW
        status = fn(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if status & 0xFFFFFFFF != _PDH_MORE_DATA:
            return {}
        buffer = (ctypes.c_byte * size.value)()
        self._check(fn(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer))
        items = ctypes.cast(buffer, ctypes.POINTER(_FmtItem))
        out: dict[int, float] = {}
        for i in range(count.value):
            name = items[i].szName or ""
            if not name.startswith("pid_"):
                continue
            try:
                pid = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            # one instance per (pid, adapter, segment kind); sum them per pid
            out[pid] = out.get(pid, 0.0) + items[i].FmtValue.largeValue / MB
        return out

    def sample(self, pids: set[int]) -> ProcessGpuMemory | None:
        """Summed dedicated/shared usage of ``pids``; ``None`` if none of them is visible."""
        if self._pdh.PdhCollectQueryData(self._query) != _ERROR_SUCCESS:
            return None
        dedicated, shared = self._read(self._dedicated), self._read(self._shared)
        seen = pids & (dedicated.keys() | shared.keys())
        if not seen:
            return None
        return ProcessGpuMemory(
            dedicated_mb=sum(dedicated.get(p, 0.0) for p in seen),
            shared_mb=sum(shared.get(p, 0.0) for p in seen),
        )

    def close(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = wintypes.HANDLE()


def open_counters() -> WddmCounters | None:
    """Best effort: ``None`` off Windows or if the counters are unavailable."""
    if not available() or os.environ.get("QUANTBENCH_DISABLE_WDDM_COUNTERS"):
        return None
    try:
        return WddmCounters()
    except OSError:
        return None
