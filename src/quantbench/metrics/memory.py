"""NVML sampling and VRAM accounting (requirements.md §5.3).

Four different VRAM numbers are reported separately because they answer different
questions, and collapsing them is a common error:

``vram_weights_mb``
    Process VRAM delta measured after model load, before any KV allocation.
``vram_kv_mb``
    KV cache allocation at the tested context/batch, engine-reported where available and
    measured as a delta otherwise.
``vram_peak_allocated_mb``
    Framework allocator peak (``torch.cuda.max_memory_allocated``). Undercounts.
``vram_peak_nvsmi_mb``
    Process-level VRAM from NVML, sampled continuously. This is the number that decides
    whether a configuration *actually fits* in 16 GB, and it exceeds the allocator figure.

The sampler runs inside the worker subprocess and attributes memory to the worker's own
PID plus any engine-server PIDs registered with :meth:`NvmlSampler.track_pid`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

MB = 1024 * 1024


class NvmlUnavailable(RuntimeError):
    """NVML could not be initialised (no driver, no GPU, or no permissions)."""


def _import_pynvml() -> Any:
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - exercised only without nvidia-ml-py
        raise NvmlUnavailable(f"nvidia-ml-py is not importable: {exc}") from exc
    return pynvml


@contextmanager
def nvml_session() -> Iterator[Any]:
    """Yield an initialised ``pynvml`` module, shutting it down on exit."""
    pynvml = _import_pynvml()
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        raise NvmlUnavailable(f"nvmlInit failed: {exc}") from exc
    try:
        yield pynvml
    finally:
        with suppress(Exception):  # shutdown failures are not actionable
            pynvml.nvmlShutdown()


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class GpuSample:
    """One NVML poll. All fields may be ``None`` if that query is unsupported."""

    t_s: float
    device_used_mb: float | None
    process_used_mb: float | None
    power_w: float | None
    temp_c: float | None
    sm_clock_mhz: float | None
    throttle_bits: int | None


@dataclass
class NvmlSampler:
    """Background NVML poller.

    ``hz`` must satisfy both §5.3 (>= 5 Hz for memory) and §5.4 (>= 10 Hz for power); the
    default of 10 Hz satisfies both with a single sampling loop.
    """

    hz: float = 10.0
    device_index: int = 0
    pids: set[int] = field(default_factory=lambda: {os.getpid()})
    samples: list[GpuSample] = field(default_factory=list)
    available: bool = False
    error: str | None = None

    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _pynvml: Any = None
    _handle: Any = None
    _t0: float = 0.0

    def track_pid(self, pid: int) -> None:
        """Attribute another process's VRAM to this run (engine servers, quantizers)."""
        self.pids.add(pid)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        try:
            self._pynvml = _import_pynvml()
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            return
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nvml-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[GpuSample]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.available and self._pynvml is not None:
            with suppress(Exception):
                self._pynvml.nvmlShutdown()
        return self.samples

    @contextmanager
    def sampling(self) -> Iterator[NvmlSampler]:
        self.start()
        try:
            yield self
        finally:
            self.stop()

    # -- polling -----------------------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / max(self.hz, 0.1)
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append(self.poll())
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay < 0:  # sampling fell behind; resynchronise rather than drift
                next_tick = time.perf_counter()
                delay = 0.0
            self._stop.wait(delay)

    def poll(self) -> GpuSample:
        p, h = self._pynvml, self._handle
        return GpuSample(
            t_s=time.perf_counter() - self._t0,
            device_used_mb=_try(lambda: p.nvmlDeviceGetMemoryInfo(h).used / MB),
            process_used_mb=self._process_memory_mb(),
            power_w=_try(lambda: p.nvmlDeviceGetPowerUsage(h) / 1000.0),
            temp_c=_try(lambda: float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU))),
            sm_clock_mhz=_try(lambda: float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM))),
            throttle_bits=_try(self._throttle_bits),
        )

    def _throttle_bits(self) -> int:
        p, h = self._pynvml, self._handle
        # NVML renamed this call; support both spellings.
        for name in (
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ):
            fn = getattr(p, name, None)
            if fn is not None:
                return int(fn(h))
        raise AttributeError("no clocks-throttle-reasons entry point in this NVML build")

    def _process_memory_mb(self) -> float | None:
        p, h = self._pynvml, self._handle
        total = 0.0
        found = False
        for getter in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(p, getter, None)
            if fn is None:
                continue
            try:
                procs = fn(h)
            except Exception:
                continue
            for proc in procs:
                if proc.pid in self.pids and proc.usedGpuMemory:
                    total += float(proc.usedGpuMemory) / MB
                    found = True
            break
        return total if found else None


def _try(fn: Any) -> Any:
    try:
        return fn()
    except Exception:
        return None


# -- pure summarisation (unit-tested without a GPU) --------------------------------------


def peak_process_mb(samples: list[GpuSample]) -> float | None:
    """Peak process-attributed VRAM; falls back to device-wide when per-PID is unavailable
    (WSL2 does not implement per-process accounting)."""
    process = [s.process_used_mb for s in samples if s.process_used_mb is not None]
    if process:
        return max(process)
    device = [s.device_used_mb for s in samples if s.device_used_mb is not None]
    return max(device) if device else None


def vram_accounting_mode(samples: list[GpuSample]) -> str | None:
    """Which VRAM accounting the samples support: ``per_process`` or ``device_wide``."""
    if any(s.process_used_mb is not None for s in samples):
        return "per_process"
    if any(s.device_used_mb is not None for s in samples):
        return "device_wide"
    return None


def effective_bpw(
    weight_bytes_non_embedding: int | None, params_non_embedding: int | None
) -> float | None:
    """``(non-embedding weight bytes * 8) / non_embedding_param_count`` (§5.3).

    Both numerator and denominator exclude embeddings. Dividing the *whole* file by the
    non-embedding parameter count would leave the embedding's quantization type in the
    numerator — exactly the distortion §5.3 sets out to remove (for Qwen3-4B Q4_K_M the
    Q6_K embedding moves the figure from 4.78 to 5.50). See docs/open-questions.md Q12.

    Returns ``None`` when either input is unknown; a guessed count would silently corrupt
    every quality-vs-bpw comparison in the repo.
    """
    if not weight_bytes_non_embedding or not params_non_embedding:
        return None
    return (weight_bytes_non_embedding * 8) / params_non_embedding


def device_query(device_index: int = 0) -> dict[str, Any]:
    """One-shot device facts for preflight and the environment record."""
    with nvml_session() as p:
        h = p.nvmlDeviceGetHandleByIndex(device_index)
        mem = p.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_name": _decode(p.nvmlDeviceGetName(h)),
            "gpu_uuid": _decode(p.nvmlDeviceGetUUID(h)),
            "driver": _decode(p.nvmlSystemGetDriverVersion()),
            "vram_total_mb": int(mem.total / MB),
            "vram_used_mb": int(mem.used / MB),
            "device_count": int(p.nvmlDeviceGetCount()),
            "temp_c": _try(lambda: float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU))),
            "power_limit_w": _try(lambda: p.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0),
            "persistence_mode": _try(lambda: int(p.nvmlDeviceGetPersistenceMode(h))),
        }


def gpu_temperature_c(device_index: int = 0) -> float | None:
    """Current GPU temperature, used by the cooldown gate in the runner (§6.3)."""
    try:
        with nvml_session() as p:
            h = p.nvmlDeviceGetHandleByIndex(device_index)
            return float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU))
    except Exception:
        return None
