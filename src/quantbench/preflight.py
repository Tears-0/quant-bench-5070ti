"""Environment preflight (requirements.md §3, §10 M0).

Everything here is *detected and recorded*, never hardcoded. The single most important
check is the PyTorch arch list: a wheel built without ``sm_120`` fails at the first kernel
launch with ``CUDA error: no kernel image is available for execution on the device``, and
the failure is far away from its cause. This module fails loudly, with a remediation hint,
before a single benchmark starts (§3.2).

``quantbench preflight`` is expected to run — and produce a useful report — on machines
with no GPU at all. It reports what is missing; it does not pretend.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from quantbench.config import HardwareConfig
from quantbench.metrics.memory import device_query
from quantbench.paths import models_dir, repo_root
from quantbench.schema import EnvironmentRecord

Severity = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    severity: Severity
    detail: str
    hint: str | None = None

    @property
    def passed(self) -> bool:
        return self.severity != "fail"


@dataclass
class PreflightReport:
    environment: EnvironmentRecord
    checks: list[Check] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, severity: Severity, detail: str, hint: str | None = None) -> None:
        self.checks.append(Check(name, severity, detail, hint))

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "environment": self.environment.model_dump(mode="json"),
            "environment_hash": self.environment.hash(),
            "checks": [c.__dict__ for c in self.checks],
            "extra": self.extra,
        }


# --------------------------------------------------------------------------------------
# host facts
# --------------------------------------------------------------------------------------


def is_wsl() -> bool:
    """WSL2 adds measurable latency variance; §3.1 requires it to be recorded."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    version_file = Path("/proc/version")
    if version_file.exists():
        return "microsoft" in version_file.read_text(errors="ignore").lower()
    return False


def total_ram_gb() -> float | None:
    # os.sysconf exists only on POSIX; look both up dynamically so this module type-checks
    # and runs on Windows, WSL2 and native Linux alike.
    sysconf_names: dict[str, int] = getattr(os, "sysconf_names", {})
    sysconf: Any = getattr(os, "sysconf", None)
    if sysconf is not None and "SC_PHYS_PAGES" in sysconf_names:
        try:
            pages = int(sysconf("SC_PHYS_PAGES"))
            page_size = int(sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):  # pragma: no cover - platform specific
            pages = page_size = 0
        if pages and page_size:
            return pages * page_size / (1024**3)
    if platform.system() == "Windows":  # pragma: no cover - platform specific

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 - Windows-only API
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / (1024**3)
    return None


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def torch_facts() -> dict[str, Any]:
    """Torch version, CUDA runtime and arch list — without importing torch eagerly."""
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    facts: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
    }
    if facts["cuda_available"]:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            facts["device_capability"] = f"{major}.{minor}"
            facts["device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            facts["error"] = str(exc)
    return facts


def nvidia_smi_driver() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return proc.stdout.strip().splitlines()[0] if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for chunk in value.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------


def capture_environment() -> EnvironmentRecord:
    """The environment block stamped into every run record (§6.8)."""
    torch_info = torch_facts()
    env = EnvironmentRecord(
        torch=torch_info.get("version"),
        cuda_runtime=torch_info.get("cuda_runtime"),
        torch_arch_list=torch_info.get("arch_list", []),
        is_wsl=is_wsl(),
        cpu=cpu_name(),
        ram_gb=round(v, 1) if (v := total_ram_gb()) else None,
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        hostname=platform.node(),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        driver=nvidia_smi_driver(),
    )
    try:
        device = device_query()
        env.gpu_name = device["gpu_name"]
        env.gpu_uuid = device["gpu_uuid"]
        env.vram_total_mb = device["vram_total_mb"]
        env.driver = device.get("driver") or env.driver
        env.nvml_driver = device.get("driver")
    except Exception:
        pass
    return env


def run_preflight(
    hardware: HardwareConfig | None = None,
    model_paths: list[Path] | None = None,
    require_gpu: bool = True,
) -> PreflightReport:
    env = capture_environment()
    report = PreflightReport(environment=env)
    torch_info = torch_facts()
    report.extra["torch"] = torch_info

    # -- GPU presence ------------------------------------------------------------------
    if env.gpu_name is None:
        report.add(
            "gpu",
            "fail" if require_gpu else "warn",
            "no NVIDIA GPU visible through NVML",
            "benchmarks need the target card; planning and plotting do not (--dry-run)",
        )
    else:
        report.add("gpu", "ok", f"{env.gpu_name} ({env.vram_total_mb} MB)")

    if hardware is not None and env.gpu_name:
        if hardware.gpu_name_match.lower() not in env.gpu_name.lower():
            report.add(
                "gpu.model",
                "warn",
                f"expected {hardware.gpu_name_match!r}, found {env.gpu_name!r}",
                "results are scoped to the configured card; update configs/hardware/",
            )
        else:
            report.add("gpu.model", "ok", env.gpu_name)

    # -- driver / CUDA -----------------------------------------------------------------
    if hardware is not None:
        if _version_tuple(env.driver) and _version_tuple(env.driver) < _version_tuple(
            hardware.min_driver
        ):
            report.add(
                "driver",
                "fail",
                f"driver {env.driver} < required {hardware.min_driver}",
                "update the NVIDIA driver",
            )
        elif env.driver:
            report.add("driver", "ok", str(env.driver))
        else:
            report.add("driver", "warn", "driver version could not be determined")

    # -- the sm_120 assertion (§3.2) ---------------------------------------------------
    required_arch = hardware.required_torch_arch if hardware else "sm_120"
    if not torch_info.get("available"):
        report.add(
            "torch",
            "warn",
            f"torch is not installed ({torch_info.get('error')})",
            "only needed by the Transformers, vLLM and ExLlamaV3 engines; install one of "
            "requirements/hf.txt, requirements/vllm.txt or requirements/exl3.txt",
        )
    elif not torch_info.get("cuda_available"):
        report.add(
            "torch.cuda", "fail" if require_gpu else "warn", "torch.cuda.is_available() is False"
        )
    else:
        arch_list = torch_info.get("arch_list", [])
        if required_arch not in arch_list:
            report.add(
                "torch.arch_list",
                "fail",
                f"{required_arch} missing from torch arch list {arch_list}",
                "install a cu128+ wheel built with sm_120 "
                "(pip install --index-url https://download.pytorch.org/whl/cu128 torch); "
                "without it every kernel launch fails with 'no kernel image is available'",
            )
        else:
            report.add("torch.arch_list", "ok", f"{required_arch} present: {arch_list}")
        if hardware and (cap := torch_info.get("device_capability")):
            severity: Severity = "ok" if cap == hardware.compute_capability else "warn"
            report.add("gpu.compute_capability", severity, f"sm {cap}")
        if _version_tuple(torch_info.get("cuda_runtime")) < _version_tuple(
            hardware.min_cuda_runtime if hardware else "12.8"
        ):
            report.add(
                "cuda.runtime",
                "fail",
                f"torch CUDA runtime {torch_info.get('cuda_runtime')} is below the "
                f"{hardware.min_cuda_runtime if hardware else '12.8'} required for sm_120",
            )
        else:
            report.add("cuda.runtime", "ok", str(torch_info.get("cuda_runtime")))

    # -- WSL flag (§3.1) ---------------------------------------------------------------
    if env.is_wsl:
        report.add(
            "wsl",
            "warn",
            "running under WSL2",
            "batch-1 latency is noisier than native Linux; this is recorded in every "
            "record and must be stated in the README",
        )
    else:
        report.add("wsl", "ok", "native")

    # -- disk (§12) --------------------------------------------------------------------
    target = models_dir() if models_dir().exists() else repo_root()
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    min_gb = hardware.min_free_disk_gb if hardware else 200
    report.add(
        "disk",
        "ok" if free_gb >= min_gb else "warn",
        f"{free_gb:.0f} GB free at {target} (want >= {min_gb} GB for the model cache)",
        "quantized artifacts run to hundreds of GB; see 'quantbench models prune'",
    )

    # -- engine binaries ---------------------------------------------------------------
    for binary in ("llama-server", "llama-perplexity", "llama-quantize", "llama-imatrix"):
        exe = binary + (".exe" if os.name == "nt" else "")
        found = shutil.which(binary) or (
            str(p)
            if (bd := os.environ.get("LLAMA_CPP_BIN_DIR")) and (p := Path(bd) / exe).exists()
            else None
        )
        report.add(
            f"binary.{binary}",
            "ok" if found else "warn",
            found or "not found",
            None if found else "see docs/install/llamacpp.md; note the renamed binaries",
        )

    # -- model artifacts ---------------------------------------------------------------
    for path in model_paths or []:
        report.add(
            f"artifact.{path.name}",
            "ok" if path.exists() else "warn",
            str(path) if path.exists() else f"missing: {path}",
        )
    return report
