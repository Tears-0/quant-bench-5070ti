"""EXL3 production via ExLlamaV3's ``convert.py``.

EXL3's continuous bits-per-weight knob is what makes a smooth quality-vs-bpw curve
possible, so this is the format that most directly tests "how much quality does a bit buy"
(§4.2, figure 2). Conversions at {3.0, 4.0, 5.0, 6.0} bpw come from the *same* source
checkpoint with the same calibration settings, so bpw is the only variable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from quantbench.quantize.base import QuantizationError, QuantizeResult, finalize, run_command


def _convert_script() -> Path:
    repo = Path(os.environ.get("EXLLAMAV3_REPO", ""))
    script = repo / "convert.py"
    if not script.exists():
        raise QuantizationError(
            f"convert.py not found at {script}; set EXLLAMAV3_REPO to an exllamav3 checkout "
            "(see docs/install/exllamav3.md)"
        )
    return script


def _version() -> str | None:
    try:
        import exllamav3

        return getattr(exllamav3, "__version__", None)
    except ImportError:
        return os.environ.get("EXLLAMAV3_VERSION")


def convert(
    model_dir: Path,
    out_dir: Path,
    bpw: float,
    head_bpw: float | None = None,
    work_dir: Path | None = None,
    calibration_dataset: str | None = None,
) -> QuantizeResult:
    """Convert an HF checkpoint to EXL3 at the requested bits-per-weight."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_convert_script()),
        "-i",
        str(model_dir),
        "-o",
        str(out_dir),
        "-b",
        str(bpw),
    ]
    if head_bpw is not None:
        cmd += ["-hb", str(head_bpw)]
    if work_dir is not None:
        cmd += ["-w", str(work_dir)]

    t0 = time.perf_counter()
    log = run_command(cmd)
    return finalize(
        QuantizeResult(
            artifact_path=out_dir,
            producer="exllamav3/convert.py",
            producer_version=_version(),
            command=cmd,
            duration_s=time.perf_counter() - t0,
            calibration_dataset=calibration_dataset or "exllamav3 built-in calibration",
            log_tail=log[-4000:],
            extra={"bpw": bpw, "head_bpw": head_bpw},
        )
    )
