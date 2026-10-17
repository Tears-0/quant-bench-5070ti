"""GGUF production: ``convert_hf_to_gguf.py`` -> ``llama-imatrix`` -> ``llama-quantize``.

The imatrix ablation is the reason this module exists rather than downloading someone
else's GGUFs: the repo builds Q4_K_M **with and without** an importance matrix from the
same F16 source with the same tool version, so the imatrix effect is isolated rather than
confounded with publisher choices (§4.2). Calibration text for the imatrix must be disjoint
from the evaluation stream (§6.7); ``build_imatrix`` refuses to reuse the eval corpus.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from quantbench.engines.llamacpp import parse_version
from quantbench.engines.server import find_binary
from quantbench.quantize.base import QuantizationError, QuantizeResult, finalize, run_command

#: Variants this repo builds. K-quants and the i-quant that needs an imatrix.
GGUF_VARIANTS = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS")
IMATRIX_REQUIRED = ("IQ4_XS",)


def _bin(name: str, env_var: str) -> str:
    if (bin_dir := os.environ.get("LLAMA_CPP_BIN_DIR")) and not os.environ.get(env_var):
        candidate = Path(bin_dir) / (name + (".exe" if os.name == "nt" else ""))
        if candidate.exists():
            return str(candidate)
    return find_binary(name, env_var)


def producer_version() -> str | None:
    """Build tag of the pinned llama.cpp release.

    ``llama-quantize`` and ``llama-imatrix`` print no version banner, but every binary in a
    release zip is built from one commit, so ``llama-server --version`` identifies them all.
    The converter script comes from the source tree of the same tag (docs/install/).
    """
    try:
        blob = run_command([_bin("llama-server", "LLAMA_SERVER_BIN"), "--version"], timeout=60)
    except (QuantizationError, OSError) as exc:
        blob = str(exc)
    version, commit = parse_version(blob)
    if version:
        return f"{version} ({commit})" if commit else version
    return os.environ.get("LLAMA_CPP_VERSION")


def convert_to_f16(model_dir: Path, out_path: Path, repo_dir: Path | None = None) -> QuantizeResult:
    """Convert an HF checkout to F16 GGUF using llama.cpp's converter script."""
    repo = repo_dir or Path(os.environ.get("LLAMA_CPP_REPO", ""))
    script = repo / "convert_hf_to_gguf.py"
    if not script.exists():
        raise QuantizationError(
            f"convert_hf_to_gguf.py not found at {script}; set LLAMA_CPP_REPO to a llama.cpp "
            "checkout (see docs/install/llamacpp.md)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sys.executable: the converter must run in the engine env that has torch + gguf-py.
    cmd = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        "f16",
    ]
    t0 = time.perf_counter()
    log = run_command(cmd)
    return finalize(
        QuantizeResult(
            artifact_path=out_path,
            producer="convert_hf_to_gguf.py",
            producer_version=producer_version(),
            command=cmd,
            duration_s=time.perf_counter() - t0,
            log_tail=log[-4000:],
        )
    )


def build_imatrix(
    f16_gguf: Path,
    calibration_file: Path,
    out_path: Path,
    eval_corpus: Path | None = None,
    n_gpu_layers: int = -1,
    chunks: int = 512,
) -> QuantizeResult:
    """Compute an importance matrix from calibration text.

    ``eval_corpus`` is passed so the function can *refuse* to calibrate on the evaluation
    corpus. Evaluating on the calibration distribution is the field's most common
    self-inflicted wound (§6.7).
    """
    if eval_corpus is not None and calibration_file.resolve() == eval_corpus.resolve():
        raise QuantizationError(
            "calibration file is the evaluation corpus; the calibration set must be "
            "disjoint from the evaluation set (§6.7)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _bin("llama-imatrix", "LLAMA_IMATRIX_BIN"),
        "-m",
        str(f16_gguf),
        "-f",
        str(calibration_file),
        "-o",
        str(out_path),
        "-ngl",
        str(n_gpu_layers),
        "--chunks",
        str(chunks),
    ]
    t0 = time.perf_counter()
    log = run_command(cmd)
    return finalize(
        QuantizeResult(
            artifact_path=out_path,
            producer="llama-imatrix",
            producer_version=producer_version(),
            command=cmd,
            duration_s=time.perf_counter() - t0,
            calibration_dataset=calibration_file.name,
            imatrix=True,
            log_tail=log[-4000:],
        )
    )


def quantize(
    f16_gguf: Path,
    variant: str,
    out_path: Path,
    imatrix_path: Path | None = None,
    calibration_dataset: str | None = None,
    threads: int | None = None,
) -> QuantizeResult:
    """Run ``llama-quantize``. Pass ``imatrix_path`` for the imatrix arm of the ablation."""
    if variant not in GGUF_VARIANTS:
        raise QuantizationError(f"unknown GGUF variant {variant!r}; known: {GGUF_VARIANTS}")
    if variant in IMATRIX_REQUIRED and imatrix_path is None:
        raise QuantizationError(f"{variant} requires an importance matrix")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_bin("llama-quantize", "LLAMA_QUANTIZE_BIN")]
    if imatrix_path is not None:
        cmd += ["--imatrix", str(imatrix_path)]
    cmd += [str(f16_gguf), str(out_path), variant]
    if threads:
        cmd.append(str(threads))

    t0 = time.perf_counter()
    log = run_command(cmd)
    return finalize(
        QuantizeResult(
            artifact_path=out_path,
            producer="llama-quantize",
            producer_version=producer_version(),
            command=cmd,
            duration_s=time.perf_counter() - t0,
            calibration_dataset=calibration_dataset,
            imatrix=imatrix_path is not None,
            imatrix_source=str(imatrix_path) if imatrix_path else None,
            log_tail=log[-4000:],
            extra={"variant": variant, "source": str(f16_gguf)},
        )
    )
