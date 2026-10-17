"""AWQ (W4A16) and FP8 (W8A8) production via ``llm-compressor``.

``AutoAWQ`` is deprecated and archived — this repo does not use it (§4.2). Its
functionality moved into ``llm-compressor``, which is also the FP8 path.

FP8 on consumer Blackwell is *conditional*: GB203 has FP8 tensor cores but the vLLM
``cutlass_scaled_mm`` path for block-scaled FP8 has had ``sm_120`` gaps. Producing the
artifact is not evidence that it runs; that is verified empirically by the vLLM engine, and
a failure is recorded as a finding (§3.2, §12).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from quantbench.quantize.base import QuantizationError, QuantizeResult, finalize


def _version() -> str | None:
    try:
        import llmcompressor

        return getattr(llmcompressor, "__version__", None)
    except ImportError:
        return None


def _load_calibration(dataset: str, n_samples: int, max_seq_len: int, tokenizer: Any) -> Any:
    """Load calibration text. Must be disjoint from the evaluation corpus (§6.7)."""
    from datasets import load_dataset

    data = load_dataset(dataset, split=f"train[:{n_samples}]")
    return data.map(
        lambda row: tokenizer(row["text"], truncation=True, max_length=max_seq_len, padding=False),
        remove_columns=data.column_names,
    )


def quantize_awq(
    model_id: str,
    out_dir: Path,
    calibration_dataset: str = "allenai/c4",
    n_samples: int = 512,
    max_seq_len: int = 2048,
    revision: str | None = None,
) -> QuantizeResult:
    """AWQ INT4 (W4A16), served by vLLM through the ``awq_marlin`` kernel path."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise QuantizationError(f"install requirements/vllm.txt for llm-compressor: {exc}") from exc

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto")
    recipe = AWQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head"])
    oneshot(
        model=model,
        dataset=_load_calibration(calibration_dataset, n_samples, max_seq_len, tokenizer),
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=n_samples,
        output_dir=str(out_dir),
    )
    tokenizer.save_pretrained(out_dir)
    return finalize(
        QuantizeResult(
            artifact_path=out_dir,
            producer="llm-compressor(AWQModifier)",
            producer_version=_version(),
            command=[
                "llmcompressor.oneshot",
                model_id,
                "W4A16",
                f"calib={calibration_dataset}",
                f"n={n_samples}",
                f"seq={max_seq_len}",
            ],
            duration_s=time.perf_counter() - t0,
            calibration_dataset=f"{calibration_dataset}[:{n_samples}]@{max_seq_len}",
            extra={"scheme": "W4A16", "ignored": ["lm_head"]},
        )
    )


def quantize_fp8(
    model_id: str,
    out_dir: Path,
    dynamic: bool = True,
    revision: str | None = None,
) -> QuantizeResult:
    """FP8 W8A8. Dynamic activation scales need no calibration data."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise QuantizationError(f"install requirements/vllm.txt for llm-compressor: {exc}") from exc

    t0 = time.perf_counter()
    scheme = "FP8_DYNAMIC" if dynamic else "FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto")
    oneshot(
        model=model,
        recipe=QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=["lm_head"]),
        output_dir=str(out_dir),
    )
    tokenizer.save_pretrained(out_dir)
    return finalize(
        QuantizeResult(
            artifact_path=out_dir,
            producer="llm-compressor(QuantizationModifier)",
            producer_version=_version(),
            command=["llmcompressor.oneshot", model_id, scheme],
            duration_s=time.perf_counter() - t0,
            calibration_dataset=None if dynamic else "activation-calibrated",
            extra={
                "scheme": scheme,
                "note": "kernel support on sm_120 is verified at run time, not here",
            },
        )
    )
