"""GPTQ INT4 production via ``GPTQModel``.

``AutoGPTQ`` is archived — this repo does not use it (§4.2). ``GPTQModel`` is the
maintained successor and carries the Blackwell-era GPTQ/AWQ kernel work, which matters
because the resulting checkpoint has to run on ``sm_120`` through either vLLM's
``gptq_marlin`` path or Transformers.
"""

from __future__ import annotations

import time
from pathlib import Path

from quantbench.quantize.base import QuantizationError, QuantizeResult, finalize


def _version() -> str | None:
    try:
        import gptqmodel

        return getattr(gptqmodel, "__version__", None)
    except ImportError:
        return None


def quantize_gptq(
    model_id: str,
    out_dir: Path,
    calibration_dataset: str = "allenai/c4",
    n_samples: int = 512,
    max_seq_len: int = 2048,
    bits: int = 4,
    group_size: int = 128,
    revision: str | None = None,
) -> QuantizeResult:
    """Produce a GPTQ INT4 (W4A16) checkpoint.

    The calibration set must be disjoint from the evaluation stream (§6.7); the dataset
    identifier and sample count are recorded in the manifest so a reviewer can check.
    """
    try:
        from datasets import load_dataset
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as exc:
        raise QuantizationError(f"install requirements/vllm.txt for GPTQModel: {exc}") from exc

    t0 = time.perf_counter()
    calibration = [
        row["text"]
        for row in load_dataset(calibration_dataset, split=f"train[:{n_samples}]")
        if row.get("text")
    ]
    config = QuantizeConfig(bits=bits, group_size=group_size, desc_act=True, sym=True)
    model = GPTQModel.load(model_id, config, revision=revision)
    model.quantize(calibration, batch_size=1)
    model.save(str(out_dir))

    return finalize(
        QuantizeResult(
            artifact_path=out_dir,
            producer="GPTQModel",
            producer_version=_version(),
            command=[
                "GPTQModel.quantize",
                model_id,
                f"bits={bits}",
                f"group_size={group_size}",
                f"calib={calibration_dataset}[:{n_samples}]",
            ],
            duration_s=time.perf_counter() - t0,
            calibration_dataset=f"{calibration_dataset}[:{n_samples}]@{max_seq_len}",
            extra={"bits": bits, "group_size": group_size, "desc_act": True, "sym": True},
        )
    )
