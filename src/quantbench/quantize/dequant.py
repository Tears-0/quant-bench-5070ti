"""One-time GGUF -> fp16 safetensors dequantization, for the Transformers-side evaluations.

llama-server serves no prompt log-probs, so lm-eval scores GGUF quants through Transformers
(open question Q15 (a)). Loading a GGUF with ``gguf_file=`` dequantizes every tensor on the
CPU, which takes minutes and was being paid once per lm-eval invocation (three per config).
This module pays it once and saves the result, with provenance, beside the GGUF.

Three choices keep the comparison clean:

* **fp16, not bf16.** bf16's 8-bit mantissa would add a rounding error per weight about as
  large as Q8_0's own quantization error; fp16 (10-bit mantissa) keeps the dequantized
  values close to what llama.cpp computes with.
* **The original HF tokenizer and config**, not the ones Transformers reconstructs from GGUF
  metadata, so the reference and every quant share one tokenizer. Architecture fields that
  differ between the GGUF-derived config and the HF config are a hard error.
* **Verified, not assumed.** ``quantbench reference``-style scoring of the dequantized
  checkpoint against the BF16 reference (top-k KL) must land near llama.cpp's own KL for
  the same quant before any lm-eval hours are spent on it.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from quantbench.quantize.base import QuantizationError, QuantizeResult, finalize

#: Config fields that must agree between the GGUF-derived and the original HF config.
ARCH_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
)


def _same(a: Any, b: Any) -> bool:
    """Equal, with floats compared at float32 precision (GGUF stores them as f32)."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return bool(a == b)


def dequantized_dir_for(gguf_path: Path, dtype: str = "float16") -> Path:
    """Where the dequantized checkpoint of ``gguf_path`` lives (a sibling directory)."""
    short = {"float16": "fp16", "bfloat16": "bf16"}.get(dtype, dtype)
    return gguf_path.with_name(f"{gguf_path.stem}-dequant-{short}")


def _arch(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ARCH_FIELDS:
        value = getattr(config, name, None)
        if name == "rope_theta" and value is None:
            # transformers 5 moved rope_theta into rope_parameters
            value = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
        out[name] = value
    return out


def dequantize_gguf(
    gguf_path: Path, hf_dir: Path, dtype: str = "float16", force: bool = False
) -> QuantizeResult:
    """Dequantize ``gguf_path`` with Transformers and save it as safetensors."""
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise QuantizationError(f"install requirements/hf.txt: {exc}") from exc

    out_dir = dequantized_dir_for(gguf_path, dtype)
    if out_dir.exists() and (out_dir / "config.json").exists() and not force:
        raise QuantizationError(f"{out_dir} already exists; pass force=True to rebuild")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        gguf_path.parent, gguf_file=gguf_path.name, dtype=torch_dtype, device_map={"": "cpu"}
    )
    reference = AutoConfig.from_pretrained(hf_dir)
    got, want = _arch(model.config), _arch(reference)
    mismatch = {
        k: (got[k], want[k])
        for k in ARCH_FIELDS
        if want[k] is not None and not _same(got[k], want[k])
    }
    if mismatch:
        raise QuantizationError(f"GGUF-derived config disagrees with the HF config: {mismatch}")

    # Transformers marks a GGUF-loaded model as quantized and refuses to save it, although
    # its weights are already plain tensors. Rebuild the architecture from the *original*
    # HF config (meta device: no extra RAM) and move the dequantized tensors in with
    # strict=True: any naming mismatch between the GGUF mapping and the HF architecture is
    # a hard error rather than a silently missing or random weight.
    with torch.device("meta"):
        fresh = AutoModelForCausalLM.from_config(reference, dtype=torch_dtype)
    state = {k: v.to(torch_dtype) for k, v in model.state_dict().items()}
    fresh.load_state_dict(state, strict=True, assign=True)
    fresh.tie_weights()
    fresh.config.dtype = dtype
    del model
    if out_dir.exists():
        shutil.rmtree(out_dir)
    fresh.save_pretrained(out_dir, safe_serialization=True)
    model = fresh
    for name in TOKENIZER_FILES:
        if (hf_dir / name).exists():
            shutil.copy2(hf_dir / name, out_dir / name)
    elapsed = time.perf_counter() - t0
    del model

    (out_dir / "quantbench_dequant.json").write_text(
        json.dumps(
            {
                "source_gguf": str(gguf_path),
                "dtype": dtype,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "config_from": str(hf_dir),
                "tokenizer_from": str(hf_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    from quantbench.reference.logits_cache import artifact_sha_from_manifest

    return finalize(
        QuantizeResult(
            artifact_path=out_dir,
            producer=f"transformers.from_pretrained(gguf_file) -> {dtype} safetensors",
            producer_version=transformers.__version__,
            command=[
                "AutoModelForCausalLM.from_pretrained",
                str(gguf_path.parent),
                f"gguf_file={gguf_path.name}",
                f"dtype={dtype}",
            ],
            duration_s=elapsed,
            extra={
                "source_gguf": str(gguf_path),
                "source_gguf_sha256": artifact_sha_from_manifest(gguf_path),
                "arch_checked": list(ARCH_FIELDS),
            },
        )
    )
