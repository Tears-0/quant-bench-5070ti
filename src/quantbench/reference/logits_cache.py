"""FP16 reference logit cache (requirements.md §5.2, "critical design element").

For Tier B (8B) and Tier C (14B) the FP16 model cannot be co-resident with the quantized
model on 16 GB — in fact an 8B model in BF16 does not fit at all (§3.3). So:

1. The FP16 model is run **once** over the fixed evaluation stream, with CPU offload as
   needed. This is slow; it happens once per (model, tokenizer, eval stream).
2. The reference top-k log-probs are persisted, normalised, so the missing tail mass is
   *known* rather than assumed (see :mod:`quantbench.metrics.quality`).
3. Every subsequent KL computation reads the cache. Nothing else re-runs FP16.

The cache key is ``(model + revision, tokenizer_sha, eval_stream_sha, topk, chunking)``.
Change any of them and you get a different reference file rather than a silently
incomparable number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantbench.config import EvalConfig, ModelConfig
from quantbench.metrics.quality import LogprobArtifact
from quantbench.paths import reference_cache_dir
from quantbench.schema import sha256_of


@dataclass
class ReferenceKey:
    model: str
    revision: str | None
    tokenizer_sha: str | None
    eval_stream_sha: str
    topk: int
    ctx: int
    stride: int
    add_bos: bool

    @property
    def digest(self) -> str:
        return sha256_of(self.__dict__).removeprefix("sha256:")[:24]

    @property
    def reference_id(self) -> str:
        """Human-readable id echoed into every quality record."""
        return f"{self.model}-fp16@{self.eval_stream_sha.removeprefix('sha256:')[:12]}"


def key_for(
    model: ModelConfig, eval_cfg: EvalConfig, eval_stream_meta: dict[str, Any]
) -> ReferenceKey:
    return ReferenceKey(
        model=model.slug,
        revision=model.revision,
        tokenizer_sha=eval_stream_meta.get("tokenizer_sha"),
        eval_stream_sha=str(eval_stream_meta["eval_stream_sha"]),
        topk=eval_cfg.reference.topk,
        ctx=eval_cfg.ppl.ctx,
        stride=eval_cfg.ppl.stride,
        add_bos=eval_cfg.ppl.add_bos,
    )


def reference_path(key: ReferenceKey) -> Path:
    return reference_cache_dir() / key.model / f"reference-{key.digest}.npz"


def meta_path(key: ReferenceKey) -> Path:
    return reference_path(key).with_suffix(".json")


def has_reference(key: ReferenceKey) -> bool:
    return reference_path(key).exists()


def load_reference(key: ReferenceKey) -> LogprobArtifact:
    path = reference_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"no FP16 reference for {key.reference_id} at {path}\n"
            "build it once with: quantbench reference build --model <model>"
        )
    return LogprobArtifact.load(path)


def write_reference_meta(key: ReferenceKey, extra: dict[str, Any]) -> Path:
    payload = {"reference_id": key.reference_id, "key": key.__dict__, **extra}
    path = meta_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
