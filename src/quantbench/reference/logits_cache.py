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


def write_reference_meta_at(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_reference_meta(key: ReferenceKey, extra: dict[str, Any]) -> Path:
    payload = {"reference_id": key.reference_id, "key": key.__dict__, **extra}
    path = meta_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# llama.cpp native KL base (GGUF formats)
# --------------------------------------------------------------------------------------
#
# For GGUF, §5.2.3's preferred path is llama-perplexity's own base-file workflow: the F16
# GGUF is run once with --kl-divergence-base, which stores the full-vocabulary log-probs of
# every scored position, and each quant is then scored against it. The KL is therefore
# exact over the whole vocabulary (no top-k truncation), computed with llama.cpp's own
# kernels, and within one engine. The header also stores the exact token ids that were
# scored, which is what lets the harness *verify* — rather than assume — that llama.cpp
# and the reference tokenizer produced the same stream (§6.6).

_KLD_MAGIC = b"_logits_"


@dataclass
class KldHeader:
    n_ctx: int
    n_vocab: int
    n_chunk: int
    tokens: Any  # numpy int32 array, n_ctx * n_chunk

    @property
    def scored_positions(self) -> int:
        """llama-perplexity scores positions n_ctx/2 .. n_ctx-2 of every chunk."""
        return self.n_chunk * (self.n_ctx - 1 - self.n_ctx // 2)


def read_kld_header(path: Path) -> KldHeader:
    import numpy as np

    with path.open("rb") as fh:
        if fh.read(8) != _KLD_MAGIC:
            raise ValueError(f"{path} is not a llama-perplexity KL-divergence base file")
        n_ctx, n_vocab, n_chunk = np.frombuffer(fh.read(12), dtype="<i4").tolist()
        tokens = np.frombuffer(fh.read(4 * n_ctx * n_chunk), dtype="<i4")
    if tokens.size != n_ctx * n_chunk:
        raise ValueError(f"{path} is truncated: expected {n_ctx * n_chunk} tokens")
    return KldHeader(n_ctx=int(n_ctx), n_vocab=int(n_vocab), n_chunk=int(n_chunk), tokens=tokens)


@dataclass
class LlamaCppBaseKey:
    model: str
    base_artifact_sha256: str
    corpus_sha: str
    ctx: int
    chunks: int

    @property
    def digest(self) -> str:
        return sha256_of(self.__dict__).removeprefix("sha256:")[:24]

    @property
    def path(self) -> Path:
        return reference_cache_dir() / self.model / f"llamacpp-f16-{self.digest}.kld"

    @property
    def reference_id(self) -> str:
        return f"{self.model}-f16gguf@{self.base_artifact_sha256.removeprefix('sha256:')[:12]}"


def llamacpp_base_key(
    model: ModelConfig, eval_cfg: EvalConfig, eval_stream_meta: dict[str, Any], base_sha: str
) -> LlamaCppBaseKey:
    return LlamaCppBaseKey(
        model=model.slug,
        base_artifact_sha256=base_sha,
        corpus_sha=str(eval_stream_meta["source_sha"]),
        ctx=eval_cfg.ppl.ctx,
        chunks=eval_cfg.native_kld_chunks,
    )


def artifact_sha_from_manifest(artifact: Path) -> str | None:
    manifest = Path(str(artifact) + ".manifest.json")
    if not manifest.exists():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
    return str(value) if value else None
