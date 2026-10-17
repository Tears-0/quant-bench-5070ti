"""Quality metrics: perplexity, KL divergence, top-k agreement, lm-eval bridge (§5.2).

**Primary metric is KL divergence against the FP16 reference**, not perplexity. A quant can
post *lower* PPL than its own FP16 parent while being demonstrably less faithful, and
calibration-set overlap makes PPL easy to game by accident. KL over the pre-sampling output
distribution measures "how differently does this model think".

Truncation and the tail term
----------------------------
Storing full-vocabulary distributions for a 300k-token evaluation stream is not practical,
so the reference artifact keeps the top-k (default 128) normalised log-probs per position.
KL is therefore computed over the reference's top-k support::

    KL_trunc = sum_{v in topk(ref)} p_v * (log p_v - log q_v)

The reference tail mass ``m = 1 - sum p_v`` is *measured* (the artifact stores normalised
log-probs, so the missing mass is known), but its distribution is not, so its contribution
to KL cannot be computed exactly. Every record therefore carries ``topk`` and
``tail_mass_estimated``; ``docs/methodology.md`` states the bound. When the candidate could
not be scored on the reference's id set, the computation degrades to the intersection of
supports and the record is flagged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from quantbench.schema import KLRecord

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Above this reference tail mass, truncated KL is meaningfully incomplete and the record
#: is flagged. 1% of probability mass outside the top-128 is already unusual for an LLM.
TAIL_MASS_WARN = 0.01


@dataclass
class LogprobArtifact:
    """Per-position scored output of one model over the fixed evaluation stream."""

    topk_ids: IntArray  # [N, K] vocabulary ids, descending probability
    topk_logprobs: FloatArray  # [N, K] normalised natural log-probs
    target_ids: IntArray  # [N] the actual next token in the stream
    target_logprobs: FloatArray  # [N] normalised log-prob of the target token
    meta: dict[str, Any]
    #: Optional: this model's log-probs at *another* model's top-k ids, aligned [N, K].
    ref_aligned_logprobs: FloatArray | None = None

    @property
    def n_positions(self) -> int:
        return int(self.topk_ids.shape[0])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "topk_ids": self.topk_ids.astype(np.int32),
            "topk_logprobs": self.topk_logprobs.astype(np.float32),
            "target_ids": self.target_ids.astype(np.int32),
            "target_logprobs": self.target_logprobs.astype(np.float32),
        }
        if self.ref_aligned_logprobs is not None:
            arrays["ref_aligned_logprobs"] = self.ref_aligned_logprobs.astype(np.float32)
        arrays["meta"] = np.frombuffer(json.dumps(self.meta).encode("utf-8"), dtype=np.uint8)
        # numpy's stub types **kwargs here as allow_pickle; the call is correct.
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: Path) -> LogprobArtifact:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                topk_ids=data["topk_ids"].astype(np.int64),
                topk_logprobs=data["topk_logprobs"].astype(np.float64),
                target_ids=data["target_ids"].astype(np.int64),
                target_logprobs=data["target_logprobs"].astype(np.float64),
                ref_aligned_logprobs=(
                    data["ref_aligned_logprobs"].astype(np.float64)
                    if "ref_aligned_logprobs" in data.files
                    else None
                ),
                meta=json.loads(bytes(data["meta"].tolist()).decode("utf-8")),
            )


def perplexity(target_logprobs: FloatArray) -> float:
    """``exp(mean negative log-likelihood)`` over scored positions."""
    if target_logprobs.size == 0:
        raise ValueError("cannot compute perplexity over zero positions")
    return float(np.exp(-target_logprobs.mean()))


def nll_sum(target_logprobs: FloatArray) -> float:
    return float(-target_logprobs.sum())


def tail_mass(topk_logprobs: FloatArray) -> FloatArray:
    """Reference probability mass outside the stored top-k, per position."""
    return np.clip(1.0 - np.exp(topk_logprobs).sum(axis=1), 0.0, 1.0)


def _align_candidate(ref: LogprobArtifact, cand: LogprobArtifact) -> tuple[FloatArray, bool]:
    """Candidate log-probs on the reference's top-k ids.

    Exact when the candidate was scored against the reference id set. Otherwise the
    candidate's own top-k is used as a lookup table and ids it never emitted are assigned
    its smallest observed log-prob as a floor — an upper bound on ``log q``, hence a lower
    bound on KL. The caller flags this case.
    """
    if cand.ref_aligned_logprobs is not None:
        return cand.ref_aligned_logprobs, True

    n, k = ref.topk_ids.shape
    aligned = np.empty((n, k), dtype=np.float64)
    floor = cand.topk_logprobs.min(axis=1)
    for i in range(n):
        lookup = dict(zip(cand.topk_ids[i].tolist(), cand.topk_logprobs[i].tolist(), strict=True))
        row_floor = floor[i]
        aligned[i] = [lookup.get(int(v), row_floor) for v in ref.topk_ids[i]]
    return aligned, False


def kl_per_position(ref: LogprobArtifact, cand: LogprobArtifact) -> tuple[FloatArray, bool]:
    """Truncated ``KL(P_ref || Q_cand)`` per position, plus an exactness flag."""
    if ref.n_positions != cand.n_positions:
        raise ValueError(
            f"reference has {ref.n_positions} positions, candidate has {cand.n_positions}; "
            "both must be scored over the identical evaluation stream (§6.6)"
        )
    aligned, exact = _align_candidate(ref, cand)
    p = np.exp(ref.topk_logprobs)
    kl = (p * (ref.topk_logprobs - aligned)).sum(axis=1)
    return np.maximum(kl, 0.0), exact


def kl_record(ref: LogprobArtifact, cand: LogprobArtifact) -> KLRecord:
    kl, exact = kl_per_position(ref, cand)
    mass = tail_mass(ref.topk_logprobs)
    return KLRecord(
        mean=float(kl.mean()),
        p90=float(np.percentile(kl, 90)),
        p99=float(np.percentile(kl, 99)),
        max=float(kl.max()),
        topk=int(ref.topk_ids.shape[1]),
        tail_mass_estimated=bool((not exact) or float(mass.mean()) > TAIL_MASS_WARN),
    )


def agreement(ref: LogprobArtifact, cand: LogprobArtifact, k: int = 1) -> float:
    """Fraction of positions where the reference's argmax is in the candidate's top-k.

    ``k=1`` is top-1 agreement; ``k=5`` is top-5 agreement (§5.2).
    """
    if k > cand.topk_ids.shape[1]:
        raise ValueError(f"candidate artifact stores only top-{cand.topk_ids.shape[1]}")
    ref_top1 = ref.topk_ids[:, 0]
    cand_topk = cand.topk_ids[:, :k]
    return float((cand_topk == ref_top1[:, None]).any(axis=1).mean())


def retention_pct(score: float | None, reference_score: float | None) -> float | None:
    """Downstream accuracy as a percentage of the FP16 reference (§5.2)."""
    if score is None or not reference_score:
        return None
    return 100.0 * score / reference_score


# --------------------------------------------------------------------------------------
# lm-evaluation-harness bridge
# --------------------------------------------------------------------------------------


def parse_lm_eval_output(payload: dict[str, Any]) -> dict[str, float | None]:
    """Normalise ``lm_eval --output_path`` JSON into ``{task: primary_metric}``.

    The primary metric per task is fixed here so that a harness version bump cannot
    silently change which number the repo publishes.
    """
    preferred = ("acc_norm,none", "exact_match,strict-match", "acc,none", "acc_norm", "acc")
    out: dict[str, float | None] = {}
    for task, metrics in (payload.get("results") or {}).items():
        if not isinstance(metrics, dict):
            continue
        value: float | None = None
        for key in preferred:
            if key in metrics and isinstance(metrics[key], int | float):
                value = float(metrics[key])
                break
        out[task] = value
    return out
