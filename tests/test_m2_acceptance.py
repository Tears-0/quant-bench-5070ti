"""M2 acceptance on real hardware records (requirements.md §10 M2).

"Tier A Q8_0 shows near-zero KL vs FP16 and Q4_K_M shows measurably higher — i.e. the
instrument demonstrably detects a known effect. Add this as a regression test with a
tolerance band."

The band was written down *before* the first real quality run, so it cannot have been
tuned to the result:

* Q8_0 is "near-zero":           mean KL < 0.005 nats/token
* Q4_K_M is "measurably higher": at least 3x Q8_0, and separated by > 4 combined standard
                                 errors (llama-perplexity reports the KL mean's uncertainty)
* sanity:                        Q4_K_M mean KL < 0.2 (a 4-bit K-quant of a 4B model far
                                 above this means a broken pipeline, not a bad quant), and
                                 Q8_0 top-1 agreement > 95%

The synthetic counterpart (tests/test_quality.py) runs everywhere; this one runs wherever
real records exist and is skipped in GPU-less CI.
"""

from __future__ import annotations

import math

import pytest

from quantbench.analysis.load import load_records
from quantbench.paths import results_dir


def _quality_records() -> dict[str, object]:
    records, _ = load_records(results_dir())
    found: dict[str, object] = {}
    for record in records:
        spec = record.spec
        if (
            record.ok
            and spec.workload.kind == "quality"
            and spec.model.name == "Qwen3-4B"
            and spec.quant.format == "gguf"
            and spec.quant.variant in {"Q8_0", "Q4_K_M"}
            and record.quality.kl_div.mean is not None
        ):
            found[spec.quant.variant] = record
    return found


RECORDS = _quality_records()

pytestmark = pytest.mark.skipif(
    not {"Q8_0", "Q4_K_M"} <= set(RECORDS),
    reason="no real Tier A Q8_0 + Q4_K_M quality records (GPU-less checkout)",
)


def _kl(variant: str) -> tuple[float, float]:
    record = RECORDS[variant]
    mean = float(record.quality.kl_div.mean)  # type: ignore[attr-defined]
    stderr = record.artifacts.get("kl_mean_stderr")  # type: ignore[attr-defined]
    return mean, float(stderr) if stderr not in (None, "None") else 0.0


def test_q8_0_is_near_zero():
    kl, _ = _kl("Q8_0")
    assert kl < 0.005, f"Q8_0 mean KL {kl:.5f} is not near zero"
    top1 = RECORDS["Q8_0"].quality.top1_agreement  # type: ignore[attr-defined]
    assert top1 is not None and top1 > 0.95


def test_q4_k_m_is_measurably_worse():
    q8, q8_err = _kl("Q8_0")
    q4, q4_err = _kl("Q4_K_M")
    assert q4 >= 3 * q8, f"Q4_K_M KL {q4:.5f} is not >= 3x Q8_0 KL {q8:.5f}"
    separation = (q4 - q8) / math.hypot(q4_err, q8_err) if (q4_err or q8_err) else math.inf
    assert separation > 4, f"Q4_K_M vs Q8_0 separated by only {separation:.1f} standard errors"
    assert q4 < 0.2, f"Q4_K_M KL {q4:.4f} is implausibly large: suspect the pipeline"


def test_both_scored_on_the_same_verified_stream():
    q8 = RECORDS["Q8_0"].quality  # type: ignore[attr-defined]
    q4 = RECORDS["Q4_K_M"].quality  # type: ignore[attr-defined]
    assert q8.eval_stream_sha == q4.eval_stream_sha
    assert q8.reference_id == q4.reference_id
    assert q8.token_stream_verified and q4.token_stream_verified
