"""Quality instrument tests, including the M2 regression band (§5.2, §10 M2).

The M2 acceptance criterion is that the instrument *demonstrably detects a known effect*:
a near-lossless quant must show near-zero KL against the reference, and a 4-bit quant must
show measurably more. These tests assert that on synthetic distributions, so the check runs
in CI; the same assertion is applied to real records when they exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_artifact
from quantbench.metrics import quality
from quantbench.metrics.quality import LogprobArtifact


def _artifact_from_logits(logits: np.ndarray, k: int, reference: LogprobArtifact | None = None):
    logprobs = (
        logits
        - np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        - logits.max(axis=1, keepdims=True)
    )
    order = np.argsort(-logprobs, axis=1)[:, :k]
    targets = np.zeros(logits.shape[0], dtype=np.int64)
    return LogprobArtifact(
        topk_ids=order.astype(np.int64),
        topk_logprobs=np.take_along_axis(logprobs, order, axis=1),
        target_ids=targets,
        target_logprobs=logprobs[np.arange(logits.shape[0]), targets],
        ref_aligned_logprobs=(
            np.take_along_axis(logprobs, reference.topk_ids, axis=1)
            if reference is not None
            else None
        ),
        meta={"synthetic": True},
    )


def test_identical_distributions_have_zero_kl():
    ref = make_artifact()
    cand = LogprobArtifact(
        topk_ids=ref.topk_ids.copy(),
        topk_logprobs=ref.topk_logprobs.copy(),
        target_ids=ref.target_ids.copy(),
        target_logprobs=ref.target_logprobs.copy(),
        ref_aligned_logprobs=ref.topk_logprobs.copy(),
        meta={},
    )
    record = quality.kl_record(ref, cand)
    assert record.mean == pytest.approx(0.0, abs=1e-12)
    assert quality.agreement(ref, cand, k=1) == 1.0


def test_kl_matches_a_hand_computed_value():
    # Two positions, three-symbol vocabulary, exact truncated KL.
    p = np.array([[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]])
    q = np.array([[0.4, 0.4, 0.2], [0.6, 0.3, 0.1]])
    ref = LogprobArtifact(
        topk_ids=np.array([[0, 1, 2], [0, 1, 2]]),
        topk_logprobs=np.log(p),
        target_ids=np.array([0, 0]),
        target_logprobs=np.log(p[:, 0]),
        meta={},
    )
    cand = LogprobArtifact(
        topk_ids=np.array([[0, 1, 2], [0, 1, 2]]),
        topk_logprobs=np.log(q),
        target_ids=np.array([0, 0]),
        target_logprobs=np.log(q[:, 0]),
        ref_aligned_logprobs=np.log(q),
        meta={},
    )
    expected = (p * (np.log(p) - np.log(q))).sum(axis=1)
    kl, exact = quality.kl_per_position(ref, cand)
    assert exact is True
    assert kl == pytest.approx(expected)
    assert quality.kl_record(ref, cand).mean == pytest.approx(expected.mean())


def test_m2_regression_instrument_detects_a_known_effect():
    """Near-lossless perturbation -> near-zero KL; 4-bit-scale perturbation -> much more.

    Tolerance band, not exact values: this guards the instrument, not the hardware.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 2.0, size=(256, 512))
    reference = _artifact_from_logits(base, k=32)

    q8_like = _artifact_from_logits(
        base + rng.normal(0.0, 0.01, size=base.shape), k=32, reference=reference
    )
    q4_like = _artifact_from_logits(
        base + rng.normal(0.0, 0.30, size=base.shape), k=32, reference=reference
    )

    kl_q8 = quality.kl_record(reference, q8_like).mean
    kl_q4 = quality.kl_record(reference, q4_like).mean
    assert kl_q8 is not None and kl_q4 is not None

    assert kl_q8 < 1e-3, "a near-lossless quant must land near zero KL"
    assert kl_q4 > 10 * kl_q8, "a 4-bit-scale perturbation must be measurably worse"
    assert quality.agreement(reference, q8_like, k=1) >= quality.agreement(reference, q4_like, k=1)


def test_missing_alignment_is_flagged_as_approximate():
    ref = make_artifact(seed=1)
    cand = make_artifact(seed=2)  # no ref_aligned_logprobs
    record = quality.kl_record(ref, cand)
    assert record.tail_mass_estimated is True
    assert record.topk == ref.topk_ids.shape[1]


def test_mismatched_streams_are_refused():
    ref = make_artifact(n=32)
    cand = make_artifact(n=16)
    with pytest.raises(ValueError, match="identical evaluation stream"):
        quality.kl_per_position(ref, cand)


def test_tail_mass_is_measured_not_assumed():
    ref = make_artifact(n=8, k=4, vocab=50)
    mass = quality.tail_mass(ref.topk_logprobs)
    assert (mass >= 0).all() and (mass <= 1).all()
    assert mass.mean() > 0, "top-4 of 50 symbols must leave measurable tail mass"


def test_perplexity_and_retention():
    logprobs = np.full(100, -np.log(7.0))
    assert quality.perplexity(logprobs) == pytest.approx(7.0)
    assert quality.nll_sum(logprobs) == pytest.approx(100 * np.log(7.0))
    assert quality.retention_pct(0.45, 0.5) == pytest.approx(90.0)
    assert quality.retention_pct(None, 0.5) is None
    assert quality.retention_pct(0.45, None) is None


def test_artifact_roundtrip(tmp_path):
    artifact = make_artifact(n=16, k=4)
    path = artifact.save(tmp_path / "art.npz")
    restored = LogprobArtifact.load(path)
    assert restored.n_positions == 16
    assert np.allclose(restored.topk_logprobs, artifact.topk_logprobs, atol=1e-6)
    assert restored.meta["synthetic"] is True


def test_lm_eval_primary_metric_is_pinned():
    payload = {
        "results": {
            "arc_challenge": {"acc,none": 0.4, "acc_norm,none": 0.44},
            "gsm8k": {"exact_match,strict-match": 0.31},
            "unscored": {"note": "n/a"},
        }
    }
    payload["results"]["arc_challenge"]["acc_norm_stderr,none"] = 0.014
    parsed = quality.parse_lm_eval_output(payload)
    assert parsed["arc_challenge"] == 0.44  # acc_norm preferred over acc
    assert parsed["arc_challenge_stderr"] == 0.014
    assert parsed["gsm8k"] == 0.31
    assert parsed["gsm8k_stderr"] is None
    assert parsed["unscored"] is None
    assert "unscored_stderr" not in parsed
