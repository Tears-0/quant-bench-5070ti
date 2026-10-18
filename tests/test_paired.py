"""Paired per-item accuracy comparison (analysis/paired.py)."""

from __future__ import annotations

import json
import math

import pytest

from quantbench.analysis.paired import compare_runs, load_samples, paired


def write_samples(directory, task, outcomes, metric="acc", filt="none", prompt="p"):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "samples_1.jsonl").open("w", encoding="utf-8") as fh:
        for i, v in enumerate(outcomes):
            row = {"doc_id": i, "filter": filt, metric: v, "prompt_hash": f"{prompt}{i}"}
            fh.write(json.dumps(row) + "\n")
            if task == "gsm8k":  # the second filter must be ignored
                other = {"doc_id": i, "filter": "flexible-extract", metric: 1.0, "prompt_hash": "x"}
                fh.write(json.dumps(other) + "\n")


def test_paired_statistics_by_hand():
    ref = {0: (1.0, "a"), 1: (1.0, "b"), 2: (0.0, "c"), 3: (0.0, "d")}
    cand = {0: (1.0, "a"), 1: (0.0, "b"), 2: (0.0, "c"), 3: (1.0, "d")}
    n, r, c, d, se, ref_only, cand_only, same = paired(ref, cand)
    assert (n, r, c, d) == (4, 0.5, 0.5, 0.0)
    # diffs = [0, -1, 0, +1]: var (ddof=1) = 2/3, se = sqrt(2/3 / 4)
    assert se == pytest.approx(math.sqrt((2 / 3) / 4))
    assert (ref_only, cand_only, same) == (1, 1, True)


def test_identical_outcomes_tie_and_gsm8k_uses_strict_match(tmp_path):
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "cand"
    outcomes = [1.0, 0.0, 1.0, 1.0]
    write_samples(ref_dir / "gsm8k", "gsm8k", outcomes, metric="exact_match", filt="strict-match")
    write_samples(cand_dir / "gsm8k", "gsm8k", outcomes, metric="exact_match", filt="strict-match")
    assert len(load_samples(ref_dir, "gsm8k")) == 4  # flexible-extract rows dropped
    [res] = compare_runs("m", ref_dir, {"q": cand_dir}, ["gsm8k"])
    assert res.delta == 0.0 and res.tied and res.prompts_identical
    assert res.retention_pct == pytest.approx(100.0)


def test_systematic_loss_is_not_tied_and_mmlu_subset_aggregates(tmp_path):
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "cand"
    for subject in ("mmlu_a", "mmlu_b"):
        write_samples(ref_dir / subject, subject, [1.0] * 200)
        write_samples(cand_dir / subject, subject, [1.0] * 150 + [0.0] * 50)
    results = compare_runs("m", ref_dir, {"q4": cand_dir}, ["mmlu_a", "mmlu_b"])
    subset = next(r for r in results if r.task == "mmlu_subset")
    assert subset.delta == pytest.approx(-0.25)
    assert not subset.tied
    assert subset.n == 400


def test_prompt_mismatch_is_reported(tmp_path):
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "cand"
    write_samples(ref_dir / "arc_challenge", "arc", [1.0, 0.0], metric="acc_norm", prompt="p")
    write_samples(
        cand_dir / "arc_challenge", "arc", [1.0, 0.0], metric="acc_norm", prompt="DIFFERENT"
    )
    [res] = compare_runs("m", ref_dir, {"q": cand_dir}, ["arc_challenge"])
    assert res.prompts_identical is False
