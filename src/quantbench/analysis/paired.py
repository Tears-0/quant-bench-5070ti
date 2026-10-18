"""Paired, per-item comparison of downstream accuracy (lm-eval) against the reference.

Every configuration answers the *same* items with byte-identical prompts (verified here via
lm-eval's per-sample ``prompt_hash``). The right test of "did quantization change the
score" is therefore paired: per item, the difference between the quant's and the
reference's outcome. Its standard error is far smaller than the one obtained by treating
the two published scores as independent samples, because both configurations get the same
easy and hard items. Reporting the unpaired error would hide real differences; reporting
no error would invent them (requirements.md §6.9).

For each (candidate, task):

* ``delta`` = mean(candidate) − mean(reference) over paired items,
* ``se`` = std(per-item differences, ddof=1) / sqrt(n), ``ci95`` = delta ± 1.96·se,
* ``retention_pct`` = 100 · candidate / reference,
* the discordant counts (reference-only correct, candidate-only correct), which are what
  actually moves the score,
* ``tied`` = the 95% interval contains zero.

The MMLU subset is the unweighted mean of its subjects (as published); its paired error
combines the subjects' errors as independent (``sqrt(sum se_i^2) / k``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

#: (metric, filter) per task: the same primary metric pinned in metrics/quality.py.
PRIMARY: dict[str, tuple[str, str]] = {
    "arc_challenge": ("acc_norm", "none"),
    "hellaswag": ("acc_norm", "none"),
    "gsm8k": ("exact_match", "strict-match"),
}
MMLU_METRIC = ("acc", "none")


def primary_metric(task: str) -> tuple[str, str]:
    return PRIMARY.get(task, MMLU_METRIC)


@dataclass
class PairedResult:
    model: str
    candidate: str  # format:variant
    task: str
    metric: str
    n: int
    reference: float
    score: float
    delta: float
    se: float
    ci_low: float
    ci_high: float
    retention_pct: float | None
    ref_only_correct: int
    cand_only_correct: int
    prompts_identical: bool
    tied: bool


def _sample_files(run_dir: Path, task: str) -> list[Path]:
    # job runner: <task>/samples_<stamp>.jsonl; lm_eval CLI: <task>/<model>/samples_<task>_*.jsonl
    return sorted((run_dir / task).rglob("samples_*.jsonl"))


def load_samples(run_dir: Path, task: str) -> dict[int, tuple[float, str]]:
    """``doc_id -> (primary metric value, prompt_hash)`` for one task of one run."""
    metric, filt = primary_metric(task)
    out: dict[int, tuple[float, str]] = {}
    for path in _sample_files(run_dir, task):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("filter", "none") != filt or metric not in row:
                    continue
                out[int(row["doc_id"])] = (float(row[metric]), str(row.get("prompt_hash", "")))
    return out


def paired(
    reference: dict[int, tuple[float, str]],
    candidate: dict[int, tuple[float, str]],
) -> tuple[int, float, float, float, float, int, int, bool]:
    ids = sorted(reference.keys() & candidate.keys())
    if not ids:
        raise ValueError("no paired items")
    ref = [reference[i][0] for i in ids]
    cand = [candidate[i][0] for i in ids]
    diff = [c - r for c, r in zip(cand, ref, strict=True)]
    n = len(ids)
    mean_d = sum(diff) / n
    var = sum((d - mean_d) ** 2 for d in diff) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    same_prompts = all(reference[i][1] == candidate[i][1] for i in ids)
    ref_only = sum(1 for r, c in zip(ref, cand, strict=True) if r > c)
    cand_only = sum(1 for r, c in zip(ref, cand, strict=True) if c > r)
    return n, sum(ref) / n, sum(cand) / n, mean_d, se, ref_only, cand_only, same_prompts


def compare_runs(
    model: str,
    reference_dir: Path,
    candidates: dict[str, Path],
    tasks: Iterable[str],
) -> list[PairedResult]:
    results: list[PairedResult] = []
    tasks = list(tasks)
    for label, cand_dir in candidates.items():
        mmlu: list[PairedResult] = []
        for task in tasks:
            ref_samples, cand_samples = (
                load_samples(reference_dir, task),
                load_samples(cand_dir, task),
            )
            if not ref_samples or not cand_samples:
                continue
            n, r, c, d, se, ref_only, cand_only, same = paired(ref_samples, cand_samples)
            res = PairedResult(
                model=model,
                candidate=label,
                task=task,
                metric=primary_metric(task)[0],
                n=n,
                reference=r,
                score=c,
                delta=d,
                se=se,
                ci_low=d - 1.96 * se,
                ci_high=d + 1.96 * se,
                retention_pct=100.0 * c / r if r else None,
                ref_only_correct=ref_only,
                cand_only_correct=cand_only,
                prompts_identical=same,
                tied=(d - 1.96 * se) <= 0.0 <= (d + 1.96 * se),
            )
            results.append(res)
            if task.startswith("mmlu_"):
                mmlu.append(res)
        if mmlu:
            k = len(mmlu)
            r = sum(m.reference for m in mmlu) / k
            c = sum(m.score for m in mmlu) / k
            d = c - r
            se = math.sqrt(sum(m.se**2 for m in mmlu)) / k
            results.append(
                PairedResult(
                    model=model,
                    candidate=label,
                    task="mmlu_subset",
                    metric="acc (macro)",
                    n=sum(m.n for m in mmlu),
                    reference=r,
                    score=c,
                    delta=d,
                    se=se,
                    ci_low=d - 1.96 * se,
                    ci_high=d + 1.96 * se,
                    retention_pct=100.0 * c / r if r else None,
                    ref_only_correct=sum(m.ref_only_correct for m in mmlu),
                    cand_only_correct=sum(m.cand_only_correct for m in mmlu),
                    prompts_identical=all(m.prompts_identical for m in mmlu),
                    tied=(d - 1.96 * se) <= 0.0 <= (d + 1.96 * se),
                )
            )
    return results


def from_frame(frame: pd.DataFrame, lm_eval_root: Path) -> pd.DataFrame:
    """Pair every lm-eval candidate with its model's BF16/FP16 lm-eval reference run."""
    if frame.empty or "run_id" not in frame.columns:
        return pd.DataFrame()
    lm = frame[frame["run_id"].str.contains("lm-eval", na=False) & (frame["status"] == "ok")]
    lm = lm[lm.get("lm_eval_limit", pd.Series(index=lm.index, dtype=float)).isna()]
    rows: list[PairedResult] = []
    for model, group in lm.groupby("model"):
        refs = group[group["format"].isin(["bf16", "fp16"])]
        if refs.empty:
            continue
        ref_dir = lm_eval_root / str(refs.iloc[0]["run_id"])
        cands = {
            f"{r.format_variant}": lm_eval_root / str(r.run_id)
            for r in group.itertuples()
            if r.format not in ("bf16", "fp16")
        }
        tasks = (
            sorted({p.name for p in ref_dir.iterdir() if p.is_dir()}) if ref_dir.exists() else []
        )
        rows.extend(compare_runs(str(model), ref_dir, cands, tasks))
    return pd.DataFrame([asdict(r) for r in rows])


def paired_path(results_parquet: Path) -> Path:
    return results_parquet.with_name("lm_eval_paired.parquet")
