"""Regenerate the tables embedded in README.md and docs/findings.md (§9.1 ``report``).

Tables are written between HTML comment markers so prose and generated content can live in
the same file without a template engine. A missing measurement renders as ``—`` with a
footnote — never as an estimate (§0.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantbench.analysis import stats

DASH = "—"

BEGIN = "<!-- BEGIN quantbench:{key} -->"
END = "<!-- END quantbench:{key} -->"


@dataclass
class Section:
    key: str
    markdown: str


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return DASH
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    if frame.empty:
        return f"_No data yet. Run the benchmark; cells stay {DASH} until a run record exists._"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(fmt(v, digits) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def recommendation_section(frame: pd.DataFrame) -> Section:
    table = stats.recommendation_table(frame)
    if table.empty:
        return Section("recommendations", markdown_table(table))
    table = table.rename(
        columns={
            "kl_budget": "KL budget (nats/token)",
            "recommendation": "use",
            "decode_tok_s_p50": "decode tok/s (p50, b1)",
            "vram_peak_nvsmi_mb": "peak VRAM, device (MiB)",
            "vram_peak_attributed_mb": "peak VRAM, this run (MiB)",
            "n_tied": "tied configs",
        }
    )
    joined = stats.quality_vs_latency(frame)
    candidates = sorted({f"{r.format_variant} ({r.engine})" for r in joined.itertuples()})
    models = sorted({str(m) for m in joined["model"].unique()})
    estimators = sorted({str(e) for e in joined.get("kl_estimator", pd.Series()).dropna()})
    latency_rows = frame[frame.get("workload_kind") == "latency"] if not frame.empty else frame
    modes = set(latency_rows.get("vram_accounting", pd.Series(dtype=str)).dropna())
    vram = (
        "Device peak includes other processes on this host (no per-process accounting); "
        "'this run' subtracts the pre-load baseline."
        if "device_wide" in modes
        else "Peak VRAM is per-process (Windows GPU Process Memory counters), the figure that "
        "decides whether it fits."
        if "per_process_wddm" in modes
        else "Peak VRAM is process-level NVML, the figure that decides whether it fits."
    )
    note = (
        f"\n\n**Chosen from {len(candidates)} candidate configuration(s) only** "
        f"({', '.join(candidates)}; model: {', '.join(models)}; KL estimator: "
        f"{', '.join(estimators) or DASH}). A row is the fastest *measured* configuration "
        "within its KL budget, not a claim about formats that have not run. Configurations "
        f"whose IQRs overlap are reported as tied rather than ranked (§6.9). {vram}"
    )
    return Section("recommendations", markdown_table(table, digits=3) + note)


def coverage_section(frame: pd.DataFrame) -> Section:
    """What ran, what OOMed, what never built — the honest matrix status (§0.2)."""
    if frame.empty or "status" not in frame.columns:
        return Section("coverage", markdown_table(pd.DataFrame()))
    pivot = frame.pivot_table(
        index=["model", "format_variant"],
        columns="status",
        values="run_id",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    return Section("coverage", markdown_table(pivot, digits=0))


def _ok(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if frame.empty or "workload_kind" not in frame.columns:
        return pd.DataFrame()
    return frame[(frame["workload_kind"] == kind) & (frame["status"] == "ok")]


def _quantized_quality(frame: pd.DataFrame) -> pd.DataFrame:
    quality = _ok(frame, "quality")
    if quality.empty or "kl_mean" not in quality.columns:
        return pd.DataFrame()
    return quality.dropna(subset=["kl_mean"])


def hypothesis_evidence(frame: pd.DataFrame) -> dict[str, str]:
    """What each hypothesis actually needs, checked against the data — not column names."""
    latency, quality = _ok(frame, "latency"), _quantized_quality(frame)
    out: dict[str, str] = {}

    variants = set(quality.get("format_variant", pd.Series(dtype=str)))
    need_h1 = {"gguf:Q5_K_M", "gguf:Q4_K_M"}
    long_ctx = (
        latency[latency.get("prompt_tokens", pd.Series(dtype=float)) >= 8192]
        if not latency.empty
        else latency
    )
    if not need_h1 <= variants:
        out["H1"] = (
            f"insufficient data: needs quality runs for {', '.join(sorted(need_h1 - variants))}"
        )
    elif long_ctx.empty:
        out["H1"] = "partial: quality pair present; long-context (>= 8k) runs missing"
    else:
        out["H1"] = "data present — see docs/findings.md"

    with_ref = (
        quality.dropna(subset=["ppl_reference"])
        if "ppl_reference" in quality
        else quality.iloc[0:0]
    )
    # count per KL estimator: rows from different estimators are not one sample (§6.1)
    if "kl_estimator" in with_ref.columns and not with_ref.empty:
        n = int(with_ref.groupby(with_ref["kl_estimator"].fillna("unknown")).size().max())
    else:
        n = len(with_ref)
    if n == 0:
        out["H2"] = (
            "insufficient data: no quantized rows with KL and a same-positions PPL reference"
        )
    elif n < 4:
        out["H2"] = f"preliminary (n={n} configurations per KL estimator) — see docs/findings.md"
    else:
        out["H2"] = f"data present (n={n}) — see docs/findings.md"

    families = set(latency.get("format", pd.Series(dtype=str)))
    batches = (
        latency.get("batch_size", pd.Series(dtype=float)).nunique() if not latency.empty else 0
    )
    if not ({"awq", "gptq"} & families):
        out["H3"] = "insufficient data: needs AWQ/GPTQ latency runs (M4)"
    elif batches < 2:
        out["H3"] = "insufficient data: needs at least two batch sizes"
    else:
        out["H3"] = "data present — see docs/findings.md"

    kv = (
        latency.groupby("format_variant")["kv_cache_dtype"].nunique()
        if not latency.empty
        else pd.Series(dtype=int)
    )
    if kv.empty or kv.max() < 2:
        out["H4"] = "insufficient data: needs a KV-cache dtype sweep (core profile)"
    else:
        out["H4"] = "data present — see docs/findings.md"
    return out


def hypotheses_section(frame: pd.DataFrame) -> Section:
    """Per-hypothesis evidence status. Absent evidence says what is missing; never a guess."""
    statements = {
        "H1": "Q5_K_M beats Q4_K_M on quality-per-VRAM below 8B, inverting under KV pressure",
        "H2": "PPL delta understates quantization damage relative to KL divergence",
        "H3": "INT4 + Marlin-class kernels beat GGUF Q4_K_M batched, but not at batch 1",
        "H4": "KV-cache quantization buys more usable context per GB than weight quantization",
    }
    evidence = hypothesis_evidence(frame) if not frame.empty else {}
    rows = [
        {
            "hypothesis": key,
            "statement": statement,
            "evidence": evidence.get(key, "insufficient data"),
            "verdict": DASH,
        }
        for key, statement in statements.items()
    ]
    return Section("hypotheses", markdown_table(pd.DataFrame(rows)))


LATENCY_COLUMNS = {
    "model": "model",
    "format_variant": "format",
    "engine": "engine",
    "batch_size": "batch",
    "prompt_tokens": "prompt",
    "generate_tokens": "gen",
    "decode_tok_s_p50": "decode tok/s (p50)",
    "decode_tok_s_iqr": "IQR",
    "ttft_ms_p50": "TTFT ms (p50)",
    "itl_ms_p50": "ITL ms (p50)",
    "itl_ms_p99": "ITL ms (p99)",
    "vram_peak_attributed_mb": "peak VRAM MiB (run)",
    "effective_bpw": "eff. bpw",
    "tokens_per_joule": "tok/J",
    "power_capped_fraction": "power-capped",
}


def latency_section(frame: pd.DataFrame) -> Section:
    """Every successful latency record, one row each — the raw numbers behind the figures."""
    if frame.empty or "workload_kind" not in frame.columns:
        return Section("latency", markdown_table(pd.DataFrame()))
    rows = frame[(frame["workload_kind"] == "latency") & (frame["status"] == "ok")]
    cols = [c for c in LATENCY_COLUMNS if c in rows.columns]
    table = rows[cols].sort_values(cols[:6]).rename(columns=LATENCY_COLUMNS)
    if "power-capped" in table.columns:
        table["power-capped"] = [
            DASH if pd.isna(v) else f"{100 * float(v):.0f}%" for v in table["power-capped"]
        ]
    note = (
        "\n\nMedians over measured repetitions; IQR is of per-request decode rate. Rows whose "
        "IQRs overlap are statistically indistinguishable (§6.9). Peak VRAM is attributable to "
        "the run; see the VRAM note in docs/methodology.md for how that is defined per host."
    )
    return Section("latency", markdown_table(table, digits=2) + (note if not table.empty else ""))


def environment_section(frame: pd.DataFrame) -> Section:
    if frame.empty:
        return Section("environment", markdown_table(pd.DataFrame()))
    cols = [
        c
        for c in ("gpu_name", "driver", "engine", "engine_version", "is_wsl", "vram_accounting")
        if c in frame.columns
    ]
    unique = frame[cols].drop_duplicates() if cols else pd.DataFrame()
    return Section("environment", markdown_table(unique))


TASK_ORDER = [
    "arc_challenge",
    "hellaswag",
    "gsm8k",
    "mmlu_subset",
    "mmlu_abstract_algebra",
    "mmlu_college_computer_science",
    "mmlu_high_school_mathematics",
    "mmlu_professional_law",
    "mmlu_world_religions",
]


def accuracy_section(paired: pd.DataFrame | None) -> Section:
    """Downstream accuracy, reference vs every quant, with *paired* per-item uncertainty."""
    if paired is None or paired.empty:
        return Section("accuracy", markdown_table(pd.DataFrame()))
    rows = []
    for model, group in paired.groupby("model"):
        cands = sorted(group["candidate"].unique())
        for task in [t for t in TASK_ORDER if t in set(group["task"])]:
            sub = group[group["task"] == task]
            row: dict[str, object] = {
                "model": model,
                "task": task,
                "n": int(sub.iloc[0]["n"]),
                "reference": f"{100 * sub.iloc[0]['reference']:.1f}",
            }
            for cand in cands:
                hit = sub[sub["candidate"] == cand]
                if hit.empty:
                    row[cand] = DASH
                    continue
                r = hit.iloc[0]
                verdict = "tied" if r["tied"] else ("**worse**" if r["delta"] < 0 else "**better**")
                row[cand] = (
                    f"{100 * r['score']:.1f} ({r['retention_pct']:.1f}%; "
                    f"Δ {100 * r['delta']:+.1f} "
                    f"[{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}], "
                    f"{verdict})"
                )
            rows.append(row)
    note = (
        "\n\nScores in %. Each cell: score (retention vs reference; paired Δ in points with its "
        '95% interval over the *same* items; verdict). "tied" means the interval contains zero '
        "(§6.9). Pairing is per item on byte-identical prompts (lm-eval `prompt_hash`), which is "
        "why the intervals are much tighter than the published per-score standard errors. "
        "MMLU subset = unweighted mean of the five pinned subjects. GGUF quants are scored "
        "dequantized through Transformers, in fp16 like the reference (open question Q15)."
    )
    return Section("accuracy", markdown_table(pd.DataFrame(rows)) + note)


def build_sections(frame: pd.DataFrame, paired: pd.DataFrame | None = None) -> list[Section]:
    return [
        accuracy_section(paired),
        recommendation_section(frame),
        coverage_section(frame),
        hypotheses_section(frame),
        latency_section(frame),
        environment_section(frame),
    ]


def apply_sections(path: Path, sections: list[Section]) -> list[str]:
    """Replace marked blocks in a markdown file. Returns the keys that were updated."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    updated: list[str] = []
    for section in sections:
        begin, end = BEGIN.format(key=section.key), END.format(key=section.key)
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            continue
        text = pattern.sub(f"{begin}\n{section.markdown}\n{end}", text)
        updated.append(section.key)
    path.write_text(text, encoding="utf-8")
    return updated
