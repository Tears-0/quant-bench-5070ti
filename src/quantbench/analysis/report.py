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
        body = markdown_table(table)
    else:
        table = table.rename(
            columns={
                "kl_budget": "KL budget (nats/token)",
                "recommendation": "use",
                "decode_tok_s_p50": "decode tok/s (p50, b1)",
                "vram_peak_nvsmi_mb": "peak VRAM, device (MB)",
                "vram_peak_attributed_mb": "peak VRAM, this run (MB)",
                "n_tied": "tied configs",
            }
        )
        body = markdown_table(table, digits=3)
    note = (
        "\n\nConfigurations whose IQRs overlap are reported as tied rather than ranked "
        "(§6.9). Peak VRAM is process-level NVML, the figure that decides whether it fits."
    )
    return Section("recommendations", body + note)


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


def hypotheses_section(frame: pd.DataFrame) -> Section:
    """Per-hypothesis verdicts. Absent evidence yields "insufficient data", not a guess."""
    checks = [
        (
            "H1",
            "Q5_K_M beats Q4_K_M on quality-per-VRAM below 8B, inverting under KV pressure",
            {"kl_mean", "vram_peak_nvsmi_mb"},
        ),
        (
            "H2",
            "PPL delta understates quantization damage relative to KL divergence",
            {"ppl_wikitext2", "kl_mean"},
        ),
        (
            "H3",
            "INT4 + Marlin-class kernels beat GGUF Q4_K_M batched, but not at batch 1",
            {"throughput_tok_s_total_p50", "batch_size"},
        ),
        (
            "H4",
            "KV-cache quantization buys more usable context per GB than weight quantization",
            {"vram_kv_mb", "prompt_tokens"},
        ),
    ]
    rows = []
    for key, statement, needed in checks:
        have = (
            needed.issubset(frame.columns)
            and not frame.dropna(subset=list(needed & set(frame.columns)), how="all").empty
        )
        rows.append(
            {
                "hypothesis": key,
                "statement": statement,
                "evidence": "data present — see docs/findings.md" if have else "insufficient data",
                "verdict": DASH,
            }
        )
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


def build_sections(frame: pd.DataFrame) -> list[Section]:
    return [
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
