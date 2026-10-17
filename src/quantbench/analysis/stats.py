"""Aggregation, Pareto extraction, and statistical honesty (requirements.md §6.9).

The rule this module exists to enforce: **if two configurations' IQRs overlap, they are
indistinguishable and the repo says so** rather than ranking them. Every ranking helper
here returns the overlap information alongside the order, so a table can render "tied"
instead of inventing a winner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Interval:
    p50: float
    iqr: float | None

    @property
    def lo(self) -> float:
        return self.p50 - (self.iqr or 0.0) / 2.0

    @property
    def hi(self) -> float:
        return self.p50 + (self.iqr or 0.0) / 2.0


def indistinguishable(a: Interval, b: Interval) -> bool:
    """True when the two IQR bands overlap."""
    return a.lo <= b.hi and b.lo <= a.hi


def rank_with_ties(
    frame: pd.DataFrame, value_col: str, iqr_col: str, higher_is_better: bool = True
) -> pd.DataFrame:
    """Rank rows, marking each row that is statistically tied with the leader.

    Adds ``rank`` and ``tied_with_leader``. The README is expected to render tied rows
    without an ordering claim.
    """
    out = frame.dropna(subset=[value_col]).copy()
    if out.empty:
        out["rank"] = pd.Series(dtype="int64")
        out["tied_with_leader"] = pd.Series(dtype="bool")
        return out
    out = out.sort_values(value_col, ascending=not higher_is_better).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    leader_value = _safe(out.loc[0, value_col]) or 0.0
    leader = Interval(leader_value, _safe(out.loc[0, iqr_col]) if iqr_col in out else None)
    iqr_values: list[object] = list(out[iqr_col]) if iqr_col in out.columns else [None] * len(out)
    out["tied_with_leader"] = [
        indistinguishable(leader, Interval(_safe(v) or 0.0, _safe(i)))
        for v, i in zip(out[value_col], iqr_values, strict=False)
    ]
    return out


def _safe(value: object) -> float | None:
    """Best-effort float, tolerating pandas' NA sentinels. Never raises."""
    if value is None or not isinstance(value, int | float | str | Decimal):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def pareto_frontier(
    frame: pd.DataFrame,
    x: str,
    y: str,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """Non-dominated rows for the (x, y) objective pair.

    The headline figure is KL divergence (minimize) against decode tok/s (maximize) at
    batch 1, so those are the defaults' orientation.
    """
    data = frame.dropna(subset=[x, y]).copy()
    if data.empty:
        return data.assign(pareto=pd.Series(dtype="bool"))
    data = data.sort_values([x, y], ascending=[minimize_x, not maximize_y])
    frontier: list[object] = []
    best_y = float("-inf") if maximize_y else float("inf")
    for idx, row in data.iterrows():
        value = _safe(row[y])
        if value is None:
            continue
        better = value > best_y if maximize_y else value < best_y
        if better or not frontier:
            frontier.append(idx)
            best_y = value
    data["pareto"] = data.index.isin(frontier)
    return data


def summarize_by(
    frame: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Median across whatever repetition structure remains above the record level.

    Within a record, repetitions are already reduced to p50/IQR by the worker; this handles
    the case of several records sharing a configuration (e.g. re-runs on different days).
    """
    available = [m for m in metrics if m in frame.columns]
    if frame.empty or not available:
        return pd.DataFrame(columns=[*keys, *available])
    grouped = frame.groupby(keys, dropna=False)[available].median(numeric_only=True)
    counts = frame.groupby(keys, dropna=False).size().rename("n_records")
    return grouped.join(counts).reset_index()


def quality_vs_latency(
    frame: pd.DataFrame,
    batch_size: int = 1,
    quality_col: str = "kl_mean",
    speed_col: str = "decode_tok_s_p50",
) -> pd.DataFrame:
    """Join quality-workload rows onto latency rows for the same configuration.

    Quality and latency are separate runs (different workload kinds, sometimes different
    engines), so the join is explicit and the resulting frame keeps ``scored_by_engine``
    to make a cross-engine quality attribution visible in the plot legend (§7.2).
    """
    keys = ["model", "format_variant", "engine", "kv_cache_dtype"]
    required = {"workload_kind", "batch_size", "model", "format_variant"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    latency = frame[(frame["workload_kind"] == "latency") & (frame["batch_size"] == batch_size)]
    quality = frame[frame["workload_kind"] == "quality"]
    if latency.empty or quality.empty:
        return pd.DataFrame()

    quality_keys = ["model", "format_variant"]
    quality_cols = [
        c
        for c in (
            quality_col,
            "ppl_wikitext2",
            "top1_agreement",
            "top5_agreement",
            "scored_by_engine",
            "kl_tail_estimated",
        )
        if c in quality.columns
    ]
    # Take only the latency-side columns. Both frames carry the full schema, so merging
    # them wholesale would collide on the quality columns and leave the latency rows'
    # empty values in place — silently emptying the join.
    latency_cols = [
        c
        for c in (
            *keys,
            speed_col,
            "decode_tok_s_iqr",
            "itl_ms_p50",
            "ttft_ms_p50",
            "vram_peak_nvsmi_mb",
            "vram_peak_attributed_mb",
            "vram_weights_mb",
            "effective_bpw",
            "bits_nominal",
            "tokens_per_joule",
            "batch_size",
            "prompt_tokens",
            "imatrix",
        )
        if c in latency.columns
    ]
    merged = latency[latency_cols].merge(
        quality[[*quality_keys, *quality_cols]].drop_duplicates(quality_keys),
        on=quality_keys,
        how="inner",
    )
    return merged


def recommendation_table(
    frame: pd.DataFrame, kl_budgets: tuple[float, ...] = (0.01, 0.05, 0.1)
) -> pd.DataFrame:
    """ "If you want X on 16 GB, use Y" (§9.3).

    For each quality budget, the fastest configuration whose KL divergence is within
    budget and whose peak VRAM fits. Empty input yields an empty table — never a guess.
    """
    joined = quality_vs_latency(frame)
    if joined.empty:
        return pd.DataFrame(
            columns=["kl_budget", "recommendation", "decode_tok_s_p50", "vram_peak_nvsmi_mb"]
        )
    rows: list[dict[str, Any]] = []
    for budget in kl_budgets:
        eligible = joined[joined["kl_mean"] <= budget].dropna(subset=["decode_tok_s_p50"])
        if eligible.empty:
            rows.append({"kl_budget": budget, "recommendation": None})
            continue
        ranked = rank_with_ties(
            eligible, "decode_tok_s_p50", "decode_tok_s_iqr", higher_is_better=True
        )
        winners = ranked[ranked["tied_with_leader"]]
        rows.append(
            {
                "kl_budget": budget,
                "recommendation": " / ".join(
                    f"{r.format_variant} ({r.engine})" for r in winners.itertuples()
                ),
                "decode_tok_s_p50": _safe(ranked.iloc[0]["decode_tok_s_p50"]),
                "vram_peak_nvsmi_mb": _safe(ranked.iloc[0].get("vram_peak_nvsmi_mb")),
                "vram_peak_attributed_mb": _safe(ranked.iloc[0].get("vram_peak_attributed_mb")),
                "n_tied": len(winners),
            }
        )
    return pd.DataFrame(rows)
