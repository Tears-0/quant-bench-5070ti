"""Every figure in the repo (requirements.md §9.2).

Rules this module enforces:

* **Deterministic.** Fixed rcParams, a fixed SVG hash salt, no timestamps in file metadata.
  Regenerating figures from unchanged results produces byte-identical files, so a diff in
  ``figures/`` always means a change in the data.
* **No GPU.** Figures regenerate from committed results on any machine.
* **No invented data.** A figure whose inputs are missing is *skipped* with a stated
  reason; it is never drawn from partial or estimated values.
* **Colorblind-safe.** The Okabe–Ito palette throughout, with marker shape carrying format
  family so the encoding survives greyscale printing.

Every figure carries a caption stating N and the hardware, per §9.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from quantbench.analysis import stats

#: Okabe–Ito, colorblind-safe.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

#: KL estimator -> (line style, legend label). Never pooled on one line (methodology §2).
ESTIMATOR_STYLES: dict[str, tuple[str, str]] = {
    "llama-perplexity": ("-", "full-vocab KL vs F16 GGUF (llama.cpp)"),
    "topk-reference": ("--", "top-128 KL vs BF16 (Transformers)"),
}

FORMAT_MARKERS = {
    "gguf": "o",
    "awq": "s",
    "gptq": "D",
    "exl3": "^",
    "fp8": "v",
    "nf4": "P",
    "bf16": "*",
    "fp16": "*",
}

VRAM_CEILING_MB = 16303  # RTX 5070 Ti, as reported by NVML; overridden by the caller


@dataclass
class FigureResult:
    name: str
    paths: list[Path] = field(default_factory=list)
    skipped: bool = False
    reason: str | None = None
    n: int = 0


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "svg.hashsalt": "quant-bench-5070ti",  # deterministic element ids
            "svg.fonttype": "none",
            "path.simplify": False,
        }
    )


_CAPTIONS: dict[int, str] = {}


def caption(fig: Figure, text: str) -> None:
    """Attach a caption; :func:`save` places it beneath everything the axes draw."""
    _CAPTIONS[id(fig)] = text


def _place_caption(fig: Figure) -> None:
    text = _CAPTIONS.pop(id(fig), None)
    if text is None:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    bottoms = [bb.y0 for ax in fig.axes if (bb := ax.get_tightbbox(renderer)) is not None]
    y_px = min(bottoms) if bottoms else 0.0
    y = y_px / fig.bbox.height - 0.02
    fig.text(0.0, y, text, fontsize=8, va="top", ha="left", wrap=True, color="#444444")


def estimator_note(frame: pd.DataFrame) -> str:
    """Which KL estimator(s) a figure's points came from (methodology §2)."""
    names = sorted(
        {str(e) for e in frame.get("kl_estimator", pd.Series(dtype=str)).dropna().unique()}
    )
    labels = {
        "llama-perplexity": "full-vocabulary KL vs F16 GGUF (llama.cpp)",
        "topk-reference": "top-k KL vs BF16 (Transformers reference)",
    }
    if not names:
        return ""
    text = "; ".join(labels.get(n, n) for n in names)
    warn = " Mixed estimators: compare within one only." if len(names) > 1 else ""
    return f"KL: {text}.{warn}"


def hardware_note(frame: pd.DataFrame) -> str:
    gpus = sorted({str(g) for g in frame.get("gpu_name", pd.Series(dtype=str)).dropna().unique()})
    drivers = sorted({str(d) for d in frame.get("driver", pd.Series(dtype=str)).dropna().unique()})
    wsl = bool(frame.get("is_wsl", pd.Series(dtype=bool)).fillna(False).any())
    parts = [", ".join(gpus) or "hardware not recorded"]
    if drivers:
        parts.append(f"driver {', '.join(drivers)}")
    if wsl:
        parts.append("WSL2 (latency-noisy)")
    return " | ".join(parts)


def vram_note(frame: pd.DataFrame) -> str:
    modes = set(frame.get("vram_accounting", pd.Series(dtype=str)).dropna().unique())
    if modes == {"per_process"}:
        return "Peak is process-level NVML, which exceeds the allocator figure."
    if modes and modes <= {"per_process", "per_process_wddm"}:
        return (
            "Peak is per-process GPU memory (Windows GPU Process Memory counters where NVML "
            "has none under WDDM), which exceeds the allocator figure."
        )
    if "device_wide" in modes:
        return (
            "Peak is device-wide NVML minus the pre-load baseline: this driver model (Windows "
            "WDDM) offers no per-process accounting, so other processes' VRAM is subtracted, "
            "not attributed."
        )
    return "VRAM accounting mode not recorded."


def save(fig: Figure, out_dir: Path, name: str, watermark: bool = False) -> list[Path]:
    if watermark:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC DATA",
            fontsize=44,
            color="#CC0000",
            alpha=0.18,
            ha="center",
            va="center",
            rotation=30,
            transform=fig.transFigure,
        )
    for ax in fig.axes:
        # Minor tick labels on a log axis collide ("3x10^-3 4x10^-3"); majors suffice.
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_formatter(NullFormatter())
    _place_caption(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, metadata in ((".svg", {"Date": None}), (".png", {"Software": None})):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def _num(value: object, default: float = float("nan")) -> float:
    """Coerce a dataframe cell to a float for plotting; never raises."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _at_batch(frame: pd.DataFrame, batch_size: int = 1) -> pd.DataFrame:
    """Rows at one batch size, tolerating frames that never recorded the column."""
    if "batch_size" not in frame.columns:
        return frame
    return frame.loc[frame["batch_size"] == batch_size]


def _skip(name: str, reason: str) -> FigureResult:
    return FigureResult(name=name, skipped=True, reason=reason)


def _color_map(values: list[str]) -> dict[str, str]:
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(sorted(values))}


# --------------------------------------------------------------------------------------
# 1. headline Pareto — the README hero image
# --------------------------------------------------------------------------------------


def fig_headline_pareto(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False
) -> FigureResult:
    name = "01-headline-pareto"
    joined = stats.quality_vs_latency(frame, batch_size=1)
    joined = joined.dropna(subset=["kl_mean", "decode_tok_s_p50"]) if not joined.empty else joined
    if joined.empty:
        return _skip(name, "no configuration has both a batch-1 latency run and a KL measurement")

    front = stats.pareto_frontier(joined, x="kl_mean", y="decode_tok_s_p50")
    colors = _color_map([str(e) for e in joined["engine"].unique()])
    fig, ax = plt.subplots()
    for row in joined.itertuples():
        family = str(row.format_variant).split(":")[0]
        vram = getattr(row, "vram_peak_attributed_mb", None)
        ax.scatter(
            _num(row.kl_mean),
            _num(row.decode_tok_s_p50),
            s=40 + (0.02 * vram if vram and pd.notna(vram) else 40),
            c=colors[str(row.engine)],
            marker=FORMAT_MARKERS.get(family, "o"),
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            str(row.format_variant).split(":")[-1],
            (_num(row.kl_mean), _num(row.decode_tok_s_p50)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#333333",
        )
    edge = front[front["pareto"]].sort_values("kl_mean")
    if len(edge) > 1:
        ax.plot(
            edge["kl_mean"], edge["decode_tok_s_p50"], color="#666666", lw=1.2, ls="--", zorder=2
        )

    ax.set_xscale("log")
    ax.set_xlabel("KL divergence vs FP16 reference (nats/token, log scale — lower is better)")
    ax.set_ylabel("decode throughput at batch 1 (tokens/s — higher is better)")
    ax.set_title("Quality cost vs decode speed, batch 1")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=e) for e, c in colors.items()]
    ax.legend(handles=handles, title="engine", loc="best")
    caption(
        fig,
        f"N = {len(joined)} configurations. Marker size ∝ peak VRAM attributed to the run, "
        f"shape = format family. Dashed line: Pareto frontier. {estimator_note(joined)} "
        f"{hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(joined))


# --------------------------------------------------------------------------------------
# 2. quality vs effective bits-per-weight
# --------------------------------------------------------------------------------------


def fig_quality_vs_bpw(frame: pd.DataFrame, out_dir: Path, watermark: bool = False) -> FigureResult:
    name = "02-quality-vs-bpw"
    data = (
        frame.dropna(subset=["effective_bpw", "kl_mean"])
        if "effective_bpw" in frame
        else pd.DataFrame()
    )
    if data.empty:
        return _skip(name, "no rows with both effective_bpw and kl_mean")
    data = data.assign(
        family=data["format_variant"].astype(str).str.split(":").str[0],
        estimator=data.get("kl_estimator", pd.Series(index=data.index, dtype=str)).fillna(
            "unknown"
        ),
    )
    colors = _color_map([str(f) for f in data["family"].unique()])

    fig, ax = plt.subplots()
    # One series per (family, estimator): a line must never connect points measured by
    # different KL estimators, or a change of instrument would read as a trend (§6.1).
    for (family, estimator), group in data.groupby(["family", "estimator"]):
        group = group.sort_values("effective_bpw")
        ax.plot(
            group["effective_bpw"],
            group["kl_mean"],
            marker=FORMAT_MARKERS.get(str(family), "o"),
            color=colors[str(family)],
            ls=ESTIMATOR_STYLES.get(str(estimator), ("-", ""))[0],
            label=f"{family} · {ESTIMATOR_STYLES.get(str(estimator), ('', str(estimator)))[1]}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("effective bits per weight (embedding-excluded)")
    ax.set_ylabel("KL divergence vs FP16 (nats/token, log scale)")
    ax.set_title("Quality per bit, by format family")
    ax.legend(title="format · KL estimator", fontsize=8)
    caption(
        fig,
        f"N = {len(data)} configurations. effective_bpw = non-embedding weight bytes x 8 / "
        f"non-embedding parameters. {estimator_note(data)} {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 3. PPL vs KL disagreement (hypothesis H2)
# --------------------------------------------------------------------------------------


def fig_ppl_vs_kl(frame: pd.DataFrame, out_dir: Path, watermark: bool = False) -> FigureResult:
    name = "03-ppl-vs-kl"
    needed = {"ppl_wikitext2", "kl_mean"}
    if not needed.issubset(frame.columns):
        return _skip(name, "records carry no perplexity or KL columns")
    if "ppl_reference" not in frame.columns:
        return _skip(name, "records carry no same-positions reference perplexity")
    # Each row's delta is against *its own* reference over the identical scored positions:
    # a single global FP16 PPL would mix chunking conventions across engines (§6.1).
    data = frame.dropna(subset=[*needed, "ppl_reference"])
    if data.empty:
        return _skip(name, "no quantized rows with KL and a same-positions reference PPL")
    data = data.assign(ppl_delta=data["ppl_wikitext2"] - data["ppl_reference"])
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="#999999", lw=0.8)
    est = data.get("kl_estimator", pd.Series(index=data.index, dtype=str)).fillna("unknown")
    data = data.assign(estimator=est)
    colors = _color_map([str(e) for e in data["estimator"].unique()])
    for row in data.itertuples():
        ax.scatter(
            _num(row.kl_mean),
            _num(row.ppl_delta),
            color=colors[str(row.estimator)],
            s=45,
            zorder=3,
        )
        ax.annotate(
            str(row.format_variant),
            (_num(row.kl_mean), _num(row.ppl_delta)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("KL divergence vs FP16 (nats/token, log scale)")
    ax.set_ylabel("perplexity delta vs FP16 reference (same scored positions)")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=c, label=ESTIMATOR_STYLES.get(e, ("", e))[1])
            for e, c in colors.items()
        ],
        title="KL estimator",
        fontsize=8,
    )
    ax.set_title("H2: does perplexity delta track KL divergence?")
    caption(
        fig,
        f"N = {len(data)}. Points far right but near zero on the y axis are quants whose "
        f"perplexity looks harmless while their output distribution has moved. Each delta "
        f"is against that row's own reference over identical positions. "
        f"{estimator_note(data)} {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 4. VRAM composition
# --------------------------------------------------------------------------------------


def fig_vram_composition(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False, ceiling_mb: float = VRAM_CEILING_MB
) -> FigureResult:
    name = "04-vram-composition"
    if "vram_weights_mb" not in frame.columns:
        return _skip(name, "records carry no VRAM columns")
    data = frame[frame["workload_kind"] == "latency"] if "workload_kind" in frame else frame
    data = data.dropna(subset=["vram_weights_mb"])
    oom = frame[frame["status"] == "oom"] if "status" in frame else pd.DataFrame()
    if data.empty and oom.empty:
        return _skip(name, "no rows with measured VRAM and no recorded OOMs")

    peak_col = (
        "vram_peak_attributed_mb"
        if "vram_peak_attributed_mb" in data.columns
        else "vram_peak_nvsmi_mb"
    )
    data = data.sort_values(peak_col, na_position="first")
    labels = [
        f"{r.format_variant}\n{r.engine} b{r.batch_size}/ctx{r.prompt_tokens}"
        for r in data.itertuples()
    ]
    weights = data["vram_weights_mb"].fillna(0.0)
    kv = data.get("vram_kv_mb", pd.Series([0.0] * len(data), index=data.index)).fillna(0.0)
    peak = data.get(peak_col, pd.Series([float("nan")] * len(data), index=data.index))
    other = (peak.fillna(weights + kv) - weights - kv).clip(lower=0.0)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * (len(data) + len(oom))), 5.0))
    x = range(len(data))
    ax.bar(x, weights, color=PALETTE[0], label="weights")
    ax.bar(x, kv, bottom=weights, color=PALETTE[1], label="KV cache")
    ax.bar(x, other, bottom=weights + kv, color=PALETTE[2], label="activations + allocator")
    for offset, row in enumerate(oom.itertuples()):
        ax.bar(
            len(data) + offset,
            ceiling_mb * 1.1,
            color="none",
            edgecolor=PALETTE[5],
            hatch="//",
            label="OOM" if offset == 0 else None,
        )
        labels.append(f"{row.format_variant}\n{row.engine} b{row.batch_size} (OOM)")
    ax.axhline(ceiling_mb, color="#CC0000", lw=1.0, ls="--")
    ax.text(
        0,
        ceiling_mb,
        f"  {ceiling_mb / 1024:.1f} GiB card total",
        color="#CC0000",
        fontsize=8,
        va="bottom",
    )
    baseline = data.get("vram_baseline_nvsmi_mb", pd.Series(dtype=float)).dropna()
    if not baseline.empty and "device_wide" in set(
        data.get("vram_accounting", pd.Series()).dropna()
    ):
        # Other processes (desktop compositor, browser) already hold this much of the card,
        # so on this host the practical ceiling for a run is lower than the card total.
        available = ceiling_mb - float(baseline.median())
        ax.axhline(available, color="#CC0000", lw=0.8, ls=":")
        ax.text(
            0,
            available,
            f"  {available / 1024:.1f} GiB available on this host (median baseline "
            f"{float(baseline.median()) / 1024:.1f} GiB in use by other processes)",
            color="#CC0000",
            fontsize=8,
            va="bottom",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("VRAM (MiB)")
    ax.set_title("Where the 16 GB goes")
    ax.legend()
    caption(
        fig,
        f"N = {len(data)} measured configurations plus {len(oom)} recorded OOMs (hatched). "
        f"{vram_note(data)} {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 5-6. scaling
# --------------------------------------------------------------------------------------


def fig_throughput_vs_batch(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False
) -> FigureResult:
    name = "05-throughput-vs-batch"
    col = "throughput_tok_s_total_p50"
    if col not in frame.columns:
        return _skip(name, "no aggregate throughput column")
    data = frame.dropna(subset=[col, "batch_size"])
    if data.empty or data["batch_size"].nunique() < 2:
        return _skip(name, "need at least two batch sizes")
    colors = _color_map([str(f) for f in data["format_variant"].unique()])
    fig, ax = plt.subplots()
    for key, group in data.groupby(["format_variant", "engine"]):
        group = group.sort_values("batch_size")
        ax.plot(
            group["batch_size"],
            group[col],
            marker="o",
            color=colors[str(key[0])],
            label=f"{key[0]} ({key[1]})",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (log scale)")
    ax.set_ylabel("aggregate generated tokens/s")
    ax.set_title("Where batched INT4 kernels overtake GGUF (H3)")
    ax.legend(fontsize=8)
    caption(fig, f"N = {len(data)} runs. {hardware_note(frame)}.")
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


def fig_ttft_vs_prompt(frame: pd.DataFrame, out_dir: Path, watermark: bool = False) -> FigureResult:
    name = "06-ttft-vs-prompt-length"
    if "ttft_ms_p50" not in frame.columns:
        return _skip(name, "no TTFT column")
    data = _at_batch(frame.dropna(subset=["ttft_ms_p50", "prompt_tokens"]))
    if data.empty or data["prompt_tokens"].nunique() < 2:
        return _skip(name, "need at least two prompt lengths at batch 1")
    colors = _color_map([str(f) for f in data["format_variant"].unique()])
    fig, ax = plt.subplots()
    for key, group in data.groupby(["format_variant", "engine"]):
        group = group.sort_values("prompt_tokens")
        ax.errorbar(
            group["prompt_tokens"],
            group["ttft_ms_p50"],
            yerr=group.get("ttft_ms_iqr", pd.Series(0.0, index=group.index)).fillna(0.0) / 2.0,
            marker="o",
            capsize=3,
            color=colors[str(key[0])],
            label=f"{key[0]} ({key[1]})",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("prompt tokens (log scale)")
    ax.set_ylabel("TTFT (ms, log scale) — median, IQR whiskers")
    ax.set_title("Prefill scaling at batch 1")
    ax.legend(fontsize=8)
    caption(fig, f"N = {len(data)} runs. {hardware_note(frame)}.")
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 7. ITL distribution
# --------------------------------------------------------------------------------------


def fig_itl_distribution(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False
) -> FigureResult:
    name = "07-itl-distribution"
    cols = ["itl_ms_p50", "itl_ms_p90", "itl_ms_p95", "itl_ms_p99"]
    if not set(cols).issubset(frame.columns):
        return _skip(name, "records carry no ITL percentiles")
    data = _at_batch(frame).dropna(subset=cols)
    if data.empty:
        return _skip(name, "no batch-1 runs with ITL percentiles")
    quantiles = [50, 90, 95, 99]
    colors = _color_map([str(f) for f in data["format_variant"].unique()])
    fig, ax = plt.subplots()
    for row in data.itertuples():
        values = [_num(getattr(row, c)) for c in cols]
        ax.plot(
            values,
            quantiles,
            marker="o",
            color=colors[str(row.format_variant)],
            label=f"{row.format_variant!s} ({row.engine!s})",
        )
    ax.set_xlabel("inter-token latency (ms)")
    ax.set_ylabel("percentile")
    ax.set_yticks(quantiles)
    ax.set_title("ITL tail at batch 1 — the mean hides this")
    ax.legend(fontsize=8)
    caption(
        fig,
        f"N = {len(data)} runs. Quantile plot built from recorded p50/p90/p95/p99; the full "
        f"per-token timestamps live in results/raw/. {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 8-9. context cliff and efficiency
# --------------------------------------------------------------------------------------


def fig_context_cliff(frame: pd.DataFrame, out_dir: Path, watermark: bool = False) -> FigureResult:
    name = "08-context-cliff"
    if "prompt_tokens" not in frame.columns or "status" not in frame.columns:
        return _skip(name, "records lack prompt length or status")
    ok = frame.loc[frame["status"] == "ok"]
    if "workload_kind" in ok.columns:
        ok = ok.loc[ok["workload_kind"] == "latency"]  # quality runs say nothing about context
    ok = _at_batch(ok)
    if ok.empty:
        return _skip(name, "no successful batch-1 runs")
    grouped = (
        ok.groupby(["format_variant", "kv_cache_dtype"], dropna=False)["prompt_tokens"]
        .max()
        .reset_index()
    )
    if grouped.empty:
        return _skip(name, "no context data")
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.4 * len(grouped))))
    labels = [f"{r.format_variant} / kv={r.kv_cache_dtype}" for r in grouped.itertuples()]
    ax.barh(labels, grouped["prompt_tokens"], color=PALETTE[0])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("maximum context that ran without OOM (prompt tokens, log scale)")
    ax.set_title("Context cliff at batch 1, with and without KV quantization (H4)")
    caption(
        fig,
        f"N = {len(ok)} successful runs. Bars show the largest *tested* context that fit, "
        f"not a theoretical maximum. {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(grouped))


def fig_efficiency(frame: pd.DataFrame, out_dir: Path, watermark: bool = False) -> FigureResult:
    name = "09-tokens-per-joule"
    if "tokens_per_joule" not in frame.columns:
        return _skip(name, "no power measurements")
    data = frame.dropna(subset=["tokens_per_joule"]).sort_values("tokens_per_joule")
    if data.empty:
        return _skip(name, "no rows with tokens_per_joule")
    labels = [f"{r.format_variant} ({r.engine}) b{r.batch_size}" for r in data.itertuples()]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35 * len(data))))
    ax.barh(labels, data["tokens_per_joule"], color=PALETTE[2])
    ax.set_xlabel("tokens per joule (higher is better)")
    ax.set_title("Efficiency ranking")
    caption(
        fig,
        f"N = {len(data)} runs. Energy is trapezoidal integration of NVML power over the "
        f"measured window only (load and warmup excluded). {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


# --------------------------------------------------------------------------------------
# 10-11. accuracy retention and the imatrix ablation
# --------------------------------------------------------------------------------------


def fig_accuracy_retention(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False, paired: pd.DataFrame | None = None
) -> FigureResult:
    name = "10-accuracy-retention"
    if paired is None or paired.empty:
        return _skip(name, "no paired lm-eval comparisons (run `quantbench analyze`)")
    tasks = [
        t
        for t in ("arc_challenge", "hellaswag", "gsm8k", "mmlu_subset")
        if t in set(paired["task"])
    ]
    data = paired[paired["task"].isin(tasks)]
    cands = sorted(data["candidate"].unique())
    if not tasks or not cands:
        return _skip(name, "no candidate rows")

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(tasks)), 5.0))
    width = 0.8 / len(cands)
    for i, cand in enumerate(cands):
        sub = data[data["candidate"] == cand].set_index("task").reindex(tasks)
        ret = sub["retention_pct"].astype(float)
        # paired 95% interval of the score, expressed as % of the reference score
        lo = 100.0 * (sub["reference"] + sub["ci_low"]) / sub["reference"]
        hi = 100.0 * (sub["reference"] + sub["ci_high"]) / sub["reference"]
        x = [j + (i - (len(cands) - 1) / 2) * width for j in range(len(tasks))]
        ax.bar(
            x,
            ret,
            width=width * 0.95,
            color=PALETTE[i % len(PALETTE)],
            label=str(cand),
            yerr=[(ret - lo).to_numpy(), (hi - ret).to_numpy()],
            capsize=3,
        )
        for xi, (task_name, row) in zip(x, sub.iterrows(), strict=False):
            if not bool(row["tied"]):
                ax.text(xi, float(hi[task_name]) + 0.3, "*", ha="center", fontsize=12)
    ax.axhline(100.0, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    lowest = float(
        min(lo.min() for lo in [100.0 * (data["reference"] + data["ci_low"]) / data["reference"]])
    )
    ax.set_ylim(min(90.0, lowest - 1.0), 103.0)
    ax.set_ylabel("score as % of reference (paired 95% interval)")
    ax.set_title("Downstream accuracy retention")
    ax.legend(fontsize=8, loc="lower left")
    n_items = int(data.drop_duplicates("task")["n"].sum())
    caption(
        fig,
        f"{n_items:,} items; every configuration answered the same items with identical "
        f"prompts. Whiskers: paired per-item 95% interval; * = interval excludes 100% "
        f"(a real difference, not noise). GGUF scored dequantized via Transformers in fp16, "
        f"like the reference. {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(data))


def fig_imatrix_ablation(
    frame: pd.DataFrame, out_dir: Path, watermark: bool = False
) -> FigureResult:
    name = "11-imatrix-ablation"
    if "imatrix" not in frame.columns or "kl_mean" not in frame.columns:
        return _skip(name, "no imatrix flag or KL column")
    data = frame.dropna(subset=["kl_mean"])
    pairs: list[tuple[str, float, float]] = []
    for (model, variant), group in data.groupby(["model", "variant"]):
        with_im = group[group["imatrix"].fillna(False).astype(bool)]["kl_mean"]
        without = group[~group["imatrix"].fillna(False).astype(bool)]["kl_mean"]
        if not with_im.empty and not without.empty:
            pairs.append((f"{model} {variant}", float(without.median()), float(with_im.median())))
    if not pairs:
        return _skip(name, "no quant level was measured both with and without an imatrix")

    fig, ax = plt.subplots()
    x = range(len(pairs))
    ax.bar(
        [i - 0.2 for i in x], [p[1] for p in pairs], width=0.4, color=PALETTE[0], label="no imatrix"
    )
    ax.bar(
        [i + 0.2 for i in x],
        [p[2] for p in pairs],
        width=0.4,
        color=PALETTE[1],
        label="with imatrix",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([p[0] for p in pairs], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("KL divergence vs FP16 (nats/token)")
    ax.set_title("Isolating the importance-matrix effect")
    ax.legend()
    caption(
        fig,
        f"N = {len(pairs)} paired comparisons at identical quant levels; the only variable "
        f"is imatrix presence. {hardware_note(frame)}.",
    )
    return FigureResult(name=name, paths=save(fig, out_dir, name, watermark), n=len(pairs))


FIGURES = [
    fig_headline_pareto,
    fig_quality_vs_bpw,
    fig_ppl_vs_kl,
    fig_vram_composition,
    fig_throughput_vs_batch,
    fig_ttft_vs_prompt,
    fig_itl_distribution,
    fig_context_cliff,
    fig_efficiency,
    fig_accuracy_retention,
    fig_imatrix_ablation,
]


def generate_all(
    frame: pd.DataFrame,
    out_dir: Path,
    only: list[str] | None = None,
    paired: pd.DataFrame | None = None,
) -> list[FigureResult]:
    """Regenerate every figure. Missing inputs skip a figure; they never fake one."""
    apply_theme()
    watermark = bool(frame.get("synthetic", pd.Series(dtype=bool)).fillna(False).any())
    results: list[FigureResult] = []
    for fn in FIGURES:
        figure_name = fn.__name__.removeprefix("fig_")
        if only and not any(token in figure_name for token in only):
            continue
        try:
            if fn is fig_accuracy_retention:
                results.append(fn(frame, out_dir, watermark, paired=paired))
            else:
                results.append(fn(frame, out_dir, watermark))
        except Exception as exc:
            results.append(FigureResult(name=figure_name, skipped=True, reason=f"error: {exc}"))
    return results


def results_table(results: list[FigureResult]) -> list[dict[str, Any]]:
    return [
        {"figure": r.name, "status": "skipped" if r.skipped else "ok", "n": r.n, "reason": r.reason}
        for r in results
    ]
