"""Latency and throughput derivation (requirements.md §5.1).

Every metric here is derived from *per-token arrival timestamps*. Engines are required to
supply them (§7.2); deriving inter-token-latency percentiles from an aggregate is
impossible, and aggregate-only measurement is the most common flaw in amateur benchmarks.

Definitions, verbatim from §5.1 and mirrored in ``docs/methodology.md``:

``ttft_ms``
    Wall time from request submission to first generated token emitted.
``itl_ms``
    Inter-token latency: per-token deltas after the first. p50/p90/p95/p99 and std are all
    reported; the mean alone is never reported.
``decode_tok_s``
    ``(generated_tokens - 1) / (t_last_token - t_first_token)``, per request.
``prefill_tok_s``
    ``prompt_tokens / ttft_ms * 1000``. This conflates scheduling overhead into prefill.
``e2e_latency_ms``
    Submission to final token.
``throughput_tok_s_total``
    Aggregate generated tokens / wall time across the whole batch (serving-side number,
    distinct from per-request decode rate).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from quantbench.schema import Distribution, LatencyRecord


def describe(samples: Sequence[float], keep_samples: bool = True) -> Distribution:
    """Robust summary of a sample vector. Empty input yields an all-``None`` distribution."""
    values = [float(v) for v in samples if v is not None and np.isfinite(v)]
    if not values:
        return Distribution(n=0, samples=[])
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = (float(x) for x in np.percentile(arr, [25, 75]))
    return Distribution(
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        iqr=q3 - q1,
        min=float(arr.min()),
        max=float(arr.max()),
        n=int(arr.size),
        samples=[round(v, 6) for v in values] if keep_samples else [],
    )


def ttft_ms(submit_ms: float, token_timestamps_ms: Sequence[float]) -> float | None:
    if not token_timestamps_ms:
        return None
    return token_timestamps_ms[0] - submit_ms


def itl_series_ms(token_timestamps_ms: Sequence[float]) -> list[float]:
    """Per-token deltas *after* the first token."""
    if len(token_timestamps_ms) < 2:
        return []
    return [float(b - a) for a, b in pairwise(token_timestamps_ms)]


def decode_tok_s(token_timestamps_ms: Sequence[float]) -> float | None:
    if len(token_timestamps_ms) < 2:
        return None
    span_s = (token_timestamps_ms[-1] - token_timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return None
    return (len(token_timestamps_ms) - 1) / span_s


def prefill_tok_s(prompt_tokens: int, ttft: float | None) -> float | None:
    if not ttft or ttft <= 0 or prompt_tokens <= 0:
        return None
    return prompt_tokens * 1000.0 / ttft


class Iteration:
    """One measured repetition: a batch of sequences, each with arrival timestamps.

    All times are milliseconds on a single monotonic clock local to the worker process.
    """

    def __init__(
        self,
        submit_ms: float,
        sequences: Sequence[Sequence[float]],
        prompt_tokens: int,
        wall_ms: float | None = None,
    ) -> None:
        self.submit_ms = submit_ms
        self.sequences = [list(s) for s in sequences]
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = sum(len(s) for s in self.sequences)
        last = max((s[-1] for s in self.sequences if s), default=submit_ms)
        self.wall_ms = wall_ms if wall_ms is not None else last - submit_ms

    @property
    def ttfts(self) -> list[float]:
        return [t for s in self.sequences if (t := ttft_ms(self.submit_ms, s)) is not None]

    @property
    def itls(self) -> list[float]:
        out: list[float] = []
        for s in self.sequences:
            out.extend(itl_series_ms(s))
        return out

    @property
    def e2es(self) -> list[float]:
        return [s[-1] - self.submit_ms for s in self.sequences if s]

    @property
    def decode_rates(self) -> list[float]:
        return [r for s in self.sequences if (r := decode_tok_s(s)) is not None]

    @property
    def total_throughput(self) -> float | None:
        if self.wall_ms <= 0 or self.generated_tokens == 0:
            return None
        return self.generated_tokens / (self.wall_ms / 1000.0)


def summarize_iterations(
    iterations: Sequence[Iteration],
    keep_timestamps: bool = True,
) -> LatencyRecord:
    """Aggregate measured repetitions into the latency block of a run record.

    Per-request metrics pool every sequence of every repetition; ``throughput_tok_s_total``
    has one value per repetition because it is a property of the batch, not the request.
    """
    ttfts: list[float] = []
    itls: list[float] = []
    e2es: list[float] = []
    decodes: list[float] = []
    prefills: list[float] = []
    totals: list[float] = []
    stamps: list[list[float]] = []

    for it in iterations:
        ttfts.extend(it.ttfts)
        itls.extend(it.itls)
        e2es.extend(it.e2es)
        decodes.extend(it.decode_rates)
        prefills.extend(
            p for t in it.ttfts if (p := prefill_tok_s(it.prompt_tokens, t)) is not None
        )
        if (total := it.total_throughput) is not None:
            totals.append(total)
        if keep_timestamps:
            # Store timestamps relative to submission so records are comparable and small.
            stamps.extend([[round(t - it.submit_ms, 4) for t in s] for s in it.sequences])

    return LatencyRecord(
        ttft_ms=describe(ttfts),
        itl_ms=describe(itls, keep_samples=False),
        decode_tok_s=describe(decodes),
        prefill_tok_s=describe(prefills),
        e2e_latency_ms=describe(e2es),
        throughput_tok_s_total=describe(totals),
        per_token_timestamps_ms=stamps,
    )
