"""Deterministic synthetic engine for CI, tests, and ``--dry-run``.

This engine fabricates timings. That is the *only* place in the repo where a number is
invented, and it is fenced off three ways:

* records it produces carry ``synthetic: true``;
* the runner writes them to ``results/dryrun/`` which is git-ignored;
* :mod:`quantbench.analysis.load` refuses to mix synthetic records into a real dataset
  unless explicitly asked, and tags every figure produced from them.

Its purpose is to exercise the schema, the planner, the runner, the aggregation and the
plotting code on a machine with no GPU (requirements.md §0.5, §11.3), not to estimate
anything about real hardware.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from quantbench.engines.base import (
    BaseEngine,
    EngineMetadata,
    GenReport,
    LoadReport,
    PreflightResult,
    ScoreReport,
    SequenceTiming,
)
from quantbench.schema import RunSpec


def _seed_from(spec: RunSpec) -> int:
    digest = hashlib.sha256(spec.hash().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class MockEngine(BaseEngine):
    name = "mock"

    def __init__(self) -> None:
        super().__init__()
        self._rng: np.random.Generator | None = None
        self._spec: RunSpec | None = None

    def preflight(self, spec: RunSpec) -> PreflightResult:
        return PreflightResult(
            ok=True,
            can_generate=True,
            can_score=True,
            warnings=["mock engine: all values are synthetic"],
        )

    def load(self, spec: RunSpec) -> LoadReport:
        self._spec = spec
        self._rng = np.random.default_rng(_seed_from(spec))
        self._loaded = True
        bits = spec.quant.bits_nominal or 16.0
        params = spec.model.params_total or 4_000_000_000
        weight_bytes = int(params * bits / 8)
        time.sleep(0.001)
        return LoadReport(
            load_time_s=float(self._rng.uniform(2.0, 6.0)),
            vram_weights_mb=weight_bytes / (1024 * 1024),
            weight_file_bytes=weight_bytes,
            extra={"synthetic": True},
        )

    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport:
        rng = self._rng if self._rng is not None else np.random.default_rng(_seed_from(spec))
        bits = spec.quant.bits_nominal or 16.0
        n_gen = spec.workload.generate_tokens
        n_prompt = spec.workload.prompt_tokens
        batch_size = max(1, spec.workload.batch_size)

        # A toy model: smaller weights decode faster, larger batches amortise, longer
        # prompts cost prefill. Shape only; magnitudes mean nothing.
        base_itl_ms = 6.0 * (bits / 4.0) ** 0.5 * (1.0 + 0.05 * np.log2(batch_size))
        prefill_ms = 0.05 * n_prompt * (bits / 4.0) ** 0.25 + 25.0

        submit_ms = 0.0
        sequences: list[SequenceTiming] = []
        for _ in range(batch_size):
            ttft = float(prefill_ms * rng.normal(1.0, 0.02))
            itls = rng.gamma(shape=40.0, scale=base_itl_ms / 40.0, size=max(n_gen - 1, 0))
            stamps = np.concatenate([[ttft], ttft + np.cumsum(itls)]) if n_gen else np.array([])
            sequences.append(
                SequenceTiming(
                    submit_ms=submit_ms,
                    token_timestamps_ms=[float(x) for x in stamps],
                    prompt_tokens=n_prompt,
                    generated_tokens=int(stamps.size),
                    finish_reason="length",
                )
            )
        wall = max(
            (s.token_timestamps_ms[-1] for s in sequences if s.token_timestamps_ms), default=0.0
        )
        kv_mb = 2 * 36 * 8 * 128 * (n_prompt + n_gen) * batch_size * 2 / (1024 * 1024)
        return GenReport(
            sequences=sequences,
            wall_time_ms=wall,
            submit_ms=submit_ms,
            vram_kv_mb=kv_mb,
            vram_peak_allocated_mb=None,
            extra={"synthetic": True},
        )

    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport:
        rng = self._rng if self._rng is not None else np.random.default_rng(_seed_from(spec))
        n = 256
        bits = spec.quant.bits_nominal or 16.0
        nll = float(rng.normal(2.0 + 0.4 / bits, 0.01) * n)
        # Synthetic quality numbers, shaped so the KL/PPL plotting path has something to
        # draw in CI. Monotone in bit width purely so the figures are legible.
        kl = 0.0 if bits >= 16 else float(0.4 / bits**2 * rng.normal(1.0, 0.05))
        return ScoreReport(
            n_positions=n,
            scored_by_engine=self.name,
            ppl=float(np.exp(nll / n)),
            nll_sum=nll,
            native_metrics={
                "kl_mean": kl,
                "kl_p90": kl * 2.1,
                "kl_p99": kl * 5.4,
                "kl_max": kl * 20.0,
                "top1_agreement": float(min(1.0, 1.0 - kl * 2)),
            },
            extra={"synthetic": True},
        )

    def describe(self) -> EngineMetadata:
        return EngineMetadata(
            name=self.name,
            version="synthetic",
            attention_backend="none",
            extra={"synthetic": True},
        )
