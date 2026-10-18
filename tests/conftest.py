"""Shared fixtures.

The whole test suite runs without a GPU: engines are replaced by the deterministic mock,
and every analysis function is exercised against fixture records (requirements.md §11.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.metrics.quality import LogprobArtifact
from quantbench.schema import (
    Distribution,
    EngineSpec,
    EnvironmentRecord,
    KLRecord,
    LatencyRecord,
    MemoryRecord,
    ModelRef,
    PowerRecord,
    ProtocolSpec,
    QualityRecord,
    QuantSpec,
    RunRecord,
    RunSpec,
    RunStatus,
    WorkloadSpec,
)


def make_spec(
    run_id: str = "test__gguf-q4km__llamacpp__b1__ctx512__gen16__latency",
    engine: str = "mock",
    fmt: str = "gguf",
    variant: str = "Q4_K_M",
    bits: float | None = 4,
    batch_size: int = 1,
    prompt_tokens: int = 64,
    generate_tokens: int = 16,
    kind: str = "latency",
    measured_iters: int = 2,
    warmup_iters: int = 1,
    **quant_kwargs: object,
) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        model=ModelRef(
            name="Qwen3-4B",
            hf_repo="Qwen/Qwen3-4B",
            revision="deadbeef",
            tier="A",
            params_total=4_000_000_000,
            params_non_embedding=3_500_000_000,
        ),
        quant=QuantSpec(format=fmt, variant=variant, bits_nominal=bits, **quant_kwargs),  # type: ignore[arg-type]
        engine=EngineSpec(name=engine, attention_backend="auto", kv_cache_dtype="f16"),
        workload=WorkloadSpec(
            kind=kind,  # type: ignore[arg-type]
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            eval_config="wikitext2" if kind == "quality" else None,
        ),
        protocol=ProtocolSpec(
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
            cooldown_target_c=None,
            timeout_s=120,
        ),
    )


def make_record(
    spec: RunSpec | None = None,
    status: RunStatus = RunStatus.ok,
    decode: float = 100.0,
    kl: float | None = 0.01,
    ppl: float | None = 9.0,
    vram_peak: float | None = 9000.0,
    synthetic: bool = False,
    tokens_per_joule: float | None = 0.5,
    lm_eval: dict[str, float | None] | None = None,
) -> RunRecord:
    spec = spec or make_spec()
    env = EnvironmentRecord(
        driver="570.10",
        cuda_runtime="12.8",
        torch="2.7.0+cu128",
        torch_arch_list=["sm_90", "sm_120"],
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        vram_total_mb=16303,
        is_wsl=False,
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    return RunRecord(
        spec_hash=spec.hash(),
        environment_hash=env.hash(),
        spec=spec,
        status=status,
        synthetic=synthetic,
        environment=env,
        latency=LatencyRecord(
            ttft_ms=Distribution(p50=40.0, p90=44.0, p99=48.0, iqr=3.0, n=3),
            itl_ms=Distribution(p50=8.0, p90=9.0, p95=9.5, p99=11.0, std=0.8, n=48),
            decode_tok_s=Distribution(p50=decode, iqr=2.0, n=3),
            prefill_tok_s=Distribution(p50=1500.0, n=3),
            throughput_tok_s_total=Distribution(p50=decode * 0.95, n=3),
        ),
        memory=MemoryRecord(
            vram_weights_mb=4200.0,
            vram_kv_mb=512.0,
            vram_peak_nvsmi_mb=vram_peak,
            effective_bpw=(spec.quant.bits_nominal or 16) + 0.5,
            load_time_s=4.0,
        ),
        power=PowerRecord(
            power_avg_w=180.0,
            power_peak_w=210.0,
            energy_j=100.0,
            tokens_per_joule=tokens_per_joule,
            gpu_temp_start_c=45.0,
            gpu_temp_end_c=60.0,
        ),
        quality=QualityRecord(
            reference_id="qwen3-4b-fp16@abc",
            eval_stream_sha="sha256:abc",
            ppl_wikitext2=ppl,
            ppl_reference=9.0 if ppl is not None else None,
            kl_div=KLRecord(
                mean=kl, p90=kl * 2 if kl else None, p99=kl * 4 if kl else None, topk=128
            ),
            top1_agreement=0.95,
            lm_eval=lm_eval or {},
            scored_by_engine="mock",
        ),
    )


def make_artifact(
    n: int = 64, k: int = 8, seed: int = 0, noise: float = 0.0, vocab: int = 100
) -> LogprobArtifact:
    """A synthetic scored artifact; ``noise`` perturbs the logits away from the reference."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(n, vocab))
    if noise:
        logits = logits + np.random.default_rng(seed + 1).normal(0.0, noise, size=logits.shape)
    logits -= logits.max(axis=1, keepdims=True)
    logprobs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    order = np.argsort(-logprobs, axis=1)[:, :k]
    topk_lp = np.take_along_axis(logprobs, order, axis=1)
    targets = rng.integers(0, vocab, size=n)
    return LogprobArtifact(
        topk_ids=order.astype(np.int64),
        topk_logprobs=topk_lp,
        target_ids=targets,
        target_logprobs=logprobs[np.arange(n), targets],
        meta={"synthetic": True, "k": k},
    )


def aligned_to(reference: LogprobArtifact, candidate_full_logprobs: np.ndarray) -> np.ndarray:
    return np.take_along_axis(candidate_full_logprobs, reference.topk_ids, axis=1)


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """An isolated results tree; the harness reads QUANTBENCH_RESULTS."""
    root = tmp_path / "results"
    (root / "raw").mkdir(parents=True)
    monkeypatch.setenv("QUANTBENCH_RESULTS", str(root))
    return root
