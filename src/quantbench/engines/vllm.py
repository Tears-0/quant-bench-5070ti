"""vLLM engine adapter (AWQ / GPTQ / FP8).

vLLM is driven as an **OpenAI-compatible server subprocess** rather than in-process, for
three reasons: streaming responses give honest per-token timestamps (§7.2); a segfault
costs one data point instead of the session (§6.4); and the CLI never has to import vLLM,
which on ``sm_120`` is exactly the import that fails (§11.1).

Blackwell hazards this adapter is built around (§3.2)
-----------------------------------------------------
* Prebuilt wheels have historically shipped without consumer-Blackwell arch flags. The
  install recipe (``docs/install/vllm.md``) pins either a source build with
  ``TORCH_CUDA_ARCH_LIST="12.0"`` or a specific nightly, and the exact artifact is recorded
  in every run record.
* AWQ on ``sm_120`` is reported to work through the ``awq_marlin`` kernel path, sometimes
  only with the Triton attention backend rather than FlashAttention. The attention backend
  is therefore a *recorded, sweepable variable* (``VLLM_ATTENTION_BACKEND``), not an
  implementation detail.
* Block-scaled FP8 via ``cutlass_scaled_mm`` has had ``sm_120`` gaps. FP8 is marked
  conditional in the format config: if the server refuses to start, the failure text lands
  in the record and in ``docs/failures.md`` — it is a finding, not a crash to hide.

Scoring: vLLM can return ``prompt_logprobs``, which is enough for PPL and for KL over its
own top-k, but not for log-probs on an arbitrary reference support. ``can_score`` is
therefore gated behind ``engine.extra.allow_native_scoring``; by default quality for
vLLM-served formats is routed to the designated scoring engine and the record says so via
``scored_by_engine`` (§7.2).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from quantbench.engines.base import (
    BaseEngine,
    EngineMetadata,
    EngineOOM,
    EngineUnsupported,
    GenReport,
    LoadReport,
    PreflightResult,
    ScoreReport,
    SequenceTiming,
)
from quantbench.engines.server import ManagedServer, find_binary, free_port, post_json, stream_sse
from quantbench.paths import logs_dir
from quantbench.schema import RunSpec

QUANT_ARG = {
    "awq": "awq_marlin",
    "gptq": "gptq_marlin",
    "fp8": "fp8",
    "bf16": None,
    "fp16": None,
}

ATTENTION_BACKENDS = {
    "auto": None,
    "flash_attn": "FLASH_ATTN",
    "flashinfer": "FLASHINFER",
    "triton": "TRITON_ATTN",
    "triton_attn": "TRITON_ATTN",
    "xformers": "XFORMERS",
}


class VllmEngine(BaseEngine):
    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self.server: ManagedServer | None = None
        self._meta: dict[str, Any] = {}
        self._model_id: str = ""

    # -- lifecycle ---------------------------------------------------------------------

    def preflight(self, spec: RunSpec) -> PreflightResult:
        warnings: list[str] = []
        try:
            binary = find_binary("vllm", "VLLM_BIN")
        except EngineUnsupported as exc:
            return PreflightResult(ok=False, reason=str(exc), can_generate=False)
        self._meta["binary"] = binary

        if spec.quant.format not in QUANT_ARG:
            return PreflightResult(
                ok=False,
                reason=f"vLLM adapter has no mapping for format {spec.quant.format!r}",
                can_generate=False,
            )
        backend = (spec.engine.attention_backend or "auto").lower()
        if backend not in ATTENTION_BACKENDS:
            return PreflightResult(ok=False, reason=f"unknown attention backend {backend!r}")
        if spec.quant.format == "fp8":
            warnings.append(
                "FP8 W8A8 on consumer Blackwell is unverified; a startup failure here is a "
                "finding for docs/failures.md, not a bug"
            )
        allow_native = bool(spec.engine.extra.get("allow_native_scoring", False))
        return PreflightResult(
            ok=True, can_generate=True, can_score=allow_native, warnings=warnings
        )

    def _server_cmd(self, spec: RunSpec, port: int) -> list[str]:
        w, e = spec.workload, spec.engine
        model = spec.quant.artifact_path or spec.model.hf_repo
        self._model_id = str(model)
        cmd = [
            self._meta.get("binary", "vllm"),
            "serve",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(w.prompt_tokens + w.generate_tokens),
            "--max-num-seqs",
            str(max(1, w.batch_size)),
            "--seed",
            str(w.seed),
            "--disable-log-requests",
            # Prefix caching would turn the second measured iteration into a cache hit and
            # silently delete prefill from the measurement (§6.5).
            "--no-enable-prefix-caching",
        ]
        if e.extra.get("enforce_eager"):
            cmd.append("--enforce-eager")
        if quant := QUANT_ARG.get(spec.quant.format):
            cmd += ["--quantization", quant]
        if e.kv_cache_dtype and e.kv_cache_dtype != "auto":
            cmd += ["--kv-cache-dtype", e.kv_cache_dtype]
        if spec.model.revision and not spec.quant.artifact_path:
            cmd += ["--revision", spec.model.revision]
        if gpu_util := e.extra.get("gpu_memory_utilization"):
            cmd += ["--gpu-memory-utilization", str(gpu_util)]
        cmd += list(e.flags)
        return cmd

    def load(self, spec: RunSpec) -> LoadReport:
        port = free_port()
        env: dict[str, str] = {}
        backend = ATTENTION_BACKENDS.get((spec.engine.attention_backend or "auto").lower())
        if backend:
            env["VLLM_ATTENTION_BACKEND"] = backend
        env["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "INFO")

        log_path = logs_dir() / f"vllm-{spec.run_id}.log"
        self.server = ManagedServer(
            cmd=self._server_cmd(spec, port),
            port=port,
            health_path="/health",
            startup_timeout_s=float(max(spec.protocol.timeout_s, 900)),
            env=env,
            log_path=log_path,
        )
        t0 = time.perf_counter()
        self.server.start()
        try:
            self.server.wait_healthy()
        except EngineUnsupported:
            self.server.classify_exit()
            raise
        load_time = time.perf_counter() - t0
        self._meta["attention_backend"] = backend or "auto"
        self._meta["kv_cache_dtype"] = spec.engine.kv_cache_dtype or "auto"
        self._loaded = True
        return LoadReport(
            load_time_s=load_time,
            vram_weights_mb=None,  # measured by the worker as an NVML delta across load
            extra={"server_pid": self.server.pid, "port": port, "log": str(log_path)},
        )

    def teardown(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self._loaded = False

    # -- generation --------------------------------------------------------------------

    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport:
        if self.server is None:
            raise EngineUnsupported("vllm: generate() called before load()")
        url = self.server.base_url + "/v1/completions"
        w = spec.workload
        t0 = time.perf_counter()

        def one(prompt_ids: list[int]) -> SequenceTiming:
            payload: dict[str, Any] = {
                "model": self._model_id,
                "prompt": prompt_ids,
                "max_tokens": w.generate_tokens,
                "min_tokens": w.generate_tokens if w.ignore_eos else 0,
                "temperature": w.temperature,
                "seed": w.seed,
                "stream": True,
                "ignore_eos": w.ignore_eos,
            }
            stamps: list[float] = []
            finish: str | None = None
            for tok in stream_sse(url, payload, t0=t0, timeout=float(spec.protocol.timeout_s)):
                if tok.text:
                    stamps.append(tok.t_ms)
                if tok.finish_reason:
                    finish = tok.finish_reason
            return SequenceTiming(
                submit_ms=0.0,
                token_timestamps_ms=stamps,
                prompt_tokens=len(prompt_ids),
                generated_tokens=len(stamps),
                finish_reason=finish,
            )

        try:
            with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
                sequences = list(pool.map(one, batch))
        except Exception as exc:
            text = self.server.tail(8000).lower() if self.server else ""
            if "out of memory" in text or "no available memory for the cache blocks" in text:
                raise EngineOOM(self.server.tail(2000) if self.server else str(exc)) from exc
            raise
        return GenReport(
            sequences=sequences,
            wall_time_ms=(time.perf_counter() - t0) * 1000.0,
            submit_ms=0.0,
            extra={"n_requests": len(batch)},
        )

    # -- scoring -----------------------------------------------------------------------

    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport:
        """Score via ``prompt_logprobs``.

        Only reachable when ``engine.extra.allow_native_scoring`` is set. The resulting KL
        is computed on vLLM's own top-k support, so records produced this way are flagged
        ``tail_mass_estimated`` downstream.
        """
        if self.server is None:
            raise EngineUnsupported("vllm: score() called before load()")
        if not spec.engine.extra.get("allow_native_scoring", False):
            raise EngineUnsupported(
                "vLLM native scoring is disabled; route quality through the designated "
                "scoring engine (see FormatConfig.scoring_engine)"
            )
        from quantbench.metrics.quality import LogprobArtifact

        ids = [int(i) for i in np.asarray(tokens["ids"], dtype=np.int64).tolist()]
        ctx = int(tokens.get("ctx", 2048))
        topk = int(tokens.get("topk", 128))
        url = self.server.base_url + "/v1/completions"

        rows_ids: list[list[int]] = []
        rows_lps: list[list[float]] = []
        tgt_ids: list[int] = []
        tgt_lps: list[float] = []

        for start in range(0, max(len(ids) - 1, 0), ctx):
            chunk = ids[start : start + ctx]
            if len(chunk) < 2:
                break
            resp = post_json(
                url,
                {
                    "model": self._model_id,
                    "prompt": chunk,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": topk,
                },
                timeout=float(spec.protocol.timeout_s),
            )
            positions = (resp["choices"][0].get("prompt_logprobs") or [])[1:]
            for pos, target in zip(positions, chunk[1:], strict=False):
                if not pos:
                    continue
                items = sorted(pos.items(), key=lambda kv: kv[1]["logprob"], reverse=True)[:topk]
                rows_ids.append([int(k) for k, _ in items])
                rows_lps.append([float(v["logprob"]) for _, v in items])
                tgt_ids.append(int(target))
                tgt_lps.append(float(pos.get(str(target), {}).get("logprob", float("nan"))))

        if not rows_ids:
            raise EngineUnsupported("vLLM returned no prompt_logprobs")
        width = min(len(r) for r in rows_ids)
        artifact = LogprobArtifact(
            topk_ids=np.array([r[:width] for r in rows_ids], dtype=np.int64),
            topk_logprobs=np.array([r[:width] for r in rows_lps], dtype=np.float64),
            target_ids=np.array(tgt_ids, dtype=np.int64),
            target_logprobs=np.array(tgt_lps, dtype=np.float64),
            meta={
                "engine": self.name,
                "model": spec.model.name,
                "quant": f"{spec.quant.format}:{spec.quant.variant}",
                "ctx": ctx,
                "topk": width,
                "eval_stream_sha": tokens.get("eval_stream_sha"),
                "note": "scored on vLLM's own top-k support; KL tail is approximate",
            },
        )
        artifact.save(tokens["out_path"])
        finite = artifact.target_logprobs[np.isfinite(artifact.target_logprobs)]
        nll = float(-finite.sum()) if finite.size else None
        return ScoreReport(
            n_positions=artifact.n_positions,
            scored_by_engine=self.name,
            artifact_path=str(tokens["out_path"]),
            ppl=float(np.exp(nll / finite.size)) if nll is not None and finite.size else None,
            nll_sum=nll,
        )

    # -- metadata ----------------------------------------------------------------------

    def describe(self) -> EngineMetadata:
        version = None
        if self.server is not None:
            try:
                info = post_json(self.server.base_url + "/version", {}, timeout=10)
                version = info.get("version")
            except Exception:
                version = None
        return EngineMetadata(
            name=self.name,
            version=version or os.environ.get("VLLM_VERSION"),
            commit=os.environ.get("VLLM_COMMIT"),
            attention_backend=self._meta.get("attention_backend"),
            kv_cache_dtype=self._meta.get("kv_cache_dtype"),
            binary_path=self._meta.get("binary"),
            kernels=["awq_marlin"] if "awq" in self._model_id.lower() else [],
        )
