"""ExLlamaV3 engine adapter (EXL3, trellis/QTIP-based quantization).

EXL3 is the strongest quality-per-bit contender in the matrix and the format most people
have not benchmarked properly; its continuous bits-per-weight knob is what makes a smooth
quality-vs-bpw curve possible (§4.2). Unlike llama.cpp and vLLM there is no maintained
HTTP server surface worth driving, so this adapter uses the Python API in-process — inside
the worker subprocess, imported lazily, so a broken ExLlamaV3 install cannot take down the
CLI (§11.1).

Per-token timestamps come from a hand-written greedy loop with an explicit
``torch.cuda.synchronize()`` per step, matching the Transformers adapter so the two are
comparable to each other (§7.2).

Status: this adapter targets the ExLlamaV3 version pinned in ``docs/install/exllamav3.md``.
It is written from that API surface but is **not yet verified against real hardware** — it
is exercised only at M4. Until then treat it as unverified; see ``docs/open-questions.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
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
from quantbench.schema import RunSpec

MB = 1024 * 1024


class ExLlamaV3Engine(BaseEngine):
    name = "exllamav3"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.cache: Any = None
        self.tokenizer: Any = None
        self._torch: Any = None
        self._meta: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------------------

    def preflight(self, spec: RunSpec) -> PreflightResult:
        try:
            import exllamav3  # noqa: F401
            import torch
        except ImportError as exc:
            return PreflightResult(
                ok=False,
                reason=f"install requirements/exl3.txt (see docs/install/exllamav3.md): {exc}",
                can_generate=False,
            )
        if not torch.cuda.is_available():
            return PreflightResult(ok=False, reason="CUDA is not available", can_generate=False)
        if "sm_120" not in torch.cuda.get_arch_list():
            return PreflightResult(
                ok=False,
                reason=f"torch build lacks sm_120 kernels: {list(torch.cuda.get_arch_list())}",
                can_generate=False,
            )
        path = spec.quant.artifact_path
        if not path or not Path(path).exists():
            return PreflightResult(ok=False, reason=f"EXL3 artifact missing: {path}")
        return PreflightResult(ok=True, can_generate=True, can_score=True)

    def load(self, spec: RunSpec) -> LoadReport:
        import torch
        from exllamav3 import Cache, Config, Model, Tokenizer

        self._torch = torch
        path = str(spec.quant.artifact_path)
        w = spec.workload
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        try:
            config = Config.from_directory(path)
            self.model = Model.from_config(config)
            self.model.load()
            weights_mb = (torch.cuda.memory_allocated() - before) / MB
            self.cache = Cache(
                self.model,
                max_num_tokens=(w.prompt_tokens + w.generate_tokens) * max(1, w.batch_size),
                **self._cache_kwargs(spec),
            )
            self.tokenizer = Tokenizer.from_config(config)
        except torch.OutOfMemoryError as exc:
            raise EngineOOM(str(exc)) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineOOM(str(exc)) from exc
            raise EngineUnsupported(str(exc)) from exc
        load_time = time.perf_counter() - t0
        self._loaded = True
        self._meta["bpw"] = spec.quant.bits_nominal
        self._meta["kv_cache_dtype"] = spec.engine.kv_cache_dtype or "fp16"

        directory = Path(path)
        weight_bytes = sum(f.stat().st_size for f in directory.glob("*.safetensors")) or None
        return LoadReport(
            load_time_s=load_time,
            vram_weights_mb=weights_mb,
            weight_file_bytes=weight_bytes,
            extra=dict(self._meta),
        )

    def _cache_kwargs(self, spec: RunSpec) -> dict[str, Any]:
        """Map this repo's KV dtype vocabulary onto ExLlamaV3's cache layer classes."""
        from exllamav3 import CacheLayer_quant

        kv = (spec.engine.kv_cache_dtype or "fp16").lower()
        if kv in {"fp16", "f16", "auto"}:
            return {}
        bits = {"q8_0": 8, "q6": 6, "q4_0": 4}.get(kv)
        if bits is None:
            raise EngineUnsupported(f"exllamav3: unsupported kv_cache_dtype {kv!r}")
        return {"layer_type": CacheLayer_quant, "k_bits": bits, "v_bits": bits}

    def teardown(self) -> None:
        if self._torch is not None:
            self.cache = None
            self.model = None
            self._torch.cuda.empty_cache()
        self._loaded = False

    # -- generation --------------------------------------------------------------------

    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport:
        torch = self._torch
        if torch is None or self.model is None:
            raise EngineUnsupported("exllamav3: generate() called before load()")

        ids = torch.tensor(batch, dtype=torch.long, device="cuda")
        n_gen = spec.workload.generate_tokens
        torch.cuda.reset_peak_memory_stats()
        kv_before = torch.cuda.memory_allocated()
        stamps: list[float] = []
        t0 = time.perf_counter()
        try:
            params = {"attn_mode": "flash_attn"}
            logits = self.model.forward(input_ids=ids, params=params, cache=self.cache)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            stamps.append((time.perf_counter() - t0) * 1000.0)
            kv_after_prefill = torch.cuda.memory_allocated()

            for _ in range(n_gen - 1):
                logits = self.model.forward(input_ids=next_tok, params=params, cache=self.cache)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                stamps.append((time.perf_counter() - t0) * 1000.0)
        except torch.OutOfMemoryError as exc:
            raise EngineOOM(str(exc)) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineOOM(str(exc)) from exc
            raise

        wall_ms = (time.perf_counter() - t0) * 1000.0
        sequences = [
            SequenceTiming(
                submit_ms=0.0,
                token_timestamps_ms=list(stamps),
                prompt_tokens=len(prompt),
                generated_tokens=len(stamps),
                finish_reason="length",
            )
            for prompt in batch
        ]
        return GenReport(
            sequences=sequences,
            wall_time_ms=wall_ms,
            submit_ms=0.0,
            vram_kv_mb=(kv_after_prefill - kv_before) / MB,
            vram_peak_allocated_mb=torch.cuda.max_memory_allocated() / MB,
            extra={"lockstep_batch": True},
        )

    # -- scoring -----------------------------------------------------------------------

    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport:
        import torch

        from quantbench.metrics.quality import LogprobArtifact

        if self.model is None:
            raise EngineUnsupported("exllamav3: score() called before load()")

        ids = np.asarray(tokens["ids"], dtype=np.int64)
        ctx = int(tokens.get("ctx", 2048))
        stride = int(tokens.get("stride", ctx))
        topk = int(tokens.get("topk", 128))
        ref_topk_ids = tokens.get("ref_topk_ids")

        chunks_ids, chunks_lps, tgt_ids, tgt_lps, aligned = [], [], [], [], []
        position = 0
        for start in range(0, max(len(ids) - 1, 0), stride):
            chunk = ids[start : start + ctx]
            if len(chunk) < 2:
                break
            inp = torch.tensor(chunk[None, :], dtype=torch.long, device="cuda")
            logits = self.model.forward(input_ids=inp).float()[0]
            logprobs = torch.log_softmax(logits[:-1], dim=-1)
            targets = inp[0, 1:]
            vals, idx = torch.topk(logprobs, k=topk, dim=-1)
            chunks_ids.append(idx.cpu().numpy())
            chunks_lps.append(vals.cpu().numpy())
            tgt_ids.append(targets.cpu().numpy())
            tgt_lps.append(logprobs.gather(1, targets[:, None]).squeeze(1).cpu().numpy())
            if ref_topk_ids is not None:
                n = logprobs.shape[0]
                ref_slice = torch.tensor(
                    np.asarray(ref_topk_ids)[position : position + n],
                    dtype=torch.long,
                    device="cuda",
                )
                aligned.append(logprobs.gather(1, ref_slice).cpu().numpy())
            position += logprobs.shape[0]

        if not chunks_ids:
            raise EngineUnsupported("evaluation stream is too short to score")
        artifact = LogprobArtifact(
            topk_ids=np.concatenate(chunks_ids).astype(np.int64),
            topk_logprobs=np.concatenate(chunks_lps).astype(np.float64),
            target_ids=np.concatenate(tgt_ids).astype(np.int64),
            target_logprobs=np.concatenate(tgt_lps).astype(np.float64),
            ref_aligned_logprobs=np.concatenate(aligned).astype(np.float64) if aligned else None,
            meta={
                "engine": self.name,
                "model": spec.model.name,
                "quant": f"{spec.quant.format}:{spec.quant.variant}",
                "ctx": ctx,
                "stride": stride,
                "topk": topk,
                "eval_stream_sha": tokens.get("eval_stream_sha"),
            },
        )
        artifact.save(Path(tokens["out_path"]))
        nll = float(-artifact.target_logprobs.sum())
        return ScoreReport(
            n_positions=artifact.n_positions,
            scored_by_engine=self.name,
            artifact_path=str(tokens["out_path"]),
            ppl=float(np.exp(nll / artifact.n_positions)),
            nll_sum=nll,
        )

    # -- metadata ----------------------------------------------------------------------

    def describe(self) -> EngineMetadata:
        version = None
        try:
            import exllamav3

            version = getattr(exllamav3, "__version__", None)
        except ImportError:
            pass
        return EngineMetadata(
            name=self.name,
            version=version,
            attention_backend="flash_attn",
            kv_cache_dtype=self._meta.get("kv_cache_dtype"),
            extra=dict(self._meta),
        )
