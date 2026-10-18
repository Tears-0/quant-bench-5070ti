"""One run, one process (requirements.md §7.3).

    python -m quantbench.worker --spec spec.json --out record.json

The worker owns exactly one :class:`~quantbench.schema.RunSpec`: it imports the engine
(lazily, here, never in the parent), executes the warmup/measured protocol, samples NVML
throughout, and writes a schema-valid record. It is designed so that *every* failure mode
still produces a record: an OOM, a build failure and a segfault are data points, not lost
runs. The parent handles timeouts and hard crashes (§7.3).

A fresh process per run also gives a fresh CUDA context, so allocator fragmentation and
engine global state cannot leak from one configuration into the next (§6.4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from quantbench.checkpoint import artifact_weight_bytes
from quantbench.config import load_eval, load_model
from quantbench.engines.base import (
    BaseEngine,
    EngineOOM,
    EngineUnsupported,
    GenReport,
    load_engine,
)
from quantbench.metrics import latency as latency_metrics
from quantbench.metrics import power as power_metrics
from quantbench.metrics import quality as quality_metrics
from quantbench.metrics.memory import (
    NvmlSampler,
    effective_bpw,
    peak_process_mb,
    peak_shared_mb,
    vram_accounting_mode,
)
from quantbench.preflight import capture_environment
from quantbench.prompts import build_prompt_batch
from quantbench.reference import eval_stream as eval_stream_mod
from quantbench.reference import logits_cache
from quantbench.schema import (
    MemoryRecord,
    PplConfig,
    QualityRecord,
    RunRecord,
    RunSpec,
    RunStatus,
)


def _now_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class Worker:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.record = RunRecord(
            spec_hash=spec.hash(),
            spec=spec,
            status=RunStatus.crashed,  # replaced on every successful path
            synthetic=spec.engine.name == "mock",
            environment=capture_environment(),
        )
        self.record.environment_hash = self.record.environment.hash()
        self.sampler = NvmlSampler(hz=spec.protocol.sample_hz)
        self.engine: BaseEngine | None = None

    # -- main ---------------------------------------------------------------------------

    def run(self) -> RunRecord:
        started = time.perf_counter()
        try:
            self.engine = load_engine(self.spec.engine.name)
            pf = self.engine.preflight(self.spec)
            self.record.warnings.extend(pf.warnings)
            if not pf.ok:
                self.record.status = RunStatus.engine_unsupported
                self.record.error = pf.reason
                return self.record

            self.engine.child_pid_hook = self.sampler.track_pid
            self.sampler.start()
            if not self.sampler.available:
                self.record.warnings.append(
                    f"NVML sampling unavailable ({self.sampler.error}); memory and power "
                    "fields will be null"
                )

            baseline = self._poll_mb()
            load_report = self.engine.load(self.spec)
            if pid := load_report.extra.get("server_pid"):
                self.sampler.track_pid(int(pid))
                time.sleep(0.5)  # let the sampler see the server's allocation
            after_load = self._poll_mb()

            self.record.engine_metadata = self.engine.describe().model_dump(mode="json")
            self.record.memory = MemoryRecord(
                vram_baseline_nvsmi_mb=baseline,
                vram_weights_mb=load_report.vram_weights_mb
                if load_report.vram_weights_mb is not None
                else (after_load - baseline if (after_load and baseline) else None),
                vram_kv_mb=load_report.kv_cache_mb_reported,
                weight_file_bytes=load_report.weight_file_bytes,
                load_time_s=load_report.load_time_s,
            )

            # Workloads may downgrade the status themselves (throttled_invalid); only the
            # untouched placeholder is promoted to ok.
            self.record.status = RunStatus.ok
            if self.spec.workload.kind == "quality":
                self._run_quality()
            else:
                self._run_latency()
        except EngineOOM as exc:
            # §5.3: OOM is a first-class outcome recorded with the context/batch that hit it.
            self.record.status = RunStatus.oom
            self.record.error = str(exc)[:8000]
        except EngineUnsupported as exc:
            self.record.status = RunStatus.engine_unsupported
            self.record.error = str(exc)[:8000]
        except Exception as exc:
            self.record.status = RunStatus.crashed
            self.record.error = f"{type(exc).__name__}: {exc}"[:8000]
            self.record.stderr_tail = traceback.format_exc()[-8000:]
        finally:
            if self.engine is not None:
                try:
                    self.engine.teardown()
                except Exception as exc:
                    self.record.warnings.append(f"teardown failed: {exc}")
            self.sampler.stop()
            self.record.duration_s = time.perf_counter() - started
        return self.record

    # -- workloads ----------------------------------------------------------------------

    def _run_latency(self) -> None:
        assert self.engine is not None
        spec = self.spec
        stream_ids = self._prompt_stream_ids()
        prompts = build_prompt_batch(spec, stream_ids)
        self.record.artifacts["prompt_source"] = prompts.source
        if prompts.source == "synthetic":
            self.record.warnings.append(
                "prompts are synthetic token ids, not text; prefill numbers are comparable "
                "across engines but not to real-text workloads"
            )

        # §6.1: discard warmups, then measure. Warmup energy is excluded from the window.
        for _ in range(spec.protocol.warmup_iters):
            self.engine.generate(prompts.ids, spec)

        window_start = self._sampler_clock()
        iterations: list[latency_metrics.Iteration] = []
        reports: list[GenReport] = []
        for _ in range(spec.protocol.measured_iters):
            report = self.engine.generate(prompts.ids, spec)
            reports.append(report)
            iterations.append(
                latency_metrics.Iteration(
                    submit_ms=report.submit_ms,
                    sequences=[s.token_timestamps_ms for s in report.sequences],
                    prompt_tokens=spec.workload.prompt_tokens,
                    wall_ms=report.wall_time_ms,
                )
            )
        window_end = self._sampler_clock()

        self.record.latency = latency_metrics.summarize_iterations(iterations)
        self._check_token_counts(reports)
        self._finalize_memory(reports)
        self._finalize_power(
            window_start, window_end, sum(it.generated_tokens for it in iterations)
        )

    def _run_quality(self) -> None:
        """Score this configuration against its FP16 reference (§5.2)."""
        assert self.engine is not None
        spec = self.spec
        if not spec.workload.eval_config:
            raise EngineUnsupported("quality workload without an eval_config")
        if self.record.synthetic:
            self._run_quality_synthetic()
            return
        model_cfg = load_model(_slug(spec.model.name))
        eval_cfg = load_eval(spec.workload.eval_config)
        if eval_cfg.kind == "lm_eval":
            window_start = self._sampler_clock()
            record = self._quality_lm_eval(eval_cfg)
            window_end = self._sampler_clock()
            self.record.quality = record
            self._finalize_power(window_start, window_end, generated_tokens=None)
            self._finalize_memory([])
            return
        try:
            stream = eval_stream_mod.load_eval_stream(
                eval_stream_mod.stream_dir(model_cfg, eval_cfg)
            )
        except (FileNotFoundError, OSError) as exc:
            raise EngineUnsupported(
                f"no evaluation stream for {model_cfg.name}/{eval_cfg.name} ({exc}); "
                "build it once with: quantbench reference build --model "
                f"{model_cfg.slug} --eval {eval_cfg.name}"
            ) from exc

        window_start = self._sampler_clock()
        if spec.engine.name == "llama.cpp":
            record = self._quality_llamacpp(model_cfg, eval_cfg, stream)
        else:
            record = self._quality_topk(model_cfg, eval_cfg, stream)
        window_end = self._sampler_clock()

        record.calibration_disjoint_from_eval = _calibration_is_disjoint(spec, eval_cfg)
        self.record.quality = record
        self._finalize_power(window_start, window_end, generated_tokens=None)
        self._finalize_memory([])

    # -- quality: this repo's top-k reference (Transformers, ExLlamaV3, vLLM) ----------

    def _quality_topk(self, model_cfg: Any, eval_cfg: Any, stream: Any) -> QualityRecord:
        assert self.engine is not None
        spec = self.spec
        key = logits_cache.key_for(model_cfg, eval_cfg, stream.meta)
        is_reference = spec.quant.format in {"bf16", "fp16"}
        add_bos = bool(eval_cfg.ppl.add_bos and stream.meta.get("bos_token_id") is not None)
        if eval_cfg.ppl.add_bos and not add_bos:
            self.record.warnings.append(
                "eval config asks for add_bos but this tokenizer has no BOS token; "
                "chunks are scored without one (recorded as add_bos=false)"
            )

        ref_artifact = None
        if not is_reference:
            try:
                ref_artifact = logits_cache.load_reference(key)
            except FileNotFoundError as exc:
                raise EngineUnsupported(str(exc)) from exc

        out_path = (
            logits_cache.reference_path(key)
            if is_reference
            else logits_cache.reference_path(key).parent / f"cand-{spec.hash()[7:31]}.npz"
        )
        tokens: dict[str, Any] = {
            "ids": stream.ids,
            "ctx": eval_cfg.ppl.ctx,
            "stride": eval_cfg.ppl.stride,
            "add_bos": add_bos,
            "topk": eval_cfg.reference.topk,
            "out_path": str(out_path),
            "eval_stream_sha": stream.sha,
            "ref_topk_ids": ref_artifact.topk_ids if ref_artifact is not None else None,
        }
        score = self.engine.score(tokens, spec)

        record = QualityRecord(
            reference_id=key.reference_id,
            eval_stream_sha=stream.sha,
            ppl_wikitext2=score.ppl,
            ppl_config=PplConfig(
                ctx=eval_cfg.ppl.ctx,
                stride=eval_cfg.ppl.stride,
                tokens=score.n_positions,
                add_bos=add_bos,
                scoring="every position of each non-overlapping chunk except the first",
            ),
            scored_by_engine=score.scored_by_engine,
            token_stream_verified=True,  # scored directly on the hashed id stream
        )
        if is_reference:
            artifact = quality_metrics.LogprobArtifact.load(out_path)
            mass = quality_metrics.tail_mass(artifact.topk_logprobs)
            meta = {
                "engine": score.scored_by_engine,
                "n_positions": score.n_positions,
                "ppl": score.ppl,
                "spec_hash": self.record.spec_hash,
                "tail_mass_mean": float(mass.mean()),
                "tail_mass_p99": float(np.percentile(mass, 99)),
                "tail_mass_max": float(mass.max()),
            }
            logits_cache.write_reference_meta(key, meta)
            self.record.artifacts["reference_logprobs"] = str(out_path)
            self.record.artifacts["reference_tail_mass"] = (
                f"mean={meta['tail_mass_mean']:.3e} p99={meta['tail_mass_p99']:.3e} "
                f"max={meta['tail_mass_max']:.3e}"
            )
            record.ppl_reference = score.ppl
            record.kl_div.topk = int(artifact.topk_ids.shape[1])
            record.kl_div.estimator = "topk-reference"
        elif score.artifact_path and ref_artifact is not None:
            candidate = quality_metrics.LogprobArtifact.load(Path(score.artifact_path))
            record.kl_div = quality_metrics.kl_record(ref_artifact, candidate)
            record.kl_div.estimator = "topk-reference"
            record.ppl_reference = quality_metrics.perplexity(ref_artifact.target_logprobs)
            record.top1_agreement = quality_metrics.agreement(ref_artifact, candidate, k=1)
            record.top5_agreement = quality_metrics.agreement(ref_artifact, candidate, k=5)
            self.record.artifacts["candidate_logprobs"] = score.artifact_path
        return record

    # -- quality: downstream accuracy via lm-evaluation-harness ------------------------

    def _quality_lm_eval(self, eval_cfg: Any) -> QualityRecord:
        assert self.engine is not None
        from quantbench import lmeval
        from quantbench.paths import results_dir

        spec = self.spec
        limit = spec.engine.extra.get("lm_eval_limit")
        if spec.engine.name == "llama.cpp":
            server = getattr(self.engine, "server", None)
            if server is None:
                raise EngineUnsupported("llama.cpp lm-eval run without a running llama-server")
            backend = "gguf"
            model_args = f"base_url={server.base_url}"
        elif spec.engine.name == "transformers":
            backend = "hf"
            source = spec.quant.artifact_path or spec.model.hf_repo
            model_args = lmeval.model_args_hf(
                str(source), spec.quant.format, dtype=eval_cfg.lm_eval_dtype
            )
        else:
            raise EngineUnsupported(f"no lm-eval backend wired for engine {spec.engine.name!r}")

        out_dir = results_dir() / "lm_eval" / spec.run_id
        from quantbench.lmeval_tune import tuned_batch_sizes, tuning_path
        from quantbench.paths import configs_dir

        tuning_file = tuning_path(configs_dir(), _slug(spec.model.name))
        tuned = tuned_batch_sizes(tuning_file)
        if not tuned:
            self.record.warnings.append(
                f"no tuned lm-eval batch sizes at {tuning_file}; every task runs at batch "
                f"{eval_cfg.lm_eval_batch_size} (quantbench lmeval-tune measures them)"
            )
        try:
            result = lmeval.run(
                backend,
                model_args,
                eval_cfg.lm_eval_tasks,
                out_dir,
                seed=spec.workload.seed,
                timeout_s=spec.protocol.timeout_s,
                limit=int(limit) if limit else None,
                batch_sizes=tuned,
                default_batch_size=eval_cfg.lm_eval_batch_size,
                on_spawn=self.sampler.track_pid,
            )
        except lmeval.LmEvalOOM as exc:
            raise EngineOOM(str(exc)) from exc
        except lmeval.LmEvalError as exc:
            text = str(exc).lower()
            if "echo" in text or "logprob" in text or "token_logprobs" in text:
                raise EngineUnsupported(
                    f"{spec.engine.name} does not serve the prompt log-probs lm-eval's "
                    f"{backend} backend needs: {exc}"
                ) from exc
            raise
        self.record.artifacts["lm_eval_output"] = str(out_dir)
        self.record.artifacts["lm_eval_version"] = str(result.lm_eval_version)
        self.record.artifacts["lm_eval_dtype"] = eval_cfg.lm_eval_dtype
        self.record.artifacts["lm_eval_batch_sizes"] = json.dumps(result.batch_sizes)
        self.record.artifacts["lm_eval_task_seconds"] = json.dumps(
            {k: round(v, 1) for k, v in result.task_seconds.items()}
        )
        if result.resumed:
            self.record.artifacts["lm_eval_resumed_tasks"] = ",".join(result.resumed)
        self.record.engine_metadata.setdefault("extra", {})["lm_eval_commands"] = [
            " ".join(c) for c in result.commands
        ]
        if limit:
            self.record.warnings.append(
                f"lm-eval ran with --limit {limit}: a plumbing check, not a publishable score"
            )
        return QualityRecord(
            lm_eval=result.scores,
            lm_eval_limit=int(limit) if limit else None,
            scored_by_engine=f"lm-eval/{backend}",
        )

    # -- quality: llama.cpp native full-vocabulary KL (GGUF) ----------------------------

    def _quality_llamacpp(self, model_cfg: Any, eval_cfg: Any, stream: Any) -> QualityRecord:
        """GGUF quality via llama-perplexity's base-file workflow (§5.2.3).

        The F16 GGUF writes the base (its full-vocabulary log-probs); every other GGUF is
        scored against it. The scored token ids are read back from the base header and
        compared with the hashed reference-tokenizer stream, so the identical-stream
        requirement (§6.6) is checked rather than assumed.
        """
        assert self.engine is not None
        spec = self.spec
        base_artifact = _f16_gguf_for(spec)
        base_sha = logits_cache.artifact_sha_from_manifest(base_artifact)
        if base_sha is None:
            raise EngineUnsupported(
                f"no provenance manifest for the F16 base {base_artifact}; produce it with "
                f"quantbench quantize --model {model_cfg.slug} --formats gguf:Q8_0"
            )
        key = logits_cache.llamacpp_base_key(model_cfg, eval_cfg, stream.meta, base_sha)
        is_base = spec.quant.format == "gguf" and spec.quant.variant.upper() == "F16"

        tokens: dict[str, Any] = {"text_path": str(stream.text_path), "ctx": key.ctx}
        if is_base:
            tokens |= {"write_kl_base": str(key.path), "chunks": key.chunks}
        else:
            if not key.path.exists():
                raise EngineUnsupported(
                    f"no llama.cpp KL base at {key.path}; build it once with: quantbench "
                    f"reference build --model {model_cfg.slug} --engine llama.cpp"
                )
            tokens["kl_base"] = str(key.path)

        score = self.engine.score(tokens, spec)
        header = logits_cache.read_kld_header(key.path)
        scored_ids = np.asarray(header.tokens, dtype=np.int64)
        reference_ids = np.asarray(stream.ids[: scored_ids.size], dtype=np.int64)
        verified = bool(
            scored_ids.size == reference_ids.size and (scored_ids == reference_ids).all()
        )
        if not verified:
            n = min(scored_ids.size, reference_ids.size)
            mismatch = int((scored_ids[:n] != reference_ids[:n]).sum())
            self.record.warnings.append(
                f"llama.cpp's token stream differs from the reference tokenizer's at "
                f"{mismatch} of {scored_ids.size} positions; quality numbers remain "
                "internally consistent (one base, one stream) but are not token-identical "
                "to the Transformers reference (§6.6)"
            )

        native = score.native_metrics
        # The reference PPL is the F16 run's own full-logit perplexity over these exact
        # positions, read from the base's metadata. llama-perplexity's "Mean PPL(base)" is
        # recomputed from the *stored* base log-probs, which are floored 16 nats below the
        # top logit; that floor over-credits rare targets (0.28% of positions on Qwen3-4B)
        # and lowers PPL(base) by ~0.6%, inflating every PPL(Q) - PPL(base). It is kept as
        # an artifact, never used as the reference. See docs/failures.md.
        if is_base:
            ppl_reference = score.ppl
        else:
            base_meta_path = key.path.with_suffix(".json")
            base_meta = (
                json.loads(base_meta_path.read_text(encoding="utf-8"))
                if base_meta_path.exists()
                else {}
            )
            ppl_reference = base_meta.get("ppl")
            if "ppl_base" in native:
                self.record.artifacts["ppl_base_native_floored"] = str(native["ppl_base"])
        record = QualityRecord(
            reference_id=key.reference_id,
            eval_stream_sha=eval_stream_mod.sha256_bytes(
                np.asarray(header.tokens, dtype="<i4").tobytes()
            ),
            ppl_wikitext2=score.ppl,
            ppl_reference=ppl_reference,
            ppl_config=PplConfig(
                ctx=header.n_ctx,
                stride=header.n_ctx,
                tokens=header.scored_positions,
                add_bos=False,
                scoring=(
                    f"llama-perplexity: positions {header.n_ctx // 2}..{header.n_ctx - 2} "
                    f"of each of {header.n_chunk} chunks"
                ),
            ),
            scored_by_engine=score.scored_by_engine,
            token_stream_verified=verified,
        )
        record.kl_div.topk = None  # full vocabulary
        record.kl_div.estimator = "llama-perplexity"
        if not is_base:
            record.kl_div.mean = native.get("kl_mean")
            record.kl_div.p90 = native.get("kl_p90")
            record.kl_div.p99 = native.get("kl_p99")
            record.kl_div.max = native.get("kl_max")
            record.top1_agreement = native.get("top1_agreement")
            self.record.artifacts["kl_mean_stderr"] = str(native.get("kl_mean_unc"))
            self.record.artifacts["top1_agreement_stderr"] = str(native.get("top1_agreement_unc"))
            self.record.warnings.append(
                "KL is llama-perplexity's full-vocabulary estimator against the F16 GGUF "
                "(within llama.cpp); compare it with top-k-reference KL from other engines "
                "only across that labelled boundary (§6.1)"
            )
        else:
            self.record.artifacts["llamacpp_kl_base"] = str(key.path)
            logits_cache.write_reference_meta_at(
                key.path.with_suffix(".json"),
                {
                    "reference_id": key.reference_id,
                    "key": key.__dict__,
                    "n_ctx": header.n_ctx,
                    "n_vocab": header.n_vocab,
                    "n_chunk": header.n_chunk,
                    "scored_positions": header.scored_positions,
                    "token_stream_verified": verified,
                    "ppl": score.ppl,
                    "bytes": key.path.stat().st_size,
                },
            )
        return record

    def _run_quality_synthetic(self) -> None:
        """Quality path for the mock engine.

        Exists so ``--dry-run`` and CI exercise the quality plumbing without a tokenizer,
        an evaluation corpus, or a reference cache. Everything it produces is fabricated
        and the record is marked ``synthetic``.
        """
        assert self.engine is not None
        window_start = self._sampler_clock()
        score = self.engine.score({"synthetic": True}, self.spec)
        window_end = self._sampler_clock()
        native = score.native_metrics
        self.record.quality = QualityRecord(
            reference_id="synthetic-reference",
            eval_stream_sha="synthetic",
            ppl_wikitext2=score.ppl,
            ppl_config=PplConfig(ctx=2048, stride=2048, tokens=score.n_positions, add_bos=True),
            scored_by_engine=score.scored_by_engine,
        )
        self.record.quality.kl_div.mean = native.get("kl_mean")
        self.record.quality.kl_div.p90 = native.get("kl_p90")
        self.record.quality.kl_div.p99 = native.get("kl_p99")
        self.record.quality.kl_div.max = native.get("kl_max")
        self.record.quality.kl_div.tail_mass_estimated = True
        self.record.quality.top1_agreement = native.get("top1_agreement")
        self.record.warnings.append("synthetic quality values from the mock engine")
        self._finalize_power(window_start, window_end, generated_tokens=None)
        self._finalize_memory([])

    # -- finalisation -------------------------------------------------------------------

    def _check_token_counts(self, reports: list[GenReport]) -> None:
        """§6.5: every run must generate exactly N tokens, or throughput is inflated."""
        want = self.spec.workload.generate_tokens
        for report in reports:
            for seq in report.sequences:
                if seq.generated_tokens != want:
                    self.record.warnings.append(
                        f"expected {want} generated tokens, got {seq.generated_tokens} "
                        f"(finish_reason={seq.finish_reason}); throughput may be inflated"
                    )
                    return

    def _finalize_memory(self, reports: list[GenReport]) -> None:
        mem = self.record.memory
        mem.vram_peak_nvsmi_mb = peak_process_mb(self.sampler.samples)
        for report in reports:
            if report.vram_kv_mb is not None:
                mem.vram_kv_mb = report.vram_kv_mb
            if report.vram_peak_allocated_mb is not None:
                mem.vram_peak_allocated_mb = max(
                    mem.vram_peak_allocated_mb or 0.0, report.vram_peak_allocated_mb
                )
        mem.vram_accounting = vram_accounting_mode(self.sampler.samples)
        mem.vram_shared_peak_mb = peak_shared_mb(self.sampler.samples)
        self._check_spill()
        if mem.vram_peak_nvsmi_mb is not None:
            if mem.vram_accounting == "device_wide" and mem.vram_baseline_nvsmi_mb is not None:
                mem.vram_peak_attributed_mb = mem.vram_peak_nvsmi_mb - mem.vram_baseline_nvsmi_mb
                note = (
                    "VRAM is device-wide (no per-process NVML accounting on this driver "
                    f"model); {mem.vram_baseline_nvsmi_mb:.0f} MB was already in use by other "
                    "processes before load and is subtracted in vram_peak_attributed_mb"
                )
                if note not in self.record.warnings:
                    self.record.warnings.append(note)
            elif (mem.vram_accounting or "").startswith("per_process"):
                mem.vram_peak_attributed_mb = mem.vram_peak_nvsmi_mb
        self._finalize_bpw()

    def _check_spill(self) -> None:
        """A run whose processes spilled VRAM into system RAM did not fit (§5.3: OOM)."""
        from quantbench.metrics.memory import spill_suspected
        from quantbench.metrics.wddm import PRESSURE_MARGIN_MB, SPILL_THRESHOLD_MB

        shared = self.record.memory.vram_shared_peak_mb
        if shared is None or not spill_suspected(
            self.sampler.samples,
            self.record.environment.vram_total_mb,
            SPILL_THRESHOLD_MB,
            PRESSURE_MARGIN_MB,
        ):
            return
        message = (
            f"sysmem fallback: {shared:.0f} MiB of this run's GPU memory spilled to system RAM "
            "(Windows 'CUDA - Sysmem Fallback Policy'). It did not fit in VRAM; on a driver "
            "without fallback this is an OOM, and any timings taken while spilled are invalid."
        )
        if self.record.status is RunStatus.ok:
            self.record.status = RunStatus.oom
            self.record.error = message
        self.record.warnings.append(message)

    def _finalize_bpw(self) -> None:
        mem = self.record.memory
        artifact = self.spec.quant.artifact_path
        if not artifact or mem.weight_bytes_non_embedding is not None:
            return
        weights = artifact_weight_bytes(Path(artifact))
        if weights is None:
            return
        mem.weight_bytes_non_embedding = weights.non_embedding
        mem.effective_bpw = effective_bpw(
            weights.non_embedding, self.spec.model.params_non_embedding
        )

    def _finalize_power(
        self, window_start: float, window_end: float, generated_tokens: int | None
    ) -> None:
        samples = self.sampler.samples
        self.record.power = power_metrics.summarize_power(
            samples, generated_tokens=generated_tokens, window=(window_start, window_end)
        )
        fraction = self.record.power.throttled_fraction or 0.0
        if fraction >= power_metrics.THROTTLE_INVALID_FRACTION:
            self.record.status = RunStatus.throttled_invalid
            self.record.warnings.append(
                f"invalidating throttle active for {fraction:.0%} of the measured window"
            )
        elif self.record.power.throttled:
            self.record.warnings.append(
                f"throttling flagged ({', '.join(self.record.power.throttle_reasons)}) for "
                f"{fraction:.0%} of the measured window"
            )

    # -- helpers ------------------------------------------------------------------------

    def _poll_mb(self) -> float | None:
        if not self.sampler.available:
            return None
        sample = self.sampler.poll()
        return (
            sample.process_used_mb if sample.process_used_mb is not None else sample.device_used_mb
        )

    def _sampler_clock(self) -> float:
        return self.sampler.samples[-1].t_s if self.sampler.samples else 0.0

    def _prompt_stream_ids(self) -> np.ndarray | None:
        """Reuse the pinned evaluation stream for prompts when it already exists."""
        try:
            model_cfg = load_model(_slug(self.spec.model.name))
            eval_cfg = load_eval(self.spec.workload.eval_config or "wikitext2")
            directory = eval_stream_mod.stream_dir(model_cfg, eval_cfg)
            if (directory / "tokens.npy").exists():
                return eval_stream_mod.load_eval_stream(directory).ids
        except Exception:
            return None
        return None


def _f16_gguf_for(spec: RunSpec) -> Path:
    """The F16 GGUF that serves as llama.cpp's within-engine reference for this model."""
    from quantbench.config import load_format
    from quantbench.planner import artifact_path_for

    model_cfg = load_model(_slug(spec.model.name))
    path = artifact_path_for(load_format("gguf"), model_cfg, "F16")
    if path is None:
        raise EngineUnsupported("configs/formats/gguf.yaml has no artifact_pattern")
    return Path(path)


def _slug(model_name: str) -> str:
    return model_name.lower().replace(".", "-").replace("_", "-").replace(" ", "-")


def _calibration_is_disjoint(spec: RunSpec, eval_cfg: Any) -> bool | None:
    """§6.7: evaluating on the calibration distribution is the field's classic own goal."""
    calib = spec.quant.calibration_dataset
    if calib is None:
        return None
    forbidden = {str(x).lower() for x in eval_cfg.calibration_disjoint_from}
    return calib.lower() not in forbidden and eval_cfg.name.lower() not in calib.lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantbench.worker")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    spec = RunSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    record = Worker(spec).run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    # Exit code carries the outcome so the parent can classify even if the file is lost.
    print(json.dumps({"run_id": spec.run_id, "status": record.status.value}))
    return 0 if record.status is RunStatus.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
