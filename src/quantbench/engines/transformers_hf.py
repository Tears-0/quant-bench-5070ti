"""HuggingFace Transformers engine.

Roles in the matrix:

* the **BF16/FP16 reference** (Tier A live on-GPU; Tier B/C with CPU offload to build the
  cached reference logits, §5.2);
* the **NF4 / bitsandbytes convenience baseline** (§4.2);
* the **fallback INT4 path** if vLLM cannot be made to work on ``sm_120`` (§12.1), reading
  GPTQ/AWQ checkpoints through ``GPTQModel``/compressed-tensors kernels;
* the **designated scoring engine** for formats whose serving engine cannot score.

Decoding is a hand-written greedy loop rather than ``model.generate`` because per-token
arrival timestamps are mandatory (§7.2). Each token is timestamped after
``torch.cuda.synchronize()``, which is what makes the timestamp mean "the token existed",
at the cost of removing CPU/GPU overlap between steps. That cost applies uniformly to every
Transformers run and is stated in ``docs/methodology.md``.
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


class TransformersEngine(BaseEngine):
    name = "transformers"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.tokenizer: Any = None
        self._torch: Any = None
        self._meta: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------------------

    def preflight(self, spec: RunSpec) -> PreflightResult:
        warnings: list[str] = []
        try:
            import torch
            import transformers  # noqa: F401
        except ImportError as exc:
            return PreflightResult(
                ok=False,
                reason=f"install requirements/hf.txt (see docs/install/transformers.md): {exc}",
                can_generate=False,
            )
        if not torch.cuda.is_available():
            return PreflightResult(ok=False, reason="CUDA is not available", can_generate=False)
        arch_list = list(torch.cuda.get_arch_list())
        if "sm_120" not in arch_list:
            return PreflightResult(
                ok=False,
                reason=(
                    "this torch build has no sm_120 kernels "
                    f"(arch list: {arch_list}); install a cu128+ wheel — see docs/install/"
                ),
                can_generate=False,
            )
        if spec.quant.format == "nf4":
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                return PreflightResult(ok=False, reason=f"bitsandbytes missing: {exc}")
            warnings.append(
                "bitsandbytes NF4 is included as a convenience baseline, not a fast path"
            )
        return PreflightResult(ok=True, can_generate=True, can_score=True, warnings=warnings)

    def _quantization_config(self, spec: RunSpec) -> Any:
        if spec.quant.format != "nf4":
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load(self, spec: RunSpec) -> LoadReport:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        source = spec.quant.artifact_path or spec.model.hf_repo
        offload = bool(spec.engine.extra.get("cpu_offload", False))
        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16 if spec.quant.format in {"bf16", "fp16"} else "auto",
            "device_map": "auto" if offload else {"": 0},
            "trust_remote_code": bool(spec.engine.extra.get("trust_remote_code", False)),
        }
        if spec.model.revision and not spec.quant.artifact_path:
            kwargs["revision"] = spec.model.revision
        if (qc := self._quantization_config(spec)) is not None:
            kwargs["quantization_config"] = qc
        if backend := spec.engine.attention_backend:
            # "sdpa" | "eager" | "flash_attention_2"; recorded as a run variable (§3.2).
            kwargs["attn_implementation"] = backend

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except torch.OutOfMemoryError as exc:
            raise EngineOOM(str(exc)) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineOOM(str(exc)) from exc
            raise EngineUnsupported(str(exc)) from exc
        self.model.eval()
        load_time = time.perf_counter() - t0
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model.hf_repo, revision=spec.model.revision
        )
        weights_mb = (torch.cuda.memory_allocated() - before) / MB
        self._meta["dtype"] = str(next(self.model.parameters()).dtype)
        self._meta["attn_implementation"] = getattr(self.model.config, "_attn_implementation", None)
        self._loaded = True

        weight_bytes: int | None = None
        if spec.quant.artifact_path and (p := Path(spec.quant.artifact_path)).exists():
            weight_bytes = (
                sum(f.stat().st_size for f in p.glob("*.safetensors"))
                if p.is_dir()
                else p.stat().st_size
            ) or None
        return LoadReport(
            load_time_s=load_time,
            vram_weights_mb=weights_mb,
            weight_file_bytes=weight_bytes,
            extra=dict(self._meta),
        )

    def teardown(self) -> None:
        if self._torch is not None:
            del self.model
            self.model = None
            self._torch.cuda.empty_cache()
        self._loaded = False

    # -- generation --------------------------------------------------------------------

    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport:
        torch = self._torch
        if torch is None or self.model is None:
            raise EngineUnsupported("transformers: generate() called before load()")
        if spec.workload.temperature != 0.0:
            raise EngineUnsupported("only greedy decoding is implemented (§6.5)")

        input_ids = torch.tensor(batch, dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones_like(input_ids)
        n_gen = spec.workload.generate_tokens
        torch.cuda.reset_peak_memory_stats()
        kv_before = torch.cuda.memory_allocated()

        stamps: list[float] = []
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                stamps.append((time.perf_counter() - t0) * 1000.0)
                kv_after_prefill = torch.cuda.memory_allocated()

                for _ in range(n_gen - 1):
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_tok)], dim=-1)
                    out = self.model(
                        input_ids=next_tok,
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    torch.cuda.synchronize()
                    stamps.append((time.perf_counter() - t0) * 1000.0)
        except torch.OutOfMemoryError as exc:
            raise EngineOOM(str(exc)) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise EngineOOM(str(exc)) from exc
            raise

        wall_ms = (time.perf_counter() - t0) * 1000.0
        # Every sequence in the batch is decoded in lockstep, so they share arrival times.
        sequences = [
            SequenceTiming(
                submit_ms=0.0,
                token_timestamps_ms=list(stamps),
                prompt_tokens=len(batch[i]),
                generated_tokens=len(stamps),
                finish_reason="length",
            )
            for i in range(len(batch))
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
        """Teacher-forced scoring over the fixed evaluation stream.

        ``tokens`` carries ``ids`` (the pinned token stream), the chunking parameters, an
        output path, and optionally ``ref_topk_ids`` so this model's log-probs can be
        recorded on the *reference's* support — which is what makes truncated KL exact
        rather than approximate (see :mod:`quantbench.metrics.quality`).
        """
        import torch

        from quantbench.metrics.quality import LogprobArtifact

        if self.model is None:
            raise EngineUnsupported("transformers: score() called before load()")

        ids = np.asarray(tokens["ids"], dtype=np.int64)
        ctx = int(tokens.get("ctx", 2048))
        stride = int(tokens.get("stride", ctx))
        topk = int(tokens.get("topk", 128))
        add_bos = bool(tokens.get("add_bos", True))
        ref_topk_ids = tokens.get("ref_topk_ids")
        bos_id = getattr(self.tokenizer, "bos_token_id", None)

        topk_ids: list[np.ndarray[Any, Any]] = []
        topk_lps: list[np.ndarray[Any, Any]] = []
        tgt_ids: list[np.ndarray[Any, Any]] = []
        tgt_lps: list[np.ndarray[Any, Any]] = []
        aligned: list[np.ndarray[Any, Any]] = []
        position = 0

        with torch.inference_mode():
            for start in range(0, max(len(ids) - 1, 0), stride):
                chunk = ids[start : start + ctx]
                if len(chunk) < 2:
                    break
                arr = chunk.tolist()
                if add_bos and bos_id is not None:
                    arr = [bos_id, *arr]
                inp = torch.tensor([arr], dtype=torch.long, device=self.model.device)
                logits = self.model(input_ids=inp).logits[0].float()
                logprobs = torch.log_softmax(logits[:-1], dim=-1)
                targets = inp[0, 1:]

                vals, idx = torch.topk(logprobs, k=topk, dim=-1)
                topk_ids.append(idx.cpu().numpy())
                topk_lps.append(vals.cpu().numpy())
                tgt_ids.append(targets.cpu().numpy())
                tgt_lps.append(logprobs.gather(1, targets[:, None]).squeeze(1).cpu().numpy())

                if ref_topk_ids is not None:
                    n = logprobs.shape[0]
                    ref_slice = torch.tensor(
                        np.asarray(ref_topk_ids)[position : position + n],
                        dtype=torch.long,
                        device=logprobs.device,
                    )
                    aligned.append(logprobs.gather(1, ref_slice).cpu().numpy())
                position += logprobs.shape[0]

        if not topk_ids:
            raise EngineUnsupported("evaluation stream is too short to score")

        artifact = LogprobArtifact(
            topk_ids=np.concatenate(topk_ids).astype(np.int64),
            topk_logprobs=np.concatenate(topk_lps).astype(np.float64),
            target_ids=np.concatenate(tgt_ids).astype(np.int64),
            target_logprobs=np.concatenate(tgt_lps).astype(np.float64),
            ref_aligned_logprobs=np.concatenate(aligned).astype(np.float64) if aligned else None,
            meta={
                "engine": self.name,
                "model": spec.model.name,
                "quant": f"{spec.quant.format}:{spec.quant.variant}",
                "ctx": ctx,
                "stride": stride,
                "add_bos": add_bos,
                "topk": topk,
                "eval_stream_sha": tokens.get("eval_stream_sha"),
            },
        )
        out_path = Path(tokens["out_path"])
        artifact.save(out_path)
        nll = float(-artifact.target_logprobs.sum())
        return ScoreReport(
            n_positions=artifact.n_positions,
            scored_by_engine=self.name,
            artifact_path=str(out_path),
            ppl=float(np.exp(nll / artifact.n_positions)),
            nll_sum=nll,
        )

    # -- metadata ----------------------------------------------------------------------

    def describe(self) -> EngineMetadata:
        version = None
        torch_version = None
        try:
            import torch
            import transformers

            version = transformers.__version__
            torch_version = torch.__version__
        except ImportError:
            pass
        return EngineMetadata(
            name=self.name,
            version=version,
            attention_backend=self._meta.get("attn_implementation"),
            extra={"torch": torch_version, **self._meta},
        )
