"""llama.cpp engine adapter.

llama.cpp is consumed as **pinned pre-built binaries invoked as subprocesses** (§11.1), not
as a Python binding: ``llama-server``, ``llama-perplexity``, ``llama-quantize`` and
``llama-imatrix`` are the maintained, well-defined surface. Note the renamed binaries —
older tutorials say ``main``/``server``/``quantize``.

* ``generate`` drives ``llama-server`` over streaming HTTP so that each token's arrival is
  timestamped individually (§7.2). Batch size N is N concurrent streaming requests against
  a server started with ``--parallel N``.
* ``score`` shells out to ``llama-perplexity``, whose ``--kl-divergence-base`` /
  ``--kl-divergence`` workflow is exactly the cached-reference design in §5.2.3. Native
  output is *normalised into this repo's schema* rather than reimplemented.

Binary locations may be overridden with ``LLAMA_CPP_BIN_DIR`` or the per-binary variables
``LLAMA_SERVER_BIN`` / ``LLAMA_PERPLEXITY_BIN``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
from quantbench.engines.server import ManagedServer, find_binary, free_port, stream_sse
from quantbench.paths import logs_dir
from quantbench.schema import RunSpec

KV_DTYPES = {"f16", "q8_0", "q4_0", "q5_1", "bf16"}

# Two banner formats exist in the wild:
#   old:  "version: 4321 (abc1234)"
#   new:  "version: 0.7.0-dev (build 11706, commit f74e2fa6a)"
# Releases are tagged by build number (b11706), so that is what gets recorded.
_VERSION_OLD_RE = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", re.IGNORECASE)
_VERSION_NEW_RE = re.compile(
    r"version:\s*\S+\s*\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)", re.IGNORECASE
)


def parse_version(text: str) -> tuple[str | None, str | None]:
    """``(build tag, commit)`` from a llama.cpp ``--version`` banner or server log."""
    for pattern in (_VERSION_NEW_RE, _VERSION_OLD_RE):
        if m := pattern.search(text):
            return f"b{m.group(1)}", m.group(2)
    return None, None


# KV allocation, as reported by the engine. Current builds log one line per device
# ("llama_kv_cache:      CUDA0 KV buffer size =   108.00 MiB", only at -lv >= 4); older
# builds logged a single "KV self size = N MiB".
_KV_BUFFER_RE = re.compile(r"llama_kv_cache:\s+(\S+)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB")
_KV_SELF_RE = re.compile(r"KV self size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
_FLASH_RE = re.compile(r"llama_context:\s+flash_attn\s*=\s*(\w+)")
_ARCHS_RE = re.compile(r"CUDA : ARCHS = ([\d,]+)")
_MODEL_BUF_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
_COMPUTE_BUF_RE = re.compile(r"sched_reserve:\s+(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB")


def _gpu_sum(pairs: list[tuple[str, str]]) -> float | None:
    """Sum MiB over GPU devices; host buffers (CPU, CPU_Mapped, CUDA_Host) are excluded."""
    gpu = [
        float(mib)
        for dev, mib in pairs
        if dev.upper().startswith("CUDA") and "HOST" not in dev.upper()
    ]
    return sum(gpu) if gpu else None


#: Server log verbosity. 4 is the lowest level that reports KV buffer sizes, and the
#: highest that emits no per-token debug lines (which would perturb the timings).
LOG_VERBOSITY = "4"


def parse_kv_mib(log_text: str) -> float | None:
    """Engine-reported KV cache allocation, summed across devices."""
    per_device = {dev: float(mib) for dev, mib in _KV_BUFFER_RE.findall(log_text)}
    gpu = {d: v for d, v in per_device.items() if not d.upper().startswith("CPU")}
    if gpu:
        return sum(gpu.values())
    if m := _KV_SELF_RE.search(log_text):
        return float(m.group(1))
    return None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def parse_load_facts(log_text: str) -> dict[str, object]:
    """What the server actually did, as opposed to what it was asked to do."""
    facts: dict[str, object] = {}
    if m := _OFFLOAD_RE.search(log_text):
        facts["layers_offloaded"] = f"{m.group(1)}/{m.group(2)}"
        facts["fully_offloaded"] = m.group(1) == m.group(2)
    if m := _FLASH_RE.search(log_text):
        facts["flash_attn_resolved"] = m.group(1)
    if m := _ARCHS_RE.search(log_text):
        facts["cuda_archs"] = m.group(1)
    if (kv := parse_kv_mib(log_text)) is not None:
        facts["kv_cache_mib_reported"] = kv
    if (weights := _gpu_sum(_MODEL_BUF_RE.findall(log_text))) is not None:
        facts["weights_gpu_mib_reported"] = weights
    host = [
        float(mib)
        for dev, mib in _MODEL_BUF_RE.findall(log_text)
        if not dev.upper().startswith("CUDA")
    ]
    if host:
        # e.g. token_embd kept host-side for the embedding lookup
        facts["weights_host_mib_reported"] = sum(host)
    if (compute := _gpu_sum(_COMPUTE_BUF_RE.findall(log_text))) is not None:
        facts["compute_gpu_mib_reported"] = compute
    return facts


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", re.IGNORECASE)
_MEAN_KLD_RE = re.compile(r"Mean\s+KLD:\s*([\d.eE+-]+)")
_MAX_KLD_RE = re.compile(r"Maximum\s+KLD:\s*([\d.eE+-]+)")
_P99_KLD_RE = re.compile(r"99\.0%\s+KLD:\s*([\d.eE+-]+)")
_P90_KLD_RE = re.compile(r"90\.0%\s+KLD:\s*([\d.eE+-]+)")
_TOP1_RE = re.compile(r"Same top p:\s*([\d.]+)")


def _bin(name: str, env_var: str) -> str:
    if (bin_dir := os.environ.get("LLAMA_CPP_BIN_DIR")) and not os.environ.get(env_var):
        candidate = Path(bin_dir) / (name + (".exe" if os.name == "nt" else ""))
        if candidate.exists():
            return str(candidate)
    return find_binary(name, env_var)


class LlamaCppEngine(BaseEngine):
    name = "llama.cpp"

    def __init__(self) -> None:
        super().__init__()
        self.server: ManagedServer | None = None
        self._meta: dict[str, Any] = {}
        self._model_path: Path | None = None
        self._load_time_s: float = 0.0

    # -- lifecycle ---------------------------------------------------------------------

    def preflight(self, spec: RunSpec) -> PreflightResult:
        warnings: list[str] = []
        try:
            server_bin = _bin("llama-server", "LLAMA_SERVER_BIN")
        except EngineUnsupported as exc:
            return PreflightResult(ok=False, reason=str(exc), can_generate=False)

        artifact = spec.quant.artifact_path
        if not artifact:
            return PreflightResult(
                ok=False, reason="quant.artifact_path is unset", can_generate=False
            )
        if not Path(artifact).exists():
            return PreflightResult(
                ok=False, reason=f"GGUF artifact missing: {artifact}", can_generate=False
            )
        kv = (spec.engine.kv_cache_dtype or "f16").lower()
        if kv not in KV_DTYPES:
            return PreflightResult(ok=False, reason=f"unsupported kv_cache_dtype {kv!r}")
        fa_requested = (spec.engine.attention_backend or "").lower() in {"cuda-fa", "fa", "flash"}
        if kv != "f16" and not (
            fa_requested or any(f.startswith("-fa") for f in spec.engine.flags)
        ):
            warnings.append(
                "quantized KV cache in llama.cpp generally requires flash attention (-fa); "
                "the run will record whatever the server reports"
            )
        try:
            can_score = bool(_bin("llama-perplexity", "LLAMA_PERPLEXITY_BIN"))
        except EngineUnsupported:
            can_score = False
            warnings.append("llama-perplexity not found; quality must be routed elsewhere")
        self._meta["server_bin"] = server_bin
        return PreflightResult(ok=True, can_generate=True, can_score=can_score, warnings=warnings)

    def _server_cmd(self, spec: RunSpec, port: int) -> list[str]:
        w, e = spec.workload, spec.engine
        ngl = "all" if e.n_gpu_layers is None or e.n_gpu_layers < 0 else str(e.n_gpu_layers)
        cmd = [
            _bin("llama-server", "LLAMA_SERVER_BIN"),
            "-m",
            str(spec.quant.artifact_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            ngl,
            # --fit defaults to "on" since ~b11xxx: it silently shrinks ctx / offload to
            # make a config fit. That would turn an OOM (a result, §5.3) into a different,
            # unrecorded experiment, so it is always off (§3.2: never silently fall back).
            "--fit",
            "off",
            "-c",
            str((w.prompt_tokens + w.generate_tokens) * max(1, w.batch_size)),
            "-np",
            str(max(1, w.batch_size)),
            "-b",
            str(max(512, min(w.prompt_tokens, 4096))),
            "--seed",
            str(w.seed),
            # Measured iterations repeat identical prompts; any prompt cache (slot-level or
            # the host-RAM cache) would delete prefill from the measurement.
            "--no-cache-prompt",
            "-cram",
            "0",
            "--no-webui",
            "--metrics",
            "-lv",
            LOG_VERBOSITY,
        ]
        kv = (e.kv_cache_dtype or "f16").lower()
        cmd += ["-ctk", kv, "-ctv", kv]
        backend = (e.attention_backend or "auto").lower()
        if backend in {"cuda-fa", "fa", "flash"}:
            cmd += ["-fa", "on"]
        elif backend in {"cuda-nofa", "nofa"}:
            cmd += ["-fa", "off"]
        else:
            cmd += ["-fa", "auto"]  # the resolved choice is read back from the log
        cmd += list(e.flags)
        return cmd

    def load(self, spec: RunSpec) -> LoadReport:
        port = free_port()
        log_path = logs_dir() / f"llamacpp-{spec.run_id}.log"
        self.server = ManagedServer(
            cmd=self._server_cmd(spec, port),
            port=port,
            health_path="/health",
            startup_timeout_s=float(spec.protocol.timeout_s),
            log_path=log_path,
        )
        t0 = time.perf_counter()
        self.server.start()
        try:
            self.server.wait_healthy()
        except EngineUnsupported:
            self.server.classify_exit()  # may raise EngineOOM with the real reason
            raise
        self._load_time_s = time.perf_counter() - t0
        self._model_path = Path(str(spec.quant.artifact_path))
        self._meta["attention_backend"] = spec.engine.attention_backend or "auto"
        self._meta["kv_cache_dtype"] = (spec.engine.kv_cache_dtype or "f16").lower()
        self._loaded = True

        log_text = self.server.tail(200_000)
        version, commit = parse_version(log_text)
        if version:
            self._meta["version"], self._meta["commit"] = version, commit
        facts = parse_load_facts(log_text)
        self._meta.update(facts)
        self._meta["cudart"] = (
            sorted(
                p.name for p in Path(self._meta.get("server_bin", "")).parent.glob("cudart*.dll")
            )
            or None
        )
        if facts.get("fully_offloaded") is False:
            raise EngineUnsupported(
                f"llama.cpp offloaded only {facts['layers_offloaded']} layers with -ngl all; "
                "a partially CPU-resident model is a different experiment"
            )
        kv_mb = parse_kv_mib(log_text)
        return LoadReport(
            load_time_s=self._load_time_s,
            # Engine-reported GPU weight buffers (§5.3 "measured or engine-reported"). A
            # device-wide NVML delta is too noisy under WDDM, where other processes share
            # the number; the worker falls back to that delta only if the log lacks this.
            vram_weights_mb=_as_float(facts.get("weights_gpu_mib_reported")),
            weight_file_bytes=self._model_path.stat().st_size
            if self._model_path.exists()
            else None,
            kv_cache_mb_reported=kv_mb,
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
            raise EngineUnsupported("llama.cpp: generate() called before load()")
        url = self.server.base_url + "/completion"
        w = spec.workload
        t0 = time.perf_counter()
        submit_ms = 0.0

        def one(prompt_ids: list[int]) -> SequenceTiming:
            payload: dict[str, Any] = {
                "prompt": prompt_ids,
                "n_predict": w.generate_tokens,
                "temperature": w.temperature,
                "top_k": 1 if w.temperature == 0 else 40,
                "seed": w.seed,
                "stream": True,
                "cache_prompt": False,
                # §6.5: fixed token count, otherwise early EOS silently inflates throughput.
                "ignore_eos": w.ignore_eos,
                "n_probs": 0,
            }
            stamps: list[float] = []
            finish: str | None = None
            for tok in stream_sse(url, payload, t0=t0, timeout=float(spec.protocol.timeout_s)):
                if tok.text:
                    stamps.append(tok.t_ms)
                if tok.finish_reason:
                    finish = tok.finish_reason
            return SequenceTiming(
                submit_ms=submit_ms,
                token_timestamps_ms=stamps,
                prompt_tokens=len(prompt_ids),
                generated_tokens=len(stamps),
                finish_reason=finish,
            )

        try:
            with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
                sequences = list(pool.map(one, batch))
        except Exception as exc:
            if self.server is not None:
                text = self.server.tail(8000).lower()
                if "out of memory" in text or "failed to allocate" in text:
                    raise EngineOOM(self.server.tail(2000)) from exc
            raise
        wall_ms = (time.perf_counter() - t0) * 1000.0
        kv_mb = parse_kv_mib(self.server.tail(400_000)) if self.server is not None else None
        return GenReport(
            sequences=sequences,
            wall_time_ms=wall_ms,
            submit_ms=submit_ms,
            vram_kv_mb=kv_mb,
            extra={"n_requests": len(batch)},
        )

    # -- scoring -----------------------------------------------------------------------

    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport:
        """Run ``llama-perplexity`` and normalise its output into this repo's schema.

        ``tokens`` is a mapping with ``text_path`` (the evaluation stream as raw text) and
        optionally ``kl_base`` (a reference file produced by ``--kl-divergence-base``). The
        PPL chunking parameters come from the eval config and are echoed into the record.
        """
        binary = _bin("llama-perplexity", "LLAMA_PERPLEXITY_BIN")
        text_path = tokens["text_path"]
        ctx = int(tokens.get("ctx", 2048))
        cmd = [
            binary,
            "-m",
            str(spec.quant.artifact_path),
            "-f",
            str(text_path),
            "-c",
            str(ctx),
            "-ngl",
            str(spec.engine.n_gpu_layers if spec.engine.n_gpu_layers is not None else -1),
            "--seed",
            str(spec.workload.seed),
        ]
        if kl_base := tokens.get("kl_base"):
            cmd += ["--kl-divergence-base", str(kl_base), "--kl-divergence"]
        elif out_base := tokens.get("write_kl_base"):
            cmd += ["--kl-divergence-base", str(out_base)]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec.protocol.timeout_s,
            check=False,
        )
        blob = proc.stdout + proc.stderr
        if proc.returncode != 0:
            if "out of memory" in blob.lower():
                raise EngineOOM(blob[-2000:])
            raise EngineUnsupported(
                f"llama-perplexity failed (rc={proc.returncode}): {blob[-2000:]}"
            )

        native: dict[str, float] = {}
        for key, pattern in (
            ("kl_mean", _MEAN_KLD_RE),
            ("kl_max", _MAX_KLD_RE),
            ("kl_p99", _P99_KLD_RE),
            ("kl_p90", _P90_KLD_RE),
            ("top1_agreement", _TOP1_RE),
        ):
            if m := pattern.search(blob):
                native[key] = float(m.group(1))
        if "top1_agreement" in native and native["top1_agreement"] > 1.0:
            native["top1_agreement"] /= 100.0  # llama.cpp prints a percentage

        ppl = float(m.group(1)) if (m := _PPL_RE.search(blob)) else None
        return ScoreReport(
            n_positions=int(tokens.get("n_positions", 0)),
            scored_by_engine=self.name,
            ppl=ppl,
            native_metrics=native,
            extra={"cmd": cmd, "stdout_tail": blob[-4000:]},
        )

    # -- metadata ----------------------------------------------------------------------

    def describe(self) -> EngineMetadata:
        version = self._meta.get("version")
        if version is None:
            try:
                proc = subprocess.run(
                    [_bin("llama-server", "LLAMA_SERVER_BIN"), "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    check=False,
                )
                version, commit = parse_version(proc.stdout + proc.stderr)
                if commit:
                    self._meta["commit"] = commit
            except Exception:
                version = None
        return EngineMetadata(
            name=self.name,
            version=version,
            commit=self._meta.get("commit"),
            attention_backend=self._meta.get("attention_backend"),
            kv_cache_dtype=self._meta.get("kv_cache_dtype"),
            binary_path=self._meta.get("server_bin"),
            extra={
                k: v
                for k, v in self._meta.items()
                if k not in {"version", "commit", "attention_backend", "kv_cache_dtype"}
            },
        )
