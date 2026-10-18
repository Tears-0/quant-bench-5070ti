"""Versioned data schemas for run specifications and run records.

Every number this repository publishes is carried by a :class:`RunRecord` produced by the
harness. Analysis code validates records on load and refuses records whose ``spec_version``
it cannot interpret (requirements.md §8).

Two invariants are enforced here rather than by convention:

* ``RunRecord.synthetic`` marks records that did not come from real hardware (mock engine,
  ``--dry-run``). The analysis layer refuses to mix synthetic and measured records.
* Optional metrics are ``None``, never zero. A missing measurement renders as ``—``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SPEC_VERSION = 1
RECORD_VERSION = 1

WorkloadKind = Literal["latency", "quality", "memory-sweep"]


def canonical_json(payload: Any) -> str:
    """Stable JSON encoding used for every hash in the repo."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# RunSpec (input, §8.1)
# --------------------------------------------------------------------------------------


class ModelRef(Frozen):
    """A model, pinned hard enough to be reproducible.

    ``params_total``/``params_non_embedding`` are ``None`` until filled in by
    ``quantbench models inspect``, which reads them from the downloaded config/weights.
    They are never guessed: ``effective_bpw`` is reported as ``None`` when the
    non-embedding parameter count is unknown (§5.3).
    """

    name: str
    hf_repo: str
    revision: str | None = None
    tier: Literal["A", "B", "C", "D"] | None = None
    params_total: int | None = None
    params_non_embedding: int | None = None
    tokenizer_sha256: str | None = None


class QuantSpec(Frozen):
    format: str  # bf16 | gguf | awq | gptq | fp8 | nf4 | exl3
    variant: str  # Q4_K_M | INT4 | 4.0bpw | ...
    bits_nominal: float | None = None
    imatrix: bool = False
    imatrix_source: str | None = None
    calibration_dataset: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    producer: str | None = None
    producer_version: str | None = None
    publisher: str | None = None  # set when the artifact was downloaded, not produced here
    publisher_revision: str | None = None


class EngineSpec(Frozen):
    name: str  # llama.cpp | vllm | exllamav3 | transformers | mock
    version: str | None = None
    commit: str | None = None
    attention_backend: str | None = None  # recorded always (§3.2)
    kv_cache_dtype: str | None = None
    n_gpu_layers: int | None = None
    flags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class WorkloadSpec(Frozen):
    kind: WorkloadKind = "latency"
    batch_size: int = 1
    prompt_tokens: int = 512
    generate_tokens: int = 128
    ignore_eos: bool = True
    temperature: float = 0.0
    seed: int = 1337
    eval_config: str | None = None  # configs/eval/<name>.yaml, quality workloads only

    @field_validator("temperature")
    @classmethod
    def _greedy_for_quality(cls, v: float, info: Any) -> float:
        kind = info.data.get("kind")
        if kind == "quality" and v != 0.0:
            raise ValueError("quality workloads must use greedy decoding (temperature=0)")
        return v


class ProtocolSpec(Frozen):
    warmup_iters: int = 2
    measured_iters: int = 5
    cooldown_target_c: int | None = 50
    cooldown_timeout_s: int = 300
    clocks_locked: bool = False
    timeout_s: int = 1800
    sample_hz: float = 10.0  # NVML sampling; §5.3 needs >=5 Hz, §5.4 needs >=10 Hz


class RunSpec(Frozen):
    run_id: str
    spec_version: int = SPEC_VERSION
    model: ModelRef
    quant: QuantSpec
    engine: EngineSpec
    workload: WorkloadSpec
    protocol: ProtocolSpec = ProtocolSpec()

    def hash(self) -> str:
        """Identity of the *experiment*, independent of the human-readable run_id."""
        payload = self.model_dump(mode="json", exclude={"run_id"})
        return sha256_of(payload)


# --------------------------------------------------------------------------------------
# RunRecord (output, §8.2)
# --------------------------------------------------------------------------------------


class RunStatus(StrEnum):
    ok = "ok"
    oom = "oom"
    engine_unsupported = "engine_unsupported"
    build_failed = "build_failed"
    timeout = "timeout"
    crashed = "crashed"
    throttled_invalid = "throttled_invalid"


class Distribution(Mutable):
    """Robust summary of repeated measurements.

    Median and IQR are the headline pair (§5.1); mean/std are carried for completeness but
    the analysis layer never ranks on them alone.
    """

    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    mean: float | None = None
    std: float | None = None
    iqr: float | None = None
    min: float | None = None
    max: float | None = None
    n: int = 0
    samples: list[float] = Field(default_factory=list)


class EnvironmentRecord(Mutable):
    driver: str | None = None
    cuda_runtime: str | None = None
    torch: str | None = None
    torch_arch_list: list[str] = Field(default_factory=list)
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    vram_total_mb: int | None = None
    is_wsl: bool = False
    cpu: str | None = None
    ram_gb: float | None = None
    os: str | None = None
    python: str | None = None
    hostname: str | None = None
    nvml_driver: str | None = None
    timestamp_utc: str | None = None

    def hash(self) -> str:
        """Environment identity used with ``spec_hash`` for resume (§7.3).

        Deliberately excludes the timestamp and hostname: the same machine on the same
        stack yesterday is the same environment today.
        """
        payload = self.model_dump(mode="json", exclude={"timestamp_utc", "hostname", "gpu_uuid"})
        return sha256_of(payload)


class LatencyRecord(Mutable):
    ttft_ms: Distribution = Field(default_factory=Distribution)
    itl_ms: Distribution = Field(default_factory=Distribution)
    decode_tok_s: Distribution = Field(default_factory=Distribution)
    prefill_tok_s: Distribution = Field(default_factory=Distribution)
    e2e_latency_ms: Distribution = Field(default_factory=Distribution)
    throughput_tok_s_total: Distribution = Field(default_factory=Distribution)
    per_token_timestamps_ms: list[list[float]] = Field(default_factory=list)


class MemoryRecord(Mutable):
    vram_weights_mb: float | None = None
    vram_kv_mb: float | None = None
    vram_peak_allocated_mb: float | None = None
    vram_peak_nvsmi_mb: float | None = None
    vram_baseline_nvsmi_mb: float | None = None
    #: Peak VRAM attributable to this run: the process-level NVML peak where the driver
    #: supports per-process accounting, otherwise device-wide peak minus the pre-load
    #: baseline. ``vram_peak_nvsmi_mb`` stays the raw number that decides whether it fit.
    vram_peak_attributed_mb: float | None = None
    #: "per_process" (NVML), "per_process_wddm" (Windows GPU Process Memory counters) or
    #: "device_wide" (no per-process figure available, e.g. WSL2).
    vram_accounting: str | None = None
    #: Windows only: peak system memory mapped for this run's GPU work ("Shared Usage"):
    #: pinned host buffers and, under dedicated-memory pressure, spilled device allocations.
    #: Spill (shared memory *while* the device is nearly full) makes the run status ``oom``.
    vram_shared_peak_mb: float | None = None
    effective_bpw: float | None = None
    weight_file_bytes: int | None = None
    weight_bytes_non_embedding: int | None = None
    load_time_s: float | None = None


class PowerRecord(Mutable):
    power_avg_w: float | None = None
    power_peak_w: float | None = None
    energy_j: float | None = None
    tokens_per_joule: float | None = None
    sm_clock_avg_mhz: float | None = None
    gpu_temp_start_c: float | None = None
    gpu_temp_end_c: float | None = None
    throttle_reasons: list[str] = Field(default_factory=list)
    #: Thermal / hardware slowdown seen in the measured window (contamination).
    throttled: bool = False
    throttled_fraction: float | None = None
    #: Board power-limit regulation seen (normal boost behaviour; flagged, not excluded).
    power_capped: bool = False
    power_capped_fraction: float | None = None
    samples_n: int = 0
    sample_hz_effective: float | None = None


class PplConfig(Mutable):
    ctx: int
    stride: int
    tokens: int  # positions actually scored
    add_bos: bool  # *effective*: false for tokenizers without a BOS token (Qwen3)
    scoring: str | None = None  # which positions of each chunk were scored, by whom


class KLRecord(Mutable):
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None
    max: float | None = None
    # KL over a top-k truncated reference distribution carries a tail term; when the tail
    # mass had to be approximated rather than measured, say so in the record (§5.2).
    topk: int | None = None  # None = full-vocabulary KL
    tail_mass_estimated: bool = False
    estimator: str | None = None  # "topk-reference" | "llama-perplexity"


class QualityRecord(Mutable):
    reference_id: str | None = None
    eval_stream_sha: str | None = None
    ppl_wikitext2: float | None = None
    ppl_config: PplConfig | None = None
    #: Perplexity of the reference over the same scored positions, so PPL deltas never
    #: mix chunking conventions (§6.1).
    ppl_reference: float | None = None
    kl_div: KLRecord = Field(default_factory=KLRecord)
    top1_agreement: float | None = None
    top5_agreement: float | None = None
    lm_eval: dict[str, float | None] = Field(default_factory=dict)
    #: Set when lm-eval ran with --limit: a plumbing check, never a publishable score.
    lm_eval_limit: int | None = None
    longctx_retrieval: dict[str, float | None] = Field(default_factory=dict)
    # §7.2: when latency and quality came from different backends, the record says so.
    scored_by_engine: str | None = None
    #: The scored token ids were checked against the hashed reference-tokenizer stream.
    token_stream_verified: bool | None = None
    calibration_disjoint_from_eval: bool | None = None


class RunRecord(Mutable):
    record_version: int = RECORD_VERSION
    spec_hash: str
    environment_hash: str | None = None
    spec: RunSpec
    status: RunStatus
    synthetic: bool = False
    environment: EnvironmentRecord = Field(default_factory=EnvironmentRecord)
    engine_metadata: dict[str, Any] = Field(default_factory=dict)
    latency: LatencyRecord = Field(default_factory=LatencyRecord)
    memory: MemoryRecord = Field(default_factory=MemoryRecord)
    power: PowerRecord = Field(default_factory=PowerRecord)
    quality: QualityRecord = Field(default_factory=QualityRecord)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    stderr_tail: str | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.ok


class SchemaVersionError(ValueError):
    """Raised when a record on disk predates or postdates what this code understands."""


def load_record(payload: dict[str, Any]) -> RunRecord:
    """Validate a record dict, refusing versions this code cannot interpret (§8.2)."""
    version = payload.get("record_version")
    if version is None:
        raise SchemaVersionError("record is missing 'record_version'")
    if version != RECORD_VERSION:
        raise SchemaVersionError(
            f"record_version {version} != supported {RECORD_VERSION}; "
            "migrate the record or pin an older quantbench"
        )
    return RunRecord.model_validate(payload)
