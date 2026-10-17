"""The ``Engine`` protocol every backend implements (requirements.md §7.2).

Keeping this interface narrow is what makes the cross-engine comparison fair. Two rules
are load-bearing:

1. :meth:`Engine.generate` **must** return per-token arrival timestamps. Aggregate-only
   timing cannot produce ITL percentiles, and ITL tails are where quantization pathologies
   show up.
2. An engine that cannot implement :meth:`Engine.score` says so in :meth:`Engine.preflight`
   (``can_score=False``). The planner then routes quality evaluation for that format to a
   designated scoring engine and the record carries ``scored_by_engine``, so a
   quality/latency split across backends is visible in the data rather than hidden.

Engine modules import their heavy third-party dependencies lazily *inside* methods. The
CLI must stay usable on a machine where vLLM will not even import (§11.1).
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from quantbench.schema import RunSpec


class EngineError(RuntimeError):
    """Base class for engine failures the runner classifies into a ``RunStatus``."""


class EngineUnsupported(EngineError):
    """This engine cannot run this spec at all (missing binary, missing kernel, ...)."""


class EngineOOM(EngineError):
    """Out of VRAM. A first-class outcome, not a bug (§5.3)."""


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PreflightResult(Report):
    ok: bool
    reason: str | None = None
    can_generate: bool = True
    can_score: bool = False
    warnings: list[str] = Field(default_factory=list)


class LoadReport(Report):
    load_time_s: float
    vram_weights_mb: float | None = None
    weight_file_bytes: int | None = None
    kv_cache_mb_reported: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SequenceTiming(Report):
    """One generated sequence.

    ``token_timestamps_ms`` are arrival times on the worker's monotonic clock, same origin
    as ``submit_ms``. Exactly ``generated_tokens`` entries.
    """

    submit_ms: float
    token_timestamps_ms: list[float]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str | None = None


class GenReport(Report):
    sequences: list[SequenceTiming]
    wall_time_ms: float
    submit_ms: float
    vram_kv_mb: float | None = None
    vram_peak_allocated_mb: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ScoreReport(Report):
    """Result of teacher-forced scoring over the fixed evaluation stream."""

    n_positions: int
    scored_by_engine: str
    artifact_path: str | None = None  # LogprobArtifact .npz
    ppl: float | None = None
    nll_sum: float | None = None
    native_metrics: dict[str, float] = Field(default_factory=dict)  # e.g. llama.cpp KL
    extra: dict[str, Any] = Field(default_factory=dict)


class EngineMetadata(Report):
    name: str
    version: str | None = None
    commit: str | None = None
    attention_backend: str | None = None
    kv_cache_dtype: str | None = None
    kernels: list[str] = Field(default_factory=list)
    binary_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Engine(Protocol):
    name: str

    def preflight(self, spec: RunSpec) -> PreflightResult: ...
    def load(self, spec: RunSpec) -> LoadReport: ...
    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport: ...
    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport: ...
    def teardown(self) -> None: ...
    def describe(self) -> EngineMetadata: ...


class BaseEngine:
    """Default implementations; subclasses override what they support.

    Deliberately concrete rather than abstract: an engine that implements only
    ``generate`` is a legitimate participant in the matrix, and the unimplemented methods
    raise :class:`EngineUnsupported`, which the runner records as an outcome.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._loaded = False

    def preflight(self, spec: RunSpec) -> PreflightResult:
        return PreflightResult(ok=True, can_generate=True, can_score=False)

    def load(self, spec: RunSpec) -> LoadReport:
        raise EngineUnsupported(f"{self.name}: load() not implemented")

    def generate(self, batch: list[list[int]], spec: RunSpec) -> GenReport:
        raise EngineUnsupported(f"{self.name}: generate() not implemented")

    def score(self, tokens: Any, spec: RunSpec) -> ScoreReport:
        raise EngineUnsupported(
            f"{self.name}: score() not implemented; route quality through a scoring engine"
        )

    def teardown(self) -> None:
        self._loaded = False

    def describe(self) -> EngineMetadata:
        return EngineMetadata(name=self.name)


#: Engine name -> import path. Import happens on demand, never at package import time.
ENGINE_REGISTRY: dict[str, str] = {
    "mock": "quantbench.engines.mock:MockEngine",
    "llama.cpp": "quantbench.engines.llamacpp:LlamaCppEngine",
    "vllm": "quantbench.engines.vllm:VllmEngine",
    "exllamav3": "quantbench.engines.exllamav3:ExLlamaV3Engine",
    "transformers": "quantbench.engines.transformers_hf:TransformersEngine",
}


def load_engine(name: str) -> BaseEngine:
    """Instantiate an engine by name, importing its module only now."""
    try:
        target = ENGINE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(ENGINE_REGISTRY))
        raise EngineUnsupported(f"unknown engine {name!r}; known engines: {known}") from exc
    module_path, _, class_name = target.partition(":")
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    engine: BaseEngine = engine_cls()
    return engine
