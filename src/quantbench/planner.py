"""Config -> shuffled, resumable run queue (requirements.md §6.2, §7.3).

The ordering rule is not cosmetic. GPU temperature drifts monotonically over a long
session, so running all Q4 configs and then all Q5 configs makes format *perfectly
confounded* with thermal state. The queue is therefore shuffled with a recorded seed, or
round-robined across format families, and the profile records which.

Resume is keyed on ``(spec_hash, environment_hash)``: re-running skips specs already
present in ``results/raw/`` unless ``--force``. An overnight run must survive a crash at
hour six.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quantbench import __version__
from quantbench.config import (
    CellConfig,
    FormatConfig,
    ModelConfig,
    ProfileConfig,
    ProtocolConfig,
    load_format,
    load_model,
    load_profile,
    parse_format_ref,
)
from quantbench.schema import (
    EngineSpec,
    ModelRef,
    ProtocolSpec,
    QuantSpec,
    RunSpec,
    WorkloadSpec,
)


class PlanningError(ValueError):
    """A profile asked for something the configs cannot express."""


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    seed: int
    order: str
    created_utc: str
    quantbench_version: str
    entries: list[RunSpec]
    #: format ref -> engine that will produce the quality numbers when the serving engine
    #: cannot score. Surfaced here so the split is visible before the run, not after.
    scoring_routes: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-").replace("--", "-")


def make_run_id(
    model: ModelConfig,
    fmt: FormatConfig,
    variant: str,
    engine: str,
    workload: WorkloadSpec,
    engine_spec: EngineSpec,
) -> str:
    """Human-readable, collision-free within a plan.

    Repetitions are *not* part of the id: one run record aggregates
    ``protocol.measured_iters`` repetitions and retains their samples, so there is no
    ``__rN`` suffix (see docs/open-questions.md).
    """
    parts = [
        slugify(model.name),
        f"{slugify(fmt.family)}-{slugify(variant)}",
        slugify(engine),
        f"b{workload.batch_size}",
        f"ctx{workload.prompt_tokens}",
        f"gen{workload.generate_tokens}",
        workload.kind,
    ]
    if engine_spec.kv_cache_dtype and engine_spec.kv_cache_dtype not in {"auto", "f16", "fp16"}:
        parts.append(f"kv-{slugify(engine_spec.kv_cache_dtype)}")
    if engine_spec.attention_backend and engine_spec.attention_backend != "auto":
        parts.append(f"attn-{slugify(engine_spec.attention_backend)}")
    return "__".join(parts)


def artifact_path_for(fmt: FormatConfig, model: ModelConfig, variant: str) -> str | None:
    if not fmt.artifact_pattern:
        return None
    return fmt.artifact_pattern.format(
        model_slug=model.slug,
        model_name=model.name,
        variant=variant,
        variant_lower=variant.lower(),
        variant_slug=slugify(variant),
        family=fmt.family,
    )


def _protocol(cell: CellConfig, profile: ProfileConfig) -> ProtocolSpec:
    cfg: ProtocolConfig = cell.protocol or profile.protocol
    return ProtocolSpec(**cfg.model_dump())


def expand_cell(
    cell: CellConfig,
    profile: ProfileConfig,
    notes: list[str],
    scoring_routes: dict[str, str],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    protocol = _protocol(cell, profile)

    for model_name, format_ref in itertools.product(cell.models, cell.formats):
        model = load_model(model_name)
        family, variant_id = parse_format_ref(format_ref)
        fmt = load_format(family)
        variant = fmt.variant(variant_id)

        engines = cell.engines or fmt.default_engines
        for engine_name in engines:
            if engine_name not in fmt.supported_engines and engine_name != "mock":
                raise PlanningError(
                    f"cell {cell.name!r}: engine {engine_name!r} does not support format "
                    f"{family!r} (supported: {fmt.supported_engines})"
                )
            if fmt.scoring_engine and fmt.scoring_engine != engine_name:
                scoring_routes[f"{family}:{variant_id}@{engine_name}"] = fmt.scoring_engine
            if fmt.conditional:
                notes.append(
                    f"{family}:{variant_id} is marked conditional — kernel support on "
                    "sm_120 is unverified; a failure here is a finding (§3.2)"
                )

            kv_dtypes = cell.kv_cache_dtypes or fmt.kv_cache_dtypes[:1]
            backends = cell.attention_backends or fmt.attention_backends[:1]

            for sweep, kv, backend in itertools.product(cell.workloads, kv_dtypes, backends):
                if kv not in fmt.kv_cache_dtypes:
                    raise PlanningError(
                        f"cell {cell.name!r}: kv_cache_dtype {kv!r} not supported by {family!r}"
                    )
                for batch, prompt, gen in itertools.product(
                    sweep.batch_size, sweep.prompt_tokens, sweep.generate_tokens
                ):
                    workload = WorkloadSpec(
                        kind=sweep.kind,
                        batch_size=batch,
                        prompt_tokens=prompt,
                        generate_tokens=gen if sweep.kind != "quality" else 0,
                        ignore_eos=sweep.ignore_eos,
                        temperature=0.0,
                        seed=profile.seed,
                        eval_config=sweep.eval_config,
                    )
                    engine_spec = EngineSpec(
                        name=engine_name,
                        attention_backend=backend,
                        kv_cache_dtype=kv,
                        n_gpu_layers=-1 if engine_name == "llama.cpp" else None,
                    )
                    quant = QuantSpec(
                        format=family,
                        variant=variant_id,
                        bits_nominal=variant.bits_nominal,
                        imatrix=variant.imatrix,
                        calibration_dataset=variant.calibration_dataset,
                        artifact_path=artifact_path_for(fmt, model, variant_id),
                        producer=fmt.producer,
                    )
                    specs.append(
                        RunSpec(
                            run_id=make_run_id(
                                model, fmt, variant_id, engine_name, workload, engine_spec
                            ),
                            model=ModelRef(
                                name=model.name,
                                hf_repo=model.hf_repo,
                                revision=model.revision,
                                tier=model.tier,
                                params_total=model.params_total,
                                params_non_embedding=model.params_non_embedding,
                                tokenizer_sha256=model.tokenizer_sha256,
                            ),
                            quant=quant,
                            engine=engine_spec,
                            workload=workload,
                            protocol=protocol,
                        )
                    )
    return specs


def order_specs(specs: list[RunSpec], order: str, seed: int) -> list[RunSpec]:
    """Break the confound between execution order and format (§6.2)."""
    if order == "as_listed":
        return list(specs)
    if order == "shuffle":
        rng = random.Random(seed)
        shuffled = list(specs)
        rng.shuffle(shuffled)
        return shuffled
    if order == "round_robin":
        buckets: dict[str, list[RunSpec]] = defaultdict(list)
        for spec in specs:
            buckets[f"{spec.quant.format}:{spec.quant.variant}"].append(spec)
        rng = random.Random(seed)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        out: list[RunSpec] = []
        for row in itertools.zip_longest(*[buckets[k] for k in sorted(buckets)]):
            out.extend(spec for spec in row if spec is not None)
        return out
    raise PlanningError(f"unknown order {order!r}")


def build_plan(
    profile_name: str, engine_override: str | None = None, kind: str | None = None
) -> Plan:
    profile: ProfileConfig = load_profile(profile_name)
    notes: list[str] = []
    scoring_routes: dict[str, str] = {}

    specs: list[RunSpec] = []
    for cell in profile.cells:
        specs.extend(expand_cell(cell, profile, notes, scoring_routes))

    if kind:
        before = len(specs)
        specs = [s for s in specs if s.workload.kind == kind]
        notes.append(f"kept {len(specs)} of {before} specs with workload kind {kind!r}")

    if engine_override:
        specs = [
            spec.model_copy(
                update={"engine": spec.engine.model_copy(update={"name": engine_override})}
            )
            for spec in specs
        ]
        notes.append(f"engine overridden to {engine_override!r} for every spec")

    # Deduplicate on experiment identity, keeping the first occurrence's run_id.
    seen: set[str] = set()
    unique: list[RunSpec] = []
    for spec in specs:
        h = spec.hash()
        if h in seen:
            continue
        seen.add(h)
        unique.append(spec)
    if len(unique) != len(specs):
        notes.append(f"deduplicated {len(specs) - len(unique)} identical specs")

    ordered = order_specs(unique, profile.order, profile.seed)
    return Plan(
        profile=profile.name,
        seed=profile.seed,
        order=profile.order,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        quantbench_version=__version__,
        entries=ordered,
        scoring_routes=scoring_routes,
        notes=notes,
    )


def completed_keys(raw_dir: Path) -> set[tuple[str, str | None]]:
    """``(spec_hash, environment_hash)`` pairs already on disk."""
    import json

    keys: set[tuple[str, str | None]] = set()
    if not raw_dir.exists():
        return keys
    for path in raw_dir.glob("*.json"):
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "spec_hash" in payload:
            keys.add((payload["spec_hash"], payload.get("environment_hash")))
    return keys


def filter_resumable(
    entries: list[RunSpec], raw_dir: Path, environment_hash: str | None
) -> tuple[list[RunSpec], list[RunSpec]]:
    """Split a plan into (to_run, already_done) for the current environment."""
    done = completed_keys(raw_dir)
    todo: list[RunSpec] = []
    skipped: list[RunSpec] = []
    for spec in entries:
        key = (spec.hash(), environment_hash)
        if key in done or (spec.hash(), None) in done:
            skipped.append(spec)
        else:
            todo.append(spec)
    return todo, skipped
