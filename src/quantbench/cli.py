"""``quantbench`` command line (requirements.md §9.1).

    quantbench preflight
    quantbench models fetch --config configs/models/qwen3-8b.yaml
    quantbench quantize --model qwen3-8b --formats gguf:Q4_K_M,gguf:Q5_K_M,awq,exl3:4.0
    quantbench reference build --model qwen3-8b
    quantbench plan --profile core --out plan.json
    quantbench run --plan plan.json [--resume] [--dry-run]
    quantbench analyze --results results/ --out results/results.parquet
    quantbench plot --results results/results.parquet --out figures/
    quantbench report

No engine, and no torch, is imported by this module. ``quantbench preflight`` has to work
on the machine where the CUDA stack is broken — that is the whole point of it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantbench import __version__
from quantbench.config import (
    ModelConfig,
    list_configs,
    load_eval,
    load_hardware,
    load_model,
    parse_format_ref,
)
from quantbench.logging_setup import setup_logging
from quantbench.paths import (
    ensure_dirs,
    figures_dir,
    models_dir,
    raw_dir,
    reference_cache_dir,
    results_dir,
)
from quantbench.planner import Plan, build_plan
from quantbench.runner import Runner, RunnerOptions

if TYPE_CHECKING:
    import pandas as pd

    # producers pull in heavy tooling and are imported at call time only
    from quantbench.quantize.base import QuantizeResult

if TYPE_CHECKING:  # imported lazily at call time; the producers pull in heavy tooling
    from quantbench.quantize.base import QuantizeResult

app = typer.Typer(
    name="quantbench",
    help="Consumer-GPU quantization & inference benchmark suite (RTX 5070 Ti, sm_120).",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Fetch and inspect model artifacts.", no_args_is_help=True)
reference_app = typer.Typer(help="Build and inspect the FP16 reference.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(reference_app, name="reference")

console = Console()


@app.callback()
def _root(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    setup_logging(verbose=verbose)


@app.command()
def version() -> None:
    """Print the harness version."""
    console.print(f"quantbench {__version__}")


# --------------------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------------------


@app.command()
def preflight(
    hardware: Annotated[str, typer.Option(help="configs/hardware/<name>.yaml")] = "5070ti",
    out: Annotated[Path | None, typer.Option(help="where to write environment.json")] = None,
    require_gpu: Annotated[bool, typer.Option("--require-gpu/--no-require-gpu")] = True,
) -> None:
    """Environment probe: sm_120 assertion, driver/CUDA/VRAM/WSL, disk, binaries."""
    from quantbench.preflight import run_preflight

    hw = load_hardware(hardware)
    report = run_preflight(hardware=hw, require_gpu=require_gpu)

    table = Table(title="preflight", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    styles = {"ok": "green", "warn": "yellow", "fail": "red"}
    for check in report.checks:
        table.add_row(check.name, f"[{styles[check.severity]}]{check.severity}[/]", check.detail)
    console.print(table)
    for check in report.checks:
        if check.severity != "ok" and check.hint:
            console.print(f"[dim]hint ({check.name}):[/] {check.hint}")

    destination = out or (results_dir() / "environment.json")
    ensure_dirs(destination.parent)
    destination.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    console.print(f"environment written to [bold]{destination}[/]")
    raise typer.Exit(0 if report.ok else 1)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


@models_app.command("fetch")
def models_fetch(
    config: Annotated[str, typer.Option(help="model name or configs/models/*.yaml path")],
    revision: Annotated[str | None, typer.Option(help="override the pinned revision")] = None,
) -> None:
    """Download a model with the ``hf`` CLI into the local model cache.

    The Hugging Face CLI is invoked as a subprocess rather than taken as a Python
    dependency, mirroring the llama.cpp binary policy (§11.1).
    """
    model = load_model(config)
    target = models_dir() / model.slug / "hf"
    ensure_dirs(target)
    rev = revision or model.revision
    if not rev:
        console.print(
            "[yellow]warning:[/] this model has no pinned revision. Pin one in the config "
            "before publishing results — an unpinned model is not reproducible (§4.1)."
        )
    cmd = ["hf", "download", model.hf_repo, "--local-dir", str(target)]
    if rev:
        cmd += ["--revision", rev]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        console.print("[red]the 'hf' CLI was not found[/]; pip install huggingface_hub[cli]")
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(exc.returncode) from exc
    console.print(f"downloaded to [bold]{target}[/]")


@models_app.command("inspect")
def models_inspect(
    config: Annotated[str, typer.Option(help="model name or config path")],
    write: Annotated[bool, typer.Option("--write", help="write counts into the config")] = False,
) -> None:
    """Count parameters from the downloaded checkpoint's safetensors headers.

    Parameter counts are never guessed: until this runs, ``effective_bpw`` stays ``None``
    in every record for that model (§5.3).
    """
    from quantbench.checkpoint import (
        count_parameters,
        tokenizer_sha256,
        update_model_config,
    )
    from quantbench.paths import configs_dir

    model = load_model(config)
    path = models_dir() / model.slug / "hf"
    try:
        counts = count_parameters(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]; run 'quantbench models fetch' first")
        raise typer.Exit(1) from exc
    tok_sha = tokenizer_sha256(path)
    console.print(f"[bold]{model.name}[/]: {counts.n_tensors} tensors")
    console.print(f"  params_total          {counts.total:,}")
    console.print(f"  embedding (excluded)  {counts.embedding:,}  {counts.embedding_tensors}")
    console.print(f"  params_non_embedding  {counts.non_embedding:,}")
    console.print(f"  tokenizer_sha256      {tok_sha}")
    if write:
        target = configs_dir() / "models" / f"{model.slug}.yaml"
        changed = update_model_config(
            target,
            {
                "params_total": counts.total,
                "params_non_embedding": counts.non_embedding,
                "tokenizer_sha256": tok_sha,
            },
        )
        console.print(f"updated {target}: {', '.join(changed)}")


@models_app.command("dequantize")
def models_dequantize(
    model: Annotated[str, typer.Option(help="model name or config path")],
    variants: Annotated[str, typer.Option(help="comma-separated GGUF variants, e.g. Q8_0,Q4_K_M")],
    dtype: Annotated[str, typer.Option(help="float16 (default) or bfloat16")] = "float16",
    force: Annotated[bool, typer.Option(help="rebuild even if present")] = False,
) -> None:
    """Dequantize GGUF quants once to safetensors for the Transformers-side evaluations.

    Run from the engine environment (requirements/hf.txt). Each output carries a provenance
    manifest naming the source GGUF and its SHA-256.
    """
    from quantbench.config import load_format
    from quantbench.planner import artifact_path_for
    from quantbench.quantize.dequant import dequantize_gguf, dequantized_dir_for

    model_cfg = load_model(model)
    hf_dir = models_dir() / model_cfg.slug / "hf"
    fmt = load_format("gguf")
    failed = []
    for variant in [v.strip() for v in variants.split(",") if v.strip()]:
        gguf = Path(str(artifact_path_for(fmt, model_cfg, variant)))
        target = dequantized_dir_for(gguf, dtype)
        if target.exists() and not force:
            console.print(f"[green]{variant}[/] already dequantized: {target}")
            continue
        console.print(f"dequantizing {gguf} -> {target} ({dtype})")
        try:
            result = dequantize_gguf(gguf, hf_dir, dtype=dtype, force=force)
        except Exception as exc:
            console.print(f"[red]{variant} failed:[/] {exc}")
            failed.append(variant)
            continue
        console.print(f"[green]{variant}[/] -> {result.artifact_path} ({result.duration_s:.0f}s)")
    if failed:
        raise typer.Exit(1)


@models_app.command("prune")
def models_prune(
    keep: Annotated[list[str] | None, typer.Option(help="artifact substrings to keep")] = None,
    yes: Annotated[bool, typer.Option("--yes", help="actually delete")] = False,
) -> None:
    """List (and with --yes delete) quantized artifacts. Never deletes without confirmation."""
    root = models_dir()
    if not root.exists():
        console.print(f"nothing to prune at {root}")
        return
    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".gguf", ".safetensors"}
        and not any(k in str(p) for k in (keep or []))
    ]
    total_gb = sum(p.stat().st_size for p in candidates) / (1024**3)
    for path in candidates:
        console.print(f"  {path}  [dim]{path.stat().st_size / (1024**3):.2f} GB[/]")
    console.print(f"[bold]{len(candidates)} artifacts, {total_gb:.1f} GB[/]")
    if not yes:
        console.print("[yellow]dry run[/] — pass --yes to delete")
        return
    for path in candidates:
        path.unlink()
    console.print(f"deleted {len(candidates)} artifacts")


# --------------------------------------------------------------------------------------
# quantize
# --------------------------------------------------------------------------------------


@app.command()
def quantize(
    model: Annotated[str, typer.Option(help="model name or config path")],
    formats: Annotated[str, typer.Option(help="comma-separated, e.g. gguf:Q4_K_M,awq,exl3:4.0")],
    calibration: Annotated[Path | None, typer.Option(help="calibration text for imatrix")] = None,
    show_only: Annotated[
        bool, typer.Option("--show-only", help="print the plan, do not run")
    ] = False,
) -> None:
    """Produce quantized artifacts locally, with provenance manifests."""
    model_cfg = load_model(model)
    source = models_dir() / model_cfg.slug / "hf"
    out_root = models_dir() / model_cfg.slug
    requests = [f.strip() for f in formats.split(",") if f.strip()]

    console.print(f"[bold]{model_cfg.name}[/] <- {source}")
    for request in requests:
        family, _, variant = request.partition(":")
        console.print(f"  {family} {variant or '(default)'} -> {out_root}")
    if show_only:
        return
    if not source.exists():
        console.print(f"[red]checkpoint missing:[/] {source}; run 'quantbench models fetch'")
        raise typer.Exit(1)

    failed: list[str] = []
    for request in requests:
        family, _, variant = request.partition(":")
        try:
            result = _dispatch_quantize(family, variant, model_cfg, source, out_root, calibration)
        except Exception as exc:
            console.print(f"[red]{request} failed:[/] {exc}")
            console.print("[dim]record this verbatim in docs/failures.md (§0.2)[/]")
            failed.append(request)
            continue
        console.print(f"[green]{request}[/] -> {result.artifact_path} ({result.sha256})")
    if failed:
        console.print(f"[red]{len(failed)} of {len(requests)} producer(s) failed:[/] {failed}")
        raise typer.Exit(1)


def _dispatch_quantize(
    family: str,
    variant: str,
    model_cfg: ModelConfig,
    source: Path,
    out_root: Path,
    calibration: Path | None,
) -> QuantizeResult:
    slug = model_cfg.slug
    repo = model_cfg.hf_repo
    revision = model_cfg.revision
    if family == "gguf":
        from quantbench.quantize import gguf

        f16 = out_root / f"{slug}-f16.gguf"
        if not f16.exists():
            gguf.convert_to_f16(source, f16)
        imatrix_path = None
        if variant.endswith("-imat") or variant in gguf.IMATRIX_REQUIRED:
            if calibration is None:
                raise ValueError("--calibration is required for imatrix variants (§6.7)")
            imatrix_path = out_root / f"{slug}-imatrix.dat"
            if not imatrix_path.exists():
                gguf.build_imatrix(f16, calibration, imatrix_path)
        base_variant = variant.removesuffix("-imat")
        out = out_root / f"{slug}-{base_variant.lower()}{'-imat' if imatrix_path else ''}.gguf"
        return gguf.quantize(f16, base_variant, out, imatrix_path=imatrix_path)
    if family == "awq":
        from quantbench.quantize import compressor

        return compressor.quantize_awq(repo, out_root / f"{slug}-awq-int4", revision=revision)
    if family == "fp8":
        from quantbench.quantize import compressor

        return compressor.quantize_fp8(repo, out_root / f"{slug}-fp8", revision=revision)
    if family == "gptq":
        from quantbench.quantize import gptqmodel

        return gptqmodel.quantize_gptq(repo, out_root / f"{slug}-gptq-int4", revision=revision)
    if family == "exl3":
        from quantbench.quantize import exl3

        variant = variant or "4.0"
        # The directory name embeds the variant string verbatim so it matches the
        # artifact_pattern in configs/formats/exl3.yaml.
        return exl3.convert(source, out_root / f"{slug}-exl3-{variant}bpw", bpw=float(variant))
    raise ValueError(f"no producer for format family {family!r}")


# --------------------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------------------


@reference_app.command("build")
def reference_build(
    model: Annotated[str, typer.Option(help="model name or config path")],
    eval_config: Annotated[
        str, typer.Option("--eval", help="configs/eval/<name>.yaml")
    ] = "wikitext2",
    engine: Annotated[
        str,
        typer.Option(
            help="transformers: BF16 top-k reference cache (all formats); "
            "llama.cpp: F16-GGUF full-vocabulary KL base (GGUF formats)"
        ),
    ] = "transformers",
    cpu_offload: Annotated[
        bool | None,
        typer.Option(
            "--cpu-offload/--no-cpu-offload",
            help="default: from the model's reference_strategy (live = on-GPU)",
        ),
    ] = None,
    force: Annotated[bool, typer.Option(help="rebuild even if cached")] = False,
) -> None:
    """Build a model's quality reference once over the fixed eval stream (§5.2).

    Slow by design, and it happens once per (model, tokenizer, eval stream). The run is
    recorded like any other, so its PPL, duration and power are in results/raw/.
    """
    from quantbench.config import load_format
    from quantbench.planner import artifact_path_for
    from quantbench.reference import eval_stream as eval_stream_mod
    from quantbench.reference import logits_cache
    from quantbench.schema import (
        EngineSpec,
        ModelRef,
        ProtocolSpec,
        QuantSpec,
        RunSpec,
        WorkloadSpec,
    )

    model_cfg = load_model(model)
    eval_cfg = load_eval(eval_config)
    stream_dir = eval_stream_mod.stream_dir(model_cfg, eval_cfg)
    if (stream_dir / "tokens.npy").exists() and not force:
        stream = eval_stream_mod.load_eval_stream(stream_dir)
    else:
        console.print("building the evaluation token stream (hashed, identical across engines)")
        stream = eval_stream_mod.build_eval_stream(model_cfg, eval_cfg, force=force)
    console.print(f"  stream: {stream.n_tokens:,} tokens, sha {stream.sha}")

    model_ref = ModelRef(
        name=model_cfg.name,
        hf_repo=model_cfg.hf_repo,
        revision=model_cfg.revision,
        tier=model_cfg.tier,
        params_total=model_cfg.params_total,
        params_non_embedding=model_cfg.params_non_embedding,
        tokenizer_sha256=model_cfg.tokenizer_sha256,
    )
    workload = WorkloadSpec(kind="quality", generate_tokens=0, eval_config=eval_cfg.name)
    protocol = ProtocolSpec(warmup_iters=0, measured_iters=1, timeout_s=24 * 3600)

    if engine == "llama.cpp":
        fmt = load_format("gguf")
        f16 = artifact_path_for(fmt, model_cfg, "F16")
        base_sha = logits_cache.artifact_sha_from_manifest(Path(str(f16)))
        if base_sha is None:
            console.print(f"[red]F16 GGUF or its manifest missing:[/] {f16}")
            raise typer.Exit(1)
        key = logits_cache.llamacpp_base_key(model_cfg, eval_cfg, stream.meta, base_sha)
        if key.path.exists() and not force:
            console.print(f"[green]llama.cpp KL base already cached[/]: {key.path}")
            return
        spec = RunSpec(
            run_id=f"{model_cfg.slug}__gguf-f16__llama-cpp__reference__{eval_cfg.name}",
            model=model_ref,
            quant=QuantSpec(
                format="gguf",
                variant="F16",
                bits_nominal=16,
                artifact_path=f16,
                artifact_sha256=base_sha,
                producer=fmt.producer,
            ),
            engine=EngineSpec(name="llama.cpp", attention_backend="cuda-fa", kv_cache_dtype="f16"),
            workload=workload,
            protocol=protocol,
        )
    elif engine == "transformers":
        key_t = logits_cache.key_for(model_cfg, eval_cfg, stream.meta)
        if logits_cache.has_reference(key_t) and not force:
            path = logits_cache.reference_path(key_t)
            console.print(f"[green]reference already cached[/]: {path}")
            return
        offload = (
            cpu_offload if cpu_offload is not None else model_cfg.reference_strategy == "cached"
        )
        local = models_dir() / model_cfg.slug / "hf"
        spec = RunSpec(
            run_id=f"{model_cfg.slug}__bf16-reference__transformers__reference__{eval_cfg.name}",
            model=model_ref,
            quant=QuantSpec(
                format="bf16",
                variant="reference",
                bits_nominal=16,
                # the pinned local checkout, so the reference never silently re-downloads
                artifact_path=str(local) if local.exists() else None,
            ),
            engine=EngineSpec(
                name="transformers",
                attention_backend="sdpa",
                extra={"cpu_offload": offload},
            ),
            workload=workload,
            protocol=protocol,
        )
    else:
        console.print(f"[red]unknown reference engine {engine!r}[/]")
        raise typer.Exit(2)

    console.print(f"running {spec.run_id}")
    record = Runner(RunnerOptions(cooldown=False)).execute(spec)
    Runner.write_record(record, raw_dir())
    q = record.quality
    console.print(
        f"status: [bold]{record.status.value}[/]  ppl={q.ppl_wikitext2}  "
        f"scored={q.ppl_config.tokens if q.ppl_config else None}  "
        f"token_stream_verified={q.token_stream_verified}  {record.duration_s:.0f}s"
    )
    for warning in record.warnings:
        console.print(f"  [yellow]warning:[/] {warning}")
    for name, value in record.artifacts.items():
        console.print(f"  [dim]{name}: {value}[/]")
    if record.error:
        console.print(f"[red]{record.error}[/]")
        raise typer.Exit(1)


@reference_app.command("list")
def reference_list() -> None:
    """Show cached FP16 references."""
    root = reference_cache_dir()
    if not root.exists():
        console.print(f"no references cached at {root}")
        return
    for meta in sorted(root.rglob("reference-*.json")):
        payload = json.loads(meta.read_text(encoding="utf-8"))
        console.print(f"{payload.get('reference_id')}  [dim]{meta}[/]")


# --------------------------------------------------------------------------------------
# plan / run
# --------------------------------------------------------------------------------------


@app.command()
def plan(
    profile: Annotated[str, typer.Option(help="smoke | core | full | path")] = "core",
    out: Annotated[Path, typer.Option(help="where to write the plan")] = Path("plan.json"),
    engine: Annotated[str | None, typer.Option(help="override every engine (e.g. mock)")] = None,
    kind: Annotated[
        str | None, typer.Option(help="keep only one workload kind: latency | quality")
    ] = None,
) -> None:
    """Expand a profile into a shuffled, resumable run queue."""
    built = build_plan(profile, engine_override=engine, kind=kind)
    built.save(out)
    console.print(
        f"[bold]{built.profile}[/]: {len(built.entries)} runs, order={built.order}, "
        f"seed={built.seed} -> {out}"
    )
    for note in built.notes:
        console.print(f"  [yellow]note:[/] {note}")
    for route, scorer in built.scoring_routes.items():
        console.print(f"  [dim]quality for {route} will be scored by {scorer}[/]")


@app.command()
def run(
    plan_path: Annotated[Path, typer.Option("--plan", help="plan.json from 'quantbench plan'")],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force", help="re-run even if a record exists")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="mock engine, synthetic output")
    ] = False,
    cooldown: Annotated[bool, typer.Option("--cooldown/--no-cooldown")] = True,
    lock_clocks: Annotated[bool, typer.Option("--lock-clocks")] = False,
    limit: Annotated[int | None, typer.Option(help="stop after N runs")] = None,
) -> None:
    """Execute a plan, one subprocess per run."""
    loaded = Plan.load(plan_path)
    options = RunnerOptions(
        dry_run=dry_run,
        force=force,
        resume=resume,
        cooldown=cooldown and not dry_run,
        lock_clocks=lock_clocks,
        limit=limit,
    )
    if dry_run:
        console.print(
            "[yellow]--dry-run[/]: every run uses the mock engine. Output is synthetic, "
            "written to results/dryrun/, and excluded from analysis by default."
        )
    runner = Runner(options)
    result = runner.run_plan(loaded)
    console.print(
        f"[bold]{len(result.records)}[/] runs -> {result.output_dir} "
        f"({len(result.skipped)} skipped as already complete)"
    )
    for status, count in sorted(result.counts.items()):
        style = "green" if status == "ok" else "yellow"
        console.print(f"  [{style}]{status}[/]: {count}")
    failures = [
        r for r in result.records if r.status.value in {"build_failed", "engine_unsupported"}
    ]
    if failures:
        console.print(
            f"[yellow]{len(failures)} configuration(s) could not run.[/] "
            "Copy the errors into docs/failures.md — negative results ship (§0.2)."
        )


# --------------------------------------------------------------------------------------
# analyze / plot / report
# --------------------------------------------------------------------------------------


@app.command()
def analyze(
    results: Annotated[Path, typer.Option(help="results dir or raw dir")] = Path("results"),
    out: Annotated[Path, typer.Option(help="parquet output")] = Path("results/results.parquet"),
    allow_synthetic: Annotated[bool, typer.Option("--allow-synthetic")] = False,
) -> None:
    """Consolidate raw records into results.parquet."""
    from quantbench.analysis.load import consolidate

    frame, problems = consolidate(results, out=out, allow_synthetic=allow_synthetic)
    console.print(f"{len(frame)} records -> [bold]{out}[/]")
    from quantbench.analysis.paired import from_frame, paired_path

    paired = from_frame(frame, Path(results) / "lm_eval")
    if not paired.empty:
        paired.to_parquet(paired_path(out), index=False)
        console.print(f"{len(paired)} paired lm-eval comparisons -> [bold]{paired_path(out)}[/]")
    if not frame.empty:
        counts = frame["status"].value_counts().to_dict()
        console.print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for problem in problems[:20]:
        console.print(f"  [yellow]{problem}[/]")


@app.command()
def plot(
    results: Annotated[Path, typer.Option(help="results.parquet or results dir")] = Path(
        "results/results.parquet"
    ),
    out: Annotated[Path, typer.Option(help="figure output directory")] = Path("figures"),
    allow_synthetic: Annotated[bool, typer.Option("--allow-synthetic")] = False,
    strict: Annotated[bool, typer.Option("--strict", help="fail if any figure is skipped")] = False,
) -> None:
    """Regenerate every figure. Runs without a GPU, from committed results."""
    from quantbench.analysis.load import read_frame
    from quantbench.analysis.plots import generate_all, results_table

    frame = read_frame(results, allow_synthetic=allow_synthetic)
    ensure_dirs(out)
    figures = generate_all(frame, out, paired=_read_paired(results))
    table = Table(title=f"figures -> {out}")
    table.add_column("figure")
    table.add_column("status")
    table.add_column("N", justify="right")
    table.add_column("reason", overflow="fold")
    for row in results_table(figures):
        style = "yellow" if row["status"] == "skipped" else "green"
        table.add_row(
            row["figure"], f"[{style}]{row['status']}[/]", str(row["n"]), row["reason"] or ""
        )
    console.print(table)
    skipped = [f for f in figures if f.skipped]
    if skipped and strict:
        raise typer.Exit(1)


@app.command()
def report(
    results: Annotated[Path, typer.Option(help="results.parquet or results dir")] = Path(
        "results/results.parquet"
    ),
    targets: Annotated[
        list[Path] | None, typer.Option("--target", help="markdown files to update")
    ] = None,
    allow_synthetic: Annotated[bool, typer.Option("--allow-synthetic")] = False,
) -> None:
    """Regenerate the generated tables in README.md and docs/findings.md."""
    from quantbench.analysis.load import read_frame
    from quantbench.analysis.report import apply_sections, build_sections

    frame = read_frame(results, allow_synthetic=allow_synthetic) if Path(results).exists() else None
    if frame is None:
        console.print(f"[yellow]{results} does not exist[/]; tables will show '—'")
        import pandas as pd

        frame = pd.DataFrame()
    sections = build_sections(frame, paired=_read_paired(results))
    files = targets or [Path("README.md"), Path("docs/findings.md")]
    for path in files:
        updated = apply_sections(path, sections)
        console.print(
            f"{path}: updated {', '.join(updated) if updated else 'nothing (no markers)'}"
        )


@app.command("lmeval-tune")
def lmeval_tune(
    model: Annotated[str, typer.Option(help="model name or config path")],
    eval_config: Annotated[str, typer.Option("--eval")] = "lm_eval_tasks",
    margin_mib: Annotated[float, typer.Option(help="VRAM kept free below the budget")] = 1536.0,
    seed: Annotated[int, typer.Option()] = 1337,
) -> None:
    """Measure the best lm-eval batch size per task within the VRAM budget.

    Run from the engine environment with nothing else on the GPU. Writes
    configs/eval/lm_eval_batch/<model>.yaml with every probe, for review.
    """
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quantbench import lmeval_tune as lt
    from quantbench.paths import configs_dir

    model_cfg = load_model(model)
    eval_cfg = load_eval(eval_config)
    hf_dir = models_dir() / model_cfg.slug / "hf"
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    console.print("[bold]1. profiling lm-eval requests[/] (lm-eval's own task objects)")
    profiles = []
    for task in eval_cfg.lm_eval_tasks:
        prof = lt.profile_task(str(task["task"]), int(task.get("num_fewshot", 0)), tokenizer, seed)
        profiles.append(prof)
        console.print(
            f"  {prof.task:34s} {prof.request_type:15s} n={prof.n_requests:6d} "
            f"max_len={prof.max_len:5d} mean={prof.mean_len:7.1f} gen={prof.max_gen_toks}"
        )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[eval_cfg.lm_eval_dtype]
    console.print(f"[bold]2. loading {hf_dir} as {eval_cfg.lm_eval_dtype}[/]")
    lm = AutoModelForCausalLM.from_pretrained(
        hf_dir, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa"
    ).eval()

    check = _fp16_sanity(lm, model_cfg)
    console.print(f"  {eval_cfg.lm_eval_dtype} sanity vs BF16 reference: {check}")

    budget = lt.budget_mib(margin_mib)
    console.print(f"[bold]3. budget[/] {budget}")
    shared = lt._spill_sampler()
    tunings = []
    for prof in profiles:
        tunings.append(
            lt.tune_task(
                lm,
                prof,
                budget["budget_reserved_mib"],
                shared,
                int(tokenizer.eos_token_id),
                log=console.print,
            )
        )
    payload = lt.to_yaml_payload(model_cfg.slug, eval_cfg.lm_eval_dtype, budget, tunings)
    payload["fp16_sanity"] = check
    out = lt.tuning_path(configs_dir(), model_cfg.slug)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Generated by `quantbench lmeval-tune`; see src/quantbench/lmeval_tune.py.\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    total = sum(t.est_minutes or 0 for t in tunings)
    console.print(f"[bold]4. chosen[/] -> {out}")
    for t in tunings:
        console.print(
            f"  {t.profile.task:34s} batch {t.batch_size:4d}  ~{t.est_minutes or 0:6.1f} min"
        )
    console.print(f"  estimated GPU time per configuration: ~{total:.0f} min (forward passes only)")


def _fp16_sanity(lm: object, model_cfg: object) -> dict[str, float | bool]:
    """fp16 activations can overflow; compare one eval chunk with the BF16 reference."""
    import numpy as np
    import torch

    from quantbench.config import load_eval as _load_eval
    from quantbench.reference import eval_stream as es
    from quantbench.reference import logits_cache

    eval_cfg = _load_eval("wikitext2")
    stream = es.load_eval_stream(es.stream_dir(model_cfg, eval_cfg))  # type: ignore[arg-type]
    ref = logits_cache.load_reference(logits_cache.key_for(model_cfg, eval_cfg, stream.meta))  # type: ignore[arg-type]
    ctx = eval_cfg.ppl.ctx
    ids = torch.tensor([stream.ids[:ctx].tolist()], device="cuda")
    with torch.inference_mode():
        logits = lm(input_ids=ids).logits[0, :-1].float()  # type: ignore[operator]
        lp = torch.log_softmax(logits, dim=-1)
        target = lp.gather(1, ids[0, 1:, None]).squeeze(1).cpu().numpy()
        top1 = lp.argmax(dim=-1).cpu().numpy()
    n = target.size
    ref_target = ref.target_logprobs[:n]
    return {
        "finite": bool(np.isfinite(target).all()),
        "nll_this_dtype": round(float(-target.mean()), 5),
        "nll_bf16_reference": round(float(-ref_target.mean()), 5),
        "abs_diff": round(float(abs(target.mean() - ref_target.mean())), 5),
        "top1_agreement": round(float((top1 == ref.topk_ids[:n, 0]).mean()), 5),
        "positions": int(n),
    }


def _read_paired(results: Path) -> pd.DataFrame | None:
    """The paired lm-eval table written by `analyze`, if any."""
    import pandas as pd

    from quantbench.analysis.paired import paired_path

    path = paired_path(Path(results)) if Path(results).suffix == ".parquet" else None
    return pd.read_parquet(path) if path is not None and path.exists() else None


@app.command("configs")
def show_configs() -> None:
    """List available configs."""
    for kind in ("hardware", "models", "formats", "profiles", "eval"):
        console.print(f"[bold]{kind}[/]: {', '.join(list_configs(kind)) or '—'}")


@app.command()
def validate(
    profile: Annotated[str | None, typer.Option(help="also expand this profile")] = None,
) -> None:
    """Validate every config file (and optionally a profile expansion) without a GPU."""
    problems: list[str] = []
    for kind, loader in (
        ("hardware", load_hardware),
        ("models", load_model),
        (
            "formats",
            lambda n: __import__("quantbench.config", fromlist=["load_format"]).load_format(n),
        ),
        ("eval", load_eval),
    ):
        for name in list_configs(kind):
            try:
                loader(name)
            except Exception as exc:
                problems.append(f"{kind}/{name}: {exc}")
    if profile:
        try:
            built = build_plan(profile)
            console.print(f"profile {profile}: {len(built.entries)} runs")
            for ref in {f"{s.quant.format}:{s.quant.variant}" for s in built.entries}:
                parse_format_ref(ref)
        except Exception as exc:
            problems.append(f"profiles/{profile}: {exc}")
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/]")
        raise typer.Exit(1)
    console.print("[green]all configs valid[/]")


@app.command("figures-dir")
def show_figures_dir() -> None:
    """Print where figures are written (used by the Makefile and CI)."""
    console.print(str(figures_dir()))


if __name__ == "__main__":  # pragma: no cover
    app()
