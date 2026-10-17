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

if TYPE_CHECKING:  # producers pull in heavy tooling and are imported at call time only
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
    cpu_offload: Annotated[
        bool, typer.Option(help="offload to CPU/RAM for models that do not fit")
    ] = True,
    force: Annotated[bool, typer.Option(help="rebuild even if cached")] = False,
) -> None:
    """Run FP16 once over the fixed eval stream and cache its top-k logits (§5.2).

    Slow by design, and it happens once per (model, tokenizer, eval stream).
    """
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
    console.print("building the evaluation token stream (hashed, byte-identical across engines)")
    stream = eval_stream_mod.build_eval_stream(model_cfg, eval_cfg, force=force)
    console.print(f"  {stream.n_tokens:,} tokens, sha {stream.sha}")

    key = logits_cache.key_for(model_cfg, eval_cfg, stream.meta)
    if logits_cache.has_reference(key) and not force:
        console.print(f"[green]reference already cached[/]: {logits_cache.reference_path(key)}")
        return

    spec = RunSpec(
        run_id=f"{model_cfg.slug}__reference-fp16__{eval_cfg.name}",
        model=ModelRef(
            name=model_cfg.name,
            hf_repo=model_cfg.hf_repo,
            revision=model_cfg.revision,
            tier=model_cfg.tier,
            params_total=model_cfg.params_total,
            params_non_embedding=model_cfg.params_non_embedding,
        ),
        quant=QuantSpec(format="bf16", variant="reference", bits_nominal=16),
        engine=EngineSpec(
            name=model_cfg.reference_engine,
            attention_backend="sdpa",
            extra={"cpu_offload": cpu_offload},
        ),
        workload=WorkloadSpec(kind="quality", generate_tokens=0, eval_config=eval_cfg.name),
        protocol=ProtocolSpec(warmup_iters=0, measured_iters=1, timeout_s=24 * 3600),
    )
    record = Runner(RunnerOptions(cooldown=False)).execute(spec)
    Runner.write_record(record, raw_dir())
    console.print(f"status: [bold]{record.status.value}[/]  ppl={record.quality.ppl_wikitext2}")
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
    figures = generate_all(frame, out)
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
    sections = build_sections(frame)
    files = targets or [Path("README.md"), Path("docs/findings.md")]
    for path in files:
        updated = apply_sections(path, sections)
        console.print(
            f"{path}: updated {', '.join(updated) if updated else 'nothing (no markers)'}"
        )


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
