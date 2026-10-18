"""lm-evaluation-harness bridge (requirements.md §5.2).

Downstream accuracy is run by the ``lm_eval`` CLI as a subprocess — never imported — so the
harness depends on its stable command line rather than its Python internals, and a broken
lm-eval install costs one record, not the session.

Two backends are wired, matching how each format is actually served:

* ``hf``   — lm-eval loads the checkpoint itself (BF16 reference, NF4, GPTQ/AWQ via
  Transformers). The worker does *not* also load it: two copies would not fit.
* ``gguf`` — lm-eval talks to the ``llama-server`` the worker started, over
  ``/v1/completions``. Log-likelihood tasks need prompt log-probs (``echo``), which
  llama-server may not provide; that is probed, and a refusal is recorded as
  ``engine_unsupported`` with the server's answer rather than papered over.

Task list, few-shot counts and the primary metric per task are pinned in
``configs/eval/lm_eval_tasks.yaml`` and :func:`quantbench.metrics.quality.parse_lm_eval_output`.
The MMLU subset is reported per subject and as ``mmlu_subset``, the unweighted mean of
the pinned subjects.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantbench.metrics.quality import parse_lm_eval_output


class LmEvalError(RuntimeError):
    """lm-eval failed; the message carries the tail of its output."""


class LmEvalOOM(LmEvalError):
    """lm-eval ran out of GPU memory (a recorded outcome, not a crash)."""


@dataclass
class LmEvalResult:
    scores: dict[str, float | None] = field(default_factory=dict)
    commands: list[list[str]] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    lm_eval_version: str | None = None
    limit: int | None = None
    batch_sizes: dict[str, int] = field(default_factory=dict)
    task_seconds: dict[str, float] = field(default_factory=dict)
    resumed: list[str] = field(default_factory=list)


def task_groups(tasks: list[dict[str, Any]]) -> dict[int, list[str]]:
    """lm-eval takes one ``--num_fewshot`` per invocation, so group tasks by it."""
    groups: dict[int, list[str]] = defaultdict(list)
    for task in tasks:
        groups[int(task.get("num_fewshot", 0))].append(str(task["task"]))
    return dict(sorted(groups.items()))


def model_args_hf(path: str, quant_format: str, dtype: str = "float16") -> str:
    if path.endswith(".gguf"):
        from quantbench.quantize.dequant import dequantized_dir_for

        dequant = dequantized_dir_for(Path(path))
        if dequant.exists():
            # one-time fp16 dequantization: fast load, original HF tokenizer and config
            return f"pretrained={dequant},dtype={dtype},attn_implementation=sdpa"
        # Open question Q15 option (a): Transformers dequantizes the GGUF weights. fp16, not
        # bf16, because fp16's 10-bit mantissa holds dequantized K-quant values more exactly.
        gguf = Path(path)
        return (
            f"pretrained={gguf.parent},gguf_file={gguf.name},dtype=float16,attn_implementation=sdpa"
        )
    args = [f"pretrained={path}", f"dtype={dtype}", "attn_implementation=sdpa"]
    if quant_format == "nf4":
        args += ["load_in_4bit=True", "bnb_4bit_quant_type=nf4"]
    return ",".join(args)


def build_command(
    backend: str,
    model_args: str,
    tasks: list[str],
    num_fewshot: int,
    out_dir: Path,
    seed: int,
    limit: int | None = None,
    batch_size: str = "8",
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--output_path",
        str(out_dir),
        "--seed",
        str(seed),
        "--log_samples",
    ]
    if backend != "gguf":  # the gguf backend sends one request at a time
        cmd += ["--batch_size", batch_size]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def collect_results(out_dir: Path) -> tuple[dict[str, float | None], list[str], str | None]:
    """Merge every ``results_*.json`` lm-eval wrote under ``out_dir``."""
    scores: dict[str, float | None] = {}
    files: list[str] = []
    version: str | None = None
    for path in sorted(out_dir.rglob("results_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores |= parse_lm_eval_output(payload)
        files.append(str(path))
        version = version or payload.get("lm_eval_version") or payload.get("git_hash")
    subjects = [v for k, v in scores.items() if k.startswith("mmlu_") and not k.endswith("_stderr")]
    if subjects and all(v is not None for v in subjects):
        scores["mmlu_subset"] = sum(v for v in subjects if v is not None) / len(subjects)
    return scores, files, version


def run(
    backend: str,
    model_args: str,
    tasks: list[dict[str, Any]],
    out_dir: Path,
    seed: int,
    timeout_s: int,
    limit: int | None = None,
    batch_sizes: dict[str, int] | None = None,
    default_batch_size: int = 1,
    on_spawn: Callable[[int], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> LmEvalResult:
    """Run the tasks strictly in sequence, each at its own measured batch size.

    ``hf``: one :mod:`quantbench.lmeval_job` process loads the model once and runs every
    task. Other backends: one ``lm_eval`` CLI invocation per task. Either way a task whose
    results file exists is skipped, so a failure or reboot costs one task, not the run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "hf":
        return _run_hf_job(
            model_args,
            tasks,
            out_dir,
            seed,
            timeout_s,
            limit,
            batch_sizes or {},
            default_batch_size,
            on_spawn,
            log,
        )
    result = LmEvalResult(limit=limit)
    for task in tasks:
        name = str(task["task"])
        fewshot = int(task.get("num_fewshot", 0))
        task_dir = out_dir / name
        batch = (batch_sizes or {}).get(name, default_batch_size)
        result.batch_sizes[name] = batch
        if limit is None and any(task_dir.rglob("results_*.json")):
            result.resumed.append(name)
            if log:
                log(f"lm-eval {name}: results present, skipped (resume)")
            continue
        cmd = build_command(
            backend, model_args, [name], fewshot, task_dir, seed, limit, batch_size=str(batch)
        )
        result.commands.append(cmd)
        t0 = time.perf_counter()
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as proc:
            if on_spawn is not None:
                on_spawn(proc.pid)  # the model lives in this process: attribute its VRAM
            try:
                output, _ = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
        (task_dir / "lm_eval.log").parent.mkdir(parents=True, exist_ok=True)
        (task_dir / "lm_eval.log").write_text(output, encoding="utf-8")
        result.task_seconds[name] = time.perf_counter() - t0
        if proc.returncode != 0:
            tail = output[-4000:]
            if "out of memory" in tail.lower():
                raise LmEvalOOM(f"lm_eval OOM on {name} at batch {batch}:\n{tail}")
            raise LmEvalError(f"lm_eval exited {proc.returncode} for {name}:\n{tail}")
        if log:
            log(f"lm-eval {name}: done in {result.task_seconds[name]:.0f}s at batch {batch}")
    result.scores, result.output_files, result.lm_eval_version = collect_results(out_dir)
    return result


def _run_hf_job(
    model_args: str,
    tasks: list[dict[str, Any]],
    out_dir: Path,
    seed: int,
    timeout_s: int,
    limit: int | None,
    batch_sizes: dict[str, int],
    default_batch_size: int,
    on_spawn: Callable[[int], None] | None,
    log: Callable[[str], None] | None,
) -> LmEvalResult:
    """One model load for all tasks (see :mod:`quantbench.lmeval_job`)."""
    from quantbench.lmeval_job import EXIT_OK, EXIT_OOM

    result = LmEvalResult(limit=limit)
    job_tasks = []
    for task in tasks:
        name = str(task["task"])
        batch = batch_sizes.get(name, default_batch_size)
        result.batch_sizes[name] = batch
        if limit is None and any((out_dir / name).rglob("results_*.json")):
            result.resumed.append(name)
        job_tasks.append(
            {"task": name, "num_fewshot": int(task.get("num_fewshot", 0)), "batch_size": batch}
        )
    if limit is None and len(result.resumed) == len(job_tasks):
        if log:
            log("lm-eval: every task already complete, nothing to run")
        result.scores, result.output_files, result.lm_eval_version = collect_results(out_dir)
        return result
    job = {
        "model_args": model_args,
        "tasks": job_tasks,
        "out_dir": str(out_dir),
        "seed": seed,
        "limit": limit,
    }
    job_path = out_dir / "job.json"
    pid_file = out_dir / "job.pid"
    pid_file.unlink(missing_ok=True)
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-m", "quantbench.lmeval_job", "--job", str(job_path)]
    result.commands.append(cmd)
    if log and result.resumed:
        log(f"lm-eval: resuming, already complete: {', '.join(result.resumed)}")
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        if on_spawn is not None:
            on_spawn(proc.pid)
            _watch_pid_file(pid_file, proc, on_spawn)  # the real interpreter's PID
        try:
            output, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
    with (out_dir / "job.log").open("a", encoding="utf-8") as fh:
        fh.write(output)
    if proc.returncode == EXIT_OOM:
        raise LmEvalOOM(f"lm-eval job ran out of GPU memory:\n{output[-4000:]}")
    if proc.returncode != EXIT_OK:
        raise LmEvalError(f"lm-eval job exited {proc.returncode}:\n{output[-4000:]}")
    for path in out_dir.rglob("results_*.json"):
        meta = json.loads(path.read_text(encoding="utf-8")).get("quantbench") or {}
        if "seconds" in meta:
            result.task_seconds[path.parent.name] = float(meta["seconds"])
    result.scores, result.output_files, result.lm_eval_version = collect_results(out_dir)
    return result


def _watch_pid_file(
    pid_file: Path, proc: subprocess.Popen[str], on_spawn: Callable[[int], None]
) -> None:
    """Register the PID the job reports for itself, as soon as it appears (see lmeval_job)."""

    def watch() -> None:
        while proc.poll() is None:
            if pid_file.exists():
                text = pid_file.read_text(encoding="utf-8").strip()
                if text.isdigit():
                    on_spawn(int(text))
                    return
            time.sleep(0.2)

    threading.Thread(target=watch, name="lmeval-pid", daemon=True).start()
