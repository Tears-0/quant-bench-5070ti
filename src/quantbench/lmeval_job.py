"""One lm-eval job per configuration: load the model once, run the tasks in sequence.

    python -m quantbench.lmeval_job --job job.json

Running ``lm_eval`` once per task (the first implementation) paid ~2 minutes of Python
imports and task indexing plus ~30 seconds of model loading for *every* task, against as
little as 40 seconds of compute: on Tier A, about 70 minutes of an idle GPU across the three
configurations. This job keeps what made per-task runs attractive and drops the cost:

* the model is loaded **once** (lm-eval's ``HFLM``), and each task runs with **its own
  measured batch size** (``configs/eval/lm_eval_batch/<model>.yaml``), set on the loaded
  model before the task starts;
* tasks run **strictly in sequence**, each saved as soon as it finishes, so a task whose
  results exist is skipped on resume and a failure costs one task;
* it is still a **subprocess** of the worker: an OOM or a crash cannot poison the harness,
  and its PID is tracked so its VRAM (and any sysmem spill) is attributed to the run.

The per-task call is ``lm_eval.simple_evaluate`` with the same seeds, few-shot count and
sample logging as the CLI, so scores are identical to a CLI run at the same batch size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: Exit codes understood by :mod:`quantbench.lmeval`.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_OOM = 3


def _model_kwargs(model_args: str) -> dict[str, Any]:
    """``"pretrained=x,dtype=float16"`` -> ``{"pretrained": "x", "dtype": "float16"}``."""
    out: dict[str, Any] = {}
    for part in model_args.split(","):
        if part.strip():
            key, _, value = part.partition("=")
            out[key.strip()] = value.strip()
    return out


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def run_job(job: dict[str, Any]) -> int:
    out_dir = Path(job["out_dir"])
    seed = int(job["seed"])
    limit = job.get("limit")
    pending = [
        t
        for t in job["tasks"]
        if limit is not None or not any((out_dir / t["task"]).rglob("results_*.json"))
    ]
    if not pending:
        print("all tasks already complete", flush=True)
        return EXIT_OK

    # heavy imports only once there is work to do
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    t_load = time.perf_counter()
    lm = HFLM(batch_size=1, **_model_kwargs(job["model_args"]))
    task_manager = TaskManager()  # index task YAMLs once, not per task
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    for task in pending:
        name, fewshot, batch = str(task["task"]), int(task["num_fewshot"]), int(task["batch_size"])
        task_dir = out_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        lm.batch_size_per_gpu = batch  # the measured batch for *this* task
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            results = simple_evaluate(
                model=lm,
                tasks=[name],
                num_fewshot=fewshot,
                limit=limit,
                log_samples=True,
                task_manager=task_manager,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed,
            )
        except Exception as exc:
            (task_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"task {name} failed: {exc}", flush=True)
            return EXIT_OOM if _is_oom(exc) else EXIT_FAILED
        seconds = time.perf_counter() - t0
        assert results is not None
        samples = results.pop("samples", {})
        results["quantbench"] = {
            "batch_size": batch,
            "num_fewshot": fewshot,
            "seconds": round(seconds, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "limit": limit,
        }
        stamp = time.strftime("%Y%m%dT%H%M%S")
        # short names: Windows paths are capped at 260 characters, and the task
        # directory already carries the task name
        with (task_dir / f"samples_{stamp}.jsonl").open("w", encoding="utf-8") as fh:
            for sample in samples.get(name, []):
                fh.write(json.dumps(sample, default=str) + "\n")
        # results file last: its existence is the resume marker
        (task_dir / f"results_{stamp}.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"task {name} done in {seconds:.0f}s at batch {batch}", flush=True)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantbench.lmeval_job")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    # On Windows a venv's python.exe is a launcher that runs the real interpreter as a
    # child, so the PID the parent sees is not the one that holds the GPU memory. Report
    # the real one; the parent attributes VRAM (and spill) to it.
    pid_file = Path(job["out_dir"]) / "job.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return run_job(job)
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        return EXIT_OOM if _is_oom(exc) else EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
