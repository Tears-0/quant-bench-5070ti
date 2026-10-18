"""Measured per-task batch sizes for lm-eval (VRAM utilisation and throughput).

lm-eval's ``--batch_size auto`` searches for the largest batch that does not OOM. On this
host CUDA does not OOM: the driver's sysmem fallback spills into system RAM instead
(docs/failures.md), so ``auto`` walks straight into a 10-50x slowdown. A single fixed batch
size is the opposite problem: sized for MMLU's longest 5-shot prompt it starves ARC's short
0-shot ones, and sized for ARC it overflows MMLU.

So the batch size is *measured*, per task, once:

1. **Profile the real requests.** lm-eval's own task objects build every request with the
   pinned few-shot count and seed; each is tokenized exactly as lm-eval's HF backend does
   (context and continuation encoded together; the model sees ``len - 1`` tokens). lm-eval
   sorts requests longest-first, so the first batch, at the longest length, is the
   memory peak.
2. **Measure the peak.** For each candidate batch size, run the backend's exact hot path at
   that worst-case length — forward + ``log_softmax`` over the full vocabulary for
   log-likelihood tasks, ``generate`` with the task's full generation budget for generative
   ones — and record the allocator peak and the throughput.
3. **Budget against reality.** The VRAM budget is the card's total minus what other
   processes hold right now, minus this process's CUDA-context overhead, minus a safety
   margin (spill was observed to begin ~0.7 GiB from full). Any probe that spills (Windows
   shared GPU memory rises) is rejected outright.
4. **Pick for throughput.** Among candidates within budget, the fastest; when a smaller
   batch is within 3% of it, the smaller one, because headroom costs nothing and
   fragmentation late in a long run is real.

The result is written to ``configs/eval/lm_eval_batch/<model>.yaml`` with every measurement,
so the choice is reviewable and the evaluation is reproducible (a fixed batch per task),
rather than depending on how much VRAM happened to be free during the run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MIB = 1024 * 1024

#: Headroom kept free below the budget. Spill was observed to begin ~0.7 GiB from full.
SAFETY_MARGIN_MIB = 1536.0
#: A smaller batch within this fraction of the best throughput is preferred.
THROUGHPUT_TOLERANCE = 0.03

LOGLIKELIHOOD_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128)
GENERATE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class TaskProfile:
    task: str
    num_fewshot: int
    request_type: str  # loglikelihood | generate_until
    n_requests: int
    max_len: int  # longest model input, tokens
    mean_len: float
    total_tokens: int  # model input tokens over all requests (generation excluded)
    max_gen_toks: int = 0


@dataclass
class Probe:
    batch_size: int
    peak_reserved_mib: float
    seconds: float
    tokens_per_s: float
    within_budget: bool
    spilled: bool = False
    error: str | None = None


@dataclass
class TaskTuning:
    profile: TaskProfile
    probes: list[Probe] = field(default_factory=list)
    batch_size: int = 1
    est_minutes: float | None = None


# --------------------------------------------------------------------------------------
# 1. request profiling (lm-eval's own task objects)
# --------------------------------------------------------------------------------------


def profile_task(
    task: str, num_fewshot: int, tokenizer: Any, seed: int, default_max_gen: int = 256
) -> TaskProfile:
    from lm_eval.tasks import TaskManager, get_task_dict

    obj = get_task_dict([task], TaskManager())[task]
    # mirror lm_eval.evaluator.simple_evaluate's few-shot handling
    if obj.get_config("num_fewshot") != 0 or num_fewshot == 0:
        obj.set_config(key="num_fewshot", value=num_fewshot)
    obj.set_fewshot_seed(seed=seed)
    obj.build_all_requests(limit=None, rank=0, world_size=1)

    lengths: list[int] = []
    request_type = "loglikelihood"
    max_gen = 0
    texts: list[str] = []
    for inst in obj.instances:
        request_type = inst.request_type
        if inst.request_type == "generate_until":
            context, gen_kwargs = inst.args
            texts.append(context)
            max_gen = max(max_gen, int((gen_kwargs or {}).get("max_gen_toks", default_max_gen)))
        else:
            context, continuation = inst.args[0], inst.args[1]
            texts.append(context + continuation)
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        n = len(ids)
        lengths.append(n if request_type == "generate_until" else max(n - 1, 1))
    return TaskProfile(
        task=task,
        num_fewshot=num_fewshot,
        request_type=request_type,
        n_requests=len(lengths),
        max_len=max(lengths),
        mean_len=sum(lengths) / len(lengths),
        total_tokens=sum(lengths),
        max_gen_toks=max_gen,
    )


# --------------------------------------------------------------------------------------
# 2-3. measurement under a real budget
# --------------------------------------------------------------------------------------


def _spill_sampler() -> Callable[[], float | None]:
    """Shared (system-RAM) GPU memory of this process, or a no-op off Windows."""
    from quantbench.metrics.wddm import open_counters

    counters = open_counters()
    if counters is None:
        return lambda: None
    pid = {os.getpid()}

    def sample() -> float | None:
        s = counters.sample(pid)
        return s.shared_mb if s is not None else None

    return sample


def _device_used_mib() -> tuple[float, float]:
    import pynvml

    pynvml.nvmlInit()
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return info.used / MIB, info.total / MIB
    finally:
        pynvml.nvmlShutdown()


def budget_mib(margin_mib: float = SAFETY_MARGIN_MIB) -> dict[str, float]:
    """Allocator budget for this process, measured *after* the model is loaded.

    budget = card total - everything currently in use - margin + what torch already
    reserved (that reservation is ours and is reused by the probes).
    """
    import torch

    used, total = _device_used_mib()
    reserved = torch.cuda.memory_reserved() / MIB
    return {
        "total_mib": total,
        "device_used_mib": used,
        "torch_reserved_mib": reserved,
        "margin_mib": margin_mib,
        "budget_reserved_mib": total - used - margin_mib + reserved,
    }


def probe_batch(
    model: Any,
    profile: TaskProfile,
    batch_size: int,
    budget_reserved_mib: float,
    shared_sample: Callable[[], float | None],
    eos_id: int,
) -> Probe:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from quantbench.metrics.wddm import SPILL_THRESHOLD_MB

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    shared_before = shared_sample() or 0.0
    length = profile.max_len
    inp = torch.randint(1000, 30000, (batch_size, length), device=model.device)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            if profile.request_type == "generate_until":
                gen = profile.max_gen_toks or 256
                model.generate(
                    input_ids=inp,
                    attention_mask=torch.ones_like(inp),
                    max_new_tokens=gen,
                    min_new_tokens=gen,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
                tokens = batch_size * (length + gen)
            else:
                # the exact hot path of lm-eval's HF _loglikelihood_tokens
                logits = model(inp).logits
                multi_logits = F.log_softmax(logits, dim=-1)
                _ = multi_logits[:, -1, :].sum().item()
                del logits, multi_logits
                tokens = batch_size * length
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        return Probe(batch_size, float("nan"), 0.0, 0.0, False, error=f"OOM: {str(exc)[:120]}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return Probe(batch_size, float("nan"), 0.0, 0.0, False, error=f"OOM: {str(exc)[:120]}")
        raise
    finally:
        del inp
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_reserved() / MIB
    shared_after = shared_sample() or 0.0
    spilled = (shared_after - shared_before) > SPILL_THRESHOLD_MB
    return Probe(
        batch_size=batch_size,
        peak_reserved_mib=peak,
        seconds=seconds,
        tokens_per_s=tokens / seconds if seconds > 0 else 0.0,
        within_budget=(peak <= budget_reserved_mib) and not spilled,
        spilled=spilled,
    )


def choose(probes: list[Probe]) -> int:
    """Fastest within budget; prefer a smaller batch within THROUGHPUT_TOLERANCE of it."""
    valid = [p for p in probes if p.within_budget and p.tokens_per_s > 0]
    if not valid:
        return 1
    best = max(p.tokens_per_s for p in valid)
    good = [p for p in valid if p.tokens_per_s >= best * (1 - THROUGHPUT_TOLERANCE)]
    return min(p.batch_size for p in good)


def tune_task(
    model: Any,
    profile: TaskProfile,
    budget_reserved_mib: float,
    shared_sample: Callable[[], float | None],
    eos_id: int,
    log: Callable[[str], None] = print,
) -> TaskTuning:
    tuning = TaskTuning(profile=profile)
    candidates = (
        GENERATE_CANDIDATES
        if profile.request_type == "generate_until"
        else LOGLIKELIHOOD_CANDIDATES
    )
    for batch in candidates:
        if batch > max(1, profile.n_requests):
            break
        # one warm-up at batch 1 so the first timing is not a cold-start artefact
        if batch == candidates[0]:
            probe_batch(model, profile, 1, budget_reserved_mib, shared_sample, eos_id)
        probe = probe_batch(model, profile, batch, budget_reserved_mib, shared_sample, eos_id)
        tuning.probes.append(probe)
        log(
            f"  {profile.task:34s} b={batch:<4d} peak {probe.peak_reserved_mib:8.0f} MiB "
            f"{probe.tokens_per_s:10.0f} tok/s {'ok' if probe.within_budget else 'OVER'}"
            f"{' SPILL' if probe.spilled else ''}{' ' + probe.error if probe.error else ''}"
        )
        if not probe.within_budget:
            break
    tuning.batch_size = choose(tuning.probes)
    chosen = next((p for p in tuning.probes if p.batch_size == tuning.batch_size), None)
    if chosen and chosen.tokens_per_s > 0:
        gen = (
            profile.max_gen_toks * profile.n_requests
            if profile.request_type == "generate_until"
            else 0
        )
        # length-sorted batches shrink after the first, so mean length is the right scale
        tokens = profile.total_tokens + gen
        tuning.est_minutes = tokens / chosen.tokens_per_s / 60.0
    return tuning


def to_yaml_payload(
    model_slug: str, dtype: str, budget: dict[str, float], tunings: list[TaskTuning]
) -> dict[str, Any]:
    return {
        "model": model_slug,
        "dtype": dtype,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget": {k: round(v, 1) for k, v in budget.items()},
        "selection": (
            "fastest batch within the VRAM budget at the task's longest request; a smaller "
            f"batch within {THROUGHPUT_TOLERANCE:.0%} of it is preferred"
        ),
        "tasks": {
            t.profile.task: {
                "batch_size": t.batch_size,
                "est_minutes": round(t.est_minutes, 1) if t.est_minutes else None,
                "profile": asdict(t.profile),
                "probes": [
                    {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(p).items()}
                    for p in t.probes
                ],
            }
            for t in tunings
        },
    }


def tuned_batch_sizes(path: Path) -> dict[str, int]:
    """``{task: batch_size}`` from a tuning file, or ``{}`` if there is none."""
    import yaml

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(task): int(entry["batch_size"])
        for task, entry in (data.get("tasks") or {}).items()
        if isinstance(entry, dict) and "batch_size" in entry
    }


def tuning_path(configs_dir: Path, model_slug: str) -> Path:
    return configs_dir / "eval" / "lm_eval_batch" / f"{model_slug}.yaml"
