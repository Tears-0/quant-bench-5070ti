"""``results/raw/*.json`` -> validated dataframe (requirements.md §7.1, §8.2).

Two guardrails live here:

* **Version validation.** Records are validated on load; a record whose ``record_version``
  this code cannot interpret is refused, not coerced.
* **Synthetic quarantine.** Records produced by the mock engine (``--dry-run``, CI) carry
  ``synthetic: true``. They are excluded by default and can never be silently mixed into a
  measured dataset; asking for them requires ``allow_synthetic=True`` and taints the whole
  frame via the ``synthetic`` column, which the plotting layer watermarks.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from quantbench.schema import RunRecord, SchemaVersionError, load_record


class MixedDatasetError(ValueError):
    """Synthetic and measured records were combined without an explicit opt-in."""


def iter_record_files(results: Path) -> Iterable[Path]:
    """Accept a raw directory, a results root, or a single file."""
    results = Path(results)
    if results.is_file():
        yield results
        return
    for candidate in (results / "raw", results):
        if candidate.is_dir():
            yield from sorted(candidate.glob("*.json"))
            return


def load_records(
    results: Path, allow_synthetic: bool = False, strict: bool = False
) -> tuple[list[RunRecord], list[str]]:
    """Load and validate records. Returns ``(records, problems)``."""
    records: list[RunRecord] = []
    problems: list[str] = []
    for path in iter_record_files(results):
        if path.name == "environment.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = load_record(payload)
        except (SchemaVersionError, ValueError, OSError) as exc:
            problems.append(f"{path.name}: {exc}")
            if strict:
                raise
            continue
        if record.synthetic and not allow_synthetic:
            problems.append(f"{path.name}: synthetic record excluded")
            continue
        records.append(record)
    return records, problems


def record_to_row(record: RunRecord) -> dict[str, Any]:
    """Flatten one record into the analysis schema. Missing values stay ``None``."""
    spec, lat, mem, pwr, qual = (
        record.spec,
        record.latency,
        record.memory,
        record.power,
        record.quality,
    )
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "spec_hash": record.spec_hash,
        "status": record.status.value,
        "synthetic": record.synthetic,
        "duration_s": record.duration_s,
        # spec
        "model": spec.model.name,
        "tier": spec.model.tier,
        "params_total": spec.model.params_total,
        "params_non_embedding": spec.model.params_non_embedding,
        "format": spec.quant.format,
        "variant": spec.quant.variant,
        "format_variant": f"{spec.quant.format}:{spec.quant.variant}",
        "bits_nominal": spec.quant.bits_nominal,
        "imatrix": spec.quant.imatrix,
        "calibration_dataset": spec.quant.calibration_dataset,
        "engine": spec.engine.name,
        "engine_version": record.engine_metadata.get("version"),
        "attention_backend": spec.engine.attention_backend,
        "kv_cache_dtype": spec.engine.kv_cache_dtype,
        "workload_kind": spec.workload.kind,
        "batch_size": spec.workload.batch_size,
        "prompt_tokens": spec.workload.prompt_tokens,
        "generate_tokens": spec.workload.generate_tokens,
        "warmup_iters": spec.protocol.warmup_iters,
        "measured_iters": spec.protocol.measured_iters,
        "clocks_locked": spec.protocol.clocks_locked,
        # latency
        "ttft_ms_p50": lat.ttft_ms.p50,
        "ttft_ms_p90": lat.ttft_ms.p90,
        "ttft_ms_p99": lat.ttft_ms.p99,
        "ttft_ms_iqr": lat.ttft_ms.iqr,
        "itl_ms_p50": lat.itl_ms.p50,
        "itl_ms_p90": lat.itl_ms.p90,
        "itl_ms_p95": lat.itl_ms.p95,
        "itl_ms_p99": lat.itl_ms.p99,
        "itl_ms_std": lat.itl_ms.std,
        "decode_tok_s_p50": lat.decode_tok_s.p50,
        "decode_tok_s_iqr": lat.decode_tok_s.iqr,
        "decode_tok_s_n": lat.decode_tok_s.n,
        "prefill_tok_s_p50": lat.prefill_tok_s.p50,
        "e2e_latency_ms_p50": lat.e2e_latency_ms.p50,
        "throughput_tok_s_total_p50": lat.throughput_tok_s_total.p50,
        # memory
        "vram_weights_mb": mem.vram_weights_mb,
        "vram_kv_mb": mem.vram_kv_mb,
        "vram_peak_allocated_mb": mem.vram_peak_allocated_mb,
        "vram_peak_nvsmi_mb": mem.vram_peak_nvsmi_mb,
        "vram_baseline_nvsmi_mb": mem.vram_baseline_nvsmi_mb,
        "vram_peak_attributed_mb": mem.vram_peak_attributed_mb,
        "vram_accounting": mem.vram_accounting,
        "weight_bytes_non_embedding": mem.weight_bytes_non_embedding,
        "effective_bpw": mem.effective_bpw,
        "load_time_s": mem.load_time_s,
        # power
        "power_avg_w": pwr.power_avg_w,
        "power_peak_w": pwr.power_peak_w,
        "energy_j": pwr.energy_j,
        "tokens_per_joule": pwr.tokens_per_joule,
        "sm_clock_avg_mhz": pwr.sm_clock_avg_mhz,
        "gpu_temp_start_c": pwr.gpu_temp_start_c,
        "gpu_temp_end_c": pwr.gpu_temp_end_c,
        "throttled": pwr.throttled,
        "throttled_fraction": pwr.throttled_fraction,
        "power_capped": pwr.power_capped,
        "power_capped_fraction": pwr.power_capped_fraction,
        # quality
        "reference_id": qual.reference_id,
        "eval_stream_sha": qual.eval_stream_sha,
        "ppl_wikitext2": qual.ppl_wikitext2,
        "ppl_reference": qual.ppl_reference,
        "kl_estimator": qual.kl_div.estimator,
        "token_stream_verified": qual.token_stream_verified,
        "lm_eval_limit": qual.lm_eval_limit,
        "kl_mean": qual.kl_div.mean,
        "kl_p90": qual.kl_div.p90,
        "kl_p99": qual.kl_div.p99,
        "kl_max": qual.kl_div.max,
        "kl_tail_estimated": qual.kl_div.tail_mass_estimated,
        "top1_agreement": qual.top1_agreement,
        "top5_agreement": qual.top5_agreement,
        "scored_by_engine": qual.scored_by_engine,
        "calibration_disjoint": qual.calibration_disjoint_from_eval,
        # provenance
        "driver": record.environment.driver,
        "cuda_runtime": record.environment.cuda_runtime,
        "torch": record.environment.torch,
        "gpu_name": record.environment.gpu_name,
        "is_wsl": record.environment.is_wsl,
        "timestamp_utc": record.environment.timestamp_utc,
        "n_warnings": len(record.warnings),
        "error": record.error,
    }
    for task, value in (qual.lm_eval or {}).items():
        row[f"lm_eval_{task}"] = value
    return row


def to_dataframe(records: list[RunRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["run_id", "status", "format_variant", "engine"])
    frame = pd.DataFrame([record_to_row(r) for r in records])
    return frame.sort_values(["model", "format_variant", "engine", "batch_size", "prompt_tokens"])


def consolidate(
    results: Path, out: Path | None = None, allow_synthetic: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """Load every record and optionally write ``results/results.parquet``."""
    records, problems = load_records(results, allow_synthetic=allow_synthetic)
    frame = to_dataframe(records)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
    return frame, problems


def read_frame(path: Path, allow_synthetic: bool = False) -> pd.DataFrame:
    """Read a consolidated parquet (or a results directory) for plotting/reporting."""
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame, _ = consolidate(path, out=None, allow_synthetic=allow_synthetic)
    if "synthetic" in frame.columns and frame["synthetic"].any() and not allow_synthetic:
        raise MixedDatasetError(
            "this dataset contains synthetic (mock-engine) records; pass allow_synthetic "
            "to work with them — figures built from them are watermarked"
        )
    return frame


def measured_only(frame: pd.DataFrame, exclude_throttled: bool = True) -> pd.DataFrame:
    """Rows fit for publication: status ok, and (by default) not thermally contaminated."""
    out = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
    if exclude_throttled and "throttled" in out.columns:
        out = out[~out["throttled"].fillna(False).astype(bool)]
    return out
