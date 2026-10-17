"""End-to-end pipeline on the mock engine: subprocess isolation, classification, resume.

This is the CI proof that the harness runs without a GPU (§0.5, §11.3).
"""

from __future__ import annotations

import json

import pytest

from conftest import make_spec
from quantbench.engines.base import load_engine
from quantbench.planner import Plan
from quantbench.runner import Runner, RunnerOptions
from quantbench.schema import RunStatus
from quantbench.worker import Worker


def test_worker_produces_a_valid_record_in_process():
    record = Worker(make_spec()).run()
    assert record.status is RunStatus.ok
    assert record.synthetic is True
    assert record.latency.ttft_ms.p50 is not None
    assert record.latency.decode_tok_s.p50 is not None
    assert record.latency.per_token_timestamps_ms
    # The mock spec has no artifact on disk, so bpw cannot be measured and is not guessed.
    assert record.memory.effective_bpw is None
    assert record.engine_metadata["name"] == "mock"


def test_worker_is_deterministic_for_a_given_spec():
    first = Worker(make_spec()).run()
    second = Worker(make_spec()).run()
    assert first.latency.ttft_ms.p50 == pytest.approx(second.latency.ttft_ms.p50)


def test_generated_token_count_is_checked():
    record = Worker(make_spec(generate_tokens=16)).run()
    assert all(len(s) == 16 for s in record.latency.per_token_timestamps_ms)
    assert not any("throughput may be inflated" in w for w in record.warnings)


def test_unknown_engine_is_recorded_not_raised():
    record = Worker(make_spec(engine="does-not-exist")).run()
    assert record.status is RunStatus.engine_unsupported
    assert "unknown engine" in (record.error or "")


def test_engine_registry_is_lazy():
    engine = load_engine("mock")
    assert engine.name == "mock"


def test_runner_executes_a_plan_in_subprocesses(results_dir, tmp_path):
    plan = Plan(
        profile="test",
        seed=1,
        order="as_listed",
        created_utc="2026-01-01T00:00:00Z",
        quantbench_version="0.1.0",
        entries=[make_spec(run_id="run-a"), make_spec(run_id="run-b", batch_size=2)],
    )
    runner = Runner(RunnerOptions(dry_run=True, cooldown=False))
    result = runner.run_plan(plan)

    assert len(result.records) == 2
    assert result.counts == {"ok": 2}
    written = sorted(p.name for p in result.output_dir.glob("*.json"))
    assert written == ["run-a.json", "run-b.json"]
    payload = json.loads((result.output_dir / "run-a.json").read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["record_version"] == 1


def test_dry_run_output_is_quarantined(results_dir):
    plan = Plan(
        profile="test",
        seed=1,
        order="as_listed",
        created_utc="2026-01-01T00:00:00Z",
        quantbench_version="0.1.0",
        entries=[make_spec(run_id="run-a")],
    )
    result = Runner(RunnerOptions(dry_run=True, cooldown=False)).run_plan(plan)
    assert "dryrun" in str(result.output_dir)
    assert not list((results_dir / "raw").glob("*.json"))


def test_resume_skips_completed_runs(results_dir):
    plan = Plan(
        profile="test",
        seed=1,
        order="as_listed",
        created_utc="2026-01-01T00:00:00Z",
        quantbench_version="0.1.0",
        entries=[make_spec(run_id="run-a")],
    )
    runner = Runner(RunnerOptions(dry_run=True, cooldown=False))
    first = runner.run_plan(plan)
    assert len(first.records) == 1

    second = runner.run_plan(plan)
    assert len(second.records) == 0
    assert len(second.skipped) == 1

    forced = Runner(RunnerOptions(dry_run=True, cooldown=False, force=True)).run_plan(plan)
    assert len(forced.records) == 1


def test_failure_classification_from_worker_output():
    runner = Runner(RunnerOptions(dry_run=True, cooldown=False))
    spec = make_spec()
    oom = runner._synthesize_failure(spec, "", "CUDA error: out of memory", 1, False, 1.0)
    assert oom.status is RunStatus.oom
    build = runner._synthesize_failure(spec, "", "no kernel image is available", 1, False, 1.0)
    assert build.status is RunStatus.build_failed
    timeout = runner._synthesize_failure(spec, "", "", -1, True, 1.0)
    assert timeout.status is RunStatus.timeout
    crash = runner._synthesize_failure(spec, "", "Segmentation fault", -11, False, 1.0)
    assert crash.status is RunStatus.crashed
    assert crash.stderr_tail == "Segmentation fault"


def test_quality_workload_without_a_stream_is_unsupported_not_crashed(monkeypatch):
    spec = make_spec(kind="quality", engine="mock").model_copy()
    # force the real (non-synthetic) path
    record = Worker(spec)
    record.record.synthetic = False
    out = record.run()
    assert out.status is RunStatus.engine_unsupported
    assert "quantbench reference build" in (out.error or "")


def test_throttled_invalid_is_not_overwritten_by_ok(monkeypatch):
    """Regression: run() used to set status=ok *after* the workload had downgraded it."""
    from quantbench.metrics import power as power_metrics
    from quantbench.schema import PowerRecord

    def hot(*_args, **_kwargs):
        return PowerRecord(throttled=True, throttled_fraction=0.9, throttle_reasons=["hw_slowdown"])

    monkeypatch.setattr(power_metrics, "summarize_power", hot)
    record = Worker(make_spec()).run()
    assert record.status is RunStatus.throttled_invalid
    assert any("invalidating throttle" in w for w in record.warnings)
