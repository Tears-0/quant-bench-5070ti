"""Planning: expansion, ordering, dedupe and resume (§6.2, §7.3)."""

from __future__ import annotations

import json
from itertools import pairwise

from conftest import make_record, make_spec
from quantbench.planner import Plan, build_plan, filter_resumable, make_run_id, order_specs


def test_profiles_expand():
    smoke = build_plan("smoke")
    assert smoke.entries, "the smoke profile must produce runs"
    assert {s.workload.kind for s in smoke.entries} == {"latency", "quality"}
    assert all(s.protocol.measured_iters >= 1 for s in smoke.entries)

    core = build_plan("core")
    assert len(core.entries) > len(smoke.entries)
    # batch 1 must be present: it is the regime most local users actually run
    assert any(s.workload.batch_size == 1 for s in core.entries)


def test_quality_runs_generate_no_tokens():
    for spec in build_plan("smoke").entries:
        if spec.workload.kind == "quality":
            assert spec.workload.generate_tokens == 0
            assert spec.workload.temperature == 0.0
            assert spec.workload.eval_config == "wikitext2"


def test_scoring_routes_are_declared_up_front():
    core = build_plan("core")
    # vLLM-served formats must have their quality routed to a scoring engine (§7.2)
    assert any("awq" in route for route in core.scoring_routes)
    assert set(core.scoring_routes.values()) == {"transformers"}


def test_conditional_formats_are_noted_not_silently_included():
    full = build_plan("full")
    assert any("conditional" in note for note in full.notes)


def test_ordering_is_seeded_and_deterministic():
    specs = build_plan("core").entries
    first = order_specs(specs, "shuffle", seed=1337)
    second = order_specs(specs, "shuffle", seed=1337)
    third = order_specs(specs, "shuffle", seed=99)
    assert [s.run_id for s in first] == [s.run_id for s in second]
    assert [s.run_id for s in first] != [s.run_id for s in third]


def test_shuffle_breaks_format_blocking():
    specs = build_plan("core").entries
    shuffled = order_specs(specs, "shuffle", seed=1337)
    formats = [s.quant.variant for s in shuffled]
    # A blocked order would have very few transitions between adjacent formats.
    transitions = sum(1 for a, b in pairwise(formats) if a != b)
    assert transitions > len(formats) * 0.5


def test_round_robin_interleaves_formats():
    specs = build_plan("core").entries
    rr = order_specs(specs, "round_robin", seed=1337)
    variants = [f"{s.quant.format}:{s.quant.variant}" for s in rr][:6]
    assert len(set(variants)) >= 5


def test_plan_is_deduplicated():
    plan = build_plan("core")
    hashes = [s.hash() for s in plan.entries]
    assert len(hashes) == len(set(hashes))


def test_run_ids_are_unique_and_readable():
    plan = build_plan("core")
    ids = [s.run_id for s in plan.entries]
    assert len(ids) == len(set(ids))
    assert all("__" in i for i in ids)


def test_resume_skips_completed_specs(tmp_path):
    spec = make_spec()
    record = make_record(spec)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / f"{spec.run_id}.json").write_text(record.model_dump_json(), encoding="utf-8")

    other = make_spec(run_id="other", batch_size=8)
    todo, skipped = filter_resumable([spec, other], raw, record.environment_hash)
    assert [s.run_id for s in skipped] == [spec.run_id]
    assert [s.run_id for s in todo] == ["other"]


def test_resume_ignores_corrupt_records(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "broken.json").write_text("{not json", encoding="utf-8")
    todo, skipped = filter_resumable([make_spec()], raw, "sha256:env")
    assert len(todo) == 1 and not skipped


def test_plan_save_load_roundtrip(tmp_path):
    plan = build_plan("smoke")
    path = plan.save(tmp_path / "plan.json")
    restored = Plan.load(path)
    assert [s.hash() for s in restored.entries] == [s.hash() for s in plan.entries]
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == plan.seed


def test_engine_override_rewrites_every_spec():
    plan = build_plan("smoke", engine_override="mock")
    assert {s.engine.name for s in plan.entries} == {"mock"}


def test_run_id_encodes_swept_variables():
    spec = make_spec()
    run_id = make_run_id(
        type("M", (), {"name": "Qwen3-8B"})(),
        type("F", (), {"family": "gguf"})(),
        "Q4_K_M",
        "llama.cpp",
        spec.workload,
        spec.engine.model_copy(update={"kv_cache_dtype": "q4_0", "attention_backend": "triton"}),
    )
    assert "kv-q4-0" in run_id and "attn-triton" in run_id
