"""Analysis: loading, the synthetic quarantine, Pareto extraction, statistical ties."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import make_record, make_spec
from quantbench.analysis import report, stats
from quantbench.analysis.load import (
    MixedDatasetError,
    consolidate,
    load_records,
    measured_only,
    read_frame,
    to_dataframe,
)
from quantbench.schema import RunStatus


def write(record, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.spec.run_id}.json"
    path.write_text(record.model_dump_json(), encoding="utf-8")
    return path


def test_load_and_flatten(tmp_path):
    raw = tmp_path / "raw"
    write(make_record(make_spec(run_id="a")), raw)
    write(make_record(make_spec(run_id="b", variant="Q5_K_M", bits=5), decode=80.0), raw)

    records, problems = load_records(tmp_path)
    assert len(records) == 2 and not problems

    frame = to_dataframe(records)
    assert set(frame["run_id"]) == {"a", "b"}
    assert frame["format_variant"].tolist() == ["gguf:Q4_K_M", "gguf:Q5_K_M"]
    assert "kl_mean" in frame.columns and "vram_peak_nvsmi_mb" in frame.columns


def test_synthetic_records_are_excluded_by_default(tmp_path):
    raw = tmp_path / "raw"
    write(make_record(make_spec(run_id="real")), raw)
    write(make_record(make_spec(run_id="fake"), synthetic=True), raw)

    records, problems = load_records(tmp_path)
    assert [r.spec.run_id for r in records] == ["real"]
    assert any("synthetic" in p for p in problems)

    permitted, _ = load_records(tmp_path, allow_synthetic=True)
    assert len(permitted) == 2


def test_read_frame_refuses_to_mix(tmp_path):
    raw = tmp_path / "raw"
    write(make_record(make_spec(run_id="fake"), synthetic=True), raw)
    frame, _ = consolidate(tmp_path, out=tmp_path / "r.parquet", allow_synthetic=True)
    assert frame["synthetic"].all()
    with pytest.raises(MixedDatasetError):
        read_frame(tmp_path / "r.parquet")


def test_corrupt_and_future_records_are_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.json").write_text("{", encoding="utf-8")
    payload = json.loads(make_record().model_dump_json())
    payload["record_version"] = 99
    (raw / "future.json").write_text(json.dumps(payload), encoding="utf-8")

    records, problems = load_records(tmp_path)
    assert not records
    assert len(problems) == 2


def test_measured_only_drops_failures_and_throttled(tmp_path):
    raw = tmp_path / "raw"
    write(make_record(make_spec(run_id="ok")), raw)
    write(make_record(make_spec(run_id="oomed"), status=RunStatus.oom), raw)
    hot = make_record(make_spec(run_id="hot"))
    hot.power.throttled = True
    write(hot, raw)

    frame = to_dataframe(load_records(tmp_path)[0])
    assert set(measured_only(frame)["run_id"]) == {"ok"}
    assert set(measured_only(frame, exclude_throttled=False)["run_id"]) == {"ok", "hot"}


def test_pareto_frontier():
    frame = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "kl_mean": [0.01, 0.02, 0.03, 0.005],
            "decode_tok_s_p50": [100.0, 90.0, 150.0, 50.0],
        }
    )
    out = stats.pareto_frontier(frame, x="kl_mean", y="decode_tok_s_p50")
    frontier = set(out[out["pareto"]]["name"])
    assert "d" in frontier and "a" in frontier and "c" in frontier
    assert "b" not in frontier, "b is dominated by a (worse KL, slower)"


def test_ranking_marks_statistical_ties():
    frame = pd.DataFrame(
        {
            "name": ["fast", "same", "slow"],
            "decode_tok_s_p50": [100.0, 99.0, 60.0],
            "decode_tok_s_iqr": [8.0, 8.0, 2.0],
        }
    )
    ranked = stats.rank_with_ties(frame, "decode_tok_s_p50", "decode_tok_s_iqr")
    tied = dict(zip(ranked["name"], ranked["tied_with_leader"], strict=False))
    assert tied["fast"] and tied["same"]
    assert not tied["slow"]


def test_intervals_overlap_logic():
    a = stats.Interval(p50=10.0, iqr=2.0)
    b = stats.Interval(p50=10.5, iqr=2.0)
    c = stats.Interval(p50=50.0, iqr=1.0)
    assert stats.indistinguishable(a, b)
    assert not stats.indistinguishable(a, c)
    assert stats.indistinguishable(stats.Interval(1.0, None), stats.Interval(1.0, None))


def test_quality_latency_join_keeps_both_sides(tmp_path):
    raw = tmp_path / "raw"
    write(make_record(make_spec(run_id="lat", kind="latency"), decode=120.0), raw)
    quality_spec = make_spec(run_id="qual", kind="quality", generate_tokens=0)
    write(make_record(quality_spec, kl=0.02), raw)

    frame = to_dataframe(load_records(tmp_path)[0])
    joined = stats.quality_vs_latency(frame)
    assert len(joined) == 1
    assert joined.iloc[0]["kl_mean"] == pytest.approx(0.02)
    assert joined.iloc[0]["decode_tok_s_p50"] == pytest.approx(120.0)


def test_recommendation_table_is_empty_without_data():
    table = stats.recommendation_table(pd.DataFrame())
    assert table.empty or table["recommendation"].isna().all()


def test_report_renders_missing_values_as_dashes():
    assert report.fmt(None) == "—"
    assert report.fmt(float("nan")) == "—"
    assert report.fmt(1.234, digits=2) == "1.23"
    assert "No data yet" in report.markdown_table(pd.DataFrame())


def test_report_updates_only_marked_blocks(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "intro\n<!-- BEGIN quantbench:recommendations -->\nold\n"
        "<!-- END quantbench:recommendations -->\noutro\n",
        encoding="utf-8",
    )
    sections = report.build_sections(pd.DataFrame())
    updated = report.apply_sections(path, sections)
    text = path.read_text(encoding="utf-8")
    assert "recommendations" in updated
    assert text.startswith("intro") and text.rstrip().endswith("outro")
    assert "old" not in text


def test_hypotheses_report_says_insufficient_data_when_empty():
    section = report.hypotheses_section(pd.DataFrame())
    assert "insufficient data" in section.markdown
    assert "H2" in section.markdown


def test_hypothesis_evidence_checks_data_not_columns():
    records = [
        make_record(make_spec(run_id="lat-q4", variant="Q4_K_M", bits=4)),
        make_record(make_spec(run_id="lat-q8", variant="Q8_0", bits=8)),
        make_record(make_spec(run_id="q-q4", variant="Q4_K_M", kind="quality", generate_tokens=0)),
        make_record(make_spec(run_id="q-q8", variant="Q8_0", kind="quality", generate_tokens=0)),
    ]
    evidence = report.hypothesis_evidence(to_dataframe(records))
    assert evidence["H1"].startswith("insufficient") and "Q5_K_M" in evidence["H1"]
    assert evidence["H2"].startswith("preliminary (n=2")
    assert evidence["H3"].startswith("insufficient") and "AWQ/GPTQ" in evidence["H3"]
    assert evidence["H4"].startswith("insufficient")


def test_recommendation_note_names_its_candidates_and_vram_mode():
    records = [
        make_record(make_spec(run_id="lat-q4", variant="Q4_K_M", bits=4)),
        make_record(make_spec(run_id="q-q4", variant="Q4_K_M", kind="quality", generate_tokens=0)),
    ]
    section = report.recommendation_section(to_dataframe(records))
    assert "Chosen from 1 candidate configuration(s) only" in section.markdown
    assert "gguf:Q4_K_M (mock)" in section.markdown
