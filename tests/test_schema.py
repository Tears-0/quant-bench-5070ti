"""Schema round-trips and the guarantees the analysis layer relies on (§10 M0)."""

from __future__ import annotations

import json

import pytest

from conftest import make_record, make_spec
from quantbench.schema import (
    RECORD_VERSION,
    RunRecord,
    RunSpec,
    RunStatus,
    SchemaVersionError,
    load_record,
)


def test_runspec_roundtrip():
    spec = make_spec()
    restored = RunSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    assert restored.hash() == spec.hash()


def test_runrecord_roundtrip():
    record = make_record()
    restored = load_record(json.loads(record.model_dump_json()))
    assert restored.spec_hash == record.spec_hash
    assert restored.latency.itl_ms.p99 == record.latency.itl_ms.p99
    assert restored.status is RunStatus.ok


def test_spec_hash_ignores_run_id_but_tracks_content():
    a = make_spec(run_id="name-a")
    b = make_spec(run_id="name-b")
    assert a.hash() == b.hash(), "run_id is cosmetic; experiment identity is the content"

    c = make_spec(run_id="name-a", batch_size=4)
    assert a.hash() != c.hash()


def test_environment_hash_ignores_timestamp_and_host():
    record = make_record()
    other = record.environment.model_copy(
        update={"timestamp_utc": "2030-01-01T00:00:00Z", "hostname": "elsewhere"}
    )
    assert other.hash() == record.environment.hash()

    upgraded = record.environment.model_copy(update={"driver": "999.99"})
    assert upgraded.hash() != record.environment.hash()


def test_record_version_is_enforced():
    payload = json.loads(make_record().model_dump_json())
    payload["record_version"] = RECORD_VERSION + 1
    with pytest.raises(SchemaVersionError):
        load_record(payload)

    del payload["record_version"]
    with pytest.raises(SchemaVersionError):
        load_record(payload)


def test_quality_workloads_must_be_greedy():
    with pytest.raises(ValueError, match="greedy"):
        make_spec(kind="quality").workload.model_copy(update={"temperature": 0.7})
        RunSpec.model_validate(
            {
                **make_spec(kind="quality").model_dump(),
                "workload": {
                    **make_spec(kind="quality").workload.model_dump(),
                    "temperature": 0.7,
                },
            }
        )


def test_missing_metrics_are_none_not_zero():
    record = RunRecord(spec_hash="sha256:x", spec=make_spec(), status=RunStatus.oom)
    assert record.memory.vram_peak_nvsmi_mb is None
    assert record.quality.kl_div.mean is None
    assert record.latency.ttft_ms.p50 is None
    assert record.power.energy_j is None


def test_unknown_fields_are_rejected():
    payload = json.loads(make_record().model_dump_json())
    payload["totally_new_field"] = 1
    with pytest.raises(ValueError, match="totally_new_field"):
        load_record(payload)
