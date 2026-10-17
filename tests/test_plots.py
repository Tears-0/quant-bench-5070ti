"""Figure regeneration: deterministic, GPU-free, and honest about missing inputs (§9.2)."""

from __future__ import annotations

import hashlib

from conftest import make_record, make_spec
from quantbench.analysis.load import to_dataframe
from quantbench.analysis.plots import FIGURES, generate_all, results_table


def rich_frame():
    """A fixture dataset broad enough to exercise every figure."""
    records = []
    for variant, bits, decode, kl in (
        ("Q8_0", 8, 60.0, 0.001),
        ("Q5_K_M", 5, 80.0, 0.01),
        ("Q4_K_M", 4, 95.0, 0.03),
    ):
        for batch in (1, 4):
            for prompt in (512, 2048):
                records.append(
                    make_record(
                        make_spec(
                            run_id=f"lat-{variant}-b{batch}-p{prompt}",
                            variant=variant,
                            bits=bits,
                            batch_size=batch,
                            prompt_tokens=prompt,
                        ),
                        decode=decode * (1 + 0.1 * batch),
                    )
                )
        records.append(
            make_record(
                make_spec(
                    run_id=f"qual-{variant}",
                    variant=variant,
                    bits=bits,
                    kind="quality",
                    generate_tokens=0,
                ),
                kl=kl,
                ppl=9.0 + kl * 20,
                lm_eval={
                    "arc_challenge": 0.50 - kl,
                    "arc_challenge_stderr": 0.01,
                    "hellaswag": 0.70 - kl,
                    "hellaswag_stderr": 0.01,
                },
            )
        )
    # the FP16 anchor, an OOM, and the imatrix pair
    records.append(
        make_record(
            make_spec(
                run_id="qual-bf16",
                fmt="bf16",
                variant="reference",
                bits=16,
                kind="quality",
                generate_tokens=0,
            ),
            kl=0.0,
            ppl=9.0,
            lm_eval={
                "arc_challenge": 0.50,
                "arc_challenge_stderr": 0.01,
                "hellaswag": 0.70,
                "hellaswag_stderr": 0.01,
            },
        )
    )
    records.append(
        make_record(
            make_spec(run_id="oom-14b", variant="Q4_K_M", batch_size=8, prompt_tokens=32768),
            status=__import__("quantbench.schema", fromlist=["RunStatus"]).RunStatus.oom,
        )
    )
    records.append(
        make_record(
            make_spec(
                run_id="qual-Q4_K_M-imat",
                variant="Q4_K_M",
                bits=4,
                kind="quality",
                generate_tokens=0,
                imatrix=True,
            ),
            kl=0.02,
        )
    )
    return to_dataframe(records)


def test_every_figure_is_produced_from_a_complete_dataset(tmp_path):
    results = generate_all(rich_frame(), tmp_path)
    assert len(results) == len(FIGURES)
    skipped = {r.name: r.reason for r in results if r.skipped}
    assert not skipped, f"unexpected skips: {skipped}"
    for result in results:
        assert [p.suffix for p in result.paths] == [".svg", ".png"]
        assert all(p.exists() and p.stat().st_size > 0 for p in result.paths)


def test_regeneration_is_byte_stable(tmp_path):
    frame = rich_frame()
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_all(frame, first)
    generate_all(frame, second)
    for path in sorted(first.glob("*")):
        other = second / path.name
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == hashlib.sha256(other.read_bytes()).hexdigest()
        ), f"{path.name} is not deterministic"


def test_missing_inputs_skip_rather_than_invent(tmp_path):
    frame = to_dataframe([make_record(make_spec(run_id="only-one"))])
    results = generate_all(frame, tmp_path)
    skipped = [r for r in results if r.skipped]
    assert skipped, "a one-row dataset cannot support every figure"
    assert all(r.reason for r in skipped), "every skip must state why"
    assert not any((tmp_path / f"{r.name}.svg").exists() for r in skipped)


def test_synthetic_frames_are_watermarked(tmp_path):
    frame = to_dataframe([make_record(make_spec(run_id="fake"), synthetic=True)])
    generate_all(frame, tmp_path)
    svgs = list(tmp_path.glob("*.svg"))
    assert svgs, "at least one figure should render"
    assert any("SYNTHETIC DATA" in p.read_text(encoding="utf-8") for p in svgs)


def test_results_table_shape(tmp_path):
    rows = results_table(generate_all(rich_frame(), tmp_path))
    assert {"figure", "status", "n", "reason"} == set(rows[0])
