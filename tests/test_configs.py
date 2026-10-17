"""Every shipped config must load, and the matrix must be internally consistent."""

from __future__ import annotations

import pytest

from quantbench.config import (
    list_configs,
    load_eval,
    load_format,
    load_hardware,
    load_model,
    load_profile,
    parse_format_ref,
)
from quantbench.engines.base import ENGINE_REGISTRY


def test_hardware_config_matches_the_target_card():
    hw = load_hardware("5070ti")
    assert hw.required_torch_arch == "sm_120"
    assert hw.compute_capability == "12.0"
    assert hw.sample_hz >= 10.0, "power needs >=10 Hz (§5.4) and memory >=5 Hz (§5.3)"


@pytest.mark.parametrize("name", list_configs("models"))
def test_models_load_and_declare_a_tier(name):
    model = load_model(name)
    assert model.tier in {"A", "B", "C", "D"}
    assert model.slug == name
    # Parameter counts are filled in from the checkpoint, never guessed (§5.3).
    assert model.params_non_embedding is None or model.params_non_embedding > 0


@pytest.mark.parametrize("name", list_configs("formats"))
def test_formats_reference_known_engines(name):
    fmt = load_format(name)
    assert fmt.variants, "a format with no variants cannot be planned"
    for engine in fmt.supported_engines + fmt.default_engines:
        assert engine in ENGINE_REGISTRY, f"{name}: unknown engine {engine}"
    if fmt.scoring_engine:
        assert fmt.scoring_engine in ENGINE_REGISTRY
    assert set(fmt.default_engines) <= set(fmt.supported_engines)


@pytest.mark.parametrize("name", list_configs("eval"))
def test_eval_configs_pin_their_chunking(name):
    cfg = load_eval(name)
    assert cfg.ppl.ctx > 0 and cfg.ppl.stride > 0
    assert cfg.reference.topk >= 1


@pytest.mark.parametrize("name", list_configs("profiles"))
def test_profiles_are_coherent(name):
    profile = load_profile(name)
    assert profile.cells
    for cell in profile.cells:
        for ref in cell.formats:
            family, variant = parse_format_ref(ref)
            fmt = load_format(family)
            fmt.variant(variant)  # raises if the variant is not declared
            if cell.kv_cache_dtypes:
                assert set(cell.kv_cache_dtypes) <= set(fmt.kv_cache_dtypes), (
                    f"{name}/{cell.name}: {family} does not support {cell.kv_cache_dtypes}"
                )
        for model in cell.models:
            load_model(model)


def test_imatrix_ablation_pair_exists():
    """The isolated-imatrix comparison is a headline result; the pair must be configured."""
    gguf = load_format("gguf")
    ids = {v.id: v for v in gguf.variants}
    assert "Q4_K_M" in ids and "Q4_K_M-imat" in ids
    assert ids["Q4_K_M"].imatrix is False
    assert ids["Q4_K_M-imat"].imatrix is True
    assert ids["Q4_K_M"].bits_nominal == ids["Q4_K_M-imat"].bits_nominal


def test_calibration_is_declared_disjoint_from_evaluation():
    wikitext = load_eval("wikitext2")
    assert wikitext.calibration_disjoint_from, "§6.7 requires an explicit disjointness rule"
    for name in list_configs("formats"):
        for variant in load_format(name).variants:
            if variant.calibration_dataset:
                assert not any(
                    banned in variant.calibration_dataset.lower()
                    for banned in wikitext.calibration_disjoint_from
                ), f"{name}:{variant.id} calibrates on the evaluation corpus"


def test_conditional_formats_are_marked():
    assert load_format("fp8").conditional is True, "FP8 on sm_120 is unverified (§3.2)"
