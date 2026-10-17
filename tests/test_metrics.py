"""Latency, power and memory maths (§5.1, §5.3, §5.4)."""

from __future__ import annotations

import math

import pytest

from quantbench.metrics import latency, power
from quantbench.metrics.memory import GpuSample, effective_bpw, peak_process_mb


def sample(t, power_w=None, temp=None, clock=None, throttle=None, dev=None, proc=None):
    return GpuSample(
        t_s=t,
        device_used_mb=dev,
        process_used_mb=proc,
        power_w=power_w,
        temp_c=temp,
        sm_clock_mhz=clock,
        throttle_bits=throttle,
    )


# -- latency ---------------------------------------------------------------------------


def test_ttft_and_itl_definitions():
    stamps = [10.0, 20.0, 33.0, 45.0]
    assert latency.ttft_ms(0.0, stamps) == 10.0
    assert latency.itl_series_ms(stamps) == [10.0, 13.0, 12.0]
    # decode rate excludes the first token by definition
    assert latency.decode_tok_s(stamps) == pytest.approx(3 / ((45.0 - 10.0) / 1000))


def test_prefill_conflates_scheduling_but_is_defined():
    assert latency.prefill_tok_s(2048, 100.0) == pytest.approx(20480.0)
    assert latency.prefill_tok_s(2048, 0.0) is None
    assert latency.prefill_tok_s(0, 100.0) is None


def test_single_token_generation_has_no_itl():
    assert latency.itl_series_ms([5.0]) == []
    assert latency.decode_tok_s([5.0]) is None


def test_describe_reports_dispersion_not_just_a_mean():
    dist = latency.describe([1.0, 2.0, 3.0, 4.0, 100.0])
    assert dist.p50 == 3.0
    assert dist.iqr == 2.0
    assert dist.max == 100.0
    assert dist.n == 5
    assert dist.p99 > dist.p90 > dist.p50


def test_describe_handles_empty_input():
    dist = latency.describe([])
    assert dist.n == 0 and dist.p50 is None


def test_summarize_iterations_pools_sequences_and_keeps_timestamps():
    it = latency.Iteration(
        submit_ms=0.0, sequences=[[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], prompt_tokens=64
    )
    record = latency.summarize_iterations([it, it])
    assert record.ttft_ms.n == 4
    assert record.itl_ms.n == 8
    # one throughput value per repetition, not per sequence
    assert record.throughput_tok_s_total.n == 2
    assert len(record.per_token_timestamps_ms) == 4
    assert record.per_token_timestamps_ms[0][0] == 10.0


# -- power -----------------------------------------------------------------------------


def test_energy_is_trapezoidal():
    samples = [sample(0.0, power_w=100.0), sample(1.0, power_w=200.0), sample(2.0, power_w=200.0)]
    assert power.integrate_energy_j(samples) == pytest.approx(150.0 + 200.0)


def test_energy_needs_two_points():
    assert power.integrate_energy_j([sample(0.0, power_w=100.0)]) is None
    assert power.integrate_energy_j([]) is None


def test_throttle_decoding_separates_flags_from_invalidation():
    assert power.decode_throttle(None) == []
    assert power.decode_throttle(0x1) == ["gpu_idle"]
    reasons = power.decode_throttle(0x20 | 0x4)
    assert reasons == ["sw_power_cap", "sw_thermal_slowdown"]
    # clocks locked is informative, not contamination
    assert not set(power.decode_throttle(0x2)) & power.INVALIDATING_REASONS
    # power-limit regulation is flagged separately, not treated as contamination
    assert "sw_power_cap" not in power.INVALIDATING_REASONS
    assert "sw_power_cap" in power.POWER_CAP_REASONS


def test_unknown_throttle_bits_are_kept_not_dropped():
    # driver 617 sets 0x400, which this NVML binding does not name
    assert power.decode_throttle(0x400) == ["unknown_0x400"]
    assert power.decode_throttle(0x404) == ["sw_power_cap", "unknown_0x400"]


def test_summarize_power_flags_throttling_and_efficiency():
    samples = [
        sample(0.0, power_w=100.0, temp=45.0, clock=2500.0, throttle=0),
        sample(1.0, power_w=300.0, temp=70.0, clock=2400.0, throttle=0x40),
    ]
    record = power.summarize_power(samples, generated_tokens=256)
    assert record.power_peak_w == 300.0
    assert record.gpu_temp_start_c == 45.0 and record.gpu_temp_end_c == 70.0
    assert record.energy_j == pytest.approx(200.0)
    assert record.tokens_per_joule == pytest.approx(256 / 200.0)
    assert record.throttled is True
    assert record.throttled_fraction == pytest.approx(0.5)
    assert "hw_thermal_slowdown" in record.throttle_reasons


def test_power_cap_is_flagged_but_not_contamination():
    samples = [sample(float(i), power_w=220.0, throttle=0x4) for i in range(10)]
    record = power.summarize_power(samples, generated_tokens=100)
    assert record.power_capped is True
    assert record.power_capped_fraction == pytest.approx(1.0)
    assert record.throttled is False
    assert record.throttled_fraction == 0.0


def test_throttled_fraction_and_window_restriction():
    samples = [sample(float(i), power_w=100.0, throttle=0x40 if i >= 8 else 0) for i in range(10)]
    assert power.throttled_fraction(samples) == pytest.approx(0.2)
    windowed = power.summarize_power(samples, window=(0.0, 5.0))
    assert windowed.throttled is False
    assert windowed.samples_n == 6


# -- memory ----------------------------------------------------------------------------


def test_effective_bpw_refuses_to_guess():
    assert effective_bpw(1_000_000_000, 2_000_000_000) == pytest.approx(4.0)
    assert effective_bpw(None, 2_000_000_000) is None
    assert effective_bpw(1_000_000_000, None) is None


def test_peak_process_falls_back_to_device_wide():
    with_proc = [sample(0.0, proc=100.0, dev=900.0), sample(1.0, proc=250.0, dev=950.0)]
    assert peak_process_mb(with_proc) == 250.0

    # WSL2 does not implement per-process accounting
    without_proc = [sample(0.0, dev=900.0), sample(1.0, dev=1200.0)]
    assert peak_process_mb(without_proc) == 1200.0
    assert peak_process_mb([]) is None


def test_no_metric_silently_becomes_zero():
    record = power.summarize_power([])
    assert record.power_avg_w is None
    assert record.energy_j is None
    assert not math.isnan(record.samples_n)
