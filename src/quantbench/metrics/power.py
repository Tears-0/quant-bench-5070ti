"""Power integration and throttle classification (requirements.md §5.4).

Pure functions over the sample series produced by :mod:`quantbench.metrics.memory`, so the
whole efficiency path is unit-testable on a machine with no GPU.

A run during which thermal or power throttling was active is flagged. The analysis layer
can exclude flagged runs; it never silently averages them in.
"""

from __future__ import annotations

from itertools import pairwise

from quantbench.metrics.memory import GpuSample
from quantbench.schema import PowerRecord

# NVML clocks-throttle-reason bits (stable across NVML versions).
THROTTLE_BITS: dict[str, int] = {
    "gpu_idle": 0x0000000000000001,
    "applications_clocks_setting": 0x0000000000000002,
    "sw_power_cap": 0x0000000000000004,
    "hw_slowdown": 0x0000000000000008,
    "sync_boost": 0x0000000000000010,
    "sw_thermal_slowdown": 0x0000000000000020,
    "hw_thermal_slowdown": 0x0000000000000040,
    "hw_power_brake_slowdown": 0x0000000000000080,
    "display_clock_setting": 0x0000000000000100,
}

#: Reasons that invalidate a comparison: the card slowed itself for thermal or electrical
#: protection, which drifts over a session and would be confounded with run order.
INVALIDATING_REASONS: frozenset[str] = frozenset(
    {
        "hw_slowdown",
        "sw_thermal_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
)

#: ``sw_power_cap`` is GPU Boost regulating clocks against the board power limit. On the
#: RTX 5070 Ti it is asserted under essentially any sustained load (observed at ~205 W avg
#: against a 300 W limit), so it is recorded as ``power_capped`` with its own fraction —
#: flagged, never hidden — but it is the card's normal operating regime, not contamination.
#: ``gpu_idle`` is normal between iterations; ``applications_clocks_setting`` means clocks
#: were locked, which is recorded as a protocol field. See docs/open-questions.md Q13.
POWER_CAP_REASONS: frozenset[str] = frozenset({"sw_power_cap"})

_KNOWN_MASK = 0
for _bit in THROTTLE_BITS.values():
    _KNOWN_MASK |= _bit


def decode_throttle(bits: int | None) -> list[str]:
    """Bitmask -> sorted reason names. Bits this NVML binding does not name are kept as
    ``unknown_0x...`` rather than dropped (newer drivers define reasons older bindings
    lack; driver 617 sets 0x400)."""
    if not bits:
        return []
    names = [name for name, bit in THROTTLE_BITS.items() if bits & bit]
    unknown = bits & ~_KNOWN_MASK
    bit = 1
    while unknown:
        if unknown & bit:
            names.append(f"unknown_{bit:#x}")
            unknown &= ~bit
        bit <<= 1
    return sorted(names)


#: Fraction of the measured window that must show an invalidating throttle reason before
#: the run is downgraded from "flagged" to ``status=throttled_invalid``. A brief power-cap
#: blip is worth flagging; a run spent mostly in thermal slowdown is not a measurement.
THROTTLE_INVALID_FRACTION = 0.25


def throttled_fraction(
    samples: list[GpuSample], reasons: frozenset[str] = INVALIDATING_REASONS
) -> float:
    """Fraction of samples during which any of ``reasons`` was active."""
    usable = [s for s in samples if s.throttle_bits is not None]
    if not usable:
        return 0.0
    hits = sum(1 for s in usable if set(decode_throttle(s.throttle_bits)) & reasons)
    return hits / len(usable)


def integrate_energy_j(samples: list[GpuSample]) -> float | None:
    """Trapezoidal integration of instantaneous power over the sampled window.

    Returns ``None`` for fewer than two usable samples: a single power reading over an
    unknown interval is not an energy measurement.
    """
    points = [(s.t_s, s.power_w) for s in samples if s.power_w is not None]
    if len(points) < 2:
        return None
    energy = 0.0
    for (t0, p0), (t1, p1) in pairwise(points):
        dt = t1 - t0
        if dt <= 0:
            continue
        energy += 0.5 * (p0 + p1) * dt
    return energy


def summarize_power(
    samples: list[GpuSample],
    generated_tokens: int | None = None,
    window: tuple[float, float] | None = None,
) -> PowerRecord:
    """Summarise a sample series into a :class:`PowerRecord`.

    ``window`` restricts integration to the measured phase, excluding model load and
    warmup — energy spent loading weights is not energy spent generating tokens.
    """
    if window is not None:
        lo, hi = window
        samples = [s for s in samples if lo <= s.t_s <= hi]

    record = PowerRecord(samples_n=len(samples))
    if not samples:
        return record

    powers = [s.power_w for s in samples if s.power_w is not None]
    temps = [s.temp_c for s in samples if s.temp_c is not None]
    clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz is not None]

    if powers:
        record.power_avg_w = sum(powers) / len(powers)
        record.power_peak_w = max(powers)
    if temps:
        record.gpu_temp_start_c = temps[0]
        record.gpu_temp_end_c = temps[-1]
    if clocks:
        record.sm_clock_avg_mhz = sum(clocks) / len(clocks)

    record.energy_j = integrate_energy_j(samples)
    if record.energy_j and generated_tokens:
        record.tokens_per_joule = generated_tokens / record.energy_j

    reasons: set[str] = set()
    for s in samples:
        reasons.update(decode_throttle(s.throttle_bits))
    record.throttle_reasons = sorted(reasons)
    record.throttled = bool(reasons & INVALIDATING_REASONS)
    record.power_capped = bool(reasons & POWER_CAP_REASONS)
    record.throttled_fraction = throttled_fraction(samples)
    record.power_capped_fraction = throttled_fraction(samples, POWER_CAP_REASONS)

    span = samples[-1].t_s - samples[0].t_s
    if span > 0:
        record.sample_hz_effective = (len(samples) - 1) / span
    return record
