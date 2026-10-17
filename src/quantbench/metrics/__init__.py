"""Measurement primitives: latency, memory, power, quality.

Everything in this package is either a pure function over samples (unit-testable without a
GPU) or a thin NVML sampler. The engines produce raw timings; these modules turn them into
the numbers defined in requirements.md §5.
"""
