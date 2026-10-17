"""Small builders shared across test modules."""

from __future__ import annotations

from quantbench.metrics.memory import GpuSample


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
