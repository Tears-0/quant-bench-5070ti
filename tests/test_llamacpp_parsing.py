"""Parsers for llama.cpp's logs and files, pinned against text captured from build b11706.

These are the brittle edge of the harness: llama.cpp changes its banners and log lines
between builds, and a silent parse failure turns a measured value into ``None``. Every
fixture below was copied from a real b11706 run on the target card.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantbench.engines.llamacpp import (
    parse_kv_mib,
    parse_load_facts,
    parse_perplexity_output,
    parse_version,
)
from quantbench.planner import build_plan
from quantbench.reference.logits_cache import read_kld_header

SERVER_LOG = """\
0.00.145.079 I cmn  common_param: system_info: n_threads = 24 | CUDA : ARCHS = 750,800,860,890,900,1200,1210 | USE_GRAPHS = 1 |
0.00.550.568 I load_tensors: offloaded 37/37 layers to GPU
0.00.550.572 I load_tensors:   CPU_Mapped model buffer size =   304.28 MiB
0.00.550.572 I load_tensors:        CUDA0 model buffer size =  2375.91 MiB
0.01.344.950 I llama_context: flash_attn            = enabled
0.01.345.357 I llama_context:  CUDA_Host  output buffer size =     0.58 MiB
0.01.345.545 I llama_kv_cache:      CUDA0 KV buffer size =   108.00 MiB
0.01.347.261 I llama_kv_cache: size =  108.00 MiB (   768 cells,  36 layers,  1/1 seqs), K (f16):   54.00 MiB, V (f16):   54.00 MiB
0.01.560.355 I sched_reserve:      CUDA0 compute buffer size =    77.76 MiB
0.01.560.357 I sched_reserve:  CUDA_Host compute buffer size =    20.77 MiB
"""

KLD_REPORT = """\
====== Perplexity statistics ======
Mean PPL(Q)                   :  12.345678 ±   0.123456
Mean PPL(base)                :  12.300000 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.95%
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   3.210000
99.9%   KLD:   0.900000
99.0%   KLD:   0.150000
90.0%   KLD:   0.030000
Median  KLD:   0.004000
RMS Δp    :  1.234 ± 0.012 %
Same top p: 96.500 ± 0.091 %
"""


def test_version_banners_old_and_new():
    assert parse_version("version: 0.7.0-dev (build 11706, commit f74e2fa6a)") == (
        "b11706",
        "f74e2fa6a",
    )
    assert parse_version("version: 4321 (abc1234)") == ("b4321", "abc1234")
    assert parse_version("no banner here") == (None, None)


def test_kv_is_engine_reported_and_summed_over_gpus_only():
    assert parse_kv_mib(SERVER_LOG) == pytest.approx(108.0)
    two_gpus = SERVER_LOG + "llama_kv_cache:      CUDA1 KV buffer size =   20.00 MiB\n"
    assert parse_kv_mib(two_gpus) == pytest.approx(128.0)
    assert parse_kv_mib("llama_new_context: KV self size  =   64.00 MiB") == pytest.approx(64.0)
    assert parse_kv_mib("nothing") is None


def test_load_facts():
    facts = parse_load_facts(SERVER_LOG)
    assert facts["fully_offloaded"] is True
    assert facts["layers_offloaded"] == "37/37"
    assert facts["flash_attn_resolved"] == "enabled"
    assert "1200" in str(facts["cuda_archs"])  # native sm_120 kernels
    assert facts["weights_gpu_mib_reported"] == pytest.approx(2375.91)  # host buffer excluded
    assert facts["weights_host_mib_reported"] == pytest.approx(304.28)
    assert facts["compute_gpu_mib_reported"] == pytest.approx(77.76)  # CUDA_Host excluded


def test_partial_offload_is_detected():
    facts = parse_load_facts("load_tensors: offloaded 20/37 layers to GPU")
    assert facts["fully_offloaded"] is False


def test_perplexity_kld_mode():
    out = parse_perplexity_output(KLD_REPORT)
    assert out["ppl"] == pytest.approx(12.345678)
    assert out["ppl_base"] == pytest.approx(12.3)
    assert out["kl_mean"] == pytest.approx(0.012345)
    assert out["kl_mean_unc"] == pytest.approx(0.000123)
    assert out["kl_p99"] == pytest.approx(0.15)
    assert out["kl_p90"] == pytest.approx(0.03)
    assert out["kl_max"] == pytest.approx(3.21)
    assert out["top1_agreement"] == pytest.approx(0.965)  # percent -> fraction


def test_perplexity_base_mode():
    out = parse_perplexity_output("Final estimate: PPL = 11.2345 +/- 0.08123")
    assert out == {"ppl": pytest.approx(11.2345)}


def test_kld_header_roundtrip(tmp_path):
    n_ctx, n_vocab, n_chunk = 8, 100, 3
    tokens = np.arange(n_ctx * n_chunk, dtype="<i4")
    path = tmp_path / "base.kld"
    path.write_bytes(
        b"_logits_"
        + np.array([n_ctx, n_vocab, n_chunk], dtype="<i4").tobytes()
        + tokens.tobytes()
        + b"\0" * 64
    )
    header = read_kld_header(path)
    assert (header.n_ctx, header.n_vocab, header.n_chunk) == (n_ctx, n_vocab, n_chunk)
    assert (header.tokens == tokens).all()
    # positions n_ctx/2 .. n_ctx-2 of each chunk are scored
    assert header.scored_positions == n_chunk * (n_ctx - 1 - n_ctx // 2)


def test_kld_header_rejects_other_files(tmp_path):
    path = tmp_path / "x.kld"
    path.write_bytes(b"notakld!" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a llama-perplexity"):
        read_kld_header(path)


def test_plan_kind_filter():
    plan = build_plan("smoke", kind="quality")
    assert plan.entries and {s.workload.kind for s in plan.entries} == {"quality"}
    assert any("kept" in note for note in plan.notes)
