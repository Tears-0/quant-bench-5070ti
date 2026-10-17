"""Parameter and weight-byte accounting read straight from checkpoint headers (§5.3)."""

from __future__ import annotations

import json
import struct

import pytest

from quantbench.checkpoint import (
    count_parameters,
    gguf_weight_bytes,
    safetensors_weight_bytes,
    update_model_config,
)
from quantbench.metrics.memory import effective_bpw, vram_accounting_mode
from tests_support import sample


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (shape, nbytes) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def gguf_string(s):
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, tensors, alignment=32):
    """Minimal GGUF v3: one string KV, one array KV, then tensor infos and padded data."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    out += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen3")
    out += gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
    out += gguf_string("a") + gguf_string("bc")
    offset, layout = 0, []
    for name, nbytes in tensors.items():
        out += gguf_string(name) + struct.pack("<I", 1) + struct.pack("<Q", nbytes)
        out += struct.pack("<IQ", 0, offset)
        layout.append(nbytes)
        offset += -(-nbytes // alignment) * alignment
    out += b"\0" * (-(-len(out) // alignment) * alignment - len(out))
    for nbytes in layout:
        out += b"\1" * nbytes + b"\0" * (-(-nbytes // alignment) * alignment - nbytes)
    path.write_bytes(out)


def test_count_parameters_excludes_embeddings(tmp_path):
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": ([100, 8], 1600),
            "model.layers.0.mlp.up_proj.weight": ([16, 8], 256),
            "lm_head.weight": ([100, 8], 1600),
        },
    )
    counts = count_parameters(tmp_path)
    assert counts.total == 800 + 128 + 800
    assert counts.embedding == 1600
    assert counts.non_embedding == 128


def test_safetensors_bytes(tmp_path):
    write_safetensors(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": ([4, 4], 32), "model.layers.0.w": ([4, 4], 32)},
    )
    weights = safetensors_weight_bytes(tmp_path)
    assert weights.total == 64 and weights.embedding == 32 and weights.non_embedding == 32


def test_gguf_bytes_by_offset(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(path, {"token_embd.weight": 64, "blk.0.attn_q.weight": 96, "output_norm.weight": 32})
    weights = gguf_weight_bytes(path)
    assert weights.per_tensor["token_embd.weight"] == 64
    assert weights.per_tensor["blk.0.attn_q.weight"] == 96
    assert weights.per_tensor["output_norm.weight"] == 32
    assert weights.embedding == 64
    assert weights.non_embedding == 128


def test_gguf_rejects_other_files(tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf_weight_bytes(path)


def test_effective_bpw_excludes_embedding_bytes():
    # Q8_0 is 34 bytes per 32 weights -> exactly 8.5 bpw.
    assert effective_bpw(34 * 1000, 32 * 1000) == pytest.approx(8.5)
    assert effective_bpw(None, 10) is None


def test_vram_accounting_mode():
    assert vram_accounting_mode([sample(0.0, proc=10.0, dev=900.0)]) == "per_process"
    assert vram_accounting_mode([sample(0.0, dev=900.0)]) == "device_wide"
    assert vram_accounting_mode([]) is None


def test_update_model_config_preserves_comments(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("# keep me\nname: X\nparams_total: null\nrevision: null\n", encoding="utf-8")
    changed = update_model_config(path, {"params_total": 42, "revision": "abc"})
    text = path.read_text(encoding="utf-8")
    assert changed == ["params_total", "revision"]
    assert text.startswith("# keep me") and "params_total: 42" in text and 'revision: "abc"' in text
