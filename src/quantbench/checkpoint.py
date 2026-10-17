"""Facts read directly from a downloaded checkpoint, never guessed.

``effective_bpw`` divides by the non-embedding parameter count (§5.3), so that count has to
come from the weights themselves. Safetensors files start with an 8-byte little-endian
header length followed by a JSON header listing every tensor's dtype and shape — enough to
count parameters exactly without loading a single weight or importing torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

#: Tensors excluded from the non-embedding count: input embeddings and the output head.
#: With tied embeddings (Qwen3-4B) the head is absent from the checkpoint and contributes 0.
EMBEDDING_PATTERNS = (re.compile(r"(^|\.)embed_tokens\."), re.compile(r"(^|\.)lm_head\."))


@dataclass
class ParamCount:
    total: int = 0
    embedding: int = 0
    embedding_tensors: list[str] = field(default_factory=list)
    n_tensors: int = 0

    @property
    def non_embedding(self) -> int:
        return self.total - self.embedding


def read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header: dict[str, dict[str, object]] = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return header


def count_parameters(checkpoint_dir: Path) -> ParamCount:
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {checkpoint_dir}")
    count = ParamCount()
    for file in files:
        for name, info in read_safetensors_header(file).items():
            shape = info.get("shape")
            numel = math.prod(int(d) for d in shape) if isinstance(shape, list) else 0
            count.total += numel
            count.n_tensors += 1
            if any(p.search(name) for p in EMBEDDING_PATTERNS):
                count.embedding += numel
                count.embedding_tensors.append(name)
    return count


def tokenizer_sha256(checkpoint_dir: Path) -> str | None:
    path = checkpoint_dir / "tokenizer.json"
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def update_model_config(config_path: Path, values: dict[str, object]) -> list[str]:
    """Set top-level ``key: value`` lines in a model YAML, preserving comments and order."""
    text = config_path.read_text(encoding="utf-8")
    changed: list[str] = []
    for key, value in values.items():
        rendered = (
            "null" if value is None else (f'"{value}"' if isinstance(value, str) else str(value))
        )
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
        if pattern.search(text):
            text, n = pattern.subn(f"{key}: {rendered}", text, count=1)
            if n:
                changed.append(key)
    config_path.write_text(text, encoding="utf-8")
    return changed


# --------------------------------------------------------------------------------------
# Weight bytes, with embeddings separable (§5.3 effective_bpw)
# --------------------------------------------------------------------------------------

#: Embedding-like tensors in GGUF naming: input embedding and (untied) output head.
GGUF_EMBEDDING_NAMES = frozenset({"token_embd.weight", "output.weight"})

_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING, _GGUF_ARRAY, _GGUF_UINT32, _GGUF_UINT64 = 8, 9, 4, 10


@dataclass
class WeightBytes:
    total: int = 0
    embedding: int = 0
    per_tensor: dict[str, int] = field(default_factory=dict)

    @property
    def non_embedding(self) -> int:
        return self.total - self.embedding


class _Reader:
    def __init__(self, fh: object) -> None:
        self.fh = fh

    def read(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        (value,) = struct.unpack(fmt, self.fh.read(size))  # type: ignore[attr-defined]
        return int(value)

    def string(self) -> str:
        length = self.read("<Q")
        return self.fh.read(length).decode("utf-8", errors="replace")  # type: ignore[attr-defined, no-any-return]

    def skip_value(self, vtype: int) -> object:
        if vtype == _GGUF_STRING:
            return self.string()
        if vtype == _GGUF_ARRAY:
            itype = self.read("<I")
            count = self.read("<Q")
            if itype in _GGUF_SCALAR_SIZES:
                self.fh.seek(_GGUF_SCALAR_SIZES[itype] * count, 1)  # type: ignore[attr-defined]
            else:
                for _ in range(count):
                    self.skip_value(itype)
            return None
        size = _GGUF_SCALAR_SIZES[vtype]
        raw = self.fh.read(size)  # type: ignore[attr-defined]
        if vtype == _GGUF_UINT32:
            return struct.unpack("<I", raw)[0]
        if vtype == _GGUF_UINT64:
            return struct.unpack("<Q", raw)[0]
        return None


def gguf_weight_bytes(path: Path) -> WeightBytes:
    """Per-tensor byte sizes from a GGUF file's header (v2/v3), without loading weights.

    Tensor sizes are taken as offset differences in the data section, so each includes up
    to ``general.alignment - 1`` bytes of padding (< 32 B per tensor; negligible).
    """
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        r = _Reader(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = r.read("<I")
        if version < 2:
            raise ValueError(f"GGUF v{version} is not supported")
        n_tensors, n_kv = r.read("<Q"), r.read("<Q")
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.read("<I")
            value = r.skip_value(vtype)
            if key == "general.alignment" and isinstance(value, int):
                alignment = value
        infos: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.read("<I")
            fh.seek(8 * n_dims, 1)
            r.read("<I")  # ggml type
            infos.append((name, r.read("<Q")))
        data_start = -(-fh.tell() // alignment) * alignment

    infos.sort(key=lambda item: item[1])
    result = WeightBytes()
    for i, (name, offset) in enumerate(infos):
        end = infos[i + 1][1] if i + 1 < len(infos) else file_size - data_start
        size = end - offset
        result.per_tensor[name] = size
        result.total += size
        if name in GGUF_EMBEDDING_NAMES:
            result.embedding += size
    return result


def safetensors_weight_bytes(directory: Path) -> WeightBytes:
    """Per-tensor byte sizes for a safetensors checkpoint (AWQ/GPTQ/FP8/EXL3 dirs)."""
    result = WeightBytes()
    for file in sorted(directory.glob("*.safetensors")):
        for name, info in read_safetensors_header(file).items():
            offsets = info.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                continue
            size = int(offsets[1]) - int(offsets[0])
            result.per_tensor[name] = size
            result.total += size
            if any(p.search(name) for p in EMBEDDING_PATTERNS):
                result.embedding += size
    return result


def artifact_weight_bytes(path: Path) -> WeightBytes | None:
    """Dispatch on artifact type. ``None`` when the layout is not understood."""
    try:
        if path.is_file() and path.suffix == ".gguf":
            return gguf_weight_bytes(path)
        if path.is_dir():
            return safetensors_weight_bytes(path)
    except (OSError, ValueError, struct.error):
        return None
    return None
