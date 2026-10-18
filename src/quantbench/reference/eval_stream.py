"""The fixed, hashed evaluation token stream (requirements.md §6.6).

All quality evaluation runs over a byte-identical, pre-tokenized stream whose SHA-256 is
recorded in every record. Different tokenization of "the same" WikiText-2 is a real and
frequently-missed source of perplexity discrepancy between tools, and it is the first thing
a sceptical reader will suspect.

The stream is built once per (model tokenizer, eval config) and cached. Both the token ids
and the exact source text are persisted: llama.cpp's tools consume text, everything else
consumes ids, and both must be provably the same material.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantbench.config import EvalConfig, ModelConfig
from quantbench.paths import reference_cache_dir


@dataclass
class EvalStream:
    ids: np.ndarray
    text_path: Path
    ids_path: Path
    meta: dict[str, Any]

    @property
    def sha(self) -> str:
        return str(self.meta["eval_stream_sha"])

    @property
    def n_tokens(self) -> int:
        return int(self.ids.size)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stream_dir(model: ModelConfig, eval_cfg: EvalConfig) -> Path:
    return reference_cache_dir() / model.slug / f"stream-{eval_cfg.name}"


def _read_source_text(eval_cfg: EvalConfig) -> str:
    """Load the evaluation corpus.

    ``local_file`` is the default and the only path with no extra dependency: point the
    eval config at a checked-out copy of the corpus. ``hf_dataset`` additionally requires
    the ``datasets`` package (optional extra, see docs/open-questions.md).
    """
    if eval_cfg.source_kind == "local_file":
        path = Path(eval_cfg.source)
        if not path.exists():
            raise FileNotFoundError(
                f"evaluation corpus not found: {path}\n"
                "download it once and point configs/eval/*.yaml at the file; the exact "
                "bytes are hashed into every quality record"
            )
        return path.read_text(encoding="utf-8")

    from datasets import load_dataset

    dataset = load_dataset(eval_cfg.source, eval_cfg.subset, split=eval_cfg.split or "test")
    return "\n\n".join(dataset["text"])


def build_eval_stream(model: ModelConfig, eval_cfg: EvalConfig, force: bool = False) -> EvalStream:
    """Tokenize the corpus with the model's tokenizer and freeze the result."""
    out_dir = stream_dir(model, eval_cfg)
    ids_path = out_dir / "tokens.npy"
    text_path = out_dir / "corpus.txt"
    meta_path = out_dir / "meta.json"
    if not force and ids_path.exists() and meta_path.exists():
        return load_eval_stream(out_dir)

    from transformers import AutoTokenizer

    text = _read_source_text(eval_cfg)
    from quantbench.paths import models_dir

    local = models_dir() / model.slug / "hf"
    tokenizer = (
        AutoTokenizer.from_pretrained(local)
        if (local / "tokenizer.json").exists()
        else AutoTokenizer.from_pretrained(model.hf_repo, revision=model.revision)
    )
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    if eval_cfg.ppl.max_tokens:
        ids = ids[: eval_cfg.ppl.max_tokens]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, ids)
    text_path.write_text(text, encoding="utf-8")
    meta = {
        "model": model.name,
        "hf_repo": model.hf_repo,
        "revision": model.revision,
        "eval_config": eval_cfg.name,
        "source": eval_cfg.source,
        "source_sha": sha256_bytes(text.encode("utf-8")),
        "tokenizer_sha": _tokenizer_sha(tokenizer),
        # Qwen3 has no BOS token: an add_bos request is then a no-op, and records say so.
        "bos_token_id": tokenizer.bos_token_id,
        "n_tokens": int(ids.size),
        "eval_stream_sha": sha256_bytes(ids.tobytes()),
        "ppl": eval_cfg.ppl.model_dump(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return EvalStream(ids=ids, text_path=text_path, ids_path=ids_path, meta=meta)


def load_eval_stream(out_dir: Path) -> EvalStream:
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    ids = np.load(out_dir / "tokens.npy")
    return EvalStream(
        ids=ids, text_path=out_dir / "corpus.txt", ids_path=out_dir / "tokens.npy", meta=meta
    )


def _tokenizer_sha(tokenizer: Any) -> str | None:
    """Hash the tokenizer's serialized form so a silent tokenizer change is detectable."""
    try:
        blob = tokenizer.backend_tokenizer.to_str()  # fast tokenizers
    except Exception:
        try:
            blob = json.dumps(tokenizer.get_vocab(), sort_keys=True)
        except Exception:
            return None
    return sha256_bytes(blob.encode("utf-8"))
