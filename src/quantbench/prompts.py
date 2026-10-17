"""Deterministic prompt construction for latency workloads.

Latency runs need prompts of an *exact* token length, identical across engines and across
repetitions, or prompt-length effects contaminate the format comparison. Two sources, in
order of preference:

1. **The pinned evaluation stream** (``reference/eval_stream``) for the model's tokenizer.
   Real text, real token distribution, byte-identical across engines (§6.6).
2. **A seeded synthetic id sequence** when no tokenized stream exists yet. Ids are drawn
   from a conservative mid-vocabulary band that is valid for every model in the ladder,
   avoiding special tokens.

Which source was used is recorded in the run record, because a synthetic prompt is not a
realistic prefill workload and must not be compared against a text prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantbench.schema import RunSpec

#: Conservative id band: above the byte-fallback/special-token region of common
#: tokenizers, below the smallest vocabulary in the model ladder.
SAFE_ID_LO = 1024
SAFE_ID_HI = 30000


@dataclass
class PromptBatch:
    ids: list[list[int]]
    source: str  # "eval_stream" | "synthetic"
    prompt_tokens: int

    def as_metadata(self) -> dict[str, object]:
        return {"prompt_source": self.source, "prompt_tokens": self.prompt_tokens}


def build_prompt_batch(spec: RunSpec, stream_ids: np.ndarray | None = None) -> PromptBatch:
    """Build ``batch_size`` prompts of exactly ``prompt_tokens`` tokens each.

    Each element of the batch gets a *different* slice so that engines with prefix caching
    cannot serve request N from request 1's cache.
    """
    n = spec.workload.prompt_tokens
    b = max(1, spec.workload.batch_size)

    if stream_ids is not None and len(stream_ids) >= n * b:
        ids = [[int(x) for x in stream_ids[i * n : (i + 1) * n]] for i in range(b)]
        return PromptBatch(ids=ids, source="eval_stream", prompt_tokens=n)

    rng = np.random.default_rng(spec.workload.seed)
    pool = rng.integers(SAFE_ID_LO, SAFE_ID_HI, size=(b, n), dtype=np.int64)
    return PromptBatch(
        ids=[[int(x) for x in row] for row in pool], source="synthetic", prompt_tokens=n
    )
