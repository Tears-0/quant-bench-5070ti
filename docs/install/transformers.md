# Install recipe: HuggingFace Transformers (reference, NF4, scoring, INT4 fallback)

This environment does four jobs:

1. the **BF16/FP16 reference** — live on Tier A, CPU-offloaded once to build the cached
   reference logits for Tiers B and C;
2. the **NF4 / bitsandbytes convenience baseline**;
3. the **designated scoring engine** for formats whose serving engine cannot return
   log-probs on the reference's support (AWQ, GPTQ, FP8 via vLLM);
4. the **INT4 fallback** if vLLM cannot be built for `sm_120` (§12.1).

## Environment

```bash
uv venv .venv-hf --python 3.12
uv pip install -r requirements/hf.txt --python .venv-hf
.venv-hf/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

`sm_120` must appear in the arch list, and the CUDA runtime must be 12.8 or newer. If it is
not, install the cu128 wheel explicitly:

```bash
uv pip install --python .venv-hf --extra-index-url https://download.pytorch.org/whl/cu128 torch
```

Do **not** pass `--index-strategy unsafe-best-match` to uv here: it will prefer a newer
CPU-only torch from PyPI over the cu128 wheel (this happened during M1; see
`docs/failures.md`). The working M1 environment is `torch 2.11.0+cu128`, whose arch list ends
in `sm_120`.

`quantbench preflight` asserts this and aborts with the same remediation hint. Without it,
the first kernel launch fails with `no kernel image is available for execution on the
device`, which is a confusing error a long way from its cause.

## Building the FP16 reference

```bash
uv run quantbench reference build --model qwen3-8b --eval wikitext2
```

This:

1. tokenizes the pinned evaluation corpus with the model's own tokenizer and hashes the
   token ids (`eval_stream_sha`) — different tokenization of "the same" WikiText-2 is a
   real source of PPL discrepancy;
2. runs FP16 once over that stream, with `device_map="auto"` CPU offload where the model
   does not fit (an 8B in BF16 does not fit in 16 GB);
3. persists the top-k normalised log-probs keyed by model, revision, tokenizer, stream,
   top-k and chunking.

It is slow, and it happens once. Everything downstream reads the cache.

## Attention backend

`engine.attention_backend` maps to `attn_implementation`: `sdpa`, `eager` or
`flash_attention_2`. It is recorded in every run record, and the resolved implementation is
read back from the loaded model rather than assumed.

## NF4

`configs/formats/nf4.yaml` loads through `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)`.
It is in the matrix as the convenience baseline people actually reach for, not because it is
expected to be fast. QLoRA training is a non-goal; NF4 appears only as an inference format.

## Timing caveat

Decoding is a hand-written greedy loop with `torch.cuda.synchronize()` after each token,
because per-token arrival timestamps are mandatory. That removes CPU/GPU overlap between
steps and costs some throughput — uniformly, for every Transformers run. It is stated in
[methodology §1](../methodology.md#where-the-clock-is) rather than subtracted.

Batched runs decode in lockstep, so all sequences in a batch share arrival times; the record
notes this with `lockstep_batch`.
