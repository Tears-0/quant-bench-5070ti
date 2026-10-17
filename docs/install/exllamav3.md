# Install recipe: ExLlamaV3 (EXL3)

EXL3 is the strongest quality-per-bit contender in the matrix and the format most people
have not benchmarked properly. Its continuous bits-per-weight knob is what makes the
quality-vs-bpw curve (figure 2) a curve rather than four disconnected points.

## Environment

```bash
uv venv .venv-exl3 --python 3.12
uv pip install -r requirements/exl3.txt --python .venv-exl3
.venv-exl3/bin/python -c "import torch, exllamav3; print(torch.cuda.get_arch_list(), exllamav3.__version__)"
```

`sm_120` must be in the arch list. If the ExLlamaV3 wheel ships prebuilt kernels without
consumer Blackwell, build it from source against the same torch:

```bash
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
git checkout <COMMIT>
TORCH_CUDA_ARCH_LIST="12.0" uv pip install --no-build-isolation -e . --python ../.venv-exl3
export EXLLAMAV3_REPO=$PWD          # convert.py lives here
export EXLLAMAV3_VERSION=<version>
```

Record the version or commit in this file in the same PR; the run records read
`EXLLAMAV3_VERSION` when the package does not expose `__version__`.

## Producing EXL3 artifacts

```bash
uv run quantbench quantize --model qwen3-8b --formats exl3:3.0,exl3:4.0,exl3:5.0,exl3:6.0
```

which shells out to `convert.py -i <hf-checkout> -o <out-dir> -b <bpw>` and writes a
`.manifest.json` beside each artifact with the command line, calibration setting, duration
and SHA-256. All bpw levels are converted from the **same** source checkpoint with identical
settings so that bits-per-weight is the only variable across the curve.

The artifact directory name embeds the variant verbatim
(`models/<slug>/<slug>-exl3-4.0bpw`) to match `configs/formats/exl3.yaml`.

## How the harness drives ExLlamaV3

In-process, inside the worker subprocess — there is no server surface worth driving.
Per-token timestamps come from a hand-written greedy loop with an explicit
`torch.cuda.synchronize()` per step, matching the Transformers adapter so the two are
comparable to each other. See [methodology §1](../methodology.md#where-the-clock-is).

KV cache dtype maps onto ExLlamaV3's quantized cache layers:

| Config value | Cache |
|---|---|
| `fp16` / `auto` | default cache layer |
| `q8_0` | `CacheLayer_quant`, 8-bit K and V |
| `q4_0` | `CacheLayer_quant`, 4-bit K and V |

## Status of this adapter

**Unverified against hardware.** The adapter is written against the API surface of the
pinned version above but has not yet been exercised on a real EXL3 artifact — that happens
at M4. Until then treat `src/quantbench/engines/exllamav3.py` as provisional: if the API has
moved, fix the adapter and record what changed here and in `docs/failures.md`. This is
tracked in [`docs/open-questions.md`](../open-questions.md).
