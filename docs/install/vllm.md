# Install recipe: vLLM on consumer Blackwell (`sm_120`)

This is the highest-risk dependency in the repo. Requirements §12 anticipates that it may
not work at all, and the fallback is documented rather than hidden: if vLLM cannot be made
to run on `sm_120` within reasonable effort, the failure is written up in
[`docs/failures.md`](../failures.md) with versions and errors, INT4 coverage moves to
Transformers + `GPTQModel` kernels, and ExLlamaV3 carries the second high-performance
engine.

## Environment

vLLM gets its own virtual environment. Its torch pin conflicts with the ExLlamaV3 and HF
sets, which is why the engine dependency sets are separate files rather than extras of one
lockfile (see open question Q8).

```bash
uv venv .venv-vllm --python 3.12
uv pip install -r requirements/vllm.txt --python .venv-vllm
```

## Verify the arch list before anything else

```bash
.venv-vllm/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

`sm_120` must appear. If it does not, the wheel was not built for consumer Blackwell and
every kernel launch will fail with
`CUDA error: no kernel image is available for execution on the device`.

## If the prebuilt wheel lacks `sm_120`

Prebuilt vLLM wheels have historically shipped without consumer-Blackwell arch flags. Build
from source against a cu128+ torch:

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout <COMMIT>
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 \
  uv pip install --no-build-isolation -e . --python ../.venv-vllm
```

Then record exactly what you used, because the run records carry it:

```bash
export VLLM_VERSION=<version-or-nightly-tag>
export VLLM_COMMIT=<git-sha>
```

Whichever artifact you ended up with — wheel URL or source commit — write it into this file
in the same PR. §11.2 requires the install to be a versioned artifact, not folklore.

## Attention backend is a variable, not a detail

FlashAttention builds for `sm_120` may be unavailable or broken depending on version;
community reports indicate AWQ works through `awq_marlin` on `sm_120`, sometimes only with
the Triton attention backend. The harness therefore treats the backend as a swept,
recorded variable via `VLLM_ATTENTION_BACKEND`:

| Config value | Env value |
|---|---|
| `flash_attn` | `FLASH_ATTN` |
| `flashinfer` | `FLASHINFER` |
| `triton` / `triton_attn` | `TRITON_ATTN` |
| `xformers` | `XFORMERS` |
| `auto` | unset (vLLM chooses; whatever it chose is still recorded) |

`configs/profiles/core.yaml` sweeps three backends on one Tier B AWQ configuration. If one
of them fails to start, that is a data point, not a broken run: the record carries
`engine_unsupported` with the server's stderr.

## FP8 is conditional

GB203 has FP8 tensor cores, but vLLM's `cutlass_scaled_mm` path for block-scaled FP8 has had
`sm_120` gaps. `configs/formats/fp8.yaml` is marked `conditional: true`, the planner emits a
note when it is included, and nothing in the README promises FP8 works until a successful
run record exists. If the server refuses to start, copy the kernel error into
`docs/failures.md` and drop FP8 from the core matrix (§12).

## How the harness drives vLLM

As an OpenAI-compatible **server subprocess**, not in-process:

* streaming responses give honest per-token arrival timestamps;
* a segfault costs one data point rather than the session;
* the CLI never imports vLLM, so `quantbench preflight` still works on a machine where the
  import is exactly what is broken.

Prefix caching is disabled explicitly (`--no-enable-prefix-caching`) — otherwise the second
measured iteration is a cache hit and prefill silently disappears from the measurement.

## Scoring

vLLM's `prompt_logprobs` exposes its own top-k, not log-probs on an arbitrary reference
support, so quality for vLLM-served formats is routed to a designated scoring engine
(Transformers on the same checkpoint) and the record carries `scored_by_engine`. Native
scoring can be enabled per spec with `engine.extra.allow_native_scoring`, in which case the
resulting KL is flagged `tail_mass_estimated`.
