# Install recipe: llama.cpp (`sm_120`)

llama.cpp is used as **pinned pre-built binaries invoked as subprocesses**, not as a Python
binding. The CLI tools are the maintained, well-defined surface, and keeping them out of
the Python process means a llama.cpp rebuild cannot disturb the harness environment.

Note the renamed binaries. Older tutorials say `main`, `server`, `quantize`; the current
names are `llama-cli`, `llama-server`, `llama-quantize`.

## What M1 actually used (native Windows, 2026-10-17)

There is no compiler toolchain on the measurement host, so the official prebuilt release
was used. The CUDA **13.4** build is the one with native `sm_120` kernels — its startup log
reports `CUDA : ARCHS = 750,800,860,890,900,1200,1210`. (The `cuda-12.4` build cannot
contain `sm_120` code; CUDA 12.8+ is required to emit it.)

| File | Source | SHA-256 |
|---|---|---|
| `llama-b11706-bin-win-cuda-13.4-x64.zip` (148 MB) | github.com/ggml-org/llama.cpp releases, tag `b11706` | `dc1c685a52f1309cfae478a6fe50c0140cee840c8b21b98fe450da16173627ed` |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` (404 MB) | same release | `738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668` |
| `b11706.zip` source (38 MB, for `convert_hf_to_gguf.py` and `gguf-py`) | same repo, tag archive | `eb8ac76a4afb2b58bacbcb9e1c95997595a2b036a240b718600791f2277ee8ae` |

`llama-server --version` → `version: 0.7.0-dev (build 11706, commit f74e2fa6a)`.

```powershell
# both zips extract into one directory; the cudart zip provides cudart64_13.dll etc.
Expand-Archive llama-b11706-bin-win-cuda-13.4-x64.zip tools\llama.cpp\b11706\bin
Expand-Archive cudart-llama-bin-win-cuda-13.4-x64.zip tools\llama.cpp\b11706\bin
Expand-Archive b11706.zip tools\llama.cpp\b11706\src
$env:LLAMA_CPP_BIN_DIR = "$PWD\tools\llama.cpp\b11706\bin"
$env:LLAMA_CPP_REPO    = "$PWD\tools\llama.cpp\b11706\src\llama.cpp-b11706"
# the converter runs in the engine env (torch + sentencepiece) with gguf-py from the same tag
uv pip install -e "$env:LLAMA_CPP_REPO\gguf-py" --python .venv-hf\Scripts\python.exe
```

`tools/` is git-ignored; the table above is the record of what was in it.

## Build from source (Linux, or Windows with a toolchain)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout <TAG>            # pin a tag, e.g. b4321 — record it in docs and in run records
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

`CMAKE_CUDA_ARCHITECTURES=120` is the whole point: a build without it produces
`no kernel image is available for execution on the device` at the first kernel launch on a
5070 Ti, far away from its cause. CUDA 12.8+ is required to compile for `sm_120` at all.

## Tell the harness where the binaries are

```bash
export LLAMA_CPP_BIN_DIR=/path/to/llama.cpp/build/bin
export LLAMA_CPP_REPO=/path/to/llama.cpp        # needed for convert_hf_to_gguf.py
export LLAMA_CPP_VERSION=b4321                  # recorded when --version cannot be parsed
```

Per-binary overrides also exist: `LLAMA_SERVER_BIN`, `LLAMA_PERPLEXITY_BIN`,
`LLAMA_QUANTIZE_BIN`, `LLAMA_IMATRIX_BIN`.

Verify:

```bash
uv run quantbench preflight
```

The four `binary.llama-*` checks must be `ok` before the GGUF matrix can run.

## Which binaries the harness calls

| Binary | Used for |
|---|---|
| `llama-server` | Generation. Driven over streaming HTTP so every token gets its own arrival timestamp; batch size N is N concurrent requests against `--parallel N`. |
| `llama-perplexity` | Quality. `--kl-divergence-base` writes the reference file, `--kl-divergence` scores against it; output is normalised into this repo's schema and tagged as a native estimator. |
| `llama-quantize` | K-quants and i-quants from one F16 conversion, with and without an imatrix. |
| `llama-imatrix` | The importance matrix, from calibration text that must be disjoint from the evaluation stream. |

The server is started with `--fit off -ngl all --no-cache-prompt -cram 0 -lv 4` and a fixed
token count, so that a config either fits as asked or OOMs, and repeated measured iterations
really do repeat the work. `--fit` defaults to **on** in current builds and would otherwise
silently shrink context or offload layers; see `docs/failures.md`.

## Flags that are recorded, not hidden

`-ngl`, `-ctk`/`-ctv` (KV cache dtype), `-fa on|off` (flash attention) and any extra flags
appear in the `engine` block of every run record. A run with `-fa on` and a run with
`-fa off` are different experiments and are planned as such.

Quantized KV cache (`q8_0`, `q4_0`) generally requires flash attention; the harness warns
when that combination is requested without it rather than silently changing the request.

## Known sharp edges

* Building against a CUDA toolkit older than 12.8 silently yields a binary with no `sm_120`
  kernels — check `CMAKE_CUDA_ARCHITECTURES` actually took effect in the build log.
* `LLAMA_CURL=OFF` avoids pulling model downloads into llama.cpp itself; artifacts are
  fetched and hashed by the harness so their provenance is recorded.
* Record the tag in `docs/failures.md` alongside any build failure. "It didn't build" is a
  finding only if the reader can tell which commit it didn't build from.
