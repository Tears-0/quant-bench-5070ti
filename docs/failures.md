# Failures, dead ends, and unsupported paths

Negative results ship. A configuration that would not build, would not load, or OOMed is a
finding about running LLMs on a 16 GB consumer Blackwell card, and it belongs here with the
exact versions and the exact error text.

This file is maintained by hand from the run records. Every entry should be traceable to a
record in `results/raw/` whose `status` is one of `oom`, `build_failed`,
`engine_unsupported`, `timeout`, `crashed` or `throttled_invalid`.

---

## Format of an entry

```markdown
### <what was attempted>

* **Status:** build_failed | engine_unsupported | oom | timeout | crashed
* **Date:** YYYY-MM-DD
* **Versions:** engine + commit, torch + arch list, CUDA, driver
* **Command / spec:** run_id or exact command line
* **Error:**
  ```
  verbatim, trimmed only at the ends
  ```
* **What we tried:** the remediation attempts, in order
* **Resolution:** fixed / worked around / dropped from the matrix (and which fallback §12 applied)
```

---

## Build and environment failures

### vLLM Python dependency set cannot be resolved on a host without CUDA

* **Status:** build_failed (tooling, not a benchmark run)
* **Date:** 2026-08-15
* **Context:** `uv sync --extra dev` on the development host (Windows 11, no `CUDA_HOME`).
* **Error:**
  ```
  RuntimeError: Cannot find CUDA_HOME. CUDA must be available in order to build the package.
  hint: `vllm` (v0.1.3) was included because `quantbench[vllm]` (v0.1.0) depends on `vllm`
  ```
* **Cause:** uv resolves *every* declared extra into one lockfile. Declaring vLLM as an
  extra therefore made the whole CLI uninstallable on any machine without a CUDA
  toolchain — including the machine that most needs `quantbench preflight` to explain why.
* **Resolution:** engine dependency sets were moved out of `[project.optional-dependencies]`
  into `requirements/{hf,vllm,exl3}.txt`, one isolated environment per engine, which is
  what §11.2 wants anyway (their torch pins conflict). Recorded as open question Q8.

---

## Environment and tooling findings (M1, 2026-10-17)

Everything below was hit while setting up and running the first real `smoke` profile on the
target card (RTX 5070 Ti, driver 617.14, Windows 11, llama.cpp b11706 `win-cuda-13.4`).
Each is either a property of the stack worth knowing or a harness defect that was fixed;
the harness fixes are covered by regression tests.

### torch resolved to a CPU-only build

* **Status:** environment error, caught before any run
* **Context:** `uv pip install -r requirements/hf.txt --index-strategy unsafe-best-match`
* **Symptom:** `AssertionError: Torch not compiled with CUDA enabled`
* **Cause:** with `unsafe-best-match`, uv picked a newer CPU wheel from PyPI (torch 2.14.1)
  over the newest cu128 wheel (2.11.0+cu128). This is exactly the §3.2 hazard, one step
  earlier than usual.
* **Resolution:** install torch from the CUDA index with uv's default index strategy.
  Verified: `2.11.0+cu128`, arch list `[sm_75 … sm_100, sm_120]`, a bf16 matmul ran on
  the card. `docs/install/transformers.md` now says not to use `unsafe-best-match`.

### `convert_hf_to_gguf.py` (b11706) needs `sentencepiece` even for BPE tokenizers

* **Status:** build_failed (producer), fixed
* **Error:**
  ```
  File ".../conversion/qwen.py", line 63, in set_vocab
      self._set_vocab_sentencepiece()
  ModuleNotFoundError: No module named 'sentencepiece'
  ```
* **Cause:** Qwen2/3 `set_vocab` tries the sentencepiece path first and only falls back to
  BPE on `FileNotFoundError`, so the import itself must succeed.
* **Resolution:** `sentencepiece` and `protobuf` added to `requirements/hf.txt` (they are in
  llama.cpp's own `requirements-convert_hf_to_gguf.txt`).

### Harness defect: producer output decoded as cp1252 on Windows

* **Status:** harness bug, fixed
* **Symptom:** `gguf:Q8_0 failed: can only concatenate str (not "NoneType") to str`
* **Cause:** `subprocess.run(text=True)` decodes with the locale codec on Windows; the
  converter's UTF-8 progress output killed the reader thread and left `stdout=None`. The
  F16 conversion itself had completed, but its provenance manifest was never written.
* **Resolution:** every subprocess call decodes as UTF-8 with `errors="replace"`. All
  artifacts were regenerated from scratch so each has a complete manifest; all three
  SHA-256s reproduced bit-for-bit, so conversion and quantization are deterministic.

### Harness defect: `quantize` exited 0 when every producer failed

* **Resolution:** the command now exits 1 and lists the failed formats.

### llama.cpp changed its version banner

* **Old:** `version: 4321 (abc1234)` — **new:** `version: 0.7.0-dev (build 11706, commit f74e2fa6a)`
* Records would have carried `version: null`. Both formats are now parsed; the build tag
  (`b11706`) is recorded because releases are tagged by build number.
* `llama-quantize` prints no version at all, so producer versions are read from
  `llama-server --version` in the same release directory.

### `llama-server` defaults that would have silently changed the experiment

| Flag | Default in b11706 | Why it matters | Pinned to |
|---|---|---|---|
| `--fit` | **on** — "adjust unset arguments to fit in device memory" | turns an OOM (a result, §5.3) into a silently shrunk context or partial offload | `--fit off` |
| `-ngl` | `auto` | partial CPU offload is a different experiment | `-ngl all`, and the run fails if the log reports fewer layers offloaded than exist |
| `--cache-prompt` / `-cram` | enabled / 8192 MiB host cache | measured iterations repeat identical prompts; a cache hit deletes prefill from TTFT | `--no-cache-prompt -cram 0` |
| log verbosity | KV buffer sizes not logged | `vram_kv_mb` was null | `-lv 4` (lowest level that logs `llama_kv_cache: CUDA0 KV buffer size`; emits no per-token debug lines) |

### No per-process VRAM accounting under Windows WDDM

* NVML returns no per-process `usedGpuMemory` on this driver model, so the sampler falls
  back to device-wide usage, which includes the desktop compositor and every other
  application. On this host **4.6–4.9 GiB was in use before any model loaded**, leaving
  ~11.3 GiB of the 15.9 GiB card for a run.
* Records now carry `vram_accounting: "device_wide"` and `vram_peak_attributed_mb` (peak
  minus pre-load baseline). `vram_weights_mb` for llama.cpp uses the engine-reported GPU
  model buffer instead of a load-time NVML delta.

### Harness defect: `vram_weights_mb` from a single early device-wide poll

* **Symptom:** Q8_0 read 3,500 MiB of weights against a 4,275 MiB attributed peak, while
  Q4_K_M agreed within 25 MiB.
* **Resolution:** engine-reported model buffer (above). After the fix the decomposition
  closes for both quants: weights + KV + compute + ~250 MiB CUDA context = attributed peak.
  The two affected records are kept in `results/superseded/2026-10-17-vram-weights-early-poll/`
  and are not read by the analysis layer.

### `sw_power_cap` is asserted under ordinary load; driver 617 sets an undocumented bit

* At ~205–220 W average against a 300 W limit, NVML reports `sw_power_cap` for part of the
  measured window (0–29% across runs). Treating it as invalidating would have discarded
  essentially every run; ignoring it would hide a real clock-regulation state. It is now
  recorded separately as `power_capped` / `power_capped_fraction` (see open question Q13).
* Bit `0x400` is set at idle and under load and is not named by nvidia-ml-py; it is
  recorded as `unknown_0x400` rather than dropped.

### Harness defect: `throttled_invalid` was overwritten by `ok`

* `Worker.run()` set `status=ok` after the workload had already downgraded it. Fixed, with a
  regression test.

### Harness defect: preflight could not find `.exe` binaries via `LLAMA_CPP_BIN_DIR`

* Fixed: the lookup appends `.exe` on Windows.

---

## Runtime failures

_The M1 `smoke` latency runs (2 configurations) completed with `status: ok`. No runtime
failures yet. This section must not be padded with anticipated failures — a predicted error
is not a finding._

Expected candidates, to be confirmed or refuted empirically (§3.2, §12):

| Candidate | What would confirm it |
|---|---|
| vLLM prebuilt wheels lack `sm_120` kernels | Server exits at startup with `no kernel image is available for execution on the device`; recorded as `build_failed` |
| AWQ needs the Triton attention backend on `sm_120` | `flash_attn` backend fails while `triton` succeeds for the same spec — the attention-backend sweep is designed to catch exactly this |
| FP8 W8A8 unsupported via `cutlass_scaled_mm` on GB203 | vLLM refuses the quantization at load with a kernel/capability error |
| Tier C (14B) OOMs at 5-bit and above | `status: "oom"` recorded with the context and batch at which it happened — the fit/no-fit table is the deliverable |
| FlashAttention build unavailable for `sm_120` | Engine preflight reports the backend as unavailable; the run records whichever backend was actually used |

## OOM ledger

Populated from records with `status: "oom"`. The fit/no-fit table is exactly what a 16 GB
owner wants, so it is published even though every row is a failure.

| Model | Format | Engine | Batch | Context | KV dtype | Peak VRAM before OOM |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
