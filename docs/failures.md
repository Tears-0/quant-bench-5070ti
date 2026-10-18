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

### No per-process VRAM accounting under Windows WDDM *(superseded the same day — see "Windows sysmem fallback" below: the WDDM performance counters provide it)*

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

## Quality-pipeline findings (M2, 2026-10-17)

### llama-perplexity's `PPL(base)` is biased low by its own 16-nat log-prob floor

* **What:** in `--kl-divergence` mode llama-perplexity reports `Mean PPL(base)` = **10.6000**
  for the F16 base, while the F16 run that wrote that base reported **10.6672** over the
  identical 40,920 positions.
* **Cause, verified:** the base file stores each log-prob as uint16 between the top logit and
  a floor 16 nats below it (`tools/perplexity/perplexity.cpp`, `log_softmax(..., uint16_t*)`).
  Reading the 12.4 GB base directly, **114 of 40,920 target tokens (0.28%) sit at the floor**,
  and recomputing PPL from the stored values reproduces 10.6000 exactly. The floor credits
  those rare targets with more probability than F16 gave them.
* **Effect:** llama.cpp's own `Mean PPL(Q) − PPL(base)` overstates every quant's PPL cost by
  ~0.067 here (Q8_0: +0.118 reported vs **+0.051** true). KL is unaffected: floored tokens
  carry < e⁻¹⁶ of the base mass.
* **Resolution:** `ppl_reference` is the F16 run's full-logit PPL from the base metadata; the
  floored value is kept as the artifact `ppl_base_native_floored`. The two records written
  before the fix are in `results/superseded/2026-10-17-floored-ppl-reference/` (their KL,
  top-1 and PPL(Q) reproduced bit-for-bit on re-run).

### Windows sysmem fallback: on this host, CUDA never runs out of memory

* **Discovered:** an lm-eval check with `--batch_size auto` sat at "100% utilization" while
  drawing 79 W (compute-bound load on this card draws 210–260 W). Windows' GPU counters
  showed the process holding 14.6 GB of dedicated VRAM **plus 17.4 GB in system RAM**.
* **Confirmed with a probe** (allocating 1 GiB tensors until past the card): 18 GiB was
  allocated on a 16 GiB card with **no OOM**. Dedicated memory plateaued at ~15.4 GB and
  every further allocation was served from system RAM over PCIe. This is NVIDIA's
  "CUDA - Sysmem Fallback Policy" (driver 536.40+), whose default on Windows is to allow it.
* **Why it matters:** §5.3 treats OOM as a first-class *result*, and Tier C's fit/no-fit
  table is the deliverable. With fallback on, a config that does not fit would instead run
  10–50× slower and be recorded as a valid — wildly slow — measurement. `--batch_size auto`
  walks straight into it, because it searches for the largest batch that does not OOM.
* **Resolution:**
  1. The sampler reads Windows' `\GPU Process Memory(pid_*)\{Dedicated,Shared} Usage`
     counters through `pdh.dll` (0.07 ms per sample). Validated: a 2,048 MiB tensor shows as
     +2,104 MiB dedicated under the right PID. This also gives **true per-process VRAM on
     Windows** (`vram_accounting: per_process_wddm`), closing the WDDM gap NVML leaves; it
     agreed with the earlier device-minus-baseline estimate within ~20 MiB.
  2. Spill = shared memory above 256 MiB **while** the device is within 1 GiB of full. Such
     a run gets `status: oom` with the spilled amount in the error. Validated live: the rule
     fires at the first spilled gigabyte and not before.
  3. A first version flagged shared memory alone and wrongly marked two llama-perplexity
     runs as OOM: WDDM also counts deliberately *pinned host* buffers (its ~1.2 GB logits
     buffer) as shared. Those records are in `results/superseded/2026-10-17-false-spill-oom/`.
  4. lm-eval uses a fixed, recorded batch size (8), never `auto`.
  5. Child processes that hold the model (llama-perplexity, lm-eval) are now tracked, so
     their VRAM is attributed to the run.
* **Recommended host setting (owner action):** NVIDIA Control Panel → Manage 3D settings →
  "CUDA - Sysmem Fallback Policy" → **Prefer No Sysmem Fallback**. With it, a config that
  does not fit fails immediately as a clean OOM instead of crawling until detected.

### Windows PyTorch has no fused attention kernel; generation is CPU-bound

* **Measured (torch 2.11.0+cu128, Windows):** `torch.backends.cuda.is_flash_attention_available()`
  is `False`, and forcing SDPA's memory-efficient kernel fails with "No available kernel". SDPA
  therefore runs on the math path, which materializes the full attention matrix: one
  2,338-token GSM8K prefill costs **2,079 MiB** over the weights (Qwen3-4B, fp16).
* **Decode is launch/Python-bound, not GPU-bound:** ~29 decode steps/s regardless of batch
  (27.9 tok/s at batch 1, 238 tok/s at batch 8). Batch is the only throughput lever for
  generation, and the math-path prefill spike is what caps it.
* **Consequence:** GSM8K through Transformers runs at batch 2 (the measured maximum inside the
  VRAM budget; batch 4 spills), roughly 2 h per configuration. llama.cpp has flash attention
  and native kernels and would be several times faster — see open question Q16.

### lm-eval batch sizes are measured, not guessed

* `--batch_size auto` probes for the largest batch that does not OOM; with sysmem fallback
  on, nothing OOMs, so it spills. A single fixed size either starves short tasks or
  overflows long ones (MMLU professional law's 5-shot prompts reach 2,158 tokens; ARC's
  0-shot ones 177).
* `quantbench lmeval-tune` builds every real request with lm-eval's own task objects,
  measures peak VRAM and throughput per batch at each task's longest request, and pins the
  fastest batch inside the budget (`configs/eval/lm_eval_batch/qwen3-4b.yaml`, every probe
  recorded). Every over-budget probe was caught, and the spill detector flagged the ones
  that spilled. Log-likelihood throughput plateaus around 7k tok/s by batch 4–8: the GPU is
  already compute-saturated, so larger batches would only add VRAM risk.

### `llama-server` (b11706) cannot serve prompt log-probs, so lm-eval's GGUF backend cannot run log-likelihood tasks

* **Probe:** `POST /v1/completions` with `"echo": true, "logprobs": 10, "max_tokens": 1`
  returned HTTP 200 with only the *generated* token's log-probs — `echo` was ignored. The
  server source has an explicit `throw std::runtime_error("Only no echo is supported")` on
  another code path.
* **Effect:** lm-eval's `gguf` backend needs prompt log-probs (`text_offset`, `tokens`,
  `token_logprobs`) for every log-likelihood task — ARC, HellaSwag, MMLU. Only generative
  GSM8K could run against llama-server.
* **Status:** open question Q15 (how to get downstream accuracy for GGUF).

### Top-128 reference support leaves 1.35% of the probability mass unaccounted for

* Measured on the Qwen3-4B BF16 reference: tail mass mean 1.35%, p99 21%, max 93%. See open
  question Q14. GGUF KL is unaffected (full-vocabulary).

### Tokenizers agree: llama.cpp and Transformers produce the identical stream

* Not a failure — recorded because it was *checked*: the 81,920 token ids stored in the
  llama.cpp KL base match the hashed Transformers stream exactly (`token_stream_verified`).

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
