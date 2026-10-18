# Methodology

Every metric this repository publishes is defined here, in the same wording used in the
code docstrings that compute it. Where a definition is ambiguous in the wild, the choice
this repo makes is stated explicitly, because ambiguous metric definitions are the single
most common reason benchmark repos get dismissed.

Snapshot date: M1 `smoke` latency runs on 2026-10-17 (driver 617.14, llama.cpp b11706,
native Windows 11), and M2 quality runs the same day: Tier A Q8_0 and Q4_K_M against both references.

---

## 1. Latency and throughput

Computed in [`src/quantbench/metrics/latency.py`](../src/quantbench/metrics/latency.py)
from **per-token arrival timestamps**, which every engine adapter is required to produce.
Deriving inter-token-latency percentiles from an aggregate is impossible, and aggregate-only
measurement is the most common flaw in amateur benchmarks.

| Metric | Definition |
|---|---|
| `ttft_ms` | Wall time from request submission to first generated token emitted. |
| `itl_ms` | Inter-token latency: per-token deltas after the first. p50, p90, p95, p99 and std are all reported. The mean alone is never reported. |
| `decode_tok_s` | `(generated_tokens - 1) / (t_last_token - t_first_token)`, per request. |
| `prefill_tok_s` | `prompt_tokens / ttft_ms × 1000`. **This conflates scheduling overhead into prefill**; it is a useful comparative number, not a kernel measurement. |
| `e2e_latency_ms` | Submission to final token. |
| `throughput_tok_s_total` | Aggregate generated tokens / wall time across the whole batch — the serving-side number, distinct from per-request decode rate. |

Aggregation: per-request metrics pool every sequence of every measured repetition;
`throughput_tok_s_total` has one value per repetition because it is a property of the
batch, not the request. **Median and IQR are reported, not mean ± std**, unless a
distribution has been verified near-normal.

### Where the clock is

Timestamps come from `time.perf_counter()` inside the worker process.

* **llama.cpp and vLLM** are driven as HTTP servers over localhost, and a token's arrival
  is timestamped the instant its SSE frame has been read. TTFT and ITL therefore include
  loopback and SSE-framing overhead. That overhead is identical for both server-based
  engines and is **not subtracted**.
* **Transformers and ExLlamaV3** run in-process with a hand-written greedy loop and an
  explicit `torch.cuda.synchronize()` per step. This makes each timestamp mean "the token
  exists", at the cost of removing CPU/GPU overlap between steps. It applies uniformly to
  both in-process engines.

These two timing regimes are not interchangeable at the microsecond level. Cross-engine
latency comparisons in this repo are made at the scale of milliseconds per token, where
the difference is immaterial; anywhere it might matter, the engine is on the plot.

---

## 2. Quality

Computed in [`src/quantbench/metrics/quality.py`](../src/quantbench/metrics/quality.py).

**KL divergence against the FP16 reference is the primary metric.** Perplexity is the
familiar number, but it is a weak instrument: a quant can post *lower* PPL than its own
FP16 parent while being demonstrably less faithful, and calibration-set overlap makes PPL
easy to game by accident. KL over the pre-sampling output distribution directly measures
"how differently does this model think".

| Metric | Definition |
|---|---|
| `ppl_wikitext2` | Perplexity on WikiText-2 (raw) at fixed context, stride and token count. The exact chunking parameters are recorded in every record (`ppl_config`), because PPL is not comparable across tools without them. |
| `kl_div.{mean,p90,p99,max}` | Per-position `KL(P_ref ‖ Q_cand)` over the reference's top-k support, summarised across positions. |
| `top1_agreement` | Fraction of positions where the candidate's argmax equals the reference's argmax. |
| `top5_agreement` | Fraction of positions where the reference's argmax is inside the candidate's top-5. |
| `lm_eval.*` | `arc_challenge` (0-shot), `hellaswag` (0-shot), `gsm8k` (8-shot exact match), and a **fixed named MMLU subset** (see `configs/eval/lm_eval_tasks.yaml`). Reported absolute and as retention % of FP16. |

Default PPL chunking: `ctx=2048`, `stride=2048` (non-overlapping — no sliding-window
credit), `add_bos=true`, capped at 335,000 tokens.

### The truncation and its tail term

Storing full-vocabulary distributions for a 300k-token stream is impractical, so the
reference artifact keeps the top-k (default 128) **normalised** log-probs per position. KL
is computed over that support:

```
KL_trunc = Σ_{v ∈ topk(ref)} p_v · (log p_v − log q_v)
```

Because the stored reference log-probs are normalised, the omitted tail mass
`m = 1 − Σ p_v` is *measured*, not assumed — but its distribution is unknown, so its
contribution to KL cannot be computed exactly. Two consequences, both recorded per record:

* `kl_div.topk` states the support size.
* `kl_div.tail_mass_estimated` is `true` when the reference tail mass exceeds 1%, **or**
  when the candidate could not be scored on the reference's id set (in which case the
  candidate's own top-k is used as a lookup table with its minimum observed log-prob as a
  floor — an upper bound on `log q`, hence a *lower* bound on KL).

The exact path is preferred everywhere it is available: the scoring engines accept the
reference's top-k ids and return their log-probs at exactly those positions.

### Downstream accuracy (lm-eval)

* **Engine:** lm-eval's `hf` backend for the reference and every quant (open question Q15 (a);
  llama-server serves no prompt log-probs). GGUF quants are dequantized **once** to fp16
  safetensors (`quantbench models dequantize`), rebuilt under the original HF config with a
  strict tensor-name match, and saved with the original HF tokenizer.
* **Validation gate:** before any lm-eval run, each dequantized checkpoint is scored on the
  BF16 top-k reference and must reproduce llama.cpp's own quality for that quant. Measured:
  Q8_0 KL 0.00115 / top-1 98.24% (llama.cpp: 0.00165 / 98.04%); Q4_K_M 0.0582 / 90.66%
  (llama.cpp: 0.0509 / 90.96%).
* **One compute dtype, fp16, for all configurations**, so weights are the only variable.
  fp16 was checked against the BF16 reference on 2,047 positions: NLL 2.31615 vs 2.3166,
  top-1 agreement 98.9%.
* **Batch size per task, measured** (`quantbench lmeval-tune`, see docs/failures.md) and
  pinned in `configs/eval/lm_eval_batch/<model>.yaml`. Fixed per task, so scores do not depend
  on how much VRAM happened to be free.
* **Jobs:** one `quantbench.lmeval_job` subprocess per configuration loads the model once,
  then runs the tasks strictly in sequence, each at its measured batch size, saving each as
  it finishes; a task whose results exist is skipped on resume. (A process per task cost
  ~2.5 minutes of imports and model loading per task, against as little as 40 seconds of
  compute.) The job reports its real interpreter PID, because a Windows venv's `python.exe`
  is a launcher, so VRAM and spill are attributed to the process that holds the model.
* **Seeds:** `--seed 1337` (random, numpy, torch and few-shot sampling).

### The FP16 reference cache

An 8B model in BF16 does not fit in 16 GB, so the FP16 model cannot be co-resident with
the quantized model on Tiers B and C. Therefore:

1. FP16 runs **once** over the fixed evaluation stream, with CPU offload as needed.
2. Its top-k log-probs are persisted, keyed by
   `(model + revision, tokenizer_sha, eval_stream_sha, topk, ctx, stride, add_bos)`.
3. Every later KL computation reads that cache. Change any key component and you get a
   different reference file rather than a silently incomparable number.

Tier A (4B) runs FP16 live on-GPU, which is what makes the cached-reference path auditable:
the same model must produce the same numbers both ways.

### Two KL estimators, labelled on every record

`quality.kl_div.estimator` says which one produced a number. They are not pooled.

| | `topk-reference` | `llama-perplexity` |
|---|---|---|
| Used for | Transformers, ExLlamaV3, vLLM (AWQ, GPTQ, FP8, NF4, EXL3) | GGUF formats |
| Reference | HF BF16 over the hashed id stream | the F16 GGUF, inside llama.cpp |
| Support | reference top-128, tail mass measured and recorded | full vocabulary; base log-probs stored as uint16 with a 16-nat floor |
| Positions | every position of each non-overlapping 2,048-token chunk except the first | positions 1,024–2,046 of each 2,048-token chunk, first 40 chunks (`native_kld_chunks`) |
| Top-k agreement | top-1 and top-5 | top-1 only ("Same top p") |

The GGUF path is llama.cpp's own base-file workflow, which §5.2.3 prefers over a
reimplementation. Its base stores ~2 bytes × vocabulary per scored token (~304 KB for
Qwen3's 151,936 tokens), so it covers a fixed, recorded 40 chunks (40,920 scored positions)
rather than the whole corpus. The base file also stores the exact token ids it scored; the
worker compares them with the hashed reference-tokenizer stream and records
`token_stream_verified`, so §6.6 is checked rather than assumed.

Every record carries `ppl_reference`, the reference's full-logit perplexity over the *same*
positions (for GGUF: the F16 run's own PPL, **not** llama-perplexity's `Mean PPL(base)`, which
is recomputed from floored stored log-probs and reads ~0.6% low — docs/failures.md),
so a PPL delta never mixes chunking conventions. `ppl_config.add_bos` is the *effective*
value: Qwen3's tokenizer has no BOS token, so it is `false` even though the eval config
requests one.

---

## 3. Memory

Computed in [`src/quantbench/metrics/memory.py`](../src/quantbench/metrics/memory.py).
Four numbers, reported separately — collapsing them is a common error:

| Metric | Definition |
|---|---|
| `vram_weights_mb` | Process VRAM delta measured after model load, before any KV allocation. |
| `vram_kv_mb` | KV cache allocation at the tested context/batch — engine-reported where available, measured as a delta otherwise. |
| `vram_peak_allocated_mb` | Framework allocator peak (`torch.cuda.max_memory_allocated`). Undercounts real usage. |
| `vram_peak_nvsmi_mb` | Process-level VRAM from NVML, sampled continuously at ≥ 5 Hz. **This is the number that decides whether a configuration actually fits**, and it exceeds the allocator figure. |
| `effective_bpw` | `(weight_file_bytes × 8) / non_embedding_param_count`. Embedding-excluded, and stated as such, because embedding quantization choices otherwise distort format comparisons. |

`effective_bpw` is computed as **(non-embedding tensor bytes × 8) / non-embedding
parameters**, both read from the artifact and checkpoint headers. The literal §5.3 formula
divides the whole file by the non-embedding count, which leaves the embedding's quant type
in the numerator (5.50 vs 4.78 for Qwen3-4B Q4_K_M); see open question Q12. It is `null`
whenever either count is unknown and is never inferred from the nominal bit width.

All `*_mb` fields are **MiB** (2^20 bytes), the unit both NVML and llama.cpp report.

### VRAM accounting modes

Every record states `vram_accounting`:

* `per_process` — NVML attributes memory to the worker's PID and any engine-server PIDs it
  registered. `vram_peak_attributed_mb` equals `vram_peak_nvsmi_mb`.
* `per_process_wddm` — Windows: NVML has no per-process figure under WDDM, so the sampler
  reads the OS's `\GPU Process Memory(pid_*)\Dedicated Usage` counter for the same PIDs
  (validated against a known 2,048 MiB allocation). The companion `Shared Usage` counter is
  recorded as `vram_shared_peak_mb`: pinned host buffers, plus — when the device is within
  1 GiB of full — device allocations spilled to system RAM by the driver's sysmem fallback.
  Such a spill means the configuration did not fit, and the run is `status: oom`.
* `device_wide` — no per-process figure at all (WSL2, or the counters unavailable). The sampler
  records device-wide usage, `vram_baseline_nvsmi_mb` is the usage just before load, and
  `vram_peak_attributed_mb = vram_peak_nvsmi_mb − baseline`. Other processes' allocations
  that change *during* a run are indistinguishable from the run's own; the baseline is
  recorded so the reader can judge. The raw `vram_peak_nvsmi_mb` is kept because it is what
  actually had to fit on that host.

Where an engine reports its own buffers (llama.cpp: `model buffer size`, `KV buffer size`,
`compute buffer size`), `vram_weights_mb` and `vram_kv_mb` use those engine-reported values,
which §5.3 permits and which are exact; the NVML load delta is only a fallback.

`OOM` is a first-class outcome (`status: "oom"`), recorded with the context and batch size
at which it occurred — not an exception that deletes the data point.

Per-process VRAM accounting is unavailable under WSL2; there the sampler falls back to
device-wide usage, which includes anything else on the card. Records made that way are
identifiable by `is_wsl: true`.

---

## 4. Power and efficiency

Computed in [`src/quantbench/metrics/power.py`](../src/quantbench/metrics/power.py) from
NVML samples taken at ≥ 10 Hz.

| Metric | Definition |
|---|---|
| `power_avg_w`, `power_peak_w` | Mean and max instantaneous board power over the measured window. |
| `energy_j` | Trapezoidal integration of power over the measured window. **Load and warmup are excluded** — energy spent loading weights is not energy spent generating tokens. |
| `tokens_per_joule` | Generated tokens in the measured window ÷ `energy_j`. The headline efficiency metric. |
| `sm_clock_avg_mhz`, `gpu_temp_start_c`, `gpu_temp_end_c` | NVML, over the same window. |
| `throttle_reasons`, `throttled` | Decoded NVML clock-throttle reason flags. |

Clock-event reasons are classified in three groups:

* **Contamination** (`hw_slowdown`, `sw_thermal_slowdown`, `hw_thermal_slowdown`,
  `hw_power_brake_slowdown`): any occurrence sets `throttled: true` and its share of the
  window is `throttled_fraction`. The analysis layer excludes these rows by default.
* **Power-limit regulation** (`sw_power_cap`): recorded as `power_capped` and
  `power_capped_fraction`. On this card it is asserted under ordinary load well below the
  300 W limit, so it is the normal boost regime rather than contamination; it is reported,
  not excluded (open question Q13).
* **Unknown bits** that the installed NVML binding does not name are kept as
  `unknown_0x…` rather than dropped (driver 617 sets `0x400`).
* A contamination reason active for **≥ 25% of the measured window** downgrades the whole
  record to `status: "throttled_invalid"`. A brief power-cap blip is worth flagging; a run
  spent mostly in thermal slowdown is not a measurement.

`gpu_idle` and `display_clock_setting` are not treated as contamination.
`applications_clocks_setting` means clocks were *locked*, which is recorded as a protocol
field (`clocks_locked`), not as throttling — but a locked-clock run and a boost-clock run
are still not comparable, so the flag travels with the data.

---

## 5. Protocol

| Control | Default | Why |
|---|---|---|
| Warmup iterations | 2, discarded | First-iteration effects (JIT, autotune, page faults) are not steady state. |
| Measured iterations | 5, retained with samples | Dispersion is a reported quantity, so it needs replicates. |
| Run order | Shuffled with a recorded seed (or round-robin across formats) | GPU temperature drifts monotonically over a long session. Running all Q4 configs then all Q5 configs makes format perfectly confounded with thermal state. |
| Cooldown gate | Wait until GPU ≤ 50 °C, up to 300 s | Same reason. The actual pre-run temperature is recorded. |
| Clock locking | Off by default, recorded when used | Consumer drivers often refuse it; comparability requires knowing which regime a run was in. |
| Process isolation | One subprocess per run | Prevents allocator fragmentation, engine global state, and one engine's OOM from poisoning the next. A segfault costs one data point, not the session. |
| Decoding | Greedy (`temperature=0`) everywhere; quality workloads reject anything else | Sampling noise is not a quantization effect. |
| Token counts | Fixed N with `ignore_eos` | Early EOS silently inflates throughput. A mismatch between requested and produced token counts raises a warning in the record. |
| Prompts | Exact token length, distinct slice per batch element | Prevents prefix-cache hits from deleting prefill from the measurement. Prefix caching is disabled explicitly where the engine supports it. |
| Engine auto-adjustment | Disabled (`llama-server --fit off -ngl all`); a run fails if fewer layers are offloaded than exist | An engine that silently shrinks context or offloads to CPU to make a config fit would turn an OOM into a different, unrecorded experiment. |
| Evaluation stream | One byte-identical pre-tokenized stream, SHA-256 recorded | Different tokenization of "the same" WikiText-2 is a real and frequently-missed source of PPL discrepancy. |
| Calibration hygiene | Calibration sets disjoint from the evaluation set, asserted in config tests | Evaluating on the calibration distribution is the field's most common self-inflicted wound. |
| Provenance | Driver, CUDA, torch + arch list, engine version/commit, attention backend, KV dtype, artifact SHA-256, quantizer version and command line, CPU/RAM, WSL flag, timestamp | Recorded per run, never in a global "environment" doc. |

### Statistical reporting

Medians and IQRs. **If two configurations' IQRs overlap, they are reported as
indistinguishable rather than ranked** — the recommendation table renders every tied
configuration rather than picking a winner
([`analysis/stats.py`](../src/quantbench/analysis/stats.py)).

---

## 6. Threats to validity

Read this section before quoting any number from this repo.

### 6.1 Cross-engine perplexity is not directly comparable

llama.cpp's perplexity computation, HF's sliding-window PPL and ExLlamaV3's evaluation
differ in chunking, stride and BOS handling. This repo's preferred resolution is (a):
compute all quality metrics in one harness over one token stream. Where an engine's native
tooling is used instead — specifically `llama-perplexity`'s `--kl-divergence-base`
workflow — the record is tagged:

* `quality.scored_by_engine` names the backend that produced the number;
* `kl_div.tail_mass_estimated` is set, because it is a different estimator from this
  repo's top-k computation;
* a warning is attached saying the value must not be compared directly against KL from
  other engines.

Plots that mix estimators label the boundary. Nothing is quietly averaged across it.

### 6.2 Quality and latency may come from different backends

vLLM does not expose log-probs on an arbitrary reference support, so quality for
vLLM-served formats is routed to a designated scoring engine (Transformers reading the same
checkpoint). The plan declares these routes before the run, and every record carries
`scored_by_engine`. A row whose speed came from vLLM and whose KL came from Transformers is
still a valid answer to "what does this artifact cost you", but it is not a measurement of
vLLM's numerics.

### 6.3 Publisher-sourced artifacts carry unknown provenance

Where an artifact is downloaded rather than produced here, the publisher and revision are
recorded and `producer` is null. Whether an importance matrix was used, and on what data,
is frequently undocumented upstream — and imatrix presence is exactly the confound this
repo's paired Q4_K_M ablation exists to isolate.

### 6.4 One card, one sample

Silicon lottery, case airflow, ambient temperature and PSU behaviour are uncontrolled. The
cooldown gate and throttle flags bound the damage; they do not eliminate it. Nothing here
generalises to a different 5070 Ti, let alone to datacenter Blackwell (`sm_100`), which is
explicitly out of scope.

### 6.5 Engine version churn

Results are a dated snapshot. Every record pins the engine version and commit, the torch
build and arch list, and the driver. A released tag carries the results that were true for
those versions.

### 6.6 WSL2

If the measurement host is WSL2, batch-1 latency is noisier than native Linux and
per-process VRAM accounting is unavailable. `is_wsl` is recorded per run and surfaces in
the README's environment table.

### 6.7 A shared desktop GPU: within-run IQR understates between-session variation

The measurement host is a desktop, and the compositor, browser and any other GPU clients
share the card. Their *work* is not controlled, only their memory is subtracted. Observed on
2026-10-17, same spec (`qwen3-4b__gguf-q4-k-m__llama-cpp__b1__ctx512__gen128__latency`), same
build, same day:

| session | decode tok/s p50 (IQR) | other processes' VRAM | power-capped |
|---|---|---|---|
| 1 (archived in `results/superseded/2026-10-17-device-wide-vram/`) | 184.7 (0.7) | ~4.7 GiB | 21% |
| 2 (current record) | 199.4 (3.9) | ~2.1 GiB | 0% |

The two IQRs are far apart, so the within-run IQR is not the whole uncertainty: rankings
between configurations closer than ~8% should not be read from single sessions. Q8_0 moved
only 0.4% across the same two sessions, so the effect is not uniform. Mitigations, in order of
strength: close other GPU clients before a run (record what was open), interleave several
sessions per configuration in the `core` profile so session-to-session variance enters the
IQR, and treat cross-session differences below that spread as ties.

### 6.8 Synthetic prompts

When no pinned evaluation stream exists for a model, latency prompts fall back to seeded
synthetic token ids. These are comparable across engines but are **not** a realistic text
workload; the record carries `prompt_source` and a warning, and such runs are excluded from
any claim about real-text prefill.

### 6.9 The mock engine

`--dry-run` and CI use a mock engine that fabricates timings. Records it produces are
marked `synthetic: true`, written to a git-ignored directory, refused by the analysis layer
unless explicitly requested, and watermarked on any figure built from them. It exists to
test the pipeline, and says nothing about hardware.
