# `quant-bench-5070ti` — Requirements Specification

> **Working title:** Consumer-GPU Quantization & Inference Benchmark Suite
> **Target hardware:** NVIDIA GeForce RTX 5070 Ti (Blackwell GB203, SM 12.0 / `sm_120`, 16 GB GDDR7)
> **Audience for the finished repo:** ML infrastructure / applied AI hiring managers, and local-LLM practitioners choosing a quant for a 16 GB card.
> **Spec version:** 1.0 — treat as the source of truth. If reality contradicts this document, update this document in the same PR.

---

## 0. Agent working agreement

Read this section before writing any code.

1. **Never fabricate a number.** Every value in a results file, table, plot, or README must be traceable to a run record produced by this harness. If a benchmark has not been run, the cell is `null` and renders as `—` with a footnote, not an estimate.
2. **Negative and null results ship.** "vLLM would not build for `sm_120` on driver X" is a finding, and belongs in `docs/failures.md` with the exact error. Do not silently drop configurations that failed.
3. **Ask before adding a dependency** that is not in §11.1. The dependency surface here is unusually fragile (CUDA 12.8+, Blackwell kernels) and every addition is a reproducibility liability.
4. **Build in milestone order (§10).** Each milestone has acceptance criteria. Do not start M(n+1) until M(n) passes. It is better to have M0–M3 excellent than M0–M6 half-broken.
5. **The harness must be runnable without a GPU** in `--dry-run` mode (validates configs, exercises the schema, regenerates plots from committed results). CI has no GPU.
6. **Prefer boring, inspectable engineering.** Subprocess isolation, JSON on disk, pure functions for analysis. No clever in-process engine juggling — see §7.3.
7. When a design decision is genuinely ambiguous, write the options into `docs/open-questions.md` and pick the conservative default rather than blocking.

---

## 1. Problem statement and thesis

Anyone can run `model.generate()`. The gap this repo fills: **on a 16 GB consumer card, which quantization format and inference engine should you actually pick, and what exactly do you give up?**

The repo must answer, with reproducible evidence:

- **Q1 — Quality cost.** How much output-distribution fidelity is lost at each bit-width and format (Q4_K_M, Q5_K_M, Q6_K, IQ4_XS, AWQ-INT4, GPTQ-INT4, FP8-W8A8, NF4, EXL3 @ various bpw) relative to a BF16/FP16 reference?
- **Q2 — Speed cost/benefit.** How do prefill throughput, decode throughput, TTFT and inter-token latency vary with format, batch size, and context length?
- **Q3 — Memory reality.** What is the *actual* VRAM footprint — weights vs. KV cache vs. activations vs. allocator overhead — and where is the cliff at which a model stops fitting in 16 GB?
- **Q4 — Efficiency.** Tokens per joule. Which configuration is cheapest to run, not just fastest?
- **Q5 — The Pareto answer.** For a given quality budget, what is the fastest configuration that fits? Produce a defensible recommendation table.

**Falsifiable claims the repo should be able to make or refute** (write these in the README as hypotheses, then answer them with data):

- H1: Q5_K_M is a strictly better quality-per-VRAM tradeoff than Q4_K_M for models ≤ 8B, but the ordering inverts once KV cache pressure at long context dominates.
- H2: Perplexity delta materially understates quantization damage compared to KL-divergence against the FP16 reference; two quants with near-identical PPL can differ substantially in top-1 token agreement.
- H3: On `sm_120`, INT4 formats with Marlin-class kernels (AWQ/GPTQ via vLLM) beat GGUF Q4_K_M on batched throughput, but GGUF wins or ties at batch size 1 — the regime most local users actually run.
- H4: KV cache quantization buys more usable context per GB than weight quantization does, per unit of quality lost.

Confirming, refuting, or being unable to resolve any of these is an acceptable outcome. Stating them up front is what makes the repo read as science rather than a spec dump.

---

## 2. Non-goals

Explicitly out of scope. State these in the README so reviewers do not read absence as oversight.

- Training, fine-tuning, or LoRA. (QLoRA/NF4 appears only as an *inference* baseline.)
- Multi-GPU, tensor parallelism, distributed serving.
- CPU-only or Apple Silicon inference.
- Subjective "vibe" quality evaluation or LLM-as-judge scoring.
- Production serving concerns: autoscaling, request routing, auth.
- Beating published SOTA numbers. The contribution is *methodology and a reproducible single-card matrix*, not a leaderboard entry.
- Datacenter Blackwell (`sm_100`). Findings are scoped to consumer GB20x.

---

## 3. Environment and hardware constraints

### 3.1 Reference environment (must be auto-detected and recorded, never hardcoded)

| Component | Expected | Notes |
|---|---|---|
| GPU | RTX 5070 Ti, 16 GB GDDR7 | compute capability 12.0 (`sm_120`) |
| Driver | ≥ 570.x | record exact version per run |
| CUDA runtime | ≥ 12.8 | `sm_120` requires 12.8+; 12.9/13.x preferred if the stack supports it |
| PyTorch | cu128 or newer build with `sm_120` in `torch.cuda.get_arch_list()` | **assert this at startup** |
| OS | Linux (native preferred; WSL2 acceptable but must be flagged in metadata) | WSL2 adds measurable latency variance — record `is_wsl` |
| Python | 3.11 or 3.12 | |

### 3.2 Blackwell-specific hazards (design around these from day one)

These are the known sharp edges as of mid-2026. The harness must fail *loudly and informatively*, never silently fall back:

- **PyTorch arch list.** Wheels not built with `sm_120` produce `CUDA error: no kernel image is available for execution on the device` or an explicit "compute capability sm_120 is not compatible" message. Add a preflight check that prints `torch.cuda.get_arch_list()` and aborts with a remediation hint if `sm_120` is absent.
- **vLLM prebuilt wheels have historically shipped without Blackwell consumer arch flags**, requiring either a source build with `TORCH_CUDA_ARCH_LIST="12.0"` or a specific nightly. Community reports indicate AWQ works on `sm_120` via the `awq_marlin` kernel path, sometimes requiring the Triton attention backend rather than FlashAttention. Treat the vLLM install as its own documented, versioned artifact (§11.2) with a fallback (§12).
- **Attention backend availability varies.** FlashAttention builds for `sm_120` may be unavailable or broken depending on version; FlashInfer or `TRITON_ATTN` may be required. **The attention backend is a recorded variable, not an implementation detail** — it materially changes latency and must appear in every run record.
- **FP8 on consumer Blackwell.** GB203 has FP8 tensor cores, but the vLLM `cutlass_scaled_mm` path for block-scaled FP8 has had `sm_120` gaps. Verify empirically before promising FP8 in the README; if unsupported, document it as a finding.
- **FP4 / NVFP4** is a stretch goal only. Verify hardware + kernel support empirically before including.

### 3.3 VRAM budget arithmetic (drives the model ladder)

16 GB is roughly 15.5 GiB usable after display/compositor overhead. Weight footprint ≈ `params × bytes_per_param`, plus KV cache ≈ `2 × layers × kv_heads × head_dim × ctx × batch × bytes`, plus activations and allocator slack.

The practical consequence: **an 8B model in BF16 does not fit** (≈16 GB weights alone, before KV). This constrains the design of the quality reference — see §5.2.

---

## 4. Experimental design

### 4.1 The model ladder

Three tiers, each answering a different question. Model choice is config-driven; these are the defaults.

| Tier | Model | Why |
|---|---|---|
| **A — Anchor** | `Qwen3-4B` (or `Llama-3.2-3B-Instruct`) | Small enough that BF16 fits comfortably alongside KV cache. This is the only tier where the FP16 reference runs *live on-GPU*, so it anchors the entire quality methodology. Full matrix runs here. |
| **B — Daily driver** | `Qwen3-8B` (or `Llama-3.1-8B-Instruct`) | The realistic 16 GB workhorse. All quants fit; BF16 does not. Reference logits are computed once and cached (§5.2). Full matrix minus live BF16. |
| **C — Frontier** | `Qwen3-14B` | The "what actually fits" tier. Several quants will OOM — *that is the result*. Reduced matrix: 4-bit and 5-bit formats, KV-cache quant sweep, context-length cliff mapping. |
| **D — Stretch (optional)** | `Qwen3-30B-A3B` (MoE) | Only if A–C complete. MoE changes the memory/compute ratio entirely and makes a good "further work" section. |

Model files must be pinned by **repo revision SHA and per-file SHA-256**. Quantized artifacts are either (a) produced locally by this repo's quantization pipeline, or (b) downloaded from a named publisher — in which case the publisher, revision, and whether an importance matrix (imatrix) was used **must** be recorded, because imatrix presence is a confound that has misled published comparisons.

### 4.2 The format × engine matrix

| Format | Producer tooling | Inference engine | Notes |
|---|---|---|---|
| BF16 / FP16 | — | HF Transformers, llama.cpp (F16 GGUF) | Reference. Tier A live; Tier B/C cached logits. |
| GGUF `Q8_0` | `llama-quantize` | llama.cpp | Near-lossless upper anchor. |
| GGUF `Q6_K` | `llama-quantize` | llama.cpp | |
| GGUF `Q5_K_M` | `llama-quantize` | llama.cpp | |
| GGUF `Q4_K_M` | `llama-quantize` | llama.cpp | The de facto default — must be characterized precisely. |
| GGUF `IQ4_XS` | `llama-quantize` (+imatrix) | llama.cpp | i-quant; tests whether imatrix beats larger K-quants at lower bpw. |
| GGUF `Q4_K_M` **+imatrix** | `llama-imatrix` → `llama-quantize` | llama.cpp | Paired with the non-imatrix build to **isolate the imatrix effect**. This is a genuinely under-reported comparison. |
| AWQ INT4 (W4A16) | `llm-compressor` | vLLM (`awq_marlin`) | **`AutoAWQ` is deprecated/archived — do not use it.** Functionality moved to `llm-compressor`. |
| GPTQ INT4 (W4A16) | `GPTQModel` | vLLM / HF Transformers | **`AutoGPTQ` is archived — do not use it.** `GPTQModel` is the maintained successor and includes Blackwell (`sm_100`+) GPTQ/AWQ kernel work. |
| FP8 W8A8 | `llm-compressor` | vLLM | Conditional on `sm_120` kernel support — verify, don't assume. |
| NF4 (bitsandbytes) | — | HF Transformers | Widely used, usually poor on latency. Include as the "convenience baseline". |
| EXL3 @ {3.0, 4.0, 5.0, 6.0} bpw | `exllamav3` `convert.py` | ExLlamaV3 | Trellis/QTIP-based; the strongest quality-per-bit contender and the format most people haven't benchmarked properly. Its variable-bpw knob makes it ideal for a smooth quality-vs-bpw curve. |

**Cross-cutting sweeps** applied where the engine supports them:

- **Batch size:** {1, 2, 4, 8, 16, 32} — batch 1 is the local-user regime and must not be buried.
- **Context / prompt length:** {512, 2048, 8192, 32768} tokens of prefill.
- **Generation length:** {128, 512} decode tokens.
- **KV cache dtype:** {fp16, q8_0, q4_0} (llama.cpp), {auto, fp8} (vLLM). Sweep on Tier B and C only.
- **Attention backend:** recorded always; swept on at least one Tier B configuration.

Full cross-product is combinatorially infeasible. Define **run profiles** in config: `smoke` (~10 runs, minutes), `core` (the headline matrix, hours), `full` (everything, overnight+). The README publishes `core`; `full` results ship if they complete.

---

## 5. Metrics — precise definitions

Every metric below must be defined in `docs/methodology.md` with the same wording used in code docstrings. Ambiguous metric definitions are the single most common reason benchmark repos get dismissed.

### 5.1 Latency and throughput

| Metric | Definition |
|---|---|
| `ttft_ms` | Wall time from request submission to first generated token emitted. |
| `itl_ms` | Inter-token latency: per-token deltas after the first. Report p50, p90, p95, p99, and std. **Never report only the mean.** |
| `decode_tok_s` | `(generated_tokens - 1) / (t_last_token - t_first_token)`, per request. |
| `prefill_tok_s` | `prompt_tokens / ttft_ms × 1000`. Note this conflates scheduling overhead into prefill; state that explicitly. |
| `e2e_latency_ms` | Submission to final token. |
| `throughput_tok_s_total` | Aggregate generated tokens / wall time across the whole batch — the serving-side number, distinct from per-request decode rate. |

Report **median and IQR across repetitions**, not mean ± std, unless the distribution is verified near-normal.

### 5.2 Quality

**Primary metric: KL divergence vs. the FP16 reference.** Perplexity is the familiar number but it is a weak instrument — a quant can post *lower* PPL than its own FP16 parent while being demonstrably less faithful, and calibration-set overlap makes PPL easy to accidentally game. KL divergence over the pre-sampling output distribution directly measures "how differently does this model think."

Required quality outputs per configuration:

- `ppl_wikitext2` — perplexity on WikiText-2 (raw), fixed context 2048, fixed stride, fixed token count. **Record the exact chunking parameters**; they are not comparable across tools otherwise.
- `kl_div_mean`, `kl_div_p90`, `kl_div_p99`, `kl_div_max` vs. FP16 reference logits.
- `top1_agreement` — fraction of positions where argmax matches the reference. Also `top5_agreement`.
- Downstream accuracy via `lm-evaluation-harness`: `arc_challenge` (0-shot), `hellaswag` (0-shot), `gsm8k` (8-shot, exact match), `mmlu` (5-shot, a fixed named subset of subjects to bound runtime). Report absolute and **retention % vs. FP16**.
- Optional: a long-context retrieval probe (needle-in-haystack style) at 8k/32k, to catch degradation that short-context PPL misses entirely.

**Reference-logit caching (critical design element).** For Tier B/C the FP16 model cannot be co-resident with the quantized model. Therefore:

1. Run the FP16 model once over the fixed evaluation token stream, offloading to CPU/RAM as needed (llama.cpp partial offload, or `device_map="auto"` with CPU offload). This is slow; it happens once.
2. Persist the reference top-k logits (k configurable, default 128, plus the full-vocab log-sum-exp needed to renormalize) to a compressed artifact keyed by `(model, tokenizer_sha, eval_stream_sha)`.
3. All subsequent KL computations read the cached reference. `llama.cpp`'s perplexity tool supports exactly this base-file workflow; ExLlamaV3 ships comparable model-diff tooling. Where a native path exists, prefer it over reimplementation, but **normalize the output into this repo's schema**.

### 5.3 Memory

Report all four, separately — collapsing them is a common error:

- `vram_weights_mb` — measured delta after model load, before any KV allocation.
- `vram_kv_mb` — measured (or engine-reported) KV cache allocation at the tested context/batch.
- `vram_peak_allocated_mb` — framework allocator peak (`torch.cuda.max_memory_allocated`) where applicable.
- `vram_peak_nvidia_smi_mb` — process-level VRAM from NVML, sampled at ≥ 5 Hz for the run duration. This is the number that determines whether it *actually fits*, and it exceeds the allocator number.
- `effective_bpw` — computed as `(weight_file_bytes × 8) / non_embedding_param_count`. Report the embedding-excluded figure and say so; embedding quantization choices otherwise distort format comparisons.

`OOM` is a first-class outcome value, not an exception. Record it with the context/batch at which it occurred.

### 5.4 Power and efficiency

- `power_avg_w`, `power_peak_w` — NVML sampled at ≥ 10 Hz for the run window.
- `energy_j` — trapezoidal integration of power over the run.
- `tokens_per_joule` — the headline efficiency metric.
- `gpu_temp_start_c`, `gpu_temp_end_c`, `sm_clock_avg_mhz`, `throttle_reasons` — NVML clock-throttle reason flags. **A run where thermal or power throttling was active must be flagged in the record**, and the analysis layer must be able to exclude flagged runs.

---

## 6. Methodological rigor requirements

This section is what separates the repo from a blog post. Implement all of it.

1. **Warmup and repetition.** Each configuration: ≥ 2 discarded warmup iterations, then ≥ 5 measured repetitions. Configurable; defaults recorded in the run record.
2. **Randomized, interleaved run order.** Do **not** run all Q4 configs then all Q5 configs — GPU temperature drifts monotonically over a long session and would be perfectly confounded with format. Shuffle the run queue with a recorded seed, or round-robin across formats.
3. **Thermal control.** Optional inter-run cooldown until GPU temp ≤ a configured threshold (default 50 °C) or a timeout. Record actual pre-run temp. Optionally lock clocks via `nvidia-smi --lock-gpu-clocks` when permissions allow, and record whether locking was active — a locked-clock run and a boost-clock run are not comparable.
4. **Process isolation.** One subprocess per run, fresh CUDA context, results written to a JSON file the parent reads. This prevents allocator fragmentation, engine global state, and one engine's OOM from poisoning the next. It also means a segfault in vLLM costs one data point, not the session.
5. **Determinism where it matters.** Fixed seeds; greedy decoding (`temperature=0`) for all quality evaluation. Latency runs use fixed token counts with `ignore_eos=True` (or equivalent) so every run generates exactly N tokens — otherwise early EOS silently inflates throughput.
6. **Identical token streams.** All quality evaluations run over a byte-identical, pre-tokenized evaluation stream, hashed and recorded. Different tokenization of "the same" WikiText-2 is a real and frequently-missed source of PPL discrepancy.
7. **Calibration hygiene.** Where a format requires calibration data (AWQ, GPTQ, imatrix, EXL3), the calibration set **must be disjoint from the evaluation set**, and this must be stated in the results. Evaluating on the calibration distribution is the field's most common self-inflicted wound.
8. **Full provenance per run.** Driver, CUDA, torch, engine version *and git SHA*, kernel/attention backend, model file SHA-256, quantizer version, quantization command line, host CPU/RAM, WSL flag, timestamp, ambient conditions if available.
9. **Statistical honesty.** Report dispersion. If two configurations' IQRs overlap, the README says they are indistinguishable rather than ranking them.

### 6.1 Validity threats to document explicitly

`docs/methodology.md` must contain a "Threats to validity" section covering at minimum:

- **Cross-engine perplexity is not directly comparable.** llama.cpp's perplexity computation, HF's sliding-window PPL, and ExLlamaV3's evaluation differ in chunking, stride, and BOS handling. Either (a) compute all quality metrics in one harness over one token stream, or (b) present within-engine comparisons only and label the boundary clearly on every plot. **Option (a) is preferred; if not achievable for some engine, say so rather than quietly mixing.**
- Publisher-sourced quants may carry unknown imatrix/calibration provenance.
- Single-GPU, single-sample-of-one-card: silicon lottery and cooling are uncontrolled.
- Engine version churn — results are a snapshot, dated in the README.
- Batch-1 measurements on WSL2 are latency-noisy relative to native Linux.

---

## 7. Repository architecture

### 7.1 Layout

```
quant-bench-5070ti/
├── README.md                     # headline chart, findings, recommendation table
├── requirements.md               # this document
├── pyproject.toml                # uv-managed
├── Makefile                      # make smoke | core | full | plots | lint
├── configs/
│   ├── hardware/5070ti.yaml
│   ├── models/{qwen3-4b,qwen3-8b,qwen3-14b}.yaml
│   ├── formats/{gguf,awq,gptq,fp8,nf4,exl3}.yaml
│   ├── profiles/{smoke,core,full}.yaml
│   └── eval/{wikitext2,lm_eval_tasks,longctx}.yaml
├── src/quantbench/
│   ├── cli.py                    # typer entrypoint
│   ├── schema.py                 # pydantic models: RunSpec, RunRecord, QualityRecord
│   ├── planner.py                # config → shuffled run queue, resumable
│   ├── runner.py                 # subprocess orchestration, timeouts, OOM capture
│   ├── engines/
│   │   ├── base.py               # Engine protocol: load/prefill/decode/teardown/report
│   │   ├── llamacpp.py
│   │   ├── vllm.py
│   │   ├── exllamav3.py
│   │   └── transformers_hf.py
│   ├── quantize/                 # producing artifacts, not just consuming them
│   │   ├── gguf.py               # llama-imatrix + llama-quantize wrappers
│   │   ├── compressor.py         # llm-compressor: AWQ, FP8
│   │   ├── gptqmodel.py
│   │   └── exl3.py
│   ├── metrics/
│   │   ├── latency.py            # TTFT/ITL/percentiles
│   │   ├── memory.py             # NVML + allocator sampling
│   │   ├── power.py              # NVML power integration, throttle flags
│   │   └── quality.py            # PPL, KL-div, top-k agreement, lm-eval bridge
│   ├── reference/
│   │   └── logits_cache.py       # FP16 reference generation + storage
│   ├── analysis/
│   │   ├── load.py               # results/*.jsonl → dataframe
│   │   ├── stats.py              # medians, IQR, pareto frontier extraction
│   │   └── plots.py              # every figure, deterministic
│   └── preflight.py              # sm_120 checks, disk space, model presence
├── results/
│   ├── raw/<run_id>.json         # one file per run, never edited
│   ├── results.parquet           # consolidated, committed
│   └── environment.json          # captured env snapshot
├── figures/                      # committed SVG + PNG
├── docs/
│   ├── methodology.md
│   ├── failures.md               # what didn't build/run and why
│   ├── open-questions.md
│   └── findings.md               # the long-form writeup
└── tests/
```

### 7.2 The `Engine` protocol

A single interface every backend implements. Keeping this narrow is what makes the comparison fair.

```python
class Engine(Protocol):
    name: str
    def preflight(self, spec: RunSpec) -> PreflightResult: ...      # can this even run? cheap.
    def load(self, spec: RunSpec) -> LoadReport: ...                # returns vram_weights_mb, load_time_s
    def generate(self, batch: list[str], spec: RunSpec) -> GenReport: ...  # per-token timestamps required
    def score(self, tokens: IntArray, spec: RunSpec) -> ScoreReport: ...   # logprobs for PPL/KL
    def teardown(self) -> None: ...
    def describe(self) -> EngineMetadata: ...                       # version, commit, kernels, backend
```

`generate` **must return per-token arrival timestamps**, not just a total. Deriving ITL percentiles from an aggregate is impossible, and aggregate-only measurement is the most common flaw in amateur benchmarks.

Engines that cannot implement `score` (e.g. a server-only path) declare so in `preflight`; the planner then routes quality evaluation for that format through a designated scoring engine and **records that the quality and latency numbers came from different backends**.

### 7.3 Orchestration model

```
plan → queue (shuffled, resumable) → for each RunSpec:
    spawn subprocess: python -m quantbench.worker --spec spec.json --out record.json
    parent enforces timeout, captures stdout/stderr, classifies exit
    on success: validate record against schema, append to results/raw/
    on OOM / build failure / timeout: write record with status + error text, continue
```

The queue is resumable: re-running skips `RunSpec`s whose `(spec_hash, environment_hash)` already exists in `results/raw/`, unless `--force`. Overnight runs must survive a crash at hour six.

---

## 8. Data schemas

### 8.1 `RunSpec` (input)

```yaml
run_id: "qwen3-8b__gguf-q4km-imatrix__llamacpp__b1__ctx2048__gen128__r3"
spec_version: 1
model:
  name: "Qwen3-8B"
  hf_repo: "Qwen/Qwen3-8B"
  revision: "<commit-sha>"
  params_total: 8190735360
  params_non_embedding: 7000000000
quant:
  format: "gguf"
  variant: "Q4_K_M"
  bits_nominal: 4
  imatrix: true
  imatrix_source: "calib/imatrix-c4-512.dat"
  calibration_dataset: "c4-en-512seq"      # null if none
  artifact_path: "models/qwen3-8b-q4km-imat.gguf"
  artifact_sha256: "..."
  producer: "llama-quantize"
  producer_version: "b4321"
engine:
  name: "llama.cpp"
  version: "b4321"
  commit: "..."
  attention_backend: "cuda-fa"
  kv_cache_dtype: "f16"
  n_gpu_layers: -1
  flags: ["--no-mmap"]
workload:
  kind: "latency"                          # latency | quality | memory-sweep
  batch_size: 1
  prompt_tokens: 2048
  generate_tokens: 128
  ignore_eos: true
  temperature: 0.0
  seed: 1337
protocol:
  warmup_iters: 2
  measured_iters: 5
  cooldown_target_c: 50
  clocks_locked: false
```

### 8.2 `RunRecord` (output)

```json
{
  "spec_hash": "sha256:...",
  "spec": { "...": "full RunSpec echoed back" },
  "status": "ok",
  "environment": {
    "driver": "570.xx", "cuda_runtime": "12.8", "torch": "2.x+cu128",
    "torch_arch_list": ["sm_90", "sm_120"],
    "gpu_name": "NVIDIA GeForce RTX 5070 Ti", "gpu_uuid": "GPU-...",
    "vram_total_mb": 16303, "is_wsl": false,
    "cpu": "...", "ram_gb": 64, "os": "...", "timestamp_utc": "..."
  },
  "latency": {
    "ttft_ms": {"p50": 0, "p90": 0, "p99": 0, "iqr": 0, "samples": []},
    "itl_ms":  {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "std": 0},
    "decode_tok_s": {"p50": 0, "iqr": 0},
    "prefill_tok_s": {"p50": 0},
    "throughput_tok_s_total": {"p50": 0},
    "per_token_timestamps_ms": [[]]
  },
  "memory": {
    "vram_weights_mb": 0, "vram_kv_mb": 0,
    "vram_peak_allocated_mb": 0, "vram_peak_nvsmi_mb": 0,
    "effective_bpw": 0.0, "load_time_s": 0.0
  },
  "power": {
    "power_avg_w": 0, "power_peak_w": 0, "energy_j": 0,
    "tokens_per_joule": 0, "sm_clock_avg_mhz": 0,
    "gpu_temp_start_c": 0, "gpu_temp_end_c": 0,
    "throttle_reasons": [], "throttled": false
  },
  "quality": {
    "reference_id": "qwen3-8b-fp16@evalstream-sha",
    "eval_stream_sha": "...",
    "ppl_wikitext2": 0.0,
    "ppl_config": {"ctx": 2048, "stride": 2048, "tokens": 335000, "add_bos": true},
    "kl_div": {"mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0},
    "top1_agreement": 0.0, "top5_agreement": 0.0,
    "lm_eval": {"arc_challenge": 0.0, "hellaswag": 0.0, "gsm8k": 0.0, "mmlu_subset": 0.0},
    "scored_by_engine": "llama.cpp"
  },
  "error": null,
  "stderr_tail": null,
  "duration_s": 0.0
}
```

`status ∈ {ok, oom, engine_unsupported, build_failed, timeout, crashed, throttled_invalid}`.

Schema is versioned (`spec_version`). Analysis code validates on load and refuses records it cannot interpret.

---

## 9. Deliverables

### 9.1 CLI

```bash
quantbench preflight                          # env checks, sm_120 assertion, disk, model presence
quantbench models fetch --config configs/models/qwen3-8b.yaml
quantbench quantize --model qwen3-8b --formats gguf:Q4_K_M,gguf:Q5_K_M,awq,gptq,exl3:4.0
quantbench reference build --model qwen3-8b   # FP16 logit cache (slow, once)
quantbench plan --profile core --out plan.json
quantbench run --plan plan.json [--resume] [--dry-run]
quantbench analyze --results results/ --out results/results.parquet
quantbench plot --results results/results.parquet --out figures/
quantbench report                             # regenerate README tables + docs/findings.md tables
```

`make core` chains plan → run → analyze → plot → report.

### 9.2 Figures (all deterministic, regenerable without a GPU from committed results)

1. **Headline Pareto** — KL-divergence (x, log scale, lower=better) vs. decode tok/s at batch 1 (y). Marker size = peak VRAM, color = engine, shape = format family. Pareto frontier drawn. **This is the README hero image.**
2. **Quality vs. effective bpw** — KL-div and PPL delta against measured bpw, one line per format family. Reveals whether EXL3/i-quants genuinely dominate K-quants per bit.
3. **PPL vs. KL disagreement plot** — scatter of PPL delta against KL divergence, to visually support or refute H2.
4. **VRAM composition** — stacked bars (weights / KV / activations+overhead) per config, with a 16 GB ceiling line. Configs that OOM shown as hatched bars past the line.
5. **Throughput vs. batch size** — per format, log-x. Shows where batched INT4 kernels overtake GGUF.
6. **TTFT vs. prompt length** — prefill scaling.
7. **ITL distribution** — ECDF or violin per format at batch 1. Tail latency is invisible in a mean.
8. **Context cliff** — max achievable context per format at batch 1, with and without KV quantization.
9. **Efficiency** — tokens per joule, ranked.
10. **Accuracy retention** — lm-eval scores as % of FP16, grouped bars with CI whiskers.
11. **imatrix ablation** — paired comparison of identical quant levels with and without importance matrix.

Style: one consistent theme, readable at README width, colorblind-safe palette, every axis labeled with units, every figure with a caption stating N and the hardware. Export SVG + PNG.

### 9.3 Written artifacts

- **README.md** — hero chart, 5-line summary of findings, the recommendation table ("if you want X on 16 GB, use Y"), reproduction instructions, snapshot date, non-goals, limitations. Reviewers will spend ninety seconds here; it must carry the whole argument.
- **docs/methodology.md** — metric definitions, protocol, threats to validity.
- **docs/findings.md** — the long-form analysis. Per-hypothesis verdicts with evidence.
- **docs/failures.md** — build failures, unsupported paths, dead ends, with exact errors and versions. This is a credibility asset, not an embarrassment.
- **results/results.parquet + raw JSON** — committed so anyone can re-analyze without a GPU.

---

## 10. Milestones and acceptance criteria

**M0 — Skeleton and preflight**
- Repo scaffolding, `pyproject.toml`, ruff + mypy + pytest green, `Makefile`.
- `quantbench preflight` detects GPU, asserts `sm_120` in torch arch list, reports driver/CUDA/VRAM/WSL, writes `results/environment.json`.
- Pydantic schemas for `RunSpec`/`RunRecord` with round-trip tests.
- ✅ *Accept when:* preflight runs clean on the 5070 Ti, and `--dry-run` planning works on a machine with no GPU.

**M1 — One engine, end to end**
- llama.cpp engine: load, generate with per-token timestamps, teardown. NVML memory + power sampling. Subprocess runner with timeout and OOM classification.
- Smoke profile: Tier A model, Q4_K_M and Q8_0, batch 1, one context length.
- ✅ *Accept when:* `make smoke` produces schema-valid records with non-null latency, memory, and power, and re-running with `--resume` skips completed runs.

**M2 — Quality pipeline**
- Fixed evaluation token stream (hashed). FP16 reference logit cache. PPL + KL-div + top-k agreement. lm-eval-harness bridge.
- ✅ *Accept when:* Tier A Q8_0 shows near-zero KL vs FP16 and Q4_K_M shows measurably higher — i.e. the instrument demonstrably detects a known effect. Add this as a regression test with a tolerance band.

**M3 — Analysis and plots**
- Parquet consolidation, median/IQR aggregation, Pareto frontier extraction, figures 1–5.
- ✅ *Accept when:* `quantbench plot` regenerates all figures from committed results on a GPU-less machine, byte-stable across runs.

**M4 — Second and third engines**
- vLLM (AWQ + GPTQ, `awq_marlin`, attention backend recorded) and ExLlamaV3 (EXL3 at multiple bpw). Document the exact install recipe for each in `docs/`. If vLLM cannot be made to work on `sm_120`, execute fallback §12.1 and write it up.
- ✅ *Accept when:* the same `RunSpec` shape drives all three engines and cross-engine caveats are enforced in code (quality metrics tagged with `scored_by_engine`).

**M5 — Full matrix + sweeps**
- Tiers A/B/C, batch and context sweeps, KV-cache quant sweep, imatrix ablation, power/efficiency. Randomized interleaved execution with cooldowns.
- ✅ *Accept when:* the `core` profile completes unattended, OOMs are recorded rather than fatal, and every hypothesis in §1 has supporting or refuting data.

**M6 — Publication**
- README with hero chart and recommendation table, findings writeup, failures doc, CI (lint + tests + plot regeneration), LICENSE, CITATION.cff, tagged release with results attached.
- ✅ *Accept when:* a stranger can clone, run `make plots`, and reproduce every figure without a GPU; and with a 16 GB Blackwell card can reproduce the numbers within stated tolerance.

---

## 11. Engineering standards

### 11.1 Approved dependencies

Core: `pydantic`, `typer`, `pyyaml`, `pandas`, `pyarrow`, `numpy`, `matplotlib`, `nvidia-ml-py` (NVML), `rich`, `pytest`, `ruff`, `mypy`.
Engines/quantizers (each optional-extra, never a hard import at module top level): `torch`, `transformers`, `accelerate`, `bitsandbytes`, `vllm`, `llmcompressor`, `gptqmodel`, `exllamav3`, `lm-eval`. llama.cpp is used as **built binaries invoked as subprocesses**, pinned by tag — not as a Python binding — because the CLI tools (`llama-bench`, `llama-perplexity`, `llama-quantize`, `llama-imatrix`) are the maintained, well-defined surface.

Note the renamed llama.cpp binaries (`llama-cli`, `llama-server`, `llama-quantize`) — older tutorials reference `main`/`server`/`quantize`.

Import engine libraries **lazily, inside the worker subprocess**. `import vllm` at package import time will break the CLI on a machine where vLLM won't build, which is precisely the machine you need the CLI to work on.

### 11.2 Environment reproducibility

- `uv` with a committed lockfile; separate dependency sets per engine since their torch requirements conflict. **Amended 2026-08-15:** these are `requirements/{hf,vllm,exl3}.txt`, one isolated environment per engine, *not* `[project.optional-dependencies]` extras. A single uv lockfile resolves all extras together, and resolving vLLM requires building it, which fails on any host without `CUDA_HOME` — making the CLI uninstallable on exactly the machine that needs `quantbench preflight` to diagnose the problem. See `docs/open-questions.md` Q8 and `docs/failures.md`. llama.cpp keeps no Python dependency surface at all.
- One documented install recipe per engine in `docs/install/<engine>.md`, including the `TORCH_CUDA_ARCH_LIST="12.0"` source-build path for vLLM and the exact wheel/nightly used.
- Optional Dockerfile per engine. Given the `sm_120` situation, containerized builds may be the only reliable reproduction path — treat as strongly recommended, not required.
- Record every version in every run record. Never rely on a global "environment" doc to describe a specific number.

### 11.3 Code quality

- Type hints throughout; `mypy --strict` on `src/quantbench/` excluding engine shims.
- Analysis functions pure and unit-tested against fixture records.
- Fixture-based tests: a `MockEngine` producing synthetic timings so the whole pipeline is testable in CI without a GPU.
- Structured logging (`rich` console + JSONL log file per session).
- CI: lint, type check, tests, and figure regeneration on every PR. No GPU jobs.

---

## 12. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| vLLM will not build/run on `sm_120` within reasonable effort | Medium-high | **12.1:** Document the failure thoroughly in `docs/failures.md` with versions and errors. Substitute HF Transformers + `GPTQModel` kernels for INT4 coverage, and lean on ExLlamaV3 as the second high-performance engine. The comparison survives; the honesty about *why* is itself a signal of competence. |
| FP8 W8A8 unsupported on consumer Blackwell path | Medium | Drop from the core matrix; document as a finding with the specific kernel error. |
| Tier C (14B) OOMs on most formats | Expected | This is a result. Publish the fit/no-fit table — it is exactly what a 16 GB owner wants. |
| Full matrix exceeds available time | High | The `smoke`/`core`/`full` profile split exists for this. Ship `core`. |
| Thermal throttling contaminates results | Medium | Cooldown gates, throttle flags in records, analysis excludes flagged runs, optional clock locking. |
| Quantized artifacts consume excessive disk | High (100s of GB) | Configurable model cache dir, `quantbench models prune`, and a disk-space check in preflight. Never auto-delete without confirmation. |
| Engine version churn invalidates results | Certain over time | Date-stamp the README, pin every version, tag a release with results attached. |
| Scope creep into training/fine-tuning | Medium | §2 non-goals are binding. |

---

## 13. Open questions for the human

Record answers in `docs/open-questions.md` as they're decided; defaults in brackets apply until then.

1. Model family — Qwen3 throughout, or mix in Llama 3.x for architecture diversity? [Qwen3 primary, one Llama tier-A run as a sanity check]
2. Are quantized artifacts produced locally end-to-end, or are some pulled from established publishers? [Produce locally where feasible; document provenance either way]
3. Native Linux or WSL2 as the primary measurement environment? [Prefer native; if WSL2, flag prominently and consider it a stated limitation]
4. Is clock locking available (permissions/driver)? Affects comparability across the session.
5. Disk budget for the model cache? Drives how much of the `full` profile is feasible.
6. Publication target — GitHub repo only, or repo + writeup post? [Repo first; `docs/findings.md` is written so it can become a post with minimal editing]
7. Include a long-context retrieval probe, or leave to further work? [Stretch goal after M5]

---

## 14. Definition of done

The project is complete when a reader who has never met you can, in under ten minutes:

- see one chart that tells them which quantization to use on a 16 GB card and what it costs them,
- find the precise definition of every metric behind that chart,
- see what was tried and failed, with errors,
- and reproduce every figure from committed data with a single command.

Everything else in this document is in service of that.
