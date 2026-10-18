# quant-bench-5070ti

**Which quantization format and inference engine should you actually run on a 16 GB
consumer card, and what exactly do you give up?**

A reproducible single-card benchmark matrix for the NVIDIA RTX 5070 Ti (Blackwell GB203,
`sm_120`, 16 GB GDDR7), covering GGUF K-quants and i-quants, AWQ, GPTQ, FP8, NF4 and EXL3
across llama.cpp, vLLM, ExLlamaV3 and HF Transformers.

> **Status: M1 and M2 complete on real hardware (Tier A `smoke` only).**
> Runs: 2026-10-17/18, RTX 5070 Ti, driver 617.14, llama.cpp b11706, native Windows 11.
> One model (Qwen3-4B), two GGUF quants against BF16/F16 references, so the tables below
> demonstrate the instrument; they are not yet the recommendation the repo exists to make.
>
> **What the instrument already shows:** Q8_0 is indistinguishable from the reference on
> every measure (KL 0.0016 nats/token, all downstream tasks tied). Q4_K_M is 39% faster and
> 38% smaller at batch 1, at a cost of KL 0.05, a changed top token at ~9% of positions, and
> statistically real but small losses on GSM8K (−2.0 points) and HellaSwag (−0.5). Details,
> uncertainties and caveats: [docs/findings.md](docs/findings.md).
>
> Host caveats (native Windows): other processes hold ~2–5 GiB of the card, and the driver's
> sysmem fallback spills allocations to system RAM instead of OOMing. VRAM is attributed
> per process from Windows' GPU counters, and a spilled run is recorded as OOM; see
> [methodology](docs/methodology.md#vram-accounting-modes) and [failures](docs/failures.md).

---

## The hero chart

![Headline Pareto: KL divergence vs decode throughput at batch 1](figures/01-headline-pareto.png)

_Figure 1 — quality cost (KL divergence vs the FP16 reference, log x, lower is better)
against decode throughput at batch 1. Marker size ∝ peak VRAM, colour = engine, shape =
format family, dashed line = Pareto frontier. **Not yet generated: no runs have been
executed.**_

## Recommendation table

<!-- BEGIN quantbench:recommendations -->
| KL budget (nats/token) | use | decode tok/s (p50, b1) | peak VRAM, device (MiB) | peak VRAM, this run (MiB) | tied configs |
| --- | --- | --- | --- | --- | --- |
| 0.010 | gguf:Q8_0 (llama.cpp) | 143.545 | 4,493.602 | 4,493.602 | 1 |
| 0.050 | gguf:Q8_0 (llama.cpp) | 143.545 | 4,493.602 | 4,493.602 | 1 |
| 0.100 | gguf:Q4_K_M (llama.cpp) | 199.428 | 2,791.586 | 2,791.586 | 1 |

**Chosen from 2 candidate configuration(s) only** (gguf:Q4_K_M (llama.cpp), gguf:Q8_0 (llama.cpp); model: Qwen3-4B; KL estimator: llama-perplexity). A row is the fastest *measured* configuration within its KL budget, not a claim about formats that have not run. Configurations whose IQRs overlap are reported as tied rather than ranked (§6.9). Peak VRAM is per-process (Windows GPU Process Memory counters), the figure that decides whether it fits.
<!-- END quantbench:recommendations -->

## Hypotheses

Stated up front, answered with data. Confirming, refuting, or being unable to resolve any
of them is an acceptable outcome.

- **H1** — Q5_K_M is a strictly better quality-per-VRAM tradeoff than Q4_K_M for models
  ≤ 8B, but the ordering inverts once KV-cache pressure at long context dominates.
- **H2** — Perplexity delta materially understates quantization damage compared to KL
  divergence against the FP16 reference; two quants with near-identical PPL can differ
  substantially in top-1 token agreement.
- **H3** — On `sm_120`, INT4 formats with Marlin-class kernels (AWQ/GPTQ via vLLM) beat
  GGUF Q4_K_M on batched throughput, but GGUF wins or ties at batch size 1 — the regime
  most local users actually run.
- **H4** — KV-cache quantization buys more usable context per GB than weight quantization
  does, per unit of quality lost.

<!-- BEGIN quantbench:hypotheses -->
| hypothesis | statement | evidence | verdict |
| --- | --- | --- | --- |
| H1 | Q5_K_M beats Q4_K_M on quality-per-VRAM below 8B, inverting under KV pressure | insufficient data: needs quality runs for gguf:Q5_K_M | — |
| H2 | PPL delta understates quantization damage relative to KL divergence | preliminary (n=2 configurations per KL estimator) — see docs/findings.md | — |
| H3 | INT4 + Marlin-class kernels beat GGUF Q4_K_M batched, but not at batch 1 | insufficient data: needs AWQ/GPTQ latency runs (M4) | — |
| H4 | KV-cache quantization buys more usable context per GB than weight quantization | insufficient data: needs a KV-cache dtype sweep (core profile) | — |
<!-- END quantbench:hypotheses -->

## What ran, what didn't

<!-- BEGIN quantbench:coverage -->
| model | format_variant | ok |
| --- | --- | --- |
| Qwen3-4B | bf16:reference | 2 |
| Qwen3-4B | gguf:F16 | 1 |
| Qwen3-4B | gguf:Q4_K_M | 4 |
| Qwen3-4B | gguf:Q8_0 | 4 |
<!-- END quantbench:coverage -->

Build failures and unsupported paths are published in [docs/failures.md](docs/failures.md)
with exact versions and errors. That file is a credibility asset, not an embarrassment.

## Downstream accuracy (lm-eval)

<!-- BEGIN quantbench:accuracy -->
| model | task | n | reference | gguf:Q4_K_M | gguf:Q8_0 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | arc_challenge | 1172 | 53.9 | 52.6 (97.6%; Δ -1.3 [-2.8, +0.3], tied) | 53.8 (99.8%; Δ -0.1 [-0.5, +0.4], tied) |
| Qwen3-4B | hellaswag | 10042 | 68.4 | 67.9 (99.2%; Δ -0.5 [-0.9, -0.1], **worse**) | 68.4 (100.0%; Δ +0.0 [-0.1, +0.1], tied) |
| Qwen3-4B | gsm8k | 1319 | 86.2 | 84.2 (97.7%; Δ -2.0 [-3.6, -0.3], **worse**) | 86.3 (100.1%; Δ +0.1 [-0.7, +0.8], tied) |
| Qwen3-4B | mmlu_subset | 2175 | 60.8 | 59.4 (97.7%; Δ -1.4 [-3.4, +0.6], tied) | 61.1 (100.5%; Δ +0.3 [-0.6, +1.1], tied) |
| Qwen3-4B | mmlu_abstract_algebra | 100 | 59.0 | 56.0 (94.9%; Δ -3.0 [-10.1, +4.1], tied) | 60.0 (101.7%; Δ +1.0 [-2.4, +4.4], tied) |
| Qwen3-4B | mmlu_college_computer_science | 100 | 68.0 | 63.0 (92.6%; Δ -5.0 [-10.1, +0.1], tied) | 67.0 (98.5%; Δ -1.0 [-3.0, +1.0], tied) |
| Qwen3-4B | mmlu_high_school_mathematics | 270 | 48.9 | 49.6 (101.5%; Δ +0.7 [-3.1, +4.6], tied) | 49.6 (101.5%; Δ +0.7 [-0.3, +1.8], tied) |
| Qwen3-4B | mmlu_professional_law | 1534 | 47.8 | 47.0 (98.2%; Δ -0.8 [-2.4, +0.7], tied) | 48.0 (100.3%; Δ +0.1 [-0.2, +0.4], tied) |
| Qwen3-4B | mmlu_world_religions | 171 | 80.1 | 81.3 (101.5%; Δ +1.2 [-1.1, +3.5], tied) | 80.7 (100.7%; Δ +0.6 [-0.6, +1.7], tied) |

Scores in %. Each cell: score (retention vs reference; paired Δ in points with its 95% interval over the *same* items; verdict). "tied" means the interval contains zero (§6.9). Pairing is per item on byte-identical prompts (lm-eval `prompt_hash`), which is why the intervals are much tighter than the published per-score standard errors. MMLU subset = unweighted mean of the five pinned subjects. GGUF quants are scored dequantized through Transformers, in fp16 like the reference (open question Q15).
<!-- END quantbench:accuracy -->

## Latency, memory and power measurements

<!-- BEGIN quantbench:latency -->
| model | format | engine | batch | prompt | gen | decode tok/s (p50) | IQR | TTFT ms (p50) | ITL ms (p50) | ITL ms (p99) | peak VRAM MiB (run) | eff. bpw | tok/J | power-capped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B | gguf:Q4_K_M | llama.cpp | 1 | 512 | 128 | 199.43 | 3.94 | 68.32 | 4.85 | 6.33 | 2,791.59 | 4.78 | 0.86 | 0% |
| Qwen3-4B | gguf:Q8_0 | llama.cpp | 1 | 512 | 128 | 143.55 | 1.22 | 60.65 | 6.84 | 7.98 | 4,493.60 | 8.50 | 0.64 | 17% |

Medians over measured repetitions; IQR is of per-request decode rate. Rows whose IQRs overlap are statistically indistinguishable (§6.9). Peak VRAM is attributable to the run; see the VRAM note in docs/methodology.md for how that is defined per host.
<!-- END quantbench:latency -->

## Measurement environment

<!-- BEGIN quantbench:environment -->
| gpu_name | driver | engine | engine_version | is_wsl | vram_accounting |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 5070 Ti | 617.14 | transformers | 5.20.1 | False | per_process_wddm |
| NVIDIA GeForce RTX 5070 Ti | 617.14 | transformers | 5.20.1 | False | device_wide |
| NVIDIA GeForce RTX 5070 Ti | 617.14 | llama.cpp | b11706 | False | device_wide |
| NVIDIA GeForce RTX 5070 Ti | 617.14 | llama.cpp | b11706 | False | per_process_wddm |
<!-- END quantbench:environment -->

---

## Reproduce it

Without a GPU — regenerate every figure from the committed results:

```bash
uv sync --extra dev
uv run quantbench plot --results results/results.parquet --out figures/
```

With a 16 GB Blackwell card:

```bash
uv run quantbench preflight                 # asserts sm_120 is in torch's arch list
uv run quantbench models fetch --config configs/models/qwen3-8b.yaml
uv run quantbench quantize --model qwen3-8b --formats gguf:Q4_K_M,gguf:Q5_K_M,awq,exl3:4.0
uv run quantbench reference build --model qwen3-8b
uv run quantbench plan --profile core --out plan.core.json
uv run quantbench run --plan plan.core.json --resume
uv run quantbench analyze && uv run quantbench plot && uv run quantbench report
```

`make core` chains plan → run → analyze → plot → report. Engine environments install
separately (`requirements/{hf,vllm,exl3}.txt`) because their torch pins conflict; see
[docs/install/](docs/install/).

Dry-run the whole pipeline on a machine with no GPU at all:

```bash
uv run quantbench plan --profile smoke --out plan.smoke.json
uv run quantbench run --plan plan.smoke.json --dry-run
```

Dry-run output is produced by a mock engine, marked `synthetic: true`, written to
`results/dryrun/`, and excluded from analysis by default. It exercises the schema and the
pipeline; it says nothing about hardware.

## How the numbers are made

The full protocol is in [docs/methodology.md](docs/methodology.md). The parts that matter
most:

- **KL divergence against the FP16 reference is the primary quality metric**, not
  perplexity. A quant can post *lower* PPL than its own FP16 parent while being
  demonstrably less faithful.
- **Per-token arrival timestamps** are recorded for every request, so ITL p90/p95/p99 are
  measured rather than inferred. Tail latency is invisible in a mean.
- **Four separate VRAM numbers** — weights, KV cache, allocator peak, and process-level
  NVML peak. Only the last one decides whether a configuration actually fits.
- **Randomized, interleaved run order** with a recorded seed, plus a cooldown gate, so GPU
  temperature drift is not confounded with format.
- **One subprocess per run**: an OOM or a segfault costs one data point, not the session,
  and `OOM` is a recorded outcome rather than an exception.
- **Statistical honesty**: medians and IQRs. Configurations whose IQRs overlap are reported
  as tied, not ranked.

## Non-goals

Absence here is deliberate, not oversight:

- Training, fine-tuning, LoRA (NF4 appears only as an *inference* baseline).
- Multi-GPU, tensor parallelism, distributed serving.
- CPU-only or Apple Silicon inference.
- Subjective "vibe" evaluation or LLM-as-judge scoring.
- Production serving concerns: autoscaling, routing, auth.
- Beating published SOTA numbers. The contribution is methodology plus a reproducible
  single-card matrix.
- Datacenter Blackwell (`sm_100`). Findings are scoped to consumer GB20x.

## Limitations

- Single GPU, single sample of one card: silicon lottery and cooling are uncontrolled.
- Engine version churn — results are a dated snapshot with every version pinned per record.
- Cross-engine perplexity is not directly comparable; quality metrics are computed in one
  harness over one token stream wherever possible, and labelled `scored_by_engine` where
  not.
- If measurements are taken under WSL2, batch-1 latency is noisier than native Linux. The
  `is_wsl` flag is recorded in every run record and surfaces in the environment table.

## Layout

```
configs/    hardware, models, formats, run profiles, eval definitions
src/        the harness (schema, planner, runner, engines, metrics, analysis)
results/    raw/<run_id>.json (never edited) + results.parquet
figures/    committed SVG + PNG, regenerable without a GPU
docs/       methodology, findings, failures, open questions, install recipes
```

## License

MIT. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
