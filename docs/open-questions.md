# Open questions

Decisions that were genuinely ambiguous. Each records the options, the conservative default
that is in force, and what would change it. Answers get written here as they are decided
(requirements.md §0.7, §13).

---

## Q1 — Model family: Qwen3 throughout, or mix in Llama 3.x?

**Default in force:** Qwen3 primary across Tiers A/B/C; one Llama-3.2-3B-Instruct Tier-A run
as an architecture-diversity sanity check, configured in the `full` profile only.

Rationale: keeping the ladder within one family means tier-to-tier differences are about
size, not architecture. The Llama run exists to check that findings are not Qwen-specific;
if it disagrees materially, that becomes a headline caveat rather than a footnote.

**Status:** open — needs a human decision on whether the Llama run is worth the download and
conversion time for the `core` profile.

---

## Q2 — Produce quantized artifacts locally, or pull from publishers?

**Default in force:** produce locally wherever the tooling exists (GGUF via
`llama-quantize`, AWQ/FP8 via `llm-compressor`, GPTQ via `GPTQModel`, EXL3 via
`convert.py`). Provenance is recorded either way in a `.manifest.json` beside the artifact.

Rationale: the imatrix ablation is only meaningful if both arms come from the same F16
source with the same tool version. Publisher GGUFs frequently carry undocumented imatrix
provenance, which is precisely the confound being isolated.

**Status:** default stands. Any downloaded artifact must set `publisher` and
`publisher_revision` in its manifest, leaving `producer` null.

---

## Q3 — Native Linux or WSL2 as the primary measurement environment?

**Default in force:** prefer native Linux. If WSL2 is used, it is flagged in every record
(`is_wsl`), surfaced in the README environment table, and stated as a limitation.

Note for this machine specifically: the target card is installed in a Windows 11 host. The
harness runs on Windows (the CLI, planner, analysis and plotting are all platform-neutral
and the test suite passes there), but §3.1 prefers Linux for measurement, and per-process
NVML VRAM accounting — the number that decides whether a config *fits* — is unavailable
under WSL2, falling back to device-wide usage.

**Decided 2026-10-17: (c) native Windows**, by the repo owner. Consequences, all recorded
per run and stated in the methodology:

* NVML has **no** per-process accounting under WDDM (contrary to the hope above), so VRAM is
  `vram_accounting: "device_wide"`; `vram_peak_attributed_mb` subtracts the pre-load
  baseline and llama.cpp weights/KV come from the engine's own buffer reports.
* Other processes held 4.6–4.9 GiB of the card during M1, so the practical ceiling on this
  host is ~11.3 GiB. Tier C fit/no-fit results are host-specific and figure 4 draws both the
  card total and the host-available line.
* vLLM has no supported native-Windows build; when M4 reaches it, §12.1 (or WSL2 for that
  engine only, flagged) applies.

---

## Q4 — Is clock locking available?

**Default in force:** off. `nvidia-smi --lock-gpu-clocks` is attempted only with
`--lock-clocks`, and whether it succeeded is recorded per run (`clocks_locked`).

Consumer drivers often refuse clock locking. A locked-clock run and a boost-clock run are
not comparable, so the flag has to travel with the data either way.

**Status:** open — needs one experiment on the target machine.

---

## Q5 — Disk budget for the model cache

**Default in force:** preflight warns below 300 GB free. `quantbench models prune` lists
artifacts and only deletes with `--yes`; nothing is ever auto-deleted.

The `full` profile across three tiers and ten-plus formats runs to hundreds of GB. How much
of `full` is feasible is a function of this number.

**Status:** open.

---

## Q6 — Publication target

**Default in force:** repo first. `docs/findings.md` is written so it can become a post
with minimal editing.

---

## Q7 — Long-context retrieval probe

**Default in force:** stretch goal after M5. `configs/eval/longctx.yaml` exists but is not
referenced by the `core` profile, so the README cannot imply long-context behaviour was
measured when it was not.

---

## Q8 — Engine dependency sets are requirements files, not extras

**Deviation from §11.2, decided 2026-08-15.**

§11.2 asks for `uv` extras per engine (`.[llamacpp]`, `.[vllm]`, `.[exl3]`). A single uv
lockfile resolves every extra together, and resolving vLLM requires building it, which
fails on any host without `CUDA_HOME` — making the CLI uninstallable on exactly the machine
that needs `quantbench preflight` to diagnose the problem (see `docs/failures.md`).

**Default in force:** `requirements/{hf,vllm,exl3}.txt`, one isolated environment per
engine, which also satisfies the underlying reason §11.2 gives (conflicting torch pins).
llama.cpp keeps no Python surface at all. Revisit if uv gains per-extra lock isolation.

---

## Q9 — Other structural deviations from §7.1

Small additions to the prescribed layout, listed so a reviewer can see they were deliberate:

| Path | Why |
|---|---|
| `src/quantbench/worker.py` | The per-run subprocess entry point named in §7.3. |
| `src/quantbench/config.py` | Pydantic models for the YAML configs the planner consumes. |
| `src/quantbench/paths.py`, `logging_setup.py`, `prompts.py` | Filesystem layout, JSONL session logging (§11.3), deterministic prompt construction. |
| `src/quantbench/engines/mock.py` | The `MockEngine` §11.3 requires for GPU-less CI. |
| `src/quantbench/engines/server.py` | Shared managed-server + SSE client used by the llama.cpp and vLLM adapters. |
| `src/quantbench/reference/eval_stream.py` | Builds and hashes the fixed evaluation token stream (§6.6); `logits_cache.py` consumes it. |
| `src/quantbench/analysis/report.py` | Backs `quantbench report` (§9.1). |
| `src/quantbench/quantize/base.py` | Shared provenance/manifest plumbing for the four producers. |
| `requirements/*.txt` | See Q8. |

---

## Q10 — `run_id` has no repetition suffix

§8.1's example ends `__r3`. In this implementation one run record aggregates
`protocol.measured_iters` repetitions and retains their per-sample values, so there is no
per-repetition record to name. Repetition count is in the record (`measured_iters`), and
the raw samples are in `latency.*.samples` and `latency.per_token_timestamps_ms`.

**Default in force:** no `__rN` suffix. Re-running the same spec later produces a second
record distinguished by its spec hash suffix, not by a repetition index.

---

## Q11 — `datasets` as an optional dependency

The evaluation corpus defaults to `source_kind: local_file`, which needs nothing beyond
§11.1. Loading corpora or calibration sets straight from the Hub (`hf_dataset`, and the
AWQ/GPTQ calibration paths) imports `datasets`, which is listed in `requirements/*.txt` but
is **not** in the approved core dependency set.

**Default in force:** local files for evaluation; `datasets` only inside producer/engine
environments. Flagged here rather than silently added to the core (§0.3).

---

## Q12 — `effective_bpw` excludes embedding *bytes*, not only embedding *parameters*

**Deviation from the literal §5.3 formula, decided 2026-10-17.**

§5.3 writes `effective_bpw = (weight_file_bytes × 8) / non_embedding_param_count` and asks
for "the embedding-excluded figure … because embedding quantization choices otherwise
distort format comparisons". Dividing the *whole file* by the *non-embedding* count leaves
the embedding's quant type in the numerator, which is the very distortion the section names.
Measured on Qwen3-4B Q4_K_M (Q6_K embedding): literal formula 5.50, embedding-excluded 4.78,
all-parameter 4.95 (matching llama.cpp's own "4.95 BPW").

**Default in force:** `(non-embedding tensor bytes × 8) / non-embedding params`, with tensor
bytes read from the GGUF / safetensors headers (`src/quantbench/checkpoint.py`). Validated
against three anchors: F16 → 16.0009, Q8_0 → 8.5013 (exactly 8.5 by construction), and the
Q4_K_M embedding alone → 6.5625 (exactly Q6_K). Records also keep `weight_file_bytes`, so the
literal figure can be recomputed by anyone who prefers it.

---

## Q13 — Is `sw_power_cap` contamination?

**Decided 2026-10-17 (conservative-but-usable default).** On this card NVML asserts
`sw_power_cap` under ordinary sustained load, well below the 300 W limit (0–29% of each M1
window). It is GPU Boost regulating clocks, i.e. the card's normal operating regime, and it
does not drift with session temperature the way thermal slowdown does.

**Default in force:** thermal/hardware slowdown reasons are contamination (`throttled`, and
`status: throttled_invalid` above 25% of the window). `sw_power_cap` is recorded separately
as `power_capped` / `power_capped_fraction`, is shown in the README table, and is *not*
excluded by default. Revisit if power-capped fraction turns out to correlate with format —
that would make it a confound, and the analysis can exclude on it.

**Status:** open for the owner to overrule.
