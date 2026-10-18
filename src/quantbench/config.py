"""YAML configuration objects.

Configs are data, not code: the planner turns (hardware, models, formats, profile) into a
list of :class:`~quantbench.schema.RunSpec`. Everything validated here fails loudly at
``quantbench plan`` time rather than six hours into an overnight run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantbench.paths import configs_dir
from quantbench.schema import WorkloadKind


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------------------


class ThermalConfig(ConfigModel):
    cooldown_target_c: int | None = 50
    cooldown_timeout_s: int = 300
    mark_throttled_invalid: bool = True


class HardwareConfig(ConfigModel):
    name: str
    gpu_name_match: str
    compute_capability: str
    required_torch_arch: str
    vram_total_mb: int
    vram_usable_mb: int
    min_driver: str
    min_cuda_runtime: str
    min_free_disk_gb: int = 200
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    sample_hz: float = 10.0
    allow_clock_lock: bool = False
    locked_sm_clock_mhz: int | None = None


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


class ModelConfig(ConfigModel):
    name: str
    slug: str
    tier: Literal["A", "B", "C", "D"]
    hf_repo: str
    revision: str | None = None
    params_total: int | None = None
    params_non_embedding: int | None = None
    tokenizer_sha256: str | None = None
    reference_engine: str = "transformers"
    reference_strategy: Literal["live", "cached"] = "cached"
    notes: str | None = None


# --------------------------------------------------------------------------------------
# formats
# --------------------------------------------------------------------------------------


class VariantConfig(ConfigModel):
    id: str
    bits_nominal: float | None = None
    imatrix: bool = False
    calibration_dataset: str | None = None
    producer_args: list[str] = Field(default_factory=list)
    notes: str | None = None


class FormatConfig(ConfigModel):
    family: str
    producer: str | None = None
    producer_module: str | None = None  # quantbench.quantize.<module>
    default_engines: list[str]
    supported_engines: list[str]
    scoring_engine: str | None = None  # §7.2 fallback when the engine cannot score
    kv_cache_dtypes: list[str] = Field(default_factory=lambda: ["auto"])
    attention_backends: list[str] = Field(default_factory=lambda: ["auto"])
    artifact_pattern: str | None = None
    variants: list[VariantConfig]
    conditional: bool = False  # verify support empirically before promising it (§3.2)
    notes: str | None = None

    def variant(self, variant_id: str) -> VariantConfig:
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise KeyError(f"format {self.family!r} has no variant {variant_id!r}")


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------


class PplConfigYaml(ConfigModel):
    ctx: int = 2048
    stride: int = 2048
    add_bos: bool = True
    max_tokens: int | None = None


class ReferenceConfig(ConfigModel):
    topk: int = 128
    store_dtype: Literal["float16", "float32"] = "float16"


class EvalConfig(ConfigModel):
    name: str
    kind: Literal["ppl_kl", "lm_eval", "longctx"] = "ppl_kl"
    source_kind: Literal["hf_dataset", "local_file"] = "local_file"
    source: str
    subset: str | None = None
    split: str | None = None
    ppl: PplConfigYaml = Field(default_factory=PplConfigYaml)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    lm_eval_tasks: list[dict[str, Any]] = Field(default_factory=list)
    #: llama-perplexity stores the *full-vocabulary* base distribution (~2 bytes x vocab per
    #: scored token: ~304 KB/token for Qwen3's 151,936-token vocabulary), so the native
    #: GGUF KL path scores a fixed, recorded number of chunks rather than the whole corpus.
    native_kld_chunks: int = 40
    #: Fixed, recorded batch size. lm-eval's 'auto' probes for the largest batch that
    #: does not OOM; with Windows sysmem fallback nothing OOMs, so 'auto' spills.
    lm_eval_batch_size: int = 1  # fallback only; tuned per task: quantbench lmeval-tune
    #: One compute dtype for the reference and every quant, so weights are the only
    #: variable. fp16 also holds dequantized GGUF values more exactly than bf16.
    lm_eval_dtype: str = "float16"
    calibration_disjoint_from: list[str] = Field(default_factory=list)
    notes: str | None = None


# --------------------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------------------


class WorkloadSweep(ConfigModel):
    kind: WorkloadKind
    batch_size: list[int] = Field(default_factory=lambda: [1])
    prompt_tokens: list[int] = Field(default_factory=lambda: [512])
    generate_tokens: list[int] = Field(default_factory=lambda: [128])
    ignore_eos: bool = True
    eval_config: str | None = None

    @model_validator(mode="after")
    def _quality_needs_eval(self) -> WorkloadSweep:
        if self.kind == "quality" and not self.eval_config:
            raise ValueError("quality workloads require an eval_config")
        return self


class ProtocolConfig(ConfigModel):
    warmup_iters: int = 2
    measured_iters: int = 5
    cooldown_target_c: int | None = 50
    cooldown_timeout_s: int = 300
    clocks_locked: bool = False
    timeout_s: int = 1800
    sample_hz: float = 10.0


class CellConfig(ConfigModel):
    """One rectangle of the matrix: models x formats x engines x workloads."""

    name: str
    models: list[str]
    formats: list[str]  # "family:variant"
    engines: list[str] | None = None
    kv_cache_dtypes: list[str] | None = None
    attention_backends: list[str] | None = None
    workloads: list[WorkloadSweep]
    protocol: ProtocolConfig | None = None
    notes: str | None = None


class ProfileConfig(ConfigModel):
    name: str
    description: str | None = None
    seed: int = 1337
    order: Literal["shuffle", "round_robin", "as_listed"] = "shuffle"
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    hardware: str = "5070ti"
    cells: list[CellConfig]


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _resolve(kind: str, name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    return configs_dir() / kind / f"{name_or_path}.yaml"


def load_hardware(name_or_path: str = "5070ti") -> HardwareConfig:
    return HardwareConfig.model_validate(_read_yaml(_resolve("hardware", name_or_path)))


def load_model(name_or_path: str) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(_resolve("models", name_or_path)))


def load_format(name_or_path: str) -> FormatConfig:
    return FormatConfig.model_validate(_read_yaml(_resolve("formats", name_or_path)))


def load_eval(name_or_path: str) -> EvalConfig:
    return EvalConfig.model_validate(_read_yaml(_resolve("eval", name_or_path)))


def load_profile(name_or_path: str) -> ProfileConfig:
    return ProfileConfig.model_validate(_read_yaml(_resolve("profiles", name_or_path)))


def list_configs(kind: str) -> list[str]:
    directory = configs_dir() / kind
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def parse_format_ref(ref: str) -> tuple[str, str]:
    """``"gguf:Q4_K_M"`` -> ``("gguf", "Q4_K_M")``."""
    if ":" not in ref:
        raise ValueError(f"format reference {ref!r} must be 'family:variant'")
    family, variant = ref.split(":", 1)
    return family.strip(), variant.strip()
