"""Filesystem layout. Every path in the repo resolves through here."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root, overridable with ``QUANTBENCH_ROOT`` (tests use this)."""
    env = os.environ.get("QUANTBENCH_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def results_dir() -> Path:
    return Path(os.environ.get("QUANTBENCH_RESULTS", repo_root() / "results"))


def raw_dir() -> Path:
    return results_dir() / "raw"


def dryrun_dir() -> Path:
    """Synthetic records live here and are never read by the analysis layer by default."""
    return results_dir() / "dryrun" / "raw"


def figures_dir() -> Path:
    return repo_root() / "figures"


def models_dir() -> Path:
    return Path(os.environ.get("QUANTBENCH_MODELS", repo_root() / "models"))


def reference_cache_dir() -> Path:
    return Path(os.environ.get("QUANTBENCH_REFCACHE", repo_root() / "reference_cache"))


def calib_dir() -> Path:
    return repo_root() / "calib"


def logs_dir() -> Path:
    return repo_root() / "logs"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
