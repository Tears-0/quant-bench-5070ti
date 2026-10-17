"""Shared plumbing for the quantization pipeline (requirements.md §4.1, §6.7, §6.8).

This repo *produces* artifacts, it does not only consume them. Whichever path an artifact
takes, its provenance is recorded in a manifest next to it:

* who produced it (tool + version) and the **exact command line**;
* the calibration dataset, if any — and calibration data must be disjoint from the
  evaluation set (§6.7);
* whether an importance matrix was used, because imatrix presence is a confound that has
  misled published comparisons (§4.1);
* the SHA-256 of the resulting file(s), which is echoed into every run record.

Downloaded artifacts get the same manifest with ``publisher``/``publisher_revision`` set
and ``producer`` null, so "we made this" and "we trusted someone else" are distinguishable
in the data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class QuantizationError(RuntimeError):
    """A producer tool failed. The error text belongs in docs/failures.md verbatim."""


@dataclass
class QuantizeResult:
    artifact_path: Path
    producer: str
    producer_version: str | None
    command: list[str]
    duration_s: float
    sha256: str | None = None
    calibration_dataset: str | None = None
    imatrix: bool = False
    imatrix_source: str | None = None
    publisher: str | None = None
    publisher_revision: str | None = None
    log_tail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "sha256": self.sha256,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "command": self.command,
            "calibration_dataset": self.calibration_dataset,
            "imatrix": self.imatrix,
            "imatrix_source": self.imatrix_source,
            "publisher": self.publisher,
            "publisher_revision": self.publisher_revision,
            "duration_s": round(self.duration_s, 2),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.extra,
        }

    def write_manifest(self) -> Path:
        path = Path(str(self.artifact_path) + ".manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file (or of the sorted concatenation of a directory's weight files)."""
    digest = hashlib.sha256()
    files = (
        sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin")) if path.is_dir() else [path]
    )
    for file in files:
        digest.update(file.name.encode("utf-8"))
        with file.open("rb") as fh:
            while block := fh.read(chunk):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def run_command(cmd: list[str], timeout: int = 24 * 3600, cwd: Path | None = None) -> str:
    """Run a producer tool, raising with its output attached on failure."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=cwd,
        check=False,
    )
    blob = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise QuantizationError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{blob[-4000:]}"
        )
    return blob


def finalize(result: QuantizeResult) -> QuantizeResult:
    """Hash the artifact and drop a manifest beside it."""
    if result.artifact_path.exists():
        result.sha256 = sha256_file(result.artifact_path)
    result.write_manifest()
    return result
