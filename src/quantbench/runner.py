"""Subprocess orchestration (requirements.md §6.3, §6.4, §7.3).

The parent process never imports an engine. It:

* gates each run on a GPU cooldown target so thermal state does not drift into the results;
* spawns ``python -m quantbench.worker`` per run, so a segfault in vLLM costs one data
  point rather than the session;
* enforces a timeout and classifies the exit;
* writes a record **whatever happens** — timeout, crash, OOM and build failure are all
  recorded outcomes, never silent drops;
* skips runs already present for this environment, so an overnight run survives a crash at
  hour six.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from quantbench.logging_setup import log_event
from quantbench.metrics.memory import gpu_temperature_c
from quantbench.paths import dryrun_dir, ensure_dirs, raw_dir
from quantbench.planner import Plan, filter_resumable
from quantbench.preflight import capture_environment
from quantbench.schema import RunRecord, RunSpec, RunStatus, load_record

log = logging.getLogger("quantbench.runner")

OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")
BUILD_MARKERS = ("no kernel image is available", "undefined symbol", "importerror")


@dataclass
class RunnerOptions:
    dry_run: bool = False
    force: bool = False
    resume: bool = True
    cooldown: bool = True
    lock_clocks: bool = False
    locked_sm_clock_mhz: int | None = None
    keep_going: bool = True
    limit: int | None = None
    python: str = sys.executable


@dataclass
class RunnerResult:
    records: list[RunRecord] = field(default_factory=list)
    skipped: list[RunSpec] = field(default_factory=list)
    output_dir: Path = Path()

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for record in self.records:
            out[record.status.value] = out.get(record.status.value, 0) + 1
        return out


class Runner:
    def __init__(self, options: RunnerOptions | None = None) -> None:
        self.options = options or RunnerOptions()
        self.environment = capture_environment()
        self.environment_hash = self.environment.hash()

    # -- thermal / clocks ---------------------------------------------------------------

    def cooldown(self, target_c: int | None, timeout_s: int) -> float | None:
        """Wait until the GPU is at or below ``target_c`` (§6.3). Returns the actual temp."""
        temp = gpu_temperature_c()
        if not self.options.cooldown or target_c is None or temp is None:
            return temp
        deadline = time.monotonic() + timeout_s
        while temp is not None and temp > target_c and time.monotonic() < deadline:
            time.sleep(5.0)
            temp = gpu_temperature_c()
        return temp

    def set_clock_lock(self, enable: bool, mhz: int | None) -> bool:
        """Lock or reset GPU clocks. A locked-clock run and a boost-clock run are not
        comparable, so whether locking was active is recorded in the protocol block."""
        if not shutil.which("nvidia-smi"):
            return False
        cmd = (
            ["nvidia-smi", "--lock-gpu-clocks", str(mhz)]
            if enable and mhz
            else ["nvidia-smi", "--reset-gpu-clocks"]
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode != 0:
            log.warning("clock lock unavailable: %s", (proc.stderr or proc.stdout).strip()[:200])
            return False
        return True

    # -- execution ----------------------------------------------------------------------

    def run_plan(
        self,
        plan: Plan,
        on_record: Callable[[RunRecord], None] | None = None,
    ) -> RunnerResult:
        out_dir = dryrun_dir() if self.options.dry_run else raw_dir()
        ensure_dirs(out_dir)
        result = RunnerResult(output_dir=out_dir)

        entries = plan.entries
        if self.options.dry_run:
            entries = [
                spec.model_copy(update={"engine": spec.engine.model_copy(update={"name": "mock"})})
                for spec in entries
            ]
        if self.options.resume and not self.options.force:
            entries, skipped = filter_resumable(entries, out_dir, self.environment_hash)
            result.skipped = skipped
        if self.options.limit:
            entries = entries[: self.options.limit]

        clocks_locked = False
        if self.options.lock_clocks and not self.options.dry_run:
            clocks_locked = self.set_clock_lock(True, self.options.locked_sm_clock_mhz)

        try:
            for index, spec in enumerate(entries, start=1):
                pre_temp = self.cooldown(
                    spec.protocol.cooldown_target_c, spec.protocol.cooldown_timeout_s
                )
                effective = spec.model_copy(
                    update={
                        "protocol": spec.protocol.model_copy(
                            update={"clocks_locked": clocks_locked}
                        )
                    }
                )
                log_event(
                    log,
                    logging.INFO,
                    f"[{index}/{len(entries)}] {spec.run_id}",
                    run_id=spec.run_id,
                    pre_run_temp_c=pre_temp,
                )
                record = self.execute(effective)
                if pre_temp is not None and record.power.gpu_temp_start_c is None:
                    record.power.gpu_temp_start_c = pre_temp
                self.write_record(record, out_dir)
                result.records.append(record)
                if on_record is not None:
                    on_record(record)
                if record.status is not RunStatus.ok and not self.options.keep_going:
                    log.error("stopping after %s (keep_going=False)", record.status.value)
                    break
        finally:
            if clocks_locked:
                self.set_clock_lock(False, None)
        return result

    def execute(self, spec: RunSpec) -> RunRecord:
        """Spawn one worker and classify its outcome."""
        with tempfile.TemporaryDirectory(prefix="quantbench-") as tmp:
            tmp_path = Path(tmp)
            spec_path = tmp_path / "spec.json"
            out_path = tmp_path / "record.json"
            spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            cmd = [
                self.options.python,
                "-m",
                "quantbench.worker",
                "--spec",
                str(spec_path),
                "--out",
                str(out_path),
            ]
            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=spec.protocol.timeout_s,
                    check=False,
                )
                stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                stdout = _as_text(exc.stdout)
                stderr = _as_text(exc.stderr)
                returncode = -1
                timed_out = True
            duration = time.perf_counter() - started

            if out_path.exists():
                try:
                    record = load_record(json.loads(out_path.read_text(encoding="utf-8")))
                    if timed_out:
                        record.status = RunStatus.timeout
                        record.error = (record.error or "") + " (parent timeout)"
                    if record.stderr_tail is None and stderr:
                        record.stderr_tail = stderr[-4000:]
                    return record
                except Exception as exc:
                    stderr = f"{stderr}\nrecord validation failed: {exc}"

            return self._synthesize_failure(spec, stdout, stderr, returncode, timed_out, duration)

    def _synthesize_failure(
        self,
        spec: RunSpec,
        stdout: str,
        stderr: str,
        returncode: int,
        timed_out: bool,
        duration: float,
    ) -> RunRecord:
        """The worker died before it could write a record. Classify from its output."""
        blob = f"{stdout}\n{stderr}".lower()
        if timed_out:
            status = RunStatus.timeout
        elif any(m in blob for m in OOM_MARKERS):
            status = RunStatus.oom
        elif any(m in blob for m in BUILD_MARKERS):
            status = RunStatus.build_failed
        else:
            status = RunStatus.crashed
        return RunRecord(
            spec_hash=spec.hash(),
            environment_hash=self.environment_hash,
            spec=spec,
            status=status,
            synthetic=spec.engine.name == "mock",
            environment=self.environment,
            error=f"worker exited rc={returncode} without a valid record",
            stderr_tail=(stderr or stdout)[-8000:],
            duration_s=duration,
        )

    @staticmethod
    def write_record(record: RunRecord, out_dir: Path) -> Path:
        ensure_dirs(out_dir)
        path = out_dir / f"{record.spec.run_id}.json"
        if path.exists():
            path = out_dir / f"{record.spec.run_id}.{record.spec_hash[7:15]}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
