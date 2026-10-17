"""Structured logging: a rich console for humans, a JSONL file per session for machines."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from quantbench.paths import ensure_dirs, logs_dir

_console: Console | None = None
_session_id: str | None = None


def console() -> Console:
    global _console
    if _console is None:
        _console = Console(stderr=False, highlight=False, soft_wrap=False)
    return _console


def session_id() -> str:
    global _session_id
    if _session_id is None:
        _session_id = (
            os.environ.get("QUANTBENCH_SESSION")
            or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        )
    return _session_id


class JsonlHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        ensure_dirs(path.parent)
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload |= extra
        if record.exc_info:
            payload["exc"] = self.format(record)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")


def setup_logging(verbose: bool = False, session_log: bool = True) -> logging.Logger:
    root = logging.getLogger("quantbench")
    if root.handlers:
        return root
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(
        RichHandler(console=console(), show_time=True, show_path=False, rich_tracebacks=True)
    )
    if session_log:
        root.addHandler(JsonlHandler(logs_dir() / f"session-{session_id()}.jsonl"))
    return root


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
