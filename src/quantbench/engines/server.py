"""Managed inference-server subprocess + SSE streaming client.

Both llama.cpp and vLLM expose their maintained, well-defined surface as an HTTP server,
and streaming responses are the only way to obtain honest per-token arrival timestamps
without patching the engine (§7.2). This module is deliberately stdlib-only: adding an
HTTP dependency to measure HTTP latency would be its own confound, and §11.1 keeps the
dependency surface small.

Timestamping contract: the clock is ``time.perf_counter()`` in the worker process, and a
token's arrival time is taken immediately after its SSE frame has been fully read. The
resulting TTFT/ITL therefore include the local loopback and SSE-framing overhead. That
overhead is identical across server-based engines and is stated in ``docs/methodology.md``
rather than subtracted.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantbench.engines.base import EngineOOM, EngineUnsupported

OOM_MARKERS = (
    "out of memory",
    "cudamalloc failed",
    "cuda error: out of memory",
    "failed to allocate",
    "torch.outofmemoryerror",
    "no available memory for the cache blocks",
)

BUILD_MARKERS = (
    "no kernel image is available",
    "not compatible with the current pytorch installation",
    "undefined symbol",
    "sm_120",
)


def find_binary(name: str, env_var: str | None = None) -> str:
    """Locate an engine binary, preferring an explicit environment override."""
    if env_var and (override := os.environ.get(env_var)):
        if Path(override).exists():
            return override
        raise EngineUnsupported(f"{env_var}={override!r} does not exist")
    found = shutil.which(name)
    if not found:
        raise EngineUnsupported(
            f"binary {name!r} not found on PATH; see docs/install/ for the pinned build"
        )
    return found


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class ManagedServer:
    """A subprocess that serves an HTTP inference API on localhost."""

    cmd: list[str]
    port: int
    health_path: str = "/health"
    startup_timeout_s: float = 600.0
    env: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None

    proc: subprocess.Popen[bytes] | None = None
    _log_fh: Any = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def start(self) -> None:
        environ = os.environ | self.env
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("wb")
            stdout: Any = self._log_fh
        else:
            stdout = subprocess.PIPE
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=environ,
        )

    def wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                self.classify_exit()
                raise EngineUnsupported(
                    f"server exited during startup (rc={self.proc.returncode}); {self.tail()}"
                )
            try:
                with urllib.request.urlopen(self.base_url + self.health_path, timeout=2.0) as resp:
                    if 200 <= resp.status < 300:
                        return
            except (urllib.error.URLError, OSError) as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self.stop()
        raise EngineUnsupported(
            f"server did not become healthy within {self.startup_timeout_s:.0f}s: {last_error}"
        )

    def classify_exit(self) -> None:
        """Turn a dead server into the most specific exception we can justify."""
        text = self.tail(8000).lower()
        if any(m in text for m in OOM_MARKERS):
            raise EngineOOM(self.tail(2000))
        if any(m in text for m in BUILD_MARKERS):
            raise EngineUnsupported(self.tail(2000))

    def tail(self, n_chars: int = 4000) -> str:
        if self.log_path is not None and self.log_path.exists():
            data = self.log_path.read_bytes()[-n_chars:]
            return data.decode("utf-8", errors="replace")
        if self.proc is not None and self.proc.stdout is not None:
            try:
                return self.proc.stdout.read(n_chars).decode("utf-8", errors="replace")
            except Exception:
                return ""
        return ""

    def stop(self, grace_s: float = 20.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.terminate()
                else:
                    self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
                with suppress(subprocess.TimeoutExpired):  # pragma: no cover
                    self.proc.wait(timeout=grace_s)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self.proc = None

    def __enter__(self) -> ManagedServer:
        self.start()
        self.wait_healthy()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


@dataclass
class StreamedToken:
    t_ms: float
    text: str
    n_tokens: int = 1
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stream_sse(
    url: str,
    payload: dict[str, Any],
    t0: float,
    timeout: float = 600.0,
    headers: dict[str, str] | None = None,
) -> Iterator[StreamedToken]:
    """POST ``payload`` and yield one :class:`StreamedToken` per SSE frame.

    ``t0`` is the ``time.perf_counter()`` origin so timestamps share the worker's clock.
    Each frame is timestamped the instant its line has been read, before parsing.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"}
        | (headers or {}),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            arrival_ms = (time.perf_counter() - t0) * 1000.0
            if not line:
                return
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded or decoded.startswith(":"):
                continue
            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()
            if decoded == "[DONE]":
                return
            try:
                frame = json.loads(decoded)
            except json.JSONDecodeError:
                continue
            yield _to_token(frame, arrival_ms)


def _to_token(frame: dict[str, Any], arrival_ms: float) -> StreamedToken:
    """Normalise llama.cpp and OpenAI-compatible frames into one shape."""
    text = ""
    finish: str | None = None
    if "choices" in frame:  # OpenAI-compatible (vLLM)
        choice = (frame.get("choices") or [{}])[0]
        text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
        finish = choice.get("finish_reason")
    else:  # llama.cpp native /completion
        text = frame.get("content", "")
        if frame.get("stop"):
            finish = str(frame.get("stop_type") or "stop")
    return StreamedToken(t_ms=arrival_ms, text=text, finish_reason=finish, raw=frame)
