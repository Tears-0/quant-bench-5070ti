"""Engine adapters.

Nothing here imports torch, vLLM, or ExLlamaV3 at module scope. ``from quantbench.engines
import base`` must succeed on a machine with no GPU and no CUDA stack.
"""

from quantbench.engines.base import (
    BaseEngine,
    Engine,
    EngineError,
    EngineMetadata,
    EngineOOM,
    EngineUnsupported,
    GenReport,
    LoadReport,
    PreflightResult,
    ScoreReport,
    SequenceTiming,
    load_engine,
)

__all__ = [
    "BaseEngine",
    "Engine",
    "EngineError",
    "EngineMetadata",
    "EngineOOM",
    "EngineUnsupported",
    "GenReport",
    "LoadReport",
    "PreflightResult",
    "ScoreReport",
    "SequenceTiming",
    "load_engine",
]
