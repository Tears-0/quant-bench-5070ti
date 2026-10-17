"""quant-bench-5070ti: consumer-GPU quantization & inference benchmark suite.

Nothing CUDA-related is imported here. The CLI must remain usable on a machine where
vLLM/ExLlamaV3/torch will not even import — that is precisely the machine on which you
need to run ``quantbench preflight`` (requirements.md §11.1).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
