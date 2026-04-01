from __future__ import annotations

import importlib
import platform
from typing import Any, Dict

import torch


def collect_env_info() -> Dict[str, Any]:
    packages = {}
    for name in ["torch", "numpy", "scipy", "nibabel", "yaml"]:
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "not-installed"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
    }
