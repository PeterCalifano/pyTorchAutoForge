from __future__ import annotations

from functools import lru_cache

import pytest
import torch


@lru_cache(maxsize=1)
def CudaIsUsable() -> bool:
    """Return True only when CUDA kernels can execute, not just be discovered."""
    if not torch.cuda.is_available():
        return False

    try:
        device_ = torch.device("cuda")
        tensor_ = torch.ones(1, device=device_)
        result_ = (tensor_ + 1.0).item()
        torch.cuda.synchronize(device_)
        return result_ == 2.0
    except Exception:
        return False


def RequireCudaUsable() -> None:
    if not CudaIsUsable():
        pytest.skip("CUDA runtime is not usable in this environment.")
