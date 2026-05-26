# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Device selection and MPS compatibility helpers.

Centralizes the cuda/mps/cpu selection logic and provides shims for ops that
either don't exist on MPS (bfloat16 autocast, pin_memory, FlashAttn, NCCL) or
exist but with different semantics. On MPS we deliberately fall back to fp32 —
MPS bfloat16 support is incomplete and several attention kernels silently
produce wrong results in bf16.
"""

import contextlib
import warnings
from typing import Optional, Union

import torch

_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned_once:
        return
    _warned_once.add(key)
    warnings.warn(message, category=UserWarning, stacklevel=2)


def get_default_device() -> torch.device:
    """Pick the best available device. Priority: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Normalize a user-supplied device argument; ``None`` picks the default."""
    if device is None:
        return get_default_device()
    return torch.device(device)


def is_cuda(device: Union[str, torch.device]) -> bool:
    return torch.device(device).type == "cuda"


def is_mps(device: Union[str, torch.device]) -> bool:
    return torch.device(device).type == "mps"


def autocast_context(
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.bfloat16,
    enabled: bool = True,
):
    """Return an autocast context appropriate for ``device``.

    On CUDA: real ``torch.autocast`` with the requested dtype.
    On MPS / CPU: nullcontext (fp32). MPS bf16 is too unreliable for SAM3 to
    use by default. See module docstring.
    """
    dev = torch.device(device)
    if not enabled:
        return contextlib.nullcontext()
    if dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if dev.type == "mps":
        _warn_once(
            "mps-autocast",
            "MPS detected: skipping bfloat16 autocast and running inference in fp32. "
            "This is slower than CUDA bf16 but avoids known MPS numerical issues.",
        )
    return contextlib.nullcontext()


def maybe_pin_memory(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
    """``tensor.pin_memory()`` is a CUDA-only optimization. No-op elsewhere."""
    if is_cuda(device):
        return tensor.pin_memory()
    return tensor


def supports_flash_attn(device: Union[str, torch.device]) -> bool:
    """FlashAttention requires CUDA Ampere+. Not available on MPS or CPU."""
    if not is_cuda(device):
        return False
    try:
        return torch.cuda.get_device_properties(0).major >= 8
    except Exception:
        return False


def setup_tf32() -> None:
    """Enable TF32 on Ampere+ CUDA. No-op on MPS/CPU."""
    if not torch.cuda.is_available():
        return
    try:
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def cuda_only_autocast(dtype: torch.dtype = torch.bfloat16):
    """Decorator: applies ``torch.autocast(device_type='cuda', dtype=...)`` to a
    method when CUDA is available at import time; otherwise it's a no-op pass-through.

    Use this on methods that originally had ``@torch.autocast(device_type="cuda", ...)``
    and that we want to run as plain fp32 on MPS / CPU.
    """
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _identity(fn):
        return fn

    return _identity


def warn_mps_fallback(component: str, reason: str) -> None:
    """One-time warning that a CUDA-only optimization is disabled on MPS."""
    _warn_once(
        f"mps-fallback-{component}",
        f"MPS: {component} disabled ({reason}). Inference will use a slower fallback.",
    )
