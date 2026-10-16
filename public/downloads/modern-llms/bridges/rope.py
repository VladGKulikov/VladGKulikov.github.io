"""B02 — adjacent-pair Rotary Position Embedding in NumPy and PyTorch."""
from __future__ import annotations

import numpy as np
import torch


def _validate_last_dim(dim: int) -> None:
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE requires a positive even last dimension")


def numpy_rope_explicit(
    x: np.ndarray,
    position_ids: np.ndarray,
    *,
    base: float = 10_000.0,
) -> np.ndarray:
    """Adjacent pairs; valid floating x[..., T, D], integer IDs[T], base > 1."""
    dim = x.shape[-1]
    inv_freq = base ** (-np.arange(0, dim, 2) / dim)
    angles = position_ids[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + angles.shape
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.result_type(x.dtype, np.float64))
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def torch_rope_explicit(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Adjacent pairs with at least float32 phase arithmetic; output keeps x.dtype."""
    dim = x.shape[-1]
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv_freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=work_dtype) / dim)
    angles = position_ids.to(device=x.device, dtype=work_dtype)[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + tuple(angles.shape)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    even, odd = x[..., 0::2].to(work_dtype), x[..., 1::2].to(work_dtype)
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


def numpy_rope(x: np.ndarray, position_ids: np.ndarray, *, base: float = 10_000.0) -> np.ndarray:
    """Rotate adjacent coordinate pairs of ``x[..., T, D]`` at integer positions ``[T]``."""
    x = np.asarray(x)
    positions = np.asarray(position_ids)
    if x.ndim < 2 or positions.ndim != 1 or x.shape[-2] != positions.size:
        raise ValueError("expected x[..., T, D] and position_ids[T]")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError("position_ids must contain integers")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("x must be a floating-point array")
    _validate_last_dim(x.shape[-1])
    if not np.isfinite(x).all() or not np.isfinite(base) or base <= 1.0:
        raise ValueError("x and base must be finite; base must exceed one")

    work = x.astype(np.float64, copy=False)
    inv_freq = base ** (-np.arange(0, x.shape[-1], 2, dtype=np.float64) / x.shape[-1])
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    shape = (1,) * (x.ndim - 2) + cos.shape
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    even, odd = work[..., 0::2], work[..., 1::2]
    out = np.empty_like(work)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out.astype(x.dtype, copy=False) if np.issubdtype(x.dtype, np.floating) else out


def torch_rope(x: torch.Tensor, position_ids: torch.Tensor, *, base: float = 10_000.0) -> torch.Tensor:
    """PyTorch equivalent of :func:`numpy_rope`, preserving device and autograd."""
    if x.ndim < 2 or position_ids.ndim != 1 or x.shape[-2] != position_ids.numel():
        raise ValueError("expected x[..., T, D] and position_ids[T]")
    if position_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("position_ids must contain integers")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    _validate_last_dim(x.shape[-1])
    if not torch.isfinite(x).all() or not np.isfinite(base) or base <= 1.0:
        raise ValueError("x and base must be finite; base must exceed one")

    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv_freq = base ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device, dtype=calc_dtype) / x.shape[-1]
    )
    angles = position_ids.to(device=x.device, dtype=calc_dtype)[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + tuple(angles.shape)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    even, odd = x[..., 0::2].to(calc_dtype), x[..., 1::2].to(calc_dtype)
    out = torch.empty_like(x, dtype=calc_dtype)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out.to(dtype=x.dtype)
