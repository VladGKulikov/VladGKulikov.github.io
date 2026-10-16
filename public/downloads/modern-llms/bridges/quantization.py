"""B19 — grouped fake quantization along the last tensor dimension."""
from __future__ import annotations

import math
import numpy as np
import torch


def numpy_grouped_fake_quant_explicit(
    x: np.ndarray,
    *,
    grid: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Minimal nearest-grid fake quantization along the last dimension."""
    pad = (-x.shape[-1]) % group_size
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    groups = padded.reshape(*x.shape[:-1], -1, group_size)
    max_abs = np.max(np.abs(groups), axis=-1, keepdims=True)
    scales = np.where(max_abs > 0, max_abs / np.max(np.abs(grid)), 1.0)
    normalized = groups / scales
    codes = np.abs(normalized[..., None] - grid).argmin(axis=-1)
    return (grid[codes] * scales).reshape(padded.shape)[..., : x.shape[-1]]


def torch_grouped_fake_quant_explicit(
    x: torch.Tensor,
    *,
    grid: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Minimal tensor counterpart; leading dimensions never share a scale."""
    grid = grid.to(device=x.device, dtype=x.dtype)
    pad = (-x.shape[-1]) % group_size
    padded = torch.nn.functional.pad(x, (0, pad)) if pad else x
    groups = padded.reshape(*x.shape[:-1], -1, group_size)
    max_abs = groups.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(max_abs > 0, max_abs / grid.abs().max(), torch.ones_like(max_abs))
    normalized = groups / scales
    codes = (normalized[..., None] - grid).abs().argmin(dim=-1)
    return (grid[codes] * scales).reshape(padded.shape)[..., : x.shape[-1]]


def numpy_grouped_fake_quant(
    x: np.ndarray,
    *,
    grid: np.ndarray,
    group_size: int,
) -> dict[str, np.ndarray | int]:
    """Nearest-grid fake quantization; leading dimensions never share a group."""
    raw_x = np.asarray(x)
    if not np.issubdtype(raw_x.dtype, np.floating):
        raise TypeError("x must be a floating-point array")
    x = raw_x.astype(np.float64, copy=False)
    grid = np.asarray(grid, dtype=np.float64)
    if isinstance(group_size, (bool, np.bool_)) or not isinstance(group_size, (int, np.integer)) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if x.ndim < 1 or x.size == 0 or grid.ndim != 1 or grid.size < 2:
        raise ValueError("x/grid/group_size have invalid shapes")
    if not np.isfinite(x).all() or not np.isfinite(grid).all() or not np.any(grid != 0):
        raise ValueError("x and grid must be finite and grid non-zero")
    if not np.any(grid == 0):
        raise ValueError("grid must contain an exact zero for zero-valued groups")
    pad = (-x.shape[-1]) % group_size
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    groups = padded.reshape(*x.shape[:-1], -1, group_size)
    max_grid = float(np.max(np.abs(grid)))
    max_abs = np.max(np.abs(groups), axis=-1, keepdims=True)
    scales = np.where(max_abs > 0, max_abs / max_grid, 1.0)
    normalized = groups / scales
    codes = np.abs(normalized[..., None] - grid).argmin(axis=-1)
    dequant = (grid[codes] * scales).reshape(padded.shape)[..., : x.shape[-1]]
    return {"codes": codes.reshape(padded.shape)[..., : x.shape[-1]], "scales": scales[..., 0], "dequantized": dequant, "padding": pad}


def torch_grouped_fake_quant(
    x: torch.Tensor,
    *,
    grid: torch.Tensor,
    group_size: int,
) -> dict[str, torch.Tensor | int]:
    """Torch equivalent of :func:`numpy_grouped_fake_quant`."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if x.ndim < 1 or x.numel() == 0 or grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("x/grid/group_size have invalid shapes")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    if not torch.isfinite(x).all() or not torch.isfinite(grid).all() or not torch.any(grid != 0):
        raise ValueError("x and grid must be finite and grid non-zero")
    if not torch.any(grid == 0):
        raise ValueError("grid must contain an exact zero for zero-valued groups")
    grid = grid.to(device=x.device, dtype=x.dtype)
    pad = (-x.shape[-1]) % group_size
    padded = torch.nn.functional.pad(x, (0, pad)) if pad else x
    groups = padded.reshape(*x.shape[:-1], -1, group_size)
    max_grid = grid.abs().max()
    max_abs = groups.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(max_abs > 0, max_abs / max_grid, torch.ones_like(max_abs))
    normalized = groups / scales
    codes = (normalized[..., None] - grid).abs().argmin(dim=-1)
    dequant = (grid[codes] * scales).reshape(padded.shape)[..., : x.shape[-1]]
    return {"codes": codes.reshape(padded.shape)[..., : x.shape[-1]], "scales": scales[..., 0], "dequantized": dequant, "padding": pad}
