"""B07/B08 — RMSNorm and SwiGLU reference/torch pairs."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def numpy_rms_norm(x: np.ndarray, weight: np.ndarray | None = None, *, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last dimension using float64 reference reductions."""
    x = np.asarray(x)
    if x.ndim == 0 or x.shape[-1] == 0 or not np.isfinite(x).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("x must be finite and eps positive")
    work = x.astype(np.float64, copy=False)
    y = work / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != (x.shape[-1],) or not np.isfinite(w).all():
            raise ValueError("weight must be finite with shape [D]")
        y = y * w
    return y.astype(x.dtype, copy=False) if np.issubdtype(x.dtype, np.floating) else y


def torch_rms_norm_explicit(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Visible tensor RMSNorm; reduction is written explicitly for teaching."""
    if not x.is_floating_point() or x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must be floating with a nonempty last axis")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    work = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    rms_inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    y = work * rms_inv
    if weight is not None:
        if weight.shape != (x.shape[-1],):
            raise ValueError("weight must have shape [D]")
        y = y * weight.to(work.dtype)
    return y.to(x.dtype)


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, *, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch RMSNorm bridge with native autograd and device handling."""
    return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)


def numpy_swiglu(
    gate: np.ndarray,
    up: np.ndarray,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> np.ndarray:
    """SwiGLU; optional limits and shift reproduce the gpt-oss-style teaching variant."""
    gate, up = np.asarray(gate), np.asarray(up)
    if gate.shape != up.shape or not np.isfinite(gate).all() or not np.isfinite(up).all():
        raise ValueError("gate and up must be finite arrays of the same shape")
    if not np.isfinite(alpha) or not np.isfinite(shift_up):
        raise ValueError("alpha and shift_up must be finite")
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (gate_limit, up_limit)):
        raise ValueError("limits must be finite and positive")
    g = gate.astype(np.float64, copy=False)
    u = up.astype(np.float64, copy=False)
    if gate_limit is not None:
        g = np.minimum(g, gate_limit)
    if up_limit is not None:
        u = np.clip(u, -up_limit, up_limit)
    sigmoid = np.exp(-np.logaddexp(0.0, -alpha * g))
    y = g * sigmoid * (u + shift_up)
    dtype = np.result_type(gate.dtype, up.dtype)
    return y.astype(dtype, copy=False) if np.issubdtype(dtype, np.floating) else y


def torch_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> torch.Tensor:
    """Tensor equivalent of :func:`numpy_swiglu`, preserving autograd."""
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if gate.dtype != up.dtype or not gate.is_floating_point() or gate.device != up.device:
        raise ValueError("gate and up must share a floating dtype and device")
    if not np.isfinite(alpha) or not np.isfinite(shift_up):
        raise ValueError("alpha and shift_up must be finite")
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (gate_limit, up_limit)):
        raise ValueError("limits must be finite and positive")
    dtype = torch.float32 if gate.dtype in (torch.float16, torch.bfloat16) else gate.dtype
    g, u = gate.to(dtype), up.to(dtype)
    if gate_limit is not None:
        g = g.clamp(max=gate_limit)
    if up_limit is not None:
        u = u.clamp(min=-up_limit, max=up_limit)
    return (g * torch.sigmoid(alpha * g) * (u + shift_up)).to(gate.dtype)
