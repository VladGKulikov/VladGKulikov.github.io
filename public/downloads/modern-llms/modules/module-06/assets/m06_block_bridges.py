"""Modern LLMs module 06: executable NumPy → PyTorch bridges (B07, B08).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
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


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((3, 5)); w_np = rng.standard_normal(5)
    np_out = numpy_rms_norm(x_np, w_np, eps=1e-6)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    explicit = torch_rms_norm_explicit(x, w, eps=1e-6)
    api = torch_rms_norm(x, w, eps=1e-6)
    np.testing.assert_allclose(np_out, explicit.detach().numpy(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: torch_rms_norm_explicit(a, b, eps=1e-6), (x, w))
    for a, b in zip(torch.autograd.grad(explicit.square().sum(), (x, w)),
                    torch.autograd.grad(api.square().sum(), (x, w))):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    for dtype in (torch.float16, torch.bfloat16):
        large = torch.full((2, 8), 1000.0, dtype=dtype)
        torch.testing.assert_close(torch_rms_norm_explicit(large), torch.ones_like(large))
    print("B07 explicit NumPy / PyTorch / F.rms_norm: PASS")

    rng = np.random.default_rng(8)
    g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
    kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
    np_out = numpy_swiglu(g_np, u_np, **kwargs)
    g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
    t_out = torch_swiglu(g, u, **kwargs)
    np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))
    extreme = np.array([-1000., -9., 0., 9., 1000.])
    with np.errstate(over='raise', invalid='raise'):
        stable = numpy_swiglu(extreme, np.ones_like(extreme), **kwargs)
    np.testing.assert_allclose(stable, torch_swiglu(torch.tensor(extreme), torch.ones(5, dtype=torch.float64), **kwargs))
    outside = torch.tensor([-9., 9.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: torch_swiglu(a, a, **kwargs), (outside,))
    print("B08 forward + gradient: PASS")
    print("Module 06 bridge asset: PASS")
