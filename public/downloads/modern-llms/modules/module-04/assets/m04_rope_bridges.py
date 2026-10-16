"""Modern LLMs module 04: executable NumPy → PyTorch bridges (B02).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


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


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((2, 5, 8))
    pos_np = np.arange(5, dtype=np.int64) + 11
    np_out = numpy_rope_explicit(x_np, pos_np)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    t_out = torch_rope_explicit(x, torch.tensor(pos_np))
    np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-12, atol=1e-12)
    t_out.square().sum().backward()
    # A rotation is orthogonal: d ||R x||^2 / dx = 2 x.
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=1e-12, atol=1e-12)
    # Half-precision IDs would merge 2048 and 2049 before the rotation.
    long_ids = torch.tensor([2048, 2049], dtype=torch.int64)
    half_x = torch.ones((2, 8), dtype=torch.float16)
    half_out = torch_rope_explicit(half_x, long_ids)
    expected = numpy_rope_explicit(half_x.numpy(), long_ids.numpy())
    np.testing.assert_allclose(half_out.numpy(), expected, rtol=1e-3, atol=1e-3)
    assert not torch.equal(half_out[0], half_out[1])
    assert long_ids.to(torch.float16)[0] == long_ids.to(torch.float16)[1]
    print("B02 outputs, analytic gradient, and float16 positions: PASS")
    print("Module 04 bridge asset: PASS")
