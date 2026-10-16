"""Modern LLMs module 11: executable NumPy → PyTorch bridges (B18, B19).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def numpy_rejection_kernel_explicit(
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal exact rejection correction, assuming valid probability vectors."""
    acceptance = np.zeros_like(p, dtype=np.float64)
    support = q > 0
    acceptance[support] = np.minimum(1.0, p[support] / q[support])
    overlap = np.minimum(p, q)
    positive = np.maximum(p - q, 0.0)
    residual_mass = positive.sum()
    residual = positive / residual_mass if residual_mass > 0 else np.zeros_like(p)
    reconstructed = overlap + residual_mass * residual
    return acceptance, residual, reconstructed


def torch_rejection_kernel_explicit(
    p: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal tensor counterpart of exact rejection correction."""
    acceptance = torch.zeros_like(p)
    support = q > 0
    acceptance[support] = torch.minimum(torch.ones_like(p[support]), p[support] / q[support])
    overlap = torch.minimum(p, q)
    positive = torch.clamp(p - q, min=0)
    residual_mass = positive.sum()
    residual = positive / residual_mass if bool(residual_mass > 0) else torch.zeros_like(p)
    reconstructed = overlap + residual_mass * residual
    return acceptance, residual, reconstructed


def torch_sample_speculative_token_explicit(
    p: torch.Tensor,
    q: torch.Tensor,
    proposal: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, bool]:
    """Sample one exact speculative step from the explicit rejection kernel."""
    if isinstance(proposal, bool) or not isinstance(proposal, int) or not 0 <= proposal < p.numel() or q[proposal] <= 0:
        raise ValueError("proposal must be an integer in the positive support of q")
    acceptance, residual, _ = torch_rejection_kernel_explicit(p, q)
    draw = torch.rand((), generator=generator, device=p.device, dtype=p.dtype)
    if draw < acceptance[proposal]:
        return int(proposal), True
    return int(torch.multinomial(residual, 1, generator=generator).item()), False


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


if __name__ == "__main__":
    p_np = np.array([0.40,0.30,0.15,0.10,0.05]); q_np = np.array([0.25,0.35,0.20,0.10,0.10])
    a_np, r_np, b_np = numpy_rejection_kernel_explicit(p_np, q_np)
    a_t, r_t, b_t = torch_rejection_kernel_explicit(torch.tensor(p_np), torch.tensor(q_np))
    np.testing.assert_allclose(a_np, a_t.numpy()); np.testing.assert_allclose(r_np, r_t.numpy()); np.testing.assert_allclose(b_np, b_t.numpy()); np.testing.assert_allclose(b_np, p_np)
    gen = torch.Generator().manual_seed(18); counts = torch.zeros(5)
    q = torch.tensor(q_np); p = torch.tensor(p_np)
    for _ in range(12000):
        proposal = int(torch.multinomial(q, 1, generator=gen))
        token, _ = torch_sample_speculative_token_explicit(p, q, proposal, generator=gen)
        counts[token] += 1
    assert torch.max(torch.abs(counts / counts.sum() - p)) < 0.02

    # A real, tiny residual must remain a probability distribution.
    near_q = np.array([1.0, 0.0]); near_p = np.array([1.0-1e-14, 1e-14])
    for kernel, args in [
        (numpy_rejection_kernel_explicit, (near_p, near_q)),
        (torch_rejection_kernel_explicit, (torch.tensor(near_p), torch.tensor(near_q))),
    ]:
        _, residual, balance = kernel(*args)
        np.testing.assert_allclose(np.asarray(residual), [0.0, 1.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(balance), near_p, rtol=0, atol=1e-15)
    # Uniform RNG can return zero: a zero-acceptance proposal must still be rejected.
    from unittest.mock import patch
    with patch.object(torch, "rand", return_value=torch.tensor(0.0, dtype=torch.float64)):
        token, accepted = torch_sample_speculative_token_explicit(
            torch.tensor([0.,1.], dtype=torch.float64),
            torch.tensor([1.,0.], dtype=torch.float64), 0, generator=gen)
        assert token == 1 and not accepted

    print("B18 exact balance, tiny residual, zero draw, Monte Carlo: PASS")

    rng = np.random.default_rng(19)
    x_np = rng.standard_normal((3,2,11))
    grid_np = np.array([-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6.0])
    np_q = numpy_grouped_fake_quant_explicit(x_np, grid=grid_np, group_size=4)
    t_q = torch_grouped_fake_quant_explicit(torch.tensor(x_np), grid=torch.tensor(grid_np), group_size=4)
    np.testing.assert_allclose(np_q, t_q.numpy())
    zeros = torch.zeros(2,3,7, dtype=torch.float64)
    zero_q = torch_grouped_fake_quant_explicit(zeros, grid=torch.tensor(grid_np), group_size=4)
    torch.testing.assert_close(zero_q, zeros)
    print("B19 explicit grouped fake quantization: PASS")
    print("Module 11 bridge asset: PASS")
