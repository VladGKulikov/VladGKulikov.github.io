"""B18 — exact one-position speculative rejection correction."""
from __future__ import annotations

import numpy as np
import torch


def _numpy_probability(values: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError(f"{name} must be a finite non-negative vector")
    if not np.isclose(x.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return x


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


def numpy_rejection_kernel(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return proposal acceptance, residual distribution, and reconstructed target."""
    p, q = _numpy_probability(p, "p"), _numpy_probability(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must share a vocabulary")
    return numpy_rejection_kernel_explicit(p, q)


def torch_rejection_kernel(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor equivalent of :func:`numpy_rejection_kernel`."""
    if p.ndim != 1 or p.numel() < 2 or q.shape != p.shape:
        raise ValueError("p and q must be matching one-dimensional probability vectors")
    if p.dtype not in (torch.float32, torch.float64) or q.dtype != p.dtype or q.device != p.device:
        raise ValueError("p and q must share a device and float32 or float64 dtype")
    if not torch.isfinite(p).all() or not torch.isfinite(q).all() or torch.any(p < 0) or torch.any(q < 0):
        raise ValueError("probabilities must be finite and non-negative")
    tolerance = 1e-12 if p.dtype == torch.float64 else 1e-7
    if not torch.allclose(p.sum(), p.new_tensor(1.0), rtol=0.0, atol=tolerance):
        raise ValueError("p must sum to one")
    if not torch.allclose(q.sum(), q.new_tensor(1.0), rtol=0.0, atol=tolerance):
        raise ValueError("q must sum to one")
    return torch_rejection_kernel_explicit(p, q)


def torch_sample_speculative_token(
    p: torch.Tensor,
    q: torch.Tensor,
    proposal: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[int, bool]:
    """Accept one draft token or sample exact residual correction."""
    acceptance, residual, _ = torch_rejection_kernel(p, q)
    if isinstance(proposal, bool) or not isinstance(proposal, int) or proposal < 0 or proposal >= p.numel() or q[proposal] <= 0:
        raise ValueError("proposal must lie in the positive support of q")
    draw = torch.rand((), generator=generator, device=p.device, dtype=p.dtype)
    if draw < acceptance[proposal]:
        return int(proposal), True
    if residual.sum() <= 0:
        raise ArithmeticError("rejected a proposal without positive residual mass")
    return int(torch.multinomial(residual, 1, generator=generator).item()), False
