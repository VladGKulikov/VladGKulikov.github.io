"""M11 minimal probability and grouped-value teaching kernels."""
import numpy as np


def probability_vector(values, *, name="p", atol=1e-12):
    """Validate a distribution without hidden clipping or renormalization."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected a finite one-dimensional vector")
    if np.any(x < 0) or not np.isclose(x.sum(), 1.0, rtol=0.0, atol=atol):
        raise ValueError(f"{name}: probabilities must be >= 0 and sum to 1")
    return x


def rejection_kernel(p_values, q_values):
    """Acceptance probabilities, residual, and exact one-position balance."""
    p = probability_vector(p_values, name="p")
    q = probability_vector(q_values, name="q")
    if p.shape != q.shape:
        raise ValueError("p and q must use the same vocabulary")

    acceptance = np.zeros_like(p)
    support = q > 0  # q_t=0 means the proposer cannot emit token t
    acceptance[support] = np.minimum(1.0, p[support] / q[support])

    overlap = np.minimum(p, q)
    positive = np.maximum(p - q, 0.0)
    residual_mass = positive.sum()
    residual = (
        positive / residual_mass
        if residual_mass > 0.0
        else np.zeros_like(p)
    )
    balance = overlap + residual_mass * residual
    np.testing.assert_allclose(balance, p, rtol=0.0, atol=1e-12)
    return acceptance, residual, balance


def quantize_grouped_e2m1(values, group_size=16):
    """Teaching E2M1-like grid with one scale per group."""
    grid = np.array(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0,
          0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=np.float64,
    )
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("expected a non-empty finite one-dimensional vector")
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)) or group_size <= 0 or x.size % group_size:
        raise ValueError("length must be divisible by a positive group_size")

    groups = x.reshape(-1, group_size)
    max_abs = np.max(np.abs(groups), axis=1, keepdims=True)
    scale = np.where(max_abs > 0, max_abs / 6.0, 1.0)
    normalized = groups / scale
    codes = np.abs(normalized[..., None] - grid).argmin(axis=-1)
    return (grid[codes] * scale).reshape(x.shape)
