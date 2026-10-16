"""Modern LLMs module 05: executable NumPy → PyTorch bridges (B04, B05).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def numpy_sdpa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Minimal visible SDPA algorithm; ``True`` in the mask means allowed."""
    output_dtype = np.result_type(q.dtype, k.dtype, v.dtype)
    if output_dtype.kind not in "fc":
        output_dtype = np.dtype(np.float64)
    work_dtype = np.result_type(output_dtype, np.float32)
    q, k, v = (x.astype(work_dtype, copy=False) for x in (q, k, v))
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    if allowed_mask is None:
        row_has_key = np.ones_like(scores[..., :1], dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(allowed_mask, dtype=bool), scores.shape)
        row_has_key = mask.any(axis=-1, keepdims=True)
        scores = np.where(mask, scores, -np.inf)

    safe_scores = np.where(row_has_key, scores, 0.0)
    shifted = safe_scores - safe_scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=-1, keepdims=True)
    weights = np.where(row_has_key, weights, 0.0)
    return np.matmul(weights, v).astype(output_dtype, copy=False)


def torch_sdpa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Line-for-line tensor counterpart of :func:`numpy_sdpa_explicit`."""
    output_dtype = q.dtype
    work_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    q, k, v = (x.to(work_dtype) for x in (q, k, v))
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale

    if allowed_mask is None:
        row_has_key = torch.ones_like(scores[..., :1], dtype=torch.bool)
    else:
        mask = torch.broadcast_to(allowed_mask.to(torch.bool), scores.shape)
        row_has_key = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.inf)

    safe_scores = torch.where(row_has_key, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1)
    weights = torch.where(row_has_key, weights, torch.zeros_like(weights))
    return (weights @ v).to(output_dtype)


def torch_sdpa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Public PyTorch SDPA API; the runtime may choose an optimized backend."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
    )


def numpy_gqa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Visible GQA: repeat K/V heads, then apply the explicit SDPA reference."""
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = np.repeat(k, group, axis=1)
    v_repeated = np.repeat(v, group, axis=1)
    return numpy_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)


def torch_gqa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Visible tensor GQA: repeat K/V heads, then use explicit tensor SDPA."""
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = k.repeat_interleave(group, dim=1)
    v_repeated = v.repeat_interleave(group, dim=1)
    return torch_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)


def torch_gqa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Public SDPA GQA path; availability depends on PyTorch backend support."""
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
        enable_gqa=True,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    q_np = rng.standard_normal((1, 2, 4, 8))
    k_np = rng.standard_normal((1, 2, 4, 8))
    v_np = rng.standard_normal((1, 2, 4, 6))
    mask_np = np.tril(np.ones((4, 4), dtype=bool))

    np_out = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
    q = torch.tensor(q_np, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(k_np, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(v_np, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(mask_np)
    explicit_out = torch_sdpa_explicit(q, k, v, allowed_mask=mask)
    api_out = torch_sdpa_optimized(q, k, v, allowed_mask=mask, dropout_p=0.0)

    torch.testing.assert_close(explicit_out, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(api_out, explicit_out, rtol=1e-10, atol=1e-10)

    mask_np[0] = False
    np_empty = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
    t_empty = torch_sdpa_explicit(q, k, v, allowed_mask=torch.tensor(mask_np))
    api_empty = torch_sdpa_optimized(q, k, v, allowed_mask=torch.tensor(mask_np), dropout_p=0.0)
    np.testing.assert_array_equal(np_empty[..., 0, :], 0.0)
    torch.testing.assert_close(t_empty[..., 0, :], torch.zeros_like(t_empty[..., 0, :]))
    torch.testing.assert_close(api_empty, t_empty, rtol=1e-10, atol=1e-10)

    grads_explicit = torch.autograd.grad(explicit_out.square().sum(), (q, k, v), retain_graph=True)
    grads_api = torch.autograd.grad(api_out.square().sum(), (q, k, v), retain_graph=True)
    for ge, ga in zip(grads_explicit, grads_api):
        torch.testing.assert_close(ge, ga, rtol=1e-9, atol=1e-10)
    explicit_out.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    print("B04 explicit NumPy / explicit PyTorch / optimized API: PASS")

    rng = np.random.default_rng(5)
    q_np = rng.standard_normal((1, 4, 3, 8))
    k_np = rng.standard_normal((1, 2, 3, 8))
    v_np = rng.standard_normal((1, 2, 3, 7))
    np_out = numpy_gqa_explicit(q_np, k_np, v_np)
    q, k, v = (torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (q_np, k_np, v_np))
    explicit = torch_gqa_explicit(q, k, v)
    api = torch_gqa_optimized(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(explicit, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
    for ge, ga in zip(torch.autograd.grad(explicit.square().sum(), (q,k,v)),
                      torch.autograd.grad(api.square().sum(), (q,k,v))):
        torch.testing.assert_close(ge, ga, rtol=1e-9, atol=1e-10)
    print("B05 explicit repetition / enable_gqa: PASS")
    print("Module 05 bridge asset: PASS")
