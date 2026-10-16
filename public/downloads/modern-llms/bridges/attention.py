"""B04/B05 — transparent SDPA and grouped-query attention bridges."""
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


def b04_parity_demo() -> None:
    """Executable parity check used by the lecture and module attachment."""
    rng = np.random.default_rng(4)
    q_np = rng.standard_normal((1, 2, 4, 8))
    k_np = rng.standard_normal((1, 2, 4, 8))
    v_np = rng.standard_normal((1, 2, 4, 6))
    mask_np = np.tril(np.ones((4, 4), dtype=bool))

    out_np = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
    q = torch.tensor(q_np, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(k_np, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(v_np, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(mask_np)
    out_explicit = torch_sdpa_explicit(q, k, v, allowed_mask=mask)
    out_api = torch_sdpa_optimized(q, k, v, allowed_mask=mask, dropout_p=0.0)

    torch.testing.assert_close(out_explicit, torch.from_numpy(out_np), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(out_api, out_explicit, rtol=1e-10, atol=1e-10)

    mask_np[0] = False
    empty_np = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
    empty_t = torch_sdpa_explicit(q, k, v, allowed_mask=torch.tensor(mask_np))
    empty_api = torch_sdpa_optimized(q, k, v, allowed_mask=torch.tensor(mask_np), dropout_p=0.0)
    np.testing.assert_array_equal(empty_np[..., 0, :], 0.0)
    torch.testing.assert_close(empty_t[..., 0, :], torch.zeros_like(empty_t[..., 0, :]))
    torch.testing.assert_close(empty_api, empty_t, rtol=1e-10, atol=1e-10)

    out_explicit.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    exp = np.exp(shifted)
    denom = exp.sum(axis=axis, keepdims=True)
    return np.divide(exp, denom, out=np.zeros_like(exp), where=denom != 0)


def numpy_sdpa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Reference SDPA for tensors ``[B,H,T,D]``; ``True`` in a bool mask means allowed."""
    q, k, v = (np.asarray(t) for t in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B,H,T,D]")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible SDPA shapes")
    if not all(np.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("q, k, v must be finite")
    scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = np.einsum("bhtd,bhsd->bhts", q.astype(np.float64), k.astype(np.float64)) * scale
    all_masked = None
    if allowed_mask is not None:
        mask = np.asarray(allowed_mask, dtype=bool)
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("allowed_mask is not broadcastable to attention scores") from exc
        all_masked = ~mask.any(axis=-1, keepdims=True)
        scores = np.where(mask, scores, -np.inf)
        # Avoid -inf - (-inf) in the fully masked-row reference path.
        scores = np.where(all_masked, 0.0, scores)
    weights = _numpy_softmax(scores, axis=-1)
    if all_masked is not None:
        weights = np.where(all_masked, 0.0, weights)
    return np.einsum("bhts,bhsd->bhtd", weights, v.astype(np.float64)).astype(
        np.result_type(q.dtype, k.dtype, v.dtype), copy=False
    )


def torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Call PyTorch SDPA; callers must pass ``dropout_p=0`` explicitly at evaluation time."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def numpy_gqa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reference GQA by explicitly repeating KV heads before SDPA."""
    q, k, v = (np.asarray(t) for t in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must have [B,H,T,D] shapes and K/V must share B,H,T")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv and batch/head dimensions must match")
    group = q.shape[1] // k.shape[1]
    return numpy_sdpa(
        q,
        np.repeat(k, group, axis=1),
        np.repeat(v, group, axis=1),
        allowed_mask=allowed_mask,
    )


def torch_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Use the public ``enable_gqa`` SDPA path; backend support is device-dependent."""
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=True,
    )
