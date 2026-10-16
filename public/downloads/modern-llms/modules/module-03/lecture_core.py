"""Minimal learner implementation displayed in both M3 lectures.

The notebook adds a structured result, padding support, and broader diagnostics.
Run validate_module.py to check both code levels against independent calculations.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor | None = None,
    *,
    empty_row: Literal["error", "zero"] = "error",
) -> tuple[Tensor, Tensor]:
    """Educational attention with an explicit fully-masked-row policy.

    q: [..., T_q, d_k]
    k: [..., T_k, d_k]
    v: [..., T_k, d_v]
    allowed: broadcastable bool mask [..., T_q, T_k], True = allowed
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("q, k and v must have at least two dimensions")
    if any(size <= 0 for t in (q, k, v) for size in t.shape):
        raise ValueError("q, k and v must have non-empty axes")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("q, k and v must be finite floating-point tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k and v must be on one device")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share d_k")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share T_k")

    dtype = torch.float64 if torch.float64 in {q.dtype, k.dtype, v.dtype} else torch.float32
    q, k, v = (t.to(dtype=dtype) for t in (q, k, v))
    batch_shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q, k, v = (t.expand(*batch_shape, *t.shape[-2:]) for t in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if not torch.isfinite(scores).all():
        raise FloatingPointError("attention scores overflowed")

    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    else:
        if allowed.dtype is not torch.bool:
            raise TypeError("allowed must be a bool mask")
        allowed = torch.broadcast_to(allowed.to(scores.device), scores.shape)

    empty = ~allowed.any(dim=-1)
    if empty_row == "error" and empty.any():
        raise ValueError("attention contains a fully masked row")

    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    row_max = torch.where(
        (~empty)[..., None],
        masked_scores.max(dim=-1, keepdim=True).values,
        torch.zeros_like(masked_scores[..., :1]),
    )
    exponentials = torch.where(
        allowed,
        torch.exp(masked_scores - row_max),
        torch.zeros_like(masked_scores),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    weights = exponentials / denominator
    output = weights @ v
    if not torch.isfinite(output).all():
        raise FloatingPointError("attention output overflowed")
    return output, weights


def causal_allowed(
    t_q: int,
    t_k: int,
    *,
    query_offset: int = 0,
    device=None,
) -> Tensor:
    if any(type(n) is not int for n in (t_q, t_k, query_offset)):
        raise TypeError("lengths and offset must be integers")
    if min(t_q, t_k) <= 0 or query_offset < 0:
        raise ValueError("positive lengths and non-negative offset required")
    q_pos = query_offset + torch.arange(t_q, device=device)[:, None]
    k_pos = torch.arange(t_k, device=device)[None, :]
    return k_pos <= q_pos


import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (d_model, n_heads)):
            raise ValueError("d_model and n_heads must be positive integers")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(z: Tensor) -> Tensor:
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        allowed = None
        if causal:
            allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        y, _ = scaled_dot_product_attention(q, k, v, allowed)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(y.to(dtype=x.dtype))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal=True)
        x = x + self.ffn(self.ln2(x))
        return x
