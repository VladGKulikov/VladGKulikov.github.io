"""Modern LLMs module 07: executable NumPy → PyTorch bridges (B09, B10).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def numpy_topk_router_explicit(
    logits: np.ndarray,
    top_k: int,
    *,
    selection_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal sigmoid router: bias changes selection, not clean mixture weights."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("expected non-empty finite logits[T,E]")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("top_k must be an integer in [1,E]")
    log_scores = -np.logaddexp(0.0, -logits)
    clean = np.exp(log_scores)
    # Without bias, monotonic sigmoid permits ranking logits before saturation.
    selection = logits if selection_bias is None else clean + selection_bias
    if selection.shape != logits.shape or not np.isfinite(selection).all():
        raise ValueError("selection_bias must broadcast to [T,E] and be finite")
    indices = np.argsort(-selection, axis=-1, kind="stable")[:, :top_k]
    selected_log = np.take_along_axis(log_scores, indices, axis=-1)
    weights = np.exp(selected_log - selected_log.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return indices, weights


def torch_topk_router_explicit(
    logits: torch.Tensor,
    top_k: int,
    *,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal tensor counterpart using sigmoid, topk, and gather."""
    if logits.ndim != 2 or not logits.numel() or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("expected non-empty finite floating logits[T,E]")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("top_k must be an integer in [1,E]")
    z = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_scores = -torch.logaddexp(torch.zeros_like(z), -z)
    selection = z if selection_bias is None else log_scores.exp() + selection_bias
    if selection.shape != logits.shape or not torch.isfinite(selection).all():
        raise ValueError("selection_bias must broadcast to [T,E] and be finite")
    _, indices = torch.topk(selection, top_k, dim=-1)
    selected_log = log_scores.gather(-1, indices)
    weights = selected_log.softmax(dim=-1)
    return indices, weights


def numpy_expert_dispatch_explicit(
    hidden: np.ndarray,
    expert_indices: np.ndarray,
    gates: np.ndarray,
    w_in: np.ndarray,
    w_out: np.ndarray,
) -> np.ndarray:
    """Minimal token/slot loop that makes dispatch and weighted combine visible."""
    if expert_indices.ndim != 2 or expert_indices.shape != gates.shape or expert_indices.shape[0] != hidden.shape[0]:
        raise ValueError("indices and gates must have matching shape [T,K]")
    if not np.issubdtype(expert_indices.dtype, np.integer):
        raise ValueError("expert indices must be integers")
    if np.any(expert_indices < 0) or np.any(expert_indices >= w_in.shape[0]):
        raise ValueError("expert index out of range")
    output = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(expert_indices.shape[1]):
            expert = int(expert_indices[token, slot])
            mid = np.tanh(hidden[token] @ w_in[expert])
            output[token] += gates[token, slot] * (mid @ w_out[expert])
    return output


def torch_expert_dispatch_explicit(
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    gates: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    """Group assignments by expert and combine them with index_add_."""
    if expert_indices.ndim != 2 or expert_indices.shape != gates.shape or expert_indices.shape[0] != hidden.shape[0]:
        raise ValueError("indices and gates must have matching shape [T,K]")
    if expert_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("expert indices must be integers")
    if torch.any(expert_indices < 0) or torch.any(expert_indices >= w_in.shape[0]):
        raise ValueError("expert index out of range")
    output = torch.zeros_like(hidden)
    for expert in range(w_in.shape[0]):
        token_ids, slots = torch.where(expert_indices == expert)
        if token_ids.numel() == 0:
            continue
        mid = torch.tanh(hidden[token_ids] @ w_in[expert])
        expert_out = (mid @ w_out[expert]) * gates[token_ids, slots, None]
        output.index_add_(0, token_ids, expert_out)
    return output


if __name__ == "__main__":
    logits_np = np.array([[2.1, -0.4, 0.8, 1.3], [0.2, 1.9, -1.1, 0.7]])
    bias_np = np.array([0.0, 0.08, -0.03, 0.02])
    i_np, w_np = numpy_topk_router_explicit(logits_np, 2, selection_bias=bias_np)
    i_t, w_t = torch_topk_router_explicit(torch.tensor(logits_np), 2, selection_bias=torch.tensor(bias_np))
    np.testing.assert_array_equal(i_np, i_t.numpy())
    np.testing.assert_allclose(w_np, w_t.numpy())
    extreme = np.array([[-1002., -1000., -1001., -1003.]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ei, ew = numpy_topk_router_explicit(extreme, 2)
    ti, tw = torch_topk_router_explicit(torch.tensor(extreme), 2)
    np.testing.assert_array_equal(ei, [[1, 2]])
    np.testing.assert_array_equal(ei, ti.numpy())
    np.testing.assert_allclose(ew, tw.numpy(), rtol=1e-12, atol=1e-12)
    z = torch.tensor(logits_np, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: torch_topk_router_explicit(x, 2)[1], (z,))
    one = torch_topk_router_explicit(z, 1)[1]
    np.testing.assert_allclose(torch.autograd.grad(one.sum(), z)[0].numpy(), 0., atol=1e-12)
    print("B09 routing, extreme logits, and selected-gate gradients: PASS")

    rng = np.random.default_rng(10)
    x_np = rng.standard_normal((6, 4))
    idx_np = np.array([[0,1],[1,2],[2,0],[0,2],[1,0],[2,1]])
    gates_np = np.full((6,2), 0.5)
    w1_np = rng.standard_normal((3,4,5)); w2_np = rng.standard_normal((3,5,4))
    np_out = numpy_expert_dispatch_explicit(x_np, idx_np, gates_np, w1_np, w2_np)
    t_out = torch_expert_dispatch_explicit(torch.tensor(x_np), torch.tensor(idx_np), torch.tensor(gates_np), torch.tensor(w1_np), torch.tensor(w2_np))
    np.testing.assert_allclose(np_out, t_out.numpy(), rtol=1e-10, atol=1e-10)
    args = [torch.tensor(x_np, requires_grad=True), torch.tensor(idx_np),
            torch.tensor(gates_np, requires_grad=True), torch.tensor(w1_np, requires_grad=True),
            torch.tensor(w2_np, requires_grad=True)]
    assert torch.autograd.gradcheck(
        lambda x, g, a, b: torch_expert_dispatch_explicit(x, args[1], g, a, b),
        (args[0], args[2], args[3], args[4]))
    for invalid in (-1, w1_np.shape[0], .5, np.nan, True):
        bad = np.full(idx_np.shape, invalid)
        for dispatch, values in (
            (numpy_expert_dispatch_explicit, (x_np,bad,gates_np,w1_np,w2_np)),
            (torch_expert_dispatch_explicit, tuple(torch.tensor(a) for a in (x_np,bad,gates_np,w1_np,w2_np))),
        ):
            try: dispatch(*values)
            except ValueError: pass
            else: raise AssertionError("invalid expert index accepted")
    duplicate = np.zeros_like(idx_np)
    expected = idx_np.shape[1] * (np.tanh(x_np @ w1_np[0]) @ w2_np[0])
    actual = numpy_expert_dispatch_explicit(x_np,duplicate,np.ones_like(gates_np),w1_np,w2_np)
    np.testing.assert_allclose(actual,expected)
    actual_t = torch_expert_dispatch_explicit(torch.tensor(x_np),torch.tensor(duplicate),
        torch.ones_like(torch.tensor(gates_np)),torch.tensor(w1_np),torch.tensor(w2_np))
    np.testing.assert_allclose(actual_t.numpy(),expected)
    print("B10 values, gradients, repeated assignments, and invalid indices: PASS")
    print("Module 07 bridge asset: PASS")
