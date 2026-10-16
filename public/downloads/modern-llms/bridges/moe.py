"""B09/B10 — visible top-k routing and expert dispatch/combine."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch


def _numpy_scores(logits: np.ndarray, activation: str) -> np.ndarray:
    if activation == "softmax":
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)
    if activation == "sigmoid":
        return np.exp(-np.logaddexp(0.0, -logits))
    raise ValueError("activation must be 'softmax' or 'sigmoid'")


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


def numpy_topk_router(
    logits: np.ndarray,
    top_k: int,
    *,
    selection_bias: np.ndarray | None = None,
    activation: str = "softmax",
    renormalize: bool = True,
    route_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select with biased scores but derive mixture weights from clean scores."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all() or isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or not (1 <= top_k <= logits.shape[1]):
        raise ValueError("expected finite logits[T,E] and 1 <= top_k <= E")
    clean = _numpy_scores(logits, activation)
    if selection_bias is None:
        selection = logits
    else:
        try:
            bias = np.broadcast_to(np.asarray(selection_bias, dtype=np.float64), clean.shape)
        except ValueError as exc:
            raise ValueError("selection_bias is not broadcastable to logits") from exc
        if not np.isfinite(bias).all():
            raise ValueError("selection_bias must be finite")
        selection = clean + bias
    if not np.isfinite(route_scale):
        raise ValueError("route_scale must be finite")
    indices = np.argsort(-selection, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(clean, indices, axis=-1)
    if renormalize:
        selected_logits = np.take_along_axis(logits, indices, axis=-1)
        log_weights = selected_logits if activation == "softmax" else -np.logaddexp(0.0, -selected_logits)
        weights = np.exp(log_weights - log_weights.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
    weights = weights * route_scale
    return indices.astype(np.int64), weights, clean


def torch_topk_router(
    logits: torch.Tensor,
    top_k: int,
    *,
    selection_bias: torch.Tensor | None = None,
    activation: str = "softmax",
    renormalize: bool = True,
    route_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch top-k route; tie indices are not guaranteed stable across implementations."""
    if logits.ndim != 2 or not logits.numel() or not logits.is_floating_point() or isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= logits.shape[1]):
        raise ValueError("expected logits[T,E] and 1 <= top_k <= E")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if not np.isfinite(route_scale):
        raise ValueError("route_scale must be finite")
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    clean = logits.softmax(-1) if activation == "softmax" else logits.sigmoid() if activation == "sigmoid" else None
    if clean is None:
        raise ValueError("activation must be 'softmax' or 'sigmoid'")
    if selection_bias is None:
        selection = logits
    else:
        try:
            selection = clean + selection_bias
        except RuntimeError as exc:
            raise ValueError("selection_bias is not broadcastable to logits") from exc
        if not torch.isfinite(selection_bias).all():
            raise ValueError("selection_bias must be finite")
    _, indices = torch.topk(selection, top_k, dim=-1)
    weights = clean.gather(-1, indices)
    if renormalize:
        selected_logits = logits.gather(-1, indices)
        log_weights = selected_logits if activation == "softmax" else -torch.logaddexp(torch.zeros_like(selected_logits), -selected_logits)
        weights = log_weights.softmax(dim=-1)
    return indices, weights * route_scale, clean


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


def numpy_expert_dispatch(
    hidden: np.ndarray,
    expert_indices: np.ndarray,
    gates: np.ndarray,
    w_in: np.ndarray,
    w_out: np.ndarray,
    *,
    capacity: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """FIFO expert execution for a small reference MoE with tanh expert activation."""
    hidden = np.asarray(hidden, dtype=np.float64)
    indices = np.asarray(expert_indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("expert indices must be integers")
    gates = np.asarray(gates, dtype=np.float64)
    w_in, w_out = np.asarray(w_in, dtype=np.float64), np.asarray(w_out, dtype=np.float64)
    if hidden.ndim != 2 or indices.ndim != 2 or gates.shape != indices.shape or indices.shape[0] != hidden.shape[0]:
        raise ValueError("expected hidden[T,D] and matching expert_indices/gates[T,K]")
    if capacity is not None and capacity <= 0:
        raise ValueError("capacity must be positive")
    if not np.isfinite(hidden).all() or not np.isfinite(gates).all() or not np.isfinite(w_in).all() or not np.isfinite(w_out).all():
        raise ValueError("hidden, gates, and expert weights must be finite")
    if w_in.ndim != 3 or w_out.ndim != 3 or w_in.shape[0] != w_out.shape[0]:
        raise ValueError("expected w_in[E,D,F] and w_out[E,F,D]")
    if w_in.shape[1] != hidden.shape[1] or w_out.shape[1] != w_in.shape[2] or w_out.shape[2] != hidden.shape[1]:
        raise ValueError("incompatible expert dimensions")
    if np.any(indices < 0) or np.any(indices >= w_in.shape[0]):
        raise ValueError("expert index out of range")
    indices = indices.astype(np.int64, copy=False)
    output = np.zeros_like(hidden)
    selected = np.bincount(indices.ravel(), minlength=w_in.shape[0]).astype(np.int64)
    kept = np.zeros(w_in.shape[0], dtype=np.int64)
    for token in range(hidden.shape[0]):
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            if capacity is not None and kept[expert] >= capacity:
                continue
            mid = np.tanh(hidden[token] @ w_in[expert])
            output[token] += gates[token, slot] * (mid @ w_out[expert])
            kept[expert] += 1
    return output, {"selected_load": selected, "kept_load": kept, "dropped_load": selected - kept}


def torch_expert_dispatch(
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    gates: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    capacity: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PyTorch expert execution using per-expert batches and ``index_add_`` combine."""
    if expert_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("expert indices must be integers")
    if hidden.ndim != 2 or expert_indices.ndim != 2 or gates.shape != expert_indices.shape or expert_indices.shape[0] != hidden.shape[0]:
        raise ValueError("expected hidden[T,D] and matching expert_indices/gates[T,K]")
    if capacity is not None and capacity <= 0:
        raise ValueError("capacity must be positive")
    if w_in.ndim != 3 or w_out.ndim != 3 or w_in.shape[0] != w_out.shape[0]:
        raise ValueError("expected w_in[E,D,F] and w_out[E,F,D]")
    if w_in.shape[1] != hidden.shape[1] or w_out.shape[1] != w_in.shape[2] or w_out.shape[2] != hidden.shape[1]:
        raise ValueError("incompatible expert dimensions")
    if torch.any(expert_indices < 0) or torch.any(expert_indices >= w_in.shape[0]):
        raise ValueError("expert index out of range")
    if not torch.isfinite(hidden).all() or not torch.isfinite(gates).all() or not torch.isfinite(w_in).all() or not torch.isfinite(w_out).all():
        raise ValueError("hidden, gates, and expert weights must be finite")
    experts = w_in.shape[0]
    flat_expert = expert_indices.reshape(-1)
    token_ids = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(expert_indices.shape[1])
    flat_gates = gates.reshape(-1)
    selected = torch.bincount(flat_expert, minlength=experts)
    kept = torch.zeros(experts, dtype=torch.long, device=hidden.device)
    output = torch.zeros_like(hidden)
    for expert in range(experts):
        assignment = torch.nonzero(flat_expert == expert, as_tuple=False).flatten()
        if capacity is not None:
            assignment = assignment[:capacity]
        kept[expert] = assignment.numel()
        if assignment.numel() == 0:
            continue
        tokens = token_ids[assignment]
        mid = torch.tanh(hidden[tokens] @ w_in[expert])
        expert_out = (mid @ w_out[expert]) * flat_gates[assignment, None]
        output.index_add_(0, tokens, expert_out)
    return output, {"selected_load": selected, "kept_load": kept, "dropped_load": selected - kept}
