#!/usr/bin/env python3
"""Reviewed calculations and learner-code contracts for Modern LLMs Module 7.

The module owns one frozen numerical payload in ``data/reference_metrics.json``.
Ordinary execution validates that payload; ``--write`` is required to replace it.
The marked educational-core regions are embedded into the RU/EN notebooks by the shared
builder so learner code and maintainer reference code cannot drift independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation

def _finite_array(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    """Return ``value`` as a finite float64 array with an optional rank contract."""

    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    """Validate an integer scalar while rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _bounded_real(
    name: str,
    value: Any,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    """Validate one finite real scalar against an explicit interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    lower_ok = result >= lower if lower_inclusive else result > lower
    upper_ok = result <= upper if upper_inclusive else result < upper
    if not (lower_ok and upper_ok):
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        raise ValueError(f"{name} must lie in {left}{lower}, {upper}{right}")
    return result


def stable_sigmoid(value: Any) -> np.ndarray:
    """Evaluate the logistic sigmoid without overflowing for large finite logits."""

    x = _finite_array("value", value)
    out = np.empty_like(x)
    nonnegative = x >= 0.0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def stable_softmax(value: Any, *, axis: int = -1) -> np.ndarray:
    """Compute a finite softmax along ``axis`` with max subtraction."""

    x = _finite_array("value", value)
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("axis is outside the array rank")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    denominator = exp_shifted.sum(axis=axis, keepdims=True)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("softmax normalization failed")
    return exp_shifted / denominator


def _log_sigmoid(value: np.ndarray) -> np.ndarray:
    """Return log(sigmoid(value)) in a numerically stable form."""

    return -np.logaddexp(0.0, -value)


def _deterministic_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Select descending scores with lower expert index breaking exact ties."""

    token_count, expert_count = scores.shape
    indices = np.arange(expert_count)
    selected = np.empty((token_count, top_k), dtype=np.int64)
    for token in range(token_count):
        order = np.lexsort((indices, -scores[token]))
        selected[token] = order[:top_k]
    return selected


@dataclass(frozen=True)
class RouterResult:
    """Validated output of one token-choice router pass.

    Shapes are ``clean_scores:[T,E]``, ``selected_experts:[T,k]``, and
    ``gate_weights:[T,k]``. ``selection_scores`` includes the optional balancing bias;
    ``gate_weights`` never does.
    """

    clean_scores: np.ndarray
    selection_scores: np.ndarray
    selected_experts: np.ndarray
    gate_weights: np.ndarray
    expert_counts: np.ndarray


@dataclass(frozen=True)
class DispatchPlan:
    """Capacity-limited token-to-expert assignments under a deterministic FIFO policy."""

    capacity_per_expert: int
    accepted: np.ndarray
    gate_weights: np.ndarray
    expert_counts: np.ndarray
    dropped_assignments: int
    fully_dropped_tokens: tuple[int, ...]


@dataclass(frozen=True)
class LoadStatistics:
    """Fractions and imbalance diagnostics for expert assignment counts."""

    fractions: np.ndarray
    coefficient_of_variation: float
    maximum_fraction: float
    minimum_fraction: float


# END EDUCATIONAL CORE: validation


# BEGIN EDUCATIONAL CORE: routing

def route_logits(
    logits: Any,
    *,
    top_k: int,
    scoring: Literal["sigmoid", "softmax"] = "sigmoid",
    selection_bias: Any | None = None,
) -> RouterResult:
    """Route a non-empty token batch with deterministic token-choice top-k.

    Parameters
    ----------
    logits:
        Router logits of shape ``[T,E]`` or one token of shape ``[E]``.
    top_k:
        Number of routed experts per token, constrained to ``1 <= k <= E``.
    scoring:
        ``"sigmoid"`` for independent V3-style scores or ``"softmax"`` for a
        categorical router.
    selection_bias:
        Optional vector ``[E]`` added **only** to the selection scores. Gate weights are
        normalized from the clean scores, preserving the aux-loss-free contract.

    Exact ties are broken by lower expert index so CPU, Colab, and different NumPy builds
    produce the same route.
    """

    raw = _finite_array("logits", logits)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ValueError("logits must have shape [T,E] or [E]")
    token_count, expert_count = raw.shape
    k = _positive_int("top_k", top_k)
    if k > expert_count:
        raise ValueError("top_k cannot exceed the number of experts")
    if scoring not in {"sigmoid", "softmax"}:
        raise ValueError("scoring must be 'sigmoid' or 'softmax'")

    if selection_bias is None:
        bias = np.zeros(expert_count, dtype=np.float64)
    else:
        bias = _finite_array("selection_bias", selection_bias, ndim=1)
        if bias.shape != (expert_count,):
            raise ValueError("selection_bias must have shape [E]")

    clean_scores = stable_sigmoid(raw) if scoring == "sigmoid" else stable_softmax(raw, axis=1)
    selection_scores = clean_scores + bias[None, :]
    # Both score functions preserve rank mathematically. Ranking raw logits when
    # no bias is supplied avoids artificial ties from saturation/underflow.
    selected = _deterministic_top_k(raw if selection_bias is None else selection_scores, k)

    selected_logits = np.take_along_axis(raw, selected, axis=1)
    if scoring == "sigmoid":
        # Normalize selected sigmoid scores in log space. This remains finite even when
        # every displayed sigmoid underflows to zero at extreme negative logits.
        log_weights = _log_sigmoid(selected_logits)
        gate_weights = stable_softmax(log_weights, axis=1)
    else:
        # Re-normalizing clean softmax scores over top-k is exactly softmax of selected logits.
        gate_weights = stable_softmax(selected_logits, axis=1)

    counts = np.bincount(selected.ravel(), minlength=expert_count).astype(np.int64)
    if not np.allclose(gate_weights.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("router gates must sum to one per token")
    return RouterResult(
        clean_scores=clean_scores,
        selection_scores=selection_scores,
        selected_experts=selected,
        gate_weights=gate_weights,
        expert_counts=counts,
    )


def expert_capacity(
    *,
    token_count: int,
    top_k: int,
    expert_count: int,
    capacity_factor: Real,
) -> int:
    """Return ``ceil(capacity_factor * T * k / E)`` assignments per expert."""

    tokens = _positive_int("token_count", token_count)
    k = _positive_int("top_k", top_k)
    experts = _positive_int("expert_count", expert_count)
    if k > experts:
        raise ValueError("top_k cannot exceed expert_count")
    factor = _bounded_real(
        "capacity_factor",
        capacity_factor,
        lower=0.0,
        upper=float("inf"),
        lower_inclusive=False,
    )
    return max(1, math.ceil(factor * tokens * k / experts))


def dispatch_with_capacity(
    router: RouterResult,
    *,
    capacity_factor: Real = 1.0,
    renormalize_kept_gates: bool = True,
) -> DispatchPlan:
    """Apply deterministic FIFO expert capacity and optionally renormalize kept gates.

    This transparent policy is intentionally simpler than production grouped dispatch. Each
    token is visited in input order and each route slot in router order. Overflow assignments
    are dropped; no hidden rerouting occurs. If ``renormalize_kept_gates`` is true, surviving
    weights are renormalized as an explicit *course policy* rather than a universal MoE rule.
    Tokens with no surviving assignment emit zero from the expert branch, remain on the
    residual path outside this educational layer, and are reported explicitly.
    """

    selected = np.asarray(router.selected_experts, dtype=np.int64)
    gates = _finite_array("router.gate_weights", router.gate_weights, ndim=2)
    if selected.ndim != 2 or selected.shape != gates.shape:
        raise ValueError("router selected_experts and gate_weights must share shape [T,k]")
    token_count, k = selected.shape
    if token_count == 0:
        raise ValueError("router output must contain at least one token")
    expert_count = int(np.asarray(router.expert_counts).size)
    if expert_count < 1 or np.any(selected < 0) or np.any(selected >= expert_count):
        raise ValueError("router contains an invalid expert index")

    capacity = expert_capacity(
        token_count=token_count,
        top_k=k,
        expert_count=expert_count,
        capacity_factor=capacity_factor,
    )
    accepted = np.zeros_like(selected, dtype=bool)
    counts = np.zeros(expert_count, dtype=np.int64)
    for token in range(token_count):
        for slot in range(k):
            expert = int(selected[token, slot])
            if counts[expert] < capacity:
                accepted[token, slot] = True
                counts[expert] += 1

    effective_gates = np.where(accepted, gates, 0.0)
    kept_mass = effective_gates.sum(axis=1, keepdims=True)
    fully_dropped = tuple(np.flatnonzero(kept_mass[:, 0] == 0.0).tolist())
    if renormalize_kept_gates:
        np.divide(
            effective_gates,
            kept_mass,
            out=effective_gates,
            where=kept_mass > 0.0,
        )
    dropped = int(accepted.size - accepted.sum())
    return DispatchPlan(
        capacity_per_expert=capacity,
        accepted=accepted,
        gate_weights=effective_gates,
        expert_counts=counts,
        dropped_assignments=dropped,
        fully_dropped_tokens=fully_dropped,
    )


def load_statistics(counts: Any) -> LoadStatistics:
    """Convert non-negative assignment counts into fractions and load CV."""

    raw = _finite_array("counts", counts, ndim=1)
    if np.any(raw < 0.0):
        raise ValueError("counts must be non-negative")
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("at least one assignment is required")
    fractions = raw / total
    mean = float(fractions.mean())
    return LoadStatistics(
        fractions=fractions,
        coefficient_of_variation=float(fractions.std() / mean),
        maximum_fraction=float(fractions.max()),
        minimum_fraction=float(fractions.min()),
    )


def aux_free_bias_step(
    selection_bias: Any,
    counts: Any,
    *,
    rate: Real,
    mode: Literal["sign", "proportional"] = "sign",
) -> np.ndarray:
    """Update expert selection biases outside backpropagation.

    ``mode="sign"`` reproduces the main update rule in Wang et al. (2024): an
    overloaded expert moves down by one fixed step and an underloaded expert moves up.
    ``mode="proportional"`` is the smoother ablation/teaching controller used in this
    module's convergence trace. Both modify selection only; clean gate values remain
    unchanged.
    """

    bias = _finite_array("selection_bias", selection_bias, ndim=1)
    stats = load_statistics(counts)
    if stats.fractions.shape != bias.shape:
        raise ValueError("selection_bias and counts must have the same shape")
    step = _bounded_real("rate", rate, lower=0.0, upper=float("inf"), lower_inclusive=False)
    error = stats.fractions.mean() - stats.fractions
    if mode == "sign":
        delta = np.sign(error)
    elif mode == "proportional":
        delta = error
    else:
        raise ValueError("mode must be 'sign' or 'proportional'")
    return bias + step * delta


def switch_auxiliary_loss(logits: Any, *, top_k: int = 1) -> float:
    """Compute ``E * sum_i f_i P_i`` for a softmax token-choice router batch."""

    raw = _finite_array("logits", logits)
    if raw.ndim != 2:
        raise ValueError("logits must have shape [T,E]")
    routed = route_logits(raw, top_k=top_k, scoring="softmax")
    token_count, expert_count = raw.shape
    assignment_fraction = routed.expert_counts / (token_count * top_k)
    mean_probability = routed.clean_scores.mean(axis=0)
    return float(expert_count * np.dot(assignment_fraction, mean_probability))


def router_z_loss(logits: Any) -> float:
    """Compute the ST-MoE router z-loss ``mean(logsumexp(logits)^2)``."""

    raw = _finite_array("logits", logits)
    if raw.ndim != 2:
        raise ValueError("logits must have shape [T,E]")
    maximum = np.max(raw, axis=1, keepdims=True)
    log_partition = maximum[:, 0] + np.log(np.exp(raw - maximum).sum(axis=1))
    return float(np.mean(log_partition**2))


# END EDUCATIONAL CORE: routing


# BEGIN EDUCATIONAL CORE: moe_layer

def swiglu_expert(inputs: Any, gate: Any, up: Any, down: Any) -> np.ndarray:
    """Apply one dense SwiGLU expert to ``inputs:[T,h]`` with explicit shape checks."""

    x = _finite_array("inputs", inputs)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("inputs must have shape [T,h] or [h]")
    w_gate = _finite_array("gate", gate, ndim=2)
    w_up = _finite_array("up", up, ndim=2)
    w_down = _finite_array("down", down, ndim=2)
    hidden = x.shape[1]
    if w_gate.shape[0] != hidden or w_up.shape != w_gate.shape:
        raise ValueError("gate and up must both have shape [h,i_exp]")
    if w_down.shape != (w_gate.shape[1], hidden):
        raise ValueError("down must have shape [i_exp,h]")
    gate_values = x @ w_gate
    up_values = x @ w_up
    return (gate_values * stable_sigmoid(gate_values) * up_values) @ w_down


@dataclass(frozen=True)
class MoEForwardResult:
    """Output plus routing metadata from the transparent educational MoE layer."""

    output: np.ndarray
    router: RouterResult
    dispatch: DispatchPlan | None


class EducationalMoE:
    """Small NumPy MoE with routed and always-on shared SwiGLU experts.

    Expert tensors use shape ``[E,h,i_exp]`` for gate/up and ``[E,i_exp,h]`` for
    down. The Python expert loop is intentionally inspectable; production code would use
    grouped GEMMs, fused dispatch, and distributed expert parallelism.
    """

    def __init__(
        self,
        router_weights: Any,
        expert_gate: Any,
        expert_up: Any,
        expert_down: Any,
        *,
        shared_gate: Any | None = None,
        shared_up: Any | None = None,
        shared_down: Any | None = None,
    ) -> None:
        """Validate and store routed/shared expert tensors without hidden casting later."""

        self.router_weights = _finite_array("router_weights", router_weights, ndim=2)
        self.expert_gate = _finite_array("expert_gate", expert_gate, ndim=3)
        self.expert_up = _finite_array("expert_up", expert_up, ndim=3)
        self.expert_down = _finite_array("expert_down", expert_down, ndim=3)
        experts, hidden, intermediate = self.expert_gate.shape
        if self.router_weights.shape != (hidden, experts):
            raise ValueError("router_weights must have shape [h,E]")
        if self.expert_up.shape != self.expert_gate.shape:
            raise ValueError("expert_up must match expert_gate")
        if self.expert_down.shape != (experts, intermediate, hidden):
            raise ValueError("expert_down must have shape [E,i_exp,h]")

        supplied = (shared_gate is not None, shared_up is not None, shared_down is not None)
        if len(set(supplied)) != 1:
            raise ValueError("shared gate/up/down must be supplied together")
        if shared_gate is None:
            self.shared_gate = np.empty((0, hidden, intermediate), dtype=np.float64)
            self.shared_up = np.empty((0, hidden, intermediate), dtype=np.float64)
            self.shared_down = np.empty((0, intermediate, hidden), dtype=np.float64)
        else:
            self.shared_gate = _finite_array("shared_gate", shared_gate, ndim=3)
            self.shared_up = _finite_array("shared_up", shared_up, ndim=3)
            self.shared_down = _finite_array("shared_down", shared_down, ndim=3)
            shared = self.shared_gate.shape[0]
            if self.shared_gate.shape[1:] != (hidden, intermediate):
                raise ValueError("shared_gate must have shape [S,h,i_exp]")
            if self.shared_up.shape != self.shared_gate.shape:
                raise ValueError("shared_up must match shared_gate")
            if self.shared_down.shape != (shared, intermediate, hidden):
                raise ValueError("shared_down must have shape [S,i_exp,h]")

    def forward(
        self,
        inputs: Any,
        *,
        top_k: int,
        scoring: Literal["sigmoid", "softmax"] = "sigmoid",
        selection_bias: Any | None = None,
        capacity_factor: Real | None = None,
    ) -> MoEForwardResult:
        """Route and combine one batch; capacity overflow follows ``dispatch_with_capacity``."""

        x = _finite_array("inputs", inputs)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.router_weights.shape[0]:
            raise ValueError("inputs must have shape [T,h]")
        router = route_logits(
            x @ self.router_weights,
            top_k=top_k,
            scoring=scoring,
            selection_bias=selection_bias,
        )
        dispatch = (
            dispatch_with_capacity(router, capacity_factor=capacity_factor)
            if capacity_factor is not None
            else None
        )
        gates = dispatch.gate_weights if dispatch is not None else router.gate_weights
        accepted = dispatch.accepted if dispatch is not None else np.ones_like(gates, dtype=bool)
        output = np.zeros_like(x)

        for expert in range(self.expert_gate.shape[0]):
            token_rows, slots = np.nonzero((router.selected_experts == expert) & accepted)
            if token_rows.size:
                expert_output = swiglu_expert(
                    x[token_rows],
                    self.expert_gate[expert],
                    self.expert_up[expert],
                    self.expert_down[expert],
                )
                output[token_rows] += gates[token_rows, slots, None] * expert_output

        for shared in range(self.shared_gate.shape[0]):
            output += swiglu_expert(
                x,
                self.shared_gate[shared],
                self.shared_up[shared],
                self.shared_down[shared],
            )
        return MoEForwardResult(output=output, router=router, dispatch=dispatch)


# END EDUCATIONAL CORE: moe_layer


# BEGIN EDUCATIONAL CORE: accounting

@dataclass(frozen=True)
class MoEModelSpec:
    """Explicit parameter-counting contract for a dense/MoE Transformer.

    The class deliberately avoids pretending that one formula covers every MoE family. The
    caller states how many dense and MoE layers exist, whether the vocabulary matrices are tied,
    whether expert/router biases are present, and how many *full* vocabulary matrices are counted
    in the active-parameter convention. ``active_vocab_matrices=1`` is the compute-oriented
    convention used for gpt-oss: the dense output projection is counted, while an indexed input
    embedding lookup is not charged as a full matrix.
    """

    vocab_size: int
    hidden_size: int
    layers: int
    attention_params_per_layer: int
    routed_experts: int
    experts_per_token: int
    expert_intermediate: int
    shared_experts: int = 0
    dense_layers: int = 0
    dense_intermediate: int = 0
    vocab_matrices_total: int = 2
    active_vocab_matrices: int = 1
    router_bias: bool = False
    expert_bias: bool = False
    dense_bias: bool = False
    sink_params_per_layer: int = 0
    extra_total_parameters: int = 0
    extra_active_parameters: int = 0

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "hidden_size", "layers", "attention_params_per_layer",
            "routed_experts", "experts_per_token", "expert_intermediate",
            "vocab_matrices_total",
        ):
            _positive_int(name, getattr(self, name))
        for name in (
            "shared_experts", "dense_layers", "dense_intermediate",
            "active_vocab_matrices", "sink_params_per_layer",
            "extra_total_parameters", "extra_active_parameters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.experts_per_token > self.routed_experts:
            raise ValueError("experts_per_token cannot exceed routed_experts")
        if self.dense_layers > self.layers:
            raise ValueError("dense_layers cannot exceed layers")
        if self.dense_layers and self.dense_intermediate == 0:
            raise ValueError("dense_intermediate is required when dense_layers > 0")
        if self.active_vocab_matrices > self.vocab_matrices_total:
            raise ValueError("active_vocab_matrices cannot exceed vocab_matrices_total")


@dataclass(frozen=True)
class ParameterBreakdown:
    """Total and active counts under one declared convention."""

    total_parameters: int
    active_parameters: int
    vocab_total: int
    vocab_active: int
    attention: int
    routed_and_shared_experts_total: int
    routed_and_shared_experts_active: int
    dense_ffn: int
    router: int
    norms: int
    sinks: int
    extra_total: int
    extra_active: int

    @property
    def total_to_active_ratio(self) -> float:
        return self.total_parameters / self.active_parameters


def gqa_attention_parameters(
    *,
    hidden_size: int,
    query_heads: int,
    kv_heads: int,
    head_width: int,
    include_bias: bool = False,
) -> int:
    """Count Q/K/V/O projections for one GQA layer.

    With ``include_bias=True`` the count matches the fused QKV bias and output bias in the
    public gpt-oss reference implementation.
    """

    hidden = _positive_int("hidden_size", hidden_size)
    q_heads = _positive_int("query_heads", query_heads)
    k_heads = _positive_int("kv_heads", kv_heads)
    width = _positive_int("head_width", head_width)
    if k_heads > q_heads:
        raise ValueError("kv_heads cannot exceed query_heads")
    q_dim = q_heads * width
    kv_dim = k_heads * width
    weights = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    biases = (q_dim + 2 * kv_dim + hidden) if include_bias else 0
    return int(weights + biases)


def deepseek_v3_mla_attention_parameters() -> int:
    """Course reconstruction of one DeepSeek-V3 MLA layer from public configuration fields.

    Projection biases are absent in the published config. The final two terms are the Q/latent
    normalization scales. This is an explicit reconstruction, not a claim that the public 671B
    headline follows a unique counting convention.
    """

    hidden, q_lora, kv_lora, rope, heads, head_width = 7168, 1536, 512, 64, 128, 128
    return int(
        hidden * q_lora
        + q_lora * (heads * (head_width + rope))
        + hidden * (kv_lora + rope)
        + kv_lora * (heads * (head_width + head_width))
        + heads * head_width * hidden
        + q_lora
        + kv_lora
    )


def expert_parameters(*, hidden_size: int, intermediate_size: int, include_bias: bool) -> int:
    """Count one SwiGLU expert with separate gate/up/down matrices."""

    hidden = _positive_int("hidden_size", hidden_size)
    intermediate = _positive_int("intermediate_size", intermediate_size)
    weights = 3 * hidden * intermediate
    biases = (2 * intermediate + hidden) if include_bias else 0
    return int(weights + biases)


def count_moe_parameters(spec: MoEModelSpec) -> ParameterBreakdown:
    """Count a model exactly under ``spec`` and expose every convention."""

    hidden = spec.hidden_size
    moe_layers = spec.layers - spec.dense_layers
    vocab_unit = spec.vocab_size * hidden
    vocab_total = spec.vocab_matrices_total * vocab_unit
    vocab_active = spec.active_vocab_matrices * vocab_unit
    attention = spec.attention_params_per_layer * spec.layers

    expert = expert_parameters(
        hidden_size=hidden,
        intermediate_size=spec.expert_intermediate,
        include_bias=spec.expert_bias,
    )
    expert_total = expert * (spec.routed_experts + spec.shared_experts) * moe_layers
    expert_active = expert * (spec.experts_per_token + spec.shared_experts) * moe_layers

    dense_unit = 3 * hidden * spec.dense_intermediate
    if spec.dense_bias:
        dense_unit += 2 * spec.dense_intermediate + hidden
    dense_ffn = dense_unit * spec.dense_layers

    router_per_layer = hidden * spec.routed_experts
    if spec.router_bias:
        router_per_layer += spec.routed_experts
    router = router_per_layer * moe_layers
    norms = 2 * hidden * spec.layers + hidden
    sinks = spec.sink_params_per_layer * spec.layers

    total = (
        vocab_total + attention + expert_total + dense_ffn + router + norms + sinks
        + spec.extra_total_parameters
    )
    active = (
        vocab_active + attention + expert_active + dense_ffn + router + norms + sinks
        + spec.extra_active_parameters
    )
    return ParameterBreakdown(
        total_parameters=int(total),
        active_parameters=int(active),
        vocab_total=int(vocab_total),
        vocab_active=int(vocab_active),
        attention=int(attention),
        routed_and_shared_experts_total=int(expert_total),
        routed_and_shared_experts_active=int(expert_active),
        dense_ffn=int(dense_ffn),
        router=int(router),
        norms=int(norms),
        sinks=int(sinks),
        extra_total=int(spec.extra_total_parameters),
        extra_active=int(spec.extra_active_parameters),
    )


def bytes_to_gib(byte_count: int) -> float:
    """Convert a non-negative byte count to binary gibibytes."""

    if isinstance(byte_count, bool) or not isinstance(byte_count, Integral) or byte_count < 0:
        raise ValueError("byte_count must be a non-negative integer")
    return int(byte_count) / 2**30


def expected_active_expert_fraction(*, experts: int, top_k: int, token_count: int) -> float:
    """Expected touched fraction under independent uniform token-choice routing."""

    expert_count = _positive_int("experts", experts)
    k = _positive_int("top_k", top_k)
    tokens = _positive_int("token_count", token_count)
    if k > expert_count:
        raise ValueError("top_k cannot exceed experts")
    return float(1.0 - (1.0 - k / expert_count) ** tokens)


def simulate_active_expert_fraction(
    *, experts: int, top_k: int, token_count: int, trials: int, seed: int
) -> float:
    """Monte Carlo check of the uniform independent-routing law."""

    expert_count = _positive_int("experts", experts)
    k = _positive_int("top_k", top_k)
    tokens = _positive_int("token_count", token_count)
    repeats = _positive_int("trials", trials)
    if k > expert_count:
        raise ValueError("top_k cannot exceed experts")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    fractions = []
    for _ in range(repeats):
        selected = np.stack([rng.choice(expert_count, size=k, replace=False) for _ in range(tokens)])
        fractions.append(np.unique(selected).size / expert_count)
    return float(np.mean(fractions))


def expert_parallel_payload_bytes(
    *, token_count: int, hidden_size: int, top_k: int, bytes_per_value: int,
    include_return: bool = True,
) -> int:
    """Idealized activation payload; metadata, padding, and topology are excluded."""

    tokens = _positive_int("token_count", token_count)
    hidden = _positive_int("hidden_size", hidden_size)
    k = _positive_int("top_k", top_k)
    width = _positive_int("bytes_per_value", bytes_per_value)
    one_way = tokens * hidden * k * width
    return one_way * (2 if include_return else 1)


# END EDUCATIONAL CORE: accounting


# BEGIN EDUCATIONAL CORE: balancing

def simulate_aux_free_balancing(*, mode: Literal["sign", "proportional"]) -> dict[str, Any]:
    """Reproduce deterministic 32-expert balancing traces.

    ``sign`` mirrors the main paper rule with a fixed-size correction. ``proportional`` is the
    smoother teaching controller used to visualize convergence; the paper also studies a
    proportional variant in an expert-parallel setting.
    """

    rng = np.random.default_rng(6)
    experts, top_k, token_count = 32, 4, 8192
    weights = rng.standard_normal((64, experts)) * 0.35
    constant = np.zeros(experts)
    constant[:4] = 2.5
    inputs = rng.standard_normal((token_count, 64))
    all_scores = stable_sigmoid(inputs @ weights + constant)

    def loads(bias: np.ndarray, scores: np.ndarray) -> np.ndarray:
        selected = _deterministic_top_k(scores + bias[None, :], top_k)
        return np.bincount(selected.ravel(), minlength=experts) / (scores.shape[0] * top_k)

    bias = np.zeros(experts)
    before = loads(bias, all_scores)
    rate = 0.001 if mode == "sign" else 0.15
    for _ in range(400):
        sample = all_scores[rng.integers(0, token_count, size=1024)]
        fractions = loads(bias, sample)
        bias = aux_free_bias_step(bias, fractions, rate=rate, mode=mode)
    after = loads(bias, all_scores)
    return {
        "mode": mode,
        "rate": rate,
        "before": before.tolist(),
        "after": after.tolist(),
        "bias": bias.tolist(),
        "uniform": 1.0 / experts,
        "cv_before": float(before.std() / before.mean()),
        "cv_after": float(after.std() / after.mean()),
    }


# END EDUCATIONAL CORE: balancing

def make_reference_payload() -> dict[str, Any]:
    """Compute every deterministic value embedded in the bilingual notebooks."""

    logits = np.array([1.2, -0.3, 2.1, 0.4, 1.7, -1.0, 0.9, 2.5])
    base = route_logits(logits, top_k=4, scoring="sigmoid")
    bias = np.zeros(8)
    bias[3] = 0.35
    biased = route_logits(logits, top_k=4, scoring="sigmoid", selection_bias=bias)

    capacity_logits = np.array(
        [[4.0, 3.0, 0.0, -1.0], [4.0, 0.0, 3.0, -1.0], [4.0, 0.0, -1.0, 3.0]] * 4,
        dtype=np.float64,
    )
    capacity_router = route_logits(capacity_logits, top_k=2, scoring="softmax")
    capacity_plan = dispatch_with_capacity(capacity_router, capacity_factor=1.0)

    aux_logits = np.array(
        [[4.0, 1.0, 0.0, -1.0], [3.0, 1.0, 0.0, -1.0],
         [4.0, 0.0, 1.0, -1.0], [3.0, 0.0, 1.0, -1.0]],
        dtype=np.float64,
    )

    gqa_with_bias = gqa_attention_parameters(
        hidden_size=2880, query_heads=64, kv_heads=8, head_width=64, include_bias=True
    )
    oss120 = count_moe_parameters(
        MoEModelSpec(
            vocab_size=201_088, hidden_size=2880, layers=36,
            attention_params_per_layer=gqa_with_bias,
            routed_experts=128, experts_per_token=4, expert_intermediate=2880,
            vocab_matrices_total=2, active_vocab_matrices=1,
            router_bias=True, expert_bias=True, sink_params_per_layer=64,
        )
    )
    oss20 = count_moe_parameters(
        MoEModelSpec(
            vocab_size=201_088, hidden_size=2880, layers=24,
            attention_params_per_layer=gqa_with_bias,
            routed_experts=32, experts_per_token=4, expert_intermediate=2880,
            vocab_matrices_total=2, active_vocab_matrices=1,
            router_bias=True, expert_bias=True, sink_params_per_layer=64,
        )
    )

    mla = deepseek_v3_mla_attention_parameters()
    deepseek = count_moe_parameters(
        MoEModelSpec(
            vocab_size=129_280, hidden_size=7168, layers=61,
            attention_params_per_layer=mla,
            routed_experts=256, experts_per_token=8, expert_intermediate=2048,
            shared_experts=1, dense_layers=3, dense_intermediate=18_432,
            vocab_matrices_total=2, active_vocab_matrices=1,
            router_bias=True,
        )
    )

    sign_balance = simulate_aux_free_balancing(mode="sign")
    proportional_balance = simulate_aux_free_balancing(mode="proportional")
    active_law: dict[str, Any] = {}
    for experts, top_k in ((128, 4), (256, 8), (896, 16)):
        for tokens in (16, 64):
            key = f"E{experts}_k{top_k}_N{tokens}"
            active_law[key] = {
                "theory": expected_active_expert_fraction(
                    experts=experts, top_k=top_k, token_count=tokens
                ),
                "simulation": simulate_active_expert_fraction(
                    experts=experts, top_k=top_k, token_count=tokens,
                    trials=400, seed=7 + experts + tokens,
                ),
            }

    score0 = float(base.clean_scores[0, 0])
    score6 = float(base.clean_scores[0, 6])
    one_step = {"e0": score0 - 0.01875, "e6": score6 + 0.01875}
    two_steps = {"e0": score0 - 0.0375, "e6": score6 + 0.0375}

    payload = {
        "router": {
            "logits": logits.tolist(),
            "scores": base.clean_scores[0].tolist(),
            "selected": base.selected_experts[0].tolist(),
            "gates": base.gate_weights[0].tolist(),
            "bias": bias.tolist(),
            "selected_with_bias": biased.selected_experts[0].tolist(),
            "gates_with_bias": biased.gate_weights[0].tolist(),
            "boundary_update": {"one_step": one_step, "two_steps": two_steps},
        },
        "capacity": {
            "logits": capacity_logits.tolist(),
            "capacity_per_expert": capacity_plan.capacity_per_expert,
            "before_counts": capacity_router.expert_counts.tolist(),
            "after_counts": capacity_plan.expert_counts.tolist(),
            "dropped_assignments": capacity_plan.dropped_assignments,
            "fully_dropped_tokens": list(capacity_plan.fully_dropped_tokens),
            "policy": "fifo_drop_then_renormalize_survivors",
        },
        "losses": {
            "aux_logits": aux_logits.tolist(),
            "switch_aux": switch_auxiliary_loss(aux_logits, top_k=1),
            "z_loss": router_z_loss(aux_logits),
        },
        "balance": {"sign": sign_balance, "proportional": proportional_balance},
        "models": {
            "oss120": asdict(oss120),
            "oss20": asdict(oss20),
            "deepseek_v3_reconstruction": asdict(deepseek),
            "deepseek_v3_official_main_total": 671_000_000_000,
            "deepseek_v3_official_active": 37_000_000_000,
            "deepseek_v3_hf_package_total": 685_000_000_000,
            "mla_params_per_layer": mla,
            "gqa_params_per_layer_with_bias": gqa_with_bias,
        },
        "combinations": {
            "C8_2": math.comb(8, 2),
            "C128_4": math.comb(128, 4),
            "C256_8_unconstrained_upper_bound": math.comb(256, 8),
        },
        "active_expert_law": active_law,
        "latent_moe": {
            "kimi_k3_hidden": 7168,
            "kimi_k3_latent": 3584,
            "payload_reduction": 7168 / 3584,
            "routed_experts": 896,
            "selected_experts": 16,
            "shared_experts": 2,
        },
        "qwen35_config": {
            "hidden_size": 2048,
            "layers": 40,
            "routed_experts": 256,
            "selected_experts": 8,
            "shared_experts": 1,
            "routed_intermediate": 512,
            "shared_intermediate": 512,
            "mtp_layers": 1,
        },
        "expert_parallel": {
            "tokens": 2048,
            "hidden_size": 7168,
            "top_k": 8,
            "bytes_per_value": 2,
            "round_trip_bytes": expert_parallel_payload_bytes(
                token_count=2048, hidden_size=7168, top_k=8,
                bytes_per_value=2, include_return=True,
            ),
        },
    }
    return payload


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize the reference payload in the notebook's canonical compact form."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest used by notebook and builder consistency checks."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_reference(reference_path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Recompute and compare the stored reference payload exactly."""

    expected = make_reference_payload()
    stored = json.loads(reference_path.read_text(encoding="utf-8"))
    if canonical_json(stored) != canonical_json(expected):
        raise AssertionError("stored Module 7 reference payload differs from recomputation")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    """Validate by default; replace the frozen payload only with ``--write``."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="replace data/reference_metrics.json")
    args = parser.parse_args(argv)
    payload = make_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.write:
        REFERENCE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {REFERENCE_PATH}")
    else:
        validate_reference()
        print(f"validated {REFERENCE_PATH}")
    print(f"sha256 {payload_sha256(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
