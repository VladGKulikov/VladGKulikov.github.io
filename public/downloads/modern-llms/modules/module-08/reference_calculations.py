#!/usr/bin/env python3
"""Reference calculations and reusable educational core for Module 8.

This module owns the frozen numerical contract for the SSM/linear-attention notebook.
The learner notebook embeds reviewed regions from this file, so the same validated code
powers the answer sheet, builders, plots, and clean-directory notebook execution.

The implementation is deliberately small and explicit. It is educational code rather
than a fused selective-scan kernel or a production inference runtime.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from teaching_core import zoh_diagonal, delta_step

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation

def _real(name: str, value: Real) -> float:
    """Return one finite real scalar, rejecting booleans and non-finite values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_real(name: str, value: Real) -> float:
    """Return one finite strictly positive scalar."""

    number = _real(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _unit_interval(name: str, value: Real) -> float:
    """Return a finite scalar in the closed unit interval."""

    number = _real(name, value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def _positive_int(name: str, value: Integral) -> int:
    """Return one strictly positive integer, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative_int(name: str, value: Integral) -> int:
    """Return one non-negative integer, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _finite_array(
    name: str,
    value: Any,
    *,
    ndim: int | None = None,
    non_empty: bool = True,
) -> np.ndarray:
    """Return a float64 array with explicit dimensionality and finiteness checks."""

    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if non_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def bytes_to_kib(byte_count: Real) -> float:
    """Convert a finite non-negative byte count to binary KiB."""

    number = _real("byte_count", byte_count)
    if number < 0.0:
        raise ValueError("byte_count must be non-negative")
    return number / 2**10


def bytes_to_mib(byte_count: Real) -> float:
    """Convert a finite non-negative byte count to binary MiB."""

    number = _real("byte_count", byte_count)
    if number < 0.0:
        raise ValueError("byte_count must be non-negative")
    return number / 2**20


def bytes_to_gib(byte_count: Real) -> float:
    """Convert a finite non-negative byte count to binary GiB."""

    number = _real("byte_count", byte_count)
    if number < 0.0:
        raise ValueError("byte_count must be non-negative")
    return number / 2**30

# END EDUCATIONAL CORE: validation


# BEGIN EDUCATIONAL CORE: zoh

@dataclass(frozen=True)
class ZOHDiscretization:
    """Exact diagonal zero-order-hold discretization for one time step.

    ``transition`` has shape ``[d_state]`` and represents the diagonal of
    ``exp(delta*A)``. ``input_matrix`` has shape ``[d_state, d_input]``.
    """

    transition: np.ndarray
    input_matrix: np.ndarray


@dataclass(frozen=True)
class SSMScanResult:
    """States, outputs, and per-step discrete parameters of a diagonal SSM scan."""

    states: np.ndarray
    outputs: np.ndarray
    transitions: np.ndarray
    input_matrices: np.ndarray


def zoh_discretize_diagonal(
    continuous_a: Any,
    continuous_b: Any,
    delta: Real,
) -> ZOHDiscretization:
    """Discretize ``dh/dt = A h + B x`` for diagonal stable ``A``.

    Parameters
    ----------
    continuous_a:
        Diagonal entries of ``A`` with shape ``[d_state]``. Non-positive values are
        accepted; positive values are rejected because this lesson models stable modes.
    continuous_b:
        Input matrix with shape ``[d_state, d_input]``. A vector is treated as one input
        channel.
    delta:
        Strictly positive time step.

    Notes
    -----
    The exact input factor is ``expm1(delta*a)/a``. At ``a=0`` the continuous limit is
    exactly ``delta``; handling that limit explicitly avoids a hidden ``0/0``.
    """

    a = _finite_array("continuous_a", continuous_a, ndim=1)
    if np.any(a > 0.0):
        raise ValueError("continuous_a must be non-positive for the stable lesson model")
    b = _finite_array("continuous_b", continuous_b)
    if b.ndim == 1:
        b = b[:, None]
    if b.ndim != 2 or b.shape[0] != a.size:
        raise ValueError("continuous_b must have shape [d_state, d_input]")
    step = _positive_real("delta", delta)

    transition, input_matrix = zoh_diagonal(a, b, step)
    if not np.all(np.isfinite(input_matrix)):
        raise FloatingPointError("ZOH discretization produced a non-finite input matrix")
    return ZOHDiscretization(transition=transition, input_matrix=input_matrix)


def scan_diagonal_ssm(
    continuous_a: Any,
    continuous_b: Any,
    output_c: Any,
    inputs: Any,
    deltas: Any,
    *,
    initial_state: Any | None = None,
    exact_input_discretization: bool = True,
) -> SSMScanResult:
    """Run a selective diagonal SSM over a non-empty sequence.

    Shapes are ``A:[d_state]``, ``B:[d_state,d_input]``, ``C:[d_output,d_state]``,
    ``inputs:[T,d_input]``, and ``deltas:[T]``. A one-dimensional input sequence is
    interpreted as ``d_input=1``. The output includes every post-update state.

    Setting ``exact_input_discretization=False`` keeps the exact transition but replaces
    the ZOH input matrix by the common small-step approximation ``delta*B``.
    """

    a = _finite_array("continuous_a", continuous_a, ndim=1)
    b = _finite_array("continuous_b", continuous_b)
    if b.ndim == 1:
        b = b[:, None]
    c = _finite_array("output_c", output_c)
    if c.ndim == 1:
        c = c[None, :]
    x = _finite_array("inputs", inputs)
    if x.ndim == 1:
        x = x[:, None]
    dt = _finite_array("deltas", deltas, ndim=1)

    if b.ndim != 2 or b.shape[0] != a.size:
        raise ValueError("continuous_b must have shape [d_state, d_input]")
    if c.ndim != 2 or c.shape[1] != a.size:
        raise ValueError("output_c must have shape [d_output, d_state]")
    if x.ndim != 2 or x.shape[1] != b.shape[1]:
        raise ValueError("inputs must have shape [T, d_input]")
    if x.shape[0] != dt.size:
        raise ValueError("inputs and deltas must have the same sequence length")
    if np.any(dt <= 0.0):
        raise ValueError("every delta must be positive")
    if np.any(a > 0.0):
        raise ValueError("continuous_a must be non-positive for the stable lesson model")

    state = (
        np.zeros(a.size, dtype=np.float64)
        if initial_state is None
        else _finite_array("initial_state", initial_state, ndim=1).copy()
    )
    if state.shape != a.shape:
        raise ValueError("initial_state must have shape [d_state]")

    states = np.empty((x.shape[0], a.size), dtype=np.float64)
    outputs = np.empty((x.shape[0], c.shape[0]), dtype=np.float64)
    transitions = np.empty_like(states)
    input_matrices = np.empty((x.shape[0], a.size, b.shape[1]), dtype=np.float64)

    for index, (token, step) in enumerate(zip(x, dt)):
        discrete = zoh_discretize_diagonal(a, b, float(step))
        input_matrix = (
            discrete.input_matrix
            if exact_input_discretization
            else float(step) * b
        )
        state = discrete.transition * state + input_matrix @ token
        states[index] = state
        outputs[index] = c @ state
        transitions[index] = discrete.transition
        input_matrices[index] = input_matrix

    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(outputs)):
        raise FloatingPointError("SSM scan produced non-finite states or outputs")
    return SSMScanResult(states, outputs, transitions, input_matrices)

# END EDUCATIONAL CORE: zoh


# BEGIN EDUCATIONAL CORE: affine_scan

@dataclass(frozen=True)
class AffineTransform:
    """Elementwise affine state transform ``h_out = multiplier*h_in + bias``."""

    multiplier: np.ndarray
    bias: np.ndarray


def compose_affine(later: AffineTransform, earlier: AffineTransform) -> AffineTransform:
    """Compose two diagonal affine transforms as ``later(earlier(h))``."""

    a2 = _finite_array("later.multiplier", later.multiplier, ndim=1)
    b2 = _finite_array("later.bias", later.bias, ndim=1)
    a1 = _finite_array("earlier.multiplier", earlier.multiplier, ndim=1)
    b1 = _finite_array("earlier.bias", earlier.bias, ndim=1)
    if a1.shape != b1.shape or a2.shape != b2.shape or a1.shape != a2.shape:
        raise ValueError("all affine components must share shape [d_state]")
    return AffineTransform(multiplier=a2 * a1, bias=a2 * b1 + b2)


def affine_scan_sequential(
    multipliers: Any,
    biases: Any,
    *,
    initial_state: Any | None = None,
) -> np.ndarray:
    """Evaluate ``h_t = a_t*h_{t-1} + b_t`` sequentially.

    ``multipliers`` and ``biases`` must both have shape ``[T, d_state]``. The returned
    array contains post-update states with the same shape.
    """

    a = _finite_array("multipliers", multipliers, ndim=2)
    b = _finite_array("biases", biases, ndim=2)
    if a.shape != b.shape:
        raise ValueError("multipliers and biases must have identical shape")
    state = (
        np.zeros(a.shape[1], dtype=np.float64)
        if initial_state is None
        else _finite_array("initial_state", initial_state, ndim=1).copy()
    )
    if state.shape != (a.shape[1],):
        raise ValueError("initial_state must have shape [d_state]")
    result = np.empty_like(b)
    for index in range(a.shape[0]):
        state = a[index] * state + b[index]
        result[index] = state
    return result


def affine_scan_chunked(
    multipliers: Any,
    biases: Any,
    *,
    chunk_size: Integral,
    initial_state: Any | None = None,
) -> np.ndarray:
    """Evaluate the same affine recurrence in non-overlapping chunks.

    Inside each chunk, the local zero-state recurrence and cumulative multipliers can be
    computed in parallel by a scan kernel. Only one boundary state crosses each chunk.
    This NumPy version keeps the mechanics transparent; it is not a fused GPU scan.
    """

    a = _finite_array("multipliers", multipliers, ndim=2)
    b = _finite_array("biases", biases, ndim=2)
    if a.shape != b.shape:
        raise ValueError("multipliers and biases must have identical shape")
    chunk = _positive_int("chunk_size", chunk_size)
    state = (
        np.zeros(a.shape[1], dtype=np.float64)
        if initial_state is None
        else _finite_array("initial_state", initial_state, ndim=1).copy()
    )
    if state.shape != (a.shape[1],):
        raise ValueError("initial_state must have shape [d_state]")

    result = np.empty_like(b)
    for start in range(0, a.shape[0], chunk):
        end = min(start + chunk, a.shape[0])
        local_a = a[start:end]
        local_b = b[start:end]

        # Zero-start local states carry all within-chunk contributions.
        zero_state = np.zeros(a.shape[1], dtype=np.float64)
        local_zero = np.empty_like(local_b)
        for offset in range(local_a.shape[0]):
            zero_state = local_a[offset] * zero_state + local_b[offset]
            local_zero[offset] = zero_state

        # The chunk-start state is multiplied by the cumulative transition to each row.
        prefix_multiplier = np.cumprod(local_a, axis=0)
        result[start:end] = local_zero + prefix_multiplier * state
        state = result[end - 1].copy()
    return result

# END EDUCATIONAL CORE: affine_scan


# BEGIN EDUCATIONAL CORE: linear_attention

def _validate_qkv(q: Any, k: Any, v: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a non-empty causal linear-attention triplet."""

    query = _finite_array("q", q, ndim=2)
    key = _finite_array("k", k, ndim=2)
    value = _finite_array("v", v, ndim=2)
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("q, k, and v must share sequence length T")
    if query.shape[1] != key.shape[1]:
        raise ValueError("q and k must share feature width d_key")
    return query, key, value


def linear_attention_sequential(q: Any, k: Any, v: Any) -> np.ndarray:
    """Causal additive linear attention with state shape ``[d_value, d_key]``."""

    query, key, value = _validate_qkv(q, k, v)
    state = np.zeros((value.shape[1], key.shape[1]), dtype=np.float64)
    output = np.empty((query.shape[0], value.shape[1]), dtype=np.float64)
    for index in range(query.shape[0]):
        state += np.outer(value[index], key[index])
        output[index] = state @ query[index]
    return output


def linear_attention_chunked(
    q: Any,
    k: Any,
    v: Any,
    *,
    chunk_size: Integral,
) -> np.ndarray:
    """Exact chunked form of the same causal additive linear attention.

    The inter-chunk term reads the carried state; the intra-chunk term is a masked
    matrix multiplication. The final partial chunk is supported explicitly.
    """

    query, key, value = _validate_qkv(q, k, v)
    chunk = _positive_int("chunk_size", chunk_size)
    state = np.zeros((value.shape[1], key.shape[1]), dtype=np.float64)
    output = np.empty((query.shape[0], value.shape[1]), dtype=np.float64)
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        q_chunk = query[start:end]
        k_chunk = key[start:end]
        v_chunk = value[start:end]
        inter = q_chunk @ state.T
        causal_scores = np.tril(q_chunk @ k_chunk.T)
        output[start:end] = inter + causal_scores @ v_chunk
        state += v_chunk.T @ k_chunk
    return output

# END EDUCATIONAL CORE: linear_attention


# BEGIN EDUCATIONAL CORE: delta

@dataclass(frozen=True)
class DeltaUpdateResult:
    """State and retrieval trace for one normalized-key delta-rule update."""

    state: np.ndarray
    normalized_key: np.ndarray
    retrieval_before_write: np.ndarray
    retrieval_after_write: np.ndarray


def normalize_key(key: Any) -> np.ndarray:
    """Return a unit-L2 key, rejecting the zero vector."""

    vector = _finite_array("key", key, ndim=1)
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("key must have non-zero norm")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def delta_update(
    state: Any,
    key: Any,
    value: Any,
    *,
    beta: Real,
    decay: Real = 1.0,
    normalize: bool = True,
) -> DeltaUpdateResult:
    """Apply a gated delta-rule write to associative state ``[d_value, d_key]``.

    With a unit key, the update is

    ``base = decay * state``
    ``new = base + beta * outer(value - base@key, key)``.

    Thus ``beta=1`` overwrites retrieval at that key exactly, while ``beta=0`` only
    applies the global decay. If ``normalize=False``, the caller must supply a unit key;
    the explicit check prevents a misleading "exact overwrite" claim for arbitrary norm.
    """

    matrix = _finite_array("state", state, ndim=2)
    raw_key = _finite_array("key", key, ndim=1)
    target = _finite_array("value", value, ndim=1)
    if matrix.shape != (target.size, raw_key.size):
        raise ValueError("state must have shape [d_value, d_key]")
    write_rate = _unit_interval("beta", beta)
    forget_rate = _unit_interval("decay", decay)
    unit_key = normalize_key(raw_key) if normalize else raw_key.copy()
    if not normalize and not np.isclose(np.linalg.norm(unit_key), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("key must have unit norm when normalize=False")

    base = forget_rate * matrix
    before = base @ unit_key
    updated = delta_step(matrix, unit_key, target, write_rate, forget_rate)
    after = updated @ unit_key
    return DeltaUpdateResult(updated, unit_key, before, after)

# END EDUCATIONAL CORE: delta


# BEGIN EDUCATIONAL CORE: recall

@dataclass(frozen=True)
class RecallCurve:
    """Point-retrieval MSE for additive associative memory at several loads."""

    stored_pairs: tuple[int, ...]
    mse: tuple[float, ...]
    key_width: int
    value_width: int
    seed: int


def associative_recall_curve(
    stored_pairs: Iterable[Integral],
    *,
    key_width: Integral,
    value_width: Integral | None = None,
    seed: Integral = 7,
) -> RecallCurve:
    """Measure interference in ``S = sum(v k^T)`` for normalized random keys."""

    widths = _positive_int("key_width", key_width)
    values = widths if value_width is None else _positive_int("value_width", value_width)
    random_seed = _non_negative_int("seed", seed)
    loads = tuple(_positive_int("stored_pairs item", item) for item in stored_pairs)
    if not loads:
        raise ValueError("stored_pairs must be non-empty")

    rng = np.random.default_rng(random_seed)
    errors: list[float] = []
    for count in loads:
        keys = rng.standard_normal((count, widths))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        targets = rng.standard_normal((count, values))
        state = targets.T @ keys
        retrieved = (state @ keys.T).T
        errors.append(float(np.mean((retrieved - targets) ** 2)))
    return RecallCurve(loads, tuple(errors), widths, values, random_seed)


def stable_softmax_rows(logits: Any) -> np.ndarray:
    """Compute row-wise softmax for a finite non-empty matrix."""

    scores = _finite_array("logits", logits, ndim=2)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    denominator = np.sum(weights, axis=1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("softmax denominator must be positive and finite")
    return weights / denominator


def softmax_addressing_mse(
    *,
    stored_pairs: Integral,
    key_width: Integral,
    value_width: Integral | None = None,
    inverse_temperature: Real = 20.0,
    seed: Integral = 107,
) -> float:
    """Return finite-temperature self-retrieval MSE from an explicit key/value table.

    This is a near-one-hot addressing contrast, not a claim that finite-temperature
    softmax is mathematically exact for arbitrary random keys.
    """

    count = _positive_int("stored_pairs", stored_pairs)
    width = _positive_int("key_width", key_width)
    values = width if value_width is None else _positive_int("value_width", value_width)
    temperature = _positive_real("inverse_temperature", inverse_temperature)
    random_seed = _non_negative_int("seed", seed)
    rng = np.random.default_rng(random_seed)
    keys = rng.standard_normal((count, width))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    targets = rng.standard_normal((count, values))
    weights = stable_softmax_rows(temperature * (keys @ keys.T))
    return float(np.mean((weights @ targets - targets) ** 2))

# END EDUCATIONAL CORE: recall


# BEGIN EDUCATIONAL CORE: memory

@dataclass(frozen=True)
class SSMStateGeometry:
    """Per-sequence inference-cache geometry for an SSM stack.

    A recurrent Mamba-style layer can retain both the main SSM state and the
    short causal-convolution state. ``conv_channels`` is explicit because
    Mamba-2 may cache x plus B/C channels rather than only ``inner_width``.
    """

    layers: int
    inner_width: int
    state_width: int
    conv_width: int = 0
    conv_channels: int = 0
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        _positive_int("layers", self.layers)
        _positive_int("inner_width", self.inner_width)
        _positive_int("state_width", self.state_width)
        _positive_int("bytes_per_value", self.bytes_per_value)
        _non_negative_int("conv_width", self.conv_width)
        _non_negative_int("conv_channels", self.conv_channels)
        if (self.conv_width == 0) != (self.conv_channels == 0):
            raise ValueError("conv_width and conv_channels must both be zero or both be positive")

    def ssm_bytes_per_sequence(self, *, batch_size: Integral = 1) -> int:
        """Return the main recurrent SSM-state payload."""

        batch = _positive_int("batch_size", batch_size)
        return batch * self.layers * self.inner_width * self.state_width * self.bytes_per_value

    def conv_bytes_per_sequence(self, *, batch_size: Integral = 1) -> int:
        """Return the short causal-convolution cache payload."""

        batch = _positive_int("batch_size", batch_size)
        return batch * self.layers * self.conv_channels * self.conv_width * self.bytes_per_value

    def bytes_per_sequence(self, *, batch_size: Integral = 1) -> int:
        """Return the complete fixed recurrent cache for one or more sequences."""

        return self.ssm_bytes_per_sequence(batch_size=batch_size) + self.conv_bytes_per_sequence(batch_size=batch_size)


@dataclass(frozen=True)
class KVCacheGeometry:
    """Per-token K/V geometry for a stack of attention layers."""

    layers: int
    kv_heads: int
    head_width: int
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        _positive_int("layers", self.layers)
        _positive_int("kv_heads", self.kv_heads)
        _positive_int("head_width", self.head_width)
        _positive_int("bytes_per_value", self.bytes_per_value)

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_width * self.bytes_per_value

    def total_bytes(self, context_tokens: Integral, *, batch_size: Integral = 1) -> int:
        tokens = _non_negative_int("context_tokens", context_tokens)
        batch = _positive_int("batch_size", batch_size)
        return batch * tokens * self.bytes_per_token


@dataclass(frozen=True)
class HybridMemoryEstimate:
    """Per-sequence KV and recurrent cache of a hybrid stack."""

    total_layers: int
    attention_layers: int
    ssm_layers: int
    context_tokens: int
    kv_bytes: int
    ssm_bytes: int
    conv_bytes: int
    recurrent_bytes: int
    total_bytes: int


def hybrid_memory_estimate(
    *,
    total_layers: Integral,
    attention_layers: Integral,
    context_tokens: Integral,
    inner_width: Integral,
    state_width: Integral,
    conv_width: Integral,
    conv_channels: Integral,
    kv_heads: Integral,
    head_width: Integral,
    bytes_per_value: Integral = 2,
) -> HybridMemoryEstimate:
    """Return KV plus complete recurrent-cache payload for one hybrid sequence."""

    total = _positive_int("total_layers", total_layers)
    if isinstance(attention_layers, (bool, np.bool_)) or not isinstance(attention_layers, Integral):
        raise TypeError("attention_layers must be an integer")
    attention = int(attention_layers)
    if attention < 0 or attention > total:
        raise ValueError("attention_layers must lie in [0, total_layers]")
    tokens = _non_negative_int("context_tokens", context_tokens)
    inner = _positive_int("inner_width", inner_width)
    state = _positive_int("state_width", state_width)
    conv = _non_negative_int("conv_width", conv_width)
    channels = _non_negative_int("conv_channels", conv_channels)
    if (conv == 0) != (channels == 0):
        raise ValueError("conv_width and conv_channels must both be zero or both be positive")
    heads = _positive_int("kv_heads", kv_heads)
    width = _positive_int("head_width", head_width)
    item_bytes = _positive_int("bytes_per_value", bytes_per_value)
    ssm_layers = total - attention

    kv_bytes = 0 if attention == 0 else KVCacheGeometry(attention, heads, width, item_bytes).total_bytes(tokens)
    if ssm_layers == 0:
        ssm_bytes = conv_bytes = recurrent_bytes = 0
    else:
        geometry = SSMStateGeometry(
            ssm_layers, inner, state, conv, channels, item_bytes
        )
        ssm_bytes = geometry.ssm_bytes_per_sequence()
        conv_bytes = geometry.conv_bytes_per_sequence()
        recurrent_bytes = geometry.bytes_per_sequence()
    return HybridMemoryEstimate(
        total_layers=total,
        attention_layers=attention,
        ssm_layers=ssm_layers,
        context_tokens=tokens,
        kv_bytes=kv_bytes,
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        recurrent_bytes=recurrent_bytes,
        total_bytes=kv_bytes + recurrent_bytes,
    )


@dataclass(frozen=True)
class DecodeFlopEstimate:
    """Arithmetic of the central token-mixing operators only.

    The numbers exclude projections, gating, convolution, FFN, weight traffic,
    communication and kernel efficiency. They must not be read as latency or
    end-to-end model speedups.
    """

    context_tokens: int
    attention_core_flops: int
    ssm_core_flops: int
    ratio: float


def decode_flop_estimate(
    *,
    context_tokens: Integral,
    model_width: Integral,
    inner_width: Integral,
    state_width: Integral,
) -> DecodeFlopEstimate:
    """Return the declared core-only scenario ``4*N*d`` versus ``5*d_inner*d_state``."""

    tokens = _positive_int("context_tokens", context_tokens)
    model = _positive_int("model_width", model_width)
    inner = _positive_int("inner_width", inner_width)
    state = _positive_int("state_width", state_width)
    attention = 4 * tokens * model
    # State update: two multiplies + one add; readout: multiply + add.
    # Leading-order counts omit one readout addition per channel.
    ssm = 5 * inner * state
    return DecodeFlopEstimate(tokens, attention, ssm, attention / ssm)

# END EDUCATIONAL CORE: memory


def build_reference_metrics() -> dict[str, Any]:
    """Build the deterministic numerical contract used across Module 8 artifacts."""

    a = np.array([-1.0, -2.0])
    b = np.ones((2, 1))
    c = np.ones((1, 2))
    inputs = np.array([[1.0], [-1.0], [2.0]])
    deltas = np.array([0.1, 1.0, 0.5])
    exact = scan_diagonal_ssm(a, b, c, inputs, deltas)
    approximate = scan_diagonal_ssm(
        a,
        b,
        c,
        inputs,
        deltas,
        exact_input_discretization=False,
    )
    b_errors = []
    for step in deltas:
        discrete = zoh_discretize_diagonal(a, b, float(step))
        approx = float(step) * b
        b_errors.append(float(abs(approx[0, 0] - discrete.input_matrix[0, 0]) / discrete.input_matrix[0, 0]))
    zero_mode = zoh_discretize_diagonal(np.array([0.0, -1.0]), np.ones((2, 1)), 0.25)

    rng = np.random.default_rng(11)
    affine_a = rng.uniform(0.70, 0.99, size=(53, 6))
    affine_b = rng.standard_normal((53, 6))
    affine_initial = rng.standard_normal(6)
    affine_seq = affine_scan_sequential(affine_a, affine_b, initial_state=affine_initial)
    affine_chunk = affine_scan_chunked(
        affine_a,
        affine_b,
        chunk_size=16,
        initial_state=affine_initial,
    )

    rng = np.random.default_rng(19)
    q = rng.standard_normal((50, 8))
    k = rng.standard_normal((50, 8))
    v = rng.standard_normal((50, 6))
    linear_seq = linear_attention_sequential(q, k, v)
    linear_chunk = linear_attention_chunked(q, k, v, chunk_size=16)

    key = np.array([1.0, 0.0])
    value_1 = np.array([1.0, 2.0])
    value_2 = np.array([3.0, 4.0])
    initial_delta_state = np.outer(value_1, key)
    half = delta_update(initial_delta_state, key, value_2, beta=0.5)
    full = delta_update(initial_delta_state, key, value_2, beta=1.0)
    gated = delta_update(initial_delta_state, key, value_2, beta=0.5, decay=0.8)
    additive = initial_delta_state + np.outer(value_2, key)

    recall = associative_recall_curve([4, 8, 16, 32, 64, 128], key_width=32, seed=7)
    softmax_mse = softmax_addressing_mse(
        stored_pairs=128,
        key_width=32,
        inverse_temperature=20.0,
        seed=107,
    )

    full_kv = KVCacheGeometry(layers=32, kv_heads=8, head_width=128, bytes_per_value=2)
    mamba1 = SSMStateGeometry(
        layers=32, inner_width=8192, state_width=16,
        conv_width=4, conv_channels=8192, bytes_per_value=2,
    )
    # In the open Mamba-2 implementation the convolution cache can contain x plus B/C channels.
    mamba2_like = SSMStateGeometry(
        layers=32, inner_width=8192, state_width=128,
        conv_width=4, conv_channels=8192 + 2 * 128, bytes_per_value=2,
    )
    hybrid = hybrid_memory_estimate(
        total_layers=32,
        attention_layers=4,
        context_tokens=256 * 1024,
        inner_width=8192,
        state_width=16,
        conv_width=4,
        conv_channels=8192,
        kv_heads=8,
        head_width=128,
        bytes_per_value=2,
    )
    flop_positions = [1024, 32768, 1_048_576]
    flops = [
        decode_flop_estimate(
            context_tokens=position,
            model_width=4096,
            inner_width=8192,
            state_width=16,
        )
        for position in flop_positions
    ]

    # Solve Delta/(1-exp(-Delta)) - 1 = 0.1 independently by bisection.
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        error = middle / (-math.expm1(-middle)) - 1.0
        if error < 0.1:
            low = middle
        else:
            high = middle
    ten_percent_root = (low + high) / 2

    paired_rng = np.random.default_rng(7)
    paired_softmax = []
    for count in recall.stored_pairs:
        keys = paired_rng.standard_normal((count, 32))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = paired_rng.standard_normal((count, 32))
        weights = stable_softmax_rows(20.0 * (keys @ keys.T))
        paired_softmax.append(float(np.mean((weights @ values - values)**2)))
    return {
        "schema_version": 2,
        "zoh": {
            "continuous_a": a.tolist(),
            "continuous_b": b.tolist(),
            "output_c": c.tolist(),
            "inputs": inputs.tolist(),
            "deltas": deltas.tolist(),
            "transitions": exact.transitions.tolist(),
            "exact_input_matrices": exact.input_matrices.tolist(),
            "exact_states": exact.states.tolist(),
            "exact_outputs": exact.outputs[:, 0].tolist(),
            "approximate_states": approximate.states.tolist(),
            "approximate_outputs": approximate.outputs[:, 0].tolist(),
            "relative_input_error_first_mode": b_errors,
            "zero_mode_input_matrix": zero_mode.input_matrix.tolist(),
            "ten_percent_relative_error_root_a_minus_1": ten_percent_root,
        },
        "affine_scan": {
            "seed": 11,
            "sequence_length": 53,
            "state_width": 6,
            "chunk_size": 16,
            "max_abs_difference": float(np.max(np.abs(affine_seq - affine_chunk))),
            "output_checksum": float(np.sum(affine_seq)),
        },
        "linear_attention": {
            "seed": 19,
            "sequence_length": 50,
            "key_width": 8,
            "value_width": 6,
            "chunk_size": 16,
            "max_abs_difference": float(np.max(np.abs(linear_seq - linear_chunk))),
            "output_checksum": float(np.sum(linear_seq)),
        },
        "delta": {
            "additive_retrieval": (additive @ key).tolist(),
            "half_retrieval": half.retrieval_after_write.tolist(),
            "full_retrieval": full.retrieval_after_write.tolist(),
            "gated_half_retrieval": gated.retrieval_after_write.tolist(),
        },
        "recall": {
            "seed": recall.seed,
            "key_width": recall.key_width,
            "value_width": recall.value_width,
            "stored_pairs": list(recall.stored_pairs),
            "mse": list(recall.mse),
            "paired_softmax_mse": paired_softmax,
            "expected_additive_mse": [(count - 1) / 32 for count in recall.stored_pairs],
            "softmax_seed": 107,
            "softmax_inverse_temperature": 20.0,
            "softmax_mse_at_128": softmax_mse,
        },
        "memory": {
            "kv_bytes_per_token_full_attention": full_kv.bytes_per_token,
            "mamba1": {
                "ssm_bytes": mamba1.ssm_bytes_per_sequence(),
                "conv_bytes": mamba1.conv_bytes_per_sequence(),
                "total_bytes": mamba1.bytes_per_sequence(),
                "first_strictly_larger_token": int(mamba1.bytes_per_sequence() // full_kv.bytes_per_token) + 1,
                "crossover_tokens_exact": mamba1.bytes_per_sequence() / full_kv.bytes_per_token,
                "crossover_first_integer_token": math.ceil(mamba1.bytes_per_sequence() / full_kv.bytes_per_token),
            },
            "mamba2_like": {
                "ssm_bytes": mamba2_like.ssm_bytes_per_sequence(),
                "conv_bytes": mamba2_like.conv_bytes_per_sequence(),
                "total_bytes": mamba2_like.bytes_per_sequence(),
                "first_strictly_larger_token": int(mamba2_like.bytes_per_sequence() // full_kv.bytes_per_token) + 1,
                "crossover_tokens_exact": mamba2_like.bytes_per_sequence() / full_kv.bytes_per_token,
                "crossover_first_integer_token": math.ceil(mamba2_like.bytes_per_sequence() / full_kv.bytes_per_token),
            },
            "full_attention_256k_bytes": full_kv.total_bytes(256 * 1024),
            "hybrid_256k": {
                "attention_layers": hybrid.attention_layers,
                "ssm_layers": hybrid.ssm_layers,
                "kv_bytes": hybrid.kv_bytes,
                "ssm_bytes": hybrid.ssm_bytes,
                "conv_bytes": hybrid.conv_bytes,
                "recurrent_bytes": hybrid.recurrent_bytes,
                "total_bytes": hybrid.total_bytes,
            },
        },
        "decode_flops": {
            "scope": "leading-order attention matmuls versus diagonal state update plus readout, with prepared coefficients; not end-to-end latency",
            "model_width": 4096, "inner_width": 8192, "state_width": 16,
            "positions": flop_positions,
            "attention_core_flops": [item.attention_core_flops for item in flops],
            "ssm_core_flops": [item.ssm_core_flops for item in flops],
            "ratios": [item.ratio for item in flops],
        },
    }


def _compare(expected: Any, actual: Any, path: str = "root") -> None:
    """Recursively compare reference structures with numerical tolerance."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"{path}: dictionary keys differ")
        for key in expected:
            _compare(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list shape differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise AssertionError(f"{path}: expected a number")
        if not math.isclose(float(expected), float(actual), rel_tol=1e-11, abs_tol=1e-12):
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Explicitly regenerate the frozen reference JSON."""

    payload = build_reference_metrics()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Validate the frozen JSON against freshly computed metrics without writing."""

    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_reference_metrics()
    _compare(expected, actual)
    return actual


def main(argv: list[str] | None = None) -> int:
    """CLI: validate by default; write only when ``--write`` is supplied."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="explicitly regenerate data/reference_metrics.json",
    )
    args = parser.parse_args(argv)
    if args.write:
        path = write_reference()
        print(f"wrote {path}")
    else:
        payload = validate_reference()
        print(
            "Module 8 reference contract validated: "
            f"ZOH steps={len(payload['zoh']['deltas'])}, "
            f"recall loads={payload['recall']['stored_pairs']}, "
            f"KV/token={payload['memory']['kv_bytes_per_token_full_attention']} B"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
