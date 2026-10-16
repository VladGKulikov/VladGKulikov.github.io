#!/usr/bin/env python3
"""Reviewed numerical core and frozen reference data for Module 4.

The module owns its numerical fixtures. They are deliberately independent of lecture prose,
notebook outputs, and visualization files, all of which may change during later editorial
revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation

def _positive_int(name: str, value: object) -> int:
    """Return a strictly positive integer and reject booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_even_int(name: str, value: object, *, minimum: int = 2) -> int:
    """Return an even integer not smaller than ``minimum``."""

    result = _positive_int(name, value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if result % 2:
        raise ValueError(f"{name} must be even")
    return result


def _finite_real(name: str, value: object) -> float:
    """Return a finite real scalar and reject booleans and complex values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _real_greater_than(name: str, value: object, lower: float) -> float:
    """Return a finite scalar strictly greater than ``lower``."""

    result = _finite_real(name, value)
    if result <= lower:
        raise ValueError(f"{name} must be greater than {lower}")
    return result


def _finite_array(
    name: str,
    value: object,
    *,
    ndim: int | None = None,
    minimum_ndim: int | None = None,
) -> np.ndarray:
    """Convert to float64 and enforce rank, non-emptiness, and finite values."""

    try:
        if np.iscomplexobj(value):
            raise TypeError(f"{name} must be real")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if minimum_ndim is not None and array.ndim < minimum_ndim:
        raise ValueError(f"{name} must have rank at least {minimum_ndim}, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _position_vector(name: str, value: object) -> np.ndarray:
    """Validate a non-empty one-dimensional vector of non-negative integer positions."""

    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer positions")
    result = raw.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{name} must contain non-negative positions")
    return result


def _inverse_frequency_vector(value: object, *, expected_pairs: int | None = None) -> np.ndarray:
    """Validate a positive one-dimensional inverse-frequency vector."""

    inv = _finite_array("inverse_frequencies", value, ndim=1)
    if np.any(inv <= 0.0):
        raise ValueError("inverse_frequencies must be strictly positive")
    if expected_pairs is not None and inv.size != expected_pairs:
        raise ValueError(
            f"inverse_frequencies must have {expected_pairs} entries, got {inv.size}"
        )
    return inv


def _sequence_tensor(name: str, value: object) -> np.ndarray:
    """Validate a finite sequence tensor with shape ``[..., sequence, head_dim]``."""

    tensor = _finite_array(name, value, minimum_ndim=2)
    _positive_even_int(f"{name}.shape[-1]", tensor.shape[-1])
    if tensor.shape[-2] <= 0:
        raise ValueError(f"{name} must have a non-empty sequence axis")
    return tensor


# END EDUCATIONAL CORE: validation


# BEGIN EDUCATIONAL CORE: sinusoidal_and_spectrum

def rope_inverse_frequencies(head_dim: int, base: Real = 10_000.0) -> np.ndarray:
    """Return ``base**(-2i/head_dim)`` for the ``head_dim/2`` RoPE pairs."""

    dim = _positive_even_int("head_dim", head_dim)
    rope_base = _real_greater_than("base", base, 1.0)
    pair_indices = np.arange(dim // 2, dtype=np.float64)
    inv = rope_base ** (-2.0 * pair_indices / dim)
    if not np.all(np.isfinite(inv)) or np.any(inv <= 0.0):
        raise FloatingPointError("inverse-frequency construction produced invalid values")
    return inv


def sinusoidal_encoding(
    positions: object,
    model_dim: int,
    *,
    base: Real = 10_000.0,
) -> np.ndarray:
    """Return the classic sinusoidal encoding with shape ``[positions, model_dim]``."""

    pos = _position_vector("positions", positions)
    dim = _positive_even_int("model_dim", model_dim)
    inv = rope_inverse_frequencies(dim, base)
    angles = pos.astype(np.float64)[:, None] * inv[None, :]
    result = np.empty((pos.size, dim), dtype=np.float64)
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("sinusoidal encoding produced non-finite values")
    return result


def wavelengths_from_inverse_frequencies(inverse_frequencies: object) -> np.ndarray:
    """Convert radians-per-position frequencies to tokens per full ``2π`` turn."""

    inv = _inverse_frequency_vector(inverse_frequencies)
    return (2.0 * math.pi) / inv


def rotations_within_context(
    inverse_frequencies: object,
    context_length: int,
) -> np.ndarray:
    """Return turns (including fractions) across a span L = context_length.

    This is the YaRN L*theta/(2*pi) diagnostic. Positions 0..L-1 have a maximum
    endpoint separation of L-1, which is a different convention.
    """

    inv = _inverse_frequency_vector(inverse_frequencies)
    length = _positive_int("context_length", context_length)
    return length * inv / (2.0 * math.pi)


def full_turn_count(inverse_frequencies: object, context_length: int) -> int:
    """Count frequency pairs that complete at least one full turn in the context."""

    return int(np.count_nonzero(rotations_within_context(inverse_frequencies, context_length) >= 1.0))


# END EDUCATIONAL CORE: sinusoidal_and_spectrum


# BEGIN EDUCATIONAL CORE: rope

def rotate_half_split(x: object) -> np.ndarray:
    """Apply the split-half quarter turn used by HF/Llama-style RoPE implementations."""

    values = _sequence_tensor("x", x)
    half = values.shape[-1] // 2
    return np.concatenate((-values[..., half:], values[..., :half]), axis=-1)


def _angles_for_sequence(
    x: np.ndarray,
    positions: np.ndarray,
    inverse_frequencies: np.ndarray,
) -> np.ndarray:
    """Build a broadcastable angle tensor for ``[..., T, D]`` sequence tensors."""

    if positions.size != x.shape[-2]:
        raise ValueError(
            f"positions length {positions.size} does not match sequence axis {x.shape[-2]}"
        )
    if inverse_frequencies.size != x.shape[-1] // 2:
        raise ValueError("inverse_frequencies do not match head_dim")
    base_angles = positions.astype(np.float64)[:, None] * inverse_frequencies[None, :]
    return base_angles.reshape((1,) * (x.ndim - 2) + base_angles.shape)


def apply_rope_adjacent(
    x: object,
    positions: object,
    inverse_frequencies: object,
) -> np.ndarray:
    """Apply RoPE to adjacent pairs ``(x_0,x_1), (x_2,x_3), ...``."""

    values = _sequence_tensor("x", x)
    pos = _position_vector("positions", positions)
    inv = _inverse_frequency_vector(
        inverse_frequencies,
        expected_pairs=values.shape[-1] // 2,
    )
    angles = _angles_for_sequence(values, pos, inv)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    first = values[..., 0::2]
    second = values[..., 1::2]
    result = np.empty_like(values, dtype=np.float64)
    result[..., 0::2] = cosine * first - sine * second
    result[..., 1::2] = sine * first + cosine * second
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("adjacent-pair RoPE produced non-finite values")
    return result


def apply_rope_split_half(
    x: object,
    positions: object,
    inverse_frequencies: object,
) -> np.ndarray:
    """Apply split-half RoPE to pairs ``(x_j, x_{j+D/2})``."""

    values = _sequence_tensor("x", x)
    pos = _position_vector("positions", positions)
    inv = _inverse_frequency_vector(
        inverse_frequencies,
        expected_pairs=values.shape[-1] // 2,
    )
    angles = _angles_for_sequence(values, pos, inv)
    repeated = np.concatenate((angles, angles), axis=-1)
    result = values * np.cos(repeated) + rotate_half_split(values) * np.sin(repeated)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("split-half RoPE produced non-finite values")
    return result


def apply_rope_complex_adjacent(
    x: object,
    positions: object,
    inverse_frequencies: object,
) -> np.ndarray:
    """Apply adjacent-pair RoPE through the equivalent complex multiplication."""

    values = _sequence_tensor("x", x)
    pos = _position_vector("positions", positions)
    inv = _inverse_frequency_vector(
        inverse_frequencies,
        expected_pairs=values.shape[-1] // 2,
    )
    angles = _angles_for_sequence(values, pos, inv)
    complex_values = values[..., 0::2] + 1j * values[..., 1::2]
    rotated = complex_values * np.exp(1j * angles)
    result = np.empty_like(values, dtype=np.float64)
    result[..., 0::2] = rotated.real
    result[..., 1::2] = rotated.imag
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("complex RoPE produced non-finite values")
    return result


def adjacent_to_split_permutation(head_dim: int) -> np.ndarray:
    """Return the coordinate permutation mapping adjacent pairs to split-half pairs."""

    dim = _positive_even_int("head_dim", head_dim)
    return np.concatenate((np.arange(0, dim, 2), np.arange(1, dim, 2)))


def rope_attention_logits(
    queries: object,
    keys: object,
    query_positions: object,
    key_positions: object,
    inverse_frequencies: object,
    *,
    pairing: str = "split_half",
) -> np.ndarray:
    """Return unscaled rotated dot products ``[..., query, key]``.

    Divide by sqrt(head_dim) before the standard attention softmax. The historical
    function name and fixture key 'logits' are retained for file compatibility.
    """

    q = _sequence_tensor("queries", queries)
    k = _sequence_tensor("keys", keys)
    if q.ndim != k.ndim or q.shape[:-2] != k.shape[:-2]:
        raise ValueError("queries and keys must have identical batch dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("queries and keys must have the same head_dim")
    if pairing == "split_half":
        apply = apply_rope_split_half
    elif pairing == "adjacent":
        apply = apply_rope_adjacent
    else:
        raise ValueError("pairing must be 'split_half' or 'adjacent'")
    q_rot = apply(q, query_positions, inverse_frequencies)
    k_rot = apply(k, key_positions, inverse_frequencies)
    return np.matmul(q_rot, np.swapaxes(k_rot, -1, -2))


# END EDUCATIONAL CORE: rope


# BEGIN EDUCATIONAL CORE: scaling

def position_interpolated_frequencies(
    inverse_frequencies: object,
    factor: Real,
) -> np.ndarray:
    """Apply position interpolation by dividing every frequency by ``factor``."""

    inv = _inverse_frequency_vector(inverse_frequencies)
    scale = _finite_real("factor", factor)
    if scale < 1.0:
        raise ValueError("factor must be at least 1")
    return inv / scale


def ntk_scaled_base(head_dim: int, base: Real, factor: Real) -> float:
    """Return the static NTK-aware base ``base * factor**(d/(d-2))``."""

    dim = _positive_even_int("head_dim", head_dim, minimum=4)
    rope_base = _real_greater_than("base", base, 1.0)
    scale = _finite_real("factor", factor)
    if scale < 1.0:
        raise ValueError("factor must be at least 1")
    result = rope_base * scale ** (dim / (dim - 2))
    if not math.isfinite(result):
        raise FloatingPointError("NTK base is non-finite")
    return result


def ntk_scaled_frequencies(head_dim: int, base: Real, factor: Real) -> np.ndarray:
    """Return inverse frequencies generated from the static NTK-aware base."""

    return rope_inverse_frequencies(head_dim, ntk_scaled_base(head_dim, base, factor))


@dataclass(frozen=True)
class YarnSpectrum:
    """Piecewise YaRN spectrum plus diagnostic boundaries and scaling metadata."""

    inverse_frequencies: np.ndarray
    original_inverse_frequencies: np.ndarray
    ramp: np.ndarray
    low: float
    high: float
    truncate: bool
    kept_pairs: int
    transition_pairs: int
    interpolated_pairs: int
    attention_factor: float
    score_factor_if_applied_to_q_and_k: float


def yarn_scaled_frequencies(
    head_dim: int,
    base: Real,
    factor: Real,
    original_context: int,
    *,
    beta_fast: Real = 32.0,
    beta_slow: Real = 1.0,
    truncate: bool = False,
) -> YarnSpectrum:
    """Construct the lecture's piecewise YaRN frequency spectrum.

    ``ramp == 0`` keeps a frequency, ``ramp == 1`` divides it by ``factor``.
    With ``truncate=False`` this follows the continuous correction bounds used by the
    original gpt-oss implementation. ``truncate=True`` reproduces the integer-boundary
    option supported by the generic Hugging Face YaRN utility.
    """

    dim = _positive_even_int("head_dim", head_dim, minimum=4)
    rope_base = _real_greater_than("base", base, 1.0)
    scale = _finite_real("factor", factor)
    if scale < 1.0:
        raise ValueError("factor must be at least 1")
    context = _positive_int("original_context", original_context)
    fast = _real_greater_than("beta_fast", beta_fast, 0.0)
    slow = _real_greater_than("beta_slow", beta_slow, 0.0)
    if fast <= slow:
        raise ValueError("beta_fast must be greater than beta_slow")
    if not isinstance(truncate, bool):
        raise TypeError("truncate must be a bool")

    original = rope_inverse_frequencies(dim, rope_base)
    pair_count = dim // 2

    def correction_dimension(rotations: float) -> float:
        """Map a target turn count to the corresponding continuous pair index."""

        return dim * math.log(context / (rotations * 2.0 * math.pi)) / (2.0 * math.log(rope_base))

    low_raw = correction_dimension(fast)
    high_raw = correction_dimension(slow)
    low = math.floor(low_raw) if truncate else low_raw
    high = math.ceil(high_raw) if truncate else high_raw
    low = max(0.0, min(float(pair_count - 1), float(low)))
    high = max(0.0, min(float(pair_count - 1), float(high)))
    if high <= low:
        raise ValueError("YaRN correction range is empty for this configuration")
    pair_indices = np.arange(pair_count, dtype=np.float64)
    ramp = np.clip((pair_indices - low) / (high - low), 0.0, 1.0)
    scaled = original * (1.0 - ramp) + (original / scale) * ramp
    attention_factor = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0
    score_factor_if_applied_to_q_and_k = attention_factor**2
    kept = int(np.count_nonzero(ramp == 0.0))
    interpolated = int(np.count_nonzero(ramp == 1.0))
    transition = pair_count - kept - interpolated
    return YarnSpectrum(
        inverse_frequencies=scaled,
        original_inverse_frequencies=original,
        ramp=ramp,
        low=low,
        high=high,
        truncate=truncate,
        kept_pairs=kept,
        transition_pairs=transition,
        interpolated_pairs=interpolated,
        attention_factor=attention_factor,
        score_factor_if_applied_to_q_and_k=score_factor_if_applied_to_q_and_k,
    )


# END EDUCATIONAL CORE: scaling


# BEGIN EDUCATIONAL CORE: alibi_and_mrope

def alibi_bias(
    slopes: object,
    query_positions: object,
    key_positions: object,
    *,
    causal: bool = True,
) -> np.ndarray:
    """Return an ALiBi tensor with shape ``[heads, query, key]``.

    In causal mode, future keys receive ``-inf`` and allowed keys receive
    ``-slope * (query_position - key_position)``. In bidirectional mode the absolute
    distance is used.
    """

    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    slope_values = _finite_array("slopes", slopes, ndim=1)
    if np.any(slope_values <= 0.0):
        raise ValueError("slopes must be strictly positive")
    q_pos = _position_vector("query_positions", query_positions)
    k_pos = _position_vector("key_positions", key_positions)
    delta = q_pos[:, None] - k_pos[None, :]
    if causal:
        allowed = delta >= 0
        base = -np.abs(delta).astype(np.float64)
        bias = slope_values[:, None, None] * base[None, :, :]
        bias = np.where(allowed[None, :, :], bias, -np.inf)
    else:
        bias = -slope_values[:, None, None] * np.abs(delta)[None, :, :]
    return bias


def apply_mrope_adjacent(
    x: object,
    coordinates: object,
    axis_sections: Sequence[int],
    inverse_frequencies: object,
) -> np.ndarray:
    """Apply adjacent-pair multi-axis RoPE.

    ``coordinates`` has shape ``[sequence, axes]``. ``axis_sections`` gives the number of
    frequency pairs assigned to each axis and must sum to ``head_dim / 2``.
    """

    values = _sequence_tensor("x", x)
    raw_coordinates = np.asarray(coordinates)
    if raw_coordinates.ndim != 2 or raw_coordinates.shape[0] != values.shape[-2]:
        raise ValueError("coordinates must have shape [sequence, axes]")
    if raw_coordinates.dtype.kind not in "iu":
        raise TypeError("coordinates must contain integer positions")
    coords = raw_coordinates.astype(np.int64, copy=False)
    if np.any(coords < 0):
        raise ValueError("coordinates must be non-negative")

    if isinstance(axis_sections, (str, bytes)):
        raise TypeError("axis_sections must be a sequence of integers")
    sections: list[int] = []
    for index, value in enumerate(axis_sections):
        sections.append(_positive_int(f"axis_sections[{index}]", value))
    if len(sections) != coords.shape[1]:
        raise ValueError("axis_sections length must match the coordinate axis count")
    pair_count = values.shape[-1] // 2
    if sum(sections) != pair_count:
        raise ValueError("axis_sections must sum to head_dim / 2")

    inv = _inverse_frequency_vector(inverse_frequencies, expected_pairs=pair_count)
    axis_index = np.repeat(np.arange(len(sections)), sections)
    base_angles = coords[:, axis_index].astype(np.float64) * inv[None, :]
    angles = base_angles.reshape((1,) * (values.ndim - 2) + base_angles.shape)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    first = values[..., 0::2]
    second = values[..., 1::2]
    result = np.empty_like(values, dtype=np.float64)
    result[..., 0::2] = cosine * first - sine * second
    result[..., 1::2] = sine * first + cosine * second
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("M-RoPE produced non-finite values")
    return result


# END EDUCATIONAL CORE: alibi_and_mrope


def _canonical_json(payload: dict[str, Any]) -> str:
    """Serialize reference data deterministically and reject non-standard JSON numbers."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_reference_payload() -> dict[str, Any]:
    """Compute the frozen numerical fixtures used by notebooks and figures."""

    sinusoidal_positions = np.arange(4, dtype=np.int64)
    sinusoidal = sinusoidal_encoding(sinusoidal_positions, 8, base=10_000.0)

    q = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float64)[None, :]
    k = np.array([0.0, 1.0, 1.0, -0.5], dtype=np.float64)[None, :]
    inv4 = rope_inverse_frequencies(4, 10_000.0)
    placements = ((3, 1), (7, 5), (2, 0))
    logits = []
    for q_pos, k_pos in placements:
        value = rope_attention_logits(q, k, [q_pos], [k_pos], inv4, pairing="adjacent")
        logits.append(float(value[0, 0]))
    rotated_q = apply_rope_adjacent(q, [3], inv4)[0]
    rotated_k = apply_rope_adjacent(k, [1], inv4)[0]

    head_dim = 64
    contexts = (4_096, 131_072)
    bases: dict[str, Any] = {}
    for base in (10_000.0, 150_000.0, 500_000.0):
        inv = rope_inverse_frequencies(head_dim, base)
        wavelengths = wavelengths_from_inverse_frequencies(inv)
        bases[str(int(base))] = {
            "inverse_frequencies": inv.tolist(),
            "wavelengths": wavelengths.tolist(),
            "minimum_wavelength_tokens": float(wavelengths[0]),
            "maximum_wavelength_tokens": float(wavelengths[-1]),
            "full_turn_counts": {
                str(context): full_turn_count(inv, context) for context in contexts
            },
        }

    scale_factor = 32.0
    base = 150_000.0
    original_context = 4_096
    raw = rope_inverse_frequencies(head_dim, base)
    pi = position_interpolated_frequencies(raw, scale_factor)
    ntk_base = ntk_scaled_base(head_dim, base, scale_factor)
    ntk = ntk_scaled_frequencies(head_dim, base, scale_factor)
    beta_fast = 32.0
    beta_slow = 1.0
    yarn = yarn_scaled_frequencies(
        head_dim,
        base,
        scale_factor,
        original_context,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        truncate=False,
    )

    slopes = np.array([1.0, 0.5], dtype=np.float64)
    alibi = alibi_bias(slopes, np.arange(4), np.arange(4), causal=True)
    alibi_allowed = [
        [
            [float(value) if math.isfinite(float(value)) else None for value in row]
            for row in head
        ]
        for head in alibi
    ]

    mrope_dim = 8
    mrope_inv = rope_inverse_frequencies(mrope_dim, 10_000.0)
    image_coordinates = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    image_input = np.arange(32, dtype=np.float64).reshape(4, 8) / 10.0
    image_output = apply_mrope_adjacent(
        image_input, image_coordinates, (2, 1, 1), mrope_inv
    )

    text_positions = np.array([4, 5, 6], dtype=np.int64)
    text_coordinates = np.repeat(text_positions[:, None], 3, axis=1)
    text_input = np.arange(24, dtype=np.float64).reshape(3, 8) / 10.0
    text_output = apply_mrope_adjacent(
        text_input, text_coordinates, (2, 1, 1), mrope_inv
    )
    text_standard_rope_output = apply_rope_adjacent(
        text_input, text_positions, mrope_inv
    )
    if not np.allclose(text_output, text_standard_rope_output, rtol=0.0, atol=1e-12):
        raise AssertionError("text M-RoPE must reduce to ordinary 1D RoPE")

    payload: dict[str, Any] = {
        "meta": {
            "module": "module-04",
            "version": 2,
            "description": "Frozen numerical contract independent of lecture prose and figures",
        },
        "sinusoidal": {
            "positions": sinusoidal_positions.tolist(),
            "model_dim": 8,
            "base": 10_000.0,
            "encoding": sinusoidal.tolist(),
        },
        "worked_rope": {
            "q": q[0].tolist(),
            "k": k[0].tolist(),
            "head_dim": 4,
            "base": 10_000.0,
            "placements": [list(item) for item in placements],
            "logits": logits,
            "rotated_q_at_3": rotated_q.tolist(),
            "rotated_k_at_1": rotated_k.tolist(),
        },
        "spectrum": {
            "head_dim": head_dim,
            "contexts": list(contexts),
            "bases": bases,
        },
        "scaling": {
            "head_dim": head_dim,
            "base": base,
            "factor": scale_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "original_context": original_context,
            "extended_context": int(original_context * scale_factor),
            "raw": raw.tolist(),
            "position_interpolation": pi.tolist(),
            "ntk_base": ntk_base,
            "ntk": ntk.tolist(),
            "yarn": {
                "inverse_frequencies": yarn.inverse_frequencies.tolist(),
                "ramp": yarn.ramp.tolist(),
                "low": yarn.low,
                "high": yarn.high,
                "truncate": yarn.truncate,
                "hf_truncated_low": math.floor(yarn.low),
                "hf_truncated_high": math.ceil(yarn.high),
                "kept_pairs": yarn.kept_pairs,
                "transition_pairs": yarn.transition_pairs,
                "interpolated_pairs": yarn.interpolated_pairs,
                "attention_factor": yarn.attention_factor,
                "score_factor_if_applied_to_q_and_k": yarn.score_factor_if_applied_to_q_and_k,
            },
        },
        "alibi": {
            "slopes": slopes.tolist(),
            "positions": list(range(4)),
            "allowed_bias_with_null_for_future": alibi_allowed,
        },
        "mrope": {
            "head_dim": mrope_dim,
            "base": 10_000.0,
            "axis_sections": [2, 1, 1],
            "image_coordinates": image_coordinates.tolist(),
            "image_input": image_input.tolist(),
            "image_output": image_output.tolist(),
            "text_positions": text_positions.tolist(),
            "text_coordinates": text_coordinates.tolist(),
            "text_input": text_input.tolist(),
            "text_output": text_output.tolist(),
            "text_standard_rope_output": text_standard_rope_output.tolist(),
            "text_matches_standard_1d": True,
        },
    }
    _canonical_json(payload)
    return payload


def validate_reference(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the frozen structure and float64 values across numerical libraries."""

    expected = build_reference_payload()
    if payload is None:
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(MODULE_ROOT.parents[1] / "tools"))
    from numeric_contract import assert_contract
    assert_contract(payload, expected)
    return expected


def write_reference() -> str:
    """Write the canonical module-level JSON and return its SHA-256 digest."""

    payload = build_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pretty = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    REFERENCE_PATH.write_text(pretty + "\n", encoding="utf-8")
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _print_summary(payload: dict[str, Any]) -> None:
    """Print a compact maintainer-facing summary of the frozen calculations."""

    worked = payload["worked_rope"]
    scaling = payload["scaling"]
    yarn = scaling["yarn"]
    spectrum = payload["spectrum"]
    print("Module 4 reference validation: PASS")
    print("worked RoPE logits:", ", ".join(f"{value:.6f}" for value in worked["logits"]))
    for base, data in spectrum["bases"].items():
        counts = data["full_turn_counts"]
        print(
            f"base={base}: lambda_min={data['minimum_wavelength_tokens']:.6f}, "
            f"lambda_max={data['maximum_wavelength_tokens']:.3f}, "
            f"turn counts={counts}"
        )
    print(
        "YaRN:",
        f"low={yarn['low']}",
        f"high={yarn['high']}",
        f"zones={yarn['kept_pairs']}/{yarn['transition_pairs']}/{yarn['interpolated_pairs']}",
        f"attention_factor={yarn['attention_factor']:.6f}",
        f"score_factor={yarn['score_factor_if_applied_to_q_and_k']:.6f}",
    )
    print(f"NTK base={scaling['ntk_base']:.6f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the committed payload or rewrite it only with explicit ``--write``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite data/reference_metrics.json from the reviewed calculation core",
    )
    args = parser.parse_args(argv)
    if args.write:
        digest = write_reference()
        print(f"wrote {REFERENCE_PATH}")
        print(f"canonical SHA-256: {digest}")
    payload = validate_reference()
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
