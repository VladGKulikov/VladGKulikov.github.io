#!/usr/bin/env python3
"""Language-neutral numerical contract for Module 6.

This module contains only small deterministic NumPy calculations used by maintainer
scripts. Learner notebooks embed the resulting JSON, so they do not depend on repository
files in Google Colab.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np


DepthMode = Literal["none", "pre", "post"]


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Return sigmoid(x) without overflow for large-magnitude inputs."""

    z = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z))


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Exact GELU x * Phi(x), evaluated with the scalar math.erf implementation."""

    z = np.asarray(x, dtype=np.float64)
    erf_values = np.fromiter(
        (math.erf(float(v) / math.sqrt(2.0)) for v in z.ravel()),
        dtype=np.float64,
        count=z.size,
    ).reshape(z.shape)
    return 0.5 * z * (1.0 + erf_values)


def gpt_oss_swiglu(
    gate: np.ndarray | float,
    up: np.ndarray | float,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:
    """Reference the public gpt-oss SwiGLU variant.

    The gate is clamped only from above, the linear branch from both sides, and the
    linear branch is shifted by +1 before multiplication.
    """

    if limit <= 0 or not math.isfinite(limit):
        raise ValueError("limit must be a finite positive number")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    g = np.minimum(np.asarray(gate, dtype=np.float64), limit)
    u = np.clip(np.asarray(up, dtype=np.float64), -limit, limit)
    return g * stable_sigmoid(alpha * g) * (u + 1.0)


def normalization_reference(eps: float = 1e-6) -> dict[str, object]:
    """Compute the lecture's LayerNorm/RMSNorm example in float64."""

    x = np.array([2.0, -1.0, 3.0, 0.0], dtype=np.float64)
    mean = float(x.mean())
    variance = float(x.var())
    rms_value = float(math.sqrt(float(np.mean(x * x)) + eps))
    layer = (x - mean) / math.sqrt(variance + eps)
    rms = x / rms_value
    return {
        "input": x.tolist(),
        "eps": eps,
        "mean": mean,
        "variance": variance,
        "rms_value": rms_value,
        "layer_norm": layer.tolist(),
        "rms_norm": rms.tolist(),
    }


@dataclass(frozen=True)
class DecoderShape:
    """Architecture fields needed for the exact dense-decoder parameter count."""

    vocab_size: int = 128_256
    d_model: int = 4_096
    n_layers: int = 32
    d_ff: int = 14_336
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
    tied_embeddings: bool = False


def parameter_breakdown(shape: DecoderShape) -> dict[str, int]:
    """Count bias-free GQA + SwiGLU decoder parameters, including norm scales."""

    q_width = shape.n_heads * shape.d_head
    kv_width = shape.n_kv_heads * shape.d_head
    attention = shape.d_model * q_width + q_width * shape.d_model
    attention += 2 * shape.d_model * kv_width
    mlp = 3 * shape.d_model * shape.d_ff
    norms = 2 * shape.d_model
    matrix_block = attention + mlp
    full_block = matrix_block + norms
    embedding_tables = (1 if shape.tied_embeddings else 2) * shape.vocab_size * shape.d_model
    final_norm = shape.d_model
    total = embedding_tables + shape.n_layers * full_block + final_norm
    return {
        "attention": int(attention),
        "mlp": int(mlp),
        "norms": int(norms),
        "matrix_block": int(matrix_block),
        "full_block": int(full_block),
        "block": int(full_block),
        "embedding_tables": int(embedding_tables),
        "final_norm": int(final_norm),
        "total": int(total),
    }


def scalar_rms(x: np.ndarray) -> float:
    """Return the float64 root-mean-square of an array."""

    z = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(z * z)))


def unit_rms(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Normalize an array to unit RMS for the placement-only toy experiment."""

    scale = max(scalar_rms(x), floor)
    return np.asarray(x, dtype=np.float64) / scale


def depth_problem(seed: int = 5, d_model: int = 256, n_layers: int = 36) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate one input and one shared weight stack for all depth regimes."""

    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(d_model)
    weights = [
        rng.standard_normal((d_model, d_model)) / math.sqrt(d_model)
        for _ in range(n_layers)
    ]
    return x0, weights


def simulate_depth(mode: DepthMode, x0: np.ndarray, weights: list[np.ndarray]) -> list[float]:
    """Run the same linear residual stack under none/pre/post RMS placement."""

    if mode not in {"none", "pre", "post"}:
        raise ValueError(f"unknown mode: {mode}")
    h = np.asarray(x0, dtype=np.float64).copy()
    curve: list[float] = []
    for weight in weights:
        if mode == "none":
            h = h + weight @ h
        elif mode == "pre":
            h = h + weight @ unit_rms(h)
        else:
            h = unit_rms(h + weight @ h)
        curve.append(scalar_rms(h))
    return curve


def build_reference_payload() -> dict[str, object]:
    """Return the complete frozen numerical payload embedded in the notebooks."""

    norm = normalization_reference()

    z_values = np.array([-9.0, -2.0, 0.0, 2.0, 7.0, 9.0], dtype=np.float64)
    oss_values = gpt_oss_swiglu(z_values, z_values)
    grid = np.linspace(-6.0, 6.0, 4001)
    gelu_max_error = float(
        np.max(np.abs(gelu_exact(grid) - grid * stable_sigmoid(1.702 * grid)))
    )

    shape = DecoderShape()
    params = parameter_breakdown(shape)

    x0, weights = depth_problem()
    curves = {mode: simulate_depth(mode, x0, weights) for mode in ("none", "pre", "post")}

    # The scalar surrogate used in the lecture to place the DyT curve. The max error is
    # intentionally not frozen because it is a sample-tail statistic.
    dyt_rng = np.random.default_rng(5)
    sample = dyt_rng.standard_normal(200_000)
    target = sample / scalar_rms(sample)
    alpha_grid = np.linspace(0.3, 2.0, 171)
    errors = [float(np.mean((np.tanh(alpha * sample) - target) ** 2)) for alpha in alpha_grid]
    best_index = int(np.argmin(errors))

    return {
        "schema_version": 3,
        "normalization": norm,
        "gpt_oss_swiglu": {
            "alpha": 1.702,
            "limit": 7.0,
            "diagonal_inputs": z_values.tolist(),
            "diagonal_outputs": oss_values.tolist(),
            "gelu_bridge_max_abs_error": gelu_max_error,
        },
        "llama_3_1_8b": {
            "shape": asdict(shape),
            "parameters": params,
            "matrix_block_over_d2": params["matrix_block"] / shape.d_model**2,
            "full_block_over_d2": params["full_block"] / shape.d_model**2,
        },
        "depth_experiment": {
            "seed": 5,
            "d_model": 256,
            "n_layers": 36,
            "shared_input_and_weights": True,
            "curves": curves,
        },
        "gpt_oss_block": {
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_experts": 128,
            "experts_per_token": 4,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "activation_defined_in_model_code": True,
        },
        "dyt_surrogate": {
            "seed": 5,
            "sample_size": 200_000,
            "alpha_grid_start": 0.3,
            "alpha_grid_stop": 2.0,
            "alpha_grid_steps": 171,
            "best_alpha": float(alpha_grid[best_index]),
            "best_mse": errors[best_index],
        },
    }
