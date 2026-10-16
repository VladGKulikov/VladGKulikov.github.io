#!/usr/bin/env python3
"""Reference calculations and frozen teaching data for Module 11.

The module owns its numerical contract independently of the lecture text. Learner
notebooks embed the canonical JSON generated here and verify its SHA-256 digest, so a
later editorial change cannot silently change the speculative-decoding or quantization
examples.

The implementation is deliberately explicit rather than fast. It validates probability
vectors, handles zero draft mass and the ``p == q`` residual edge case, implements a
constant-distribution speculative-sampling cycle without unsafe division, and exposes a
small grouped E2M1 / symmetric INT4 teaching quantizer. The bandwidth model is a
transparent arithmetic ceiling, not a hardware benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"

E2M1_GRID = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=np.float64,
)


def _positive_int(name: str, value: int) -> int:
    """Validate and return a strictly positive, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: int) -> int:
    """Validate and return a non-negative, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_non_negative(name: str, value: float) -> float:
    """Validate and return a finite real number greater than or equal to zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _positive_finite(name: str, value: float) -> float:
    """Validate and return a strictly positive finite real number."""

    number = _finite_non_negative(name, value)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _probability(name: str, value: float) -> float:
    """Validate and return a finite probability in ``[0, 1]``."""

    number = _finite_non_negative(name, value)
    if number > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def probability_vector(name: str, values: Sequence[float], *, atol: float = 1e-12) -> np.ndarray:
    """Return a validated one-dimensional probability vector as ``float64``.

    The function rejects rather than silently normalizes malformed inputs. This is
    important in speculative decoding: a hidden renormalization can make the acceptance
    proof appear to pass while the model probabilities are already wrong.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must not contain negative probabilities")
    total = float(array.sum(dtype=np.float64))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name} must sum to 1 within atol={atol}; got {total}")
    return array.copy()


def validate_probability_pair(
    p_values: Sequence[float], q_values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return target and draft probability vectors with matching vocabulary size."""

    p = probability_vector("p", p_values)
    q = probability_vector("q", q_values)
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape; got {p.shape} and {q.shape}")
    return p, q


@dataclass(frozen=True)
class RejectionTrace:
    """One-token speculative-rejection decomposition for fixed ``p`` and ``q``."""

    p: np.ndarray
    q: np.ndarray
    acceptance_by_token: np.ndarray
    alpha: float
    residual: np.ndarray
    residual_mass: float
    total_variation: float
    balance: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the trace."""

        return {
            "p": self.p.tolist(),
            "q": self.q.tolist(),
            "acceptance_by_token": self.acceptance_by_token.tolist(),
            "alpha": self.alpha,
            "residual": self.residual.tolist(),
            "residual_mass": self.residual_mass,
            "total_variation": self.total_variation,
            "balance": self.balance.tolist(),
        }


def rejection_trace(p_values: Sequence[float], q_values: Sequence[float]) -> RejectionTrace:
    """Return the exact speculative-rejection balance for one target/draft pair.

    For entries with ``q[t] == 0`` the draft can never propose token ``t``. The returned
    per-token acceptance value is therefore set to zero by convention, while the missing
    target mass is recovered by the residual distribution. If ``p == q``, rejection has
    zero probability; the residual is the all-zero vector rather than ``0/0``.
    """

    p, q = validate_probability_pair(p_values, q_values)
    acceptance = np.zeros_like(p)
    support = q > 0.0
    acceptance[support] = np.minimum(1.0, p[support] / q[support])

    overlap = np.minimum(p, q)
    alpha = float(overlap.sum(dtype=np.float64))
    positive_difference = np.maximum(p - q, 0.0)
    residual_mass = float(positive_difference.sum(dtype=np.float64))
    expected_residual_mass = 1.0 - alpha
    if not math.isclose(residual_mass, expected_residual_mass, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("positive residual mass must equal 1 - alpha")

    residual = (
        positive_difference / residual_mass
        if residual_mass > 0.0
        else np.zeros_like(positive_difference)
    )
    balance = overlap + residual_mass * residual
    total_variation = 0.5 * float(np.abs(p - q).sum(dtype=np.float64))
    if not np.allclose(balance, p, atol=1e-12, rtol=0.0):
        raise AssertionError("rejection balance failed to reconstruct p")
    if not math.isclose(alpha, 1.0 - total_variation, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("alpha must equal 1 - total variation distance")

    return RejectionTrace(
        p=p,
        q=q,
        acceptance_by_token=acceptance,
        alpha=alpha,
        residual=residual,
        residual_mass=residual_mass,
        total_variation=total_variation,
        balance=balance,
    )


def expected_tokens_per_cycle(alpha: float, gamma: int) -> float:
    """Return expected emitted tokens for constant acceptance probability ``alpha``.

    ``gamma`` is the number of draft tokens. The implementation uses the finite sum
    ``1 + alpha + ... + alpha**gamma`` so the ``alpha == 1`` case is exact and never
    evaluates the removable ``0/0`` singularity of the closed form.
    """

    a = _probability("alpha", alpha)
    g = _non_negative_int("gamma", gamma)
    return math.fsum(a**i for i in range(g + 1))


def speculative_speedup(alpha: float, gamma: int, draft_cost: float) -> float:
    """Return the transparent ``E / (1 + gamma*c)`` speedup scenario."""

    g = _non_negative_int("gamma", gamma)
    c = _finite_non_negative("draft_cost", draft_cost)
    return expected_tokens_per_cycle(alpha, g) / (1.0 + g * c)


def optimal_gamma(alpha: float, draft_cost: float, gamma_max: int) -> tuple[int, float]:
    """Return the smallest integer draft length attaining the maximal scenario speedup."""

    maximum = _non_negative_int("gamma_max", gamma_max)
    candidates = [(g, speculative_speedup(alpha, g, draft_cost)) for g in range(maximum + 1)]
    best_value = max(value for _, value in candidates)
    best_gamma = min(g for g, value in candidates if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-15))
    return best_gamma, best_value


@dataclass(frozen=True)
class SpeculativeCycleResult:
    """Tokens and control-flow metadata emitted by one speculative cycle."""

    tokens: tuple[int, ...]
    accepted_draft_tokens: int
    rejected: bool
    target_batch_calls: int

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a cycle must emit at least one token")
        _non_negative_int("accepted_draft_tokens", self.accepted_draft_tokens)
        calls = _positive_int("target_batch_calls", self.target_batch_calls)
        if calls != 1:
            raise ValueError("one speculative cycle must use exactly one target batch call")


def verify_speculative_batch(
    draft_tokens: Sequence[int],
    draft_probabilities: Sequence[Sequence[float]],
    target_probabilities: Sequence[Sequence[float]],
    bonus_probabilities: Sequence[float],
    rng: np.random.Generator,
) -> SpeculativeCycleResult:
    """Verify one draft using probabilities returned by a single target batch call.

    Shapes are ``draft_tokens: [gamma]``, ``draft_probabilities: [gamma, vocab]``,
    ``target_probabilities: [gamma, vocab]`` and ``bonus_probabilities: [vocab]``.
    No model function is called here: the target rows are stored first and then reused
    by the left-to-right accept/reject loop.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if any(isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)) for token in draft_tokens):
        raise ValueError("draft tokens must be integers")
    tokens = tuple(int(token) for token in draft_tokens)
    if not tokens:
        raise ValueError("draft_tokens must be non-empty")
    q_rows = np.asarray(draft_probabilities, dtype=np.float64)
    p_rows = np.asarray(target_probabilities, dtype=np.float64)
    if q_rows.ndim != 2 or p_rows.ndim != 2 or q_rows.shape != p_rows.shape:
        raise ValueError("draft and target probability tables must share [gamma, vocab]")
    if q_rows.shape[0] != len(tokens):
        raise ValueError("probability-table gamma must match draft_tokens")
    q_valid = [probability_vector(f"q[{i}]", row) for i, row in enumerate(q_rows)]
    p_valid = [probability_vector(f"p[{i}]", row) for i, row in enumerate(p_rows)]
    bonus = probability_vector("bonus_probabilities", bonus_probabilities)
    if bonus.size != q_rows.shape[1]:
        raise ValueError("bonus vocabulary size must match the probability tables")

    if any(token < 0 or token >= bonus.size for token in tokens):
        raise ValueError("draft token outside vocabulary")
    if any(q_valid[i][token] <= 0 for i, token in enumerate(tokens)):
        raise ValueError("every proposal must have positive draft probability")
    output: list[int] = []
    accepted = 0
    for i, token in enumerate(tokens):
        if token < 0 or token >= bonus.size:
            raise ValueError(f"draft token {token} is outside the vocabulary")
        q_token = float(q_valid[i][token])
        if q_token <= 0.0:
            raise ValueError("a sampled draft token must have positive q probability")
        acceptance = min(1.0, float(p_valid[i][token]) / q_token)
        if float(rng.random()) < acceptance:
            output.append(token)
            accepted += 1
            continue
        trace = rejection_trace(p_valid[i], q_valid[i])
        if trace.residual_mass <= 0.0:
            raise AssertionError("rejection occurred although residual mass is zero")
        correction = int(rng.choice(bonus.size, p=trace.residual))
        output.append(correction)
        return SpeculativeCycleResult(tuple(output), accepted, True, 1)

    output.append(int(rng.choice(bonus.size, p=bonus)))
    return SpeculativeCycleResult(tuple(output), accepted, False, 1)


def speculative_cycle_constant(
    p_values: Sequence[float],
    q_values: Sequence[float],
    gamma: int,
    rng: np.random.Generator,
) -> SpeculativeCycleResult:
    """Run one exact speculative-sampling cycle for fixed target and draft distributions.

    This teaching helper isolates rejection-sampling correctness from autoregressive
    state. A real implementation obtains a different ``p_i`` and ``q_i`` at each
    position, but applies the same local acceptance and residual rule.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    g = _positive_int("gamma", gamma)
    trace = rejection_trace(p_values, q_values)
    p, q = trace.p, trace.q
    output: list[int] = []
    accepted = 0

    for _ in range(g):
        token = int(rng.choice(q.size, p=q))
        q_token = float(q[token])
        if q_token <= 0.0:
            raise AssertionError("numpy sampled a zero-probability draft token")
        acceptance = min(1.0, float(p[token]) / q_token)
        if float(rng.random()) < acceptance:
            output.append(token)
            accepted += 1
            continue

        if trace.residual_mass <= 0.0:
            raise AssertionError("rejection occurred although residual mass is zero")
        correction = int(rng.choice(p.size, p=trace.residual))
        output.append(correction)
        return SpeculativeCycleResult(tuple(output), accepted, True, 1)

    bonus = int(rng.choice(p.size, p=p))
    output.append(bonus)
    return SpeculativeCycleResult(tuple(output), accepted, False, 1)


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    """Return a non-empty finite one-dimensional ``float64`` vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class GroupedQuantization:
    """Dequantized values and one positive scale per fixed-size group."""

    values: np.ndarray
    scales: np.ndarray
    group_size: int


def quantize_grouped_e2m1(values: Sequence[float], group_size: int = 16) -> GroupedQuantization:
    """Quantize a vector with an educational grouped E2M1 value grid.

    The function models the value grid and one scale per group. It does not claim to
    reproduce every encoding detail, second-level scale, or hardware rounding rule of
    NVFP4/MXFP4. All-zero groups use scale 1 and remain exactly zero, avoiding ``0/0``.
    """

    x = _finite_vector("values", values)
    group = _positive_int("group_size", group_size)
    if x.size % group != 0:
        raise ValueError("values length must be divisible by group_size")
    matrix = x.reshape(-1, group)
    max_abs = np.max(np.abs(matrix), axis=1, keepdims=True)
    scales = np.where(max_abs > 0.0, max_abs / 6.0, 1.0)
    normalized = matrix / scales
    indices = np.argmin(np.abs(normalized[..., None] - E2M1_GRID), axis=-1)
    dequantized = E2M1_GRID[indices] * scales
    return GroupedQuantization(dequantized.reshape(x.shape), scales[:, 0], group)


def quantize_grouped_symmetric_int4(
    values: Sequence[float], group_size: int = 16
) -> GroupedQuantization:
    """Quantize a vector to symmetric signed INT4 levels ``[-7, 7]`` per group."""

    x = _finite_vector("values", values)
    group = _positive_int("group_size", group_size)
    if x.size % group != 0:
        raise ValueError("values length must be divisible by group_size")
    matrix = x.reshape(-1, group)
    max_abs = np.max(np.abs(matrix), axis=1, keepdims=True)
    scales = np.where(max_abs > 0.0, max_abs / 7.0, 1.0)
    codes = np.clip(np.rint(matrix / scales), -7.0, 7.0)
    dequantized = codes * scales
    return GroupedQuantization(dequantized.reshape(x.shape), scales[:, 0], group)


def mean_squared_error(reference: Sequence[float], approximation: Sequence[float]) -> float:
    """Return MSE after validating equal finite one-dimensional shapes."""

    ref = _finite_vector("reference", reference)
    app = _finite_vector("approximation", approximation)
    if ref.shape != app.shape:
        raise ValueError("reference and approximation must have the same shape")
    return float(np.mean(np.square(ref - app), dtype=np.float64))


def weight_payload_bytes(parameter_count: int, bits_per_weight: float) -> float:
    """Return raw weight payload bytes for a given effective bits-per-weight value."""

    parameters = _positive_int("parameter_count", parameter_count)
    bits = _positive_finite("bits_per_weight", bits_per_weight)
    return parameters * bits / 8.0


def memory_bound_ceiling_tokens_per_second(
    bandwidth_bytes_per_second: float, parameter_count: int, bits_per_weight: float
) -> float:
    """Return a weight-stream-only arithmetic token/s ceiling.

    Bandwidth is interpreted in decimal bytes/s. The denominator is the raw weight
    payload read once per generated token. KV traffic, activations, kernels, batching,
    cache effects, and runtime overhead are intentionally outside this scenario.
    """

    bandwidth = _positive_finite("bandwidth_bytes_per_second", bandwidth_bytes_per_second)
    return bandwidth / weight_payload_bytes(parameter_count, bits_per_weight)


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical compact UTF-8 JSON."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reference_payload() -> dict[str, Any]:
    """Build the canonical numerical contract for both Module 11 notebooks."""

    p = [0.40, 0.30, 0.15, 0.10, 0.05]
    q = [0.25, 0.35, 0.20, 0.10, 0.10]
    trace = rejection_trace(p, q)
    support_gap = rejection_trace([0.5, 0.5], [1.0, 0.0])
    identical = rejection_trace([0.55, 0.45], [0.55, 0.45])

    grid = {
        f"{alpha:.1f}": {
            str(gamma): expected_tokens_per_cycle(alpha, gamma)
            for gamma in (1, 3, 5, 7)
        }
        for alpha in (0.6, 0.7, 0.8, 0.9)
    }
    gamma_star, speedup_star = optimal_gamma(0.8, 0.1, 10)

    bandwidth = 3.35e12  # decimal B/s: 3.35 TB/s
    parameters = 8_000_000_000
    formats = {
        "bf16": 16.0,
        "FP8": 8.0,
        "NVFP4 (16 + E4M3)": 4.5,
        "MXFP4 (32 + E8M0)": 4.25,
        "ideal W4": 4.0,
    }
    format_rows: dict[str, Any] = {}
    for name, bits in formats.items():
        payload_bytes = weight_payload_bytes(parameters, bits)
        format_rows[name] = {
            "bits_per_weight": bits,
            "payload_bytes": payload_bytes,
            "payload_GB": payload_bytes / 1e9,
            "payload_GiB": payload_bytes / (2.0**30),
            "weight_stream_ceiling_tokens_per_second": (
                memory_bound_ceiling_tokens_per_second(bandwidth, parameters, bits)
            ),
        }

    rng = np.random.default_rng(7)
    vector = rng.standard_normal(16)
    vector[3] *= 8.0
    fp4 = quantize_grouped_e2m1(vector, group_size=16)
    int4 = quantize_grouped_symmetric_int4(vector, group_size=16)
    mse_fp4 = mean_squared_error(vector, fp4.values)
    mse_int4 = mean_squared_error(vector, int4.values)

    mtp_alpha, mtp_gamma, mtp_cost = 0.875, 1, 0.05
    combined = (
        speculative_speedup(mtp_alpha, mtp_gamma, mtp_cost)
        * format_rows["NVFP4 (16 + E4M3)"]["weight_stream_ceiling_tokens_per_second"]
    )

    return {
        "schema_version": 1,
        "trace": trace.as_dict(),
        "edge_cases": {
            "q_zero_support_gap": support_gap.as_dict(),
            "identical_distributions": identical.as_dict(),
        },
        "expected_tokens": {
            "grid": grid,
            "calibration_alpha_0_7_gamma_5": expected_tokens_per_cycle(0.7, 5),
            "trace_alpha_0_85_gamma_4": expected_tokens_per_cycle(0.85, 4),
            "alpha_one_gamma_4": expected_tokens_per_cycle(1.0, 4),
            "alpha_zero_gamma_4": expected_tokens_per_cycle(0.0, 4),
        },
        "speedup": {
            "external": {
                "alpha": 0.8,
                "draft_cost": 0.1,
                "gamma_star": gamma_star,
                "speedup_star": speedup_star,
                "speedup_gamma_5": speculative_speedup(0.8, 5, 0.1),
                "speedup_gamma_9": speculative_speedup(0.8, 9, 0.1),
            },
            "mtp_scenario": {
                "alpha": mtp_alpha,
                "gamma": mtp_gamma,
                "draft_cost": mtp_cost,
                "expected_tokens": expected_tokens_per_cycle(mtp_alpha, mtp_gamma),
                "speedup": speculative_speedup(mtp_alpha, mtp_gamma, mtp_cost),
            },
        },
        "monte_carlo": {
            "seed": 7,
            "cycles": 20_000,
            "gamma": 4,
            "distribution_atol": 0.01,
            "mean_length_atol": 0.04,
        },
        "roofline": {
            "bandwidth_bytes_per_second": bandwidth,
            "bandwidth_TB_per_second_decimal": bandwidth / 1e12,
            "parameter_count": parameters,
            "formats": format_rows,
            "speculative_bf16": {
                "alpha": 0.8,
                "gamma": 5,
                "draft_cost": 0.06,
                "ceiling_tokens_per_second": (
                    speculative_speedup(0.8, 5, 0.06)
                    * format_rows["bf16"]["weight_stream_ceiling_tokens_per_second"]
                ),
            },
            "combined_nvfp4_mtp_scenario_tokens_per_second": combined,
            "combined_scenario_vs_bf16": combined / format_rows["bf16"]["weight_stream_ceiling_tokens_per_second"],
        },
        "quantization": {
            "seed": 7,
            "group_size": 16,
            "vector": vector.tolist(),
            "e2m1_dequantized": fp4.values.tolist(),
            "int4_dequantized": int4.values.tolist(),
            "e2m1_scale": fp4.scales.tolist(),
            "int4_scale": int4.scales.tolist(),
            "mse_e2m1": mse_fp4,
            "mse_int4": mse_int4,
            "int4_over_e2m1_mse_ratio": mse_int4 / mse_fp4,
        },
    }


def load_reference_payload(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load the frozen JSON payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference_payload(payload: Mapping[str, Any]) -> None:
    """Recompute the canonical payload and require exact JSON-equivalent values."""

    expected = build_reference_payload()
    if payload != expected:
        raise AssertionError("reference_metrics.json differs from current calculations")


def main() -> None:
    """Validate the frozen payload, or regenerate it only with explicit ``--write``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="intentionally rewrite reference_metrics.json")
    args = parser.parse_args()

    payload = build_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.write:
        REFERENCE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {REFERENCE_PATH}")
    else:
        if not REFERENCE_PATH.exists():
            raise SystemExit(f"missing {REFERENCE_PATH}; run with --write after review")
        validate_reference_payload(load_reference_payload())
        print(f"validated {REFERENCE_PATH}")
    print(f"sha256={sha256_json(payload)}")


if __name__ == "__main__":
    main()
