#!/usr/bin/env python3
"""Reference calculations and frozen teaching data for Module 9.

Module 9 owns its numerical contract independently of mutable lecture prose.  The
RU/EN learner notebooks embed the canonical JSON generated here and verify its
SHA-256 digest, while the answer sheet is read-only unless ``--write`` is passed.

The code is intentionally explicit rather than optimized.  It validates all public
inputs, separates binary memory units from decimal compute rates, models the assumptions
behind scaling-law and wall-clock estimates, handles rectangular Newton--Schulz updates,
and distinguishes parameter correction in QK-Clip from merely clipping a displayed
logit value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation

def _real(name: str, value: float) -> float:
    """Return a finite, non-boolean real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_finite(name: str, value: float) -> float:
    """Return a strictly positive finite real number."""

    number = _real(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative_finite(name: str, value: float) -> float:
    """Return a non-negative finite real number."""

    number = _real(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_int(name: str, value: int) -> int:
    """Return a strictly positive, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: int) -> int:
    """Return a non-negative, non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _fraction(name: str, value: float, *, allow_zero: bool = True) -> float:
    """Return a finite fraction in ``[0, 1]`` or ``(0, 1]``."""

    number = _real(name, value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not lower_ok or number > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return number


def bytes_to_gib(byte_count: float) -> float:
    """Convert a finite non-negative byte count to binary GiB."""

    return _non_negative_finite("byte_count", byte_count) / 2**30


def bytes_to_decimal_gb(byte_count: float) -> float:
    """Convert a finite non-negative byte count to decimal GB."""

    return _non_negative_finite("byte_count", byte_count) / 10**9

# END EDUCATIONAL CORE: validation


# BEGIN EDUCATIONAL CORE: scaling

@dataclass(frozen=True)
class ComputeAllocation:
    """A model/data allocation under a fixed training-compute budget."""

    compute_flops: float
    model_parameters: float
    training_tokens: float
    predicted_loss: float

    @property
    def tokens_per_parameter(self) -> float:
        """Return the data-to-model ratio for this allocation."""

        return self.training_tokens / self.model_parameters


@dataclass(frozen=True)
class ScalingLaw:
    """Two-term Chinchilla-style loss fit with an explicit compute convention.

    ``compute_multiplier`` is the declared approximation in
    ``training_flops = compute_multiplier * N * D``.  It is a modelling convention,
    not a measured wall-clock quantity.
    """

    irreducible_loss: float
    model_coefficient: float
    data_coefficient: float
    model_exponent: float
    data_exponent: float
    compute_multiplier: float = 6.0

    def __post_init__(self) -> None:
        _non_negative_finite("irreducible_loss", self.irreducible_loss)
        _positive_finite("model_coefficient", self.model_coefficient)
        _positive_finite("data_coefficient", self.data_coefficient)
        _positive_finite("model_exponent", self.model_exponent)
        _positive_finite("data_exponent", self.data_exponent)
        _positive_finite("compute_multiplier", self.compute_multiplier)

    def loss(self, model_parameters: float, training_tokens: float) -> float:
        """Return the fitted loss for positive ``N`` and ``D``."""

        n = _positive_finite("model_parameters", model_parameters)
        d = _positive_finite("training_tokens", training_tokens)
        return (
            self.irreducible_loss
            + self.model_coefficient * n ** (-self.model_exponent)
            + self.data_coefficient * d ** (-self.data_exponent)
        )

    def training_flops(self, model_parameters: float, training_tokens: float) -> float:
        """Return the declared arithmetic training-FLOP estimate ``kappa*N*D``."""

        n = _positive_finite("model_parameters", model_parameters)
        d = _positive_finite("training_tokens", training_tokens)
        return self.compute_multiplier * n * d

    def optimal_split(self, compute_flops: float) -> ComputeAllocation:
        """Return the analytical compute-optimal ``N`` and ``D`` for this fit."""

        c = _positive_finite("compute_flops", compute_flops)
        alpha = self.model_exponent
        beta = self.data_exponent
        n = (
            (alpha * self.model_coefficient)
            / (beta * self.data_coefficient)
            * (c / self.compute_multiplier) ** beta
        ) ** (1.0 / (alpha + beta))
        d = c / (self.compute_multiplier * n)
        return ComputeAllocation(c, n, d, self.loss(n, d))

    def tokens_for_target_loss(self, model_parameters: float, target_loss: float) -> float:
        """Return tokens needed for ``target_loss`` at fixed model size.

        The fitted law has a model-size floor
        ``E + A*N**(-alpha)``.  A target at or below that floor is unattainable for
        finite data and is rejected explicitly rather than producing a complex value.
        """

        n = _positive_finite("model_parameters", model_parameters)
        target = _positive_finite("target_loss", target_loss)
        residual = (
            target
            - self.irreducible_loss
            - self.model_coefficient * n ** (-self.model_exponent)
        )
        if residual <= 0.0:
            raise ValueError("target_loss is below the fitted floor for this model size")
        return (self.data_coefficient / residual) ** (1.0 / self.data_exponent)


@dataclass(frozen=True)
class BreakEvenResult:
    """Training and inference arithmetic for two equal-quality model choices."""

    target_loss: float
    small_training_flops: float
    large_training_tokens: float
    large_training_flops: float
    training_flops_saved_by_large_model: float
    extra_inference_flops_per_token_for_large_model: float
    break_even_generated_tokens: float


def inference_break_even(
    law: ScalingLaw,
    *,
    small_model_parameters: float,
    small_training_tokens: float,
    large_model_parameters: float,
    inference_flop_multiplier: float = 2.0,
) -> BreakEvenResult:
    """Compare an overtrained small model with an equal-quality larger model.

    The function assumes an arithmetic inference cost of
    ``inference_flop_multiplier * N`` per generated token.  It requires the larger
    model to use less training compute and more inference compute; otherwise a positive
    break-even does not exist under the declared scenario.
    """

    n_small = _positive_finite("small_model_parameters", small_model_parameters)
    d_small = _positive_finite("small_training_tokens", small_training_tokens)
    n_large = _positive_finite("large_model_parameters", large_model_parameters)
    if n_large <= n_small:
        raise ValueError("large_model_parameters must exceed small_model_parameters")
    infer_k = _positive_finite("inference_flop_multiplier", inference_flop_multiplier)

    target = law.loss(n_small, d_small)
    d_large = law.tokens_for_target_loss(n_large, target)
    c_small = law.training_flops(n_small, d_small)
    c_large = law.training_flops(n_large, d_large)
    training_saved = c_small - c_large
    inference_delta = infer_k * (n_large - n_small)
    if training_saved <= 0.0:
        raise ValueError("the larger model does not save training FLOPs in this scenario")
    return BreakEvenResult(
        target_loss=target,
        small_training_flops=c_small,
        large_training_tokens=d_large,
        large_training_flops=c_large,
        training_flops_saved_by_large_model=training_saved,
        extra_inference_flops_per_token_for_large_model=inference_delta,
        break_even_generated_tokens=training_saved / inference_delta,
    )


@dataclass(frozen=True)
class TrainingRunEstimate:
    """A transparent arithmetic runtime estimate for one pretraining run."""

    model_parameters: float
    training_tokens: float
    accelerator_count: int
    peak_flops_per_accelerator: float
    model_flop_utilization: float
    total_training_flops: float
    effective_cluster_flops_per_second: float
    estimated_seconds: float
    estimated_days: float
    training_tokens_per_second: float


def estimate_training_run(
    law: ScalingLaw,
    *,
    model_parameters: float,
    training_tokens: float,
    accelerator_count: int,
    peak_flops_per_accelerator: float,
    model_flop_utilization: float,
) -> TrainingRunEstimate:
    """Return an arithmetic duration estimate under a declared MFU assumption.

    Peak FLOPs are decimal FLOP/s.  The estimate excludes downtime, checkpoint stalls,
    data-loader starvation, communication imbalance, failures, and warm-up unless their
    effect is already folded into the supplied MFU.
    """

    n = _positive_finite("model_parameters", model_parameters)
    d = _positive_finite("training_tokens", training_tokens)
    devices = _positive_int("accelerator_count", accelerator_count)
    peak = _positive_finite("peak_flops_per_accelerator", peak_flops_per_accelerator)
    mfu = _fraction("model_flop_utilization", model_flop_utilization, allow_zero=False)
    total = law.training_flops(n, d)
    effective_rate = devices * peak * mfu
    seconds = total / effective_rate
    return TrainingRunEstimate(
        model_parameters=n,
        training_tokens=d,
        accelerator_count=devices,
        peak_flops_per_accelerator=peak,
        model_flop_utilization=mfu,
        total_training_flops=total,
        effective_cluster_flops_per_second=effective_rate,
        estimated_seconds=seconds,
        estimated_days=seconds / 86_400.0,
        training_tokens_per_second=d / seconds,
    )

# END EDUCATIONAL CORE: scaling


# BEGIN EDUCATIONAL CORE: mixture

@dataclass(frozen=True)
class DataSlice:
    """One unique-token pool and its unnormalized sampling weight."""

    name: str
    unique_tokens: float
    sampling_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _positive_finite("unique_tokens", self.unique_tokens)
        _positive_finite("sampling_weight", self.sampling_weight)


@dataclass(frozen=True)
class MixtureAllocation:
    """Allocated tokens and effective epochs for one data slice."""

    name: str
    normalized_weight: float
    allocated_tokens: float
    unique_tokens: float
    effective_epochs: float


def allocate_data_mixture(
    slices: Sequence[DataSlice], total_training_tokens: float
) -> tuple[MixtureAllocation, ...]:
    """Allocate a token budget by normalized weights and report effective epochs."""

    total = _positive_finite("total_training_tokens", total_training_tokens)
    items = tuple(slices)
    if not items:
        raise ValueError("slices must be non-empty")
    if not all(isinstance(item, DataSlice) for item in items):
        raise TypeError("every slice must be a DataSlice")
    names = [item.name for item in items]
    if len(names) != len(set(names)):
        raise ValueError("slice names must be unique")
    weight_sum = math.fsum(item.sampling_weight for item in items)
    allocations = tuple(
        MixtureAllocation(
            name=item.name,
            normalized_weight=item.sampling_weight / weight_sum,
            allocated_tokens=total * item.sampling_weight / weight_sum,
            unique_tokens=item.unique_tokens,
            effective_epochs=(total * item.sampling_weight / weight_sum) / item.unique_tokens,
        )
        for item in items
    )
    if not math.isclose(
        math.fsum(item.allocated_tokens for item in allocations),
        total,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise AssertionError("mixture allocations do not sum to the requested token budget")
    return allocations

# END EDUCATIONAL CORE: mixture


# BEGIN EDUCATIONAL CORE: muon_qk

@dataclass(frozen=True)
class SpectralSummary:
    """Singular-value and orthogonality diagnostics for a matrix update."""

    minimum_singular_value: float
    maximum_singular_value: float
    condition_number: float
    orthogonality_error_frobenius: float


def spectral_summary(matrix: Sequence[Sequence[float]]) -> SpectralSummary:
    """Return diagnostics against the identity on the smaller matrix dimension."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    singular = np.linalg.svd(array, compute_uv=False)
    minimum = float(singular.min())
    maximum = float(singular.max())
    condition = math.inf if minimum == 0.0 else maximum / minimum
    if array.shape[0] >= array.shape[1]:
        gram = array.T @ array
        identity = np.eye(array.shape[1], dtype=np.float64)
    else:
        gram = array @ array.T
        identity = np.eye(array.shape[0], dtype=np.float64)
    error = float(np.linalg.norm(gram - identity, ord="fro"))
    return SpectralSummary(minimum, maximum, condition, error)


def newton_schulz_orthogonalize(
    matrix: Sequence[Sequence[float]],
    *,
    quintic_steps: int = 5,
    cubic_steps: int = 0,
) -> np.ndarray:
    """Return a rectangular Newton--Schulz polar-factor approximation.

    The matrix is Frobenius-normalized before the iterations.  Tall matrices are
    transposed so the left-Gram matrix uses the smaller dimension, then transposed back.  The five
    quintic steps reproduce the Muon teaching trace; optional cubic steps illustrate
    how a costlier classical tail tightens the singular-value band.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    q_steps = _non_negative_int("quintic_steps", quintic_steps)
    c_steps = _non_negative_int("cubic_steps", cubic_steps)
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        raise ValueError("matrix must not be all zeros")

    transposed = array.shape[0] > array.shape[1]
    work = array.T.copy() if transposed else array.copy()
    work /= scale
    work /= np.linalg.norm(work, ord="fro")
    for _ in range(q_steps):
        gram = work @ work.T
        work = 3.4445 * work - 4.7750 * gram @ work + 2.0315 * gram @ gram @ work
    for _ in range(c_steps):
        work = 1.5 * work - 0.5 * (work @ work.T) @ work
    return work.T if transposed else work


@dataclass(frozen=True)
class QKClipResult:
    """Per-head QK-Clip scales for one monitoring step."""

    max_logits: np.ndarray
    threshold: float
    split: float
    gamma: np.ndarray
    query_scale: np.ndarray
    key_scale: np.ndarray
    corrected_max_logits: np.ndarray


def qk_clip_scales(
    max_logits: Sequence[float],
    threshold: float,
    *,
    split: float = 0.5,
) -> QKClipResult:
    """Return per-head parameter scales that restore monitored maxima to ``threshold``.

    ``split`` assigns ``gamma**split`` to Q and ``gamma**(1-split)`` to K.  The
    correction is an intervention at the current step; continued training can regrow
    the weights, so monitoring and reapplication remain necessary.
    """

    values = np.asarray(max_logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("max_logits must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("max_logits must contain finite non-negative values")
    tau = _positive_finite("threshold", threshold)
    share = _fraction("split", split)
    gamma = np.ones_like(values)
    over = values > tau
    gamma[over] = tau / values[over]
    q_scale = gamma**share
    k_scale = gamma ** (1.0 - share)
    corrected = values * q_scale * k_scale
    if np.any(corrected > tau * (1.0 + 1e-12)):
        raise AssertionError("QK-Clip failed to satisfy the threshold")
    return QKClipResult(values.copy(), tau, share, gamma, q_scale, k_scale, corrected)

# END EDUCATIONAL CORE: muon_qk


# BEGIN EDUCATIONAL CORE: memory_pipeline

@dataclass(frozen=True)
class AdamWStatePolicy:
    """Per-parameter byte components for mixed-precision AdamW state accounting."""

    model_weight_bytes: int = 2
    gradient_bytes: int = 2
    master_weight_bytes: int = 4
    first_moment_bytes: int = 4
    second_moment_bytes: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("model_weight_bytes", self.model_weight_bytes),
            ("gradient_bytes", self.gradient_bytes),
            ("master_weight_bytes", self.master_weight_bytes),
            ("first_moment_bytes", self.first_moment_bytes),
            ("second_moment_bytes", self.second_moment_bytes),
        ):
            _positive_int(name, value)

    @property
    def optimizer_state_bytes(self) -> int:
        """Return master weights plus first and second moments per parameter."""

        return self.master_weight_bytes + self.first_moment_bytes + self.second_moment_bytes

    @property
    def total_bytes_per_parameter(self) -> int:
        """Return all declared state bytes per parameter."""

        return self.model_weight_bytes + self.gradient_bytes + self.optimizer_state_bytes


def zero_state_bytes_per_rank(
    parameter_count: int,
    world_size: int,
    zero_stage: int,
    policy: AdamWStatePolicy | None = None,
) -> float:
    """Return the idealized parameter-state bytes per rank for ZeRO stages 0--3.

    Stage 0 replicates everything. Stage 1 shards optimizer state, stage 2 also shards
    gradients, and stage 3 also shards model weights. Activations, communication buffers,
    temporary workspaces, fragmentation, and framework metadata are deliberately absent.
    """

    parameters = _positive_int("parameter_count", parameter_count)
    world = _positive_int("world_size", world_size)
    stage = _non_negative_int("zero_stage", zero_stage)
    if stage > 3:
        raise ValueError("zero_stage must be one of 0, 1, 2, 3")
    p = policy or AdamWStatePolicy()
    weights = parameters * p.model_weight_bytes
    gradients = parameters * p.gradient_bytes
    optimizer = parameters * p.optimizer_state_bytes
    if stage == 0:
        return float(weights + gradients + optimizer)
    if stage == 1:
        return float(weights + gradients + optimizer / world)
    if stage == 2:
        return float(weights + (gradients + optimizer) / world)
    return float((weights + gradients + optimizer) / world)


def pipeline_bubble_fraction(stages: int, microbatches: int) -> float:
    """Return the idealized 1F1B bubble fraction ``(p-1)/(m+p-1)``."""

    p = _positive_int("stages", stages)
    m = _positive_int("microbatches", microbatches)
    return (p - 1) / (m + p - 1)


def minimum_microbatches_for_bubble(stages: int, maximum_bubble: float) -> int:
    """Return the smallest integer ``m`` whose idealized bubble is below the limit."""

    p = _positive_int("stages", stages)
    limit = _fraction("maximum_bubble", maximum_bubble, allow_zero=False)
    if p == 1:
        return 1
    candidate = max(1, math.ceil((p - 1) * (1.0 / limit - 1.0)))
    while pipeline_bubble_fraction(p, candidate) >= limit:
        candidate += 1
    while candidate > 1 and pipeline_bubble_fraction(p, candidate - 1) < limit:
        candidate -= 1
    return candidate

# END EDUCATIONAL CORE: memory_pipeline


def _allocation_dict(allocation: ComputeAllocation) -> dict[str, float]:
    """Serialize one compute allocation into JSON-friendly scalar fields."""

    return {
        "compute_flops": allocation.compute_flops,
        "model_parameters": allocation.model_parameters,
        "training_tokens": allocation.training_tokens,
        "tokens_per_parameter": allocation.tokens_per_parameter,
        "predicted_loss": allocation.predicted_loss,
    }


def _summary_dict(summary: SpectralSummary) -> dict[str, float]:
    """Serialize singular-value diagnostics into JSON-friendly scalar fields."""

    return {
        "minimum_singular_value": summary.minimum_singular_value,
        "maximum_singular_value": summary.maximum_singular_value,
        "condition_number": summary.condition_number,
        "orthogonality_error_frobenius": summary.orthogonality_error_frobenius,
    }


def compute_reference() -> dict[str, Any]:
    """Compute the complete deterministic numerical contract for Module 9."""

    law = ScalingLaw(
        irreducible_loss=1.69,
        model_coefficient=406.4,
        data_coefficient=410.7,
        model_exponent=0.34,
        data_exponent=0.28,
        compute_multiplier=6.0,
    )
    budgets = (7.23e23, 2.6e24, 1.0e25, 1.0e26)
    allocations = [law.optimal_split(budget) for budget in budgets]

    n_small = 8.03e9
    d_small = 15.0e12
    c_small = law.training_flops(n_small, d_small)
    small_optimum = law.optimal_split(c_small)
    small_loss = law.loss(n_small, d_small)
    break_even = inference_break_even(
        law,
        small_model_parameters=n_small,
        small_training_tokens=d_small,
        large_model_parameters=70.0e9,
        inference_flop_multiplier=2.0,
    )
    run = estimate_training_run(
        law,
        model_parameters=n_small,
        training_tokens=d_small,
        accelerator_count=2_048,
        peak_flops_per_accelerator=1.0e15,
        model_flop_utilization=0.42,
    )

    mixture_slices = (
        DataSlice("web", 6.0e12, 0.55),
        DataSlice("code", 1.0e12, 0.20),
        DataSlice("math", 0.4e12, 0.15),
        DataSlice("books", 0.6e12, 0.10),
    )
    mixture = allocate_data_mixture(mixture_slices, d_small)

    rng = np.random.default_rng(7)
    gradient = rng.standard_normal((64, 32))
    before = spectral_summary(gradient)
    quintic = newton_schulz_orthogonalize(gradient, quintic_steps=5, cubic_steps=0)
    after_quintic = spectral_summary(quintic)
    with_cubic_tail = newton_schulz_orthogonalize(gradient, quintic_steps=5, cubic_steps=3)
    after_cubic = spectral_summary(with_cubic_tail)
    wide = gradient.T
    wide_result = newton_schulz_orthogonalize(wide, quintic_steps=5, cubic_steps=0)

    qk = qk_clip_scales([45.0, 120.0, 260.0, 80.0], 100.0, split=0.5)
    regrown = qk.corrected_max_logits * 1.1**3
    clipped_view = np.minimum(regrown, qk.threshold)
    reapplied = qk_clip_scales(regrown, qk.threshold, split=0.5)

    policy = AdamWStatePolicy()
    memory_rows = []
    for stage in range(4):
        byte_count = zero_state_bytes_per_rank(7_000_000_000, 64, stage, policy)
        memory_rows.append(
            {
                "zero_stage": stage,
                "bytes_per_rank": byte_count,
                "decimal_gb_per_rank": bytes_to_decimal_gb(byte_count),
                "gib_per_rank": bytes_to_gib(byte_count),
            }
        )

    bubble_rows = [
        {
            "stages": 8,
            "microbatches": microbatches,
            "bubble_fraction": pipeline_bubble_fraction(8, microbatches),
        }
        for microbatches in (32, 128)
    ]

    return {
        "schema_version": 1,
        "scaling_law": {
            "irreducible_loss": law.irreducible_loss,
            "model_coefficient": law.model_coefficient,
            "data_coefficient": law.data_coefficient,
            "model_exponent": law.model_exponent,
            "data_exponent": law.data_exponent,
            "compute_multiplier": law.compute_multiplier,
        },
        "optimal_allocations": [_allocation_dict(item) for item in allocations],
        "small_model_scenario": {
            "model_parameters": n_small,
            "training_tokens": d_small,
            "training_flops": c_small,
            "tokens_per_parameter": d_small / n_small,
            "predicted_loss": small_loss,
            "same_budget_optimum": _allocation_dict(small_optimum),
            "loss_gap_to_same_budget_optimum": small_loss - small_optimum.predicted_loss,
        },
        "break_even": {
            "large_model_parameters": 70.0e9,
            "target_loss": break_even.target_loss,
            "small_training_flops": break_even.small_training_flops,
            "large_training_tokens": break_even.large_training_tokens,
            "large_training_flops": break_even.large_training_flops,
            "training_flops_saved_by_large_model": break_even.training_flops_saved_by_large_model,
            "extra_inference_flops_per_token_for_large_model": break_even.extra_inference_flops_per_token_for_large_model,
            "break_even_generated_tokens": break_even.break_even_generated_tokens,
        },
        "training_run": {
            "model_parameters": run.model_parameters,
            "training_tokens": run.training_tokens,
            "accelerator_count": run.accelerator_count,
            "peak_flops_per_accelerator": run.peak_flops_per_accelerator,
            "model_flop_utilization": run.model_flop_utilization,
            "total_training_flops": run.total_training_flops,
            "effective_cluster_flops_per_second": run.effective_cluster_flops_per_second,
            "estimated_seconds": run.estimated_seconds,
            "estimated_days": run.estimated_days,
            "training_tokens_per_second": run.training_tokens_per_second,
        },
        "data_mixture": {
            "total_training_tokens": d_small,
            "slices": [
                {
                    "name": item.name,
                    "unique_tokens": item.unique_tokens,
                    "sampling_weight": item.sampling_weight,
                }
                for item in mixture_slices
            ],
            "allocations": [
                {
                    "name": item.name,
                    "normalized_weight": item.normalized_weight,
                    "allocated_tokens": item.allocated_tokens,
                    "unique_tokens": item.unique_tokens,
                    "effective_epochs": item.effective_epochs,
                }
                for item in mixture
            ],
        },
        "muon": {
            "seed": 7,
            "shape": [64, 32],
            "quintic_steps": 5,
            "cubic_steps": 3,
            "before": _summary_dict(before),
            "after_quintic": _summary_dict(after_quintic),
            "after_cubic_tail": _summary_dict(after_cubic),
            "wide_shape": list(wide_result.shape),
            "wide_matches_transpose": bool(np.allclose(wide_result, quintic.T, atol=1e-12, rtol=0.0)),
        },
        "qk_clip": {
            "max_logits": qk.max_logits.tolist(),
            "threshold": qk.threshold,
            "split": qk.split,
            "gamma": qk.gamma.tolist(),
            "query_scale": qk.query_scale.tolist(),
            "key_scale": qk.key_scale.tolist(),
            "corrected_max_logits": qk.corrected_max_logits.tolist(),
            "regrown_after_three_10pct_steps": regrown.tolist(),
            "clipped_view_after_regrowth": clipped_view.tolist(),
            "reapplied_corrected_max_logits": reapplied.corrected_max_logits.tolist(),
        },
        "training_state_memory": {
            "parameter_count": 7_000_000_000,
            "world_size": 64,
            "policy": {
                "model_weight_bytes": policy.model_weight_bytes,
                "gradient_bytes": policy.gradient_bytes,
                "master_weight_bytes": policy.master_weight_bytes,
                "first_moment_bytes": policy.first_moment_bytes,
                "second_moment_bytes": policy.second_moment_bytes,
                "total_bytes_per_parameter": policy.total_bytes_per_parameter,
            },
            "rows": memory_rows,
        },
        "pipeline": {
            "rows": bubble_rows,
            "minimum_microbatches_for_5pct": minimum_microbatches_for_bubble(8, 0.05),
        },
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return compact deterministic JSON used for notebook embedding and hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Recompute and write the reviewed Module 9 numerical contract."""

    payload = compute_reference()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load the frozen Module 9 numerical contract."""

    return json.loads(path.read_text(encoding="utf-8"))


def _assert_reference_close(
    stored: Any,
    fresh: Any,
    *,
    path: str = "root",
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-12,
) -> None:
    """Compare frozen data structurally while allowing only round-off-scale float drift.

    SVD diagnostics can differ by a few ulps across NumPy/BLAS builds even when the
    matrix, seed, and algorithm are identical. Structure, integers, strings, booleans,
    keys, and list lengths remain exact; only finite floating-point leaves use a strict
    tolerance.
    """

    if isinstance(stored, bool) or isinstance(fresh, bool):
        if type(stored) is not type(fresh) or stored != fresh:
            raise AssertionError(f"reference mismatch at {path}: {stored!r} != {fresh!r}")
        return
    if isinstance(stored, (int, float)) and isinstance(fresh, (int, float)):
        if isinstance(stored, int) and isinstance(fresh, int):
            if stored != fresh:
                raise AssertionError(f"reference mismatch at {path}: {stored!r} != {fresh!r}")
            return
        left = float(stored)
        right = float(fresh)
        if not (math.isfinite(left) and math.isfinite(right)):
            if left != right:
                raise AssertionError(f"reference mismatch at {path}: {left!r} != {right!r}")
            return
        if not math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(
                f"reference mismatch at {path}: {left!r} != {right!r} "
                f"(rel_tol={rel_tol}, abs_tol={abs_tol})"
            )
        return
    if type(stored) is not type(fresh):
        raise AssertionError(
            f"reference type mismatch at {path}: {type(stored).__name__} != "
            f"{type(fresh).__name__}"
        )
    if isinstance(stored, Mapping):
        if set(stored) != set(fresh):
            raise AssertionError(
                f"reference key mismatch at {path}: {sorted(stored)} != {sorted(fresh)}"
            )
        for key in sorted(stored):
            _assert_reference_close(
                stored[key], fresh[key], path=f"{path}.{key}", rel_tol=rel_tol, abs_tol=abs_tol
            )
        return
    if isinstance(stored, list):
        if len(stored) != len(fresh):
            raise AssertionError(
                f"reference length mismatch at {path}: {len(stored)} != {len(fresh)}"
            )
        for index, (left, right) in enumerate(zip(stored, fresh)):
            _assert_reference_close(
                left, right, path=f"{path}[{index}]", rel_tol=rel_tol, abs_tol=abs_tol
            )
        return
    if stored != fresh:
        raise AssertionError(f"reference mismatch at {path}: {stored!r} != {fresh!r}")


def validate_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Validate the frozen contract with exact structure and ulp-scale float tolerance."""

    stored = load_reference(path)
    fresh = compute_reference()
    _assert_reference_close(stored, fresh)
    return stored


def _print_summary(payload: Mapping[str, Any]) -> None:
    """Print a compact human-readable validation summary."""

    small = payload["small_model_scenario"]
    br = payload["break_even"]
    run = payload["training_run"]
    muon = payload["muon"]
    qk = payload["qk_clip"]
    memory = payload["training_state_memory"]
    pipeline = payload["pipeline"]
    print("Module 9 reference contract: PASS")
    print(f"SHA-256: {payload_sha256(payload)}")
    print(
        "8.03B/15T: "
        f"loss={small['predicted_loss']:.6f}, "
        f"gap={small['loss_gap_to_same_budget_optimum']:.6f}, "
        f"break-even={br['break_even_generated_tokens']:.6e} generated tokens"
    )
    print(
        f"training scenario: {run['estimated_days']:.3f} days at "
        f"{run['accelerator_count']} accelerators and MFU={run['model_flop_utilization']:.2f}"
    )
    print(
        "Muon singular band: "
        f"[{muon['after_quintic']['minimum_singular_value']:.6f}, "
        f"{muon['after_quintic']['maximum_singular_value']:.6f}]"
    )
    print(f"QK corrected maxima: {qk['corrected_max_logits']}")
    print(
        "AdamW state stage 0/3: "
        f"{memory['rows'][0]['gib_per_rank']:.3f} GiB -> "
        f"{memory['rows'][3]['gib_per_rank']:.3f} GiB"
    )
    print(
        "1F1B bubble: "
        f"{pipeline['rows'][0]['bubble_fraction']:.4%} -> "
        f"{pipeline['rows'][1]['bubble_fraction']:.4%}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for read-only validation or explicit reference regeneration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write freshly computed reference data after an intentional review",
    )
    args = parser.parse_args(argv)
    if args.write:
        path = write_reference()
        print(path)
        payload = load_reference(path)
    else:
        payload = validate_reference()
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
