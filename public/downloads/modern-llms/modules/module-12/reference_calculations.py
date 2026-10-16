#!/usr/bin/env python3
"""Reviewed numerical core and frozen reference data for Module 12.

The module owns its numerical fixtures. They are deliberately independent of lecture prose
and visualization files, both of which may change during later editorial revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation

def _positive_int(name: str, value: object) -> int:
    """Return a strictly positive Python integer and reject booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(name: str, value: object) -> int:
    """Return a non-negative Python integer and reject booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_real(name: str, value: object) -> float:
    """Return a finite real scalar and reject booleans and complex values."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_real(
    name: str,
    value: object,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> float:
    """Validate one finite scalar against an explicit interval."""

    result = _finite_real(name, value)
    lower_ok = result >= lower if lower_inclusive else result > lower
    upper_ok = result <= upper if upper_inclusive else result < upper
    if not (lower_ok and upper_ok):
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        raise ValueError(f"{name} must lie in {left}{lower}, {upper}{right}")
    return result


def _finite_array(
    name: str,
    value: object,
    *,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type = np.float64,
) -> np.ndarray:
    """Convert to a NumPy array and enforce shape rank and finite values."""

    array = np.asarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_tokens(name: str, value: object, *, ndim: int = 1) -> np.ndarray:
    """Validate an integer token tensor without silently truncating floats."""

    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer token IDs")
    return raw.astype(np.int64, copy=False)


# END EDUCATIONAL CORE: validation


# BEGIN EDUCATIONAL CORE: masking_and_loss

@dataclass(frozen=True)
class CorruptionResult:
    """One absorbing-mask forward sample for a one-dimensional token sequence."""

    clean_tokens: np.ndarray
    noisy_tokens: np.ndarray
    masked_positions: np.ndarray
    eligible_positions: np.ndarray
    time: float
    mask_id: int


def linear_mask_schedule(length: int, steps: int) -> tuple[int, ...]:
    """Return monotone target mask counts from ``length`` to zero.

    ``ceil`` gives a declared policy for non-divisible lengths. Duplicate counts are allowed
    when ``steps > length``; such a step performs a model evaluation but opens no token.
    """

    n = _positive_int("length", length)
    s = _positive_int("steps", steps)
    counts = tuple((n * remaining + s - 1) // s for remaining in range(s, -1, -1))
    if counts[0] != n or counts[-1] != 0:
        raise AssertionError("schedule endpoints are inconsistent")
    if any(left < right for left, right in zip(counts, counts[1:])):
        raise AssertionError("mask schedule must be non-increasing")
    return counts


def elbo_time_weight(time: Real) -> float:
    """Return the absorbing-diffusion slice weight ``1 / t`` for ``0 < t <= 1``."""

    t = _bounded_real("time", time, lower=0.0, upper=1.0, lower_inclusive=False)
    return 1.0 / t


def corrupt_absorbing(
    tokens: object,
    *,
    time: Real,
    mask_id: int,
    rng: np.random.Generator,
    eligible_positions: object | None = None,
) -> CorruptionResult:
    """Replace eligible tokens independently by ``mask_id`` with probability ``time``.

    The mask token must be outside the clean sequence so that masked and observed positions
    remain unambiguous. ``eligible_positions`` can protect a fixed prefix or special tokens.
    """

    clean = _integer_tokens("tokens", tokens)
    if isinstance(mask_id, bool) or not isinstance(mask_id, Integral):
        raise TypeError("mask_id must be an integer")
    mask = int(mask_id)
    if np.any(clean == mask):
        raise ValueError("mask_id must not occur in the clean sequence")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    t = _bounded_real("time", time, lower=0.0, upper=1.0)

    if eligible_positions is None:
        eligible = np.ones(clean.shape, dtype=bool)
    else:
        eligible = np.asarray(eligible_positions)
        if eligible.shape != clean.shape or eligible.dtype.kind != "b":
            raise ValueError("eligible_positions must be a boolean vector matching tokens")
        eligible = eligible.astype(bool, copy=False)

    selected = eligible & (rng.random(clean.shape[0]) < t)
    noisy = clean.copy()
    noisy[selected] = mask
    return CorruptionResult(
        clean_tokens=clean.copy(),
        noisy_tokens=noisy,
        masked_positions=selected,
        eligible_positions=eligible.copy(),
        time=t,
        mask_id=mask,
    )


def stable_log_softmax(logits: object) -> np.ndarray:
    """Compute row-wise log-softmax for a finite ``[positions, vocabulary]`` tensor."""

    values = _finite_array("logits", logits, ndim=2)
    maxima = np.max(values, axis=-1, keepdims=True)
    shifted = values - maxima
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    result = shifted - log_normalizer
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("log-softmax produced a non-finite value")
    return result


def masked_diffusion_loss(
    logits: object,
    target_tokens: object,
    masked_positions: object,
    *,
    time: Real,
) -> float:
    """Return ``(1/t) * sum(-log p_theta(x_i | x_t))`` on masked positions only."""

    log_probs = stable_log_softmax(logits)
    targets = _integer_tokens("target_tokens", target_tokens)
    if targets.shape[0] != log_probs.shape[0]:
        raise ValueError("target_tokens and logits must have the same position axis")
    if np.any(targets < 0) or np.any(targets >= log_probs.shape[1]):
        raise ValueError("target token ID is outside the vocabulary")
    selected = np.asarray(masked_positions)
    if selected.shape != targets.shape or selected.dtype.kind != "b":
        raise ValueError("masked_positions must be a boolean vector matching target_tokens")
    selected = selected.astype(bool, copy=False)
    weight = elbo_time_weight(time)
    nll = -log_probs[np.arange(targets.size), targets]
    loss = weight * float(nll[selected].sum())
    if not math.isfinite(loss):
        raise FloatingPointError("diffusion loss is non-finite")
    return loss


# END EDUCATIONAL CORE: masking_and_loss


# BEGIN EDUCATIONAL CORE: sampler

@dataclass(frozen=True)
class DenoiseStep:
    """Inspectable state transition produced by one denoising model call."""

    step: int
    target_masked: int
    remasked_positions: tuple[int, ...]
    opened_positions: tuple[int, ...]
    masked_after: int
    sequence: tuple[int, ...]


@dataclass(frozen=True)
class DenoiseResult:
    """Final sequence and full trace of a masked-diffusion sampling run."""

    final_tokens: np.ndarray
    schedule: tuple[int, ...]
    steps: tuple[DenoiseStep, ...]
    model_calls: int


def stable_softmax(logits: object) -> np.ndarray:
    """Compute row-wise softmax for finite logits and verify probability normalization."""

    log_probs = stable_log_softmax(logits)
    probabilities = np.exp(log_probs)
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
        raise FloatingPointError("softmax rows do not sum to one")
    return probabilities


def sample_masked_diffusion(
    predict: Callable[[np.ndarray, int], object],
    *,
    length: int,
    steps: int,
    mask_id: int,
    rng: np.random.Generator,
    selection_policy: Literal["confidence", "random", "entropy"] = "confidence",
    token_policy: Literal["greedy", "categorical"] = "greedy",
    remask_fraction: Real = 0.0,
) -> DenoiseResult:
    """Run a transparent masked-diffusion sampler with one model call per step.

    ``predict`` receives a copy of the current token vector and the one-based step number,
    and must return finite logits of shape ``[length, vocabulary]``. The mask ID must lie
    outside that vocabulary. Opened tokens remain fixed unless explicitly selected for
    remasking. Confidence ties are resolved by lower position index.
    """

    n = _positive_int("length", length)
    s = _positive_int("steps", steps)
    if isinstance(mask_id, bool) or not isinstance(mask_id, Integral):
        raise TypeError("mask_id must be an integer")
    mask = int(mask_id)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if selection_policy not in {"confidence", "random", "entropy"}:
        raise ValueError("selection_policy must be 'confidence', 'random', or 'entropy'")
    if token_policy not in {"greedy", "categorical"}:
        raise ValueError("token_policy must be 'greedy' or 'categorical'")
    remask = _bounded_real("remask_fraction", remask_fraction, lower=0.0, upper=1.0)

    schedule = linear_mask_schedule(n, s)
    sequence = np.full(n, mask, dtype=np.int64)
    trace: list[DenoiseStep] = []
    stored_confidence = np.zeros(n)
    expected_vocabulary = None

    for step_number, target_masked in enumerate(schedule[1:], start=1):
        # Select from scores saved when tokens were last predicted. Mask BEFORE
        # this call, so the network conditions on the actual corrupted input.
        opened_before = np.flatnonzero(sequence != mask)
        remasked = np.empty(0, dtype=np.int64)
        remask_count = math.floor(remask * opened_before.size)
        if remask_count:
            order = np.lexsort((opened_before, stored_confidence[opened_before]))
            remasked = opened_before[order[:remask_count]]
            sequence[remasked] = mask
        raw_logits = predict(sequence.copy(), step_number)
        logits = _finite_array("predict output", raw_logits, ndim=2)
        if logits.shape[0] != n:
            raise ValueError("predict output must have one row per sequence position")
        vocabulary = logits.shape[1]
        if vocabulary <= 0:
            raise ValueError("predict output vocabulary must be non-empty")
        if 0 <= mask < vocabulary:
            raise ValueError("mask_id must lie outside the model vocabulary")
        if expected_vocabulary is None:
            expected_vocabulary = vocabulary
        elif vocabulary != expected_vocabulary:
            raise ValueError("predict changed vocabulary size between calls")
        probabilities = stable_softmax(logits)

        if token_policy == "greedy":
            candidates = np.argmax(probabilities, axis=-1).astype(np.int64)
        else:
            candidates = np.array(
                [rng.choice(vocabulary, p=row) for row in probabilities],
                dtype=np.int64,
            )
        candidate_confidence = probabilities[np.arange(n), candidates]

        masked_now = np.flatnonzero(sequence == mask)
        open_count = masked_now.size - target_masked
        if open_count < 0:
            raise AssertionError("sampler has fewer masks than the schedule requires")
        if selection_policy == "confidence":
            order = np.lexsort((masked_now, -candidate_confidence[masked_now]))
            opened = masked_now[order[:open_count]]
        elif selection_policy == "entropy":
            log_probabilities = stable_log_softmax(logits)
            entropy = -(probabilities * log_probabilities).sum(axis=-1)
            order = np.lexsort((masked_now, entropy[masked_now]))
            opened = masked_now[order[:open_count]]
        else:
            opened = rng.permutation(masked_now)[:open_count]
        sequence[opened] = candidates[opened]
        stored_confidence[opened] = candidate_confidence[opened]

        masked_after = int(np.count_nonzero(sequence == mask))
        if masked_after != target_masked:
            raise AssertionError("sampler did not reach the scheduled mask count")
        trace.append(
            DenoiseStep(
                step=step_number,
                target_masked=target_masked,
                remasked_positions=tuple(sorted(map(int, remasked))),
                opened_positions=tuple(sorted(map(int, opened))),
                masked_after=masked_after,
                sequence=tuple(map(int, sequence)),
            )
        )

    if np.any(sequence == mask):
        raise AssertionError("final sequence still contains mask tokens")
    return DenoiseResult(
        final_tokens=sequence.copy(),
        schedule=schedule,
        steps=tuple(trace),
        model_calls=s,
    )


def make_contextual_toy_predictor(
    target_tokens: object,
    *,
    vocabulary_size: int,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Create a deterministic toy model whose odd positions need their left anchor.

    Even positions are high-confidence anchors. An odd position initially prefers a wrong
    token; once its left anchor is opened correctly, the target becomes high-confidence.
    This makes reveal order and remasking observable without training a neural network.
    """

    target = _integer_tokens("target_tokens", target_tokens)
    vocab = _positive_int("vocabulary_size", vocabulary_size)
    if np.any(target < 0) or np.any(target >= vocab):
        raise ValueError("target token ID is outside vocabulary_size")

    def predict(sequence: np.ndarray, step: int) -> np.ndarray:
        """Return deterministic contextual logits for one sampler step."""

        current = _integer_tokens("sequence", sequence)
        if current.shape != target.shape:
            raise ValueError("sequence shape differs from the toy target")
        _positive_int("step", step)
        logits = np.full((target.size, vocab), -2.5, dtype=np.float64)
        for position, target_id in enumerate(target):
            wrong_id = (int(target_id) + 1) % vocab
            if position % 2 == 0:
                logits[position, int(target_id)] = 3.5
                logits[position, wrong_id] = 0.0
            elif current[position - 1] == target[position - 1]:
                logits[position, int(target_id)] = 3.3
                logits[position, wrong_id] = 0.0
            else:
                logits[position, wrong_id] = 1.5
                logits[position, int(target_id)] = 1.1
        return logits

    return predict


# END EDUCATIONAL CORE: sampler


# BEGIN EDUCATIONAL CORE: block_diffusion

def block_diffusion_attention_mask(prefix_length: int, block_length: int) -> np.ndarray:
    """Return a full boolean mask for a fixed causal prefix plus one bidirectional block.

    Prefix queries remain causal and cannot read the current block. Queries in the current
    block may read every prefix token and every token of their own block. The fixed prefix is
    therefore cacheable between denoising steps, while the current block is recomputed.
    """

    prefix = _nonnegative_int("prefix_length", prefix_length)
    block = _positive_int("block_length", block_length)
    total = prefix + block
    allowed = np.zeros((total, total), dtype=bool)
    if prefix:
        allowed[:prefix, :prefix] = np.tril(np.ones((prefix, prefix), dtype=bool))
    allowed[prefix:, :total] = True
    return allowed


# END EDUCATIONAL CORE: block_diffusion


# BEGIN EDUCATIONAL CORE: roofline

@dataclass(frozen=True)
class HardwareSpec:
    """Decimal-bandwidth and FLOP/s inputs for a transparent roofline scenario."""

    memory_bandwidth_bytes_per_s: float
    peak_flops_per_s: float

    def __post_init__(self) -> None:
        """Validate positive finite hardware rates."""

        object.__setattr__(
            self,
            "memory_bandwidth_bytes_per_s",
            _bounded_real(
                "memory_bandwidth_bytes_per_s",
                self.memory_bandwidth_bytes_per_s,
                lower=0.0,
                upper=float("inf"),
                lower_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "peak_flops_per_s",
            _bounded_real(
                "peak_flops_per_s",
                self.peak_flops_per_s,
                lower=0.0,
                upper=float("inf"),
                lower_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class DenseModelSpec:
    """Dense model size and approximate arithmetic per token."""

    parameter_count: int
    flops_per_parameter_per_token: float = 2.0

    def __post_init__(self) -> None:
        """Validate model size and per-token arithmetic coefficient."""

        object.__setattr__(self, "parameter_count", _positive_int("parameter_count", self.parameter_count))
        object.__setattr__(
            self,
            "flops_per_parameter_per_token",
            _bounded_real(
                "flops_per_parameter_per_token",
                self.flops_per_parameter_per_token,
                lower=0.0,
                upper=float("inf"),
                lower_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class RooflineEstimate:
    """Arithmetic floor for one repeated full-sequence or block denoising workload."""

    weight_bits: float
    weight_payload_bytes: float
    weight_payload_decimal_gb: float
    weight_payload_gib: float
    memory_time_s: float
    compute_time_s: float
    step_time_s: float
    tokens_per_s: float
    bottleneck: str


def bytes_to_gib(byte_count: Real) -> float:
    """Convert a non-negative byte count to binary GiB."""

    value = _bounded_real("byte_count", byte_count, lower=0.0, upper=float("inf"))
    return value / 2**30


def bytes_to_decimal_gb(byte_count: Real) -> float:
    """Convert a non-negative byte count to decimal GB."""

    value = _bounded_real("byte_count", byte_count, lower=0.0, upper=float("inf"))
    return value / 1e9


def estimate_roofline(
    model: DenseModelSpec,
    hardware: HardwareSpec,
    *,
    positions_per_step: int,
    denoise_steps: int,
    weight_bits: Real,
) -> RooflineEstimate:
    """Estimate a weight-stream/compute floor for repeated full-width model evaluations.

    The estimate reads the weight payload once per model call and counts dense arithmetic as
    ``flops_per_parameter_per_token * P * positions_per_step``. It excludes KV traffic,
    activations, communication, kernels, runtime overhead, batching, and measured utilization.
    """

    positions = _positive_int("positions_per_step", positions_per_step)
    steps = _positive_int("denoise_steps", denoise_steps)
    bits = _bounded_real("weight_bits", weight_bits, lower=0.0, upper=float("inf"), lower_inclusive=False)
    payload = model.parameter_count * bits / 8.0
    memory_time = payload / hardware.memory_bandwidth_bytes_per_s
    compute_time = (
        model.flops_per_parameter_per_token * model.parameter_count * positions
        / hardware.peak_flops_per_s
    )
    step_time = max(memory_time, compute_time)
    throughput = positions / (steps * step_time)
    bottleneck = "memory" if memory_time >= compute_time else "compute"
    return RooflineEstimate(
        weight_bits=bits,
        weight_payload_bytes=payload,
        weight_payload_decimal_gb=bytes_to_decimal_gb(payload),
        weight_payload_gib=bytes_to_gib(payload),
        memory_time_s=memory_time,
        compute_time_s=compute_time,
        step_time_s=step_time,
        tokens_per_s=throughput,
        bottleneck=bottleneck,
    )


def maximum_steps_strictly_faster(
    *,
    positions_per_step: int,
    diffusion_step_time_s: Real,
    autoregressive_step_time_s: Real,
) -> int:
    """Return the largest integer ``S`` with ``N/(S*t_diff) > 1/t_ar``."""

    positions = _positive_int("positions_per_step", positions_per_step)
    t_diff = _bounded_real(
        "diffusion_step_time_s",
        diffusion_step_time_s,
        lower=0.0,
        upper=float("inf"),
        lower_inclusive=False,
    )
    t_ar = _bounded_real(
        "autoregressive_step_time_s",
        autoregressive_step_time_s,
        lower=0.0,
        upper=float("inf"),
        lower_inclusive=False,
    )
    ratio = positions * t_ar / t_diff
    return max(0, math.ceil(ratio) - 1)


# END EDUCATIONAL CORE: roofline


def _probability_logits(target: np.ndarray, correct_probabilities: np.ndarray, vocabulary: int) -> np.ndarray:
    """Construct finite logits whose target probabilities equal the supplied values."""

    probs = np.empty((target.size, vocabulary), dtype=np.float64)
    for row, (token, p_correct) in enumerate(zip(target, correct_probabilities)):
        probs[row] = (1.0 - p_correct) / (vocabulary - 1)
        probs[row, token] = p_correct
    return np.log(probs)


def _static_trace_payload() -> dict[str, Any]:
    """Reproduce the small confidence-based trace used in the lecture."""

    target = np.array([3, 1, 4, 1, 0, 2, 4, 3], dtype=np.int64)
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((target.size, 5))
    logits[np.arange(target.size), target] += 2.0

    def predict(_: np.ndarray, __: int) -> np.ndarray:
        return logits

    result = sample_masked_diffusion(
        predict,
        length=target.size,
        steps=4,
        mask_id=-1,
        rng=np.random.default_rng(99),
        selection_policy="confidence",
    )
    probabilities = stable_softmax(logits)
    candidates = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    return {
        "target": target.tolist(),
        "candidates": candidates.tolist(),
        "confidence": confidence.tolist(),
        "opened_order": [list(step.opened_positions) for step in result.steps],
        "final": result.final_tokens.tolist(),
        "correct": int(np.count_nonzero(result.final_tokens == target)),
        "model_calls": result.model_calls,
    }


def _policy_payload() -> dict[str, Any]:
    """Compare reveal policies on a contextual deterministic toy model."""

    target = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0], dtype=np.int64)
    predictor = make_contextual_toy_predictor(target, vocabulary_size=6)
    trials = 200
    metrics: dict[str, float] = {}
    policies = {
        "confidence": ("confidence", 0.0),
        "random": ("random", 0.0),
        "random_with_remask": ("random", 0.25),
    }
    for name, (selection, remask) in policies.items():
        accuracies = []
        for trial in range(trials):
            result = sample_masked_diffusion(
                predictor,
                length=target.size,
                steps=4,
                mask_id=-1,
                rng=np.random.default_rng(1000 + trial),
                selection_policy=selection,
                remask_fraction=remask,
            )
            accuracies.append(float(np.mean(result.final_tokens == target)))
        metrics[name] = float(np.mean(accuracies))
    return {"trials": trials, "target": target.tolist(), **metrics}


def _roofline_payload() -> dict[str, Any]:
    """Build the generic dense-8B teaching roofline scenario used by notebook and figures.

    This scenario is deliberately not identified with DiffusionGemma or any other released
    model. It isolates weight-stream and dense arithmetic floors for a hypothetical 8B model.
    """

    hardware = HardwareSpec(memory_bandwidth_bytes_per_s=3.35e12, peak_flops_per_s=990e12)
    model = DenseModelSpec(parameter_count=8_000_000_000)
    ar = estimate_roofline(model, hardware, positions_per_step=1, denoise_steps=1, weight_bits=16)
    diffusion = {
        str(steps): asdict(
            estimate_roofline(
                model,
                hardware,
                positions_per_step=1024,
                denoise_steps=steps,
                weight_bits=16,
            )
        )
        for steps in (16, 64, 128)
    }
    diff64 = RooflineEstimate(**diffusion["64"])
    parity = maximum_steps_strictly_faster(
        positions_per_step=1024,
        diffusion_step_time_s=diff64.step_time_s,
        autoregressive_step_time_s=ar.step_time_s,
    )
    block_bf16 = estimate_roofline(
        model,
        hardware,
        positions_per_step=256,
        denoise_steps=16,
        weight_bits=16,
    )
    block_4bit = estimate_roofline(
        model,
        hardware,
        positions_per_step=256,
        denoise_steps=16,
        weight_bits=4,
    )
    return {
        "scenario": "generic_dense_8b_h100_lower_bound",
        "hardware": asdict(hardware),
        "model": asdict(model),
        "autoregressive": {
            **asdict(ar),
            "ideal_arithmetic_share_of_step": ar.compute_time_s / ar.step_time_s,
        },
        "diffusion": diffusion,
        "maximum_steps_strictly_faster": parity,
        "batch1_throughput_ratio_at_64_steps": diff64.tokens_per_s / ar.tokens_per_s,
        "ar_compute_time_fraction_of_step": ar.compute_time_s / ar.step_time_s,
        "batch_parity_under_linear_ar_scaling": diff64.tokens_per_s / ar.tokens_per_s,
        "block_bf16": asdict(block_bf16),
        "block_4bit": asdict(block_4bit),
        "block_steps_for_1000_tokens_per_s_4bit": 256 / (1000 * block_4bit.step_time_s),
        "scope": (
            "Arithmetic lower-bound scenario only: excludes KV traffic, activations, "
            "communication, routing, kernel/runtime overhead, batching, and measured utilization."
        ),
    }


def _diffusiongemma_passport_payload() -> dict[str, Any]:
    """Return released-model facts and arithmetic-only payload bounds for DiffusionGemma.

    The 4-bit payloads below are intentionally labelled idealized. They do not estimate one
    diffusion model call, because a MoE call can touch the union of experts selected by all
    canvas positions and because checkpoint metadata/precision are heterogeneous.
    """

    total = 25_200_000_000
    active = 3_800_000_000
    full_4bit_bytes = total * 4 / 8
    active_4bit_bytes = active * 4 / 8
    return {
        "source_snapshot": "2026-10-11",
        "release_date": "2026-06-10",
        "total_parameters": total,
        "active_parameters": active,
        "layers": 30,
        "routed_experts": 128,
        "active_routed_experts": 8,
        "shared_experts": 1,
        "canvas_tokens": 256,
        "context_tokens": 262_144,
        "sliding_window_tokens": 1_024,
        "committed_tokens_per_pass_min": 15,
        "committed_tokens_per_pass_max": 20,
        "h100_precision": "FP8",
        "h100_developer_reported_tokens_per_s_lower_bound": 1_100,
        "native_nvfp4_hardware_family": "NVIDIA Blackwell",
        "idealized_full_bank_4bit_bytes": full_4bit_bytes,
        "idealized_full_bank_4bit_decimal_gb": bytes_to_decimal_gb(full_4bit_bytes),
        "idealized_full_bank_4bit_gib": bytes_to_gib(full_4bit_bytes),
        "idealized_active_4bit_bytes": active_4bit_bytes,
        "idealized_active_4bit_decimal_gb": bytes_to_decimal_gb(active_4bit_bytes),
        "idealized_active_4bit_gib": bytes_to_gib(active_4bit_bytes),
        "traffic_note": (
            "Neither idealized payload equals measured bytes per diffusion pass; actual traffic "
            "depends on expert union, precision, metadata, cache, kernels, and runtime."
        ),
    }


def make_reference_payload() -> dict[str, Any]:
    """Return the complete deterministic Module 12 numerical contract."""

    clean = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    eligible = np.array([False, False, True, True, True, True, True, True])
    corruption = corrupt_absorbing(
        clean,
        time=0.5,
        mask_id=-1,
        rng=np.random.default_rng(73),
        eligible_positions=eligible,
    )

    target = np.array([0, 1, 2, 3], dtype=np.int64)
    p_correct = np.array([0.9, 0.7, 0.6, 0.8], dtype=np.float64)
    logits = _probability_logits(target, p_correct, vocabulary=5)
    selected = np.ones(target.shape, dtype=bool)
    loss = masked_diffusion_loss(logits, target, selected, time=0.5)

    block_mask = block_diffusion_attention_mask(prefix_length=3, block_length=4)
    return {
        "schedule": {
            "length": 8,
            "steps": 4,
            "counts": list(linear_mask_schedule(8, 4)),
            "weights": {str(t): elbo_time_weight(t) for t in (0.25, 0.5, 0.75)},
        },
        "corruption": {
            "clean_tokens": clean.tolist(),
            "eligible_positions": eligible.tolist(),
            "time": 0.5,
            "mask_id": -1,
            "masked_positions": np.flatnonzero(corruption.masked_positions).tolist(),
            "noisy_tokens": corruption.noisy_tokens.tolist(),
        },
        "loss": {
            "target_tokens": target.tolist(),
            "correct_probabilities": p_correct.tolist(),
            "time": 0.5,
            "masked_positions_count": int(selected.sum()),
            "reduction": "sum_over_masked_positions_then_multiply_by_1_over_t",
            "value": loss,
        },
        "trace": _static_trace_payload(),
        "policy": _policy_payload(),
        "block_mask": {
            "prefix_length": 3,
            "block_length": 4,
            "allowed": block_mask.astype(int).tolist(),
        },
        "roofline": _roofline_payload(),
        "diffusiongemma_passport": _diffusiongemma_passport_payload(),
    }


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize the payload exactly as embedded in learner notebooks."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the digest used by notebook/builder/reference consistency checks."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_reference(reference_path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Recompute and compare the frozen reference payload exactly."""

    expected = make_reference_payload()
    stored = json.loads(reference_path.read_text(encoding="utf-8"))
    if canonical_json(stored) != canonical_json(expected):
        raise AssertionError("stored Module 12 reference payload differs from recomputation")
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
