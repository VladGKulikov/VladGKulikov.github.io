#!/usr/bin/env python3
"""Deterministic reference calculations for RL for LLM, Module 12.

The module is about accounting and contracts, not a distributed runtime.  The
functions below isolate the arithmetic that should remain true regardless of
which training or inference framework is used: model-state and KV-cache memory,
pipeline timing, straggler utilization, queue stability, policy lag,
importance weights, checkpoint cadence, and schema validation.

Only NumPy and the Python standard library are required.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Any

import numpy as np


BYTES_PER_GIB = 1024**3
BYTES_PER_GB = 10**9


def completion_token_mask(token_ids: np.ndarray, eos_token_ids: Sequence[int]) -> np.ndarray:
    """Mask a generated suffix through its first EOS, including that EOS.

    The input excludes the padded prompt. This supports PAD == EOS: padding
    after the first EOS is inactive, while the first EOS remains an action.
    Rows without EOS use their full width (a generation length limit).
    """
    ids = np.asarray(token_ids)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("token_ids must be a two-dimensional integer array")
    eos = np.asarray(eos_token_ids)
    if eos.ndim != 1 or (eos.size and not np.issubdtype(eos.dtype, np.integer)):
        raise ValueError("eos_token_ids must be a sequence of integers")
    is_eos = np.isin(ids, eos)
    # Count EOS strictly before the current position, retaining the first EOS.
    return (np.cumsum(is_eos, axis=1) - is_eos) == 0


@dataclass(frozen=True)
class MemoryProfile:
    """Logical model-state memory under an explicit storage profile."""

    total_bytes: int
    parameter_bytes: int
    trainable_extra_bytes: int
    trainable_parameters: int


@dataclass(frozen=True)
class PeakMemory:
    """Peak memory across named phases."""

    phase: str
    peak_bytes: int
    phase_totals: dict[str, int]


@dataclass(frozen=True)
class PipelineReport:
    """Timing summary for a two-stage rollout/training pipeline."""

    batches: int
    sequential_total: float
    pipeline_total: float
    steady_state_period: float
    speedup: float


@dataclass(frozen=True)
class StragglerReport:
    """Utilization of a synchronous group that waits for its slowest member."""

    total_work: float
    wall_clock: float
    capacity: float
    utilization: float
    idle_fraction: float


@dataclass(frozen=True)
class LagReport:
    """Summary of behavior-policy lag in optimizer-version units."""

    lags: tuple[int, ...]
    mean_lag: float
    max_lag: int
    stale_fraction: float


@dataclass(frozen=True)
class SyncSchedule:
    """Exact arithmetic for periodic rollout-weight synchronization."""

    sync_every: int
    updates: int
    synchronization_count: int
    lag_sequence: tuple[int, ...]
    mean_lag: float
    max_lag: int


@dataclass(frozen=True)
class CheckpointCadence:
    """First-order checkpoint overhead model."""

    interval_seconds: float
    write_fraction: float
    expected_lost_work_fraction: float
    total_fraction: float


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0 or (positive and value == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _finite_vector(values: Sequence[float], name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def model_state_bytes(
    total_parameters: int,
    *,
    trainable_parameters: int | None = None,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_state_bytes: int = 8,
    master_parameter_bytes: int = 0,
) -> MemoryProfile:
    """Return logical model-state memory under an explicit profile.

    Base parameters are stored for *all* model parameters. Gradients, optimizer
    state, and an optional master copy are stored only for trainable parameters.
    This distinction makes the same function useful for full fine-tuning and
    adapter training.
    """

    total_parameters = _positive_int(total_parameters, "total_parameters")
    if trainable_parameters is None:
        trainable_parameters = total_parameters
    trainable_parameters = _positive_int(
        trainable_parameters, "trainable_parameters", allow_zero=True
    )
    if trainable_parameters > total_parameters:
        raise ValueError("trainable_parameters cannot exceed total_parameters")

    byte_fields = {
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "master_parameter_bytes": master_parameter_bytes,
    }
    for name, value in byte_fields.items():
        _positive_int(value, name, allow_zero=True)

    base = total_parameters * parameter_bytes
    extra_per_trainable = gradient_bytes + optimizer_state_bytes + master_parameter_bytes
    extra = trainable_parameters * extra_per_trainable
    return MemoryProfile(
        total_bytes=base + extra,
        parameter_bytes=base,
        trainable_extra_bytes=extra,
        trainable_parameters=trainable_parameters,
    )


def ideal_sharded_memory(
    sharded_bytes: int,
    shards: int,
    *,
    replicated_bytes: int = 0,
    temporary_peak_bytes: int = 0,
) -> int:
    """Idealized per-rank footprint plus replicated and temporary memory.

    Integer ceiling is used because a byte cannot be divided among ranks.
    """

    sharded_bytes = _positive_int(sharded_bytes, "sharded_bytes", allow_zero=True)
    shards = _positive_int(shards, "shards")
    replicated_bytes = _positive_int(replicated_bytes, "replicated_bytes", allow_zero=True)
    temporary_peak_bytes = _positive_int(
        temporary_peak_bytes, "temporary_peak_bytes", allow_zero=True
    )
    local_shard = (sharded_bytes + shards - 1) // shards
    return local_shard + replicated_bytes + temporary_peak_bytes


def kv_cache_bytes(
    *,
    num_sequences: int,
    tokens_per_sequence: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    element_bytes: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Useful KV-cache payload, optionally idealized over tensor parallelism.

    The formula is ``2 * tokens * layers * kv_heads * head_dim * bytes``.  The
    factor two is for keys and values.  Allocator pages, block tables, alignment,
    and temporary kernels are deliberately excluded.
    """

    values = {
        "num_sequences": num_sequences,
        "tokens_per_sequence": tokens_per_sequence,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "element_bytes": element_bytes,
        "tensor_parallel_size": tensor_parallel_size,
    }
    for name, value in values.items():
        _positive_int(value, name)

    total_tokens = num_sequences * tokens_per_sequence
    payload = 2 * total_tokens * num_layers * num_kv_heads * head_dim * element_bytes
    return (payload + tensor_parallel_size - 1) // tensor_parallel_size


def peak_memory(phases: Mapping[str, Sequence[int] | int]) -> PeakMemory:
    """Return the phase with the largest sum of non-negative memory components."""

    if not phases:
        raise ValueError("phases must be non-empty")
    totals: dict[str, int] = {}
    for phase, components in phases.items():
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase names must be non-empty strings")
        if isinstance(components, int) and not isinstance(components, bool):
            values = [components]
        else:
            values = list(components)  # type: ignore[arg-type]
        if not values:
            raise ValueError(f"phase {phase!r} must contain at least one component")
        total = 0
        for value in values:
            total += _positive_int(value, f"memory component in {phase}", allow_zero=True)
        totals[phase] = total
    peak_phase = max(totals, key=totals.get)
    return PeakMemory(peak_phase, totals[peak_phase], totals)


def informative_throughput(groups_per_second: float, degenerate_fraction: float) -> float:
    """Groups per second multiplied by the non-degenerate group fraction."""

    groups_per_second = _finite_nonnegative(groups_per_second, "groups_per_second")
    degenerate_fraction = float(degenerate_fraction)
    if not isfinite(degenerate_fraction) or not 0.0 <= degenerate_fraction <= 1.0:
        raise ValueError("degenerate_fraction must lie in [0, 1]")
    return groups_per_second * (1.0 - degenerate_fraction)


def pipeline_report(rollout_seconds: float, train_seconds: float, batches: int) -> PipelineReport:
    """Sequential and ideal two-stage pipeline timing for ``batches`` batches."""

    rollout_seconds = _finite_nonnegative(rollout_seconds, "rollout_seconds", positive=True)
    train_seconds = _finite_nonnegative(train_seconds, "train_seconds", positive=True)
    batches = _positive_int(batches, "batches")
    sequential_total = batches * (rollout_seconds + train_seconds)
    period = max(rollout_seconds, train_seconds)
    pipeline_total = rollout_seconds + train_seconds + (batches - 1) * period
    return PipelineReport(
        batches=batches,
        sequential_total=sequential_total,
        pipeline_total=pipeline_total,
        steady_state_period=period,
        speedup=sequential_total / pipeline_total,
    )


def straggler_report(durations: Sequence[float], *, workers: int | None = None) -> StragglerReport:
    """Utilization for a synchronous wave that ends at the slowest duration."""

    times = _finite_vector(durations, "durations")
    if np.any(times < 0.0):
        raise ValueError("durations must be non-negative")
    if workers is None:
        workers = int(times.size)
    workers = _positive_int(workers, "workers")
    if workers < times.size:
        raise ValueError("workers must be at least the number of simultaneous tasks")
    wall = float(times.max())
    if wall == 0.0:
        utilization = 1.0
        capacity = 0.0
    else:
        capacity = workers * wall
        utilization = float(times.sum() / capacity)
    return StragglerReport(
        total_work=float(times.sum()),
        wall_clock=wall,
        capacity=float(capacity),
        utilization=utilization,
        idle_fraction=1.0 - utilization,
    )


def queue_load(arrival_rate: float, service_rate: float) -> float:
    """Return utilization ``rho=lambda/mu`` for consistent work units."""

    arrival_rate = _finite_nonnegative(arrival_rate, "arrival_rate")
    service_rate = _finite_nonnegative(service_rate, "service_rate", positive=True)
    return arrival_rate / service_rate


def queue_is_stable(arrival_rate: float, service_rate: float) -> bool:
    """Necessary first-order stability condition for a stationary single queue."""

    return queue_load(arrival_rate, service_rate) < 1.0


def sequence_log_importance(
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
) -> float:
    """Log importance ratio of one autoregressive sequence."""

    current = _finite_vector(current_logprobs, "current_logprobs")
    behavior = _finite_vector(behavior_logprobs, "behavior_logprobs")
    if current.shape != behavior.shape:
        raise ValueError("current_logprobs and behavior_logprobs must have equal shape")
    if np.any(current > 1e-12) or np.any(behavior > 1e-12):
        raise ValueError("log probabilities must be non-positive")
    return float(np.sum(current - behavior))


def importance_weights(
    current_logprobs: Sequence[float],
    behavior_logprobs: Sequence[float],
    *,
    min_log_ratio: float | None = None,
    max_log_ratio: float | None = None,
) -> np.ndarray:
    """Exponentiated pointwise log-probability differences with optional clipping."""

    current = _finite_vector(current_logprobs, "current_logprobs")
    behavior = _finite_vector(behavior_logprobs, "behavior_logprobs")
    if current.shape != behavior.shape:
        raise ValueError("current_logprobs and behavior_logprobs must have equal shape")
    if np.any(current > 1e-12) or np.any(behavior > 1e-12):
        raise ValueError("log probabilities must be non-positive")
    log_ratios = current - behavior
    if min_log_ratio is not None:
        min_log_ratio = float(min_log_ratio)
        if not isfinite(min_log_ratio):
            raise ValueError("min_log_ratio must be finite")
        log_ratios = np.maximum(log_ratios, min_log_ratio)
    if max_log_ratio is not None:
        max_log_ratio = float(max_log_ratio)
        if not isfinite(max_log_ratio):
            raise ValueError("max_log_ratio must be finite")
        log_ratios = np.minimum(log_ratios, max_log_ratio)
    if min_log_ratio is not None and max_log_ratio is not None and min_log_ratio > max_log_ratio:
        raise ValueError("min_log_ratio cannot exceed max_log_ratio")
    return np.exp(log_ratios)


def effective_sample_size(weights: Sequence[float], *, normalized: bool = False) -> float:
    """Return ``(sum w)^2 / sum(w^2)`` for non-negative importance weights."""

    values = _finite_vector(weights, "weights")
    if np.any(values < 0.0):
        raise ValueError("weights must be non-negative")
    denominator = float(np.dot(values, values))
    if denominator == 0.0:
        raise ValueError("at least one weight must be positive")
    ess = float(values.sum() ** 2 / denominator)
    return ess / values.size if normalized else ess


def policy_lag(current_version: int, behavior_versions: Sequence[int]) -> LagReport:
    """Return non-negative optimizer-version lag for a rollout batch."""

    current_version = _positive_int(current_version, "current_version", allow_zero=True)
    versions_array = np.asarray(behavior_versions)
    if versions_array.ndim != 1 or versions_array.size == 0:
        raise ValueError("behavior_versions must be a non-empty one-dimensional sequence")
    if not np.issubdtype(versions_array.dtype, np.integer):
        raise ValueError("behavior_versions must contain integers")
    versions = versions_array.astype(int)
    if np.any(versions < 0):
        raise ValueError("behavior versions must be non-negative")
    lags = current_version - versions
    if np.any(lags < 0):
        raise ValueError("behavior version cannot be newer than current_version")
    return LagReport(
        lags=tuple(int(value) for value in lags),
        mean_lag=float(lags.mean()),
        max_lag=int(lags.max()),
        stale_fraction=float(np.mean(lags > 0)),
    )


def synchronization_transfer_time(
    payload_bytes: int,
    bandwidth_gb_per_second: float,
    *,
    setup_seconds: float = 0.0,
    efficiency: float = 1.0,
) -> float:
    """First-order transfer time using decimal GB/s and an efficiency factor."""

    payload_bytes = _positive_int(payload_bytes, "payload_bytes", allow_zero=True)
    bandwidth = _finite_nonnegative(
        bandwidth_gb_per_second, "bandwidth_gb_per_second", positive=True
    )
    setup_seconds = _finite_nonnegative(setup_seconds, "setup_seconds")
    efficiency = float(efficiency)
    if not isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must lie in (0, 1]")
    effective_bytes_per_second = bandwidth * BYTES_PER_GB * efficiency
    return setup_seconds + payload_bytes / effective_bytes_per_second


def synchronization_schedule(sync_every: int, updates: int) -> SyncSchedule:
    """Lag sequence when rollout weights are refreshed every ``sync_every`` updates.

    Version lag follows ``0, 1, ..., sync_every-1`` and resets at each refresh.
    """

    sync_every = _positive_int(sync_every, "sync_every")
    updates = _positive_int(updates, "updates")
    lags = tuple(step % sync_every for step in range(updates))
    synchronizations = 1 + (updates - 1) // sync_every
    return SyncSchedule(
        sync_every=sync_every,
        updates=updates,
        synchronization_count=synchronizations,
        lag_sequence=lags,
        mean_lag=float(np.mean(lags)),
        max_lag=max(lags),
    )


def synchronization_cadence_cost(
    sync_every: int,
    *,
    sync_seconds: float,
    lag_cost_per_version: float,
) -> float:
    """Per-update cost ``sync_time/K + lag_cost*(K-1)/2``.

    This is an explicit teaching approximation, not a universal quality model.
    """

    sync_every = _positive_int(sync_every, "sync_every")
    sync_seconds = _finite_nonnegative(sync_seconds, "sync_seconds")
    lag_cost_per_version = _finite_nonnegative(
        lag_cost_per_version, "lag_cost_per_version"
    )
    return sync_seconds / sync_every + lag_cost_per_version * (sync_every - 1) / 2.0


def optimal_sync_every(
    candidates: Sequence[int],
    *,
    sync_seconds: float,
    lag_cost_per_version: float,
) -> tuple[int, float]:
    """Return the candidate cadence minimizing the teaching cost model."""

    if not candidates:
        raise ValueError("candidates must be non-empty")
    unique: list[int] = []
    for candidate in candidates:
        candidate = _positive_int(candidate, "candidate")
        if candidate not in unique:
            unique.append(candidate)
    costs = {
        candidate: synchronization_cadence_cost(
            candidate,
            sync_seconds=sync_seconds,
            lag_cost_per_version=lag_cost_per_version,
        )
        for candidate in unique
    }
    winner = min(costs, key=costs.get)
    return winner, costs[winner]


def optimal_checkpoint_interval(checkpoint_seconds: float, mtbf_seconds: float) -> float:
    """Young/Daly first-order optimum ``sqrt(2*C*M)`` without restart cost."""

    checkpoint_seconds = _finite_nonnegative(
        checkpoint_seconds, "checkpoint_seconds", positive=True
    )
    mtbf_seconds = _finite_nonnegative(mtbf_seconds, "mtbf_seconds", positive=True)
    return sqrt(2.0 * checkpoint_seconds * mtbf_seconds)


def checkpoint_cadence(
    interval_seconds: float,
    *,
    checkpoint_seconds: float,
    mtbf_seconds: float,
) -> CheckpointCadence:
    """Approximate checkpoint writing and expected lost-work fractions."""

    interval_seconds = _finite_nonnegative(
        interval_seconds, "interval_seconds", positive=True
    )
    checkpoint_seconds = _finite_nonnegative(
        checkpoint_seconds, "checkpoint_seconds", positive=True
    )
    mtbf_seconds = _finite_nonnegative(mtbf_seconds, "mtbf_seconds", positive=True)
    write_fraction = checkpoint_seconds / interval_seconds
    lost_fraction = interval_seconds / (2.0 * mtbf_seconds)
    return CheckpointCadence(
        interval_seconds=interval_seconds,
        write_fraction=write_fraction,
        expected_lost_work_fraction=lost_fraction,
        total_fraction=write_fraction + lost_fraction,
    )


REQUIRED_ROLLOUT_FIELDS = {
    "sample_id",
    "prompt_id",
    "policy_version",
    "prompt_tokens",
    "response_tokens",
    "behavior_logprobs",
    "decoding",
    "reward",
    "stop_reason",
    "verifier_version",
}


def rollout_contract_errors(record: Mapping[str, Any]) -> list[str]:
    """Return human-readable errors for the minimal one-answer rollout schema."""

    if not isinstance(record, Mapping):
        return ["record must be a mapping"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ROLLOUT_FIELDS - set(record))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))

    for key in ("sample_id", "prompt_id", "stop_reason", "verifier_version"):
        if key in record and (not isinstance(record[key], str) or not record[key]):
            errors.append(f"{key} must be a non-empty string")

    if "policy_version" in record:
        value = record["policy_version"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append("policy_version must be a non-negative integer")

    for key in ("prompt_tokens", "response_tokens"):
        if key in record:
            value = record[key]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                errors.append(f"{key} must be a sequence of token ids")
            elif any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
                errors.append(f"{key} must contain non-negative integer token ids")

    if "behavior_logprobs" in record:
        value = record["behavior_logprobs"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            errors.append("behavior_logprobs must be a sequence")
        else:
            try:
                logprobs = np.asarray(value, dtype=float)
                if logprobs.ndim != 1 or not np.isfinite(logprobs).all() or np.any(logprobs > 1e-12):
                    errors.append("behavior_logprobs must be finite non-positive values")
                if "response_tokens" in record and isinstance(record["response_tokens"], Sequence):
                    if len(logprobs) != len(record["response_tokens"]):
                        errors.append("behavior_logprobs must align with response_tokens")
            except (TypeError, ValueError):
                errors.append("behavior_logprobs must be numeric")

    for key in ("decoding", "reward"):
        if key in record and not isinstance(record[key], Mapping):
            errors.append(f"{key} must be a mapping")
    return errors


REQUIRED_CHECKPOINT_FIELDS = {
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "global_step",
    "policy_version",
    "rng_state",
    "data_cursor",
    "rollout_schema_version",
    "tokenizer_revision",
    "verifier_version",
    "decoding_config_hash",
}


def checkpoint_manifest_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Validate the semantic manifest of an RL checkpoint."""

    if not isinstance(manifest, Mapping):
        return ["manifest must be a mapping"]
    errors: list[str] = []
    missing = sorted(REQUIRED_CHECKPOINT_FIELDS - set(manifest))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    for key in ("global_step", "policy_version"):
        if key in manifest:
            value = manifest[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{key} must be a non-negative integer")
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rollout_schema_version",
        "tokenizer_revision",
        "verifier_version",
        "decoding_config_hash",
    ):
        if key in manifest and (not isinstance(manifest[key], str) or not manifest[key]):
            errors.append(f"{key} must be a non-empty string")
    if "rng_state" in manifest and not isinstance(manifest["rng_state"], Mapping):
        errors.append("rng_state must be a mapping")
    return errors


def diagnose_pipeline(metrics: Mapping[str, float]) -> list[str]:
    """Map a small set of measured symptoms to testable hypotheses.

    The function intentionally returns hypotheses rather than diagnoses.  It is
    used by exercises to demonstrate that the next action should be a targeted
    measurement, not a framework switch.
    """

    required = {
        "trainer_idle_fraction",
        "rollout_idle_fraction",
        "verifier_fraction",
        "weight_sync_fraction",
        "queue_growth_per_minute",
        "normalized_ess",
        "oom_events",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError("missing metrics: " + ", ".join(missing))
    values = {key: float(metrics[key]) for key in required}
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("metrics must be finite")
    for key in (
        "trainer_idle_fraction",
        "rollout_idle_fraction",
        "verifier_fraction",
        "weight_sync_fraction",
        "normalized_ess",
    ):
        if not 0.0 <= values[key] <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1]")
    if values["oom_events"] < 0.0:
        raise ValueError("oom_events must be non-negative")

    hypotheses: list[str] = []
    if values["trainer_idle_fraction"] > 0.25 and values["queue_growth_per_minute"] <= 0.0:
        hypotheses.append("rollout, verification, or input supply is starving the trainer")
    if values["rollout_idle_fraction"] > 0.25 and values["queue_growth_per_minute"] > 0.0:
        hypotheses.append("training consumption or backpressure is limiting rollout workers")
    if values["verifier_fraction"] > 0.30:
        hypotheses.append("the verifier is on the critical path")
    if values["weight_sync_fraction"] > 0.20:
        hypotheses.append("weight synchronization is a first-order bottleneck")
    if values["normalized_ess"] < 0.50:
        hypotheses.append("policy lag or training-inference mismatch is concentrating importance weights")
    if values["oom_events"] > 0.0:
        hypotheses.append("measure phase-specific peaks, KV cache, activations, and transfer buffers")
    if not hypotheses:
        hypotheses.append("no dominant failure is identified; inspect a stage-level trace")
    return hypotheses


def human_bytes(num_bytes: int, *, binary: bool = True) -> str:
    """Compact deterministic formatter used by the notebook and solutions."""

    num_bytes = _positive_int(num_bytes, "num_bytes", allow_zero=True)
    base = 1024.0 if binary else 1000.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"] if binary else ["B", "kB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < base or unit == units[-1]:
            break
        value /= base
    return f"{value:.3f} {unit}"


if __name__ == "__main__":
    profile = model_state_bytes(
        7_000_000_000,
        parameter_bytes=2,
        gradient_bytes=2,
        optimizer_state_bytes=8,
        master_parameter_bytes=4,
    )
    kv = kv_cache_bytes(
        num_sequences=64,
        tokens_per_sequence=4096,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        element_bytes=2,
    )
    print("7B teaching profile:", human_bytes(profile.total_bytes, binary=False))
    print("KV teaching example:", human_bytes(kv))
    print("pipeline:", pipeline_report(90.0, 10.0, 20))
