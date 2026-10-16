#!/usr/bin/env python3
"""Deterministic infrastructure simulations for RL for LLM, Module 12.

This file does not pretend to emulate a GPU cluster cycle by cycle.  It models
the contracts that matter for an RL data pipeline:

* rollout workers create variable-length records under a published policy version;
* a bounded queue applies backpressure;
* a trainer consumes token-budgeted batches and advances the current version;
* synchronization publishes a new rollout version after a measurable delay;
* stale records can be rejected by an explicit version-age policy;
* synthetic importance ratios expose how policy lag can concentrate weights;
* a failure drill distinguishes a checkpoint file from a resumable experiment.

The simulation is intentionally small enough for a CPU notebook, but its state
machine mirrors the same decisions needed in a distributed implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite
from typing import Literal, Sequence

import numpy as np

from module_12_reference import effective_sample_size


Mode = Literal["sequential", "bounded_async"]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a synthetic RL infrastructure experiment."""

    mode: Mode = "bounded_async"
    duration_seconds: float = 240.0
    dt_seconds: float = 0.10
    rollout_workers: int = 8
    rollout_tokens_per_second: float = 320.0
    trainer_tokens_per_second: float = 2200.0
    batch_tokens: int = 4096
    mean_rollout_tokens: float = 640.0
    rollout_length_sigma: float = 0.55
    verifier_seconds: float = 0.12
    sync_every_updates: int = 1
    sync_seconds: float = 0.80
    max_queue_tokens: int = 16384
    max_policy_lag: int = 2
    log_ratio_std_per_update: float = 0.09
    seed: int = 12016


@dataclass(frozen=True)
class RolloutRecord:
    """One complete rollout as seen by the trainer."""

    rollout_id: int
    token_count: int
    behavior_version: int
    started_at: float
    completed_at: float


@dataclass(frozen=True)
class BatchRecord:
    """Diagnostics for one optimizer update."""

    update: int
    current_version_before: int
    published_version: int
    token_count: int
    rollout_count: int
    mean_lag: float
    max_lag: int
    normalized_ess: float
    started_at: float
    completed_at: float


@dataclass(frozen=True)
class SimulationResult:
    """Summary and event traces from a pipeline simulation."""

    config: PipelineConfig
    wall_time: float
    generated_tokens: int
    trained_tokens: int
    completed_rollouts: int
    updates: int
    discarded_rollouts: int
    discarded_tokens: int
    backpressure_seconds: float
    rollout_utilization: float
    trainer_utilization: float
    generated_tokens_per_second: float
    trained_tokens_per_second: float
    mean_policy_lag: float
    p95_policy_lag: float
    mean_normalized_ess: float
    min_normalized_ess: float
    max_queue_tokens_observed: int
    timeline: tuple[dict[str, float], ...]
    batches: tuple[BatchRecord, ...]


@dataclass(frozen=True)
class CheckpointState:
    """Minimal state needed by the deterministic recovery drill."""

    optimizer_step: int
    current_version: int
    published_version: int
    dataset_cursor: int
    rng_state: dict
    configuration_hash: str
    rollout_schema_version: str
    tokenizer_id: str
    verifier_version: str


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of an uninterrupted versus resumed execution comparison."""

    uninterrupted: tuple[int, ...]
    resumed: tuple[int, ...]
    exact_match: bool
    broken_resume: tuple[int, ...]
    broken_match: bool


def _validate_config(cfg: PipelineConfig) -> None:
    if cfg.mode not in {"sequential", "bounded_async"}:
        raise ValueError("mode must be 'sequential' or 'bounded_async'")
    positive_float = (
        "duration_seconds",
        "dt_seconds",
        "rollout_tokens_per_second",
        "trainer_tokens_per_second",
        "batch_tokens",
        "mean_rollout_tokens",
        "verifier_seconds",
        "sync_every_updates",
        "sync_seconds",
        "max_queue_tokens",
        "log_ratio_std_per_update",
    )
    for name in positive_float:
        value = float(getattr(cfg, name))
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.rollout_workers <= 0:
        raise ValueError("rollout_workers must be positive")
    if cfg.max_policy_lag < 0:
        raise ValueError("max_policy_lag must be non-negative")
    if cfg.rollout_length_sigma < 0:
        raise ValueError("rollout_length_sigma must be non-negative")
    if cfg.dt_seconds > cfg.duration_seconds:
        raise ValueError("dt_seconds cannot exceed duration_seconds")


def _sample_rollout_tokens(rng: np.random.Generator, cfg: PipelineConfig) -> int:
    """Sample a positive heavy-tailed response length with the requested mean."""

    sigma = cfg.rollout_length_sigma
    if sigma == 0:
        return max(1, int(round(cfg.mean_rollout_tokens)))
    mu = np.log(cfg.mean_rollout_tokens) - 0.5 * sigma**2
    return max(1, int(round(rng.lognormal(mu, sigma))))


def _batch_diagnostics(
    selected: Sequence[RolloutRecord],
    *,
    current_version: int,
    rng: np.random.Generator,
    cfg: PipelineConfig,
) -> tuple[float, int, float]:
    lags = np.array(
        [max(0, current_version - item.behavior_version) for item in selected],
        dtype=float,
    )
    # The log-ratio model is explicitly synthetic.  Longer sequences and larger
    # version lag both increase dispersion, which is the mechanism being taught.
    lengths = np.array([item.token_count for item in selected], dtype=float)
    scales = (
        cfg.log_ratio_std_per_update
        * lags
        * np.sqrt(np.maximum(lengths, 1.0) / cfg.mean_rollout_tokens)
    )
    log_ratios = rng.normal(0.0, scales)
    # Numerically safe enough for the small teaching configuration.
    weights = np.exp(np.clip(log_ratios, -12.0, 12.0))
    normalized_ess = effective_sample_size(weights, normalized=True)
    return float(lags.mean()) if lags.size else 0.0, int(lags.max()) if lags.size else 0, float(normalized_ess)


def _simulate_sequential(cfg: PipelineConfig) -> SimulationResult:
    """Strict generate → train → synchronize cycles with no stale queue."""

    rng = np.random.default_rng(cfg.seed)
    now = 0.0
    rollout_busy = 0.0
    trainer_busy = 0.0
    generated_tokens = 0
    trained_tokens = 0
    completed = 0
    version = 0
    update = 0
    batches: list[BatchRecord] = []
    timeline: list[dict[str, float]] = []

    while now < cfg.duration_seconds:
        records: list[RolloutRecord] = []
        batch_tokens = 0
        wave_start = now
        # Each wave uses all rollout workers; waves repeat until the batch token
        # budget is met.  The slowest rollout in a wave defines wall-clock time.
        while batch_tokens < cfg.batch_tokens and now < cfg.duration_seconds:
            wave: list[tuple[int, float]] = []
            for _ in range(cfg.rollout_workers):
                tokens = _sample_rollout_tokens(rng, cfg)
                seconds = tokens / cfg.rollout_tokens_per_second + cfg.verifier_seconds
                wave.append((tokens, seconds))
            wave_wall = max(seconds for _, seconds in wave)
            for tokens, seconds in wave:
                records.append(
                    RolloutRecord(
                        rollout_id=completed,
                        token_count=tokens,
                        behavior_version=version,
                        started_at=now,
                        completed_at=now + seconds,
                    )
                )
                completed += 1
                generated_tokens += tokens
                batch_tokens += tokens
                rollout_busy += seconds
            now += wave_wall

        if not records or now >= cfg.duration_seconds:
            break

        selected: list[RolloutRecord] = []
        selected_tokens = 0
        for record in records:
            selected.append(record)
            selected_tokens += record.token_count
            if selected_tokens >= cfg.batch_tokens:
                break

        train_start = now
        train_seconds = selected_tokens / cfg.trainer_tokens_per_second
        now += train_seconds
        trainer_busy += train_seconds
        trained_tokens += selected_tokens

        before = version
        version += 1
        now += cfg.sync_seconds
        trainer_busy += cfg.sync_seconds
        update += 1
        batches.append(
            BatchRecord(
                update=update,
                current_version_before=before,
                published_version=version,
                token_count=selected_tokens,
                rollout_count=len(selected),
                mean_lag=0.0,
                max_lag=0,
                normalized_ess=1.0,
                started_at=train_start,
                completed_at=now,
            )
        )
        timeline.append(
            {
                "time": min(now, cfg.duration_seconds),
                "queue_tokens": 0.0,
                "current_version": float(version),
                "published_version": float(version),
                "mean_lag": 0.0,
                "normalized_ess": 1.0,
            }
        )

    wall = min(max(now, 1e-12), cfg.duration_seconds)
    rollout_capacity = cfg.rollout_workers * wall
    return SimulationResult(
        config=cfg,
        wall_time=wall,
        generated_tokens=generated_tokens,
        trained_tokens=trained_tokens,
        completed_rollouts=completed,
        updates=update,
        discarded_rollouts=0,
        discarded_tokens=0,
        backpressure_seconds=0.0,
        rollout_utilization=min(1.0, rollout_busy / max(rollout_capacity, 1e-12)),
        trainer_utilization=min(1.0, trainer_busy / max(wall, 1e-12)),
        generated_tokens_per_second=generated_tokens / wall,
        trained_tokens_per_second=trained_tokens / wall,
        mean_policy_lag=0.0,
        p95_policy_lag=0.0,
        mean_normalized_ess=1.0 if batches else float("nan"),
        min_normalized_ess=1.0 if batches else float("nan"),
        max_queue_tokens_observed=0,
        timeline=tuple(timeline),
        batches=tuple(batches),
    )


def _simulate_bounded_async(cfg: PipelineConfig) -> SimulationResult:
    """Separate rollout and training pools with a bounded queue and backpressure."""

    rng = np.random.default_rng(cfg.seed)
    steps = int(np.ceil(cfg.duration_seconds / cfg.dt_seconds))
    now = 0.0

    # Per worker: remaining seconds, tokens, behavior version, start time, id.
    workers: list[dict[str, float | int] | None] = [None] * cfg.rollout_workers
    queue: list[RolloutRecord] = []
    next_rollout_id = 0

    current_version = 0
    published_version = 0
    trainer_remaining = 0.0
    sync_remaining = 0.0
    pending_batch: tuple[list[RolloutRecord], float, float, int] | None = None

    generated_tokens = 0
    trained_tokens = 0
    completed_rollouts = 0
    discarded_rollouts = 0
    discarded_tokens = 0
    backpressure_seconds = 0.0
    rollout_busy_worker_seconds = 0.0
    trainer_busy_seconds = 0.0
    max_queue = 0
    batches: list[BatchRecord] = []
    timeline: list[dict[str, float]] = []

    def queue_tokens() -> int:
        return int(sum(item.token_count for item in queue))

    for tick in range(steps):
        now = min(tick * cfg.dt_seconds, cfg.duration_seconds)

        # Progress active rollouts.
        for idx, state in enumerate(workers):
            if state is None:
                continue
            rollout_busy_worker_seconds += cfg.dt_seconds
            state["remaining"] = float(state["remaining"]) - cfg.dt_seconds
            if float(state["remaining"]) <= 1e-12:
                record = RolloutRecord(
                    rollout_id=int(state["rollout_id"]),
                    token_count=int(state["tokens"]),
                    behavior_version=int(state["behavior_version"]),
                    started_at=float(state["started_at"]),
                    completed_at=now,
                )
                queue.append(record)
                generated_tokens += record.token_count
                completed_rollouts += 1
                workers[idx] = None

        # Complete a synchronization and publish one atomic version.
        if sync_remaining > 0.0:
            sync_remaining = max(0.0, sync_remaining - cfg.dt_seconds)
            trainer_busy_seconds += cfg.dt_seconds
            if sync_remaining == 0.0:
                published_version = current_version

        # Progress or complete a training batch.
        if trainer_remaining > 0.0:
            trainer_remaining = max(0.0, trainer_remaining - cfg.dt_seconds)
            trainer_busy_seconds += cfg.dt_seconds
            if trainer_remaining == 0.0 and pending_batch is not None:
                selected, train_started, mean_lag, max_lag = pending_batch
                current_version += 1
                update = current_version
                trained = sum(item.token_count for item in selected)
                trained_tokens += trained
                _, _, ess = _batch_diagnostics(
                    selected,
                    current_version=update - 1,
                    rng=rng,
                    cfg=cfg,
                )
                if update % cfg.sync_every_updates == 0:
                    sync_remaining = cfg.sync_seconds
                batches.append(
                    BatchRecord(
                        update=update,
                        current_version_before=update - 1,
                        published_version=published_version,
                        token_count=trained,
                        rollout_count=len(selected),
                        mean_lag=mean_lag,
                        max_lag=max_lag,
                        normalized_ess=ess,
                        started_at=train_started,
                        completed_at=now,
                    )
                )
                pending_batch = None

        # Remove records that violate the declared version-age contract.
        retained: list[RolloutRecord] = []
        for item in queue:
            lag = current_version - item.behavior_version
            if lag > cfg.max_policy_lag:
                discarded_rollouts += 1
                discarded_tokens += item.token_count
            else:
                retained.append(item)
        queue = retained

        # Start a new trainer batch if enough tokens are ready and no train/sync is active.
        if trainer_remaining <= 0.0 and sync_remaining <= 0.0 and queue_tokens() >= cfg.batch_tokens:
            selected: list[RolloutRecord] = []
            selected_tokens = 0
            while queue and selected_tokens < cfg.batch_tokens:
                item = queue.pop(0)
                selected.append(item)
                selected_tokens += item.token_count
            lags = [current_version - item.behavior_version for item in selected]
            mean_lag = float(np.mean(lags)) if lags else 0.0
            max_lag = int(max(lags)) if lags else 0
            trainer_remaining = selected_tokens / cfg.trainer_tokens_per_second
            pending_batch = (selected, now, mean_lag, max_lag)

        # Bounded queue: do not start new rollouts while already over the cap.
        over_cap = queue_tokens() >= cfg.max_queue_tokens
        if over_cap:
            backpressure_seconds += cfg.dt_seconds
        else:
            for idx, state in enumerate(workers):
                if state is not None:
                    continue
                # Recheck because multiple completions can overshoot a bounded queue.
                if queue_tokens() >= cfg.max_queue_tokens:
                    break
                tokens = _sample_rollout_tokens(rng, cfg)
                duration = tokens / cfg.rollout_tokens_per_second + cfg.verifier_seconds
                workers[idx] = {
                    "remaining": duration,
                    "tokens": tokens,
                    "behavior_version": published_version,
                    "started_at": now,
                    "rollout_id": next_rollout_id,
                }
                next_rollout_id += 1

        q_tokens = queue_tokens()
        max_queue = max(max_queue, q_tokens)
        if tick % max(1, int(round(1.0 / cfg.dt_seconds))) == 0:
            latest = batches[-1] if batches else None
            timeline.append(
                {
                    "time": now,
                    "queue_tokens": float(q_tokens),
                    "current_version": float(current_version),
                    "published_version": float(published_version),
                    "mean_lag": float(latest.mean_lag if latest else 0.0),
                    "normalized_ess": float(latest.normalized_ess if latest else 1.0),
                }
            )

    wall = cfg.duration_seconds
    lag_values = [batch.mean_lag for batch in batches]
    ess_values = [batch.normalized_ess for batch in batches]
    rollout_capacity = cfg.rollout_workers * wall
    return SimulationResult(
        config=cfg,
        wall_time=wall,
        generated_tokens=generated_tokens,
        trained_tokens=trained_tokens,
        completed_rollouts=completed_rollouts,
        updates=len(batches),
        discarded_rollouts=discarded_rollouts,
        discarded_tokens=discarded_tokens,
        backpressure_seconds=backpressure_seconds,
        rollout_utilization=min(1.0, rollout_busy_worker_seconds / max(rollout_capacity, 1e-12)),
        trainer_utilization=min(1.0, trainer_busy_seconds / max(wall, 1e-12)),
        generated_tokens_per_second=generated_tokens / wall,
        trained_tokens_per_second=trained_tokens / wall,
        mean_policy_lag=float(np.mean(lag_values)) if lag_values else float("nan"),
        p95_policy_lag=float(np.quantile(lag_values, 0.95)) if lag_values else float("nan"),
        mean_normalized_ess=float(np.mean(ess_values)) if ess_values else float("nan"),
        min_normalized_ess=float(np.min(ess_values)) if ess_values else float("nan"),
        max_queue_tokens_observed=max_queue,
        timeline=tuple(timeline),
        batches=tuple(batches),
    )


def simulate_pipeline(config: PipelineConfig) -> SimulationResult:
    """Run a deterministic sequential or bounded-asynchronous experiment."""

    _validate_config(config)
    if config.mode == "sequential":
        return _simulate_sequential(config)
    return _simulate_bounded_async(config)


def result_row(result: SimulationResult) -> dict[str, float | int | str]:
    """Compact table row for notebooks and reports."""

    return {
        "mode": result.config.mode,
        "sync_every": result.config.sync_every_updates,
        "queue_cap": result.config.max_queue_tokens,
        "max_lag_contract": result.config.max_policy_lag,
        "updates": result.updates,
        "trained_tok_s": round(result.trained_tokens_per_second, 2),
        "generated_tok_s": round(result.generated_tokens_per_second, 2),
        "mean_lag": round(result.mean_policy_lag, 3),
        "p95_lag": round(result.p95_policy_lag, 3),
        "mean_nESS": round(result.mean_normalized_ess, 4),
        "min_nESS": round(result.min_normalized_ess, 4),
        "discarded_rollouts": result.discarded_rollouts,
        "max_queue_tokens": result.max_queue_tokens_observed,
        "backpressure_s": round(result.backpressure_seconds, 2),
    }


def _rng_state_copy(rng: np.random.Generator) -> dict:
    # ``bit_generator.state`` is a nested JSON-compatible mapping for PCG64.
    state = rng.bit_generator.state
    return {
        "bit_generator": state["bit_generator"],
        "state": dict(state["state"]),
        "has_uint32": int(state["has_uint32"]),
        "uinteger": int(state["uinteger"]),
    }


def make_checkpoint(
    *,
    optimizer_step: int,
    current_version: int,
    published_version: int,
    dataset_cursor: int,
    rng: np.random.Generator,
    configuration_hash: str = "cfg:module12:v2026.24",
    rollout_schema_version: str = "rollout.v3",
    tokenizer_id: str = "Qwen/Qwen3-0.6B",
    verifier_version: str = "verifier.math.v2",
) -> CheckpointState:
    """Capture deterministic state for the failure drill."""

    if min(optimizer_step, current_version, published_version, dataset_cursor) < 0:
        raise ValueError("checkpoint counters must be non-negative")
    if published_version > current_version:
        raise ValueError("published_version cannot exceed current_version")
    return CheckpointState(
        optimizer_step=int(optimizer_step),
        current_version=int(current_version),
        published_version=int(published_version),
        dataset_cursor=int(dataset_cursor),
        rng_state=_rng_state_copy(rng),
        configuration_hash=str(configuration_hash),
        rollout_schema_version=str(rollout_schema_version),
        tokenizer_id=str(tokenizer_id),
        verifier_version=str(verifier_version),
    )


def restore_rng(checkpoint: CheckpointState) -> np.random.Generator:
    """Restore the exact PCG64 state stored in ``checkpoint``."""

    rng = np.random.default_rng()
    rng.bit_generator.state = checkpoint.rng_state
    return rng


def recovery_drill(seed: int = 12016, before: int = 8, after: int = 8) -> RecoveryResult:
    """Compare uninterrupted, correct-resume, and broken-resume random streams."""

    if before <= 0 or after <= 0:
        raise ValueError("before and after must be positive")
    baseline_rng = np.random.default_rng(seed)
    uninterrupted = tuple(int(x) for x in baseline_rng.integers(0, 10_000, size=before + after))

    run_rng = np.random.default_rng(seed)
    prefix = tuple(int(x) for x in run_rng.integers(0, 10_000, size=before))
    checkpoint = make_checkpoint(
        optimizer_step=3,
        current_version=3,
        published_version=3,
        dataset_cursor=before,
        rng=run_rng,
    )
    resumed_rng = restore_rng(checkpoint)
    suffix = tuple(int(x) for x in resumed_rng.integers(0, 10_000, size=after))
    resumed = prefix + suffix

    # A common broken recovery restores model weights but silently reinitializes RNG.
    broken_rng = np.random.default_rng(seed + 1)
    broken = prefix + tuple(int(x) for x in broken_rng.integers(0, 10_000, size=after))
    return RecoveryResult(
        uninterrupted=uninterrupted,
        resumed=resumed,
        exact_match=resumed == uninterrupted,
        broken_resume=broken,
        broken_match=broken == uninterrupted,
    )


def checkpoint_as_dict(checkpoint: CheckpointState) -> dict:
    """JSON-ready checkpoint manifest used by the notebook."""

    return asdict(checkpoint)


__all__ = [
    "BatchRecord",
    "CheckpointState",
    "PipelineConfig",
    "RecoveryResult",
    "RolloutRecord",
    "SimulationResult",
    "checkpoint_as_dict",
    "make_checkpoint",
    "recovery_drill",
    "restore_rng",
    "result_row",
    "simulate_pipeline",
]
