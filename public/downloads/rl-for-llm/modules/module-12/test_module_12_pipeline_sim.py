#!/usr/bin/env python3
"""Tests for the Module 12 infrastructure simulator."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from module_12_pipeline_sim import (
    PipelineConfig,
    checkpoint_as_dict,
    make_checkpoint,
    recovery_drill,
    restore_rng,
    result_row,
    simulate_pipeline,
)


def test_sequential_has_zero_lag_and_unit_ess() -> None:
    result = simulate_pipeline(PipelineConfig(mode="sequential", duration_seconds=60.0))
    assert result.updates > 0
    assert result.mean_policy_lag == 0.0
    assert result.min_normalized_ess == 1.0
    assert result.discarded_rollouts == 0


def test_async_produces_updates_and_version_lag() -> None:
    result = simulate_pipeline(PipelineConfig(mode="bounded_async", duration_seconds=60.0))
    assert result.updates > 0
    assert result.trained_tokens > 0
    assert result.mean_policy_lag > 0.0
    assert 0.0 < result.min_normalized_ess <= 1.0


def test_async_is_deterministic() -> None:
    cfg = PipelineConfig(mode="bounded_async", duration_seconds=30.0, seed=7)
    first = simulate_pipeline(cfg)
    second = simulate_pipeline(cfg)
    assert first == second


def test_queue_cap_creates_backpressure_under_overload() -> None:
    cfg = PipelineConfig(
        mode="bounded_async",
        duration_seconds=30.0,
        rollout_workers=12,
        trainer_tokens_per_second=900.0,
        max_queue_tokens=4000,
        max_policy_lag=8,
        sync_every_updates=1,
    )
    result = simulate_pipeline(cfg)
    assert result.backpressure_seconds > 0.0
    # In-flight completions may overshoot a hard start-admission cap.
    assert result.max_queue_tokens_observed >= cfg.max_queue_tokens


def test_stale_contract_discards_records() -> None:
    cfg = PipelineConfig(
        mode="bounded_async",
        duration_seconds=60.0,
        sync_every_updates=3,
        max_policy_lag=1,
    )
    result = simulate_pipeline(cfg)
    assert result.discarded_rollouts > 0


def test_more_rollout_workers_raise_generated_throughput() -> None:
    base = PipelineConfig(mode="bounded_async", duration_seconds=40.0, rollout_workers=4)
    more = replace(base, rollout_workers=8)
    r1 = simulate_pipeline(base)
    r2 = simulate_pipeline(more)
    assert r2.generated_tokens_per_second > r1.generated_tokens_per_second


def test_result_row_has_stable_schema() -> None:
    row = result_row(simulate_pipeline(PipelineConfig(mode="sequential", duration_seconds=20.0)))
    assert set(row) == {
        "mode",
        "sync_every",
        "queue_cap",
        "max_lag_contract",
        "updates",
        "trained_tok_s",
        "generated_tok_s",
        "mean_lag",
        "p95_lag",
        "mean_nESS",
        "min_nESS",
        "discarded_rollouts",
        "max_queue_tokens",
        "backpressure_s",
    }


def test_checkpoint_restores_exact_rng_stream() -> None:
    rng = np.random.default_rng(42)
    _ = rng.integers(0, 100, size=5)
    checkpoint = make_checkpoint(
        optimizer_step=3,
        current_version=3,
        published_version=2,
        dataset_cursor=17,
        rng=rng,
    )
    expected = rng.integers(0, 100, size=10)
    restored = restore_rng(checkpoint)
    actual = restored.integers(0, 100, size=10)
    assert np.array_equal(expected, actual)


def test_checkpoint_manifest_is_json_ready() -> None:
    rng = np.random.default_rng(1)
    checkpoint = make_checkpoint(
        optimizer_step=1,
        current_version=1,
        published_version=1,
        dataset_cursor=2,
        rng=rng,
    )
    manifest = checkpoint_as_dict(checkpoint)
    assert manifest["rollout_schema_version"] == "rollout.v3"
    assert manifest["rng_state"]["bit_generator"] == "PCG64"


def test_checkpoint_rejects_future_published_version() -> None:
    with pytest.raises(ValueError):
        make_checkpoint(
            optimizer_step=1,
            current_version=1,
            published_version=2,
            dataset_cursor=0,
            rng=np.random.default_rng(0),
        )


def test_recovery_drill_distinguishes_complete_and_broken_resume() -> None:
    result = recovery_drill(seed=123, before=5, after=7)
    assert result.exact_match
    assert not result.broken_match
    assert result.resumed == result.uninterrupted
    assert result.broken_resume != result.uninterrupted


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_seconds", 0.0),
        ("dt_seconds", 0.0),
        ("rollout_workers", 0),
        ("rollout_tokens_per_second", 0.0),
        ("trainer_tokens_per_second", 0.0),
        ("batch_tokens", 0),
        ("max_queue_tokens", 0),
        ("max_policy_lag", -1),
    ],
)
def test_invalid_config_is_rejected(field: str, value: float | int) -> None:
    cfg = replace(PipelineConfig(), **{field: value})
    with pytest.raises(ValueError):
        simulate_pipeline(cfg)


def test_timeline_is_monotone_in_time() -> None:
    result = simulate_pipeline(PipelineConfig(mode="bounded_async", duration_seconds=20.0))
    times = [row["time"] for row in result.timeline]
    assert times == sorted(times)


def test_batch_versions_advance_monotonically() -> None:
    result = simulate_pipeline(PipelineConfig(mode="bounded_async", duration_seconds=60.0))
    updates = [batch.update for batch in result.batches]
    assert updates == list(range(1, len(updates) + 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
