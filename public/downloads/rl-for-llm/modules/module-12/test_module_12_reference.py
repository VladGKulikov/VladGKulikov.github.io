#!/usr/bin/env python3
"""Deterministic tests for Module 12 reference calculations."""
from __future__ import annotations

from math import isclose, log
from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_12_reference import (  # noqa: E402
    BYTES_PER_GIB,
    checkpoint_cadence,
    checkpoint_manifest_errors,
    completion_token_mask,
    diagnose_pipeline,
    effective_sample_size,
    human_bytes,
    ideal_sharded_memory,
    importance_weights,
    informative_throughput,
    kv_cache_bytes,
    model_state_bytes,
    optimal_checkpoint_interval,
    optimal_sync_every,
    peak_memory,
    pipeline_report,
    policy_lag,
    queue_is_stable,
    queue_load,
    rollout_contract_errors,
    sequence_log_importance,
    straggler_report,
    synchronization_cadence_cost,
    synchronization_schedule,
    synchronization_transfer_time,
)

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(label)
    PASSED += 1
    print(f"[PASS] {label}")


def main() -> None:
    profile_12 = model_state_bytes(
        7_000_000_000,
        parameter_bytes=2,
        gradient_bytes=2,
        optimizer_state_bytes=8,
        master_parameter_bytes=0,
    )
    check(profile_12.total_bytes == 84_000_000_000, "7B 12-byte model-state profile")

    profile_16 = model_state_bytes(
        7_000_000_000,
        parameter_bytes=2,
        gradient_bytes=2,
        optimizer_state_bytes=8,
        master_parameter_bytes=4,
    )
    check(profile_16.total_bytes == 112_000_000_000, "7B 16-byte model-state profile")
    check(profile_16.parameter_bytes == 14_000_000_000, "7B bf16 base-weight footprint")

    lora = model_state_bytes(
        7_020_000_000,
        trainable_parameters=20_000_000,
        parameter_bytes=2,
        gradient_bytes=2,
        optimizer_state_bytes=8,
        master_parameter_bytes=4,
    )
    check(lora.total_bytes == 14_320_000_000, "7B base plus 20M added adapters: all weights and adapter state")
    check(lora.parameter_bytes == 14_040_000_000, "total_parameters includes added adapter weights")
    suffix = np.array([[9, 2, 2, 2], [4, 5, 6, 7], [3, 0, 0, 0], [2, 2, 2, 2]])
    mask = completion_token_mask(suffix, [2, 3])
    check(mask.sum(axis=1).tolist() == [2, 4, 1, 1], "suffix lengths include first EOS and exclude padding")
    check(completion_token_mask(np.empty((2, 0), dtype=int), [2]).shape == (2, 0), "empty suffix")

    sharded = ideal_sharded_memory(
        profile_16.total_bytes,
        8,
        replicated_bytes=2_000_000_000,
        temporary_peak_bytes=4_000_000_000,
    )
    check(sharded == 20_000_000_000, "idealized sharded footprint plus replicated peak")

    one_token_kv = kv_cache_bytes(
        num_sequences=1,
        tokens_per_sequence=1,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        element_bytes=2,
    )
    check(one_token_kv == 131_072, "KV cache bytes per token")

    kv_32gib = kv_cache_bytes(
        num_sequences=64,
        tokens_per_sequence=4096,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        element_bytes=2,
    )
    check(kv_32gib == 32 * BYTES_PER_GIB, "64x4096 KV example equals 32 GiB")

    kv_tp4 = kv_cache_bytes(
        num_sequences=64,
        tokens_per_sequence=4096,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        element_bytes=2,
        tensor_parallel_size=4,
    )
    check(kv_tp4 == 8 * BYTES_PER_GIB, "idealized TP4 KV payload")

    peak = peak_memory(
        {
            "rollout": [14_000_000_000, 8 * BYTES_PER_GIB],
            "train": [20_000_000_000, 18_000_000_000],
            "sync": [14_000_000_000, 14_000_000_000, 2_000_000_000],
        }
    )
    check(peak.phase == "train" and peak.peak_bytes == 38_000_000_000, "phase-specific peak memory")

    check(isclose(informative_throughput(12.5, 0.9), 1.25), "informative group throughput")

    pipe_one = pipeline_report(90.0, 10.0, 1)
    check(isclose(pipe_one.pipeline_total, 100.0) and isclose(pipe_one.speedup, 1.0), "one batch has no overlap gain")

    pipe_twenty = pipeline_report(90.0, 10.0, 20)
    check(isclose(pipe_twenty.pipeline_total, 1810.0), "twenty-batch ideal pipeline time")
    check(isclose(pipe_twenty.steady_state_period, 90.0), "pipeline period is the slow stage")
    check(isclose((90.0 + 10.0) / 90.0, 1.1111111111111112), "steady-state period speedup example")

    straggler = straggler_report([4, 4, 5, 5, 5, 6, 6, 20])
    check(isclose(straggler.total_work, 55.0) and isclose(straggler.wall_clock, 20.0), "straggler work and wall time")
    check(isclose(straggler.utilization, 55 / 160), "straggler utilization 55/160")

    check(isclose(queue_load(8.0, 10.0), 0.8), "queue load")
    check(queue_is_stable(8.0, 10.0), "queue stability when arrival is slower")
    check(not queue_is_stable(10.0, 10.0), "strict stability excludes equal rates")

    current = [-0.4, -1.1, -0.2]
    behavior = [-0.5, -1.0, -0.3]
    check(isclose(sequence_log_importance(current, behavior), 0.1), "sequence log importance ratio")
    weights = importance_weights(current, behavior)
    check(np.allclose(weights, np.exp([0.1, -0.1, 0.1])), "pointwise importance weights")
    clipped = importance_weights(current, behavior, min_log_ratio=-0.05, max_log_ratio=0.05)
    check(np.allclose(clipped, np.exp([0.05, -0.05, 0.05])), "clipped log importance ratios")

    check(isclose(effective_sample_size([1, 1, 1, 1]), 4.0), "ESS of equal weights")
    check(isclose(effective_sample_size([1, 1, 1, 1], normalized=True), 1.0), "normalized ESS of equal weights")
    concentrated = effective_sample_size([100.0, 1.0, 1.0, 1.0])
    check(1.0 < concentrated < 1.1, "ESS detects a dominant weight")

    lag = policy_lag(9, [9, 8, 7, 9])
    check(lag.lags == (0, 1, 2, 0) and isclose(lag.mean_lag, 0.75), "policy lag summary")
    check(lag.max_lag == 2 and isclose(lag.stale_fraction, 0.5), "maximum lag and stale fraction")

    transfer = synchronization_transfer_time(
        14_000_000_000,
        100.0,
        setup_seconds=0.05,
        efficiency=0.7,
    )
    check(isclose(transfer, 0.25), "first-order 14 GB transfer time")

    schedule = synchronization_schedule(4, 10)
    check(schedule.lag_sequence == (0, 1, 2, 3, 0, 1, 2, 3, 0, 1), "periodic synchronization lag sequence")
    check(schedule.synchronization_count == 3 and isclose(schedule.mean_lag, 1.3), "synchronization count and finite-run mean lag")

    check(isclose(synchronization_cadence_cost(4, sync_seconds=8.0, lag_cost_per_version=1.0), 3.5), "sync cadence teaching cost")
    winner, cost = optimal_sync_every([1, 2, 4, 8, 16], sync_seconds=8.0, lag_cost_per_version=1.0)
    check(winner == 4 and isclose(cost, 3.5), "optimal cadence among discrete candidates")

    interval = optimal_checkpoint_interval(120.0, 86_400.0)
    check(isclose(interval, 4553.679830642467), "checkpoint interval sqrt(2CM)")
    cadence = checkpoint_cadence(interval, checkpoint_seconds=120.0, mtbf_seconds=86_400.0)
    check(isclose(cadence.write_fraction, cadence.expected_lost_work_fraction), "checkpoint optimum balances two first-order costs")

    valid_rollout = {
        "sample_id": "run42-p008-g03",
        "prompt_id": "p008",
        "policy_version": 127,
        "prompt_tokens": [101, 42, 17],
        "response_tokens": [314, 159, 2],
        "behavior_logprobs": [-0.31, -1.07, -0.22],
        "decoding": {"temperature": 0.8, "top_p": 1.0},
        "reward": {"correctness": 1.0, "format": 0.0},
        "stop_reason": "eos",
        "verifier_version": "arith-v4",
    }
    check(rollout_contract_errors(valid_rollout) == [], "valid rollout contract")
    broken_rollout = dict(valid_rollout, behavior_logprobs=[-0.31, -1.07])
    check(any("align" in error for error in rollout_contract_errors(broken_rollout)), "rollout logprob alignment error")

    valid_checkpoint = {
        "model_state": "model.safetensors",
        "optimizer_state": "optimizer.pt",
        "scheduler_state": "scheduler.pt",
        "global_step": 2048,
        "policy_version": 512,
        "rng_state": {"python": "...", "numpy": "...", "torch": "..."},
        "data_cursor": {"epoch": 3, "offset": 1200},
        "rollout_schema_version": "2",
        "tokenizer_revision": "abc123",
        "verifier_version": "arith-v4",
        "decoding_config_hash": "sha256:deadbeef",
    }
    check(checkpoint_manifest_errors(valid_checkpoint) == [], "valid checkpoint manifest")
    broken_checkpoint = dict(valid_checkpoint)
    broken_checkpoint.pop("optimizer_state")
    check(any("optimizer_state" in error for error in checkpoint_manifest_errors(broken_checkpoint)), "missing optimizer state is detected")

    hypotheses = diagnose_pipeline(
        {
            "trainer_idle_fraction": 0.35,
            "rollout_idle_fraction": 0.05,
            "verifier_fraction": 0.40,
            "weight_sync_fraction": 0.25,
            "queue_growth_per_minute": 0.0,
            "normalized_ess": 0.40,
            "oom_events": 2.0,
        }
    )
    check(len(hypotheses) == 5, "pipeline diagnostics return five testable hypotheses")
    neutral = diagnose_pipeline(
        {
            "trainer_idle_fraction": 0.05,
            "rollout_idle_fraction": 0.05,
            "verifier_fraction": 0.05,
            "weight_sync_fraction": 0.05,
            "queue_growth_per_minute": 0.0,
            "normalized_ess": 0.95,
            "oom_events": 0.0,
        }
    )
    check(neutral == ["no dominant failure is identified; inspect a stage-level trace"], "neutral diagnostic requests a trace")

    check(human_bytes(32 * BYTES_PER_GIB) == "32.000 GiB", "binary byte formatter")

    invalid_calls = (
        lambda: model_state_bytes(0),
        lambda: model_state_bytes(10, trainable_parameters=11),
        lambda: ideal_sharded_memory(1, 0),
        lambda: kv_cache_bytes(num_sequences=0, tokens_per_sequence=1, num_layers=1, num_kv_heads=1, head_dim=1, element_bytes=2),
        lambda: peak_memory({}),
        lambda: informative_throughput(1.0, 1.2),
        lambda: pipeline_report(0.0, 1.0, 1),
        lambda: straggler_report([1.0, 2.0], workers=1),
        lambda: queue_load(1.0, 0.0),
        lambda: sequence_log_importance([-0.1], [-0.1, -0.2]),
        lambda: importance_weights([0.1], [-0.2]),
        lambda: effective_sample_size([0.0, 0.0]),
        lambda: policy_lag(3, [4]),
        lambda: synchronization_transfer_time(1, 0.0),
        lambda: synchronization_schedule(0, 2),
        lambda: optimal_checkpoint_interval(0.0, 1.0),
        lambda: checkpoint_manifest_errors([]),  # type: ignore[arg-type]
        lambda: diagnose_pipeline({}),
    )
    for call in invalid_calls:
        try:
            result = call()
            if isinstance(result, list) and result:
                continue
        except (ValueError, TypeError):
            continue
        raise AssertionError("invalid-input check did not fail")
    check(True, "invalid inputs are rejected or reported")

    print(f"\n{PASSED} checks passed")


if __name__ == "__main__":
    main()
