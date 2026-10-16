#!/usr/bin/env python3
"""Deterministic tests for Module 11 reference mathematics and contracts."""
from __future__ import annotations

from math import isclose
from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_11_reference import (  # noqa: E402
    build_policy_mask,
    collector_target,
    degenerate_group_probability,
    discounted_return,
    discounted_td_sum,
    expected_degenerate_fraction,
    expected_groups_per_informative,
    first_visit_milestone_rewards,
    informative_group_probability,
    loss_reductions,
    macro_action_logprob,
    masked_weighted_mean,
    minimum_group_size_for_informative_probability,
    potential_shaping,
    provenance_mask,
    run_keydoor_episode,
    semantic_action_probability,
    semi_markov_return,
    semi_markov_terminal_return,
    shaped_return_identity,
    synchronous_straggler_report,
    td_residuals,
    trajectory_mask_report,
    turn_probability_ratio,
    validate_rollout_record,
)

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(label)
    PASSED += 1
    print(f"[PASS] {label}")


def valid_record() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "environment_id": "keydoor",
        "environment_version": "1",
        "token_ids": [10, 11, 12, 13],
        "policy_mask": [0, 1, 1, 0],
        "behavior_logprobs": [-0.2, -0.4],
        "policy_versions": {"1": "p-7"},
        "turn_ids": [0, 1, 1, 1],
        "action_ids": [2, 4],
        "observations": ["start", "clue_red", "opened_red"],
        "rewards": [0.0, 1.0],
        "stop_reason": "success",
        "terminated": True,
        "truncated": False,
        "infra_error": False,
    }


def main() -> None:
    mask = build_policy_mask(["user", "assistant", "tool", "assistant"], [2, 3, 4, 1])
    check(mask == [0, 0, 1, 1, 1, 0, 0, 0, 0, 1], "role-based token mask")
    check(np.array_equal(provenance_mask(["template", "policy", "environment", "controller"]), [0, 1, 0, 0]), "provenance mask")

    report = trajectory_mask_report([("system", 20), ("assistant", 30), ("tool", 270), ("assistant", 10)])
    check(report.policy_tokens == 40 and report.total_tokens == 330, "trajectory token counts")
    check(isclose(report.masked_fraction, 290 / 330), "trajectory masked fraction")
    check(isclose(report.dilution_factor, 8.25), "trajectory dilution factor")
    check(isclose(masked_weighted_mean([2, 10, 4, 20], [1, 0, 1, 0]), 3.0), "masked mean")
    check(isclose(macro_action_logprob([-0.2, -0.5, -0.1]), -0.8), "macro-action log-probability")
    check(isclose(turn_probability_ratio([-0.1, -0.4], [-0.2, -0.5]), np.exp(0.2)), "turn probability ratio")
    check(isclose(turn_probability_ratio([-100.0], [-1.0], max_log_ratio=8.0), np.exp(-8.0)), "ratio clipping in log space")

    string_probs = {'{"x":1}': 0.25, '{"x": 1}': 0.35, "stop": 0.40}
    parser = {'{"x":1}': "call_x_1", '{"x": 1}': "call_x_1", "stop": "stop"}
    check(isclose(semantic_action_probability(string_probs, parser, "call_x_1"), 0.60), "many-to-one parser probability")

    rewards = [0.0, 0.0, 1.0]
    values = [0.20, 0.35, 0.60, 0.0]
    td = td_residuals(rewards, values, 0.9, terminated=[False, False, True])
    check(np.allclose(td, [0.115, 0.19, 0.4]), "terminal-reward TD residuals")
    check(isclose(discounted_td_sum(td, 0.9), 0.61), "discounted TD residual sum")
    check(isclose(discounted_return(rewards, 0.9) - values[0], 0.61), "TD telescoping identity")
    check(isclose(collector_target(0.0, 0.7, gamma=0.9, terminated=False), 0.63), "collector boundary bootstraps")
    check(isclose(collector_target(0.0, 0.7, gamma=0.9, terminated=True), 0.0), "true terminal removes bootstrap")

    shaped = potential_shaping([0.0, 0.0, 1.0], [0.0, 0.3, 0.6, 0.0], 0.9)
    check(np.allclose(shaped, [0.27, 0.24, 0.4]), "potential-shaped rewards")
    original, shaped_return, predicted = shaped_return_identity([0.0, 0.0, 1.0], [0.0, 0.3, 0.6, 0.0], 0.9)
    check(isclose(original, 0.81) and isclose(shaped_return, 0.81) and isclose(predicted, 0.81), "shaped-return identity")
    shifted = shaped_return_identity([0.0, 0.0, 1.0], [0.2, 0.3, 0.6, 0.0], 0.9)
    check(isclose(shifted[1], 0.61), "initial potential shifts finite-horizon return")

    check(isclose(informative_group_probability(0.01, 8), 0.07725530557207982), "rare-success group informativeness")
    check(isclose(degenerate_group_probability(0.5, 8), 1 / 128), "balanced group degeneracy")
    check(isclose(expected_groups_per_informative(0.05, 16), 1.786118294993429), "expected groups per informative group")
    heterogeneous = expected_degenerate_fraction([0.01, 0.5, 0.99], 8)
    check(heterogeneous > 0.61 and heterogeneous > degenerate_group_probability(0.5, 8), "prompt-conditional degeneracy")
    check(minimum_group_size_for_informative_probability(0.02, 0.5) == 35, "minimum informative group size")

    reductions = loss_reductions([[0.2, 0.2], [0.1] * 8])
    check(isclose(reductions["token_mean"], 0.12), "token-level reduction")
    check(isclose(reductions["trajectory_mean"], 0.15), "trajectory-level reduction")
    check(isclose(semi_markov_return([0.0, 1.0], [2, 3], 0.9), 0.59049), "semi-Markov completion-time return")
    check(isclose(semi_markov_terminal_return([2, 3], 1.0, 0.9), 0.59049), "semi-Markov terminal wrapper")

    straggler = synchronous_straggler_report([1.0, 1.0, 2.0, 8.0])
    check(isclose(straggler.total_work, 12.0) and isclose(straggler.wall_clock, 8.0), "straggler work and wall clock")
    check(isclose(straggler.utilization, 0.375) and isclose(straggler.idle_fraction, 0.625), "synchronous utilization")

    milestone = first_visit_milestone_rewards(["start", "login", "search", "login", "checkout"], {"login": 0.2, "search": 0.3, "checkout": 0.5})
    check(np.allclose(milestone, [0.0, 0.2, 0.3, 0.0, 0.5]), "first-visit milestone rewards")

    record = valid_record()
    check(validate_rollout_record(record) == [], "valid rollout record")
    check(any("terminal status" in e for e in validate_rollout_record(dict(record, truncated=True))), "terminal-status conflict")
    check(any("zeros and ones" in e for e in validate_rollout_record(dict(record, policy_mask=[0, 1, 2, 0]))), "invalid mask")
    check(any("missing policy versions" in e for e in validate_rollout_record(dict(record, policy_versions={}))), "missing policy version")
    check(any("one more item" in e for e in validate_rollout_record(dict(record, observations=["start"]))), "turn-shape mismatch")
    check(any("stop_reason" in e for e in validate_rollout_record(dict(record, terminated=False, stop_reason="success"))), "orphan stop reason")

    success = run_keydoor_episode(["inspect", "pickup_red", "open_red", "finish"], key_color="red", door_color="red")
    check(success["success"] and success["reward"] == 1.0 and success["terminated"], "successful key-and-door episode")
    early = run_keydoor_episode(["finish"], key_color="red", door_color="red")
    check(not early["success"] and early["terminated"] and not early["truncated"], "early finish is task failure")
    wrong = run_keydoor_episode(["inspect", "pickup_blue", "open_blue", "finish"], key_color="red", door_color="red")
    check(wrong["invalid_actions"] >= 2 and not wrong["success"], "wrong key is detected")
    truncated = run_keydoor_episode(["inspect", "inspect", "inspect"], key_color="blue", door_color="blue", max_steps=3)
    check(truncated["truncated"] and not truncated["terminated"], "unfinished episode is truncated")

    invalid_calls = (
        lambda: build_policy_mask(["assistant"], [-1]),
        lambda: provenance_mask(["mystery"]),
        lambda: trajectory_mask_report([("tool", 10)]),
        lambda: masked_weighted_mean([1.0], [0]),
        lambda: turn_probability_ratio([-0.2], [-0.2, -0.3]),
        lambda: semantic_action_probability({"a": 0.5}, {"a": "x"}, "x"),
        lambda: td_residuals([1.0], [0.0], 0.9),
        lambda: potential_shaping([1.0], [0.0], 0.9),
        lambda: informative_group_probability(1.2, 4),
        lambda: minimum_group_size_for_informative_probability(0.0, 0.5),
        lambda: semi_markov_return([1.0], [0], 0.9),
        lambda: synchronous_straggler_report([1.0, 0.0]),
        lambda: run_keydoor_episode([], key_color="green", door_color="red"),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid-input check did not fail")
    check(True, "invalid inputs raise ValueError")

    print(f"\n{PASSED} tests passed")


if __name__ == "__main__":
    main()
