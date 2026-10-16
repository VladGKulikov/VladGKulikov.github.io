#!/usr/bin/env python3
"""Deterministic tests for the Module 10 reasoning-maze search laboratory."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from module_10_search import (  # noqa: E402
    CallBudget,
    ReasoningMaze,
    beam_search,
    best_first_search,
    best_of_n_proxy,
    evaluate_search_methods,
    maze_to_array,
    mcts_search,
    repeated_sampling,
    sample_trajectory,
)

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    if not condition:
        raise AssertionError(label)
    PASSED += 1
    print(f"[PASS] {label}")


def main() -> None:
    env = ReasoningMaze.course_maze()
    state = env.initial_state()
    check(env.shortest_distance(env.start) == 12, "maze has a known shortest route")
    check(set(env.available_actions(state)) == {"U", "R"}, "initial legal actions are correct")
    actions, probs = env.proposal_distribution(state)
    check(len(actions) == len(probs) and np.isclose(probs.sum(), 1.0), "proposal distribution is normalized")
    check(np.all(probs > 0), "proposal policy has support on every legal action")
    check(env.process_score(state, mode="good", seed=2) == env.process_score(state, mode="good", seed=2), "prefix score is deterministic")

    budget = CallBudget(18)
    trajectory = sample_trajectory(env, budget, np.random.default_rng(0))
    check(budget.used <= 18 and trajectory.depth == budget.used, "trajectory respects the model-call budget")
    check(len(trajectory.path) == trajectory.depth + 1, "trajectory stores every visited state")
    check(len(set(trajectory.path)) == len(trajectory.path), "fresh moves prevent avoidable oscillation")

    repeated = repeated_sampling(env, 64, seed=0)
    bon = best_of_n_proxy(env, 64, seed=0)
    check(repeated.coverage and repeated.success, "repeated sampling can find an exact success")
    check(bon.coverage, "Best-of-N records candidate coverage separately")
    check(bon.model_calls <= 64 and repeated.model_calls <= 64, "sampling methods respect call budgets")

    beam = beam_search(env, 64, beam_width=3, score_mode="good", seed=0)
    best = best_first_search(env, 24, score_mode="good", seed=0)
    mcts = mcts_search(env, 32, score_mode="good", seed=0)
    check(beam.success and env.is_goal(beam.selected_state), "beam search reaches the goal with sufficient budget")
    check(best.success and env.is_goal(best.selected_state), "best-first search reaches the goal")
    check(mcts.success and env.is_goal(mcts.selected_state), "MCTS reaches the goal")
    check(beam.model_calls <= 64 and best.model_calls <= 24 and mcts.model_calls <= 32, "search methods respect call budgets")

    budgets = [12, 16, 24, 32, 48]
    good = evaluate_search_methods(budgets, seeds=range(20), score_mode="good")
    biased = evaluate_search_methods(budgets, seeds=range(20), score_mode="biased")
    check(set(good) == {"repeated", "best_of_n_proxy", "beam", "best_first", "mcts"}, "benchmark returns all methods")
    check(all(values.shape == (5,) for values in good.values()), "benchmark curves match budget shape")
    check(all(np.all((0.0 <= values) & (values <= 1.0)) for values in good.values()), "success rates lie in [0, 1]")
    check(good["best_first"][1] >= biased["best_first"][1], "biased prefix score does not improve early best-first success")
    check(good["beam"][3] >= biased["beam"][3], "biased scorer can hurt early beam pruning")

    grid = maze_to_array(env, best.selected_state)
    check(grid.shape == (7, 7), "maze renderer returns the expected grid")
    check(grid[env.start] == 2 and grid[env.goal] == 3 and grid[env.decoy] == 4, "maze renderer preserves semantic markers")

    invalid_calls = (
        lambda: CallBudget(0),
        lambda: env.proposal_distribution(state, temperature=0),
        lambda: beam_search(env, 10, beam_width=0),
        lambda: mcts_search(env, 10, exploration=-1),
        lambda: evaluate_search_methods([], seeds=[0]),
        lambda: evaluate_search_methods([10], seeds=[]),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid-input check did not fail")
    check(True, "invalid search inputs raise ValueError")
    print(f"\n{PASSED} tests passed")


if __name__ == "__main__":
    main()
