#!/usr/bin/env python3
"""Minimal but realistic test-time search laboratory for Module 10.

The environment is a small grid "reasoning maze".  A frozen proposal policy
plays the role of an LLM that proposes the next reasoning step.  An exact goal
check plays the role of an outcome verifier, while a configurable prefix scorer
plays the role of a process reward/value model.  The module compares repeated
sampling, Best-of-N selection, beam search, best-first search, and a compact
PUCT-style Monte Carlo tree search under an explicit model-call budget.

The environment is deliberately small so that every result is reproducible on
CPU.  It is not presented as a benchmark for language models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import exp, isfinite, log, sqrt
from collections import deque
from typing import Iterable, Literal, Sequence
import hashlib

import numpy as np

Action = Literal["U", "D", "L", "R"]
ACTIONS: tuple[Action, ...] = ("U", "D", "L", "R")
DELTAS: dict[Action, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class MazeState:
    position: tuple[int, int]
    depth: int = 0
    path: tuple[tuple[int, int], ...] = ()
    actions: tuple[Action, ...] = ()


@dataclass(frozen=True)
class ReasoningMaze:
    height: int
    width: int
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: frozenset[tuple[int, int]]
    decoy: tuple[int, int]
    max_steps: int = 18
    model_decoy_bias: float = 0.9

    @classmethod
    def course_maze(cls, *, model_decoy_bias: float = 0.9) -> "ReasoningMaze":
        """Return the fixed maze used in the lecture and notebooks."""
        walls = frozenset({
            (0, 3), (1, 1), (1, 3), (1, 5),
            (2, 1), (2, 5),
            (3, 1), (3, 2), (3, 3), (3, 5),
            (4, 3), (4, 5),
            (5, 1), (5, 3),
        })
        return cls(
            height=7,
            width=7,
            start=(6, 0),
            goal=(0, 6),
            walls=walls,
            decoy=(0, 0),
            max_steps=18,
            model_decoy_bias=float(model_decoy_bias),
        )

    def initial_state(self) -> MazeState:
        return MazeState(position=self.start, depth=0, path=(self.start,), actions=())

    def is_goal(self, state: MazeState) -> bool:
        return state.position == self.goal

    def is_terminal(self, state: MazeState) -> bool:
        return self.is_goal(state) or state.depth >= self.max_steps

    def in_bounds(self, position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width

    def available_actions(self, state: MazeState) -> tuple[Action, ...]:
        if self.is_terminal(state):
            return ()
        valid: list[Action] = []
        fresh: list[Action] = []
        for action in ACTIONS:
            dr, dc = DELTAS[action]
            nxt = (state.position[0] + dr, state.position[1] + dc)
            if self.in_bounds(nxt) and nxt not in self.walls:
                valid.append(action)
                if nxt not in state.path:
                    fresh.append(action)
        # A reasoning trace normally should not oscillate between identical
        # prefixes.  Revisit only when every legal move would revisit.
        return tuple(fresh if fresh else valid)

    def step(self, state: MazeState, action: Action) -> MazeState:
        if action not in self.available_actions(state):
            raise ValueError(f"action {action!r} is not available from {state.position}")
        dr, dc = DELTAS[action]
        nxt = (state.position[0] + dr, state.position[1] + dc)
        return MazeState(
            position=nxt,
            depth=state.depth + 1,
            path=state.path + (nxt,),
            actions=state.actions + (action,),
        )

    def shortest_distance(self, position: tuple[int, int]) -> int | None:
        """Exact unweighted shortest-path distance to the goal."""
        if position == self.goal:
            return 0
        queue: deque[tuple[tuple[int, int], int]] = deque([(position, 0)])
        seen = {position}
        while queue:
            current, distance = queue.popleft()
            for action in ACTIONS:
                dr, dc = DELTAS[action]
                nxt = (current[0] + dr, current[1] + dc)
                if not self.in_bounds(nxt) or nxt in self.walls or nxt in seen:
                    continue
                if nxt == self.goal:
                    return distance + 1
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        return None

    def _stable_noise(self, state: MazeState, seed: int) -> float:
        payload = f"{seed}|{state.position}|{state.depth}|{state.actions}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        integer = int.from_bytes(digest, "little")
        return ((integer / (2**64 - 1)) - 0.5) * 2.0

    def proposal_distribution(self, state: MazeState, *, temperature: float = 1.0) -> tuple[tuple[Action, ...], np.ndarray]:
        """Frozen proposal policy over valid next steps.

        The policy mostly prefers progress toward the goal, but it is also
        attracted to a visually salient decoy.  This creates a controlled
        analogue of a model with a plausible but systematic reasoning bias.
        """
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        actions = self.available_actions(state)
        if not actions:
            return (), np.empty(0, dtype=np.float64)
        old_goal = abs(state.position[0] - self.goal[0]) + abs(state.position[1] - self.goal[1])
        old_decoy = abs(state.position[0] - self.decoy[0]) + abs(state.position[1] - self.decoy[1])
        logits: list[float] = []
        action_bias = {"U": 0.12, "R": 0.05, "D": -0.08, "L": -0.03}
        for action in actions:
            nxt = self.step(state, action).position
            goal = abs(nxt[0] - self.goal[0]) + abs(nxt[1] - self.goal[1])
            decoy = abs(nxt[0] - self.decoy[0]) + abs(nxt[1] - self.decoy[1])
            progress = old_goal - goal
            decoy_progress = old_decoy - decoy
            revisit = 1.0 if nxt in state.path else 0.0
            logits.append(1.15 * progress + self.model_decoy_bias * decoy_progress - 1.4 * revisit + action_bias[action])
        arr = np.asarray(logits, dtype=np.float64) / temperature
        arr -= arr.max()
        probs = np.exp(arr)
        probs /= probs.sum()
        return actions, probs

    def process_score(
        self,
        state: MazeState,
        *,
        mode: Literal["good", "biased"] = "good",
        seed: int = 0,
    ) -> float:
        """Score a prefix in [0, 1].

        ``good`` uses exact shortest-path geometry plus small deterministic
        noise.  ``biased`` adds a strong attraction to the decoy, modelling a
        process scorer that rewards a locally convincing but globally bad cue.
        """
        if self.is_goal(state):
            return 1.0
        distance = self.shortest_distance(state.position)
        if distance is None:
            base = 0.0
        else:
            remaining = max(self.max_steps - state.depth, 0)
            reachable = 1.0 if distance <= remaining else 0.0
            base = reachable * exp(-0.22 * distance)
        noise = 0.035 * self._stable_noise(state, seed)
        if mode == "good":
            score = base + noise
        elif mode == "biased":
            decoy_distance = abs(state.position[0] - self.decoy[0]) + abs(state.position[1] - self.decoy[1])
            decoy_bonus = 1.05 * exp(-0.52 * decoy_distance)
            score = 0.42 * base + decoy_bonus + noise
        else:
            raise ValueError("mode must be 'good' or 'biased'")
        return float(np.clip(score, 0.0, 1.0))

    def terminal_proxy(self, state: MazeState, *, seed: int = 0) -> float:
        """Imperfect final score used in the Best-of-N stress test."""
        success = 1.0 if self.is_goal(state) else 0.0
        decoy_distance = abs(state.position[0] - self.decoy[0]) + abs(state.position[1] - self.decoy[1])
        decoy_bonus = 1.15 * exp(-0.9 * decoy_distance)
        length_penalty = 0.015 * state.depth
        noise = 0.08 * self._stable_noise(state, seed + 991)
        return float(success + decoy_bonus - length_penalty + noise)


@dataclass
class CallBudget:
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or int(self.limit) != self.limit or self.limit <= 0:
            raise ValueError("limit must be a positive integer")
        self.limit = int(self.limit)

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self, amount: int = 1) -> bool:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.used + amount > self.limit:
            return False
        self.used += amount
        return True


@dataclass(frozen=True)
class SearchResult:
    method: str
    success: bool
    selected_state: MazeState
    model_calls: int
    generated_trajectories: int
    coverage: bool
    selected_proxy: float | None = None


def _sample_action(env: ReasoningMaze, state: MazeState, rng: np.random.Generator, temperature: float) -> Action:
    actions, probs = env.proposal_distribution(state, temperature=temperature)
    if not actions:
        raise RuntimeError("cannot sample an action from a terminal state")
    return actions[int(rng.choice(len(actions), p=probs))]


def sample_trajectory(
    env: ReasoningMaze,
    budget: CallBudget,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    start: MazeState | None = None,
) -> MazeState:
    state = env.initial_state() if start is None else start
    while not env.is_terminal(state) and budget.consume():
        state = env.step(state, _sample_action(env, state, rng, temperature))
    return state


def repeated_sampling(
    env: ReasoningMaze,
    model_calls: int,
    *,
    seed: int = 0,
    temperature: float = 1.0,
) -> SearchResult:
    """Sample complete trajectories until the model-call budget is exhausted.

    An exact outcome verifier returns the first successful trajectory if one is
    present.  Therefore selected success equals empirical coverage in this
    method; this is the oracle-verifier baseline, not majority voting.
    """
    budget = CallBudget(model_calls)
    rng = np.random.default_rng(seed)
    trajectories: list[MazeState] = []
    while budget.remaining > 0:
        before = budget.used
        state = sample_trajectory(env, budget, rng, temperature=temperature)
        if budget.used == before:
            break
        trajectories.append(state)
        if env.is_goal(state):
            return SearchResult("repeated", True, state, budget.used, len(trajectories), True)
    selected = trajectories[-1] if trajectories else env.initial_state()
    return SearchResult("repeated", False, selected, budget.used, len(trajectories), False)


def best_of_n_proxy(
    env: ReasoningMaze,
    model_calls: int,
    *,
    seed: int = 0,
    temperature: float = 1.0,
) -> SearchResult:
    """Generate trajectories and select the largest imperfect terminal score."""
    budget = CallBudget(model_calls)
    rng = np.random.default_rng(seed)
    trajectories: list[MazeState] = []
    while budget.remaining > 0:
        before = budget.used
        trajectories.append(sample_trajectory(env, budget, rng, temperature=temperature))
        if budget.used == before:
            trajectories.pop()
            break
    if not trajectories:
        initial = env.initial_state()
        return SearchResult("best_of_n_proxy", False, initial, budget.used, 0, False, env.terminal_proxy(initial, seed=seed))
    scores = np.asarray([env.terminal_proxy(state, seed=seed + i) for i, state in enumerate(trajectories)])
    selected = trajectories[int(np.argmax(scores))]
    return SearchResult(
        "best_of_n_proxy",
        env.is_goal(selected),
        selected,
        budget.used,
        len(trajectories),
        any(env.is_goal(s) for s in trajectories),
        float(scores.max()),
    )


def beam_search(
    env: ReasoningMaze,
    model_calls: int,
    *,
    beam_width: int = 4,
    score_mode: Literal["good", "biased"] = "good",
    seed: int = 0,
) -> SearchResult:
    if isinstance(beam_width, bool) or int(beam_width) != beam_width or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    budget = CallBudget(model_calls)
    beam = [env.initial_state()]
    best = beam[0]
    while beam and budget.remaining > 0:
        candidates: list[MazeState] = []
        for state in beam:
            if env.is_goal(state):
                return SearchResult("beam", True, state, budget.used, 0, True)
            if env.is_terminal(state) or not budget.consume():
                candidates.append(state)
                continue
            for action in env.available_actions(state):
                child = env.step(state, action)
                if env.is_goal(child):
                    return SearchResult("beam", True, child, budget.used, 0, True)
                candidates.append(child)
        if not candidates:
            break
        candidates.sort(
            key=lambda s: (env.process_score(s, mode=score_mode, seed=seed), -s.depth, s.actions),
            reverse=True,
        )
        beam = candidates[: int(beam_width)]
        if env.process_score(beam[0], mode=score_mode, seed=seed) > env.process_score(best, mode=score_mode, seed=seed):
            best = beam[0]
        if all(env.is_terminal(s) for s in beam):
            break
    success_states = [s for s in beam if env.is_goal(s)]
    selected = success_states[0] if success_states else best
    return SearchResult("beam", bool(success_states), selected, budget.used, 0, bool(success_states))


def best_first_search(
    env: ReasoningMaze,
    model_calls: int,
    *,
    score_mode: Literal["good", "biased"] = "good",
    seed: int = 0,
) -> SearchResult:
    budget = CallBudget(model_calls)
    initial = env.initial_state()
    counter = 0
    queue: list[tuple[float, int, MazeState]] = [(-env.process_score(initial, mode=score_mode, seed=seed), counter, initial)]
    best = initial
    seen_best_depth: dict[tuple[int, int], int] = {initial.position: 0}
    while queue and budget.remaining > 0:
        _, _, state = heappop(queue)
        if env.is_goal(state):
            return SearchResult("best_first", True, state, budget.used, 0, True)
        if env.is_terminal(state) or not budget.consume():
            continue
        for action in env.available_actions(state):
            child = env.step(state, action)
            if env.is_goal(child):
                return SearchResult("best_first", True, child, budget.used, 0, True)
            prior_depth = seen_best_depth.get(child.position)
            if prior_depth is not None and prior_depth <= child.depth:
                continue
            seen_best_depth[child.position] = child.depth
            counter += 1
            score = env.process_score(child, mode=score_mode, seed=seed)
            heappush(queue, (-score, counter, child))
            if score > env.process_score(best, mode=score_mode, seed=seed):
                best = child
    return SearchResult("best_first", False, best, budget.used, 0, False)


@dataclass
class _MCTSNode:
    state: MazeState
    parent: "_MCTSNode | None" = None
    action_from_parent: Action | None = None
    prior: float = 1.0
    children: dict[Action, "_MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def mcts_search(
    env: ReasoningMaze,
    model_calls: int,
    *,
    score_mode: Literal["good", "biased"] = "good",
    seed: int = 0,
    exploration: float = 1.25,
) -> SearchResult:
    """Compact PUCT-style MCTS with a frozen proposal prior and prefix value."""
    if not isfinite(exploration) or exploration < 0:
        raise ValueError("exploration must be finite and non-negative")
    budget = CallBudget(model_calls)
    root = _MCTSNode(env.initial_state())
    best = root.state
    found_goal: MazeState | None = None

    while budget.remaining > 0:
        node = root
        path = [node]
        # Selection through already expanded nodes.
        while node.expanded and node.children and not env.is_terminal(node.state):
            parent_visits = max(node.visits, 1)
            node = max(
                node.children.values(),
                key=lambda child: child.q + exploration * child.prior * sqrt(parent_visits) / (1 + child.visits),
            )
            path.append(node)

        # Expansion is one proposal-model call returning all legal actions.
        if not env.is_terminal(node.state) and budget.consume():
            actions, priors = env.proposal_distribution(node.state)
            for action, prior in zip(actions, priors):
                child_state = env.step(node.state, action)
                node.children[action] = _MCTSNode(
                    state=child_state,
                    parent=node,
                    action_from_parent=action,
                    prior=float(prior),
                )
                if env.is_goal(child_state):
                    found_goal = child_state
            node.expanded = True
            if node.children:
                # Evaluate the most promising newly created child first.
                node = max(
                    node.children.values(),
                    key=lambda child: env.process_score(child.state, mode=score_mode, seed=seed),
                )
                path.append(node)

        if env.is_goal(node.state):
            value = 1.0
            found_goal = node.state
        elif env.is_terminal(node.state):
            value = 0.0
        else:
            value = env.process_score(node.state, mode=score_mode, seed=seed)
        if value > env.process_score(best, mode=score_mode, seed=seed):
            best = node.state
        for item in reversed(path):
            item.visits += 1
            item.value_sum += value
        if found_goal is not None:
            return SearchResult("mcts", True, found_goal, budget.used, 0, True)

    return SearchResult("mcts", False, best, budget.used, 0, False)


def evaluate_search_methods(
    budgets: Sequence[int],
    *,
    seeds: Iterable[int] = range(100),
    score_mode: Literal["good", "biased"] = "good",
    model_decoy_bias: float = 0.9,
) -> dict[str, np.ndarray]:
    """Return success rates for the core methods over deterministic seeds."""
    budget_array = np.asarray(budgets, dtype=np.int64)
    if budget_array.ndim != 1 or budget_array.size == 0 or np.any(budget_array <= 0):
        raise ValueError("budgets must be a non-empty sequence of positive integers")
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    methods = ("repeated", "best_of_n_proxy", "beam", "best_first", "mcts")
    rows = {name: np.zeros(budget_array.size, dtype=np.float64) for name in methods}
    for j, calls in enumerate(budget_array):
        outcomes = {name: [] for name in methods}
        for seed in seed_list:
            env = ReasoningMaze.course_maze(model_decoy_bias=model_decoy_bias)
            outcomes["repeated"].append(repeated_sampling(env, int(calls), seed=seed).success)
            outcomes["best_of_n_proxy"].append(best_of_n_proxy(env, int(calls), seed=seed).success)
            outcomes["beam"].append(beam_search(env, int(calls), beam_width=3, score_mode=score_mode, seed=seed).success)
            outcomes["best_first"].append(best_first_search(env, int(calls), score_mode=score_mode, seed=seed).success)
            outcomes["mcts"].append(mcts_search(env, int(calls), score_mode=score_mode, seed=seed).success)
        for name in methods:
            rows[name][j] = float(np.mean(outcomes[name]))
    return rows


def maze_to_array(env: ReasoningMaze, state: MazeState | None = None) -> np.ndarray:
    """Integer grid for plotting: 0 free, 1 wall, 2 start, 3 goal, 4 decoy, 5 path."""
    grid = np.zeros((env.height, env.width), dtype=np.int8)
    for row, col in env.walls:
        grid[row, col] = 1
    grid[env.start] = 2
    grid[env.goal] = 3
    grid[env.decoy] = 4
    if state is not None:
        for pos in state.path[1:-1]:
            if grid[pos] == 0:
                grid[pos] = 5
        if state.position not in {env.start, env.goal, env.decoy}:
            grid[state.position] = 5
    return grid
