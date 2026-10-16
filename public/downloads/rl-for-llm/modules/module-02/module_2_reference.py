#!/usr/bin/env python3
"""Reference implementations for RL for LLM, Module 2.

The code deliberately uses small tabular environments and NumPy only.  It is
not a high-performance RL library; it is a compact, testable implementation of
value functions, Bellman operators, Monte Carlo prediction, temporal-
difference prediction, SARSA, Q-learning, and Double Q-learning.  The
implementations are intentionally explicit about terminal states, external
truncation, synchronous backups, and reproducible random-number streams.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _as_float_array(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_tabular_model(
    transitions: ArrayLike,
    rewards: ArrayLike,
    terminals: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Validate and normalize the representation used throughout the module.

    Parameters
    ----------
    transitions:
        ``P[s, a, s_next]`` transition probabilities.
    rewards:
        ``R[s, a, s_next]`` expected immediate rewards conditional on the
        transition.
    terminals:
        Boolean mask of terminal states.  Terminal values are fixed to zero.
    """
    p = _as_float_array(transitions, name="transitions", ndim=3)
    r = _as_float_array(rewards, name="rewards", ndim=3)
    if p.shape != r.shape:
        raise ValueError("transitions and rewards must have the same shape")
    n_states, n_actions, n_next = p.shape
    if n_states != n_next:
        raise ValueError("transitions must have shape (n_states, n_actions, n_states)")
    if np.any(p < 0.0):
        raise ValueError("transition probabilities must be non-negative")
    row_sums = p.sum(axis=2)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each transition row P[s,a,:] must sum to one")

    if terminals is None:
        terminal_mask = np.zeros(n_states, dtype=bool)
    else:
        terminal_mask = np.asarray(terminals, dtype=bool)
        if terminal_mask.shape != (n_states,):
            raise ValueError("terminals must have shape (n_states,)")
    return p, r, terminal_mask


def validate_policy(policy: ArrayLike, n_states: int | None = None, n_actions: int | None = None) -> FloatArray:
    pi = _as_float_array(policy, name="policy", ndim=2)
    if n_states is not None and pi.shape[0] != n_states:
        raise ValueError("policy has the wrong number of states")
    if n_actions is not None and pi.shape[1] != n_actions:
        raise ValueError("policy has the wrong number of actions")
    if np.any(pi < 0.0):
        raise ValueError("policy probabilities must be non-negative")
    if not np.allclose(pi.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("each policy row must sum to one")
    return pi


def v_from_q(q_values: ArrayLike, policy: ArrayLike) -> FloatArray:
    """Compute ``V(s) = sum_a pi(a|s) Q(s,a)``."""
    q = _as_float_array(q_values, name="q_values", ndim=2)
    pi = validate_policy(policy, *q.shape)
    return np.sum(pi * q, axis=1)


def advantage(q_values: ArrayLike, policy: ArrayLike) -> FloatArray:
    """Compute ``A(s,a) = Q(s,a) - V(s)``."""
    q = _as_float_array(q_values, name="q_values", ndim=2)
    return q - v_from_q(q, policy)[:, None]


def policy_induced_model(
    transitions: ArrayLike,
    rewards: ArrayLike,
    policy: ArrayLike,
    terminals: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Return ``P_pi`` and expected one-step reward vector ``r_pi``."""
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    pi = validate_policy(policy, p.shape[0], p.shape[1])
    p_pi = np.einsum("sa,san->sn", pi, p)
    r_pi = np.einsum("sa,san,san->s", pi, p, r)
    # Terminal values are represented as zero, so outgoing terminal rewards are
    # irrelevant.  Set the corresponding rows to an absorbing zero-reward form.
    p_pi = p_pi.copy()
    r_pi = r_pi.copy()
    for s in np.flatnonzero(terminal_mask):
        p_pi[s] = 0.0
        p_pi[s, s] = 1.0
        r_pi[s] = 0.0
    return p_pi, r_pi, terminal_mask


def exact_policy_value(
    transitions: ArrayLike,
    rewards: ArrayLike,
    policy: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
) -> FloatArray:
    """Solve the Bellman expectation equations as a linear system.

    The function supports ``gamma == 1`` only when the non-terminal linear
    system is nonsingular (for example, a proper episodic absorbing chain).
    """
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    p_pi, r_pi, terminal_mask = policy_induced_model(transitions, rewards, policy, terminals)
    nonterminal = np.flatnonzero(~terminal_mask)
    values = np.zeros(p_pi.shape[0], dtype=np.float64)
    if nonterminal.size == 0:
        return values
    p_nn = p_pi[np.ix_(nonterminal, nonterminal)]
    system = np.eye(nonterminal.size, dtype=np.float64) - gamma * p_nn
    try:
        values[nonterminal] = np.linalg.solve(system, r_pi[nonterminal])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the Bellman linear system is singular; use gamma < 1 or a proper episodic model"
        ) from exc
    return values


def q_from_v(
    transitions: ArrayLike,
    rewards: ArrayLike,
    values: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
) -> FloatArray:
    """Compute one Bellman action-value backup from a state-value vector."""
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    v = _as_float_array(values, name="values", ndim=1)
    if v.shape != (p.shape[0],):
        raise ValueError("values has the wrong shape")
    continuation = v.copy()
    continuation[terminal_mask] = 0.0
    q = np.sum(p * (r + gamma * continuation[None, None, :]), axis=2)
    q[terminal_mask] = 0.0
    return q


def iterative_policy_evaluation(
    transitions: ArrayLike,
    rewards: ArrayLike,
    policy: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[FloatArray, int]:
    """Synchronous iterative policy evaluation."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_sweeps, int) or isinstance(max_sweeps, bool) or max_sweeps <= 0:
        raise ValueError("max_sweeps must be a positive integer")
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    pi = validate_policy(policy, p.shape[0], p.shape[1])
    values = np.zeros(p.shape[0], dtype=np.float64)
    for sweep in range(1, max_sweeps + 1):
        q = q_from_v(p, r, values, gamma, terminal_mask)
        updated = np.sum(pi * q, axis=1)
        updated[terminal_mask] = 0.0
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < tolerance:
            return values, sweep
    raise RuntimeError("policy evaluation did not converge within max_sweeps")


def value_iteration(
    transitions: ArrayLike,
    rewards: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[FloatArray, int]:
    """Synchronous value iteration for a finite tabular MDP."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_sweeps, int) or isinstance(max_sweeps, bool) or max_sweeps <= 0:
        raise ValueError("max_sweeps must be a positive integer")
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    values = np.zeros(p.shape[0], dtype=np.float64)
    for sweep in range(1, max_sweeps + 1):
        q = q_from_v(p, r, values, gamma, terminal_mask)
        updated = np.max(q, axis=1)
        updated[terminal_mask] = 0.0
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < tolerance:
            return values, sweep
    raise RuntimeError("value iteration did not converge within max_sweeps")


def value_iteration_trace(
    transitions: ArrayLike,
    rewards: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
    max_sweeps: int = 100_000,
) -> tuple[FloatArray, int]:
    """Return every synchronous value-iteration iterate, including ``V_0``.

    The first row is the all-zero initialization and row ``k`` is the value
    vector after ``k`` complete Bellman-optimality sweeps.  Keeping the trace is
    inexpensive for the small teaching environments and makes value
    propagation directly inspectable in figures and notebooks.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_sweeps, int) or isinstance(max_sweeps, bool) or max_sweeps <= 0:
        raise ValueError("max_sweeps must be a positive integer")
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    values = np.zeros(p.shape[0], dtype=np.float64)
    trace = [values.copy()]
    for sweep in range(1, max_sweeps + 1):
        q = q_from_v(p, r, values, gamma, terminal_mask)
        updated = np.max(q, axis=1)
        updated[terminal_mask] = 0.0
        trace.append(updated.copy())
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < tolerance:
            return np.stack(trace, axis=0), sweep
    raise RuntimeError("value iteration did not converge within max_sweeps")


def greedy_policy(
    transitions: ArrayLike,
    rewards: ArrayLike,
    values: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
) -> IntArray:
    """Return a deterministic greedy policy with lowest-index tie breaking."""
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    q = q_from_v(p, r, values, gamma, terminal_mask)
    actions = np.argmax(q, axis=1).astype(np.int64)
    actions[terminal_mask] = 0
    return actions


def bellman_expectation_residual(
    transitions: ArrayLike,
    rewards: ArrayLike,
    policy: ArrayLike,
    values: ArrayLike,
    gamma: float,
    terminals: ArrayLike | None = None,
) -> FloatArray:
    """Return ``T_pi V - V``."""
    p, r, terminal_mask = validate_tabular_model(transitions, rewards, terminals)
    pi = validate_policy(policy, p.shape[0], p.shape[1])
    v = _as_float_array(values, name="values", ndim=1)
    if v.shape != (p.shape[0],):
        raise ValueError("values has the wrong shape")
    backed_up = np.sum(pi * q_from_v(p, r, v, gamma, terminal_mask), axis=1)
    backed_up[terminal_mask] = 0.0
    return backed_up - v


def n_step_target(
    rewards: Sequence[float],
    bootstrap_value: float,
    gamma: float,
    *,
    terminated_at_end: bool = False,
) -> float:
    """Compute an n-step target with an explicit terminal bootstrap mask."""
    reward_array = _as_float_array(rewards, name="rewards", ndim=1)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not isinstance(terminated_at_end, (bool, np.bool_)):
        raise TypeError("terminated_at_end must be boolean")
    total = 0.0
    discount = 1.0
    for reward in reward_array:
        total += discount * float(reward)
        discount *= gamma
    if bool(terminated_at_end):
        return total
    if not math.isfinite(bootstrap_value):
        raise ValueError("bootstrap_value must be finite for a non-terminal target")
    return total + discount * float(bootstrap_value)


def td_target(
    reward: float,
    next_value: float,
    gamma: float,
    *,
    terminated: bool,
    truncated: bool = False,
) -> float:
    """One-step TD target with separate termination and truncation flags.

    ``terminated`` masks the continuation because the task ended.  An external
    ``truncated`` flag is accepted and validated but does not, by itself, mask
    bootstrapping.  If a time limit is part of the task definition, that limit
    belongs in the state and its exhaustion should be represented as a true
    terminal transition by the caller.
    """
    if not math.isfinite(reward) or not math.isfinite(gamma):
        raise ValueError("reward and gamma must be finite")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not isinstance(terminated, (bool, np.bool_)):
        raise TypeError("terminated must be boolean")
    if not isinstance(truncated, (bool, np.bool_)):
        raise TypeError("truncated must be boolean")
    del truncated  # semantic documentation: external truncation preserves bootstrap
    if bool(terminated):
        return float(reward)
    if not math.isfinite(next_value):
        raise ValueError("next_value must be finite for a non-terminal target")
    return float(reward + gamma * next_value)


def q_learning_target(
    reward: float,
    next_q_values: ArrayLike,
    gamma: float,
    *,
    terminated: bool,
) -> float:
    if isinstance(terminated, (bool, np.bool_)) and bool(terminated):
        return td_target(reward, 0.0, gamma, terminated=True)
    next_q = _as_float_array(next_q_values, name="next_q_values", ndim=1)
    return td_target(reward, float(np.max(next_q)), gamma, terminated=terminated)


def sarsa_target(
    reward: float,
    next_q_value: float,
    gamma: float,
    *,
    terminated: bool,
) -> float:
    return td_target(reward, next_q_value, gamma, terminated=terminated)


def double_q_target(
    reward: float,
    selector_q: ArrayLike,
    evaluator_q: ArrayLike,
    gamma: float,
    *,
    terminated: bool,
) -> float:
    """Double-Q target: one estimator selects, the other evaluates."""
    if isinstance(terminated, (bool, np.bool_)) and bool(terminated):
        return td_target(reward, 0.0, gamma, terminated=True)
    selector = _as_float_array(selector_q, name="selector_q", ndim=1)
    evaluator = _as_float_array(evaluator_q, name="evaluator_q", ndim=1)
    if selector.shape != evaluator.shape:
        raise ValueError("selector_q and evaluator_q must have the same shape")
    selected = int(np.argmax(selector))
    return td_target(reward, float(evaluator[selected]), gamma, terminated=terminated)


def build_gridworld_4x4() -> tuple[FloatArray, FloatArray, BoolArray]:
    """Sutton--Barto 4x4 gridworld used for policy evaluation."""
    n = 4
    n_states = n * n
    n_actions = 4  # up, right, down, left
    terminals = np.zeros(n_states, dtype=bool)
    terminals[[0, n_states - 1]] = True
    p = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    r = np.zeros_like(p)

    for state in range(n_states):
        for action in range(n_actions):
            if terminals[state]:
                p[state, action, state] = 1.0
                continue
            row, col = divmod(state, n)
            if action == 0:
                row = max(row - 1, 0)
            elif action == 1:
                col = min(col + 1, n - 1)
            elif action == 2:
                row = min(row + 1, n - 1)
            else:
                col = max(col - 1, 0)
            next_state = row * n + col
            p[state, action, next_state] = 1.0
            r[state, action, next_state] = -1.0
    return p, r, terminals


def build_frozen_lake_4x4() -> tuple[FloatArray, FloatArray, BoolArray]:
    """Build the default slippery 4x4 FrozenLake-v1 model.

    Actions: 0 left, 1 down, 2 right, 3 up.  The intended action and the two
    perpendicular actions each occur with probability 1/3.
    """
    layout = ("SFFF", "FHFH", "FFFH", "HFFG")
    n = 4
    n_states = n * n
    n_actions = 4
    holes = {row * n + col for row in range(n) for col in range(n) if layout[row][col] == "H"}
    goal = n_states - 1
    terminals = np.zeros(n_states, dtype=bool)
    terminals[list(holes | {goal})] = True
    p = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    r = np.zeros_like(p)

    def move(state: int, action: int) -> int:
        row, col = divmod(state, n)
        if action == 0:
            col = max(col - 1, 0)
        elif action == 1:
            row = min(row + 1, n - 1)
        elif action == 2:
            col = min(col + 1, n - 1)
        else:
            row = max(row - 1, 0)
        return row * n + col

    for state in range(n_states):
        for action in range(n_actions):
            if terminals[state]:
                p[state, action, state] = 1.0
                continue
            for actual_action in ((action - 1) % 4, action, (action + 1) % 4):
                next_state = move(state, actual_action)
                p[state, action, next_state] += 1.0 / 3.0
                if next_state == goal:
                    r[state, action, next_state] = 1.0
    return p, r, terminals


@dataclass
class CorridorEnv:
    """Five-state absorbing corridor with a fair left/right action space."""

    n_cells: int = 5

    def __post_init__(self) -> None:
        if self.n_cells < 3 or self.n_cells % 2 == 0:
            raise ValueError("n_cells must be an odd integer >= 3")
        self.state = self.n_cells // 2

    def reset(self) -> int:
        self.state = self.n_cells // 2
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in (0, 1):
            raise ValueError("action must be 0 (left) or 1 (right)")
        self.state += -1 if action == 0 else 1
        terminated = self.state in (0, self.n_cells - 1)
        reward = 1.0 if self.state == self.n_cells - 1 else 0.0
        return self.state, reward, terminated


def exact_corridor_values(n_cells: int = 5) -> FloatArray:
    """Exact state values for the fair absorbing random walk.

    Reward ``+1`` is emitted on the transition into the right terminal.  Under
    the usual episodic convention, both terminal state values are therefore
    fixed to zero; the non-terminal values equal right-terminal hitting
    probabilities.
    """
    if n_cells < 3 or n_cells % 2 == 0:
        raise ValueError("n_cells must be an odd integer >= 3")
    values = np.linspace(0.0, 1.0, n_cells, dtype=np.float64)
    values[-1] = 0.0
    return values


def _corridor_episode(rng: np.random.Generator, env: CorridorEnv) -> tuple[list[int], list[float]]:
    states: list[int] = []
    rewards: list[float] = []
    state = env.reset()
    for _ in range(100_000):
        states.append(state)
        action = int(rng.integers(0, 2))
        _, reward, terminated = env.step(action)
        rewards.append(reward)
        if terminated:
            return states, rewards
        state = env.state
    raise RuntimeError("corridor episode exceeded the safety cap")


def monte_carlo_prediction_corridor(
    n_episodes: int,
    *,
    seed: int = 0,
    gamma: float = 1.0,
    first_visit: bool = True,
) -> FloatArray:
    """Incremental first-visit or every-visit Monte Carlo prediction."""
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    rng = np.random.default_rng(seed)
    env = CorridorEnv()
    values = np.zeros(env.n_cells, dtype=np.float64)
    counts = np.zeros(env.n_cells, dtype=np.int64)

    for _ in range(n_episodes):
        states, rewards = _corridor_episode(rng, env)
        returns = np.empty(len(rewards), dtype=np.float64)
        g = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            g = rewards[index] + gamma * g
            returns[index] = g
        seen: set[int] = set()
        for state, g_t in zip(states, returns, strict=True):
            if first_visit and state in seen:
                continue
            seen.add(state)
            counts[state] += 1
            values[state] += (g_t - values[state]) / counts[state]
    return values


def td0_prediction_corridor(
    n_episodes: int,
    *,
    seed: int = 0,
    alpha: float = 0.1,
    gamma: float = 1.0,
) -> FloatArray:
    """TD(0) prediction for the fair policy in the corridor."""
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    rng = np.random.default_rng(seed)
    env = CorridorEnv()
    values = np.zeros(env.n_cells, dtype=np.float64)

    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(100_000):
            action = int(rng.integers(0, 2))
            next_state, reward, terminated = env.step(action)
            target = td_target(reward, values[next_state], gamma, terminated=terminated)
            values[state] += alpha * (target - values[state])
            if terminated:
                break
            state = next_state
        else:
            raise RuntimeError("corridor episode exceeded the safety cap")
    return values


def n_step_td_prediction_corridor(
    n_episodes: int,
    *,
    n_steps: int,
    seed: int = 0,
    alpha: float = 0.1,
    gamma: float = 1.0,
) -> FloatArray:
    """Episodic n-step TD prediction in the absorbing corridor.

    Each episode is sampled completely and then used for sequential n-step
    updates.  When the n-step window reaches the true terminal transition, the
    continuation term is masked; otherwise it bootstraps from the current
    value estimate of the state at the end of the window.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not all(math.isfinite(x) for x in (alpha, gamma)):
        raise ValueError("alpha and gamma must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    rng = np.random.default_rng(seed)
    env = CorridorEnv()
    values = np.zeros(env.n_cells, dtype=np.float64)
    for _ in range(n_episodes):
        states, rewards = _corridor_episode(rng, env)
        horizon = len(rewards)
        for t, state in enumerate(states):
            end = min(t + n_steps, horizon)
            observed = rewards[t:end]
            if end < horizon:
                bootstrap = values[states[end]]
                target = n_step_target(observed, bootstrap, gamma, terminated_at_end=False)
            else:
                target = n_step_target(observed, 0.0, gamma, terminated_at_end=True)
            values[state] += alpha * (target - values[state])
    return values


@dataclass
class CliffEnv:
    """Deterministic 4x12 cliff-walking environment."""

    rows: int = 4
    cols: int = 12

    def __post_init__(self) -> None:
        self.start = (self.rows - 1) * self.cols
        self.goal = self.rows * self.cols - 1
        self.cliff = set(range(self.start + 1, self.goal))
        self.state = self.start

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> int:
        self.state = self.start
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in range(4):
            raise ValueError("action must be in {0,1,2,3}")
        row, col = divmod(self.state, self.cols)
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            col = min(col + 1, self.cols - 1)
        elif action == 2:
            row = min(row + 1, self.rows - 1)
        else:
            col = max(col - 1, 0)
        next_state = row * self.cols + col
        if next_state in self.cliff:
            self.state = self.start
            return self.state, -100.0, False
        self.state = next_state
        return self.state, -1.0, self.state == self.goal


def epsilon_greedy(q_values: FloatArray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(0, q_values.shape[1]))
    return int(np.argmax(q_values[state]))


def train_cliff_control(
    method: Literal["q_learning", "sarsa"],
    *,
    seed: int,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    max_steps: int = 1_000,
) -> tuple[FloatArray, FloatArray]:
    """Train tabular Q-learning or SARSA and return Q plus episode returns."""
    if method not in {"q_learning", "sarsa"}:
        raise ValueError("method must be 'q_learning' or 'sarsa'")
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    rng = np.random.default_rng(seed)
    env = CliffEnv()
    q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns = np.zeros(n_episodes, dtype=np.float64)

    for episode in range(n_episodes):
        state = env.reset()
        total = 0.0
        action = epsilon_greedy(q, state, epsilon, rng) if method == "sarsa" else None
        for _ in range(max_steps):
            if method == "q_learning":
                action = epsilon_greedy(q, state, epsilon, rng)
            assert action is not None
            next_state, reward, terminated = env.step(action)
            total += reward

            if method == "q_learning":
                target = q_learning_target(reward, q[next_state], gamma, terminated=terminated)
                q[state, action] += alpha * (target - q[state, action])
                state = next_state
            else:
                if terminated:
                    target = reward
                    q[state, action] += alpha * (target - q[state, action])
                else:
                    next_action = epsilon_greedy(q, next_state, epsilon, rng)
                    target = sarsa_target(reward, q[next_state, next_action], gamma, terminated=False)
                    q[state, action] += alpha * (target - q[state, action])
                    state, action = next_state, next_action
            if terminated:
                break
        episode_returns[episode] = total
    return q, episode_returns


def train_cliff_double_q(
    *,
    seed: int,
    n_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    max_steps: int = 1_000,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Train tabular Double Q-learning on CliffWalking.

    Behavior is epsilon-greedy with respect to ``Q_A + Q_B``.  On every
    transition one table is selected at random for the update: that table
    chooses the maximizing action and the other table evaluates it.
    """
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not all(math.isfinite(x) for x in (alpha, gamma, epsilon)):
        raise ValueError("alpha, gamma, and epsilon must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    rng = np.random.default_rng(seed)
    env = CliffEnv()
    q_a = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    q_b = np.zeros_like(q_a)
    episode_returns = np.zeros(n_episodes, dtype=np.float64)

    for episode in range(n_episodes):
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = epsilon_greedy(q_a + q_b, state, epsilon, rng)
            next_state, reward, terminated = env.step(action)
            total += reward
            update_a = bool(rng.integers(0, 2))
            if update_a:
                target = double_q_target(
                    reward,
                    q_a[next_state],
                    q_b[next_state],
                    gamma,
                    terminated=terminated,
                )
                q_a[state, action] += alpha * (target - q_a[state, action])
            else:
                target = double_q_target(
                    reward,
                    q_b[next_state],
                    q_a[next_state],
                    gamma,
                    terminated=terminated,
                )
                q_b[state, action] += alpha * (target - q_b[state, action])
            state = next_state
            if terminated:
                break
        episode_returns[episode] = total
    return q_a, q_b, episode_returns


def greedy_cliff_rollout(q_values: ArrayLike, *, max_steps: int = 1_000) -> tuple[float, list[int]]:
    q = _as_float_array(q_values, name="q_values", ndim=2)
    env = CliffEnv()
    if q.shape != (env.n_states, env.n_actions):
        raise ValueError("q_values has the wrong shape")
    state = env.reset()
    total = 0.0
    states = [state]
    for _ in range(max_steps):
        action = int(np.argmax(q[state]))
        state, reward, terminated = env.step(action)
        total += reward
        states.append(state)
        if terminated:
            return total, states
    return total, states


if __name__ == "__main__":
    grid_p, grid_r, grid_t = build_gridworld_4x4()
    uniform = np.full((16, 4), 0.25, dtype=np.float64)
    exact = exact_policy_value(grid_p, grid_r, uniform, gamma=1.0, terminals=grid_t)
    print("4x4 random-policy values:")
    print(" ".join(f"{value:.4f}" for value in exact))

    lake_p, lake_r, lake_t = build_frozen_lake_4x4()
    lake_values, sweeps = value_iteration(lake_p, lake_r, gamma=0.9, terminals=lake_t)
    lake_policy = greedy_policy(lake_p, lake_r, lake_values, gamma=0.9, terminals=lake_t)
    print(f"FrozenLake sweeps: {sweeps}")
    print(" ".join(f"{value:.4f}" for value in lake_values))
    print(" ".join(str(int(action)) for action in lake_policy))

    for method in ("q_learning", "sarsa"):
        q, returns = train_cliff_control(
            method,
            seed=0,
            n_episodes=500,
            alpha=0.5,
            gamma=1.0,
            epsilon=0.1,
        )
        greedy_return, _ = greedy_cliff_rollout(q)
        print(f"{method}: greedy={greedy_return:.1f}, train_mean={returns.mean():.2f}")
