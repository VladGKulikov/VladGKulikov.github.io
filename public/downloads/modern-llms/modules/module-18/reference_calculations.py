#!/usr/bin/env python3
"""Reference calculations and safety helpers for Module 18.

This module owns a frozen, lecture-independent numerical contract for the learner
notebooks.  It covers context growth, a conditional latency estimate, retry reliability,
subagent parallelism, and the CodeAct toy scenario.  The learner notebook embeds the
resulting JSON, so it does not import repository files in Colab.

The safety helpers deliberately implement only a tiny arithmetic language.  They are
not a Python sandbox and must not be generalized into ``python_exec`` without process
isolation, resource limits, filesystem/network policy, and an explicit threat model.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"
KIB = 2**10
MIB = 2**20
GIB = 2**30


class UnsafeExpressionError(ValueError):
    """Raised when an arithmetic expression leaves the allowed language."""


class DuplicateKeyError(ValueError):
    """Raised when JSON contains a duplicate object key."""


def _positive_int(name: str, value: int) -> int:
    """Validate and return one strictly positive non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(name: str, value: float) -> float:
    """Validate and return one strictly positive finite real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _probability(name: str, value: float) -> float:
    """Validate and return one real probability in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class ContextScenario:
    """Token and KV accounting for one append-only agent trajectory."""

    base_tokens: int
    delta_tokens: int
    steps: int
    kv_bytes_per_token: int

    def __post_init__(self) -> None:
        _positive_int("base_tokens", self.base_tokens)
        _positive_int("delta_tokens", self.delta_tokens)
        _positive_int("steps", self.steps)
        _positive_int("kv_bytes_per_token", self.kv_bytes_per_token)

    def naive_prefill_tokens(self) -> int:
        """Return tokens processed when every step re-prefills the whole history."""

        return self.steps * self.base_tokens + self.delta_tokens * self.steps * (self.steps - 1) // 2

    def cached_prefill_tokens(self) -> int:
        """Return tokens processed when the immutable prefix is cached exactly once."""

        return self.base_tokens + self.delta_tokens * (self.steps - 1)

    def final_context_tokens(self) -> int:
        """Return the context length after all step suffixes have been appended."""

        return self.base_tokens + self.delta_tokens * self.steps

    def final_kv_gib(self) -> float:
        """Return the KV payload in GiB for the final context."""

        return self.final_context_tokens() * self.kv_bytes_per_token / GIB


@dataclass(frozen=True)
class LatencyScenario:
    """A conditional arithmetic estimate for one tool-use step."""

    parameter_count: int
    new_prefill_tokens: int
    effective_flops_per_second: float
    decode_tokens: int
    decode_tokens_per_second: float
    tool_latency_ms: float

    def __post_init__(self) -> None:
        _positive_int("parameter_count", self.parameter_count)
        _positive_int("new_prefill_tokens", self.new_prefill_tokens)
        _positive_finite("effective_flops_per_second", self.effective_flops_per_second)
        _positive_int("decode_tokens", self.decode_tokens)
        _positive_finite("decode_tokens_per_second", self.decode_tokens_per_second)
        _positive_finite("tool_latency_ms", self.tool_latency_ms)

    def prefill_floor_ms(self) -> float:
        """Return the dense 2P-token conditional estimate in milliseconds."""

        flops = 2.0 * self.parameter_count * self.new_prefill_tokens
        return flops / self.effective_flops_per_second * 1_000.0

    def decode_ms(self) -> float:
        """Return decode time from the stated token throughput."""

        return self.decode_tokens / self.decode_tokens_per_second * 1_000.0

    def total_ms(self) -> float:
        """Return the additive scenario total for prefill, decode, and the tool."""

        return self.prefill_floor_ms() + self.decode_ms() + self.tool_latency_ms


def retry_step_success(success_probability: float, retries: int) -> float:
    """Return step success under independent attempts and at most ``retries`` retries."""

    p = _probability("success_probability", success_probability)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    return 1.0 - (1.0 - p) ** (retries + 1)


def trajectory_success(success_probability: float, steps: int, retries: int = 0) -> float:
    """Return all-step success under independent steps and retry attempts."""

    steps = _positive_int("steps", steps)
    return retry_step_success(success_probability, retries) ** steps


def expected_attempts(success_probability: float, retries: int) -> float:
    """Return expected attempts for a truncated independent-retry process."""

    p = _probability("success_probability", success_probability)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    failure = 1.0 - p
    return sum(failure**attempt for attempt in range(retries + 1))


def required_step_probability(target_trajectory_success: float, steps: int) -> float:
    """Invert ``p**steps`` for a target trajectory success probability."""

    target = _probability("target_trajectory_success", target_trajectory_success)
    steps = _positive_int("steps", steps)
    return target ** (1.0 / steps)


@dataclass(frozen=True)
class WorkflowScenario:
    """Parallel-work scenario with explicit serial overhead and KV accounting."""

    work_minutes: float
    plan_minutes: float
    merge_minutes: float
    agents: int
    context_tokens_per_agent: int
    kv_bytes_per_token: int

    def __post_init__(self) -> None:
        _positive_finite("work_minutes", self.work_minutes)
        _positive_finite("plan_minutes", self.plan_minutes)
        _positive_finite("merge_minutes", self.merge_minutes)
        _positive_int("agents", self.agents)
        _positive_int("context_tokens_per_agent", self.context_tokens_per_agent)
        _positive_int("kv_bytes_per_token", self.kv_bytes_per_token)

    def parallel_minutes(self, agents: int | None = None) -> float:
        """Return plan + divided work + merge for the requested agent count."""

        n = self.agents if agents is None else _positive_int("agents", agents)
        return self.plan_minutes + self.work_minutes / n + self.merge_minutes

    def work_baseline_speedup(self, agents: int | None = None) -> float:
        """Compare with the stated 400-minute useful-work baseline.

        This reproduces the lecture's historical ×8.9/×16 convention.  It is not the
        end-to-end Amdahl speedup because the one-agent denominator omits plan+merge.
        """

        return self.work_minutes / self.parallel_minutes(agents)

    def end_to_end_speedup(self, agents: int | None = None) -> float:
        """Compare with an explicit one-agent wall-clock baseline including overhead."""

        one_agent = self.plan_minutes + self.work_minutes + self.merge_minutes
        return one_agent / self.parallel_minutes(agents)

    def work_baseline_limit(self) -> float:
        """Return the historical work-baseline ceiling as agent count tends to infinity."""

        return self.work_minutes / (self.plan_minutes + self.merge_minutes)

    def end_to_end_limit(self) -> float:
        """Return the strict end-to-end Amdahl ceiling."""

        one_agent = self.plan_minutes + self.work_minutes + self.merge_minutes
        return one_agent / (self.plan_minutes + self.merge_minutes)

    def kv_gib(self) -> float:
        """Return aggregate KV payload in GiB for all active agent contexts."""

        total_bytes = self.agents * self.context_tokens_per_agent * self.kv_bytes_per_token
        return total_bytes / GIB


@dataclass(frozen=True)
class CodeActScenario:
    """Toy token and round-trip comparison between atomic JSON calls and one code block."""

    json_actions: int
    tokens_per_json_action: int
    code_tokens: int
    code_round_trips: int = 1

    def __post_init__(self) -> None:
        _positive_int("json_actions", self.json_actions)
        _positive_int("tokens_per_json_action", self.tokens_per_json_action)
        _positive_int("code_tokens", self.code_tokens)
        _positive_int("code_round_trips", self.code_round_trips)

    def json_tokens(self) -> int:
        """Return total JSON-call tokens in the toy scenario."""

        return self.json_actions * self.tokens_per_json_action

    def token_ratio(self) -> float:
        """Return JSON tokens divided by code-block tokens."""

        return self.json_tokens() / self.code_tokens

    def round_trip_ratio(self) -> float:
        """Return JSON round trips divided by code-block round trips."""

        return self.json_actions / self.code_round_trips


class SafeArithmeticEvaluator(ast.NodeVisitor):
    """Evaluate a bounded arithmetic expression without executing Python code."""

    _binary: Mapping[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary: Mapping[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(
        self,
        *,
        max_nodes: int = 64,
        max_abs_value: float = 1e12,
        max_power: float = 12.0,
    ) -> None:
        self.max_nodes = _positive_int("max_nodes", max_nodes)
        self.max_abs_value = _positive_finite("max_abs_value", max_abs_value)
        self.max_power = _positive_finite("max_power", max_power)
        self._seen = 0

    def evaluate(self, expression: str) -> int | float:
        """Parse and evaluate one expression from the allowed arithmetic subset."""

        if not isinstance(expression, str) or not expression.strip():
            raise UnsafeExpressionError("expression must be a non-empty string")
        if len(expression) > 128:
            raise UnsafeExpressionError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError("invalid arithmetic syntax") from exc
        self._seen = 0
        value = self.visit(tree)
        return self._checked(value)

    def visit(self, node: ast.AST) -> int | float:  # type: ignore[override]
        """Count AST nodes before dispatching to the allow-listed visitors."""

        self._seen += 1
        if self._seen > self.max_nodes:
            raise UnsafeExpressionError("expression is too complex")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> int | float:  # noqa: N802
        """Evaluate the expression body."""

        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int | float:  # noqa: N802
        """Accept finite real numeric literals but reject booleans and complex values."""

        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpressionError("only real numeric literals are allowed")
        return self._checked(value)

    def visit_BinOp(self, node: ast.BinOp) -> int | float:  # noqa: N802
        """Evaluate one allow-listed binary arithmetic operation."""

        operation = self._binary.get(type(node.op))
        if operation is None:
            raise UnsafeExpressionError(f"operator {type(node.op).__name__} is not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > self.max_power:
            raise UnsafeExpressionError("power is outside the allowed range")
        try:
            value = operation(left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise UnsafeExpressionError(f"arithmetic failed: {type(exc).__name__}") from exc
        return self._checked(value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int | float:  # noqa: N802
        """Evaluate unary plus or minus."""

        operation = self._unary.get(type(node.op))
        if operation is None:
            raise UnsafeExpressionError(f"operator {type(node.op).__name__} is not allowed")
        return self._checked(operation(self.visit(node.operand)))

    def generic_visit(self, node: ast.AST) -> Any:
        """Reject every AST construct not handled above."""

        raise UnsafeExpressionError(f"syntax {type(node).__name__} is not allowed")

    def _checked(self, value: int | float) -> int | float:
        """Reject non-finite and excessively large intermediate values."""

        try:
            numeric = float(value)
        except (TypeError, OverflowError) as exc:
            raise UnsafeExpressionError("result is not a finite real number") from exc
        if not math.isfinite(numeric):
            raise UnsafeExpressionError("result is not finite")
        if abs(numeric) > self.max_abs_value:
            raise UnsafeExpressionError("result exceeds the configured magnitude limit")
        return value


def safe_calculate(expression: str) -> int | float:
    """Evaluate one bounded arithmetic expression through the AST allow-list."""

    return SafeArithmeticEvaluator().evaluate(expression)


def strict_json_loads(payload: str) -> Any:
    """Parse JSON while rejecting duplicate object keys at every nesting level."""

    if not isinstance(payload, str):
        raise TypeError("payload must be a string")

    if len(payload) > 4096:
        raise ValueError("JSON payload exceeds 4096 characters")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("JSON number exceeds the finite range")
        return value

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        """Build one object while rejecting duplicate keys."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        """Reject NaN and infinities, which are outside the JSON standard."""
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_nonstandard_constant,
        parse_float=finite_float,
    )


def build_reference_payload() -> dict[str, Any]:
    """Build the frozen numerical contract consumed by notebooks and visualizations."""

    context = ContextScenario(3_000, 1_500, 20, 128 * KIB)
    latency = LatencyScenario(
        parameter_count=8_030_000_000,
        new_prefill_tokens=context.delta_tokens,
        effective_flops_per_second=990e12 * 0.5,
        decode_tokens=90,
        decode_tokens_per_second=209.4,
        tool_latency_ms=300.0,
    )
    workflow = WorkflowScenario(
        work_minutes=400.0,
        plan_minutes=5.0,
        merge_minutes=20.0,
        agents=20,
        context_tokens_per_agent=30_000,
        kv_bytes_per_token=128 * KIB,
    )
    codeact = CodeActScenario(5, 90, 160)

    payload = {
        "schema_version": 1,
        "context": {
            **asdict(context),
            "naive_prefill_tokens": context.naive_prefill_tokens(),
            "cached_prefill_tokens": context.cached_prefill_tokens(),
            "ratio": context.naive_prefill_tokens() / context.cached_prefill_tokens(),
            "final_context_tokens": context.final_context_tokens(),
            "final_kv_gib": context.final_kv_gib(),
        },
        "latency": {
            **asdict(latency),
            "prefill_floor_ms": latency.prefill_floor_ms(),
            "decode_ms": latency.decode_ms(),
            "total_ms": latency.total_ms(),
            "decode_share": latency.decode_ms() / latency.total_ms(),
            "trajectory_20_s": latency.total_ms() * 20 / 1_000.0,
        },
        "reliability": {
            "p98_20": trajectory_success(0.98, 20),
            "p95_20": trajectory_success(0.95, 20),
            "retry98": retry_step_success(0.98, 1),
            "retry95": retry_step_success(0.95, 1),
            "retry98_20": trajectory_success(0.98, 20, 1),
            "retry95_20": trajectory_success(0.95, 20, 1),
            "required_p_50_for_90pct": required_step_probability(0.90, 50),
            "expected_attempts_p95_one_retry": expected_attempts(0.95, 1),
        },
        "workflow": {
            **asdict(workflow),
            "parallel_minutes": workflow.parallel_minutes(),
            "work_baseline_speedup": workflow.work_baseline_speedup(),
            "end_to_end_speedup": workflow.end_to_end_speedup(),
            "work_baseline_limit": workflow.work_baseline_limit(),
            "end_to_end_limit": workflow.end_to_end_limit(),
            "kv_gib": workflow.kv_gib(),
        },
        "codeact": {
            **asdict(codeact),
            "json_tokens": codeact.json_tokens(),
            "token_ratio": codeact.token_ratio(),
            "round_trip_ratio": codeact.round_trip_ratio(),
            "fsm_comparison_ratio": 150_000 / (2 * 8_030_000_000),
        },
        "safe_calculator": {
            "accepted": {
                "13.96 * 2": 27.92,
                "(7 + 5) / 3": 4.0,
                "2 ** 10": 1024,
            },
            "rejected": [
                "__import__('os').system('echo unsafe')",
                "open('/etc/passwd').read()",
                "(1).__class__",
                "[1, 2][0]",
                "10 ** 100",
                "(-1) ** 0.5",
                "0 ** -1",
                "1 / 0",
                "True + 1",
            ],
        },
        "agent_trace": {
            "expected_answer": "27.92",
            "expected_tool_calls": 3,
            "expected_tool_errors": 1,
            "expected_status": "completed",
            "expected_event_kinds": [
                "assistant_action",
                "tool_error",
                "assistant_action",
                "tool_result",
                "assistant_action",
                "tool_result",
                "assistant_final",
                "verification_pass",
            ],
        },
    }
    return payload


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize the reference payload deterministically with a trailing newline."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def validate_safety_helpers(payload: Mapping[str, Any]) -> None:
    """Exercise the frozen calculator and strict-JSON negative contracts."""

    for expression, expected in payload["safe_calculator"]["accepted"].items():
        actual = safe_calculate(expression)
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"safe calculator drift: {expression}")
    for expression in payload["safe_calculator"]["rejected"]:
        try:
            safe_calculate(expression)
        except UnsafeExpressionError:
            pass
        else:
            raise AssertionError(f"unsafe expression was accepted: {expression}")

    for invalid_json in (
        '{"tool":"a","tool":"b","args":{}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
    ):
        try:
            strict_json_loads(invalid_json)
        except (DuplicateKeyError, ValueError):
            pass
        else:
            raise AssertionError(f"non-strict JSON was accepted: {invalid_json}")


def validate_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Recompute and compare the on-disk payload, then test safety helpers."""

    expected = build_reference_payload()
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise AssertionError(f"reference payload drift: {path}")
    validate_safety_helpers(observed)
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the frozen payload, or rewrite it only with explicit ``--write``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite reference_metrics.json")
    args = parser.parse_args(argv)
    payload = build_reference_payload()
    if args.write:
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(canonical_json(payload), encoding="utf-8")
        print(f"wrote {REFERENCE_PATH}")
    else:
        validate_reference()
        print(f"validated {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
