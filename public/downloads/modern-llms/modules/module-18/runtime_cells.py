"""Executable M18 notebook runtime sections, embedded without repository imports."""
# Sections execute in teaching order after imports and reference accounting.
# SECTION:CORE_REGISTRY_CODE

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


class DuplicateKeyError(ValueError):
    """Duplicate key in a JSON object."""


class ProtocolError(ValueError):
    """Malformed model response or action envelope."""


def strict_json_loads(payload: str) -> Any:
    """Parse JSON and reject duplicate keys at every object level."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")

    if len(payload) > 4096:
        raise ProtocolError("JSON payload exceeds 4096 characters")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ProtocolError("JSON number exceeds the finite range")
        return value

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build one object while rejecting duplicate keys."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        """Reject NaN and infinities, which are outside the JSON standard."""
        raise ProtocolError(f"non-standard JSON constant: {value}")

    return json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_nonstandard_constant,
        parse_float=finite_float,
    )


@dataclass(frozen=True)
class ArgRule:
    """One JSON-compatible tool-argument contract."""
    kind: str
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"string", "number", "integer"}:
            raise ValueError(f"unsupported argument kind: {self.kind}")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean")
        if self.kind == "string":
            if self.min_value is not None or self.max_value is not None:
                raise ValueError("string rules do not accept numeric bounds")
            if self.max_length is not None:
                positive_int("max_length", self.max_length)
            return
        if self.max_length is not None:
            raise ValueError("numeric rules do not accept max_length")
        bounds: dict[str, float] = {}
        for label, bound in (("min_value", self.min_value), ("max_value", self.max_value)):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise TypeError(f"{label} must be a real number")
            number = float(bound)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite")
            bounds[label] = number
        if ("min_value" in bounds and "max_value" in bounds and
                bounds["min_value"] > bounds["max_value"]):
            raise ValueError("min_value must not exceed max_value")

    def validate(self, name: str, value: Any) -> Any:
        """Validate and normalize one argument value."""
        if self.kind == "string":
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must not be empty")
            if self.max_length is not None and len(value) > self.max_length:
                raise ValueError(f"{name} is too long")
            return value
        if self.kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            if self.min_value is not None and number < self.min_value:
                raise ValueError(f"{name} is below the minimum")
            if self.max_value is not None and number > self.max_value:
                raise ValueError(f"{name} is above the maximum")
            return number
        if self.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{name} is below the minimum")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{name} is above the maximum")
            return value
        raise ValueError(f"unsupported argument kind: {self.kind}")

    def json_schema(self) -> dict[str, Any]:
        """Return the prompt-facing JSON-schema fragment for this rule."""
        schema: dict[str, Any] = {"type": self.kind}
        if self.kind == "string":
            schema["minLength"] = 1
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        return schema


@dataclass(frozen=True)
class ToolSpec:
    """Allow-listed tool implementation and its argument/output contract."""
    name: str
    description: str
    arguments: Mapping[str, ArgRule]
    function: Callable[..., Any] = field(repr=False)
    max_output_chars: int = 256

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", self.name) is None:
            raise ValueError("tool name must be a short snake_case identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        for argument_name, rule in self.arguments.items():
            if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", argument_name) is None:
                raise ValueError(f"invalid argument name: {argument_name}")
            if not isinstance(rule, ArgRule):
                raise TypeError(f"argument rule for {argument_name} must be ArgRule")
        positive_int("max_output_chars", self.max_output_chars)
        if not callable(self.function):
            raise TypeError("tool function must be callable")

    def validate_arguments(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Reject missing/extra arguments and validate every value."""
        if not isinstance(raw, Mapping):
            raise TypeError("args must be a JSON object")
        expected = set(self.arguments)
        supplied = set(raw)
        required = {name for name, rule in self.arguments.items() if rule.required}
        missing = sorted(required - supplied)
        extra = sorted(supplied - expected)
        if missing:
            raise ValueError(f"missing arguments: {missing}")
        if extra:
            raise ValueError(f"unexpected arguments: {extra}")
        return {
            name: self.arguments[name].validate(name, raw[name])
            for name in sorted(supplied)
        }

    def prompt_schema(self) -> dict[str, Any]:
        """Return one stable tool descriptor for the model prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: rule.json_schema() for name, rule in sorted(self.arguments.items())
                },
                "required": sorted(
                    name for name, rule in self.arguments.items() if rule.required
                ),
                "additionalProperties": False,
            },
        }


@dataclass(frozen=True)
class ToolObservation:
    """A bounded result returned to the model without a traceback."""
    ok: bool
    content: str
    error_code: str | None = None
    truncated: bool = False


class ToolRegistry:
    """Allow-list, validate, execute, serialize, and cap local teaching tools."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        if not specs:
            raise ValueError("at least one tool is required")
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("tool names must be unique")

    def prompt_schemas(self) -> list[dict[str, Any]]:
        """Return deterministic prompt-facing schemas."""
        return [self._specs[name].prompt_schema() for name in sorted(self._specs)]

    def invoke(self, name: str, raw_args: Mapping[str, Any]) -> ToolObservation:
        """Execute one allow-listed tool and return a bounded observation."""
        if not isinstance(name, str):
            return ToolObservation(False, "tool name must be a string", "unknown_tool")
        spec = self._specs.get(name)
        if spec is None:
            return ToolObservation(False, f"unknown tool: {name}", "unknown_tool")
        try:
            args = spec.validate_arguments(raw_args)
            value = spec.function(**args)
            if isinstance(value, str):
                content = value
            else:
                content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False)
        except (TypeError, ValueError, KeyError, ArithmeticError) as exc:
            # A typed, bounded message is useful to the model; a traceback is not.
            message = f"{type(exc).__name__}: {exc}"
            return ToolObservation(False, message[:spec.max_output_chars], "tool_error")
        truncated = len(content) > spec.max_output_chars
        if truncated:
            content = content[:spec.max_output_chars - 1] + "…"
        return ToolObservation(True, content, truncated=truncated)


# Duplicate keys and Python's non-standard NaN/Infinity spellings are rejected.
try:
    strict_json_loads('{"tool":"lookup","tool":"calculator","args":{}}')
except DuplicateKeyError:
    pass
else:
    raise AssertionError("duplicate JSON keys were accepted")

for invalid_json in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
    try:
        strict_json_loads(invalid_json)
    except ProtocolError:
        pass
    else:
        raise AssertionError(f"non-standard JSON number was accepted: {invalid_json}")

# Invalid contracts fail when the registry is defined, not at first invocation.
for invalid_rule in (
    lambda: ArgRule("string", min_value=0),
    lambda: ArgRule("number", max_length=8),
    lambda: ArgRule("integer", min_value=2, max_value=1),
    lambda: ArgRule("number", min_value="0"),
):
    try:
        invalid_rule()
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("invalid argument contract was accepted")

print(MSG["registry_ok"])

# END:CORE_REGISTRY_CODE

# SECTION:CORE_CALCULATOR_CODE

import ast
import operator


class UnsafeExpressionError(ValueError):
    """The expression leaves the calculator's bounded arithmetic language."""


class SafeArithmeticEvaluator(ast.NodeVisitor):
    """Evaluate numbers and allow-listed operators without executing Python code."""

    BINARY = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def __init__(self, *, max_nodes: int = 64, max_abs: float = 1e12,
                 max_power: float = 12) -> None:
        self.max_nodes = positive_int("max_nodes", max_nodes)
        self.max_abs = positive_finite("max_abs", max_abs)
        self.max_power = positive_finite("max_power", max_power)
        self._seen = 0

    def evaluate(self, expression: str) -> int | float:
        """Parse and evaluate one expression from the allowed subset."""
        if not isinstance(expression, str) or not expression.strip():
            raise UnsafeExpressionError("expression must be a non-empty string")
        if len(expression) > 128:
            raise UnsafeExpressionError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError("invalid arithmetic syntax") from exc
        self._seen = 0
        return self._checked(self.visit(tree))

    def visit(self, node: ast.AST) -> int | float:  # type: ignore[override]
        """Count every AST node before allow-list dispatch."""
        self._seen += 1
        if self._seen > self.max_nodes:
            raise UnsafeExpressionError("expression is too complex")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> int | float:  # noqa: N802
        """Evaluate the root expression body."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int | float:  # noqa: N802
        """Accept finite real numeric literals, but never booleans."""
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError("only real numeric literals are allowed")
        return self._checked(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> int | float:  # noqa: N802
        """Evaluate an allow-listed binary operation."""
        operation = self.BINARY.get(type(node.op))
        if operation is None:
            raise UnsafeExpressionError(f"operator {type(node.op).__name__} is not allowed")
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > self.max_power:
            raise UnsafeExpressionError("power is outside the allowed range")
        try:
            return self._checked(operation(left, right))
        except (ArithmeticError, OverflowError) as exc:
            raise UnsafeExpressionError(f"arithmetic failed: {type(exc).__name__}") from exc

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int | float:  # noqa: N802
        """Evaluate unary plus or minus."""
        operation = self.UNARY.get(type(node.op))
        if operation is None:
            raise UnsafeExpressionError(f"operator {type(node.op).__name__} is not allowed")
        return self._checked(operation(self.visit(node.operand)))

    def generic_visit(self, node: ast.AST) -> Any:
        """Reject names, calls, attributes, indexing, containers, and all other syntax."""
        raise UnsafeExpressionError(f"syntax {type(node).__name__} is not allowed")

    def _checked(self, value: int | float) -> int | float:
        """Reject non-finite or excessively large intermediate results."""
        try:
            number = float(value)
        except (TypeError, OverflowError) as exc:
            raise UnsafeExpressionError("result is not a finite real number") from exc
        if not math.isfinite(number):
            raise UnsafeExpressionError("result is not finite")
        if abs(number) > self.max_abs:
            raise UnsafeExpressionError("result exceeds the magnitude limit")
        return value


def calculator(expression: str) -> int | float:
    """Evaluate one bounded arithmetic expression; this is not general Python."""
    return SafeArithmeticEvaluator().evaluate(expression)


FACTS = {"sensor_batch": 13.96}


def lookup(key: str) -> float:
    """Return one allow-listed fact from the frozen teaching dictionary."""
    return FACTS[key]


registry = ToolRegistry([
    ToolSpec(
        name="calculator",
        description="Evaluate bounded arithmetic; no names or function calls.",
        arguments={"expression": ArgRule("string", max_length=128)},
        function=calculator,
    ),
    ToolSpec(
        name="lookup",
        description="Read one fact from the frozen teaching dictionary.",
        arguments={"key": ArgRule("string", max_length=64)},
        function=lookup,
    ),
])

for expression, expected in REFERENCE["safe_calculator"]["accepted"].items():
    assert math.isclose(float(calculator(expression)), float(expected), abs_tol=1e-12)

for expression in REFERENCE["safe_calculator"]["rejected"]:
    try:
        calculator(expression)
    except UnsafeExpressionError:
        pass
    else:
        raise AssertionError(f"unsafe expression was accepted: {expression}")

assert registry.invoke("calculator", {"expression": "13.96 * 2"}).content == "27.92"
print(json.dumps(registry.prompt_schemas(), ensure_ascii=False, indent=2))
print(MSG["calculator_ok"])

# END:CORE_CALCULATOR_CODE

# SECTION:CORE_AGENT_CODE

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ActionRequest:
    """A validated action envelope before tool-argument validation."""
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class FinalResponse:
    """A non-empty final answer emitted by the model."""
    answer: str


@dataclass(frozen=True)
class TraceEvent:
    """One deterministic learner-visible event in the agent trajectory."""
    step: int
    kind: str
    detail: str
    tool: str | None = None


@dataclass(frozen=True)
class AgentLimits:
    """Hard budgets enforced by the harness rather than requested from the model."""
    max_steps: int = 10
    max_tool_calls: int = 8
    max_history_chars: int = 8_000
    max_model_response_chars: int = 4_000

    def __post_init__(self) -> None:
        positive_int("max_steps", self.max_steps)
        positive_int("max_tool_calls", self.max_tool_calls)
        positive_int("max_history_chars", self.max_history_chars)
        positive_int("max_model_response_chars", self.max_model_response_chars)


@dataclass(frozen=True)
class AgentResult:
    """Stop status, final answer, counters, and a typed event trace."""
    status: str
    answer: str | None
    tool_calls: int
    history_chars: int
    events: tuple[TraceEvent, ...]


class LLM(Protocol):
    """Minimal callable protocol used by the educational harness."""
    def __call__(self, prompt: str) -> str: ...


def parse_agent_response(response: str) -> ActionRequest | FinalResponse:
    """Parse one strict `Action:` or non-empty `Answer:` response."""
    if not isinstance(response, str):
        raise ProtocolError("model response must be a string")
    response = response.strip()
    if response.startswith("Answer:"):
        answer = response[len("Answer:"):].strip()
        if not answer:
            raise ProtocolError("final answer must not be empty")
        return FinalResponse(answer)
    if response.startswith("Action:"):
        payload = strict_json_loads(response[len("Action:"):].strip())
        if not isinstance(payload, dict) or set(payload) != {"tool", "args"}:
            raise ProtocolError("action must contain exactly tool and args")
        if not isinstance(payload["tool"], str) or not payload["tool"]:
            raise ProtocolError("tool must be a non-empty string")
        if not isinstance(payload["args"], dict):
            raise ProtocolError("args must be a JSON object")
        return ActionRequest(payload["tool"], payload["args"])
    raise ProtocolError("response must start with Action: or Answer:")


def build_prompt(history: list[str], registry: ToolRegistry) -> str:
    """Build a deterministic prompt from history and registry-owned schemas."""
    schemas = json.dumps(registry.prompt_schemas(), ensure_ascii=False, sort_keys=True)
    return (
        "You are a tool-using agent. Tools=" + schemas + "\n"
        "Return exactly one of:\n"
        "Action: {\"tool\": \"name\", \"args\": {...}}\n"
        "Answer: final text\n"
        "History:\n" + "\n".join(history) + "\nAssistant:"
    )


def history_size(history: list[str]) -> int:
    """Return the exact character count of the serialized history body."""
    return len("\n".join(history))


def run_agent(llm: LLM, registry: ToolRegistry, user_query: str,
              *, limits: AgentLimits = AgentLimits(),
              verify: Callable[[str, tuple[TraceEvent, ...]], bool] | None = None) -> AgentResult:
    """Run a bounded ReAct-like loop and return a typed, traceback-free trace."""
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")
    history = [f"User: {user_query.strip()}"]
    events: list[TraceEvent] = []
    tool_calls = 0
    if history_size(history) > limits.max_history_chars:
        return AgentResult("history_budget", None, tool_calls,
                           history_size(history), tuple(events))

    for step in range(1, limits.max_steps + 1):
        response = llm(build_prompt(history, registry))
        if isinstance(response, str) and len(response) > limits.max_model_response_chars:
            events.append(TraceEvent(step, "budget_error", "model response is too long"))
            return AgentResult("response_budget", None, tool_calls,
                               history_size(history), tuple(events))
        try:
            parsed = parse_agent_response(response)
        except (ValueError, TypeError, RecursionError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            events.append(TraceEvent(step, "protocol_error", detail))
            history.append(f"Error: {detail}")
            if history_size(history) > limits.max_history_chars:
                return AgentResult("history_budget", None, tool_calls,
                                   history_size(history), tuple(events))
            continue

        if isinstance(parsed, FinalResponse):
            candidate = f"Assistant: Answer: {parsed.answer}"
            candidate_size = history_size([*history, candidate])
            if candidate_size > limits.max_history_chars:
                events.append(TraceEvent(step, "budget_error", "max_history_chars exceeded"))
                return AgentResult("history_budget", None, tool_calls,
                                   history_size(history), tuple(events))
            events.append(TraceEvent(step, "assistant_final", parsed.answer))
            history.append(candidate)
            if verify is None:
                return AgentResult("answer_proposed", parsed.answer, tool_calls,
                                   candidate_size, tuple(events))
            if verify(parsed.answer, tuple(events)) is True:
                events.append(TraceEvent(step, "verification_pass", "task criterion met"))
                return AgentResult("completed", parsed.answer, tool_calls,
                                   candidate_size, tuple(events))
            events.append(TraceEvent(step, "verification_fail", "task criterion not met"))
            history.append("Error: task criterion not met; revise the answer")
            if history_size(history) > limits.max_history_chars:
                return AgentResult("history_budget", None, tool_calls,
                                   history_size(history), tuple(events))
            continue

        events.append(TraceEvent(step, "assistant_action",
                                 json.dumps(parsed.args, ensure_ascii=False, sort_keys=True),
                                 parsed.tool))
        history.append(f"Assistant: {response.strip()}")
        if history_size(history) > limits.max_history_chars:
            return AgentResult("history_budget", None, tool_calls,
                               history_size(history), tuple(events))
        if tool_calls >= limits.max_tool_calls:
            events.append(TraceEvent(step, "budget_error", "max_tool_calls exhausted",
                                     parsed.tool))
            return AgentResult("tool_budget", None, tool_calls,
                               history_size(history), tuple(events))

        observation = registry.invoke(parsed.tool, parsed.args)
        tool_calls += 1
        kind = "tool_result" if observation.ok else "tool_error"
        events.append(TraceEvent(step, kind, observation.content, parsed.tool))
        label = "Observation" if observation.ok else "Error"
        history.append(f"{label}: {observation.content}")
        if history_size(history) > limits.max_history_chars:
            return AgentResult("history_budget", None, tool_calls,
                               history_size(history), tuple(events))

    return AgentResult("max_steps", None, tool_calls, history_size(history), tuple(events))


class ScriptedLLM:
    """Deterministic offline model stub that consumes one response per call."""

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("responses must not be empty")
        self.responses = tuple(responses)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        """Return the next scripted response and fail if the script is exhausted."""
        if not isinstance(prompt, str) or "Tools=" not in prompt:
            raise ValueError("the harness prompt is malformed")
        if self.calls >= len(self.responses):
            raise RuntimeError("scripted LLM exhausted")
        response = self.responses[self.calls]
        self.calls += 1
        return response

# END:CORE_AGENT_CODE

# SECTION:CORE_TRACE_CODE

SCRIPT = [
    'Action: {"tool":"lookup","args":{"key":"sensor_bach"}}',
    'Action: {"tool":"lookup","args":{"key":"sensor_batch"}}',
    'Action: {"tool":"calculator","args":{"expression":"13.96 * 2"}}',
    'Answer: 27.92',
]


class ObservationPolicy:
    """Offline feedback policy; no trained LLM or memorized script pointer."""
    def __call__(self, prompt):
        if "Observation: 27.92" in prompt:
            return "Answer: 27.92"
        if "Observation: 13.96" in prompt:
            return 'Action: {"tool":"calculator","args":{"expression":"13.96 * 2"}}'
        if "KeyError" in prompt:
            return 'Action: {"tool":"lookup","args":{"key":"sensor_batch"}}'
        return 'Action: {"tool":"lookup","args":{"key":"sensor_bach"}}'


def verify_double(answer, events):
    """Task-specific oracle checks the value and a successful calculator record."""
    return answer == "27.92" and any(e.kind == "tool_result" and
        e.tool == "calculator" and e.detail == "27.92" for e in events)


def run_script() -> AgentResult:
    """Execute the frozen trace with a fresh deterministic model stub."""
    return run_agent(ObservationPolicy(), registry, "Double the fictional sensor_batch value", verify=verify_double)


result = run_script()
expected = REFERENCE["agent_trace"]
assert result.status == expected["expected_status"] == "completed"
assert result.answer == expected["expected_answer"]
assert result.tool_calls == expected["expected_tool_calls"]
assert sum(event.kind == "tool_error" for event in result.events) == expected["expected_tool_errors"]
assert [event.kind for event in result.events] == expected["expected_event_kinds"]

# Determinism is an observable contract, not an assumption.
assert run_script() == result
for event in result.events:
    print(f"step={event.step} kind={event.kind:<16} tool={event.tool or '-':<10} {event.detail}")
print(f"status={result.status} answer={result.answer} tool_calls={result.tool_calls} "
      f"history_chars={result.history_chars}")
print(MSG["trace_ok"])

# END:CORE_TRACE_CODE

# SECTION:CORE_NEGATIVE_CODE

def expect_status(script: list[str], limits: AgentLimits, expected: str) -> AgentResult:
    """Run one bounded script and assert its explicit stop status."""
    outcome = run_agent(ScriptedLLM(script), registry, "test", limits=limits)
    assert outcome.status == expected, (outcome.status, expected)
    return outcome


# Schema failures do not reach tool execution.
for response in [
    'Action: {"tool":"calculator","args":{}}',
    'Action: {"tool":"calculator","args":{"expression":"1+1","extra":0}}',
    'Action: {"tool":"calculator","args":{"expression":true}}',
    'Action: {"tool":"missing","args":{}}',
]:
    parsed = parse_agent_response(response)
    assert isinstance(parsed, ActionRequest)
    observation = registry.invoke(parsed.tool, parsed.args)
    assert not observation.ok

# A long output is capped before it can flood history.
def echo(text: str) -> str:
    """Return text for an observation-size test."""
    return text

small_registry = ToolRegistry([
    ToolSpec("echo", "Echo bounded text.", {"text": ArgRule("string", max_length=100)},
             echo, max_output_chars=16)
])
long_observation = small_registry.invoke("echo", {"text": "x" * 40})
assert long_observation.ok and long_observation.truncated
assert len(long_observation.content) == 16 and long_observation.content.endswith("…")

# Hard harness budgets produce typed stop states.
repeat_action = ['Action: {"tool":"lookup","args":{"key":"sensor_batch"}}'] * 3
expect_status(repeat_action, AgentLimits(max_steps=3, max_tool_calls=1), "tool_budget")
expect_status(["nonsense", "nonsense"], AgentLimits(max_steps=2), "max_steps")
expect_status(["Answer: " + "x" * 50],
              AgentLimits(max_model_response_chars=20), "response_budget")
history_limited = expect_status(
    ['Action: {"tool":"lookup","args":{"key":"sensor_batch"}}'],
    AgentLimits(max_steps=1, max_history_chars=20),
    "history_budget",
)
assert history_limited.tool_calls == 0

# Duplicate top-level keys fail during strict parsing.
try:
    parse_agent_response('Action: {"tool":"lookup","tool":"calculator","args":{}}')
except DuplicateKeyError:
    pass
else:
    raise AssertionError("duplicate action keys were accepted")

print(MSG["negative_ok"])

# END:CORE_NEGATIVE_CODE
