import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    """Exact scalar contract for a trusted, local teaching handler."""
    arguments: dict
    handler: object


def parse_action(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def finite(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value

    if not isinstance(payload, str) or len(payload) > 4096:
        raise ValueError("invalid payload size or type")
    action = json.loads(payload, object_pairs_hook=unique,
                        parse_float=finite, parse_constant=finite)
    if not isinstance(action, dict) or set(action) != {"tool", "args"}:
        raise ValueError("expected exactly tool and args")
    if not isinstance(action["tool"], str) or not isinstance(action["args"], dict):
        raise ValueError("invalid action envelope")
    return action


def react_loop(model, toolbox, query, verify, max_steps=10, max_attempts=6):
    """Bounded read-only teaching loop; completion needs an external verifier."""
    for limit in (max_steps, max_attempts):
        if type(limit) is not int or limit <= 0:
            raise ValueError("budgets must be positive integers")
    events = [{"kind": "user", "text": query}]
    attempts = 0
    for step in range(max_steps):
        response = model(tuple(events))
        if not isinstance(response, str) or len(response) > 4096:
            return {"status": "invalid_response", "events": events}
        if response.startswith("Answer:"):
            answer = response[7:].strip()
            if answer and verify(answer, tuple(events)) is True:
                return {"status": "completed", "answer": answer, "events": events}
            events.append({"kind": "verification_fail"})
            continue
        if not response.startswith("Action:"):
            events.append({"kind": "protocol_error"})
            continue
        if attempts >= max_attempts:
            return {"status": "attempt_budget", "events": events}
        attempts += 1  # Includes malformed and rejected action proposals.
        try:
            action = parse_action(response[7:].strip())
            tool = toolbox.get(action["tool"])
            if tool is None:
                raise ValueError("unknown tool")
            args = action["args"]
            if set(args) != set(tool.arguments):
                raise ValueError("wrong argument names")
            if any(type(args[k]) is not t for k, t in tool.arguments.items()):
                raise TypeError("wrong scalar type")
            result = tool.handler(**args)
            content = json.dumps(result, allow_nan=False)
            if len(content) > 256:
                raise ValueError("observation too long")
            events.append({"kind": "tool_result", "value": result})
        except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError) as error:
            events.append({"kind": "tool_error", "error_type": type(error).__name__})
    return {"status": "max_steps", "events": events}
