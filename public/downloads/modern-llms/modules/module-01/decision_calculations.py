"""Offline cost-per-accepted-task example; all prices and outcomes are synthetic."""
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path


# BEGIN EDUCATIONAL CORE: decision_costs
def task_cost(row, tasks):
    """Charge every attempt; count each accepted task once. None means no successes."""
    for key in ("input_tokens", "output_tokens", "successes"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(tasks) is not int or tasks <= 0 or row["successes"] > tasks:
        raise ValueError("invalid task count or success count")
    prices = [Decimal(str(row[k])) for k in
              ("input_usd_per_million", "output_usd_per_million", "tool_usd")]
    if any(not p.is_finite() or p < 0 for p in prices):
        raise ValueError("costs must be finite and nonnegative")
    input_price, output_price, tool_cost = prices
    total = (row["input_tokens"] * input_price + row["output_tokens"] * output_price) / Decimal(1_000_000) + tool_cost
    accepted = row["successes"]
    return {
        "name": row["name"],
        "total_usd": total,
        "success_rate": Decimal(accepted) / tasks,
        "usd_per_success": total / accepted if accepted else None,
    }
# END EDUCATIONAL CORE: decision_costs


def validate():
    scenario = json.loads((Path(__file__).parent / "data/decision_scenario.json").read_text(encoding="utf-8"))
    results = [task_cost(row, scenario["tasks"]) for row in scenario["candidates"]]
    # Independent arithmetic: integer token counts and exact rational prices.
    for row, result in zip(scenario["candidates"], results):
        rational_total = sum(Fraction(row[k]) * Fraction(row[p]) / 1_000_000 for k, p in
                             (("input_tokens", "input_usd_per_million"), ("output_tokens", "output_usd_per_million"))) + Fraction(row["tool_usd"])
        assert Fraction(result["total_usd"]) == rational_total
        assert Fraction(result["usd_per_success"]) == rational_total / row["successes"]
    assert [r["total_usd"] for r in results] == [Decimal("3.60"), Decimal("2.80")]
    assert [r["usd_per_success"] for r in results] == [Decimal("0.060"), Decimal("0.035")]
    assert task_cost(dict(scenario["candidates"][0], successes=0), 100)["usd_per_success"] is None
    for changed in ({"successes": 101}, {"input_tokens": -1}, {"tool_usd": "NaN"}):
        try:
            task_cost(dict(scenario["candidates"][0], **changed), 100)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid input: {changed}")
    for result in results:
        print(result)
    return results


if __name__ == "__main__":
    validate()
