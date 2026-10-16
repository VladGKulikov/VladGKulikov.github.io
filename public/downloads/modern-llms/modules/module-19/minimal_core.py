import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: str
    expected: str
    slice_name: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.example_id, self.prompt, self.expected, self.slice_name)):
            raise ValueError("example fields must be non-empty strings")


def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip().casefold() == expected.strip().casefold())


def evaluate(
    model_fn: Callable[[str], str],
    dataset: Sequence[Example],
    metrics: Mapping[str, Callable[[str, str], float]],
) -> dict:
    dataset = tuple(dataset)
    if not callable(model_fn) or not metrics or any(not isinstance(k, str) or not k or not callable(v) for k, v in metrics.items()):
        raise ValueError("supply a callable model and named callable metrics")
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if any(not isinstance(ex, Example) for ex in dataset):
        raise TypeError("expected Example records")
    if len({ex.example_id for ex in dataset}) != len(dataset):
        raise ValueError("example_id values must be unique")

    records = []
    for ex in dataset:
        prediction = model_fn(ex.prompt)  # the only model call for this item
        if not isinstance(prediction, str):
            raise TypeError("model response must be text")
        scores = {
            name: metric(prediction, ex.expected)
            for name, metric in metrics.items()
        }
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("metric scores must be finite numbers in [0, 1]")
        records.append({
            "id": ex.example_id,
            "slice": ex.slice_name,
            "expected": ex.expected,
            "prediction": prediction,
            "scores": scores,
        })

    aggregate = {
        name: sum(row["scores"][name] for row in records) / len(records)
        for name in metrics
    }
    return {
        "records": records,
        "aggregate": aggregate,
        "model_calls": len(records),
    }
