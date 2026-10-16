"""CPU calibration experiment over authored, saved response records (no LLM)."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_records(records):
    if not records:
        raise ValueError("records must be nonempty")
    ids = set()
    for row in records:
        ident = row["id"]
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("unique nonempty string ids required")
        ids.add(ident)
        if not all(isinstance(row[key], str) for key in ("prediction", "expected")):
            raise ValueError("responses and targets must be strings")
        q = row["confidence"]
        if isinstance(q, bool) or not isinstance(q, (float, int)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")


def correctness(row):
    # Same explicitly declared policy as the lecture's exact_match.
    return int(row["prediction"].strip().casefold() == row["expected"].strip().casefold())


def brier_score(records):
    validate_records(records)
    return sum((row["confidence"] - correctness(row)) ** 2 for row in records) / len(records)


def reliability_groups(records):
    """Group identical authored scores; general data require declared bin edges."""
    validate_records(records)
    result = []
    for q in sorted({row["confidence"] for row in records}):
        group = [row for row in records if row["confidence"] == q]
        result.append(dict(confidence=q, count=len(group),
                           accuracy=sum(map(correctness, group)) / len(group)))
    return result


def selective_report(records, threshold):
    validate_records(records)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    accepted = [row for row in records if row["confidence"] >= threshold]
    errors = sum(1 - correctness(row) for row in accepted)
    return dict(threshold=threshold, accepted=len(accepted), errors=errors,
                coverage=len(accepted) / len(records),
                risk=errors / len(accepted) if accepted else None)


def authored_records():
    # These counts define a teaching fixture, not sampled model behavior.
    # Confidence is part of each authored response; it is not fitted here.
    records = []
    for q, count, successes in [(0.95, 4, 4), (0.8, 10, 6), (0.6, 6, 3)]:
        for j in range(count):
            i = len(records)
            records.append(dict(id=f"cal-{i:02}", prompt=f"{i} + 1 = ?",
                                prediction=str(i + 1 if j < successes else i + 2),
                                expected=str(i + 1), confidence=q))
    return records


def run_calibration_lab():
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        path = Path(folder) / "calibration_responses.json"
        path.write_text(json.dumps(authored_records()), encoding="utf-8")
        saved = json.loads(path.read_text(encoding="utf-8"))
    # Every score below comes from the same saved records, without new calls.
    score = brier_score(saved)
    groups = reliability_groups(saved)
    curve = [selective_report(saved, t) for t in (0.0, 0.6, 0.8, 0.9, 1.0)]
    assert sum(map(correctness, saved)) == 13
    assert math.isclose(score, 0.2185, abs_tol=1e-14)
    assert groups[1] == dict(confidence=0.8, count=10, accuracy=0.6)
    assert curve[2]["accepted"] == 14 and curve[2]["errors"] == 4
    assert curve[3]["accepted"] == 4 and curve[3]["risk"] == 0
    assert curve[4]["coverage"] == 0 and curve[4]["risk"] is None
    print("Brier:", score, "accuracy:", 13 / 20)
    print("Confidence groups:", groups)
    print("Coverage and selective risk:", curve)
    return saved, groups, curve


if __name__ == "__main__":
    run_calibration_lab()
