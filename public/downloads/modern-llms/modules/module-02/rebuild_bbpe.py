"""Deliberately retrain the canonical M2 artifact; then regenerate its metrics.

This mutating maintainer command is separate from the read-only answer sheet.
"""
import base64
import json
from m2_reference import MODEL_PATH, TRAIN_CORPUS_PATH, train_deduplicated, write_reference


def main():
    merges = train_deduplicated(TRAIN_CORPUS_PATH.read_text(encoding="utf-8"), 4096)
    if len(merges) != 4096:
        raise ValueError("Corpus exhausted before 4096 merges")
    pieces = {i: bytes([i]) for i in range(256)}
    for (left, right), new_id in merges.items():
        pieces[new_id] = pieces[left] + pieces[right]
    model = {"version": "bbpe-course-1",
             "vocab": {base64.b64encode(piece).decode("ascii"): i for i, piece in pieces.items()},
             "merges": [[left, right, new_id] for (left, right), new_id in merges.items()]}
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reference = write_reference()
    print(json.dumps({k: reference[k] for k in ("curve", "holdout_ru_curve", "fertility", "greeting_ids", "spacing")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
