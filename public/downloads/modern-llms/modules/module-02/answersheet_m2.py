#!/usr/bin/env python3
"""Fast release validator and answer sheet for Module 2.

The default mode is read-only: it recomputes the committed metrics from the
frozen Module-2 corpus and the frozen 4096-merge BBPE artifact, validates the
first 256 learned merges against a fresh teaching-trainer run, and prints the
numbers used by the lecture. Use ``--write`` only after an intentional data or
algorithm change; it rewrites reference JSON used by RU/EN visualizations.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from m2_reference import (  # noqa: E402
    FERTILITY_TEXTS,
    PATTERN,
    SAMPLE_EN,
    SAMPLE_RU,
    TRAIN_CORPUS_PATH,
    ReferenceTokenizer,
    compute_reference,
    load_model,
    load_text,
    merge_pair_in_chunk,
    merge_table_from_model,
    train_deduplicated,
    validate_reference,
    write_reference,
)

TOY = "низко низко летали ласточки, ласточки летали низко"


def toy_trace(steps: int = 6) -> tuple[list[tuple[tuple[int, int], int, int]], int]:
    """Return ``(pair, frequency, new_id)`` rows and final token count."""

    chunks = [list(chunk.encode("utf-8")) for chunk in PATTERN.findall(TOY)]
    rows: list[tuple[tuple[int, int], int, int]] = []
    for step in range(steps):
        counts: Counter[tuple[int, int]] = Counter()
        for chunk in chunks:
            counts.update(zip(chunk[:-1], chunk[1:]))
        pair = max(counts, key=counts.get)
        new_id = 256 + step
        rows.append((pair, counts[pair], new_id))
        chunks = [merge_pair_in_chunk(chunk, pair, new_id) for chunk in chunks]
    return rows, sum(len(chunk) for chunk in chunks)


def validate_model_prefix(prefix_merges: int = 256) -> None:
    """Prove that the frozen artifact starts with freshly trained exact merges."""

    text = load_text(TRAIN_CORPUS_PATH)
    model = load_model()
    expected = list(merge_table_from_model(model, prefix_merges).items())
    actual = list(train_deduplicated(text, prefix_merges).items())
    if actual != expected:
        raise AssertionError(
            f"Frozen BBPE artifact diverges within its first {prefix_merges} merges"
        )


def print_report(reference: dict) -> None:
    """Print the compact numeric answer sheet used during editorial review."""

    print("== Module 2: frozen-data contract ==")
    corpus = reference["corpus"]
    model = reference["model"]
    print(
        f"train: {corpus['train_bytes']:,} bytes / {corpus['train_characters']:,} chars "
        f"| eval: {corpus['eval_bytes']:,} bytes / {corpus['eval_characters']:,} chars"
    )
    print(
        f"model: {model['merges']} merges, |V|={model['vocab_size']}, "
        f"sha256={model['sha256'][:12]}…"
    )

    rows, final_count = toy_trace()
    assert final_count == 62
    print("\n== Toy trace ==")
    for index, (pair, frequency, new_id) in enumerate(rows, start=1):
        print(f"{index}: pair={pair}, frequency={frequency}, new_id={new_id}")
    print(f"93 bytes -> {final_count} tokens after six merges")

    print("\n== Compression curve ==")
    print("merges | RU train | RU held-out | EN held-out | RU sample | EN sample")
    for key, values in reference["curve"].items():
        print(
            f"{int(key):6d} | {values[0]:8.3f} | {reference['holdout_ru_curve'][key]:11.3f} | {values[1]:11.3f} | "
            f"{values[2]:9d} | {values[3]:9d}"
        )

    print("\n== Fertility ==")
    for name, values in reference["fertility"].items():
        print(
            f"{name:9s}: {values[0]:2d} tokens / {values[1]:2d} words | "
            f"fertility={values[2]:.2f}, bytes/token={values[3]:.2f}"
        )

    tokenizer = ReferenceTokenizer(merge_table_from_model(load_model()))
    assert tokenizer.encode("Привет") == reference["greeting_ids"] == [943, 370, 301]
    for text, expected in reference["spacing"].items():
        assert len(text.encode("utf-8")) == expected["bytes"]
        assert len(tokenizer.encode(text)) == expected["tokens"]
    assert len(tokenizer.encode(SAMPLE_RU)) == reference["curve"]["4096"][2]
    assert len(tokenizer.encode(SAMPLE_EN)) == reference["curve"]["4096"][3]
    for name, text in FERTILITY_TEXTS.items():
        assert len(tokenizer.encode(text)) == reference["fertility"][name][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite reference_metrics.json and both RU/EN visualization data files",
    )
    args = parser.parse_args()

    reference = compute_reference()
    if args.write:
        write_reference(reference)
        print(f"wrote {MODULE_ROOT / 'data' / 'reference_metrics.json'}")
    else:
        validate_reference()

    validate_model_prefix(256)
    print_report(reference)
    import runpy
    runpy.run_path(str(MODULE_ROOT / "embedding_lab.py"))
    print("\nAll Module-2 reference checks passed.")


if __name__ == "__main__":
    main()
