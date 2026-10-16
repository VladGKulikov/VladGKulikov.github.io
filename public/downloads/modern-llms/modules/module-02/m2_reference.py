#!/usr/bin/env python3
"""Shared, language-neutral reference computations for Module 2.

This file is maintainer tooling, not learner-facing notebook code. It owns the
frozen Module-2 corpus and the precomputed 4096-merge BBPE artifact. The
published RU/EN notebooks embed compressed copies of those assets so that
"Run all" works in a clean Google Colab session without repository-relative
files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import regex as re

MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = MODULE_ROOT / "data"
TRAIN_CORPUS_PATH = DATA_DIR / "m2_train_ru.txt"
EVAL_CORPUS_PATH = DATA_DIR / "m2_eval_en.txt"
HOLDOUT_RU_PATH = DATA_DIR / "m2_eval_ru.txt"
MODEL_PATH = DATA_DIR / "bbpe_ru_4096.json"
REFERENCE_PATH = DATA_DIR / "reference_metrics.json"

PATTERN = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

CURVE_MERGES = (0, 64, 256, 1024, 4096)
SAMPLE_RU = "Внимание — это всё, что вам нужно."
SAMPLE_EN = "Attention is all you need."
FERTILITY_TEXTS = {
    "RU проза": "Языковые модели предсказывают следующий токен по контексту.",
    "EN prose": "Language models predict the next token from context.",
    "Python": "def encode(text):\n    return [merges.get(p, p) for p in pairs(text)]",
    "Числа": "1234 + 5678 = 6912; 2026-07-04",
}

Pair = tuple[int, int]
MergeTable = dict[Pair, int]


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""

    return hashlib.sha256(data).hexdigest()


def load_text(path: Path) -> str:
    """Load a UTF-8 text asset and fail with a path-rich error."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Module-2 asset is missing: {path}") from exc


def load_model() -> dict:
    """Load and minimally validate the frozen BBPE JSON artifact."""

    obj = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if obj.get("version") != "bbpe-course-1":
        raise ValueError(f"Unexpected BBPE artifact version: {obj.get('version')!r}")
    merges = obj.get("merges")
    if not isinstance(merges, list) or len(merges) != 4096:
        raise ValueError("The frozen BBPE artifact must contain exactly 4096 merges")
    return obj


def merge_table_from_model(model: dict, prefix: int | None = None) -> MergeTable:
    """Convert serialized ``[left, right, new_id]`` rows to an ordered dict.

    Python dictionaries preserve insertion order. That order is the BPE merge
    rank and is therefore part of the tokenizer specification.
    """

    rows = model["merges"] if prefix is None else model["merges"][:prefix]
    return {(int(left), int(right)): int(new_id) for left, right, new_id in rows}


def merge_pair_in_chunk(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Merge every non-overlapping occurrence of ``pair`` in one token chunk."""

    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def weighted_pair_counts(
    unique_chunks: dict[str, list[int]], frequencies: dict[str, int]
) -> dict[Pair, int]:
    """Count adjacent token pairs, weighting each unique chunk by multiplicity."""

    counts: dict[Pair, int] = {}
    for chunk_text, ids in unique_chunks.items():
        frequency = frequencies[chunk_text]
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] = counts.get(pair, 0) + frequency
    return counts


def train_deduplicated(text: str, num_merges: int) -> MergeTable:
    """Train the exact teaching BBPE with chunk deduplication.

    The dictionary insertion order preserves the same first-seen tie-break as
    the naive corpus scan. The routine is suitable for release checks and
    small teaching runs; a production trainer would maintain incremental pair
    indexes rather than rescan every unique chunk after each merge.
    """

    if type(num_merges) is not int or num_merges < 0:
        raise ValueError("num_merges must be a non-negative integer")
    frequencies: dict[str, int] = {}
    for chunk in re.findall(PATTERN, text):
        frequencies[chunk] = frequencies.get(chunk, 0) + 1
    unique_chunks = {chunk: list(chunk.encode("utf-8")) for chunk in frequencies}
    merges: MergeTable = {}
    for merge_index in range(num_merges):
        counts = weighted_pair_counts(unique_chunks, frequencies)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new_id = 256 + merge_index
        for chunk, ids in unique_chunks.items():
            if len(ids) > 1:
                unique_chunks[chunk] = merge_pair_in_chunk(ids, pair, new_id)
        merges[pair] = new_id
    return merges


@dataclass(frozen=True)
class ReferenceTokenizer:
    """Small deterministic BBPE encoder used by release checks.

    This is intentionally simple. It is fast enough for the small frozen
    corpus, but it is not presented as a production tokenizer implementation.
    """

    merges: MergeTable

    def encode_chunk(self, ids: list[int]) -> list[int]:
        """Apply the currently present learned pair with the smallest rank."""

        while len(ids) > 1:
            best_pair: Pair | None = None
            best_rank: int | None = None
            for pair in zip(ids[:-1], ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None or best_rank is None:
                break
            ids = merge_pair_in_chunk(ids, best_pair, best_rank)
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 text after the module's fixed pre-tokenization regex."""

        out: list[int] = []
        for chunk in re.findall(PATTERN, text):
            out.extend(self.encode_chunk(list(chunk.encode("utf-8"))))
        return out


def bytes_per_token(tokenizer: ReferenceTokenizer, text: str) -> float:
    """Return UTF-8 bytes per emitted token; reject empty token sequences."""

    token_count = len(tokenizer.encode(text))
    if token_count == 0:
        raise ValueError("bytes_per_token is undefined for an empty token sequence")
    return len(text.encode("utf-8")) / token_count


def compute_reference() -> dict:
    """Compute all language-neutral reference values used by M2 artifacts."""

    train_bytes = TRAIN_CORPUS_PATH.read_bytes()
    eval_bytes = EVAL_CORPUS_PATH.read_bytes()
    holdout_bytes = HOLDOUT_RU_PATH.read_bytes()
    model_bytes = MODEL_PATH.read_bytes()
    train_text = train_bytes.decode("utf-8")
    eval_text = eval_bytes.decode("utf-8")
    holdout_text = holdout_bytes.decode("utf-8")
    model = load_model()
    manifest = json.loads((DATA_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        raw = (DATA_DIR / name).read_bytes()
        if sha256_bytes(raw) != item["sha256"] or len(raw) != item["bytes"]:
            raise AssertionError(f"Corpus provenance mismatch: {name}")

    curve: dict[str, list[float | int]] = {}
    holdout_curve = {}
    token_counts = {}
    for merge_count in CURVE_MERGES:
        tokenizer = ReferenceTokenizer(merge_table_from_model(model, merge_count))
        curve[str(merge_count)] = [
            round(bytes_per_token(tokenizer, train_text), 3),
            round(bytes_per_token(tokenizer, eval_text), 3),
            len(tokenizer.encode(SAMPLE_RU)),
            len(tokenizer.encode(SAMPLE_EN)),
        ]
        holdout_curve[str(merge_count)] = round(bytes_per_token(tokenizer, holdout_text), 3)
        token_counts[str(merge_count)] = {
            "ru_train": len(tokenizer.encode(train_text)),
            "ru_eval": len(tokenizer.encode(holdout_text)),
            "en_eval": len(tokenizer.encode(eval_text)),
        }

    tokenizer = ReferenceTokenizer(merge_table_from_model(model))
    aligned_eval = [row for line in (DATA_DIR / "pud_parallel.jsonl").read_text(encoding="utf-8").splitlines()
                    if (row := json.loads(line))["split"] == "eval"]
    aligned_counts = {lang: sum(len(tokenizer.encode(row[lang])) for row in aligned_eval)
                      for lang in ("ru", "en")}
    fertility: dict[str, list[float | int]] = {}
    for name, text in FERTILITY_TEXTS.items():
        tokens = tokenizer.encode(text)
        word_count = max(len(re.findall(r"\p{L}+|\p{N}+", text)), 1)
        fertility[name] = [
            len(tokens),
            word_count,
            round(len(tokens) / word_count, 2),
            round(len(text.encode("utf-8")) / len(tokens), 2),
        ]

    return {
        "schema_version": 2,
        "corpus": {
            "source": "UD PUD 2.15",
            "manifest_sha256": sha256_bytes((DATA_DIR / "corpus_manifest.json").read_bytes()),
            "train_file": TRAIN_CORPUS_PATH.name,
            "train_sha256": sha256_bytes(train_bytes),
            "train_bytes": len(train_bytes),
            "train_characters": len(train_text),
            "eval_file": EVAL_CORPUS_PATH.name,
            "eval_sha256": sha256_bytes(eval_bytes),
            "eval_bytes": len(eval_bytes),
            "eval_characters": len(eval_text),
            "holdout_ru_file": HOLDOUT_RU_PATH.name,
            "holdout_ru_sha256": sha256_bytes(holdout_bytes),
            "holdout_ru_bytes": len(holdout_bytes),
            "holdout_ru_characters": len(holdout_text),
        },
        "model": {
            "file": MODEL_PATH.name,
            "sha256": sha256_bytes(model_bytes),
            "version": model["version"],
            "base_tokens": 256,
            "merges": len(model["merges"]),
            "vocab_size": 256 + len(model["merges"]),
        },
        "samples": {"ru": SAMPLE_RU, "en": SAMPLE_EN},
        "curve": curve,
        "holdout_ru_curve": holdout_curve,
        "corpus_token_counts": token_counts,
        "aligned_eval_token_counts": aligned_counts,
        "greeting_ids": tokenizer.encode("Привет"),
        "spacing": {s: {"bytes": len(s.encode("utf-8")), "tokens": len(tokenizer.encode(s)),
                        "bytes_per_token": bytes_per_token(tokenizer, s)} for s in ("one two", "one  two")},
        "fertility": fertility,
    }


def write_reference(reference: dict | None = None) -> dict:
    """Write canonical metrics and the two visualization data copies."""

    reference = compute_reference() if reference is None else reference
    REFERENCE_PATH.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    curve = json.dumps(reference["curve"], ensure_ascii=False, indent=2) + "\n"
    fertility = json.dumps(reference["fertility"], ensure_ascii=False, indent=2) + "\n"
    for language in ("ru", "en"):
        viz_dir = MODULE_ROOT / language / "viz"
        (viz_dir / "curve_data.json").write_text(curve, encoding="utf-8")
        (viz_dir / "fertility_data.json").write_text(fertility, encoding="utf-8")
    return reference


def validate_reference(reference: dict | None = None) -> dict:
    """Recompute the reference and raise if any committed value is stale."""

    committed = (
        json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        if reference is None
        else reference
    )
    computed = compute_reference()
    if committed != computed:
        raise AssertionError(
            "Module-2 reference data is stale. Run answersheet_m2.py --write "
            "and review the resulting lecture/visual changes."
        )
    for language in ("ru", "en"):
        viz_dir = MODULE_ROOT / language / "viz"
        curve = json.loads((viz_dir / "curve_data.json").read_text(encoding="utf-8"))
        fertility = json.loads(
            (viz_dir / "fertility_data.json").read_text(encoding="utf-8")
        )
        if curve != computed["curve"] or fertility != computed["fertility"]:
            raise AssertionError(f"Visualization data is stale for language={language}")
    return computed


if __name__ == "__main__":
    ref = validate_reference()
    print(json.dumps(ref, ensure_ascii=False, indent=2))
