"""Corpus preparation: filters and deduplication (M9 §5).

Pipeline stages are ordered deliberately from cheap to expensive. Heuristics
cost one pass over each string, deduplication costs hashing, and a model-based
classifier costs inference. The classifier therefore sees only what survives
the heuristics and deduplication, and the early stages determine pipeline cost.

The first four stages are implemented here. A FineWeb-Edu-style quality
classifier is outside the capstone: it requires a separate trained model and,
at educational scale, introduces more noise than signal.

`pipeline()` returns a funnel with the document count after every stage—the
same funnel shown in the module 9 illustration.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field


CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """Apply NFC, collapse spaces, and remove control characters.

    NFC matters for Cyrillic: SHORT I (U+0439) can be encoded as one code point
    or as I (U+0438) plus a combining breve. Without normalization, BPE sees
    two different words.
    """
    text = unicodedata.normalize("NFC", text)
    text = CONTROL.sub("", text)
    text = WHITESPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


@dataclass
class FilterConfig:
    min_chars: int = 40
    max_chars: int = 100_000
    max_symbol_ratio: float = 0.3      # fraction of non-letter, non-space characters
    max_line_repeat: float = 0.3       # fraction of repeated lines within a document
    min_mean_word_len: float = 2.0
    max_mean_word_len: float = 12.0


def quality_reasons(doc: str, cfg: FilterConfig) -> list[str]:
    """List rejection reasons; an empty list means the document passes.

    Return reasons rather than a Boolean: corpus debugging must reveal which
    heuristic consumed half the data.
    """
    reasons = []
    if len(doc) < cfg.min_chars:
        reasons.append("too_short")
    if len(doc) > cfg.max_chars:
        reasons.append("too_long")
    letters = sum(1 for c in doc if c.isalpha() or c.isspace())
    if doc and 1 - letters / len(doc) > cfg.max_symbol_ratio:
        reasons.append("symbol_ratio")
    lines = [l for l in doc.split("\n") if l.strip()]
    if len(lines) > 3:
        unique = len(set(lines))
        if 1 - unique / len(lines) > cfg.max_line_repeat:
            reasons.append("line_repeat")
    words = doc.split()
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not (cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len):
            reasons.append("word_length")
    return reasons


def dedup_exact(docs: list[str]) -> list[str]:
    """Exact SHA-256 deduplication: cheap and removes most copies."""
    seen: set[str] = set()
    out = []
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append(doc)
    return out


def _shingles(doc: str, size: int = 5) -> set[str]:
    words = doc.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def minhash_signature(doc: str, permutations: int = 64, shingle: int = 5) -> tuple[int, ...]:
    """MinHash signature: minimum shingle hash for each permutation scheme.

    The Jaccard-similarity estimate is the fraction of matching signature
    positions. Exact set intersections would cost O(n²) document comparisons;
    MinHash reduces this to short-vector comparisons, and LSH banding reduces
    it further to dictionary lookup.
    """
    grams = _shingles(doc, shingle)
    if not grams:
        return tuple([0] * permutations)
    signature = []
    for p in range(permutations):
        salt = str(p).encode()
        signature.append(min(int.from_bytes(
            hashlib.blake2b(salt + g.encode("utf-8"), digest_size=8).digest(), "big")
            for g in grams))
    return tuple(signature)


def jaccard_estimate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def minhash_dedup(docs: list[str], threshold: float = 0.8, permutations: int = 64,
                  bands: int = 16, shingle: int = 5) -> list[str]:
    """Near-duplicate removal with MinHash and LSH banding (M9 §5).

    Split each signature into `bands`. Two documents become candidates when at
    least one whole band matches. Candidates are then checked using the Jaccard
    estimate over the full signature; banding only narrows the search.
    """
    if permutations % bands:
        raise ValueError("permutations должно делиться на bands")
    rows = permutations // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    signatures = [minhash_signature(d, permutations, shingle) for d in docs]

    keep, dropped = [], set()
    for i, sig in enumerate(signatures):
        candidates: set[int] = set()
        band_keys = [(b, sig[b * rows:(b + 1) * rows]) for b in range(bands)]
        for key in band_keys:
            candidates.update(buckets.get(key, ()))
        if any(jaccard_estimate(sig, signatures[j]) >= threshold
               for j in candidates if j not in dropped):
            dropped.add(i)
            continue
        for key in band_keys:
            buckets.setdefault(key, []).append(i)
        keep.append(i)
    return [docs[i] for i in keep]


@dataclass
class PipelineReport:
    stages: list[tuple[str, int]] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, count: int) -> None:
        self.stages.append((name, count))

    @property
    def survival(self) -> float:
        return self.stages[-1][1] / max(self.stages[0][1], 1)

    def __str__(self) -> str:
        lines = ["pipeline funnel:"]
        first = max(self.stages[0][1], 1)
        for name, count in self.stages:
            lines.append(f"  {name:<20} {count:>8,}  ({count / first:.1%})")
        if self.reasons:
            lines.append("rejection reasons: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.reasons.items(), key=lambda x: -x[1])))
        return "\n".join(lines)


def pipeline(docs: list[str], cfg: FilterConfig | None = None,
             near_dup_threshold: float = 0.8) -> tuple[list[str], PipelineReport]:
    """Full pass: normalization → heuristics → exact dedup → near-dup."""
    cfg = cfg or FilterConfig()
    report = PipelineReport()
    report.add("input", len(docs))

    docs = [normalize(d) for d in docs]
    report.add("normalization", len(docs))

    survivors = []
    for doc in docs:
        reasons = quality_reasons(doc, cfg)
        if reasons:
            for r in reasons:
                report.reasons[r] = report.reasons.get(r, 0) + 1
        else:
            survivors.append(doc)
    report.add("heuristics", len(survivors))

    survivors = dedup_exact(survivors)
    report.add("exact dedup", len(survivors))

    survivors = minhash_dedup(survivors, threshold=near_dup_threshold)
    report.add("near-dup dedup", len(survivors))
    return survivors, report
