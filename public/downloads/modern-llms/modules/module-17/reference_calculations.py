#!/usr/bin/env python3
"""Reference calculations and validation helpers for Module 17.

The module owns a frozen, lecture-independent numerical contract for the learner
notebooks.  The implementation is deliberately transparent: it exposes chunk
boundaries, sparse and dense index shapes, reciprocal-rank contributions, retrieval
metrics, context-budget decisions, and binary memory units.

The dense branch is a small TF-IDF + truncated-SVD (LSA) index.  It is a genuine dense
vector-search path, but it is intentionally labelled as a teaching proxy rather than a
neural embedding model.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _positive_int(name: str, value: int) -> int:
    """Validate and return one strictly positive integer."""

    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(name: str, value: float) -> float:
    """Validate and return one strictly positive finite float."""

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def tokenize(text: str) -> tuple[str, ...]:
    """Find word spans in the original text, then casefold each word for search."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(match.group().casefold() for match in WORD_RE.finditer(text))


@dataclass(frozen=True)
class Chunk:
    """One contiguous token window with source and offset metadata."""

    chunk_id: str
    source_id: str
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    text: str
    token_count: int


def chunk_document(
    source_id: str,
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[Chunk, ...]:
    """Split one document into deterministic overlapping token windows.

    Whitespace-only documents return an empty tuple explicitly.  Non-empty documents
    are covered without gaps; the final chunk is not duplicated when the last regular
    window already reaches the end.
    """

    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    chunk_size = _positive_int("chunk_size", chunk_size)
    if not isinstance(overlap, (int, np.integer)) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")
    overlap = int(overlap)
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    spans = tuple(WORD_RE.finditer(text))
    if not spans:
        return ()
    stride = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        # Character offsets refer to the original Python string, not bytes or
        # casefolded text. Preserve case, punctuation, and spacing inside the span.
        char_start = spans[start].start()
        char_end = spans[end - 1].end()
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}:c{index:03d}",
                source_id=source_id,
                start_token=start,
                end_token=end,
                start_char=char_start,
                end_char=char_end,
                text=text[char_start:char_end],
                token_count=end - start,
            )
        )
        if end == len(spans):
            break
        start += stride
        index += 1
    return tuple(chunks)


def analytical_chunk_count(total_tokens: int, chunk_size: int, overlap: int) -> int:
    """Return the exact sliding-window count for a tokenized document."""

    total_tokens = _positive_int("total_tokens", total_tokens)
    chunk_size = _positive_int("chunk_size", chunk_size)
    if not isinstance(overlap, (int, np.integer)) or isinstance(overlap, bool):
        raise TypeError("overlap must be an integer")
    overlap = int(overlap)
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if total_tokens <= chunk_size:
        return 1
    stride = chunk_size - overlap
    return (total_tokens - chunk_size + stride - 1) // stride + 1


def _validate_ids(ids):
    """IDs remain nonempty strings; coercion must not merge 1 and '1'."""
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("document IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("document IDs must be unique")


def _validate_judgments(ranking, relevance):
    """Reject duplicate ranks, fractional/negative grades and empty gold sets."""
    _validate_ids(ranking)
    _validate_ids(list(relevance))
    if any(isinstance(g, bool) or not isinstance(g, (int, np.integer))
           or not 0 <= g <= 30 for g in relevance.values()):
        raise ValueError("grades must be integers from 0 through 30")
    if not any(g > 0 for g in relevance.values()):
        raise ValueError("at least one positive judgment is required")


def stable_ranking(scores: Sequence[float], document_ids: Sequence[str]) -> tuple[str, ...]:
    """Sort scores descending with document ID as a deterministic tie-break."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or score_array.size != len(document_ids):
        raise ValueError("scores and document_ids must be aligned rank-1 sequences")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores contain NaN or infinity")
    _validate_ids(document_ids)
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document_ids must be unique")
    return tuple(
        document_id
        for _, document_id in sorted(
            zip(score_array.tolist(), map(str, document_ids)),
            key=lambda pair: (-pair[0], pair[1]),
        )
    )


@dataclass(frozen=True)
class SearchResult:
    """One ranked document and its branch score."""

    document_id: str
    score: float
    rank: int


class BM25Index:
    """A small exact BM25 index with explicit empty-input policy."""

    def __init__(
        self,
        document_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(document_ids) != len(texts) or not document_ids:
            raise ValueError("document_ids and texts must be equally sized and non-empty")
        _validate_ids(document_ids)
        self.document_ids = tuple(map(str, document_ids))
        self.tokens = tuple(tokenize(text) for text in texts)
        if any(not tokens for tokens in self.tokens):
            raise ValueError("empty documents must be removed before indexing")
        self.k1 = _positive_finite("k1", k1)
        self.b = float(b)
        if not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("b must lie in [0,1]")
        n_documents = len(self.tokens)
        self.average_length = sum(map(len, self.tokens)) / n_documents
        self.document_frequency = Counter(
            token for document_tokens in self.tokens for token in set(document_tokens)
        )
        self.idf = {
            token: math.log(
                (n_documents - frequency + 0.5) / (frequency + 0.5) + 1.0
            )
            for token, frequency in self.document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Return one BM25 score per indexed document."""

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must contain at least one word token")
        if not any(token in self.idf for token in query_tokens):
            raise ValueError("query has no indexed lexical terms")
        scores = np.zeros(len(self.tokens), dtype=np.float64)
        for row, document_tokens in enumerate(self.tokens):
            frequencies = Counter(document_tokens)
            length_norm = 1.0 - self.b + self.b * len(document_tokens) / self.average_length
            for token in query_tokens:
                tf = frequencies.get(token, 0)
                if tf == 0 or token not in self.idf:
                    continue
                scores[row] += (
                    self.idf[token]
                    * tf
                    * (self.k1 + 1.0)
                    / (tf + self.k1 * length_norm)
                )
        return scores

    def search(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        """Rank all documents or the requested top-k with stable ties."""

        scores = self.scores(query)
        ranking = stable_ranking(scores, self.document_ids)
        if top_k is not None:
            top_k = _positive_int("top_k", top_k)
            ranking = ranking[:top_k]
        score_by_id = dict(zip(self.document_ids, scores.tolist()))
        return tuple(
            SearchResult(document_id, float(score_by_id[document_id]), rank)
            for rank, document_id in enumerate(ranking, start=1)
        )


class LSAIndex:
    """Exact cosine search over dense TF-IDF + SVD document vectors."""

    def __init__(self, document_ids: Sequence[str], texts: Sequence[str], *, dimension: int = 4) -> None:
        if len(document_ids) != len(texts) or not document_ids:
            raise ValueError("document_ids and texts must be equally sized and non-empty")
        _validate_ids(document_ids)
        self.document_ids = tuple(map(str, document_ids))
        tokenized = tuple(tokenize(text) for text in texts)
        if any(not tokens for tokens in tokenized):
            raise ValueError("empty documents must be removed before indexing")
        vocabulary = sorted({token for tokens in tokenized for token in tokens})
        self.vocabulary = {token: index for index, token in enumerate(vocabulary)}
        counts = np.zeros((len(tokenized), len(vocabulary)), dtype=np.float64)
        document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
        for row, tokens in enumerate(tokenized):
            frequencies = Counter(tokens)
            for token, count in frequencies.items():
                counts[row, self.vocabulary[token]] = count
                document_frequency[self.vocabulary[token]] += 1.0
        self.idf = np.log((1.0 + len(tokenized)) / (1.0 + document_frequency)) + 1.0
        tf = np.where(counts > 0.0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
        tfidf = tf * self.idf
        _, singular_values, vt = np.linalg.svd(tfidf, full_matrices=False)
        numerical_rank = int(np.sum(singular_values > 1e-10))
        dimension = min(_positive_int("dimension", dimension), numerical_rank)
        self.projection = vt[:dimension].T
        dense = tfidf @ self.projection
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("at least one document has a zero dense representation")
        self.document_vectors = dense / norms

    def encode_query(self, query: str) -> np.ndarray:
        """Map one non-empty in-vocabulary query to a unit dense vector."""

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must contain at least one word token")
        counts = Counter(query_tokens)
        sparse = np.zeros(len(self.vocabulary), dtype=np.float64)
        for token, count in counts.items():
            position = self.vocabulary.get(token)
            if position is not None:
                sparse[position] = (1.0 + math.log(count)) * self.idf[position]
        dense = sparse @ self.projection
        norm = float(np.linalg.norm(dense))
        if norm <= 0.0:
            raise ValueError("query contains no terms represented by the dense index")
        return dense / norm

    def scores(self, query: str) -> np.ndarray:
        """Return exact cosine scores against every document vector."""

        scores = self.document_vectors @ self.encode_query(query)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("dense search produced NaN or infinity")
        return scores

    def search(self, query: str, *, top_k: int | None = None) -> tuple[SearchResult, ...]:
        """Rank exact dense cosine scores with stable ties."""

        scores = self.scores(query)
        ranking = stable_ranking(scores, self.document_ids)
        if top_k is not None:
            top_k = _positive_int("top_k", top_k)
            ranking = ranking[:top_k]
        score_by_id = dict(zip(self.document_ids, scores.tolist()))
        return tuple(
            SearchResult(document_id, float(score_by_id[document_id]), rank)
            for rank, document_id in enumerate(ranking, start=1)
        )


@dataclass(frozen=True)
class FusedResult:
    """One RRF result with a per-branch contribution trace."""

    document_id: str
    score: float
    rank: int
    contributions: Mapping[str, float]
    source_ranks: Mapping[str, int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: float = 60.0,
    weights: Mapping[str, float] | None = None,
    depth: int | None = None,
) -> tuple[FusedResult, ...]:
    """Accumulate ranks in linear time, then sort unique candidates.

    Ranks are one-based. Duplicate document IDs inside one branch are rejected because
    they would double-count evidence.  The implementation builds rank maps once and
    never calls ``list.index`` inside a loop.
    """

    k = _positive_finite("k", k)
    if not rankings:
        raise ValueError("at least one ranking is required")
    if depth is not None:
        depth = _positive_int("depth", depth)
    branch_weights = {name: 1.0 for name in rankings}
    if weights is not None:
        if set(weights) != set(rankings):
            raise ValueError("weights must contain exactly the ranking branch names")
        branch_weights = {
            name: _positive_finite(f"weight[{name}]", weights[name]) for name in rankings
        }
    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for branch_name, ranking in rankings.items():
        _validate_ids(ranking)
        sequence = tuple(ranking)
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"ranking {branch_name!r} contains duplicate document IDs")
        if depth is not None:
            sequence = sequence[:depth]
        for rank, document_id in enumerate(sequence, start=1):
            contribution = branch_weights[branch_name] / (k + rank)
            scores[document_id] = scores.get(document_id, 0.0) + contribution
            contributions.setdefault(document_id, {})[branch_name] = contribution
            source_ranks.setdefault(document_id, {})[branch_name] = rank
    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return tuple(
        FusedResult(
            document_id=document_id,
            score=float(scores[document_id]),
            rank=rank,
            contributions=dict(sorted(contributions[document_id].items())),
            source_ranks=dict(sorted(source_ranks[document_id].items())),
        )
        for rank, document_id in enumerate(ordered, start=1)
    )


def _character_trigrams(token: str) -> frozenset[str]:
    """Return boundary-aware character trigrams for one token."""

    wrapped = f"^{token}$"
    return frozenset(wrapped[index : index + 3] for index in range(max(0, len(wrapped) - 2)))


def _token_similarity(left: str, right: str) -> float:
    """Return Jaccard similarity between two token trigram sets."""

    a = _character_trigrams(left)
    b = _character_trigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_interaction_score(query: str, document: str) -> float:
    """Score a query/document pair with a transparent MaxSim-style proxy.

    This is not a neural cross-encoder.  It deliberately exposes the per-query-token
    maximum interaction used by late-interaction retrieval families.
    """

    query_tokens = tokenize(query)
    document_tokens = tokenize(document)
    if not query_tokens:
        raise ValueError("query must contain at least one word token")
    if not document_tokens:
        raise ValueError("document must contain at least one word token")
    return float(
        sum(
            max(_token_similarity(query_token, document_token) for document_token in document_tokens)
            for query_token in query_tokens
        )
        / len(query_tokens)
    )


def rerank_candidates(
    query: str,
    candidate_ids: Sequence[str],
    documents: Mapping[str, str],
) -> tuple[SearchResult, ...]:
    """Rerank a candidate list with the token-interaction proxy."""

    _validate_ids(candidate_ids)
    candidate_ids = tuple(candidate_ids)
    if not candidate_ids:
        raise ValueError("candidate_ids must be non-empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    missing = [document_id for document_id in candidate_ids if document_id not in documents]
    if missing:
        raise KeyError(f"missing documents: {missing}")
    scores = [token_interaction_score(query, documents[document_id]) for document_id in candidate_ids]
    ranking = stable_ranking(scores, candidate_ids)
    score_by_id = dict(zip(candidate_ids, scores))
    return tuple(
        SearchResult(document_id, float(score_by_id[document_id]), rank)
        for rank, document_id in enumerate(ranking, start=1)
    )


def reciprocal_rank(ranking: Sequence[str], relevance: Mapping[str, int]) -> float:
    """Return reciprocal rank of the first positively relevant document."""
    _validate_judgments(ranking, relevance)

    if not any(int(grade) > 0 for grade in relevance.values()):
        raise ValueError("relevance must contain at least one positive grade")
    for rank, document_id in enumerate(ranking, start=1):
        if int(relevance.get(document_id, 0)) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Return set recall among all positively relevant documents."""
    _validate_judgments(ranking, relevance)

    k = _positive_int("k", k)
    relevant = {document_id for document_id, grade in relevance.items() if int(grade) > 0}
    if not relevant:
        raise ValueError("relevance must contain at least one positive grade")
    return len(set(ranking[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranking: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Return nDCG@k with gains 2^grade-1 and log2 discount."""
    _validate_judgments(ranking, relevance)

    k = _positive_int("k", k)
    grades = [int(grade) for grade in relevance.values()]
    if any(grade < 0 for grade in grades):
        raise ValueError("relevance grades must be non-negative")
    if not any(grade > 0 for grade in grades):
        raise ValueError("relevance must contain at least one positive grade")
    dcg = sum(
        (2 ** int(relevance.get(document_id, 0)) - 1.0) / math.log2(rank + 1.0)
        for rank, document_id in enumerate(ranking[:k], start=1)
    )
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum((2**grade - 1.0) / math.log2(rank + 1.0) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    query_records: Sequence[Mapping[str, object]],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Aggregate MRR, Recall@k, and nDCG@k over one frozen query set."""

    k = _positive_int("k", k)
    if not query_records or len({r["query_id"] for r in query_records}) != len(query_records):
        raise ValueError("queries must be nonempty with unique IDs")
    if len(rankings) != len(query_records):
        raise ValueError("rankings must contain exactly one entry per query")
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for record in query_records:
        query_id = str(record["query_id"])
        if query_id not in rankings:
            raise KeyError(f"missing ranking for {query_id}")
        relevance = dict(record["relevance"])
        ranking = tuple(rankings[query_id])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"ranking for {query_id} contains duplicates")
        reciprocal_ranks.append(reciprocal_rank(ranking, relevance))
        recalls.append(recall_at_k(ranking, relevance, k))
        ndcgs.append(ndcg_at_k(ranking, relevance, k))
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        f"recall_at_{k}": float(np.mean(recalls)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
    }


@dataclass(frozen=True)
class ContextItem:
    """One selected chunk and its explicit budget contribution."""

    citation: str
    document_id: str
    source_id: str
    token_count: int
    score: float
    text: str


@dataclass(frozen=True)
class ContextPack:
    """A budgeted retrieval context with selected and skipped trace data."""

    items: tuple[ContextItem, ...]
    used_tokens: int
    max_tokens: int
    skipped: Mapping[str, str]
    rendered_text: str


def pack_context(
    ranked_ids: Sequence[str],
    scores: Mapping[str, float],
    documents: Mapping[str, Mapping[str, object]],
    *,
    max_tokens: int,
    max_per_source: int = 1,
    citation_overhead_tokens: int | None = None,
) -> ContextPack:
    """Deduplicate and pack whole chunks under an explicit token budget.

    Exact normalized-text duplicates and per-source overflow are skipped.  Chunks are
    never silently truncated; a chunk either fits in full or is traced as skipped.
    """

    max_tokens = _positive_int("max_tokens", max_tokens)
    max_per_source = _positive_int("max_per_source", max_per_source)
    if citation_overhead_tokens is not None:
        raise ValueError("count the serialized citation instead of a guessed overhead")
    _validate_ids(ranked_ids)
    ranked_ids = tuple(ranked_ids)
    if not ranked_ids:
        raise ValueError("ranked_ids must be nonempty")
    selected: list[ContextItem] = []
    skipped: dict[str, str] = {}
    seen_texts: set[str] = set()
    per_source: Counter[str] = Counter()
    used = 0
    for document_id in ranked_ids:
        if document_id not in documents or document_id not in scores:
            raise KeyError(f"missing document or score for {document_id}")
        record = documents[document_id]
        text = record["text"]
        source_id = record["source_id"]
        _validate_ids([source_id])
        if not math.isfinite(float(scores[document_id])):
            raise ValueError("context scores must be finite")
        tokens = tokenize(text)
        if not tokens:
            skipped[document_id] = "empty"
            continue
        normalized = " ".join(tokens)
        if normalized in seen_texts:
            skipped[document_id] = "duplicate_text"
            continue
        if per_source[source_id] >= max_per_source:
            skipped[document_id] = "source_limit"
            continue
        citation = f"[{source_id}:{document_id}]"
        cost = len(tokenize(citation + " " + text))
        if used + cost > max_tokens:
            skipped[document_id] = "budget"
            continue
        citation = f"[{source_id}:{document_id}]"
        selected.append(
            ContextItem(
                citation=citation,
                document_id=document_id,
                source_id=source_id,
                token_count=cost,
                score=float(scores[document_id]),
                text=text,
            )
        )
        used += cost
        seen_texts.add(normalized)
        per_source[source_id] += 1
    if not selected:
        raise ValueError("no chunk fits the context budget")
    rendered = "\n\n".join(f"{item.citation} {item.text}" for item in selected)
    assert len(tokenize(rendered)) == used
    return ContextPack(tuple(selected), used, max_tokens, dict(skipped), rendered)


def index_storage_gib(num_vectors: int, dimension: int, bits_per_component: float) -> float:
    """Return raw vector payload in GiB using powers-of-two units."""

    num_vectors = _positive_int("num_vectors", num_vectors)
    dimension = _positive_int("dimension", dimension)
    bits_per_component = _positive_finite("bits_per_component", bits_per_component)
    return num_vectors * dimension * bits_per_component / 8.0 / 2**30


def brute_ann_scenario(
    *,
    num_vectors: int,
    dimension: int,
    cpu_flops_per_second: float,
    visited_vectors: int,
) -> dict[str, float]:
    """Return a transparent exact-vs-ANN arithmetic scenario."""

    num_vectors = _positive_int("num_vectors", num_vectors)
    dimension = _positive_int("dimension", dimension)
    visited_vectors = _positive_int("visited_vectors", visited_vectors)
    cpu_flops_per_second = _positive_finite("cpu_flops_per_second", cpu_flops_per_second)
    exact_flops = 2.0 * num_vectors * dimension
    ann_flops = 2.0 * visited_vectors * dimension
    exact_seconds = exact_flops / cpu_flops_per_second
    ann_milliseconds = ann_flops / cpu_flops_per_second * 1e3
    return {
        "exact_flops": exact_flops,
        "exact_seconds": exact_seconds,
        "ann_flops": ann_flops,
        "ann_milliseconds": ann_milliseconds,
        "speedup": exact_seconds * 1e3 / ann_milliseconds,
    }


def cascade_latency_scenario() -> dict[str, float]:
    """Return a scenario mixing 2PT compute estimates and an assumed decode rate."""

    peak = 990e12
    mfu = 0.5
    effective = peak * mfu
    encoder_parameters = 0.6e9
    query_tokens = 32
    candidates = 100
    rerank_tokens = 432
    generator_parameters = 8.03e9
    context_tokens = 512 + 8 * 400
    answer_tokens = 300
    decode_tokens_per_second = 209.4
    stages = {
        "embed_ms": 2 * encoder_parameters * query_tokens / effective * 1e3,
        "ann_ms": brute_ann_scenario(
            num_vectors=10_000_000,
            dimension=1024,
            cpu_flops_per_second=10e9,
            visited_vectors=2_000,
        )["ann_milliseconds"],
        "rerank_ms": candidates * 2 * encoder_parameters * rerank_tokens / effective * 1e3,
        "prefill_ms": 2 * generator_parameters * context_tokens / effective * 1e3,
        "decode_ms": answer_tokens / decode_tokens_per_second * 1e3,
    }
    total = sum(stages.values())
    return {
        **stages,
        "total_ms": total,
        "context_tokens": context_tokens,
        "decode_share": stages["decode_ms"] / total,
        "retrieval_before_prefill_share": (
            stages["embed_ms"] + stages["ann_ms"] + stages["rerank_ms"]
        )
        / total,
    }


def long_context_vs_rag(*, long_tokens: int = 200_000, rag_tokens: int = 3_712) -> dict[str, float]:
    """Compare prefill, KV payload, and token count under one explicit model geometry."""

    long_tokens = _positive_int("long_tokens", long_tokens)
    rag_tokens = _positive_int("rag_tokens", rag_tokens)
    if rag_tokens >= long_tokens:
        raise ValueError("rag_tokens must be smaller than long_tokens")
    generator_parameters = 8.03e9
    effective_flops = 990e12 * 0.5
    kv_bytes_per_token = 128 * 2**10
    long_prefill_seconds = 2 * generator_parameters * long_tokens / effective_flops
    rag_prefill_seconds = 2 * generator_parameters * rag_tokens / effective_flops
    ratio = long_tokens / rag_tokens
    return {
        "long_tokens": long_tokens,
        "rag_tokens": rag_tokens,
        "long_prefill_seconds": long_prefill_seconds,
        "rag_prefill_seconds": rag_prefill_seconds,
        "long_kv_gib": long_tokens * kv_bytes_per_token / 2**30,
        "rag_kv_gib": rag_tokens * kv_bytes_per_token / 2**30,
        "ratio": ratio,
        "kv_bytes_per_token": kv_bytes_per_token,
    }


# Authored synthetic diagnostic collection, frozen separately from lecture prose.
# Seven development queries; no held-out evaluation or cross-language comparison.
DATASETS: dict[str, dict[str, object]] = {
    "ru": {
        "documents": [
            {"document_id": "d00", "source_id": "ann", "text": "HNSW строит многослойный граф малого мира для приближённого поиска ближайших соседей среди векторов."},
            {"document_id": "d01", "source_id": "ann", "text": "IVF PQ делит векторы на кластеры и кодирует остатки, уменьшая память индекса для сотен миллионов объектов."},
            {"document_id": "d02", "source_id": "hybrid", "text": "Reciprocal rank fusion RRF объединяет ранги BM25 и dense поиска без калибровки их оценок."},
            {"document_id": "d03", "source_id": "sparse", "text": "BM25 использует частоту терма, обратную документную частоту и нормировку длины документа."},
            {"document_id": "d04", "source_id": "dense", "text": "Dense bi encoder независимо кодирует запрос и документ в векторы, после чего сравнивает их косинусом."},
            {"document_id": "d05", "source_id": "rerank", "text": "Cross encoder совместно читает запрос и документ и точнее оценивает релевантность, но требует отдельного прохода на каждую пару."},
            {"document_id": "d06", "source_id": "chunking", "text": "Overlap между соседними чанками сохраняет локальный контекст на границе разбиения документа."},
            {"document_id": "d07", "source_id": "chunking", "text": "Late chunking сначала кодирует длинный документ целиком, поэтому местоимения и анафоры сохраняют контекст после нарезки."},
            {"document_id": "d08", "source_id": "routing", "text": "Long context помещает целый длинный документ в промпт и подходит для целостного анализа одного файла."},
            {"document_id": "d09", "source_id": "rag", "text": "RAG извлекает релевантные фрагменты корпуса, собирает контекст с цитатами и передаёт его генератору."},
            {"document_id": "d10", "source_id": "metrics", "text": "nDCG учитывает градуированную релевантность и логарифмически штрафует низкие позиции."},
            {"document_id": "d11", "source_id": "metrics", "text": "MRR измеряет обратный ранг первого релевантного результата и игнорирует остальные найденные документы."},
            {"document_id": "d12", "source_id": "packing", "text": "Context packing удаляет дубликаты, соблюдает токенный бюджет и сохраняет ссылки на источники."},
        ],
        "queries": [
            {"query_id": "q0", "text": "как слить результаты bm25 и dense", "relevance": {"d02": 3, "d09": 1}},
            {"query_id": "q1", "text": "граф соседей для approximate nearest neighbor", "relevance": {"d00": 3, "d01": 1}},
            {"query_id": "q2", "text": "позиция первого полезного результата", "relevance": {"d11": 3, "d10": 1}},
            {"query_id": "q3", "text": "как сохранить анафоры после нарезки документа", "relevance": {"d07": 3, "d06": 2}},
            {"query_id": "q4", "text": "почему cross encoder нельзя запустить на всём корпусе", "relevance": {"d05": 3, "d04": 1}},
            {"query_id": "q5", "text": "когда лучше прочитать весь длинный файл вместо retrieval", "relevance": {"d08": 3, "d09": 2}},
            {"query_id": "q6", "text": "context packing: как собрать найденные фрагменты с цитатами в лимит токенов", "relevance": {"d12": 3, "d09": 2}},
        ],
    },
    "en": {
        "documents": [
            {"document_id": "d00", "source_id": "ann", "text": "HNSW builds a multi-layer small-world graph for approximate nearest-neighbor search over vectors."},
            {"document_id": "d01", "source_id": "ann", "text": "IVF PQ partitions vectors into clusters and quantizes residuals, reducing index memory for hundreds of millions of objects."},
            {"document_id": "d02", "source_id": "hybrid", "text": "Reciprocal rank fusion RRF combines BM25 and dense-search rankings without calibrating their scores."},
            {"document_id": "d03", "source_id": "sparse", "text": "BM25 uses term frequency, inverse document frequency, and document-length normalization."},
            {"document_id": "d04", "source_id": "dense", "text": "A dense bi encoder encodes the query and document independently as vectors and compares them with cosine similarity."},
            {"document_id": "d05", "source_id": "rerank", "text": "A cross encoder reads the query and document jointly and scores relevance more accurately, but needs a separate forward pass for every pair."},
            {"document_id": "d06", "source_id": "chunking", "text": "Overlap between adjacent chunks preserves local context at a document boundary."},
            {"document_id": "d07", "source_id": "chunking", "text": "Late chunking first encodes the long document as a whole, so pronouns and anaphora retain context after splitting."},
            {"document_id": "d08", "source_id": "routing", "text": "Long context places the entire long document in the prompt and suits holistic analysis of one file."},
            {"document_id": "d09", "source_id": "rag", "text": "RAG retrieves relevant corpus passages, assembles a cited context, and sends it to the generator."},
            {"document_id": "d10", "source_id": "metrics", "text": "nDCG supports graded relevance and applies a logarithmic discount to lower ranks."},
            {"document_id": "d11", "source_id": "metrics", "text": "MRR measures the reciprocal rank of the first relevant result and ignores additional relevant documents."},
            {"document_id": "d12", "source_id": "packing", "text": "Context packing removes duplicates, respects a token budget, and preserves source citations."},
        ],
        "queries": [
            {"query_id": "q0", "text": "how to merge bm25 and dense results", "relevance": {"d02": 3, "d09": 1}},
            {"query_id": "q1", "text": "a graph for approximate nearest neighbor search", "relevance": {"d00": 3, "d01": 1}},
            {"query_id": "q2", "text": "position of the first useful result", "relevance": {"d11": 3, "d10": 1}},
            {"query_id": "q3", "text": "preserve anaphora after splitting a document", "relevance": {"d07": 3, "d06": 2}},
            {"query_id": "q4", "text": "why a cross encoder cannot score the whole corpus", "relevance": {"d05": 3, "d04": 1}},
            {"query_id": "q5", "text": "when to read the whole long file instead of retrieval", "relevance": {"d08": 3, "d09": 2}},
            {"query_id": "q6", "text": "pack retrieved passages with citations into a token limit", "relevance": {"d12": 3, "d09": 2}},
        ],
    },
}


def compute_dataset_metrics(language: str) -> dict[str, object]:
    """Build every teaching ranking and metric for one language dataset."""

    if language not in DATASETS:
        raise ValueError("language must be 'ru' or 'en'")
    dataset = DATASETS[language]
    documents = list(dataset["documents"])
    queries = list(dataset["queries"])
    document_ids = [str(record["document_id"]) for record in documents]
    texts = [str(record["text"]) for record in documents]
    text_by_id = dict(zip(document_ids, texts))
    bm25 = BM25Index(document_ids, texts)
    dense = LSAIndex(document_ids, texts, dimension=4)
    rankings: dict[str, dict[str, tuple[str, ...]]] = {
        "bm25": {},
        "dense_lsa": {},
        "rrf": {},
        "reranked": {},
    }
    traces: dict[str, list[dict[str, object]]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        query_text = str(query["text"])
        bm25_results = bm25.search(query_text)
        dense_results = dense.search(query_text)
        bm25_ranking = tuple(result.document_id for result in bm25_results)
        dense_ranking = tuple(result.document_id for result in dense_results)
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_ranking, "dense_lsa": dense_ranking},
            k=60.0,
            depth=8,
        )
        fused_ranking = tuple(result.document_id for result in fused)
        reranked_head = rerank_candidates(query_text, fused_ranking[:8], text_by_id)
        reranked_ranking = tuple(result.document_id for result in reranked_head) + fused_ranking[8:]
        rankings["bm25"][query_id] = bm25_ranking
        rankings["dense_lsa"][query_id] = dense_ranking
        rankings["rrf"][query_id] = fused_ranking
        rankings["reranked"][query_id] = reranked_ranking
        if query_id == "q0":
            traces[query_id] = [
                {
                    "document_id": result.document_id,
                    "rank": result.rank,
                    "score": result.score,
                    "source_ranks": dict(result.source_ranks),
                    "contributions": dict(result.contributions),
                }
                for result in fused[:5]
            ]
        if query_id == "q6":
            score_maps[query_id] = {
                result.document_id: result.score for result in reranked_head
            }
    metrics = {
        method: evaluate_rankings(method_rankings, queries, k=5)
        for method, method_rankings in rankings.items()
    }
    documents_for_pack = {
        str(record["document_id"]): {
            "source_id": str(record["source_id"]),
            "text": str(record["text"]),
        }
        for record in documents
    }
    q6_ranking = rankings["reranked"]["q6"]
    q6_scores = score_maps["q6"]
    # The reranker scores only the first eight candidates; the pack consumes that head.
    pack = pack_context(
        q6_ranking[:8],
        q6_scores,
        documents_for_pack,
        max_tokens=42,
        max_per_source=1,
    )
    return {
        "metrics": metrics,
        "rankings": {
            method: {query_id: list(ranking) for query_id, ranking in method_rankings.items()}
            for method, method_rankings in rankings.items()
        },
        "rrf_trace_q0": traces["q0"],
        "context_pack_q6": {
            "selected_ids": [item.document_id for item in pack.items],
            "citations": [item.citation for item in pack.items],
            "used_tokens": pack.used_tokens,
            "max_tokens": pack.max_tokens,
            "skipped": dict(pack.skipped),
            "rendered_text": pack.rendered_text,
        },
    }


def build_reference_payload() -> dict[str, object]:
    """Return the complete frozen numerical contract for Module 17."""

    storage = {
        "num_vectors": 10_000_000,
        "dimension": 1024,
        "fp32_gib": index_storage_gib(10_000_000, 1024, 32),
        "fp16_gib": index_storage_gib(10_000_000, 1024, 16),
        "int8_gib": index_storage_gib(10_000_000, 1024, 8),
        "binary_gib": index_storage_gib(10_000_000, 1024, 1),
        "mrl256_int8_gib": index_storage_gib(10_000_000, 256, 8),
    }
    total_tokens = 20_000_000
    chunks = analytical_chunk_count(total_tokens, 400, 64)
    encoder_flops = 2 * 0.6e9 * chunks * 400
    corpus = {
        "total_tokens": total_tokens,
        "chunk_size": 400,
        "overlap": 64,
        "chunks": chunks,
        "encoder_seconds_floor": encoder_flops / (990e12 * 0.5),
        "fp32_mib": chunks * 1024 * 4 / 2**20,
        "int8_mib": chunks * 1024 / 2**20,
        "binary_mib": chunks * 1024 / 8 / 2**20,
    }
    chunk_example_text = (
        "retrieval first splits a long source into overlapping windows while keeping "
        "source identifiers offsets and citations so later context packing remains traceable"
    )
    chunk_example = [
        asdict(chunk)
        for chunk in chunk_document(
            "demo",
            chunk_example_text,
            chunk_size=8,
            overlap=3,
        )
    ]
    return {
        "schema_version": 2,
        "storage": storage,
        "ann": brute_ann_scenario(
            num_vectors=10_000_000,
            dimension=1024,
            cpu_flops_per_second=10e9,
            visited_vectors=2_000,
        ),
        "corpus": corpus,
        "chunk_example": chunk_example,
        "cascade": cascade_latency_scenario(),
        "long_context_vs_rag": long_context_vs_rag(),
        "datasets": DATASETS,
        "retrieval_results": {
            language: compute_dataset_metrics(language) for language in ("ru", "en")
        },
    }


def canonical_json(payload: Mapping[str, object]) -> str:
    """Serialize a payload deterministically for hashing and notebook embedding."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_reference(path: Path = REFERENCE_PATH) -> Path:
    """Write the deterministic reference JSON and return its path."""

    payload = build_reference_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_reference(path: Path = REFERENCE_PATH) -> dict[str, object]:
    """Recompute the contract and require exact JSON equality."""

    expected = build_reference_payload()
    actual = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json(actual) != canonical_json(expected):
        raise AssertionError(f"reference drift detected: {path}")
    return actual


def main() -> None:
    """Validate by default; write only when --write is explicitly supplied."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite reference_metrics.json")
    args = parser.parse_args()
    if args.write:
        path = write_reference()
        print(f"wrote: {path}")
    payload = validate_reference()
    print(f"read-only validation passed: {REFERENCE_PATH}")
    print("storage GiB:", {key: round(value, 4) for key, value in payload["storage"].items() if key.endswith("_gib")})
    print("chunks:", payload["corpus"]["chunks"], "binary MiB:", round(payload["corpus"]["binary_mib"], 3))
    for language in ("ru", "en"):
        print(language, payload["retrieval_results"][language]["metrics"])


if __name__ == "__main__":
    main()
