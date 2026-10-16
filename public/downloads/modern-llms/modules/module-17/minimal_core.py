"""RRF and exact late interaction: executable, model-independent mechanisms."""
import numpy as np


def rrf(rankings, k=60):
    """Accumulate one-based reciprocal ranks, then sort U unique candidates."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")
    scores = {}
    for ranking in rankings:
        if any(not isinstance(doc, str) or not doc for doc in ranking):
            raise ValueError("IDs must be nonempty strings")
        if len(ranking) != len(set(ranking)):
            raise ValueError("duplicate document in a branch")
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc)), scores


def maxsim(query, document):
    """Q:[nq,d], D:[nd,d]; sum the best document match for each query row."""
    q, d = np.asarray(query, float), np.asarray(document, float)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]:
        raise ValueError("expected two matrices with one shared feature dimension")
    if not all(q.shape) or not all(d.shape):
        raise ValueError("empty token matrices")
    if not np.isfinite(q).all() or not np.isfinite(d).all():
        raise ValueError("non-finite vectors")
    if not (np.allclose(np.linalg.norm(q, axis=1), 1)
            and np.allclose(np.linalg.norm(d, axis=1), 1)):
        raise ValueError("normalize token vectors before cosine MaxSim")
    similarities = q @ d.T
    return float(similarities.max(axis=1).sum()), similarities


order, scores = rrf([["a", "b"], ["b", "c"]])
assert order == ["b", "a", "c"]
assert np.isclose(scores["b"], 1 / 62 + 1 / 61)
score, matrix = maxsim(np.eye(2), np.array([[1.0, 0.0]]))
assert score == 1.0 and matrix.shape == (2, 1)
