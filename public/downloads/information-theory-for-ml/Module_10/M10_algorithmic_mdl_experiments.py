from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_experiment"
OUT.mkdir(exist_ok=True)

SEED = 20260713
rng = np.random.default_rng(SEED)


def repeat_to_length(pattern: bytes, n: int) -> bytes:
    return (pattern * (n // len(pattern) + 1))[:n]


def python_prng_bytes(n: int, seed: int = 7) -> bytes:
    # A short generator plus a short seed describes this object, even though
    # generic compressors will usually see random-looking bytes.
    r = random.Random(seed)
    return bytes(r.randrange(256) for _ in range(n))


def hash_counter_stream(n: int, seed: bytes = b"course-m10") -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:n])


COMPRESSORS: dict[str, Callable[[bytes], bytes]] = {
    "gzip": lambda x: gzip.compress(x, compresslevel=9),
    "bz2": lambda x: bz2.compress(x, compresslevel=9),
    "lzma": lambda x: lzma.compress(x, preset=9),
}


def compressor_experiment(n: int = 65536) -> list[dict[str, float | str]]:
    corpus = {
        "zeros": b"\x00" * n,
        "periodic": repeat_to_length(b"abc123|", n),
        "English-like": repeat_to_length(
            b"prediction and compression are two views of the same log-loss; ", n
        ),
        "PRNG seed=7": python_prng_bytes(n, 7),
        "SHA256 stream": hash_counter_stream(n),
    }

    rows: list[dict[str, float | str]] = []
    for name, data in corpus.items():
        for comp_name, comp in COMPRESSORS.items():
            packed = comp(data)
            rows.append(
                {
                    "object": name,
                    "compressor": comp_name,
                    "raw_bytes": len(data),
                    "compressed_bytes": len(packed),
                    "bits_per_byte": 8.0 * len(packed) / len(data),
                }
            )

    names = list(corpus)
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for j, comp_name in enumerate(COMPRESSORS):
        vals = [
            next(
                float(row["bits_per_byte"])
                for row in rows
                if row["object"] == name and row["compressor"] == comp_name
            )
            for name in names
        ]
        ax.bar(x + (j - 1) * width, vals, width=width, label=comp_name)
    ax.axhline(8.0, linewidth=1.2, linestyle="--", label="raw: 8 bits/byte")
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("bits per input byte")
    ax.set_title("Practical compressors are computable upper-bound mechanisms, not K-oracles")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "M10_experiment_compressor_bounds.png", dpi=190)
    plt.close(fig)

    return rows


@dataclass
class PrequentialResult:
    name: str
    total_bits: float
    bits_per_label: float
    final_accuracy: float
    cumulative_points: list[tuple[int, float]]


def make_classification_representations(n: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = rng.normal(size=(n, 4))
    logits = 1.3 * z[:, 0] - 0.9 * z[:, 1] + 0.5 * z[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    easy = z + rng.normal(scale=0.08, size=z.shape)
    nuisance = rng.normal(size=(n, 96))
    hard = np.concatenate([z + rng.normal(scale=0.18, size=z.shape), nuisance], axis=1)
    return easy, hard, y.astype(int)


def prequential_code(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    boundaries: list[int],
) -> PrequentialResult:
    n = len(y)
    if boundaries[-1] != n:
        raise ValueError("last boundary must equal dataset size")

    first = boundaries[0]
    total_bits = float(first)  # uniform binary code for first labels
    points: list[tuple[int, float]] = [(first, total_bits)]

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs", random_state=SEED),
        )
        model.fit(X[:left], y[:left])
        probs = model.predict_proba(X[left:right])
        block_nats = log_loss(y[left:right], probs, labels=[0, 1], normalize=False)
        total_bits += float(block_nats / math.log(2.0))
        points.append((right, total_bits))

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, C=1.0, solver="lbfgs", random_state=SEED),
    )
    split = int(0.8 * n)
    final_model.fit(X[:split], y[:split])
    acc = accuracy_score(y[split:], final_model.predict(X[split:]))

    return PrequentialResult(
        name=name,
        total_bits=total_bits,
        bits_per_label=total_bits / n,
        final_accuracy=float(acc),
        cumulative_points=points,
    )


def prequential_experiment() -> list[PrequentialResult]:
    X_easy, X_hard, y = make_classification_representations()
    boundaries = [64, 128, 256, 512, 1024, 2048, 3500, len(y)]
    results = [
        prequential_code(X_easy, y, "4 informative dimensions", boundaries),
        prequential_code(X_hard, y, "4 signal + 96 nuisance dimensions", boundaries),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for result in results:
        xs = [p[0] for p in result.cumulative_points]
        ys = [p[1] / p[0] for p in result.cumulative_points]
        ax.plot(xs, ys, marker="o", label=result.name)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="uniform binary code")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of labels encoded")
    ax.set_ylabel("cumulative bits per label")
    ax.set_title("Prequential MDL rewards representations that are learned quickly")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "M10_experiment_prequential.png", dpi=190)
    plt.close(fig)
    return results


def ncd(x: bytes, y: bytes, compressor: Callable[[bytes], bytes]) -> float:
    cx = len(compressor(x))
    cy = len(compressor(y))
    cxy = min(len(compressor(x + y)), len(compressor(y + x)))
    return (cxy - min(cx, cy)) / max(cx, cy)


def mutate(data: bytes, every: int, replacement: bytes) -> bytes:
    out = bytearray(data)
    for i in range(0, len(out), every):
        out[i : i + len(replacement)] = replacement[: min(len(replacement), len(out) - i)]
    return bytes(out)


def ncd_experiment() -> tuple[list[str], np.ndarray]:
    n = 24000
    objects = {
        "text-A": repeat_to_length(b"the model predicts the next token and pays in bits. ", n),
        "text-B": mutate(repeat_to_length(b"the model predicts the next token and pays in bits. ", n), 337, b"loss"),
        "code-A": repeat_to_length(b"for(i=0;i<n;i++){sum+=a[i];}\n", n),
        "code-B": mutate(repeat_to_length(b"for(i=0;i<n;i++){sum+=a[i];}\n", n), 401, b"while"),
        "dna-A": repeat_to_length(b"ACGTACGTGATTACA", n),
        "dna-B": mutate(repeat_to_length(b"ACGTACGTGATTACA", n), 389, b"TTAA"),
    }
    labels = list(objects)
    comp = lambda b: bz2.compress(b, compresslevel=9)
    matrix = np.zeros((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            matrix[i, j] = ncd(objects[a], objects[b], comp)

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    im = ax.imshow(matrix, vmin=0.0, vmax=max(1.0, float(matrix.max())))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Normalized compression distance: shared regularity becomes proximity")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="NCD")
    fig.tight_layout()
    fig.savefig(OUT / "M10_experiment_ncd.png", dpi=190)
    plt.close(fig)
    return labels, matrix


def main() -> None:
    compressor_rows = compressor_experiment()
    prequential_results = prequential_experiment()
    labels, matrix = ncd_experiment()

    summary = {
        "compressor_rows": compressor_rows,
        "prequential": [
            {
                "name": r.name,
                "total_bits": r.total_bits,
                "bits_per_label": r.bits_per_label,
                "final_accuracy": r.final_accuracy,
            }
            for r in prequential_results
        ],
        "ncd_labels": labels,
        "ncd_matrix": matrix.tolist(),
    }
    (OUT / "M10_experiment_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Compression experiment (bits per byte):")
    for row in compressor_rows:
        print(
            f"  {row['object']:14s} {row['compressor']:4s}: "
            f"{float(row['bits_per_byte']):.3f}"
        )
    print("\nPrequential results:")
    for r in prequential_results:
        print(
            f"  {r.name}: {r.bits_per_label:.4f} bits/label, "
            f"final accuracy={r.final_accuracy:.4f}"
        )
    print("\nNCD matrix:")
    print(np.array2string(matrix, precision=3, suppress_small=True))


if __name__ == "__main__":
    main()
