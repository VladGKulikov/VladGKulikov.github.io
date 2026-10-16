#!/usr/bin/env python3
"""Reviewed numerical core and frozen reference data for Module 3.

The learner notebooks embed a module-owned numerical contract.  The same reviewed
PyTorch teaching core is extracted into both RU and EN notebooks, so lecture examples,
answer sheets, builders, and the published notebooks cannot silently diverge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    """Return SHA-256 of canonical JSON bytes."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _softmax(values: list[float]) -> list[float]:
    """Stable scalar-list softmax used only to build frozen reference numbers."""

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _attention_reference() -> dict[str, Any]:
    """Return the three-token causal-attention worked example."""

    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    v = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]
    allowed = [[column <= row for column in range(3)] for row in range(3)]
    scale = 1.0 / math.sqrt(2.0)

    scores: list[list[float | None]] = []
    weights: list[list[float]] = []
    outputs: list[list[float]] = []
    for row in range(3):
        raw = [sum(q[row][d] * k[column][d] for d in range(2)) * scale for column in range(3)]
        visible = [raw[column] for column in range(3) if allowed[row][column]]
        visible_weights = iter(_softmax(visible))
        row_weights = [next(visible_weights) if allowed[row][column] else 0.0 for column in range(3)]
        row_scores = [raw[column] if allowed[row][column] else None for column in range(3)]
        output = [sum(row_weights[column] * v[column][d] for column in range(3)) for d in range(2)]
        scores.append(row_scores)
        weights.append(row_weights)
        outputs.append(output)

    return {
        "q": q,
        "k": k,
        "v": v,
        "allowed": allowed,
        "scores": scores,
        "weights": weights,
        "output": outputs,
    }


def build_reference_payload() -> dict[str, Any]:
    """Build all version-stable numerical reference values for Module 3."""

    layer_input = [1.0, 2.0, 3.0, 4.0]
    mean = sum(layer_input) / len(layer_input)
    variance = sum((value - mean) ** 2 for value in layer_input) / len(layer_input)
    normalized = [(value - mean) / math.sqrt(variance) for value in layer_input]

    logits = [1.2, 0.3, -0.4, 2.0]
    probabilities = _softmax(logits)
    target = 3
    nll = -math.log(probabilities[target])

    positional_table = []
    for position in range(5):
        row: list[float] = []
        for coordinate in range(0, 4, 2):
            divisor = 10_000.0 ** (coordinate / 4)
            row.extend((math.sin(position / divisor), math.cos(position / divisor)))
        positional_table.append(row)

    d_model = 512
    d_ff = 2048
    mha = 4 * d_model * d_model
    ffn = 2 * d_model * d_ff

    payload: dict[str, Any] = {
        "schema_version": 1,
        "attention": _attention_reference(),
        "positional_encoding": {
            "length": 5,
            "d_model": 4,
            "table": positional_table,
        },
        "layer_norm": {
            "input": layer_input,
            "mean": mean,
            "variance": variance,
            "normalized": normalized,
        },
        "next_token_nll": {
            "logits": logits,
            "target": target,
            "probabilities": probabilities,
            "nll": nll,
        },
        "parameter_floor": {
            "d_model": d_model,
            "d_ff": d_ff,
            "mha_weights": mha,
            "ffn_weights": ffn,
            "encoder_layer": mha + ffn,
            "decoder_layer": 2 * mha + ffn,
            "ffn_share_encoder": ffn / (mha + ffn),
        },
        "scaling_experiment": {
            "seed": 18,
            "samples": 50_000,
            "d_k": 64,
            "unscaled_variance_target": 64.0,
            "scaled_variance_target": 1.0,
            "relative_tolerance": 0.08,
        },
        "tiny_lm": {
            "seed": 18,
            "vocab_size": 6,
            "period": [0, 1, 2, 3, 4, 5],
            "context": 12,
            "training_repeats": 20,
            "steps": 121,
            "learning_rate": 0.03,
            "new_tokens": 15,
            "loss_threshold": 0.02,
            "accuracy_threshold": 0.99,
            "continuation_accuracy_threshold": 0.95,
        },
    }
    return payload


def validate_reference_payload(payload: object) -> dict[str, Any]:
    """Validate a loaded reference payload against the reviewed calculation."""

    expected = build_reference_payload()
    if not isinstance(payload, dict):
        raise ValueError("reference payload must be a JSON object")
    if canonical_json(payload) != canonical_json(expected):
        raise ValueError("reference payload differs from reviewed Module 3 calculations")
    return expected


def load_reference_payload() -> dict[str, Any]:
    """Load and validate the committed Module 3 reference payload."""

    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    return validate_reference_payload(payload)


def write_reference_payload() -> Path:
    """Write the reviewed payload intentionally; never called by default."""

    payload = build_reference_payload()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return REFERENCE_PATH


# BEGIN EDUCATIONAL CORE: transformer_core

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

EmptyRowPolicy = Literal["error", "zero"]

# BEGIN EDUCATIONAL CORE: validation_helpers

def _positive_int(name: str, value: object) -> int:
    """Return a strictly positive integer and reject booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(name: str, value: object) -> int:
    """Return a non-negative integer and reject booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def expect_raises(error_type: type[BaseException], fn, *args, **kwargs) -> str:
    """Run ``fn`` and return the expected exception message."""

    try:
        fn(*args, **kwargs)
    except error_type as exc:
        return str(exc)
    raise AssertionError(f"expected {error_type.__name__}")


def _finite_positive_real(name: str, value: object) -> float:
    """Return a positive finite scalar."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _floating_tensor(name: str, value: object, *, min_ndim: int) -> Tensor:
    """Validate a finite floating-point tensor without silently changing shape."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} may not contain an empty axis")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


# END EDUCATIONAL CORE: validation_helpers

# BEGIN EDUCATIONAL CORE: positional_and_masks

def sinusoidal_encoding(
    length: int,
    d_model: int,
    *,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return the Vaswani sinusoidal table with shape ``[length, d_model]``.

    This paired implementation requires an even model width.  Position is the first
    axis; sine occupies even coordinates and cosine occupies odd coordinates.
    """

    length = _positive_int("length", length)
    d_model = _positive_int("d_model", d_model)
    base = _finite_positive_real("base", base)
    if base <= 1.0:
        raise ValueError("base must be greater than 1")
    if d_model % 2:
        raise ValueError("d_model must be even for paired sine/cosine coordinates")
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError("dtype must be torch.float32 or torch.float64")

    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    pair_index = torch.arange(0, d_model, 2, device=device, dtype=dtype)
    divisor = torch.pow(torch.tensor(base, device=device, dtype=dtype), pair_index / d_model)
    table = torch.empty((length, d_model), device=device, dtype=dtype)
    table[:, 0::2] = torch.sin(position / divisor)
    table[:, 1::2] = torch.cos(position / divisor)
    return table


def causal_allowed(
    t_q: int,
    t_k: int,
    *,
    query_offset: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a boolean causal mask of shape ``[T_q, T_k]``.

    ``query_offset`` is the absolute position of local query row zero.  It is zero for
    ordinary self-attention and equals the cached-prefix length for incremental decode.
    ``True`` means that the query may read the key.
    """

    t_q = _positive_int("t_q", t_q)
    t_k = _positive_int("t_k", t_k)
    if isinstance(query_offset, bool) or not isinstance(query_offset, Integral):
        raise TypeError("query_offset must be an integer")
    query_offset = int(query_offset)
    if query_offset < 0:
        raise ValueError("query_offset must be non-negative")
    q_pos = query_offset + torch.arange(t_q, device=device)[:, None]
    k_pos = torch.arange(t_k, device=device)[None, :]
    return k_pos <= q_pos


# END EDUCATIONAL CORE: positional_and_masks

# BEGIN EDUCATIONAL CORE: attention_core

@dataclass(frozen=True)
class AttentionResult:
    """Transparent outputs of one scaled dot-product attention call."""

    output: Tensor
    weights: Tensor
    scores: Tensor
    empty_rows: Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor | None = None,
    *,
    empty_row: EmptyRowPolicy = "error",
) -> AttentionResult:
    """Compute numerically stable educational scaled dot-product attention.

    Shapes:
        ``q [..., T_q, d_k]``
        ``k [..., T_k, d_k]``
        ``v [..., T_k, d_v]``
        ``allowed`` broadcasts to ``[..., T_q, T_k]`` and uses ``True = allowed``.

    The calculation uses at least float32 for logits.  A fully masked query row is not
    a probability distribution: ``empty_row='error'`` rejects it, while ``'zero'``
    returns zero weights and a zero output for explicitly padded query rows.
    """

    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")
    q = _floating_tensor("q", q, min_ndim=2)
    k = _floating_tensor("k", k, min_ndim=2)
    v = _floating_tensor("v", v, min_ndim=2)
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same device")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share d_k")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share T_k")
    try:
        batch_shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    except RuntimeError as exc:
        raise ValueError("q/k/v batch dimensions are not broadcast-compatible") from exc

    compute_dtype = torch.float64 if torch.float64 in {q.dtype, k.dtype, v.dtype} else torch.float32
    qf, kf, vf = (tensor.to(dtype=compute_dtype) for tensor in (q, k, v))
    # V can introduce a batch axis too; the mask belongs to the joint batch shape.
    qf, kf, vf = (t.expand(*batch_shape, *t.shape[-2:]) for t in (qf, kf, vf))
    scores = (qf @ kf.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if not torch.isfinite(scores).all():
        raise FloatingPointError("q @ k.T overflowed; rescale the inputs")

    if allowed is None:
        allowed_b = torch.ones_like(scores, dtype=torch.bool)
    else:
        if not isinstance(allowed, Tensor):
            raise TypeError("allowed must be a torch.Tensor")
        if allowed.dtype is not torch.bool:
            raise TypeError("allowed must use bool dtype")
        try:
            allowed_b = torch.broadcast_to(allowed.to(device=scores.device), scores.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"allowed shape {tuple(allowed.shape)} does not broadcast to {tuple(scores.shape)}"
            ) from exc

    empty_rows = ~allowed_b.any(dim=-1)
    if empty_row == "error" and empty_rows.any():
        locations = torch.nonzero(empty_rows, as_tuple=False)[:8].cpu().tolist()
        raise ValueError(f"attention contains fully masked rows: {locations}")

    masked_scores = scores.masked_fill(~allowed_b, float("-inf"))
    row_has_support = ~empty_rows.unsqueeze(-1)
    row_max = torch.where(
        row_has_support,
        masked_scores.max(dim=-1, keepdim=True).values,
        torch.zeros_like(masked_scores[..., :1]),
    )
    exponentials = torch.where(
        allowed_b,
        torch.exp(masked_scores - row_max),
        torch.zeros_like(masked_scores),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    weights = exponentials / safe_denominator
    output = weights @ vf
    if not torch.isfinite(weights).all() or not torch.isfinite(output).all():
        raise FloatingPointError("attention produced a non-finite weight or output")
    return AttentionResult(output=output, weights=weights, scores=masked_scores, empty_rows=empty_rows)


# END EDUCATIONAL CORE: attention_core

# BEGIN EDUCATIONAL CORE: transformer_modules

class MultiHeadSelfAttention(nn.Module):
    """Minimal self-attention with explicit head geometry and padding semantics."""

    def __init__(self, d_model: int, n_heads: int, *, causal: bool = True) -> None:
        """Validate head geometry and create Q/K/V/O projections."""

        super().__init__()
        d_model = _positive_int("d_model", d_model)
        n_heads = _positive_int("n_heads", n_heads)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = bool(causal)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        token_valid: Tensor | None = None,
        empty_row: EmptyRowPolicy = "error",
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention to ``x [B,T,D]``.

        ``token_valid [B,T]`` marks real tokens.  Invalid query rows are fully masked;
        use ``empty_row='zero'`` when padding rows should yield zero attention output.
        """

        x = _floating_tensor("x", x, min_ndim=3)
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        batch, length, width = x.shape
        if token_valid is None:
            token_valid = torch.ones((batch, length), dtype=torch.bool, device=x.device)
        else:
            if not isinstance(token_valid, Tensor) or token_valid.dtype is not torch.bool:
                raise TypeError("token_valid must be a bool tensor")
            if token_valid.shape != (batch, length):
                raise ValueError(f"token_valid must have shape {(batch, length)}")
            token_valid = token_valid.to(device=x.device)

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            """Reshape ``[B,T,D]`` into ``[B,H,T,d_head]`` without copying semantics."""

            return tensor.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        allowed = token_valid[:, None, :, None] & token_valid[:, None, None, :]
        if self.causal:
            allowed &= causal_allowed(length, length, device=x.device)[None, None, :, :]
        result = scaled_dot_product_attention(q, k, v, allowed, empty_row=empty_row)
        merged = result.output.transpose(1, 2).contiguous().view(batch, length, width)
        output = self.out(merged.to(dtype=x.dtype))
        return (output, result.weights) if return_weights else output


class FeedForward(nn.Module):
    """Position-wise two-layer GELU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Create the two affine maps and intermediate GELU activation."""

        super().__init__()
        d_model = _positive_int("d_model", d_model)
        d_ff = _positive_int("d_ff", d_ff)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Preserve all leading axes and the residual width ``d_model``."""

        return self.net(x)


class DecoderBlock(nn.Module):
    """Minimal pre-norm decoder block with causal self-attention and an FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        """Create two pre-norm residual branches with fixed model width."""

        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x + Attention(LN(x)) + FFN(LN(...))`` with the same shape."""

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# END EDUCATIONAL CORE: transformer_modules

# BEGIN EDUCATIONAL CORE: parameter_floor

@dataclass(frozen=True)
class ParameterFloor:
    """Bias/norm-free parameter floor for the original Transformer sublayers."""

    mha_weights: int
    ffn_weights: int
    encoder_layer: int
    decoder_layer: int
    ffn_share_encoder: float


def transformer_parameter_floor(d_model: int, d_ff: int) -> ParameterFloor:
    """Count Q/K/V/O and two- or three-sublayer weights, excluding biases and norms."""

    d_model = _positive_int("d_model", d_model)
    d_ff = _positive_int("d_ff", d_ff)
    mha = 4 * d_model * d_model
    ffn = 2 * d_model * d_ff
    return ParameterFloor(
        mha_weights=mha,
        ffn_weights=ffn,
        encoder_layer=mha + ffn,
        decoder_layer=2 * mha + ffn,
        ffn_share_encoder=ffn / (mha + ffn),
    )


# END EDUCATIONAL CORE: parameter_floor

# BEGIN EDUCATIONAL CORE: tiny_lm

class TinyLM(nn.Module):
    """One-block causal language model used only for a bounded overfit smoke test."""

    def __init__(
        self,
        *,
        vocab_size: int = 6,
        d_model: int = 24,
        n_heads: int = 4,
        d_ff: int = 48,
        max_length: int = 32,
    ) -> None:
        """Create a bounded one-block causal LM with tied token/output weights."""

        super().__init__()
        self.vocab_size = _positive_int("vocab_size", vocab_size)
        self.max_length = _positive_int("max_length", max_length)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_length, d_model)
        self.block = DecoderBlock(d_model, n_heads, d_ff)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map integer IDs ``[B,T]`` to next-token logits ``[B,T,V]``."""

        if not isinstance(token_ids, Tensor) or token_ids.dtype is not torch.long:
            raise TypeError("token_ids must be a torch.long tensor")
        if token_ids.ndim != 2 or any(size <= 0 for size in token_ids.shape):
            raise ValueError("token_ids must have non-empty shape [B,T]")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("sequence exceeds max_length")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= self.vocab_size:
            raise ValueError("token_ids contain an ID outside the vocabulary")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.block(hidden)
        return self.head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(self, token_ids: Tensor, new_tokens: int) -> Tensor:
        """Greedily append ``new_tokens`` IDs while cropping the model context."""

        new_tokens = _non_negative_int("new_tokens", new_tokens)
        generated = token_ids.clone()
        was_training = self.training
        self.eval()
        try:
            # Cropping resets learned positions; this is not KV-cached decoding.
            for _ in range(new_tokens):
                context = generated[:, -self.max_length :]
                next_id = self(context)[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_id), dim=1)
        finally:
            self.train(was_training)
        return generated

# END EDUCATIONAL CORE: tiny_lm

# END EDUCATIONAL CORE: transformer_core


def main(argv: list[str] | None = None) -> int:
    """Validate the committed payload, or regenerate it only with ``--write``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="intentionally rewrite reference_metrics.json")
    args = parser.parse_args(argv)
    if args.write:
        path = write_reference_payload()
        payload = load_reference_payload()
        print(f"wrote {path}")
    else:
        payload = load_reference_payload()
        print(f"validated {REFERENCE_PATH}")
    print("sha256", sha256_json(payload))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
