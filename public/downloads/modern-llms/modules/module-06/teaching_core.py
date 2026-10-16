"""Visible NumPy teaching kernels embedded in both M6 notebooks."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import numpy as np
NormPlacement = Literal["pre", "peri"]
DepthMode = Literal["none", "pre", "post"]

def _floating_array(x: np.ndarray) -> np.ndarray:
    """Validate and return a floating NumPy array without changing its shape."""

    array = np.asarray(x)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("normalization expects a floating-point array")
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("the normalized last axis must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("inputs must be finite")
    return array


def _statistics_dtype(dtype: np.dtype) -> np.dtype:
    """Use at least float32 for reductions while preserving float64 inputs."""

    return np.dtype(np.float32) if np.dtype(dtype).itemsize < 4 else np.dtype(dtype)


def root_mean_square(x: np.ndarray, axis: int | None = -1, keepdims: bool = False) -> np.ndarray:
    """Compute RMS in a reduction dtype of at least float32."""

    array = _floating_array(x)
    work = array.astype(_statistics_dtype(array.dtype), copy=False)
    return np.sqrt(np.mean(work * work, axis=axis, keepdims=keepdims))


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray | float = 1.0,
    beta: np.ndarray | float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Layer-normalize `[..., d_model]` over the last axis and preserve dtype."""

    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    array = _floating_array(x)
    work_dtype = _statistics_dtype(array.dtype)
    work = array.astype(work_dtype, copy=False)
    mean = work.mean(axis=-1, keepdims=True)
    variance = np.mean((work - mean) ** 2, axis=-1, keepdims=True)
    result = (work - mean) / np.sqrt(variance + eps)
    for value in (gamma, beta):
        if np.shape(value) not in ((), (array.shape[-1],)) or not np.isfinite(value).all():
            raise ValueError("affine parameters must be finite scalars or [D] vectors")
    result = result * np.asarray(gamma, dtype=work_dtype) + np.asarray(beta, dtype=work_dtype)
    return result.astype(array.dtype, copy=False)


def rms_norm(
    x: np.ndarray,
    gamma: np.ndarray | float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS-normalize `[..., d_model]` over the last axis and preserve dtype."""

    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    array = _floating_array(x)
    work_dtype = _statistics_dtype(array.dtype)
    work = array.astype(work_dtype, copy=False)
    mean_square = np.mean(work * work, axis=-1, keepdims=True)
    result = work / np.sqrt(mean_square + eps)
    if np.shape(gamma) not in ((), (array.shape[-1],)) or not np.isfinite(gamma).all():
        raise ValueError("gamma must be a finite scalar or [D] vector")
    result = result * np.asarray(gamma, dtype=work_dtype)
    return result.astype(array.dtype, copy=False)


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Compute sigmoid without `exp` overflow on large negative inputs."""

    z = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z))


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Evaluate exact GELU = x * Phi(x) without a SciPy dependency."""

    z = np.asarray(x, dtype=np.float64)
    erf_values = np.fromiter(
        (math.erf(float(v) / math.sqrt(2.0)) for v in z.ravel()),
        dtype=np.float64,
        count=z.size,
    ).reshape(z.shape)
    return 0.5 * z * (1.0 + erf_values)


def silu(x: np.ndarray | float) -> np.ndarray:
    """Return SiLU/Swish: x * sigmoid(x)."""

    z = np.asarray(x, dtype=np.float64)
    return z * stable_sigmoid(z)


def gpt_oss_swiglu(
    gate: np.ndarray | float,
    up: np.ndarray | float,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> np.ndarray:
    """Apply the public gpt-oss gate clamp, up clamp, and `(up + 1)` shift."""

    if limit <= 0 or not math.isfinite(limit):
        raise ValueError("limit must be finite and positive")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    g = np.minimum(np.asarray(gate, dtype=np.float64), limit)
    u = np.clip(np.asarray(up, dtype=np.float64), -limit, limit)
    return g * stable_sigmoid(alpha * g) * (u + 1.0)


def attention_logits(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return scaled dot-product logits for `[T_q, d_head]` and `[T_k, d_head]`."""

    q = _floating_array(q)
    k = _floating_array(k)
    if q.ndim != 2 or k.ndim != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must be rank-2 with the same d_head")
    dtype = np.result_type(q.dtype, k.dtype, np.float32)
    return q.astype(dtype) @ k.astype(dtype).T / math.sqrt(q.shape[-1])


def qk_rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each query/key vector across its head dimension."""

    return rms_norm(np.asarray(x, dtype=np.float64), eps=eps)


@dataclass(frozen=True)
class BlockConfig:
    """Shape contract for the educational dense GQA + SwiGLU block."""

    d_model: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate canonical GQA dimensions and the numerical epsilon."""

        dims=(self.d_model, self.d_ff, self.n_heads, self.n_kv_heads, self.d_head)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in dims):
            raise ValueError("all dimensions must be positive integers")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("canonical block requires n_heads * d_head == d_model")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.eps <= 0 or not math.isfinite(self.eps):
            raise ValueError("eps must be finite and positive")


def causal_allowed(n_tokens: int) -> np.ndarray:
    """Return `[T, T]` boolean mask with `True` for visible keys."""

    if isinstance(n_tokens, bool) or not isinstance(n_tokens, (int, np.integer)) or n_tokens <= 0:
        raise ValueError("n_tokens must be a positive integer")
    return np.tri(n_tokens, n_tokens, dtype=bool)


def masked_softmax(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Row-wise softmax with `True = allowed`; reject fully masked rows."""

    scores = np.asarray(logits, dtype=np.float64)
    visible = np.broadcast_to(np.asarray(allowed, dtype=bool), scores.shape)
    if np.any(~np.any(visible, axis=-1)):
        raise ValueError("every query row must expose at least one key")
    if not np.isfinite(scores[visible]).all():
        raise FloatingPointError("allowed attention logits must be finite")
    masked = np.where(visible, scores, -np.inf)
    row_max = np.max(masked, axis=-1, keepdims=True)
    numerators = np.where(visible, np.exp(masked - row_max), 0.0)
    return numerators / np.sum(numerators, axis=-1, keepdims=True)


def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Reference attention for q/k/v shaped `[H, T, d_head]`."""

    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must share shape [H, T, d_head]")
    logits = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    return masked_softmax(logits, allowed) @ v


def grouped_query_attention(
    q_flat: np.ndarray,
    k_flat: np.ndarray,
    v_flat: np.ndarray,
    config: BlockConfig,
    allowed: np.ndarray,
) -> np.ndarray:
    """Apply GQA to flat projections and return `[T, d_model]`."""

    q_flat = np.asarray(q_flat)
    k_flat = np.asarray(k_flat)
    v_flat = np.asarray(v_flat)
    if q_flat.ndim != 2 or k_flat.ndim != 2 or v_flat.ndim != 2:
        raise ValueError("q_flat, k_flat, and v_flat must be rank-2")
    n_tokens = q_flat.shape[0]
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.shape != (n_tokens, n_tokens):
        raise ValueError("allowed must have shape [T, T]")
    expected = (
        (n_tokens, config.d_model),
        (n_tokens, config.n_kv_heads * config.d_head),
        (n_tokens, config.n_kv_heads * config.d_head),
    )
    if (q_flat.shape, k_flat.shape, v_flat.shape) != expected:
        raise ValueError(f"projection shapes must be {expected}")
    q = q_flat.reshape(n_tokens, config.n_heads, config.d_head).transpose(1, 0, 2)
    k = k_flat.reshape(n_tokens, config.n_kv_heads, config.d_head).transpose(1, 0, 2)
    v = v_flat.reshape(n_tokens, config.n_kv_heads, config.d_head).transpose(1, 0, 2)
    repeats = config.n_heads // config.n_kv_heads
    k = np.repeat(k, repeats, axis=0)
    v = np.repeat(v, repeats, axis=0)
    return sdpa(q, k, v, allowed).transpose(1, 0, 2).reshape(n_tokens, config.d_model)


class PreNormBlock:
    """Small NumPy block that exposes both residual updates for inspection."""

    def __init__(self, config: BlockConfig, rng: np.random.Generator):
        """Sample one deterministic bias-free block from ``rng``."""

        self.config = config

        def weight(fan_in: int, fan_out: int) -> np.ndarray:
            """Return a fan-in-scaled matrix shaped ``[fan_in, fan_out]``."""

            return rng.standard_normal((fan_in, fan_out)) / math.sqrt(fan_in)

        d, i = config.d_model, config.d_ff
        q_width = config.n_heads * config.d_head
        kv_width = config.n_kv_heads * config.d_head
        self.gamma_attn = np.ones(d)
        self.gamma_ffn = np.ones(d)
        self.wq = weight(d, q_width)
        self.wk = weight(d, kv_width)
        self.wv = weight(d, kv_width)
        self.wo = weight(q_width, d)
        self.w_gate = weight(d, i)
        self.w_up = weight(d, i)
        self.w_down = weight(i, d)

    def attention_update(self, x: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        """Return the attention branch update before residual addition."""

        z = rms_norm(x, self.gamma_attn, self.config.eps)
        mixed = grouped_query_attention(
            z @ self.wq,
            z @ self.wk,
            z @ self.wv,
            self.config,
            allowed,
        )
        return mixed @ self.wo

    def ffn_update(self, x: np.ndarray) -> np.ndarray:
        """Return the standard SwiGLU FFN update before residual addition."""

        z = rms_norm(x, self.gamma_ffn, self.config.eps)
        gate = z @ self.w_gate
        up = z @ self.w_up
        return (silu(gate) * up) @ self.w_down

    def forward(
        self,
        x: np.ndarray,
        allowed: np.ndarray,
        *,
        placement: NormPlacement = "pre",
        enable_attention: bool = True,
        enable_ffn: bool = True,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run pre-norm or Peri-LN and return output plus both updates."""

        if placement not in ("pre", "peri"):
            raise ValueError("placement must be 'pre' or 'peri'")
        h = np.asarray(x, dtype=np.float64)
        if h.ndim != 2 or h.shape[-1] != self.config.d_model:
            raise ValueError("x must have shape [T, d_model]")
        if not np.isfinite(h).all():
            raise FloatingPointError("x must contain only finite values")

        attn_update = np.zeros_like(h)
        if enable_attention:
            attn_update = self.attention_update(h, allowed)
            if placement == "peri":
                attn_update = rms_norm(attn_update, eps=self.config.eps)
        after_attention = h + attn_update

        ffn_update = np.zeros_like(h)
        if enable_ffn:
            ffn_update = self.ffn_update(after_attention)
            if placement == "peri":
                ffn_update = rms_norm(ffn_update, eps=self.config.eps)
        output = after_attention + ffn_update
        return output, {
            "attention_update": attn_update,
            "after_attention": after_attention,
            "ffn_update": ffn_update,
        }


@dataclass(frozen=True)
class DecoderShape:
    """Fields needed to count a bias-free dense decoder-only Transformer."""

    vocab_size: int
    d_model: int
    n_layers: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    tied_embeddings: bool = False

    def __post_init__(self) -> None:
        """Validate the dimensions assumed by the exact parameter counter."""

        dims=(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.d_ff,
            self.n_heads,
            self.n_kv_heads,
            self.d_head,
        )
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in dims):
            raise ValueError("all shape fields must be positive integers")
        if not isinstance(self.tied_embeddings, bool):
            raise ValueError("tied_embeddings must be boolean")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model for this counter")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


@dataclass(frozen=True)
class ParameterBreakdown:
    """Exact count split for the bias-free decoder assumed below."""

    attention: int
    mlp: int
    norms: int
    matrix_block: int
    full_block: int
    block: int  # backwards-compatible alias of full_block in the frozen contract
    embedding_tables: int
    final_norm: int
    total: int


def count_decoder_parameters(shape: DecoderShape) -> ParameterBreakdown:
    """Count GQA + SwiGLU weights, norm scales, embeddings, and final norm."""

    q_width = shape.n_heads * shape.d_head
    kv_width = shape.n_kv_heads * shape.d_head
    attention = shape.d_model * q_width + q_width * shape.d_model
    attention += 2 * shape.d_model * kv_width
    mlp = 3 * shape.d_model * shape.d_ff
    norms = 2 * shape.d_model
    matrix_block = attention + mlp
    full_block = matrix_block + norms
    tables = (1 if shape.tied_embeddings else 2) * shape.vocab_size * shape.d_model
    final_norm = shape.d_model
    total = tables + shape.n_layers * full_block + final_norm
    return ParameterBreakdown(
        attention, mlp, norms, matrix_block, full_block, full_block, tables, final_norm, total
    )


def scalar_rms(x: np.ndarray) -> float:
    """Return float64 RMS for a scalar diagnostic."""

    z = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(z * z)))


def unit_rms(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Normalize to unit RMS; `floor` only guards a degenerate zero vector."""

    if floor <= 0 or not math.isfinite(floor):
        raise ValueError("floor must be finite and positive")
    return np.asarray(x, dtype=np.float64) / max(scalar_rms(x), floor)


def make_depth_problem(seed: int, d_model: int, n_layers: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate one input and one weight stack shared by all modes."""

    if d_model <= 0 or n_layers <= 0:
        raise ValueError("d_model and n_layers must be positive")
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(d_model)
    weights = [
        rng.standard_normal((d_model, d_model)) / math.sqrt(d_model)
        for _ in range(n_layers)
    ]
    return x0, weights


def simulate_depth(mode: DepthMode, x0: np.ndarray, weights: list[np.ndarray]) -> list[float]:
    """Run none/pre/post placement on the same linear residual stack."""

    if mode not in ("none", "pre", "post"):
        raise ValueError("unknown mode")
    h = np.asarray(x0, dtype=np.float64).copy()
    if h.ndim != 1 or not weights:
        raise ValueError("x0 must be a vector and weights must be non-empty")
    curve = []
    for weight in weights:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (h.size, h.size):
            raise ValueError("every weight must have shape [d_model, d_model]")
        if mode == "none":
            h = h + weight @ h
        elif mode == "pre":
            h = h + weight @ unit_rms(h)
        else:
            h = unit_rms(h + weight @ h)
        curve.append(scalar_rms(h))
    return curve


def make_block_stack(
    seed: int,
    n_layers: int,
    config: BlockConfig,
) -> list[PreNormBlock]:
    """Create one deterministic stack reused by pre and Peri-LN traces."""

    rng = np.random.default_rng(seed)
    return [PreNormBlock(config, rng) for _ in range(n_layers)]


def trace_block_stack(
    blocks: list[PreNormBlock],
    x0: np.ndarray,
    allowed: np.ndarray,
    placement: NormPlacement,
) -> dict[str, list[float]]:
    """Record residual, attention-update, and FFN-update RMS by layer."""

    x = np.asarray(x0, dtype=np.float64).copy()
    trace = {"residual": [], "attention_update": [], "ffn_update": []}
    for block in blocks:
        x, details = block.forward(x, allowed, placement=placement)
        trace["residual"].append(scalar_rms(x))
        trace["attention_update"].append(scalar_rms(details["attention_update"]))
        trace["ffn_update"].append(scalar_rms(details["ffn_update"]))
    return trace


def dynamic_tanh(
    x: np.ndarray,
    alpha: float,
    gamma: np.ndarray | float = 1.0,
    beta: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Element-wise DyT-style transform `gamma*tanh(alpha*x) + beta`."""

    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    z = np.asarray(x, dtype=np.float64)
    return np.asarray(gamma) * np.tanh(alpha * z) + np.asarray(beta)


def dyisru(x: np.ndarray, b: float, gamma: np.ndarray | float = 1.0, beta: np.ndarray | float = 0.0) -> np.ndarray:
    """DyISRU: gamma*x/sqrt(b+x**2)+beta, with b>0; sqrt(D) may be absorbed in gamma."""

    if not math.isfinite(b) or b <= 0:
        raise ValueError("b must be finite and positive")
    z = np.asarray(x, dtype=np.float64)
    return np.asarray(gamma) * z / np.hypot(math.sqrt(b), z) + np.asarray(beta)


def sinkhorn_mix(logits: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Approximate a doubly stochastic matrix; callers must check both residuals."""
    log_a = np.asarray(logits, dtype=np.float64).copy()
    if log_a.ndim != 2 or log_a.shape[0] != log_a.shape[1] or not log_a.size or not np.isfinite(log_a).all():
        raise ValueError("logits must be a finite, nonempty square matrix")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    # Alternate row/column normalization in log space to avoid exp overflow.
    for _ in range(iterations):
        log_a -= np.logaddexp.reduce(log_a, axis=1, keepdims=True)
        log_a -= np.logaddexp.reduce(log_a, axis=0, keepdims=True)
    return np.exp(log_a)


def swiglu_mlp(x, w_gate, w_up, w_down):
    """Row-vector SwiGLU: two projections, product, then output projection."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def swiglu_oss_mlp(x, w_gate, w_up, w_down, alpha=1.702, limit=7.0):
    """Apply the gpt-oss activation before the down projection."""
    return gpt_oss_swiglu(x @ w_gate, x @ w_up, alpha=alpha, limit=limit) @ w_down


class PreNormStep:
    """Inspect two sequential residual updates; attention accepts a mask."""
    def __init__(self, norm1, norm2, attention, ffn):
        self.norm1, self.norm2 = norm1, norm2
        self.attention, self.ffn = attention, ffn

    def __call__(self, x, mask):
        h = x + self.attention(self.norm1(x), mask)
        return h + self.ffn(self.norm2(h))
