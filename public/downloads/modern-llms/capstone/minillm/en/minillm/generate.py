"""MiniLLM inference: sampling, speculative decoding, and quantization.

Implemented mechanisms and their course references:

  prefill / decode as distinct phases .............. M10 §5
  prefix reuse ...................................... M10 §6
  distribution-preserving speculative decoding ..... M11 §3–4
  MXFP4 / NVFP4: group + scale ...................... M11 §11–13
  int8 per-channel ................................. M11 §12

An honest quantization boundary: this module implements **fake quant**. Weights
are quantized and immediately dequantized back to the working dtype. This
measures **quality**—how much the format damages the model—but says nothing
about speed. Real speedups require a kernel that reads packed 4-bit values from
memory (M11 §14). Accordingly, `bits_per_weight()` computes hypothetical storage
rather than the size of the tensor used here.

Shape contract: logits [B, T, V]; decode uses only the final position. Every
generator uses batch size 1. Batching is a separate module 10 topic; adding it
here would hide the mechanism behind indexing arithmetic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .model import KVCache, MiniLLM
from .precision import (autocast_context, dtype_name, model_parameter_dtype,
                        resolve_compute_dtype)


# =============================================================================
# Candidate filtering (M15 §3—sampling controls)
# =============================================================================
@dataclass
class SamplingConfig:
    temperature: float = 1.0     # 0.0 → greedy selection
    top_k: int = 0               # 0 disables the filter
    top_p: float = 1.0           # 1.0 disables the filter
    min_p: float = 0.0           # 0.0 disables the filter
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature должна быть неотрицательной")
        if self.top_k < 0:
            raise ValueError("top_k должен быть неотрицательным")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p должен лежать в интервале (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p должен лежать в интервале [0, 1]")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def logits_to_probs(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """logits [V] -> a [V] distribution after every filter.

    Order matters: temperature changes the shape, while top-k/top-p/min-p trim
    the tail of that changed distribution. Reversing the order would produce a
    different behavioral model under the same config.
    """
    if cfg.greedy:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs

    logits = logits.float() / cfg.temperature
    probs = torch.softmax(logits, dim=-1)

    if cfg.top_k and cfg.top_k < probs.numel():
        threshold = probs.topk(cfg.top_k).values[-1]
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

    if cfg.top_p < 1.0:
        ordered, index = probs.sort(descending=True)
        cumulative = ordered.cumsum(-1)
        # Keep the shortest prefix whose mass covers top_p. The token that
        # crosses the threshold is included, or top_p → 0 would yield an empty set.
        keep = cumulative - ordered < cfg.top_p
        keep[0] = True
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[index] = keep
        probs = torch.where(mask, probs, torch.zeros_like(probs))

    if cfg.min_p > 0.0:
        probs = torch.where(probs >= cfg.min_p * probs.max(), probs,
                            torch.zeros_like(probs))

    total = probs.sum()
    return probs / total if total > 0 else torch.full_like(probs, 1.0 / probs.numel())


def sample_from(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def _generator_for(device: torch.device, seed: int | None) -> torch.Generator | None:
    """The generator must live on the same device as the distribution."""
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _synchronize(device: torch.device) -> None:
    """CUDA is asynchronous; without a barrier wall time measures launch, not work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================
# Standard generation (M10 §5)
# =============================================================================
@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    kv_bytes: int = 0
    accepted: int = 0            # speculative decoding only
    proposed: int = 0
    target_forwards: int = 0     # all target-model forwards
    draft_forwards: int = 0      # all draft-model forwards
    target_prefill_forwards: int = 0
    draft_prefill_forwards: int = 0
    target_decode_forwards: int = 0
    draft_decode_forwards: int = 0
    parameter_dtype: str = "unknown"
    compute_dtype: str = "unknown"
    cache_dtype: str = "empty"
    cache_tokens: int = 0

    @property
    def decode_tokens_per_s(self) -> float:
        return self.generated_tokens / max(self.decode_seconds, 1e-9)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else float("nan")

    @property
    def tokens_per_target_forward(self) -> float:
        """Primary speculative metric (M11 §5): tokens per target-model forward.

        Prefill is accounted for separately in the statistics.
        """
        if not self.target_decode_forwards:
            return float("nan")
        return self.generated_tokens / self.target_decode_forwards

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        out["decode_tokens_per_s"] = round(self.decode_tokens_per_s, 2)
        if self.proposed:
            out["acceptance_rate"] = round(self.acceptance_rate, 4)
            out["tokens_per_target_forward"] = round(self.tokens_per_target_forward, 3)
        return out


@torch.no_grad()
def generate(model: MiniLLM, prompt: list[int], max_new_tokens: int,
             sampling: SamplingConfig | None = None,
             cache: KVCache | None = None,
             stop_tokens: set[int] | None = None,
             compute_dtype: str = "auto",
             finalize_cache: bool = False
             ) -> tuple[list[int], GenerationStats]:
    """Autoregressive generation with a precise cache and timing contract.

    Prefill processes the uncached prompt. Decode performs only the forwards
    required to obtain requested tokens. By default the returned cache trails the
    last generated token by one position; ``finalize_cache=True`` adds a separately
    timed forward so callers can continue from a fully materialized cache.

    A supplied prefix cache is trusted to correspond exactly to the prefix tokens;
    this function cannot reconstruct or verify that equality from K/V tensors alone.
    """
    sampling = sampling or SamplingConfig()
    if not prompt:
        raise ValueError("prompt не должен быть пустым")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens должен быть неотрицательным")
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    actual_compute = resolve_compute_dtype(device, compute_dtype)
    stats = GenerationStats(prompt_tokens=len(prompt),
                            parameter_dtype=model_parameter_dtype(model),
                            compute_dtype=actual_compute)
    generator = _generator_for(device, sampling.seed)
    cache = cache if cache is not None else model.new_cache()
    tokens = list(prompt)

    pending = tokens[len(cache):]
    if not pending:
        raise ValueError("кэш уже покрывает весь промпт — передайте хотя бы один новый токен")
    _synchronize(device)
    start = time.perf_counter()
    with autocast_context(device, actual_compute):
        logits, _ = model(torch.tensor([pending], device=device), cache=cache)
    stats.target_forwards += 1
    stats.target_prefill_forwards += 1
    _synchronize(device)
    stats.prefill_seconds = time.perf_counter() - start

    _synchronize(device)
    start = time.perf_counter()
    last_generated: int | None = None
    for index in range(max_new_tokens):
        probs = logits_to_probs(logits[0, -1], sampling)
        next_id = sample_from(probs, generator)
        tokens.append(next_id)
        last_generated = next_id
        stats.generated_tokens += 1
        should_stop = bool(stop_tokens and next_id in stop_tokens)
        if should_stop or index + 1 == max_new_tokens:
            break
        with autocast_context(device, actual_compute):
            logits, _ = model(torch.tensor([[next_id]], device=device), cache=cache)
        stats.target_forwards += 1
        stats.target_decode_forwards += 1
    _synchronize(device)
    stats.decode_seconds = time.perf_counter() - start

    if finalize_cache and last_generated is not None and len(cache) < len(tokens):
        _synchronize(device)
        finalize_start = time.perf_counter()
        with autocast_context(device, actual_compute):
            model(torch.tensor([[last_generated]], device=device), cache=cache)
        stats.target_forwards += 1
        _synchronize(device)
        # Finalization is not part of time-to-token throughput.
        stats.__dict__["cache_finalize_seconds"] = time.perf_counter() - finalize_start

    stats.kv_bytes = cache.nbytes()
    stats.cache_tokens = len(cache)
    stats.cache_dtype = dtype_name(cache.dtype())
    if was_training:
        model.train()
    return tokens, stats


# =============================================================================
# Speculative decoding (M11 §3–4)
# =============================================================================
@torch.no_grad()
def speculative_generate(target: MiniLLM, draft: MiniLLM, prompt: list[int],
                         max_new_tokens: int, lookahead: int = 4,
                         sampling: SamplingConfig | None = None,
                         compute_dtype: str = "auto"
                         ) -> tuple[list[int], GenerationStats]:
    """Speculative decoding with symmetric prefill/decode timing.

    Both models prefill the prompt before decode timing starts. A target cycle
    verifies all draft proposals in one batched forward. Draft and target caches
    may trail a corrected or bonus token; the next cycle catches up explicitly.
    """
    sampling = sampling or SamplingConfig()
    if not prompt:
        raise ValueError("prompt не должен быть пустым")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens должен быть неотрицательным")
    if lookahead < 1:
        raise ValueError("lookahead должен быть не меньше единицы")
    if sampling.greedy:
        raise ValueError("спекулятивный декод здесь реализован для стохастического семплирования")
    device = next(target.parameters()).device
    if next(draft.parameters()).device != device:
        raise ValueError("target и draft должны быть на одном устройстве")
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target и draft должны иметь один словарь")
    actual_compute = resolve_compute_dtype(device, compute_dtype)
    target_was_training, draft_was_training = target.training, draft.training
    target.eval(); draft.eval()
    generator = _generator_for(device, sampling.seed)
    tokens = list(prompt)
    stats = GenerationStats(prompt_tokens=len(prompt),
                            parameter_dtype=model_parameter_dtype(target),
                            compute_dtype=actual_compute)
    target_cache, draft_cache = target.new_cache(), draft.new_cache()
    bounded_windows = [w for w in (*target_cache.windows, *draft_cache.windows) if w is not None]
    if bounded_windows and lookahead > min(bounded_windows):
        raise ValueError("lookahead cannot exceed the smallest bounded cache window")

    def forward(model: MiniLLM, cache: KVCache, new_ids: list[int], *, phase: str) -> torch.Tensor:
        if not new_ids:
            raise ValueError("forward requires at least one token")
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unknown generation phase: {phase}")
        if model is target:
            stats.target_forwards += 1
            if phase == "prefill":
                stats.target_prefill_forwards += 1
            else:
                stats.target_decode_forwards += 1
        else:
            stats.draft_forwards += 1
            if phase == "prefill":
                stats.draft_prefill_forwards += 1
            else:
                stats.draft_decode_forwards += 1
        with autocast_context(device, actual_compute):
            logits, _ = model(torch.tensor([new_ids], device=device), cache=cache)
        return logits[0]

    # Symmetric prompt prefill for both models.
    _synchronize(device)
    prefill_start = time.perf_counter()
    target_next = forward(target, target_cache, tokens, phase="prefill")[-1]
    draft_next = forward(draft, draft_cache, tokens, phase="prefill")[-1]
    _synchronize(device)
    stats.prefill_seconds = time.perf_counter() - prefill_start

    _synchronize(device)
    decode_start = time.perf_counter()
    while stats.generated_tokens < max_new_tokens:
        base_len = len(tokens)
        # Catch up caches after a corrected or bonus token from the previous cycle.
        if len(draft_cache) < base_len:
            draft_next = forward(draft, draft_cache, tokens[len(draft_cache):], phase="decode")[-1]
        if len(target_cache) < base_len:
            target_next = forward(target, target_cache, tokens[len(target_cache):], phase="decode")[-1]

        drafted: list[int] = []
        q_dists: list[torch.Tensor] = []
        logits_cursor = draft_next
        for i in range(lookahead):
            q = logits_to_probs(logits_cursor, sampling)
            token = sample_from(q, generator)
            drafted.append(token)
            q_dists.append(q)
            if i + 1 < lookahead:
                logits_cursor = forward(draft, draft_cache, [token], phase="decode")[-1]

        # p(draft[0]) comes from the current target-next distribution. One batched
        # target forward over all proposals gives p(draft[1:]) and the bonus p.
        target_logits = forward(target, target_cache, drafted, phase="decode")
        p_dists = [logits_to_probs(target_next, sampling)]
        p_dists.extend(logits_to_probs(target_logits[i], sampling)
                       for i in range(lookahead))

        accepted, evaluated = 0, 0
        rejected = False
        for i, token in enumerate(drafted):
            if stats.generated_tokens >= max_new_tokens:
                break
            evaluated += 1
            p, q = p_dists[i][token], q_dists[i][token]
            threshold = torch.rand((), device=device, generator=generator).item()
            if q <= 0 or threshold < min(1.0, (p / q).item()):
                tokens.append(token)
                accepted += 1
                stats.generated_tokens += 1
            else:
                residual = (p_dists[i] - q_dists[i]).clamp(min=0)
                corrected = residual / residual.sum() if residual.sum() > 0 else p_dists[i]
                tokens.append(sample_from(corrected, generator))
                stats.generated_tokens += 1
                rejected = True
                break
        if not rejected and accepted == lookahead and stats.generated_tokens < max_new_tokens:
            tokens.append(sample_from(p_dists[lookahead], generator))
            stats.generated_tokens += 1

        stats.accepted += accepted
        stats.proposed += evaluated
        keep = base_len + accepted
        target_cache.truncate(keep)
        draft_cache.truncate(min(keep, len(draft_cache)))
        # target_next/draft_next are refreshed by catch-up at the next cycle.

    _synchronize(device)
    stats.decode_seconds = time.perf_counter() - decode_start
    stats.kv_bytes = target_cache.nbytes()
    stats.cache_tokens = len(target_cache)
    stats.cache_dtype = dtype_name(target_cache.dtype())
    if target_was_training: target.train()
    if draft_was_training: draft.train()
    return tokens, stats


# =============================================================================
# Weight quantization (M11 §11–13)
# =============================================================================
# E2M1 levels: sign + 2 exponent bits + 1 mantissa bit, eight magnitudes.
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def bits_per_weight(group: int, scale_bits: int, element_bits: int = 4) -> float:
    """Bits per weight for a “group + one scale” format.

    MXFP4: group 32, E8M0 scale (8 bits) → (32·4 + 8)/32 = 4.25.
    NVFP4: group 16, E4M3 scale (8 bits) → (16·4 + 8)/16 = 4.50.
    Module 11 gives these exact values; here they are derived, not quoted.
    """
    return (group * element_bits + scale_bits) / group


def quantize_fp4(w: torch.Tensor, group: int = 32, power_of_two_scale: bool = True
                 ) -> torch.Tensor:
    """Row-wise grouped FP4 fake quantization.

    Groups are formed along the last tensor dimension and never cross output rows
    or MoE expert boundaries. The returned tensor is dequantized to the working
    dtype; this is a quality probe, not a packed-kernel implementation.
    """
    shape = w.shape
    if w.ndim == 0:
        raise ValueError("FP4 quantization requires at least one dimension")
    width = shape[-1] if w.ndim > 1 else w.numel()
    rows = w.reshape(-1, width)
    pad = (-width) % group
    if pad:
        rows = torch.cat([rows, rows.new_zeros(rows.shape[0], pad)], dim=-1)
    blocks = rows.view(rows.shape[0], -1, group)

    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / E2M1_LEVELS[-1]
    if power_of_two_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))

    normalized = (blocks / scale).abs()
    levels = E2M1_LEVELS.to(w.device, w.dtype)
    index = (normalized[..., None] - levels).abs().argmin(dim=-1)
    dequantized = levels[index] * scale * blocks.sign()
    restored = dequantized.reshape(rows.shape[0], -1)[..., :width]
    return restored.reshape(shape)


def quantize_int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 per output channel (row of nn.Linear.weight)."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
    return torch.round(w / scale).clamp(-127, 127) * scale


QUANTIZERS = {
    "mxfp4": lambda w: quantize_fp4(w, group=32, power_of_two_scale=True),
    "nvfp4": lambda w: quantize_fp4(w, group=16, power_of_two_scale=False),
    "int8": quantize_int8_per_channel,
}

FORMAT_BITS = {
    "mxfp4": bits_per_weight(32, 8),
    "nvfp4": bits_per_weight(16, 8),
    "int8": 8.0,
}


def _packed_storage_bits(shape: torch.Size, method: str) -> int:
    """Conceptual packed payload for one tensor, including group padding/scales.

    This is still a teaching estimate: alignment, tensor headers and hardware-
    specific global metadata are outside the contract.
    """
    elements = math.prod(shape)
    width = shape[-1] if len(shape) > 1 else elements
    rows = elements // max(width, 1)
    if method == "int8":
        return elements * 8 + rows * 32       # fp32 scale per output row
    group = 32 if method == "mxfp4" else 16
    groups_per_row = math.ceil(width / group)
    local = rows * groups_per_row * (group * 4 + 8)
    if method == "nvfp4":
        local += 32                            # one conceptual global scale/tensor
    return local


@torch.no_grad()
def quantize_model(model: MiniLLM, method: str = "mxfp4",
                   skip_embeddings: bool = True) -> dict[str, float]:
    """Fake-quantize linear-layer and MoE-expert weights in place.

    Embeddings are skipped by default: the first and last layers are more
    sensitive, and production recipes almost always keep them at high precision
    (M11 §13). Return a summary of error and declared storage size.
    """
    if method not in QUANTIZERS:
        raise ValueError(f"неизвестный формат: {method!r}; есть {sorted(QUANTIZERS)}")
    quantizer = QUANTIZERS[method]

    quantized, errors, elements, packed_bits = 0, [], 0, 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if skip_embeddings and "lm_head" in name:
                continue
            w = module.weight.data
            q = quantizer(w)
            errors.append(((q - w).norm() / w.norm().clamp(min=1e-12)).item())
            elements += w.numel()
            packed_bits += _packed_storage_bits(w.shape, method)
            module.weight.data = q
            quantized += 1
    for name, param in model.named_parameters():
        if name.endswith(("gate_w", "up_w", "down_w")):       # MoE experts
            q = quantizer(param.data)
            errors.append(((q - param.data).norm() / param.data.norm()).item())
            elements += param.numel()
            packed_bits += _packed_storage_bits(param.shape, method)
            param.data = q
            quantized += 1

    theoretical_bits = FORMAT_BITS[method]
    effective_bits = packed_bits / max(elements, 1)
    return {
        "method": method,
        "tensors_quantized": quantized,
        "elements": elements,
        "bits_per_weight": theoretical_bits,
        "effective_bits_per_weight": round(effective_bits, 6),
        "stored_bytes": math.ceil(packed_bits / 8),
        "bf16_bytes": elements * 2,
        "fp32_bytes": elements * 4,
        "compression": round(16.0 / effective_bits, 3),
        "compression_vs_bf16": round(16.0 / effective_bits, 3),
        "compression_vs_runtime_fp32": round(32.0 / effective_bits, 3),
        "mean_relative_error": round(sum(errors) / len(errors), 5) if errors else 0.0,
        "scope": "quantized tensors only; row-wise fake-quant teaching model",
        "storage_note": (
            "includes row/group padding and conceptual local scales; excludes "
            "container headers, alignment and hardware-specific metadata; NVFP4 "
            "values/scales are not bit-exact E2M1/E4M3 encodings"
        ),
    }


# =============================================================================
# Demonstration
# =============================================================================
def demo(steps: int = 300) -> None:
    """Train profile S, then demonstrate generation and quantization."""
    import copy as _copy

    from .config import MiniLLMConfig
    from .data import ByteTokenizer, TokenStream, toy_corpus
    from .train import TrainConfig, train

    tok = ByteTokenizer()
    cfg = MiniLLMConfig.profile("S", vocab_size=tok.vocab_size,
                                max_position_embeddings=256)
    tcfg = TrainConfig(steps=steps, seq_len=128, micro_batch_size=8,
                       log_every=steps)
    stream = TokenStream.from_text(toy_corpus(), tok, tcfg.seq_len)
    model = MiniLLM(cfg)
    print(f"training MiniLLM-S ({model.num_parameters():,} parameters)…")
    train(model, stream, tcfg)

    prompt = tok.encode("коты спят")
    out, stats = generate(model, prompt, 60, SamplingConfig(temperature=0.7,
                                                           top_p=0.9, seed=0))
    print("\ngeneration:", repr(tok.decode(out)))
    print("statistics:", stats.as_dict())

    print("\nquantization (fake quant: quality, not speed):")
    for method in ("int8", "nvfp4", "mxfp4"):
        clone = _copy.deepcopy(model)
        report = quantize_model(clone, method)
        from .train import evaluate
        loss = evaluate(clone, stream, tcfg, batches=4)
        print(f"  {method:6} {report['bits_per_weight']:>5.2f} bits/weight · "
              f"compression ×{report['compression']:.2f} · "
              f"relative error {report['mean_relative_error']:.4f} · "
              f"loss {loss:.4f}")


if __name__ == "__main__":
    demo()
