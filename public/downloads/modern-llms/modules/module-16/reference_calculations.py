#!/usr/bin/env python3
"""Reference calculations and validation helpers for Module 16.

The module owns a frozen, lecture-independent numerical contract for the learner
notebooks.  Functions are intentionally small and explicit: they expose tensor shapes,
spatial divisibility, binary memory units, and failure modes rather than hiding them
behind a model framework.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


def _positive_int(name: str, value: int) -> int:
    """Validate and return one strictly positive integer."""

    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(name: str, value: float) -> float:
    """Validate and return one strictly positive finite floating-point value."""

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class PatchLayout:
    """Spatial patch layout for one image before or after token merging."""

    image_height: int
    image_width: int
    patch_size: int
    merge_factor: int
    grid_height: int
    grid_width: int
    token_count: int


def patch_layout(
    image_height: int,
    image_width: int,
    *,
    patch_size: int = 14,
    merge_factor: int = 1,
) -> PatchLayout:
    """Return the exact patch grid and token count for a rectangular image.

    ``merge_factor=m`` groups each spatial ``m × m`` block into one visual token.
    Both the image and the resulting patch grid must be exactly divisible; silent
    truncation would under-count tokens and is therefore rejected.
    """

    image_height = _positive_int("image_height", image_height)
    image_width = _positive_int("image_width", image_width)
    patch_size = _positive_int("patch_size", patch_size)
    merge_factor = _positive_int("merge_factor", merge_factor)
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    if grid_height % merge_factor or grid_width % merge_factor:
        raise ValueError("the patch grid must be divisible by merge_factor on both axes")
    merged_h = grid_height // merge_factor
    merged_w = grid_width // merge_factor
    return PatchLayout(
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
        merge_factor=merge_factor,
        grid_height=grid_height,
        grid_width=grid_width,
        token_count=merged_h * merged_w,
    )


def patchify_rgb(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert one ``[H, W, C]`` image into flattened row-major patch vectors.

    Returns ``patches`` with shape ``[N_patches, patch_size² × C]`` and the unmerged
    spatial grid ``(grid_height, grid_width)``.  The function accepts any numeric dtype
    but rejects non-finite floating-point values.
    """

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("image must have shape [H, W, C]")
    height, width, channels = array.shape
    if channels <= 0:
        raise ValueError("image must contain at least one channel")
    layout = patch_layout(height, width, patch_size=patch_size)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image dtype must be numeric")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("image contains NaN or infinity")
    patches = (
        array.reshape(layout.grid_height, patch_size, layout.grid_width, patch_size, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(layout.token_count, patch_size * patch_size * channels)
    )
    return patches, (layout.grid_height, layout.grid_width)


def merge_spatial_tokens(
    tokens: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    merge_factor: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Group each spatial ``m × m`` block and concatenate its feature vectors.

    ``tokens`` may be ``[N, D]`` or batched ``[B, N, D]``.  The output keeps batch
    order and has shape ``[N/m², D·m²]`` or ``[B, N/m², D·m²]``.  This is the
    transparent pixel-unshuffle part of a merger; a learned projection can follow it.
    """

    array = np.asarray(tokens)
    if array.ndim not in {2, 3}:
        raise ValueError("tokens must have shape [N, D] or [B, N, D]")
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two integers")
    grid_height, grid_width = grid_shape
    grid_height = _positive_int("grid_height", grid_height)
    grid_width = _positive_int("grid_width", grid_width)
    merge_factor = _positive_int("merge_factor", merge_factor)
    n_tokens = array.shape[-2]
    feature_dim = array.shape[-1]
    if feature_dim == 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("tokens must have nonempty numeric features")
    if n_tokens != grid_height * grid_width:
        raise ValueError("token count does not match grid_shape")
    if grid_height % merge_factor or grid_width % merge_factor:
        raise ValueError("grid_shape must be divisible by merge_factor")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("tokens contain NaN or infinity")

    batched = array.ndim == 3
    work = array if batched else array[None, ...]
    batch = work.shape[0]
    merged_h = grid_height // merge_factor
    merged_w = grid_width // merge_factor
    merged = (
        work.reshape(batch, merged_h, merge_factor, merged_w, merge_factor, feature_dim)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(batch, merged_h * merged_w, feature_dim * merge_factor * merge_factor)
    )
    return (merged if batched else merged[0]), (merged_h, merged_w)


def count_projector_parameters(
    vision_dim: int,
    llm_dim: int,
    *,
    hidden_dim: int | None = None,
    bias: bool = True,
) -> int:
    """Count parameters of a two-layer MLP vision projector exactly."""

    vision_dim = _positive_int("vision_dim", vision_dim)
    llm_dim = _positive_int("llm_dim", llm_dim)
    hidden_dim = llm_dim if hidden_dim is None else _positive_int("hidden_dim", hidden_dim)
    total = vision_dim * hidden_dim + hidden_dim * llm_dim
    if bias:
        total += hidden_dim + llm_dim
    return total


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Evaluate exact GELU in float64 without requiring SciPy."""

    array = np.asarray(x, dtype=np.float64)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * array * (1.0 + erf(array / math.sqrt(2.0)))


@dataclass(frozen=True)
class ProjectorWeights:
    """Weights of a two-layer MLP mapping vision features into the LLM width."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def validate(self) -> tuple[int, int, int]:
        """Validate matrix compatibility and return ``(vision, hidden, llm)`` widths."""

        w1 = np.asarray(self.w1)
        b1 = np.asarray(self.b1)
        w2 = np.asarray(self.w2)
        b2 = np.asarray(self.b2)
        if w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
            raise ValueError("projector weights must be rank-2 matrices and rank-1 biases")
        vision_dim, hidden_dim = w1.shape
        if b1.shape != (hidden_dim,) or w2.shape[0] != hidden_dim:
            raise ValueError("w1/b1/w2 hidden dimensions are incompatible")
        llm_dim = w2.shape[1]
        if b2.shape != (llm_dim,):
            raise ValueError("w2 and b2 output dimensions are incompatible")
        for name, value in {"w1": w1, "b1": b1, "w2": w2, "b2": b2}.items():
            if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and numeric")
        return vision_dim, hidden_dim, llm_dim


def init_toy_projector(
    vision_dim: int,
    hidden_dim: int,
    llm_dim: int,
    *,
    seed: int = 14,
) -> ProjectorWeights:
    """Create deterministic small weights for a CPU-safe teaching forward pass."""

    vision_dim = _positive_int("vision_dim", vision_dim)
    hidden_dim = _positive_int("hidden_dim", hidden_dim)
    llm_dim = _positive_int("llm_dim", llm_dim)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 1.0 / math.sqrt(vision_dim), size=(vision_dim, hidden_dim))
    w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, llm_dim))
    return ProjectorWeights(
        w1=w1.astype(np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=w2.astype(np.float32),
        b2=np.zeros(llm_dim, dtype=np.float32),
    )


def project_vision(features: np.ndarray, weights: ProjectorWeights) -> np.ndarray:
    """Project ``[..., N_visual, D_vision]`` features into ``D_llm`` with GELU."""

    array = np.asarray(features)
    if array.ndim < 2:
        raise ValueError("features must have shape [..., N_visual, D_vision]")
    vision_dim, _, _ = weights.validate()
    if array.shape[-1] != vision_dim:
        raise ValueError("features last dimension does not match projector vision_dim")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("features must be finite and numeric")
    work = array.astype(np.float32, copy=False)
    hidden = gelu_exact(work @ weights.w1 + weights.b1).astype(np.float32)
    output = hidden @ weights.w2 + weights.b2
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("projector produced NaN or infinity")
    return output.astype(np.float32, copy=False)


@dataclass(frozen=True)
class MultimodalSequence:
    """Decoder input after replacing one image placeholder by continuous features."""

    embeddings: np.ndarray
    token_labels: np.ndarray
    modality_ids: np.ndarray
    position_ids: np.ndarray
    attention_mask: np.ndarray


def assemble_multimodal_sequence(
    token_ids: np.ndarray,
    token_embeddings: np.ndarray,
    *,
    image_placeholder_id: int,
    image_embeddings: np.ndarray,
    image_start_embedding: np.ndarray,
    image_end_embedding: np.ndarray,
    ignore_index: int = -100,
) -> MultimodalSequence:
    """Replace exactly one placeholder token by start/vision/end embeddings.

    Text positions retain their token IDs as labels.  Continuous visual tokens and
    image-boundary embeddings receive ``ignore_index``.  The returned causal mask is
    ``[sequence, sequence]`` and positions are contiguous across modalities.
    """

    token_ids = np.asarray(token_ids)
    token_embeddings = np.asarray(token_embeddings)
    image_embeddings = np.asarray(image_embeddings)
    image_start_embedding = np.asarray(image_start_embedding)
    image_end_embedding = np.asarray(image_end_embedding)
    if token_ids.ndim != 1 or not np.issubdtype(token_ids.dtype, np.integer):
        raise ValueError("token_ids must be a rank-1 integer array")
    if token_embeddings.ndim != 2 or token_embeddings.shape[0] != token_ids.size:
        raise ValueError("token_embeddings must have shape [len(token_ids), D]")
    if image_embeddings.ndim != 2 or image_embeddings.shape[0] == 0:
        raise ValueError("image_embeddings must have shape [N_visual>0, D]")
    width = token_embeddings.shape[1]
    if image_embeddings.shape[1] != width:
        raise ValueError("text and image embeddings must share the LLM width")
    if image_start_embedding.shape != (width,) or image_end_embedding.shape != (width,):
        raise ValueError("image boundary embeddings must have shape [D]")
    for name, value in {
        "token_embeddings": token_embeddings,
        "image_embeddings": image_embeddings,
        "image_start_embedding": image_start_embedding,
        "image_end_embedding": image_end_embedding,
    }.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or infinity")
    if (isinstance(image_placeholder_id, bool)
            or not isinstance(image_placeholder_id, (int, np.integer))):
        raise ValueError("image_placeholder_id must be an integer")
    if np.any(token_ids < 0) or width == 0:
        raise ValueError("token IDs must be nonnegative and embedding width positive")
    placeholder_positions = np.flatnonzero(token_ids == image_placeholder_id)
    if placeholder_positions.size != 1:
        raise ValueError("token_ids must contain exactly one image placeholder")
    placeholder = int(placeholder_positions[0])

    before_embeddings = token_embeddings[:placeholder]
    after_embeddings = token_embeddings[placeholder + 1 :]
    visual_block = np.concatenate(
        [image_start_embedding[None, :], image_embeddings, image_end_embedding[None, :]],
        axis=0,
    )
    embeddings = np.concatenate([before_embeddings, visual_block, after_embeddings], axis=0)

    labels = np.concatenate(
        [
            token_ids[:placeholder],
            np.full(visual_block.shape[0], ignore_index, dtype=np.int64),
            token_ids[placeholder + 1 :],
        ]
    ).astype(np.int64)
    modality_ids = np.concatenate(
        [
            np.zeros(before_embeddings.shape[0], dtype=np.int8),
            np.full(visual_block.shape[0], 1, dtype=np.int8),
            np.zeros(after_embeddings.shape[0], dtype=np.int8),
        ]
    )
    sequence = embeddings.shape[0]
    position_ids = np.arange(sequence, dtype=np.int64)
    attention_mask = np.tril(np.ones((sequence, sequence), dtype=bool))
    return MultimodalSequence(
        embeddings=embeddings.astype(np.float32, copy=False),
        token_labels=labels,
        modality_ids=modality_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Stable log-sum-exp for finite arrays."""

    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def l2_normalize_rows(embeddings: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a finite ``[batch, width]`` matrix and reject zero rows."""

    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] == 0:
        raise ValueError("embeddings must have shape [batch>=2, width>0]")
    if not np.all(np.isfinite(array)):
        raise ValueError("embeddings contain NaN or infinity")
    eps = _positive_finite("eps", eps)
    scale = np.max(np.abs(array), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("embedding rows must have non-zero norm")
    scaled = array / scale
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    if np.any(scale <= eps / norms):
        raise ValueError("embedding row norm must exceed eps")
    return scaled / norms


def symmetric_infonce_from_similarity(similarity: np.ndarray, temperature: float) -> dict[str, float]:
    """Return stable image→text, text→image, and symmetric InfoNCE losses.

    ``similarity`` must be a finite square matrix whose diagonal contains the aligned
    image-text pairs.  Log-sum-exp arithmetic avoids overflow at small temperatures.
    """

    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1] or similarity.shape[0] < 2:
        raise ValueError("similarity must be a square [batch>=2, batch] matrix")
    if not np.all(np.isfinite(similarity)):
        raise ValueError("similarity contains NaN or infinity")
    temperature = _positive_finite("temperature", temperature)
    # Center BEFORE dividing; equal huge finite scores need not overflow.
    def directional(values):
        with np.errstate(over='ignore', invalid='ignore'):
            logits = (values - np.max(values, axis=1, keepdims=True)) / temperature
            loss = _logsumexp(logits, axis=1) - np.diag(logits)
        if not np.all(np.isfinite(loss)):
            raise FloatingPointError("InfoNCE exceeds the float64 range")
        return float(np.mean(loss))
    image_to_text = directional(similarity)
    text_to_image = directional(similarity.T)
    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "symmetric": 0.5 * (image_to_text + text_to_image),
    }


@dataclass(frozen=True)
class KVCacheSpec:
    """K+V geometry for one decoder token across all layers."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_element: int

    def bytes_per_token(self) -> int:
        """Return exact K+V bytes per token across all layers."""

        n_layers = _positive_int("n_layers", self.n_layers)
        n_kv_heads = _positive_int("n_kv_heads", self.n_kv_heads)
        head_dim = _positive_int("head_dim", self.head_dim)
        bytes_per_element = _positive_int("bytes_per_element", self.bytes_per_element)
        return 2 * n_layers * n_kv_heads * head_dim * bytes_per_element


def image_kv_metrics(
    visual_tokens: int,
    spec: KVCacheSpec,
    *,
    device_budget_gib: float,
    weights_gib: float,
    runtime_reserve_gib: float = 0.0,
) -> dict[str, float | int]:
    """Compute one image's KV memory and a KV-only capacity upper bound.

    The capacity ignores activations, allocator fragmentation, temporary buffers, and
    scheduling overhead.  It is therefore explicitly an upper bound, not a deployment
    promise.
    """

    visual_tokens = _positive_int("visual_tokens", visual_tokens)
    device_budget_gib = _positive_finite("device_budget_gib", device_budget_gib)
    weights_gib = _positive_finite("weights_gib", weights_gib)
    runtime_reserve_gib = float(runtime_reserve_gib)
    if not math.isfinite(runtime_reserve_gib) or runtime_reserve_gib < 0.0:
        raise ValueError("runtime_reserve_gib must be finite and non-negative")
    available_gib = device_budget_gib - weights_gib - runtime_reserve_gib
    if available_gib <= 0.0:
        raise ValueError("weights plus reserve leave no positive KV budget")
    kv_bytes = visual_tokens * spec.bytes_per_token()
    kv_mib = kv_bytes / 2**20
    capacity_upper_bound = int((available_gib * 2**30) // kv_bytes)
    return {
        "visual_tokens": visual_tokens,
        "kv_bytes": kv_bytes,
        "kv_mib": kv_mib,
        "available_gib": available_gib,
        "capacity_upper_bound": capacity_upper_bound,
    }


def estimate_dense_training_hours(
    n_parameters: float,
    training_tokens: float,
    *,
    peak_flops_per_second: float,
    mfu: float,
    flops_multiplier: float = 6.0,
) -> float:
    """Return a transparent dense-equivalent training scenario in hours."""

    n_parameters = _positive_finite("n_parameters", n_parameters)
    training_tokens = _positive_finite("training_tokens", training_tokens)
    peak_flops_per_second = _positive_finite("peak_flops_per_second", peak_flops_per_second)
    mfu = float(mfu)
    if not math.isfinite(mfu) or not 0.0 < mfu <= 1.0:
        raise ValueError("mfu must lie in (0, 1]")
    flops_multiplier = _positive_finite("flops_multiplier", flops_multiplier)
    return flops_multiplier * n_parameters * training_tokens / (peak_flops_per_second * mfu) / 3600.0


def uniform_sample_indices(total_frames: int, max_frames: int) -> np.ndarray:
    """Choose deterministic, approximately uniform frame indices including endpoints."""

    total_frames = _positive_int("total_frames", total_frames)
    max_frames = _positive_int("max_frames", max_frames)
    if max_frames >= total_frames:
        return np.arange(total_frames, dtype=np.int64)
    if max_frames == 1:
        return np.array([0], dtype=np.int64)
    indices = np.rint(np.linspace(0, total_frames - 1, max_frames)).astype(np.int64)
    if np.unique(indices).size != max_frames:
        raise AssertionError("uniform sampler produced duplicate indices")
    if indices[0] != 0 or indices[-1] != total_frames - 1:
        raise AssertionError("uniform sampler must include first and last frame")
    return indices


def video_budget(
    *,
    duration_seconds: float,
    source_fps: float,
    context_tokens: int,
    text_reserve_tokens: int,
    visual_tokens_per_frame: int,
    kv_spec: KVCacheSpec,
) -> dict[str, float | int]:
    """Compute frame capacity, sampling interval, and KV for one video prompt."""

    duration_seconds = _positive_finite("duration_seconds", duration_seconds)
    source_fps = _positive_finite("source_fps", source_fps)
    context_tokens = _positive_int("context_tokens", context_tokens)
    if not isinstance(text_reserve_tokens, (int, np.integer)) or isinstance(text_reserve_tokens, bool):
        raise ValueError("text_reserve_tokens must be an integer")
    text_reserve_tokens = int(text_reserve_tokens)
    if text_reserve_tokens < 0 or text_reserve_tokens >= context_tokens:
        raise ValueError("text_reserve_tokens must lie in [0, context_tokens)")
    visual_tokens_per_frame = _positive_int("visual_tokens_per_frame", visual_tokens_per_frame)
    total_frames = int(math.floor(duration_seconds * source_fps + 1e-12))
    if total_frames < 1:
        raise ValueError("the floor-based source-frame convention yields zero frames")
    visual_budget = context_tokens - text_reserve_tokens
    max_frames = visual_budget // visual_tokens_per_frame
    if max_frames <= 0:
        raise ValueError("the context budget cannot fit one frame")
    selected_frames = min(total_frames, max_frames)
    sample_interval_seconds = duration_seconds / selected_frames
    used_visual_tokens = selected_frames * visual_tokens_per_frame
    full_prompt_tokens = text_reserve_tokens + used_visual_tokens
    kv_bytes = full_prompt_tokens * kv_spec.bytes_per_token()
    return {
        "total_frames": total_frames,
        "visual_budget_tokens": visual_budget,
        "max_frames": max_frames,
        "selected_frames": selected_frames,
        "sample_interval_seconds": sample_interval_seconds,
        "used_visual_tokens": used_visual_tokens,
        "visual_kv_gib": used_visual_tokens * kv_spec.bytes_per_token() / 2**30,
        "full_prompt_tokens": full_prompt_tokens,
        "full_prompt_kv_gib": kv_bytes / 2**30,
    }


def build_reference_payload() -> dict[str, Any]:
    """Build the frozen numerical contract used by notebooks and visualizations."""

    patches = {
        "llava336": patch_layout(336, 336).token_count,
        "next672": 5 * patch_layout(336, 336).token_count,
        "qwen1008": patch_layout(1008, 1008).token_count,
        "qwen1008_merged": patch_layout(1008, 1008, merge_factor=2).token_count,
        "rectangular_56x28": patch_layout(56, 28).token_count,
        "rectangular_56x28_merged": patch_layout(56, 28, merge_factor=2).token_count,
    }

    kv_spec = KVCacheSpec(n_layers=32, n_kv_heads=8, head_dim=128, bytes_per_element=2)
    kv_metrics = {
        name: image_kv_metrics(
            tokens,
            kv_spec,
            device_budget_gib=80.0,
            weights_gib=15.0,
        )
        for name, tokens in {
            "llava336": patches["llava336"],
            "next672": patches["next672"],
            "qwen1008_merged": patches["qwen1008_merged"],
        }.items()
    }

    n = 4
    similarity = np.full((n, n), 0.2, dtype=np.float64)
    np.fill_diagonal(similarity, 0.8)
    infonce_tau01 = symmetric_infonce_from_similarity(similarity, 0.1)
    infonce_tau05 = symmetric_infonce_from_similarity(similarity, 0.5)
    p_tau01 = 1.0 / (1.0 + (n - 1) * math.exp((0.2 - 0.8) / 0.1))

    projector_params = count_projector_parameters(1024, 4096, hidden_dim=4096)
    stage1_tokens = 558_000 * 600
    stage2_tokens = 665_000 * 1_000
    cost_args = {
        "n_parameters": 7e9,
        "peak_flops_per_second": 990e12,
        "mfu": 0.40,
        "flops_multiplier": 6.0,
    }
    stage1_hours = estimate_dense_training_hours(training_tokens=stage1_tokens, **cost_args)
    stage2_hours = estimate_dense_training_hours(training_tokens=stage2_tokens, **cost_args)

    video_full = video_budget(
        duration_seconds=2 * 3600,
        source_fps=1.0,
        context_tokens=256_000,
        text_reserve_tokens=4_000,
        visual_tokens_per_frame=patches["llava336"],
        kv_spec=kv_spec,
    )
    video_merged = video_budget(
        duration_seconds=2 * 3600,
        source_fps=1.0,
        context_tokens=256_000,
        text_reserve_tokens=4_000,
        visual_tokens_per_frame=patches["llava336"] // 4,
        kv_spec=kv_spec,
    )

    return {
        "schema_version": 1,
        "patches": patches,
        "projector": {
            "vision_dim": 1024,
            "hidden_dim": 4096,
            "llm_dim": 4096,
            "parameter_count": projector_params,
        },
        "kv": {
            "spec": {
                "n_layers": kv_spec.n_layers,
                "n_kv_heads": kv_spec.n_kv_heads,
                "head_dim": kv_spec.head_dim,
                "bytes_per_element": kv_spec.bytes_per_element,
                "bytes_per_token": kv_spec.bytes_per_token(),
                "kib_per_token": kv_spec.bytes_per_token() / 2**10,
            },
            "device_budget_gib": 80.0,
            "weights_gib": 15.0,
            "per_image": kv_metrics,
        },
        "infonce": {
            "batch_size": n,
            "positive_similarity": 0.8,
            "negative_similarity": 0.2,
            "tau_0_1_probability": p_tau01,
            "tau_0_1": infonce_tau01,
            "tau_0_5": infonce_tau05,
            "uniform_floor": math.log(n),
        },
        "training_floor": {
            "stage1_tokens": stage1_tokens,
            "stage2_tokens": stage2_tokens,
            "stage1_hours": stage1_hours,
            "stage2_hours": stage2_hours,
            "total_hours": stage1_hours + stage2_hours,
            **cost_args,
        },
        "video": {
            "source_frames": 7200,
            "without_merger": video_full,
            "with_merger_2x2": video_merged,
        },
    }


def write_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Write the canonical payload with deterministic formatting and return it."""

    payload = build_reference_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite the frozen reference JSON")
    args = parser.parse_args()
    payload = write_reference() if args.write else build_reference_payload()
    print(REFERENCE_PATH if args.write else "read-only: pass --write to update reference JSON")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
