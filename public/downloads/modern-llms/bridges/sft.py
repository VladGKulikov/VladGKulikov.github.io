"""B20/B21 — SFT labels and segment-isolated packing artifacts."""
from __future__ import annotations

import numpy as np
import torch


def numpy_assistant_labels(input_ids: np.ndarray, assistant_mask: np.ndarray, *, ignore_index: int = -100) -> np.ndarray:
    ids, mask = np.asarray(input_ids), np.asarray(assistant_mask)
    if ids.dtype != np.int64 or mask.dtype != np.bool_:
        raise TypeError("use int64 token IDs and a boolean assistant mask")
    if np.any(ids < 0):
        raise ValueError("input token IDs must be nonnegative")
    if ids.shape != mask.shape:
        raise ValueError("input_ids and assistant_mask must have identical shapes")
    return np.where(mask, ids, ignore_index).astype(np.int64)


def torch_assistant_labels(input_ids: torch.Tensor, assistant_mask: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    if input_ids.dtype != torch.long or assistant_mask.dtype != torch.bool:
        raise TypeError("use torch.long token IDs and a boolean assistant mask")
    if input_ids.device != assistant_mask.device:
        raise ValueError("input IDs and mask must share a device")
    if torch.any(input_ids < 0):
        raise ValueError("input token IDs must be nonnegative")
    if input_ids.shape != assistant_mask.shape:
        raise ValueError("input_ids and assistant_mask must have identical shapes")
    labels = input_ids.clone()
    return labels.masked_fill(~assistant_mask.to(torch.bool), ignore_index)


def numpy_segment_position_ids(segment_ids: np.ndarray, *, padding_id: int = -1) -> np.ndarray:
    """Reset positions at every contiguous segment boundary; padding receives position zero."""
    segments = np.asarray(segment_ids)
    if segments.ndim not in (1, 2):
        raise ValueError("segment_ids must be [T] or [B,T]")
    batch = segments[None, :] if segments.ndim == 1 else segments
    positions = np.zeros_like(batch, dtype=np.int64)
    for row in range(batch.shape[0]):
        for col in range(1, batch.shape[1]):
            current, previous = batch[row, col], batch[row, col - 1]
            if current != padding_id and current == previous:
                positions[row, col] = positions[row, col - 1] + 1
    return positions[0] if segments.ndim == 1 else positions


def torch_segment_position_ids(segment_ids: torch.Tensor, *, padding_id: int = -1) -> torch.Tensor:
    """Pure-tensor counterpart of :func:`numpy_segment_position_ids`."""
    if segment_ids.ndim not in (1, 2):
        raise ValueError("segment_ids must be [T] or [B,T]")
    batch = segment_ids.unsqueeze(0) if segment_ids.ndim == 1 else segment_ids
    positions = torch.zeros_like(batch, dtype=torch.long)
    for col in range(1, batch.shape[-1]):
        continues = (batch[:, col] != padding_id) & (batch[:, col] == batch[:, col - 1])
        positions[:, col] = torch.where(continues, positions[:, col - 1] + 1, 0)
    return positions[0] if segment_ids.ndim == 1 else positions


def numpy_block_causal_mask(segment_ids: np.ndarray, *, padding_id: int = -1) -> np.ndarray:
    """Bool allowed-mask for causal attention inside each contiguous non-padding segment."""
    segments = np.asarray(segment_ids)
    if segments.ndim not in (1, 2):
        raise ValueError("segment_ids must be [T] or [B,T]")
    batch = segments[None, :] if segments.ndim == 1 else segments
    t = batch.shape[-1]
    boundaries = np.ones_like(batch, dtype=bool)
    if t > 1:
        boundaries[:, 1:] = batch[:, 1:] != batch[:, :-1]
    run_ids = np.cumsum(boundaries, axis=-1)
    same = run_ids[:, :, None] == run_ids[:, None, :]
    valid = (batch[:, :, None] != padding_id) & (batch[:, None, :] != padding_id)
    causal = np.arange(t)[None, None, :] <= np.arange(t)[None, :, None]
    out = same & valid & causal
    return out[0] if segments.ndim == 1 else out


def torch_block_causal_mask(segment_ids: torch.Tensor, *, padding_id: int = -1) -> torch.Tensor:
    if segment_ids.ndim not in (1, 2):
        raise ValueError("segment_ids must be [T] or [B,T]")
    batch = segment_ids.unsqueeze(0) if segment_ids.ndim == 1 else segment_ids
    t = batch.shape[-1]
    boundaries = torch.ones_like(batch, dtype=torch.bool)
    if t > 1:
        boundaries[:, 1:] = batch[:, 1:] != batch[:, :-1]
    run_ids = boundaries.to(torch.long).cumsum(dim=-1)
    same = run_ids[:, :, None] == run_ids[:, None, :]
    valid = (batch[:, :, None] != padding_id) & (batch[:, None, :] != padding_id)
    idx = torch.arange(t, device=batch.device)
    causal = idx[None, None, :] <= idx[None, :, None]
    out = same & valid & causal
    return out[0] if segment_ids.ndim == 1 else out
