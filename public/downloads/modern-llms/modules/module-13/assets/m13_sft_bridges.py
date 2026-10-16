"""Modern LLMs module 13: executable NumPy → PyTorch bridges (B20, B21).

This file is generated from the canonical ``bridges/`` package.  It contains
all learner-facing implementations shown in the lecture and their parity
checks.  Run it directly with Python.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


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


if __name__ == "__main__":
    ids_np = np.array([10,11,12,13,14,15]); mask_np = np.array([0,0,0,1,1,1], dtype=bool)
    np_labels = numpy_assistant_labels(ids_np, mask_np); t_labels = torch_assistant_labels(torch.tensor(ids_np), torch.tensor(mask_np))
    np.testing.assert_array_equal(np_labels, t_labels.numpy()); assert np_labels.tolist() == [-100,-100,-100,13,14,15]
    # Labels are stored at their own positions. Shift them once for explicit CE.
    scores = torch.zeros((len(ids_np), 20), dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(scores[:-1], t_labels[1:], ignore_index=-100)
    torch.testing.assert_close(loss, torch.log(torch.tensor(20.,dtype=torch.float64)))
    loss.backward()
    assert torch.count_nonzero(scores.grad[-1]) == 0
    assert torch.count_nonzero(scores.grad[:2]) == 0
    for ids_bad, mask_bad in [(ids_np.astype(float),mask_np),
                               (ids_np,mask_np.astype(int)),
                               (np.array([-1,1]),np.array([False,True]))]:
        for run in (lambda: numpy_assistant_labels(ids_bad,mask_bad),
                    lambda: torch_assistant_labels(torch.tensor(ids_bad),torch.tensor(mask_bad))):
            try:
                run()
            except (TypeError,ValueError):
                pass
            else:
                raise AssertionError("invalid label input was accepted")
    print("B20 assistant labels: PASS")

    segments_np = np.array([0,0,0,1,1,1,-1]); np_pos = numpy_segment_position_ids(segments_np); np_mask = numpy_block_causal_mask(segments_np)
    t_segments = torch.tensor(segments_np); t_pos = torch_segment_position_ids(t_segments); t_mask = torch_block_causal_mask(t_segments)
    np.testing.assert_array_equal(np_pos, t_pos.numpy()); np.testing.assert_array_equal(np_mask, t_mask.numpy())
    rng = np.random.default_rng(21); x = torch.tensor(rng.standard_normal((1,1,7,4)), dtype=torch.float64); mask = t_mask[None,None]
    out_a = torch.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=mask)
    x_changed = x.clone(); x_changed[..., :3, :] += 1000
    out_b = torch.nn.functional.scaled_dot_product_attention(x_changed, x_changed, x_changed, attn_mask=mask)
    torch.testing.assert_close(out_a[..., 3:6, :], out_b[..., 3:6, :], rtol=0, atol=0)
    repeated = np.array([0,0,1,1,0,0])
    np.testing.assert_array_equal(numpy_segment_position_ids(repeated), [0,1,0,1,0,1])
    repeat_mask = numpy_block_causal_mask(repeated)
    assert not repeat_mask[4,0] and repeat_mask[5,4]
    print("B21 packed isolation: PASS")
    print("Module 13 bridge asset: PASS")
