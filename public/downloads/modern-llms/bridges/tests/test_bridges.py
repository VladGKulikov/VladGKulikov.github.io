#!/usr/bin/env python3
"""Executable acceptance tests for Bridge Layer Release A."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from bridges import (
    NumpyKVCache,
    NumpyKVCacheExplicit,
    TorchKVCache,
    TorchKVCacheExplicit,
    numpy_assistant_labels,
    numpy_block_causal_mask,
    numpy_expert_dispatch,
    numpy_expert_dispatch_explicit,
    numpy_gqa,
    numpy_gqa_explicit,
    numpy_grouped_fake_quant,
    numpy_grouped_fake_quant_explicit,
    numpy_rejection_kernel,
    numpy_rejection_kernel_explicit,
    numpy_rms_norm,
    numpy_rope,
    numpy_rope_explicit,
    numpy_sdpa,
    numpy_sdpa_explicit,
    numpy_segment_position_ids,
    numpy_swiglu,
    numpy_topk_router,
    numpy_topk_router_explicit,
    torch_assistant_labels,
    torch_block_causal_mask,
    torch_expert_dispatch,
    torch_expert_dispatch_explicit,
    torch_gqa,
    torch_gqa_explicit,
    torch_gqa_optimized,
    torch_grouped_fake_quant,
    torch_grouped_fake_quant_explicit,
    torch_rejection_kernel,
    torch_rejection_kernel_explicit,
    torch_rms_norm,
    torch_rms_norm_explicit,
    torch_rope,
    torch_rope_explicit,
    torch_sample_speculative_token,
    torch_sample_speculative_token_explicit,
    torch_sdpa,
    torch_sdpa_explicit,
    torch_sdpa_optimized,
    torch_segment_position_ids,
    torch_swiglu,
    torch_topk_router,
    torch_topk_router_explicit,
)


def b02() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, 8)); pos = np.arange(5, dtype=np.int64) + 11
    expected = numpy_rope_explicit(x, pos)
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    actual = torch_rope_explicit(x_t, torch.tensor(pos))
    np.testing.assert_allclose(expected, actual.detach().numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_rope(x, pos), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch_rope(x_t, torch.tensor(pos)), actual, rtol=1e-12, atol=1e-12)
    actual.square().sum().backward()
    assert x_t.grad is not None and torch.isfinite(x_t.grad).all()
    try:
        torch_rope(torch.ones(1, 2, 4, dtype=torch.long), torch.arange(2))
    except TypeError:
        pass
    else:
        raise AssertionError("integer RoPE input must be rejected")


def b04() -> None:
    rng = np.random.default_rng(4)
    q_np = rng.standard_normal((1, 2, 4, 8))
    k_np = rng.standard_normal((1, 2, 4, 8))
    v_np = rng.standard_normal((1, 2, 4, 6))
    mask_np = np.tril(np.ones((4, 4), dtype=bool))

    expected = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
    q = torch.tensor(q_np, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(k_np, dtype=torch.float64, requires_grad=True)
    v = torch.tensor(v_np, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(mask_np)
    explicit = torch_sdpa_explicit(q, k, v, allowed_mask=mask)
    optimized = torch_sdpa_optimized(q, k, v, allowed_mask=mask, dropout_p=0.0)

    torch.testing.assert_close(explicit, torch.from_numpy(expected), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(optimized, explicit, rtol=1e-10, atol=1e-10)

    # The robust historical wrapper remains equivalent on the same normal mask.
    robust_np = numpy_sdpa(q_np, k_np, v_np, allowed_mask=mask_np)
    robust_t = torch_sdpa(q, k, v, allowed_mask=mask, dropout_p=0.0)
    np.testing.assert_allclose(robust_np, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(robust_t, explicit, rtol=1e-10, atol=1e-10)

    mask_with_empty_row = mask_np.copy()
    mask_with_empty_row[0] = False
    expected_empty = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_with_empty_row)
    actual_empty = torch_sdpa_explicit(q, k, v, allowed_mask=torch.tensor(mask_with_empty_row))
    np.testing.assert_array_equal(expected_empty[..., 0, :], 0.0)
    torch.testing.assert_close(actual_empty[..., 0, :], torch.zeros_like(actual_empty[..., 0, :]))

    explicit.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def b05() -> None:
    rng = np.random.default_rng(5)
    q = rng.standard_normal((1, 4, 3, 8)); k = rng.standard_normal((1, 2, 3, 8)); v = rng.standard_normal((1, 2, 3, 7))
    expected = numpy_gqa_explicit(q, k, v)
    q_t, k_t, v_t = (torch.tensor(a, dtype=torch.float64) for a in (q, k, v))
    explicit = torch_gqa_explicit(q_t, k_t, v_t)
    optimized = torch_gqa_optimized(q_t, k_t, v_t, dropout_p=0.0)
    np.testing.assert_allclose(expected, explicit.numpy(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(optimized, explicit, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numpy_gqa(q, k, v), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(torch_gqa(q_t, k_t, v_t, dropout_p=0.0), explicit, rtol=1e-10, atol=1e-10)


def b07() -> None:
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((3, 5)); w_np = rng.standard_normal(5)
    expected = numpy_rms_norm(x_np, w_np, eps=1e-6)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True); w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    explicit = torch_rms_norm_explicit(x, w, eps=1e-6)
    api = torch_rms_norm(x, w, eps=1e-6)
    np.testing.assert_allclose(expected, explicit.detach().numpy(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: torch_rms_norm_explicit(a, b, eps=1e-6), (x, w))


def b08() -> None:
    rng = np.random.default_rng(8)
    g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
    kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
    expected = numpy_swiglu(g_np, u_np, **kwargs)
    g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
    actual = torch_swiglu(g, u, **kwargs)
    np.testing.assert_allclose(expected, actual.detach().numpy(), rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))


def b09() -> None:
    logits = np.array([[2.1, -0.4, 0.8, 1.3], [0.2, 1.9, -1.1, 0.7]]); bias = np.array([0.0, 0.08, -0.03, 0.02])
    i_np, w_np = numpy_topk_router_explicit(logits, 2, selection_bias=bias)
    i_t, w_t = torch_topk_router_explicit(torch.tensor(logits), 2, selection_bias=torch.tensor(bias))
    np.testing.assert_array_equal(i_np, i_t.numpy()); np.testing.assert_allclose(w_np, w_t.numpy())
    i_robust, w_robust, _ = numpy_topk_router(logits, 2, selection_bias=bias, activation="sigmoid")
    i_t_robust, w_t_robust, _ = torch_topk_router(torch.tensor(logits), 2, selection_bias=torch.tensor(bias), activation="sigmoid")
    np.testing.assert_array_equal(i_robust, i_np); np.testing.assert_allclose(w_robust, w_np)
    np.testing.assert_array_equal(i_t_robust.numpy(), i_np); np.testing.assert_allclose(w_t_robust.numpy(), w_np)


def b10() -> None:
    rng = np.random.default_rng(10)
    x = rng.standard_normal((6, 4)); idx = np.array([[0,1],[1,2],[2,0],[0,2],[1,0],[2,1]]); gates = np.full((6,2), 0.5)
    w1 = rng.standard_normal((3,4,5)); w2 = rng.standard_normal((3,5,4))
    expected = numpy_expert_dispatch_explicit(x, idx, gates, w1, w2)
    actual = torch_expert_dispatch_explicit(torch.tensor(x), torch.tensor(idx), torch.tensor(gates), torch.tensor(w1), torch.tensor(w2))
    np.testing.assert_allclose(expected, actual.numpy(), rtol=1e-10, atol=1e-10)
    robust_np, expected_load = numpy_expert_dispatch(x, idx, gates, w1, w2, capacity=99)
    robust_t, actual_load = torch_expert_dispatch(torch.tensor(x), torch.tensor(idx), torch.tensor(gates), torch.tensor(w1), torch.tensor(w2), capacity=99)
    np.testing.assert_allclose(robust_np, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(robust_t.numpy(), expected, rtol=1e-10, atol=1e-10)
    for key in expected_load: np.testing.assert_array_equal(expected_load[key], actual_load[key].numpy())
    # A malformed route must not wrap around, silently disappear, or truncate.
    for invalid in (-1, 3, .5, np.nan, True):
        bad = np.full(idx.shape, invalid)
        for dispatch, args in (
            (numpy_expert_dispatch_explicit, (x,bad,gates,w1,w2)),
            (numpy_expert_dispatch, (x,bad,gates,w1,w2)),
            (torch_expert_dispatch_explicit, tuple(torch.tensor(a) for a in (x,bad,gates,w1,w2))),
            (torch_expert_dispatch, tuple(torch.tensor(a) for a in (x,bad,gates,w1,w2))),
        ):
            try: dispatch(*args)
            except ValueError: pass
            else: raise AssertionError(f"invalid expert index accepted: {invalid}")
    # Two identical slots contribute twice, under the stated additive contract.
    duplicate = np.zeros((6,2), dtype=np.int64)
    doubled = numpy_expert_dispatch_explicit(x, duplicate, np.ones((6,2)), w1, w2)
    np.testing.assert_allclose(doubled, 2 * (np.tanh(x @ w1[0]) @ w2[0]))
    duplicate_t = torch_expert_dispatch_explicit(torch.tensor(x), torch.tensor(duplicate),
        torch.ones((6,2),dtype=torch.float64), torch.tensor(w1), torch.tensor(w2))
    np.testing.assert_allclose(duplicate_t.numpy(), doubled)


def b17() -> None:
    a, b = NumpyKVCacheExplicit(window=3), TorchKVCacheExplicit(window=3)
    for value in range(5):
        k = np.full((1,1,1,2), value, dtype=np.float64); v = k + 100
        a.append(k, v); b.append(torch.tensor(k), torch.tensor(v))
    np.testing.assert_allclose(a.k, b.k.numpy()); np.testing.assert_array_equal(a.positions, b.positions.numpy())
    assert (a.start_pos, a.next_pos) == (2, 5)
    a.truncate(4); b.truncate(4); np.testing.assert_allclose(a.k, b.k.numpy())
    # Cached single-query attention equals recomputation over the retained absolute window.
    rng = np.random.default_rng(17)
    keys = rng.standard_normal((1, 1, 6, 4)); values = rng.standard_normal((1, 1, 6, 4)); queries = rng.standard_normal((1, 1, 6, 4))
    cache = TorchKVCacheExplicit(window=3)
    for step in range(6):
        cache.append(torch.tensor(keys[..., step:step+1, :]), torch.tensor(values[..., step:step+1, :]))
        cached = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), cache.k, cache.v)
        lo = max(0, step - 2)
        full = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), torch.tensor(keys[..., lo:step+1, :]), torch.tensor(values[..., lo:step+1, :]))
        torch.testing.assert_close(cached, full)


    # Rejected input must not mutate state; rollback cannot resurrect evicted tokens.
    for cache_type, convert in ((NumpyKVCacheExplicit, np.asarray), (TorchKVCacheExplicit, torch.tensor)):
        c = cache_type(window=3)
        c.truncate(0)
        x = convert(np.ones((1,1,5,2), dtype=np.float64))
        c.append(x,x)
        assert (c.start_pos,c.next_pos) == (2,5)
        for bad in (-1,1,6,2.5,True):
            try: c.truncate(bad)
            except ValueError: pass
            else: raise AssertionError("invalid rollback accepted")
        assert (c.start_pos,c.next_pos) == (2,5)
        try: c.append(convert(np.zeros((1,2,1,2))),convert(np.zeros((1,2,1,2))))
        except ValueError: pass
        else: raise AssertionError("geometry change accepted")
        c.truncate(2)
        assert c.k.shape[-2] == 0
        c.append(convert(np.ones((1,1,1,2))),convert(np.ones((1,1,1,2))))
        assert (c.start_pos,c.next_pos) == (2,3)
        for window in (0,-1,2.5,True):
            try: cache_type(window=window)
            except ValueError: pass
            else: raise AssertionError("invalid window accepted")
    a = NumpyKVCacheExplicit(window=3)
    a.append(np.ones((1,1,10,2)),np.ones((1,1,10,2)))
    assert a.k.flags.owndata and a.k.nbytes == 3*2*8
    t = TorchKVCacheExplicit(window=3)
    t.append(torch.ones(1,1,10,2),torch.ones(1,1,10,2))
    assert t.k.untyped_storage().nbytes() == t.k.numel()*t.k.element_size()


def b18() -> None:
    p_np = np.array([0.40,0.30,0.15,0.10,0.05]); q_np = np.array([0.25,0.35,0.20,0.10,0.10])
    a_np, r_np, balance_np = numpy_rejection_kernel_explicit(p_np, q_np)
    p, q = torch.tensor(p_np), torch.tensor(q_np)
    a_t, r_t, balance_t = torch_rejection_kernel_explicit(p, q)
    np.testing.assert_allclose(a_np, a_t.numpy()); np.testing.assert_allclose(r_np, r_t.numpy()); np.testing.assert_allclose(balance_np, balance_t.numpy()); np.testing.assert_allclose(balance_np, p_np)
    gen = torch.Generator().manual_seed(18); counts = torch.zeros(5)
    for _ in range(20000):
        proposal = int(torch.multinomial(q, 1, generator=gen)); token, _ = torch_sample_speculative_token_explicit(p, q, proposal, generator=gen); counts[token] += 1
    assert torch.max(torch.abs(counts / counts.sum() - p)) < 0.015
    q_zero = torch.tensor([0.0, 0.4, 0.3, 0.2, 0.1], dtype=p.dtype); torch_rejection_kernel(p, q_zero)
    a_eq, r_eq, balance_eq = torch_rejection_kernel(p, p); torch.testing.assert_close(balance_eq, p); assert r_eq.sum() == 0

    # A real, tiny residual must remain a probability distribution.
    near_q = np.array([1.0, 0.0]); near_p = np.array([1.0-1e-14, 1e-14])
    for kernel, args in [
        (numpy_rejection_kernel_explicit, (near_p, near_q)),
        (torch_rejection_kernel_explicit, (torch.tensor(near_p), torch.tensor(near_q))),
    ]:
        _, residual, balance = kernel(*args)
        np.testing.assert_allclose(np.asarray(residual), [0.0, 1.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.asarray(balance), near_p, rtol=0, atol=1e-15)
    # Uniform RNG can return zero: a zero-acceptance proposal must still be rejected.
    from unittest.mock import patch
    with patch.object(torch, "rand", return_value=torch.tensor(0.0, dtype=torch.float64)):
        token, accepted = torch_sample_speculative_token_explicit(
            torch.tensor([0.,1.], dtype=torch.float64),
            torch.tensor([1.,0.], dtype=torch.float64), 0, generator=gen)
        assert token == 1 and not accepted


def b19() -> None:
    rng = np.random.default_rng(19); x = rng.standard_normal((3,2,11)); grid = np.array([-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6.0])
    expected = numpy_grouped_fake_quant_explicit(x, grid=grid, group_size=4)
    actual = torch_grouped_fake_quant_explicit(torch.tensor(x), grid=torch.tensor(grid), group_size=4)
    np.testing.assert_allclose(expected, actual.numpy())
    robust_np = numpy_grouped_fake_quant(x, grid=grid, group_size=4)
    robust_t = torch_grouped_fake_quant(torch.tensor(x), grid=torch.tensor(grid), group_size=4)
    np.testing.assert_allclose(robust_np["dequantized"], expected)
    torch.testing.assert_close(robust_t["dequantized"], actual)
    zeros = torch.zeros(2, 3, 7, dtype=torch.float64)
    zero_q = torch_grouped_fake_quant_explicit(zeros, grid=torch.tensor(grid), group_size=4)
    torch.testing.assert_close(zero_q, zeros)


def b20() -> None:
    ids = np.array([10,11,12,13,14,15]); mask = np.array([0,0,0,1,1,1], dtype=bool)
    expected = numpy_assistant_labels(ids, mask); actual = torch_assistant_labels(torch.tensor(ids), torch.tensor(mask))
    np.testing.assert_array_equal(expected, actual.numpy()); assert expected.tolist() == [-100,-100,-100,13,14,15]


def b21() -> None:
    segments = np.array([0,0,0,1,1,1,-1]); expected_pos = numpy_segment_position_ids(segments); expected_mask = numpy_block_causal_mask(segments)
    t_segments = torch.tensor(segments); actual_pos = torch_segment_position_ids(t_segments); actual_mask = torch_block_causal_mask(t_segments)
    np.testing.assert_array_equal(expected_pos, actual_pos.numpy()); np.testing.assert_array_equal(expected_mask, actual_mask.numpy())
    rng = np.random.default_rng(21); x = torch.tensor(rng.standard_normal((1,1,7,4)), dtype=torch.float64); mask = actual_mask[None,None]
    out_a = F.scaled_dot_product_attention(x, x, x, attn_mask=mask)
    changed = x.clone(); changed[..., :3, :] += 1000
    out_b = F.scaled_dot_product_attention(changed, changed, changed, attn_mask=mask)
    torch.testing.assert_close(out_a[..., 3:6, :], out_b[..., 3:6, :], rtol=0, atol=0)
    repeated = np.array([0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(numpy_segment_position_ids(repeated), [0, 1, 0, 1, 0, 1])
    repeated_mask = numpy_block_causal_mask(repeated)
    assert not repeated_mask[4, 0] and repeated_mask[5, 4]


def main() -> int:
    tests = [b02,b04,b05,b07,b08,b09,b10,b17,b18,b19,b20,b21]
    for test in tests:
        test(); print(f"{test.__name__.upper()}: PASS")
    print(f"Bridge Layer Release A: {len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
