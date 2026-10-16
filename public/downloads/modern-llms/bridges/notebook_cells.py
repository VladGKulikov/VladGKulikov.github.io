"""Generate self-contained learner-notebook cells for canonical bridges."""
from __future__ import annotations

import inspect
import textwrap
from collections.abc import Iterable
from typing import Any

from . import attention, cache, moe, normalization, quantization, rope, sft, speculative


FUNCTIONS: dict[str, list[Any]] = {
    "B02": [rope.numpy_rope_explicit, rope.torch_rope_explicit],
    "B04": [
        attention.numpy_sdpa_explicit,
        attention.torch_sdpa_explicit,
        attention.torch_sdpa_optimized,
    ],
    "B05": [
        attention.numpy_sdpa_explicit,
        attention.torch_sdpa_explicit,
        attention.numpy_gqa_explicit,
        attention.torch_gqa_explicit,
        attention.torch_gqa_optimized,
    ],
    "B07": [
        normalization.numpy_rms_norm,
        normalization.torch_rms_norm_explicit,
        normalization.torch_rms_norm,
    ],
    "B08": [normalization.numpy_swiglu, normalization.torch_swiglu],
    "B09": [moe.numpy_topk_router_explicit, moe.torch_topk_router_explicit],
    "B10": [moe.numpy_expert_dispatch_explicit, moe.torch_expert_dispatch_explicit],
    "B17": [cache.NumpyKVCacheExplicit, cache.TorchKVCacheExplicit],
    "B18": [
        speculative.numpy_rejection_kernel_explicit,
        speculative.torch_rejection_kernel_explicit,
        speculative.torch_sample_speculative_token_explicit,
    ],
    "B19": [
        quantization.numpy_grouped_fake_quant_explicit,
        quantization.torch_grouped_fake_quant_explicit,
    ],
    "B20": [sft.numpy_assistant_labels, sft.torch_assistant_labels],
    "B21": [
        sft.numpy_segment_position_ids,
        sft.torch_segment_position_ids,
        sft.numpy_block_causal_mask,
        sft.torch_block_causal_mask,
    ],
}

TITLES = {
    "B02": ("RoPE: NumPy и PyTorch", "RoPE: NumPy and PyTorch"),
    "B04": ("SDPA: прозрачная формула и тензорный API", "SDPA: transparent math and tensor API"),
    "B05": ("GQA: явное повторение и `enable_gqa`", "GQA: explicit repetition and `enable_gqa`"),
    "B07": ("RMSNorm: выход и градиент", "RMSNorm: output and gradient"),
    "B08": ("SwiGLU: обычный и ограниченный варианты", "SwiGLU: standard and bounded variants"),
    "B09": ("MoE top-k: выбор и веса смеси", "MoE top-k: selection and mixture weights"),
    "B10": ("MoE dispatch/combine", "MoE dispatch/combine"),
    "B17": ("KV-cache: append, окно и rollback", "KV cache: append, window, and rollback"),
    "B18": ("Точная спекулятивная коррекция", "Exact speculative correction"),
    "B19": ("Групповая имитационная квантизация", "Grouped fake quantization"),
    "B20": ("Метки только для ответа ассистента", "Assistant-only labels"),
    "B21": ("Изоляция упакованных сегментов", "Packed-segment isolation"),
}

TEST_CODE: dict[str, str] = {
    "B02": '''
rng = np.random.default_rng(2)
x_np = rng.standard_normal((2, 5, 8))
pos_np = np.arange(5, dtype=np.int64) + 11
np_out = numpy_rope_explicit(x_np, pos_np)
x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
t_out = torch_rope_explicit(x, torch.tensor(pos_np))
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-12, atol=1e-12)
t_out.square().sum().backward()
# A rotation is orthogonal: d ||R x||^2 / dx = 2 x.
torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=1e-12, atol=1e-12)
# Half-precision IDs would merge 2048 and 2049 before the rotation.
long_ids = torch.tensor([2048, 2049], dtype=torch.int64)
half_x = torch.ones((2, 8), dtype=torch.float16)
half_out = torch_rope_explicit(half_x, long_ids)
expected = numpy_rope_explicit(half_x.numpy(), long_ids.numpy())
np.testing.assert_allclose(half_out.numpy(), expected, rtol=1e-3, atol=1e-3)
assert not torch.equal(half_out[0], half_out[1])
assert long_ids.to(torch.float16)[0] == long_ids.to(torch.float16)[1]
print("B02 outputs, analytic gradient, and float16 positions: PASS")
''',
    "B04": '''
rng = np.random.default_rng(4)
q_np = rng.standard_normal((1, 2, 4, 8))
k_np = rng.standard_normal((1, 2, 4, 8))
v_np = rng.standard_normal((1, 2, 4, 6))
mask_np = np.tril(np.ones((4, 4), dtype=bool))

np_out = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
q = torch.tensor(q_np, dtype=torch.float64, requires_grad=True)
k = torch.tensor(k_np, dtype=torch.float64, requires_grad=True)
v = torch.tensor(v_np, dtype=torch.float64, requires_grad=True)
mask = torch.tensor(mask_np)
explicit_out = torch_sdpa_explicit(q, k, v, allowed_mask=mask)
api_out = torch_sdpa_optimized(q, k, v, allowed_mask=mask, dropout_p=0.0)

torch.testing.assert_close(explicit_out, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api_out, explicit_out, rtol=1e-10, atol=1e-10)

mask_np[0] = False
np_empty = numpy_sdpa_explicit(q_np, k_np, v_np, allowed_mask=mask_np)
t_empty = torch_sdpa_explicit(q, k, v, allowed_mask=torch.tensor(mask_np))
api_empty = torch_sdpa_optimized(q, k, v, allowed_mask=torch.tensor(mask_np), dropout_p=0.0)
np.testing.assert_array_equal(np_empty[..., 0, :], 0.0)
torch.testing.assert_close(t_empty[..., 0, :], torch.zeros_like(t_empty[..., 0, :]))
torch.testing.assert_close(api_empty, t_empty, rtol=1e-10, atol=1e-10)

grads_explicit = torch.autograd.grad(explicit_out.square().sum(), (q, k, v), retain_graph=True)
grads_api = torch.autograd.grad(api_out.square().sum(), (q, k, v), retain_graph=True)
for ge, ga in zip(grads_explicit, grads_api):
    torch.testing.assert_close(ge, ga, rtol=1e-9, atol=1e-10)
explicit_out.sum().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
print("B04 explicit NumPy / explicit PyTorch / optimized API: PASS")
''',
    "B05": '''
rng = np.random.default_rng(5)
q_np = rng.standard_normal((1, 4, 3, 8))
k_np = rng.standard_normal((1, 2, 3, 8))
v_np = rng.standard_normal((1, 2, 3, 7))
np_out = numpy_gqa_explicit(q_np, k_np, v_np)
q, k, v = (torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (q_np, k_np, v_np))
explicit = torch_gqa_explicit(q, k, v)
api = torch_gqa_optimized(q, k, v, dropout_p=0.0)
torch.testing.assert_close(explicit, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
for ge, ga in zip(torch.autograd.grad(explicit.square().sum(), (q,k,v)),
                  torch.autograd.grad(api.square().sum(), (q,k,v))):
    torch.testing.assert_close(ge, ga, rtol=1e-9, atol=1e-10)
print("B05 explicit repetition / enable_gqa: PASS")
''',
    "B07": '''
rng = np.random.default_rng(7)
x_np = rng.standard_normal((3, 5)); w_np = rng.standard_normal(5)
np_out = numpy_rms_norm(x_np, w_np, eps=1e-6)
x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
explicit = torch_rms_norm_explicit(x, w, eps=1e-6)
api = torch_rms_norm(x, w, eps=1e-6)
np.testing.assert_allclose(np_out, explicit.detach().numpy(), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_rms_norm_explicit(a, b, eps=1e-6), (x, w))
for a, b in zip(torch.autograd.grad(explicit.square().sum(), (x, w)),
                torch.autograd.grad(api.square().sum(), (x, w))):
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
for dtype in (torch.float16, torch.bfloat16):
    large = torch.full((2, 8), 1000.0, dtype=dtype)
    torch.testing.assert_close(torch_rms_norm_explicit(large), torch.ones_like(large))
print("B07 explicit NumPy / PyTorch / F.rms_norm: PASS")
''',
    "B08": '''
rng = np.random.default_rng(8)
g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
np_out = numpy_swiglu(g_np, u_np, **kwargs)
g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
t_out = torch_swiglu(g, u, **kwargs)
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))
extreme = np.array([-1000., -9., 0., 9., 1000.])
with np.errstate(over='raise', invalid='raise'):
    stable = numpy_swiglu(extreme, np.ones_like(extreme), **kwargs)
np.testing.assert_allclose(stable, torch_swiglu(torch.tensor(extreme), torch.ones(5, dtype=torch.float64), **kwargs))
outside = torch.tensor([-9., 9.], dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(lambda a: torch_swiglu(a, a, **kwargs), (outside,))
print("B08 forward + gradient: PASS")
''',
    "B09": '''
logits_np = np.array([[2.1, -0.4, 0.8, 1.3], [0.2, 1.9, -1.1, 0.7]])
bias_np = np.array([0.0, 0.08, -0.03, 0.02])
i_np, w_np = numpy_topk_router_explicit(logits_np, 2, selection_bias=bias_np)
i_t, w_t = torch_topk_router_explicit(torch.tensor(logits_np), 2, selection_bias=torch.tensor(bias_np))
np.testing.assert_array_equal(i_np, i_t.numpy())
np.testing.assert_allclose(w_np, w_t.numpy())
extreme = np.array([[-1002., -1000., -1001., -1003.]])
with np.errstate(over="raise", invalid="raise", divide="raise"):
    ei, ew = numpy_topk_router_explicit(extreme, 2)
ti, tw = torch_topk_router_explicit(torch.tensor(extreme), 2)
np.testing.assert_array_equal(ei, [[1, 2]])
np.testing.assert_array_equal(ei, ti.numpy())
np.testing.assert_allclose(ew, tw.numpy(), rtol=1e-12, atol=1e-12)
z = torch.tensor(logits_np, requires_grad=True)
assert torch.autograd.gradcheck(lambda x: torch_topk_router_explicit(x, 2)[1], (z,))
one = torch_topk_router_explicit(z, 1)[1]
np.testing.assert_allclose(torch.autograd.grad(one.sum(), z)[0].numpy(), 0., atol=1e-12)
print("B09 routing, extreme logits, and selected-gate gradients: PASS")
''',
    "B10": '''
rng = np.random.default_rng(10)
x_np = rng.standard_normal((6, 4))
idx_np = np.array([[0,1],[1,2],[2,0],[0,2],[1,0],[2,1]])
gates_np = np.full((6,2), 0.5)
w1_np = rng.standard_normal((3,4,5)); w2_np = rng.standard_normal((3,5,4))
np_out = numpy_expert_dispatch_explicit(x_np, idx_np, gates_np, w1_np, w2_np)
t_out = torch_expert_dispatch_explicit(torch.tensor(x_np), torch.tensor(idx_np), torch.tensor(gates_np), torch.tensor(w1_np), torch.tensor(w2_np))
np.testing.assert_allclose(np_out, t_out.numpy(), rtol=1e-10, atol=1e-10)
args = [torch.tensor(x_np, requires_grad=True), torch.tensor(idx_np),
        torch.tensor(gates_np, requires_grad=True), torch.tensor(w1_np, requires_grad=True),
        torch.tensor(w2_np, requires_grad=True)]
assert torch.autograd.gradcheck(
    lambda x, g, a, b: torch_expert_dispatch_explicit(x, args[1], g, a, b),
    (args[0], args[2], args[3], args[4]))
for invalid in (-1, w1_np.shape[0], .5, np.nan, True):
    bad = np.full(idx_np.shape, invalid)
    for dispatch, values in (
        (numpy_expert_dispatch_explicit, (x_np,bad,gates_np,w1_np,w2_np)),
        (torch_expert_dispatch_explicit, tuple(torch.tensor(a) for a in (x_np,bad,gates_np,w1_np,w2_np))),
    ):
        try: dispatch(*values)
        except ValueError: pass
        else: raise AssertionError("invalid expert index accepted")
duplicate = np.zeros_like(idx_np)
expected = idx_np.shape[1] * (np.tanh(x_np @ w1_np[0]) @ w2_np[0])
actual = numpy_expert_dispatch_explicit(x_np,duplicate,np.ones_like(gates_np),w1_np,w2_np)
np.testing.assert_allclose(actual,expected)
actual_t = torch_expert_dispatch_explicit(torch.tensor(x_np),torch.tensor(duplicate),
    torch.ones_like(torch.tensor(gates_np)),torch.tensor(w1_np),torch.tensor(w2_np))
np.testing.assert_allclose(actual_t.numpy(),expected)
print("B10 values, gradients, repeated assignments, and invalid indices: PASS")
''',
    "B17": '''
np_cache = NumpyKVCacheExplicit(window=3); t_cache = TorchKVCacheExplicit(window=3)
for value in range(5):
    k_np = np.full((1,1,1,2), value, dtype=np.float64); v_np = k_np + 100
    np_cache.append(k_np, v_np); t_cache.append(torch.tensor(k_np), torch.tensor(v_np))
np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
np.testing.assert_array_equal(np_cache.positions, t_cache.positions.numpy())
assert (np_cache.start_pos, np_cache.next_pos) == (2,5)
np_cache.truncate(4); t_cache.truncate(4)
np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
rng = np.random.default_rng(17)
keys = rng.standard_normal((1,1,6,4)); values = rng.standard_normal((1,1,6,4)); queries = rng.standard_normal((1,1,6,4))
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

print("B17 window, ownership, rollback, and invalid inputs: PASS")
''',
    "B18": '''
p_np = np.array([0.40,0.30,0.15,0.10,0.05]); q_np = np.array([0.25,0.35,0.20,0.10,0.10])
a_np, r_np, b_np = numpy_rejection_kernel_explicit(p_np, q_np)
a_t, r_t, b_t = torch_rejection_kernel_explicit(torch.tensor(p_np), torch.tensor(q_np))
np.testing.assert_allclose(a_np, a_t.numpy()); np.testing.assert_allclose(r_np, r_t.numpy()); np.testing.assert_allclose(b_np, b_t.numpy()); np.testing.assert_allclose(b_np, p_np)
gen = torch.Generator().manual_seed(18); counts = torch.zeros(5)
q = torch.tensor(q_np); p = torch.tensor(p_np)
for _ in range(12000):
    proposal = int(torch.multinomial(q, 1, generator=gen))
    token, _ = torch_sample_speculative_token_explicit(p, q, proposal, generator=gen)
    counts[token] += 1
assert torch.max(torch.abs(counts / counts.sum() - p)) < 0.02

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

print("B18 exact balance, tiny residual, zero draw, Monte Carlo: PASS")
''',
    "B19": '''
rng = np.random.default_rng(19)
x_np = rng.standard_normal((3,2,11))
grid_np = np.array([-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6.0])
np_q = numpy_grouped_fake_quant_explicit(x_np, grid=grid_np, group_size=4)
t_q = torch_grouped_fake_quant_explicit(torch.tensor(x_np), grid=torch.tensor(grid_np), group_size=4)
np.testing.assert_allclose(np_q, t_q.numpy())
zeros = torch.zeros(2,3,7, dtype=torch.float64)
zero_q = torch_grouped_fake_quant_explicit(zeros, grid=torch.tensor(grid_np), group_size=4)
torch.testing.assert_close(zero_q, zeros)
print("B19 explicit grouped fake quantization: PASS")
''',
    "B20": '''
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
''',
    "B21": '''
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
''',
}


def _source(objects: Iterable[Any]) -> str:
    return "\n\n".join(textwrap.dedent(inspect.getsource(obj)).strip() for obj in objects)


def bridge_notebook_cells(ids: Iterable[str], lang: str) -> list[tuple[str, str]]:
    """Return self-contained markdown/code cells for selected bridge IDs."""
    if lang not in {"ru", "en"}:
        raise ValueError("lang must be ru or en")
    setup = '''
import importlib.util
import subprocess
import sys

missing = []
if importlib.util.find_spec("numpy") is None:
    missing.append("numpy>=1.26")
if importlib.util.find_spec("torch") is None:
    missing.append("torch>=2.5")
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import torch
import torch.nn.functional as F
import math
'''
    cells: list[tuple[str, str]] = []
    for bridge_id in ids:
        ru, en = TITLES[bridge_id]
        title = ru if lang == "ru" else en
        intro = (
            "Эта ячейка встраивает канонический NumPy→PyTorch-мост и выполняет проверку эквивалентности. Код самодостаточен и не импортирует файлы репозитория."
            if lang == "ru"
            else "This cell embeds the canonical NumPy→PyTorch bridge and runs its parity check. It is self-contained and does not import repository files."
        )
        if bridge_id == 'B04':
            intro += ('\n\nИспользуем конечные вещественные Q/K/V с непустыми совместимыми формами '
                      '`[..., T_q, D]`, `[..., T_k, D]`, `[..., T_k, D_v]`. Для API тип и устройство '
                      'общие, булева маска находится на том же устройстве. В этом коротком мосте '
                      'пустая строка всегда даёт ноль; выше основной код позволяет выбрать ошибку '
                      'или нулевой результат. Проверка ниже сравнивает обе реализации и API, '
                      'включая пустую строку и градиенты.' if lang == 'ru' else
                      '\n\nInputs are finite floating Q/K/V with nonempty compatible shapes '
                      '`[..., T_q, D]`, `[..., T_k, D]`, and `[..., T_k, D_v]`. API comparisons '
                      'share a dtype and device; the Boolean mask is on that device. This short '
                      'bridge always returns zero for an empty row, whereas the main reference '
                      'above offers an error/zero choice. The checks compare both explicit paths '
                      'and the API, including empty rows and gradients.')
        elif bridge_id == 'B05':
            intro += ('\n\nЗдесь головы выделены в ось 1: Q `[B,H_q,T_q,D]`, K '
                      '`[B,H_kv,T_k,D]`, V `[B,H_kv,T_k,D_v]`; все оси непусты, `H_q` кратно '
                      '`H_kv`. Сохраняются допущения B04 о значениях, типе, устройстве и маске. '
                      'Основной пример выше объединял головы в `[T,H*D]`; это другой интерфейс, '
                      'к которому нельзя применить повтор по оси 1 без предварительной смены формы.'
                      if lang == 'ru' else
                      '\n\nHeads occupy axis 1 here: Q `[B,H_q,T_q,D]`, K `[B,H_kv,T_k,D]`, '
                      'and V `[B,H_kv,T_k,D_v]`, with nonempty axes and `H_q` divisible by `H_kv`. '
                      'Keep B04\'s value, dtype, device, and mask assumptions. The earlier reference '
                      'flattened heads into `[T,H*D]`; reshape that interface before repeating '
                      'anything along axis 1.')
        elif bridge_id == "B07":
            intro += "\n\n" + ('Сравниваем конечный вещественный вход `[..., D]`, `D>0`, и необязательный вес `[D]` при конечном `eps>0`. Для сопоставления с API вход и вес имеют общий плавающий тип и устройство. Предполагаем, что квадраты, сумма и результат помещаются в рабочий формат: сам по себе конечный FP64-вход этого не гарантирует. NumPy раскрывает редукцию, PyTorch добавляет автоматическое дифференцирование.' if lang == "ru" else 'Compare finite floating inputs `[..., D]`, `D>0`, and an optional weight `[D]`, with finite `eps>0`. API comparisons use a common floating dtype and device for input and weight. Squares, their sum, and the output must fit the working format; merely being finite in FP64 does not ensure this. NumPy exposes the reduction, while PyTorch adds autograd.')
        elif bridge_id == "B10":
            intro += "\n\n" + ('Входы имеют формы hidden `[T,D]`, индексы и веса смеси `[T,K]`, матрицы экспертов `[E,D,F]` и `[E,F,D]`. Индексы — целые числа от 0 до E−1; одинаковые номера в разных назначениях допустимы, их вклады складываются. Вещественные массивы конечны и используют совместимый плавающий тип; в PyTorch все тензоры находятся на одном устройстве. Минимальный мост проверяет индексы, затем раскрывает отправку и суммирование; вместимость здесь не ограничена.' if lang == "ru" else 'Use hidden `[T,D]`, indices and gates `[T,K]`, and expert weights `[E,D,F]` and `[E,F,D]`. Indices are integers from 0 through E−1; repeated assignments are allowed and their contributions add. Floating arrays are finite with compatible dtypes; PyTorch tensors share a device. The minimal bridge validates indices before exposing dispatch and combination, without a capacity limit.')
        if bridge_id in ("B18", "B19", "B20"):
            # Keep the explicit-kernel preconditions identical to the lecture.
            from bridges.render import PRESENTATIONS
            presentation = PRESENTATIONS[bridge_id]
            intro += "\n\n" + (presentation.torch_ru if lang == "ru" else presentation.torch_en)
        cells.append(("markdown", f"## PyTorch bridge · {bridge_id} — {title}\n\n{intro}"))
        cells.append(("code", textwrap.dedent(setup).strip() + "\n\n" + _source(FUNCTIONS[bridge_id]) + "\n\n" + textwrap.dedent(TEST_CODE[bridge_id]).strip()))
    return cells
