# Module 5. Modern Attention Mechanisms

> Open lecture edition. Self-checks, exercises, solutions, and notebooks mentioned in reading routes are available in the [complete course on Stepik](https://stepik.org/297276).

*Modern LLMs · October 2026 revision · sources checked on October 10, 2026*

Module 3 built a Transformer; Module 4 supplied its sense of position. We now ask **how attention can use a long context within a limited memory and compute budget**. Three decisions matter: what to retain from past tokens, how to compute interactions with it, and which positions a query may use.

**Prerequisites.** Matrix multiplication, softmax, and attention heads from M3; rotary positional embeddings (RoPE) from M4 for the MLA discussion. GPU memory terminology is introduced here.

**Reading routes**

- **First pass:** §§1–2 → §4 up to the implementation panel → §§5–11 → §§15–19. In §7, establish what is cached before following the algebra. Use §17 and exercises 1–4 to check that you can explain which resource each method changes.
- **Implementation pass:** return to the expandable B04/B05 panels in §§4/6, §§12–14, and the notebook. Compare direct and streaming execution and reproduce mask and head-order errors. Exercises 6–7 connect these experiments to the derivations.
- **Reference pass:** §3 provides statistical connections; §§5, 9, 11, and 15 help you read unfamiliar architectures. Exercise 5 is an optional GPU project. Sources and the transition to M6 close the module.

## 1. What grows with the context?

Suppose a model has processed a prompt and is generating its next token. A **query** $q$ is compared with the **keys** $k_j$ of visible positions; the resulting weights mix their **values** $v_j$. An attention head performs this operation in its own feature space. Write $H_q$ for the number of query heads and $H_{kv}$ for the number of K/V heads. Several query heads can share one set of keys and values.

Previously computed K/V can be retained in a **KV cache**. For one sequence, $L$ identical layers, and equal key/value width $d_h$, one retained token costs

$$b_{KV}=2LH_{kv}d_hb,$$

where $b$ is bytes per element; the factor 2 counts both keys and values. With $L=36$, $H_{kv}=8$, $d_h=64$, and BF16 ($b=2$), this is 73,728 bytes, or **72 KiB per token**. If every layer retains all 131,072 positions, the K/V payload is **9 GiB**. Here $1\text{ KiB}=2^{10}$ bytes and $1\text{ GiB}=2^{30}$ bytes. Windows and heterogeneous layers require the layer-by-layer sum in §5.

A second cost appears during **prefill**, processing the input sequence. A simple implementation materializes an $N\times N$ score matrix per head. At $N=8192$, one FP32 matrix occupies $8192^2\cdot4=256$ MiB. This temporary array serves a different purpose from the cache that survives into later steps. During single-token **decode**, the new query usually compares with $N$ retained keys: its score row has $N$ entries, not $N^2$.

MQA, GQA, and MLA change the retained representation. FlashAttention reduces movement of intermediate data. Windows and sparse patterns reduce selected interactions. These approaches can be combined, so a method name alone does not determine resource use.

![VIZ m5/01 — three directions in attention design](assets/modern-llms/en/module-04/m5_01_attention_timeline.svg)
*State compression, execution scheduling, and position selection solve different parts of the problem.*

## 2. Three directions in attention design

**Multi-Head Attention (MHA)** gives every query head its own K/V projections. At fixed query-head count, this is the largest of the following three cache layouts.

**Multi-Query Attention (MQA)** shares one K/V set across all query heads. With other dimensions fixed, it reduces cache size by $H_q$ relative to MHA, although aggressive sharing can reduce quality.

**Grouped-Query Attention (GQA)** divides query heads into groups, each with its own K/V set. Group size $g=H_q/H_{kv}$ must be an integer. $H_{kv}=H_q$ recovers MHA; $H_{kv}=1$ recovers MQA. The GQA authors obtained quality close to MHA and speed comparable to MQA in their uptraining experiments, which continued training existing models. This is evidence for their protocol, not a guarantee for arbitrary $g$. [GQA](https://arxiv.org/abs/2305.13245).

**Multi-head Latent Attention (MLA)** caches a compact intermediate representation, the joint KV latent, together with a small positional component (§7). DeepSeek-V2 reported a 93.3% KV-cache reduction against DeepSeek 67B and up to 5.76× higher maximum generation throughput in its system comparison. That baseline differs from our arithmetic example in §5. [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

The second direction changes execution. **FlashAttention** processes small array blocks, or **tiles**, in fast accelerator memory. **HBM (High Bandwidth Memory)** is GPU main memory; **SRAM (Static Random-Access Memory)** is smaller, fast on-chip storage. Tiled execution updates softmax incrementally instead of writing the entire probability matrix to HBM, while preserving the mathematical attention operation. [FlashAttention](https://arxiv.org/abs/2205.14135).

The third direction changes available interactions. Sliding windows retain nearby positions. StreamingLLM also keeps a few initial real tokens; §10 explains their role. MoBA, NSA, and DSA use query/key content to select blocks or tokens (§11). Selection can reduce the main attention calculation without making every unselected cache entry disposable: a later query may need it.

## 3. Classical connections: smoothing, distributions, and state

This section can be read later. It connects attention to familiar mathematical ideas without introducing prerequisites for §4.

### Kernel smoothing

The Nadaraya–Watson estimator averages values using a similarity function $\kappa$:

$$\hat f(q)=\frac{\sum_i\kappa(q,k_i)v_i}{\sum_i\kappa(q,k_i)}.$$

For visible keys in softmax attention, use $\kappa(q,k_i)=\exp(q^\top k_i/\sqrt{d_h})$. Index $i$ runs over keys, with corresponding values $v_i$. The analogy is useful, but Q/K/V are learned together and later layers transform the resulting mixture. Here a *kernel* is a similarity function; a GPU kernel in §8 is executable code.

### The Gibbs distribution

With unscaled scores $s_i=q^\top k_i$ and temperature $T>0$,

$$p_i=\frac{e^{s_i/T}}{\sum_j e^{s_j/T}}$$

has the Gibbs form. The energy convention $e^{-E_i/T}$ gives $E_i=-s_i$. Choosing $T=\sqrt{d_h}$ recovers the SDPA scale, motivated by the variance calculation in §4. If scores have already been divided by $\sqrt{d_h}$, applying that factor again changes the function.

### State for continuing the computation

At fixed weights and position rules, the KV cache retains past features needed by the next attention step. MHA stores per-head K/V, GQA shares them within groups, and MLA keeps a shared latent plus positional state. Recurrent and state-space mechanisms may instead use a fixed-size state (Module 8). Complete decoding also needs the current token, positions, and generation rules.

## 4. SDPA, masks, and the square-root scale

**SDPA—Scaled Dot-Product Attention** is

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right)V.
$$

Here $Q\in\mathbb R^{T_q\times d_h}$, $K\in\mathbb R^{T_k\times d_h}$, and $V\in\mathbb R^{T_k\times d_v}$. An additive mask $M$ uses zero for an allowed edge and $-\infty$ for a forbidden one.

Softmax normalizes each query row across keys. Causality compares **token positions**, $p_k\le p_q$; only aligned Q/K origins reduce this to $j\le i$. A window of $W$ positions, including the current one, requires $0\le p_q-p_k<W$.

A query at position 5 can see keys 0–5. But `tril(ones(1, 6))` keeps only key 0: row number has been mistaken for token position. After a prefix of length $P$, query row $i$ has position $P+i$. PyTorch SDPA uses an upper-left causal mask for rectangular inputs with `is_causal=True`; here we pass an explicit position-aware mask and `is_causal=False`. SDPA boolean masks use `True = allowed`, unlike `MultiheadAttention.key_padding_mask`. Set `dropout_p=0.0` when testing without random weight dropping. [PyTorch contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

![VIZ m5/07 — keys visible to each query](assets/modern-llms/en/module-04/m5_07_masks.png)
*Rows are queries; columns are keys. Initial real tokens add visible positions. The learned sink in §10 is not a token position.*

The $1/\sqrt{d_h}$ factor comes from a variance calculation. Under the idealized assumption that query and key components are independent, zero-mean, and unit-variance,

$$
\operatorname{Var}(q^\top k)=d_h.
$$

Without scaling, the typical logit magnitude grows as $\sqrt{d_h}$ and softmax becomes increasingly saturated. The assumptions are not a permanent description of trained representations; they explain why the original parameterization starts from a dimension-independent logit scale.

A fully masked row has no probability distribution. Stable softmax implementations usually subtract the row maximum; for all-$-\infty$ input, that already creates $-\infty-(-\infty)$ and can propagate NaNs. An API must choose a policy: reject the mask, return a defined zero output for known padding rows, or supply a finite extra score with a zero value (the learned sink in §10). The notebook tests all three cases explicitly.

**<strong>Implementation pass: NumPy → PyTorch → API</strong>**

### NumPy → PyTorch · B04 — SDPA line by line

Assume validated, finite floating Q/K/V with nonempty axes and compatible shapes `[..., T_q, D]`, `[..., T_k, D]`, and `[..., T_k, D_v]`. Leading axes can represent batches and heads. API comparisons use a shared dtype and device, with a Boolean mask on that device broadcastable to the scores. This minimal example deliberately returns zero for an empty row; §12 provides the separate `error`/`zero` choice. NumPy exposes scores, masking, softmax, and the V reduction before the PyTorch API delegates execution.

```python
import math
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Explicit NumPy implementation

```python
def numpy_sdpa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    output_dtype = np.result_type(q.dtype, k.dtype, v.dtype)
    if output_dtype.kind not in "fc":
        output_dtype = np.dtype(np.float64)
    work_dtype = np.result_type(output_dtype, np.float32)
    q, k, v = (x.astype(work_dtype, copy=False) for x in (q, k, v))
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    if allowed_mask is None:
        row_has_key = np.ones_like(scores[..., :1], dtype=bool)
    else:
        mask = np.broadcast_to(np.asarray(allowed_mask, dtype=bool), scores.shape)
        row_has_key = mask.any(axis=-1, keepdims=True)
        scores = np.where(mask, scores, -np.inf)

    safe_scores = np.where(row_has_key, scores, 0.0)
    shifted = safe_scores - safe_scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=-1, keepdims=True)
    weights = np.where(row_has_key, weights, 0.0)
    return np.matmul(weights, v).astype(output_dtype, copy=False)
```

Every stage is visible in NumPy: $QK^\top$, the $1/\sqrt{d_h}$ scale, mask broadcasting, softmax over keys, and the $V$ reduction. `row_has_key` defines the fully masked-row policy: zero output.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_sdpa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    output_dtype = q.dtype
    work_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    q, k, v = (x.to(work_dtype) for x in (q, k, v))
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale

    if allowed_mask is None:
        row_has_key = torch.ones_like(scores[..., :1], dtype=torch.bool)
    else:
        mask = torch.broadcast_to(allowed_mask.to(torch.bool), scores.shape)
        row_has_key = mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~mask, -torch.inf)

    safe_scores = torch.where(row_has_key, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1)
    weights = torch.where(row_has_key, weights, torch.zeros_like(weights))
    return (weights @ v).to(output_dtype)
```

`Tensor.transpose`, `dim=-1`, and `masked_fill` express the same chain with autograd support. FP16/BF16 inputs are widened to FP32 before dot products, and the output returns to Q's dtype. All tensors remain on their shared device.

#### 3. Optimized or library PyTorch API

```python
def torch_sdpa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
    )
```

The one-line API hands the same operator to PyTorch's dispatcher, which may select a math, memory-efficient, FlashAttention, or another available backend.

| Aspect | NumPy | PyTorch |
|---|---|---|
| K transpose | `np.swapaxes` | `Tensor.transpose` |
| Softmax axis | `axis=-1` | `dim=-1` |
| Boolean mask | implemented explicitly | API uses `True` for allowed edges |
| Empty row | course policy: zero output | API behavior is guarded by regression tests |
| Dropout | none | evaluation must pass `dropout_p=0.0` |
| Backend | a sequence of NumPy operations | eager or fused kernel |

**<strong>Executable parity and edge-case check</strong>**

```python
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
```

Complete executable file: `m05_attention_bridges.py`

Official contract: [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 5. Worked storage arithmetic: four head layouts and one hybrid model

Keep the depth and head dimension fixed at 36 and 64, respectively, and vary only the number of KV heads:

| Layout | $H_{kv}$ | Group size $g$ | All-full bf16 cache per token |
|---|---:|---:|---:|
| MHA | 64 | 1 | 576 KiB |
| GQA | 8 | 8 | 72 KiB |
| MQA | 1 | 64 | 9 KiB |

At 131,072 tokens these become 72 GiB, 9 GiB, and 1.125 GiB. The first and third rows are hypothetical versions of the same depth and head size; they are not claims about released gpt-oss checkpoints.

Now restore the actual gpt-oss layer schedule. Each GQA layer needs $2\cdot8\cdot64\cdot2=2048$ bytes per retained position. If the runtime caps each local cache at 128 positions, 18 full layers cost $18\cdot131072\cdot2048=4.5$ GiB and 18 local layers cost $18\cdot128\cdot2048=4.5$ MiB. The total is **4.50439453125 GiB**. A window mask alone does not prove that old entries are freed.

For ordinary K/V, $B$ equal-length sequences and heterogeneous layers, use

$$M_{KV}=B\sum_{\ell=1}^L n_\ell H_{kv,\ell}(d_{k,\ell}+d_{v,\ell})b_\ell.$$

Here $n_\ell=N$ for a full layer and $n_\ell=\min(N,W_\ell)$ for a bounded local cache. Sum over sequences when their lengths differ. This is array payload: it excludes alignment, paging, replication, and quantization metadata. MLA stores a different object and needs its own formula.

In the educational DeepSeek-like MLA geometry, 61 layers retain a 512-dimensional latent and a 64-dimensional decoupled RoPE component:

$$
b_{\text{MLA}}=(512+64)\cdot61\cdot2=68.625\ \text{KiB/token}.
$$

A selected standard MHA baseline with 128 heads of width 128 requires

$$
b_{\text{MHA}}=2\cdot61\cdot128\cdot128\cdot2
=3.8125\ \text{MiB/token}.
$$

The raw storage ratio is 56.8889×. This number should not be confused with the DeepSeek-V2 paper's 93.3% system-level cache reduction against DeepSeek 67B; the baselines are different.

![VIZ m5/02 — cache footprints](assets/modern-llms/en/module-04/m5_02_kv_per_token.png)
*KV memory at 131,072 positions: all-full MHA/GQA/MQA, the gpt-oss layout with window-capped local caches, and a separate MLA geometry.*

## 6. GQA is a grouping invariant, not a quality guarantee

Let $g=H_q/H_{kv}$ be an integer group size. Query head $h$, counted from zero, uses KV head $j(h)=\lfloor h/g\rfloor$, so

$$
O_h=\operatorname{softmax}\left(Q_hK_{j(h)}^\top/\sqrt{d_h}+M\right)V_{j(h)}.
$$

For $H_q=64$ and $H_{kv}=8$, one KV head therefore serves eight consecutive query heads. An implementation may expose this by reshaping Q to

```text
[tokens, H_kv, H_q/H_kv, head_dim]
```

while K and V have shape

```text
[tokens, H_kv, head_dim].
```

When using an MHA reference implementation, each KV head must be repeated consecutively inside its group:

```text
kv0, kv0, ..., kv0, kv1, kv1, ..., kv1.
```

Repeating the complete KV block instead produces a tensor of the right shape with the wrong head assignment. This is a classic silent bug and is covered by a negative notebook test.

Functionally, correct GQA equals MHA with repeated K/V heads. Different fused kernels can still differ by floating-point rounding, so bitwise equality is not a portable promise.

The architectural tradeoff is straightforward. Fewer KV heads reduce cache and memory bandwidth. More KV heads provide more distinct key/value subspaces. GQA offers discrete choices between those endpoints; it does not make group size eight a theorem.

**<strong>Implementation pass: NumPy → PyTorch → API</strong>**

### NumPy → PyTorch · B05 — GQA: repeating K/V and `enable_gqa`

This block fixes four axes: Q `[B,H_q,T_q,D]`, K `[B,H_kv,T_k,D]`, and V `[B,H_kv,T_k,D_v]`. Heads occupy axis 1, and `H_q` is divisible by positive `H_kv`. The finite-value, shared dtype/device, and Boolean-mask assumptions from B04 still apply. NumPy repeats K/V heads explicitly; supported PyTorch backends express grouping with `enable_gqa=True`. Section 12's flattened `[T,H*D]` layout is a separate interface: reshape and transpose it before using these functions.

```python
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Explicit NumPy implementation

```python
def numpy_gqa_explicit(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = np.repeat(k, group, axis=1)
    v_repeated = np.repeat(v, group, axis=1)
    return numpy_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)
```

GQA does not change the attention equation. NumPy explicitly repeats every K/V head `g = H_q/H_{kv}` times and then reuses the SDPA reference.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_gqa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    group = q.shape[1] // k.shape[1]
    k_repeated = k.repeat_interleave(group, dim=1)
    v_repeated = v.repeat_interleave(group, dim=1)
    return torch_sdpa_explicit(q, k_repeated, v_repeated, allowed_mask=allowed_mask)
```

The explicit PyTorch path uses `repeat_interleave`. It is a useful shape and head-order reference, but it materializes repeated K/V tensors.

#### 3. Optimized or library PyTorch API

```python
def torch_gqa_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed_mask,
        dropout_p=dropout_p,
        enable_gqa=True,
    )
```

`enable_gqa=True` expresses grouping without explicit user-code repetition; availability and efficiency depend on the backend.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Requirement | `Hq % Hkv == 0` | same |
| K/V | `np.repeat(..., axis=1)` | `repeat_interleave(..., dim=1)` |
| Reference memory | materialized repeated arrays | materialized repeated tensors |
| Optimized path | none | `enable_gqa=True` |
| Test | matches MHA after repetition | explicit path ↔ API |

**<strong>Executable parity and edge-case check</strong>**

```python
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
```

Complete executable file: `m05_attention_bridges.py`

Official contract: [PyTorch SDPA GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 7. MLA: cache a shared latent and move the projections

**Multi-head Latent Attention (MLA)** retains a compact vector $c_t=W^{DKV}h_t$ shared across heads, together with a small positional component. Here a **latent** is an intermediate representation from which linear maps produce key and value features. Why can inference avoid reconstructing the full K/V cache?

For one head, write $k_t=A c_t$ and $v_t=B c_t$, with $c_t\in\mathbb R^{d_c}$, $A\in\mathbb R^{d_k\times d_c}$, and $B\in\mathbb R^{d_v\times d_c}$. Then

$$q^\top k_t=q^\top A c_t=(A^\top q)^\top c_t.$$

Moving $A$ to the query side produces one transformed query that can be compared with every cached latent. Once attention weights $p_t$ are available, value computation can be rearranged too:

$$\sum_t p_t v_t=B\left(\sum_t p_t c_t\right).$$

If an output projection $W^O$ follows, the product $W^OB$ can be combined in advance. For multiple heads, apply the argument to each head's $A_h$, $B_h$, and corresponding output-projection block. This **matrix absorption** changes the stored representation and dimensions of the computation; the remaining projections still have a cost.

**Where does position enter?** RoPE is linear at a fixed position. But rotating reconstructed keys gives, for query position $m$ and key position $n$,

$$(R_mq)^\top R_n A c_n=q^\top R_{n-m}A c_n.$$

The effective query $A^\top R_{n-m}^\top q$ now depends on $n$. It cannot be computed once and reused for all keys as before. Comparing $RA$ with $AR$ would also be ill-formed in general: $A$ is rectangular.

DeepSeek-V2 therefore uses a separate small RoPE branch. Content and positional terms are added to form the logit. Content projections can still be absorbed, while the positional key is cached alongside the latent. The notebook verifies both score terms and the projected output. This is an algebraic equivalence; model quality requires separate empirical evaluation. [DeepSeek-V2, §2.1](https://arxiv.org/html/2405.04434v5).

![VIZ m5/04 — computing in latent space](assets/modern-llms/en/module-04/m5_04_mla_latent.svg)
*Key expansion moves to the query side; value expansion moves after aggregation. The positional branch remains separate.*

## 8. FlashAttention: dense attention without the quadratic HBM object

For complexity estimates here, take equal key/value width $d=d_h=d_v$. Write $P$ for the matrix of attention probabilities.

A materializing implementation often writes the score matrix, reads it for softmax, writes probabilities, and reads them again for the value product. The arithmetic is not the only cost; the memory path is expensive.

FlashAttention tiles Q, K, and V so that the active blocks fit in on-chip **SRAM—Static Random-Access Memory**. Each query row carries three online-softmax quantities across key tiles:

- the running maximum $m$;
- the running normalizer $\ell$;
- the running value numerator or output accumulator $n$.

When a later tile raises the maximum, earlier state is multiplied by $e^{m_{old}-m_{new}}$. The final output is $n/\ell$.

For one query, let $x_j$ be scores of allowed keys in the new tile and $v_j$ their values. Update the running state as follows:

$$m'=\max(m,\max_j x_j),\quad \alpha=e^{m-m'},\quad \widetilde p_j=e^{x_j-m'},$$
$$\ell'=\alpha\ell+\sum_j\widetilde p_j,\qquad n'=\alpha n+\sum_j\widetilde p_jv_j,\qquad O=n'/\ell'.$$

Initialize $m=-\infty$, $\ell=0$, $n=0$. Set $\alpha=0$ for the first nonempty tile and skip fully masked tiles. If every tile is empty, apply the explicit policy from §4. This avoids $-\infty-(-\infty)$. Section 18 and the notebook work through numerical updates.

A linear-memory claim must include **all** arrays: prebuilding a dense $N\times N$ mask keeps a quadratic allocation. Our streaming code can generate a causal mask for each tile. It is a transparent CPU reference, not a performance implementation of FlashAttention.

![VIZ m5/05 — online softmax](assets/modern-llms/en/module-04/m5_05_online_softmax.svg)
*An online softmax row carries its maximum, normalization sum, and value accumulator across tiles.*

It is important to state the complexity precisely.

| Property | Materializing dense attention | FlashAttention |
|---|---|---|
| Dense pairwise arithmetic | $O(N^2d)$ | $O(N^2d)$ |
| Full $N\times N$ object in HBM | yes | no |
| Additional stored memory | quadratic | linear in sequence length |
| HBM traffic | high | substantially reduced; depends on SRAM/tile geometry |

The IO analysis is parameterized by fast-memory capacity $M$; it is not a universal reduction to $O(Nd)$ traffic. FlashAttention is called exact because it computes every dense interaction rather than selecting a subset. Floating-point order and kernel approximations may still cause small numerical differences from a naive reference.

FlashAttention-4 reports up to 1,613 TFLOP/s in BF16 on B200, approximately 71% of peak, with up to 1.3× speedup over cuDNN 9.13 and 2.7× over Triton in the authors' configurations. [Primary paper](https://arxiv.org/abs/2603.05451).

![VIZ m5/06 — tiled IO](assets/modern-llms/en/module-04/m5_06_tiling_memory.svg)
*Tiling removes the quadratic HBM intermediate, not the quadratic dense-attention arithmetic.*

## 9. Windowed attention and layer schedules

A sliding-window layer allows each query to see only the most recent $W$ keys. Its decode cache can stop growing after $W$ positions if old storage is discarded or reused. A stack of local layers can propagate information farther than one window; the reach discussed below is a dependency bound. It does not guarantee that a fact survives many transformations or match a direct global edge.

Pure windowed decoder stacks do exist. Mistral-7B-v0.1 with window 4096 is a documented example. Hybrid models add global layers to shorten long-range paths.

| Model family | Published layout |
|---|---|
| Mistral-7B-v0.1 | sliding-window attention throughout the decoder |
| gpt-oss-120b | 18 window-128 and 18 full layers, alternating |
| Gemma 2 | window-4096 and full-attention layers interleaved 1:1 within an 8192-token context |
| Gemma 4 | hybrid local/global attention; final layer global |

This is why a cache estimate must be layer-aware. A global `sliding_window` field or a single context length does not reveal how many layers retain which positions.

![VIZ m5/03 — hybrid schedules](assets/modern-llms/en/module-04/m5_03_layer_alternation.svg)
*Local and global layers create different dependency paths and different cache-growth laws.*

The window includes the current position. With $L$ consecutive local layers, equal window $W$, and no dilation, the furthest dependency is $L(W-1)$ positions back. Bounded cache storage additionally requires eviction or reuse; a mask by itself does not bound allocation. Check the checkpoint version: later Mistral releases need not share v0.1’s window policy.

## 10. “Attention sink” refers to two different objects

StreamingLLM observed that trained models often place large attention mass on initial real tokens even when those tokens are not semantically important. Dropping their K/V state from a rolling window can destabilize the model. Keeping a few initial tokens alongside the recent window restores behavior. This is an empirical property of existing attention maps. [StreamingLLM](https://arxiv.org/abs/2309.17453).

gpt-oss implements a different mechanism. Each query head owns a learned scalar $s_h$. The scalar is appended as an extra score column before softmax. After normalization, that column is removed before multiplying by V, so it receives probability mass but contributes no value vector.

For 128 token scores all equal to zero,

$$
p_{sink}=\frac{e^{s_h}}{128+e^{s_h}}.
$$

At $s_h=0,2,5$, the sink receives roughly 0.78%, 5.46%, and 53.69% of the mass.

A learned constant score is not equivalent to an ordinary fixed key. A key would produce $q^\top k_{sink}$ and vary with the query. The exact abstraction is an extra score slot with zero value. It could be represented through dot-product attention only by extending the space with a constant coordinate or by adding kernel support.

![VIZ m5/08 — two sinks](assets/modern-llms/en/module-04/m5_08_sinks.svg)
*StreamingLLM preserves initial real K/V entries; gpt-oss adds a learned score slot without a value contribution.*

A useful check: shifting only token logits while leaving the sink fixed changes the output. Softmax shift invariance requires shifting every logit, including the sink.

## 11. Sparse attention: selecting a subset of positions

A window selects keys by distance. **Content-based selection** estimates relevance from query and key features. Given a selected set $I(q)$, the main attention operation is

$$O(q)=\operatorname{softmax}\left(qK_{I(q)}^\top/\sqrt{d_h}\right)V_{I(q)}.$$

This is exact attention on the selected subset, but generally differs from attention on every visible position. A cheaper selector can miss information the main operation needs.

| Mechanism | Selection unit | Rule and conditions |
|---|---|---|
| Fixed sparse mask | blocks or positions | Predetermined pattern; quality depends on the model and task |
| MoBA — Mixture of Block Attention | K/V blocks | Query dot mean block key, followed by top-block selection |
| NSA — Native Sparse Attention | compressed summaries, selected blocks, local window | Jointly trained branches designed for hardware execution |
| DSA — DeepSeek Sparse Attention | tokens | A trained lightweight indexer followed by *top-k*: keep the $k$ highest-scoring positions |

**MoBA adds no separate trainable router parameters.** Its block score uses existing Q/K representations, whose projections learn with the model. The current block is always included with a causal mask; future blocks are excluded. Parameter-free gating does not guarantee that swapping attention in an arbitrary checkpoint preserves quality: the paper studies training and hybrid schedules. [MoBA, §2.2](https://arxiv.org/html/2502.13189v1).

**NSA** combines compression, block selection, and a local window. The compressed branch provides a coarse view, selected blocks supply detail, and the window covers nearby context. The branches learn together; reported gains belong to the authors' models and kernels. [NSA](https://arxiv.org/abs/2502.11089).

**DSA** debuted in DeepSeek-V3.2-Exp on September 29, 2025 and continued in DeepSeek-V3.2. The indexer is a smaller relevance-scoring mechanism that proposes positions for the more expensive main attention. Its computation, memory, and training must be counted too. An indexer that scans every key may still perform quadratic prefill work, even when each comparison is cheap. [V3.2 technical report](https://arxiv.org/abs/2512.02556).

For one query, $N$ keys, main comparison width $d$, and indexer width $r$, a simple two-stage dot-product count is $Nr+kd$. Relative to $Nd$, this is $r/d+k/N$, **not just $k/N$**. It still excludes indexer projections, top-k selection, softmax, value aggregation, and data movement. The project demo separates shortlist recall, final retrieval, and attention-output error.

As of October 10, 2026, this direction also includes compressing several positions into one cache entry. DeepSeek-V4 combines **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**. The DeepSeek-V4.1-Flash report adds CSA2, cross-layer KV reuse, and FP4 caching. These combine storage, selection, and memory placement in new ways; our MLA byte counts do not transfer to them. A claim of subquadratic execution needs to cover the whole pipeline. [V4 developer model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [V4.1-Flash report](https://arxiv.org/abs/2609.19969).

## 12. A transparent NumPy reference

The reference uses an additive mask: 0 allows a pair and `-inf` forbids it. It makes shapes and empty-row behavior explicit on small inputs. Section 13 and the notebook add streaming execution and MLA experiments.

**<strong>Open the SDPA and GQA implementation</strong>**

```python
import math
from typing import Literal
import numpy as np

EmptyRowPolicy = Literal["error", "zero"]

def make_mask(n: int, window: int | None = None, n_sink: int = 0) -> np.ndarray:
    """Return an additive causal mask of shape [n, n].

    Allowed entries are 0 and forbidden entries are -inf. `window` keeps only the
    latest `window` causal keys; `n_sink` preserves initial real positions as StreamingLLM-style anchors.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive")
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
        raise ValueError("window must be positive")
    if isinstance(n_sink, bool) or not isinstance(n_sink, int) or not 0 <= n_sink <= n:
        raise ValueError("n_sink must satisfy 0 <= n_sink <= n")
    if n_sink and window is None:
        raise ValueError("n_sink is meaningful only together with a finite window")

    query_pos = np.arange(n)[:, None]
    key_pos = np.arange(n)[None, :]
    allowed = key_pos <= query_pos
    if window is not None:
        allowed &= (query_pos - key_pos) < window
        allowed |= (key_pos < n_sink) & (key_pos <= query_pos)
    return np.where(allowed, 0.0, -np.inf)

def sdpa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    sink: float | np.ndarray | None = None,
    empty_row: EmptyRowPolicy = "error",
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute transparent scaled dot-product attention.

    Shapes:
        Q [..., T_q, d_k], K [..., T_k, d_k], V [..., T_k, d_v].
        `mask` must broadcast to [..., T_q, T_k]. A finite entry is an additive
        bias; -inf forbids the key. A per-head sink should be reshaped so it
        broadcasts to [..., T_q, 1].

    `empty_row="error"` rejects a row with no key and no sink. `"zero"` returns
    zero weights/output for that row, which is useful for explicit padding rows.
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")

    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if min(Q.ndim, K.ndim, V.ndim) < 2 or any(0 in x.shape for x in (Q, K, V)):
        raise ValueError("Q, K, and V must each have at least two dimensions")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same d_k")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same T_k")
    try:
        np.broadcast_shapes(Q.shape[:-2], K.shape[:-2], V.shape[:-2])
    except ValueError as exc:
        raise ValueError("Q/K/V batch dimensions are not broadcast-compatible") from exc
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q, K, and V must contain only finite values")

    # Keep float32/float64 inputs in their precision; integer examples become float64.
    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)
    Q, K, V = (x.astype(dtype, copy=False) for x in (Q, K, V))
    d_k = Q.shape[-1]
    scale = np.asarray(1.0 / math.sqrt(d_k), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    if not np.isfinite(scores).all():
        raise FloatingPointError("Q @ K.T overflowed; rescale the inputs")

    if mask is not None:
        mask = np.asarray(mask, dtype=dtype)
        if np.isnan(mask).any() or np.isposinf(mask).any():
            raise ValueError("mask may contain finite biases and -inf, but not NaN/+inf")
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError(f"mask shape {mask.shape} does not broadcast to {scores.shape}") from exc
        finite_bias = np.isfinite(mask)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = scores + mask
        if np.any(finite_bias & ~np.isfinite(scores)):
            raise FloatingPointError(
                "adding the finite mask bias overflowed; reduce the bias magnitude"
            )

    has_token = np.isfinite(scores).any(axis=-1, keepdims=True)
    token_max = np.max(scores, axis=-1, keepdims=True)

    sink_logits = None
    if sink is not None:
        sink_logits = np.asarray(sink, dtype=dtype)
        if not np.isfinite(sink_logits).all():
            raise ValueError("sink logits must be finite")
        try:
            sink_logits = np.broadcast_to(sink_logits, token_max.shape)
        except ValueError as exc:
            raise ValueError(
                f"sink shape {sink_logits.shape} does not broadcast to {token_max.shape}"
            ) from exc
        row_max = np.maximum(token_max, sink_logits)
    else:
        if empty_row == "error" and not has_token.all():
            bad = np.argwhere(~has_token.squeeze(-1))
            raise ValueError('Attention contains fully masked rows' + f": {bad[:8].tolist()}")
        # A zero-policy row gets a harmless max of 0; all its exponentials stay 0.
        row_max = np.where(has_token, token_max, np.zeros_like(token_max))

    token_exp = np.where(
        np.isfinite(scores), np.exp(scores - row_max), np.zeros_like(scores)
    )
    denominator = token_exp.sum(axis=-1, keepdims=True)
    if sink_logits is not None:
        denominator += np.exp(sink_logits - row_max)

    weights = np.divide(
        token_exp,
        denominator,
        out=np.zeros_like(token_exp),
        where=denominator > 0,
    )
    output = weights @ V
    return (output, weights) if return_weights else output

def gqa(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Educational grouped-query attention for 2D projected sequences.

    Q [T, H*d_k], K [T, H_kv*d_k], V [T, H_kv*d_v].
    The output has shape [T, H*d_v].
    """
    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if any(x.ndim != 2 or 0 in x.shape for x in (Q, K, V)):
        raise ValueError("GQA expects 2D projected sequences")
    if not (Q.shape[0] == K.shape[0] == V.shape[0]):
        raise ValueError("Q, K, and V must have the same sequence length")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in (n_heads, n_kv_heads)) or n_heads % n_kv_heads:
        raise ValueError("n_heads must be a positive multiple of n_kv_heads")
    if Q.shape[1] % n_heads:
        raise ValueError("Q width must be divisible by n_heads")
    if K.shape[1] % n_kv_heads or V.shape[1] % n_kv_heads:
        raise ValueError("K/V widths must be divisible by n_kv_heads")

    T = Q.shape[0]
    d_k = Q.shape[1] // n_heads
    if K.shape[1] // n_kv_heads != d_k:
        raise ValueError("Q and K head dimensions must match")
    d_v = V.shape[1] // n_kv_heads

    q = Q.reshape(T, n_heads, d_k).transpose(1, 0, 2)
    k = K.reshape(T, n_kv_heads, d_k).transpose(1, 0, 2)
    v = V.reshape(T, n_kv_heads, d_v).transpose(1, 0, 2)
    group_size = n_heads // n_kv_heads
    # repeat preserves groups [kv0, kv0, ..., kv1, kv1, ...].
    k = np.repeat(k, group_size, axis=0)
    v = np.repeat(v, group_size, axis=0)
    z = sdpa(q, k, v, mask)
    return z.transpose(1, 0, 2).reshape(T, n_heads * d_v)
```

`n_sink` in `make_mask` retains initial real positions. `sink` in `sdpa` adds a logit with a zero value. The tests treat them separately.

## 14. Reading gpt-oss configuration and code

Use a reproducible external example: **gpt-oss-120b** configuration revision `b5c939de…` and the `openai/gpt-oss` reference code at `7b583341…`. The data notes provide full revisions and SHA-256 hashes. M5 owns its configuration snapshot, so edits to neighbouring modules cannot alter the experiment.

**Head grouping.** Queries have shape `[tokens, H_kv, H_q/H_kv, head_dim]`; keys and values have shape `[tokens, H_kv, head_dim]`. Here 64 query heads share eight K/V sets, eight queries per group.

**Layer masks.** `AttentionBlock` gives even indices, starting at zero, a window of 128. Odd indices use full causal attention: 18 local and 18 full layers in the 36-layer model.

**Learned sink.** Each query head has one scalar in `sinks`. The `sdpa` function appends it to the scores, normalizes the whole row, and removes the final column before multiplying by V. This implements §10. RoPE acts on Q/K before comparison.

**What this code establishes.** The reference PyTorch function materializes a full score matrix and does not implement a bounded rolling KV cache. The mask tells us which keys a layer needs; the serving system must discard or reuse old storage. Thus **4.504 GiB is a K/V payload calculation assuming window-capped local storage**, not the measured peak allocation of this function. The notebook checks the configuration and arithmetic; a separate validator compares the sink/GQA formula with the pinned upstream function.

![VIZ m5/09 — from configuration to cache accounting](assets/modern-llms/en/module-04/m5_09_config_attention.svg)
*Configuration supplies dimensions, code supplies grouping and masks, and the storage policy determines actual allocation.*

## 15. The landscape on October 10, 2026

This table is a map of mechanisms. Examples identify a version or flag variation within a family. Sharing an attention label does not imply equal memory use or speed.

| Example | Retained representation | Visible positions and conditions |
|---|---|---|
| Llama 3.x | GQA | Full causal attention; head counts vary with model size |
| Mistral-7B-v0.1 | GQA | Window 4096 throughout the stack; specific to this version |
| gpt-oss-120b | GQA, eight KV heads | Alternating window-128 and full attention; learned sink |
| Gemma 2 / Gemma 4 | Variant-dependent dimensions | Local and global layers; Gemma 4 ends with a global layer |
| DeepSeek-V2/V3 | MLA | Shared latent and separate positional component |
| DeepSeek-V3.2 | MLA plus DSA indexer | Main attention over selected tokens; include indexer costs |
| DeepSeek-V4 / V4.1-Flash | Compressed entries; V4.1 also reuses KV across layers | Compression, selection, and local mechanisms combine; derive storage for the specific format |
| FlashAttention-4 | Does not define a model's KV representation | Execution strategy; mask and precision support are kernel-dependent |

These rows draw on public configurations and reports: [Llama 3](https://arxiv.org/abs/2407.21783), [Mistral v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json), [Gemma 2](https://arxiv.org/abs/2408.00118), [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4). For gpt-oss see §14; for MLA, DSA, V4, and FlashAttention see §§7–11.

When reading a new architecture, record **what is stored, how many positions each layer needs, how positions are selected, and how the computation runs**. Add numerical formats, sequence batch size, and cache placement. Context length then becomes part of a testable resource estimate.

## 16. Multimodal attention: where image information is stored

With **early fusion**, visual representations join the language decoder's sequence. Those positions inherit the decoder's attention mechanism and KV format.

Consider a hypothetical decoder with full attention in every layer and a K/V payload of 72 KiB per position. For one sequence, 4,096 additional visual positions require

$$4096\cdot72\text{ KiB}=288\text{ MiB}.$$

This counts only the added K/V, excluding weights, vision-encoder activations, and temporary arrays. Windowed layers require a separate count of visual positions still retained in each window.

With **cross-attention**, a vision encoder produces a separate feature sequence that language queries attend to. The encoder need not run again at every language layer. However, different cross-attention layers may project and cache their own visual K/V, so those costs must be counted separately.

The connection between modalities determines both projection shapes and where state is stored. Module 16 develops these architectures further.

## 19. Key takeaways

![VIZ m5/10 — attention on one page](assets/modern-llms/en/module-04/m5_10_cheatsheet.svg)
*Four independent axes form an attention passport: cache geometry, layer schedule, position selection, and execution kernel.*

- **Cache arithmetic is layer-specific.** $2LH_{kv}d_hb$ counts bytes per retained position across identical layers; sum heterogeneous layers using their retained lengths.
- **MQA and GQA reduce KV heads.** A common group ratio is not a universal quality optimum.
- **MLA changes the cached object.** A raw element ratio and a paper's end-to-end cache reduction are different comparisons.
- **FlashAttention does not linearize dense attention.** It eliminates the quadratic HBM intermediate and reduces IO.
- **Windowed layers permit bounded local state if storage is reused or evicted.** Depth creates possible long paths, not guaranteed global recall.
- **Attention sink has two meanings.** Initial real-token sinks and learned score slots must be distinguished.
- **Sparse attention selects positions.** Selection cost, training requirements, and evidence status matter as much as asymptotics.
- **Context length is not an attention passport.** State, connectivity, dtype, workload, and implementation determine cost.

**Next: M6.** Attention mixes information across positions. We next examine how normalization, nonlinearities, and residual connections control its scale and transformation within a block.

## 20. primary sources

**Primary sources:**

- Vaswani et al., *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shazeer, MQA — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., GQA — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- DeepSeek-V2 / MLA — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- FlashAttention — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- FlashAttention-4 — [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
- Mistral 7B — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- StreamingLLM — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- MoBA — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- NSA — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- DSA / DeepSeek-V3.2-Exp — [official release](https://api-docs.deepseek.com/news/news250929)
- DeepSeek-V3.2 — [technical report](https://arxiv.org/abs/2512.02556)
- DeepSeek-V4 — [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- DeepSeek-V4.1-Flash — [technical report](https://arxiv.org/abs/2609.19969)
- gpt-oss — [official repository](https://github.com/openai/gpt-oss)

**Further course connections.** KV paging, offloading, and prefill/decode disaggregation are developed in Module 10. Multimodal attention placement is developed in Module 16.

---

*Landscape verified: 10 October 2026. Memory figures are reproducible calculations for stated layouts; external performance numbers retain their authors' hardware and protocol.*
