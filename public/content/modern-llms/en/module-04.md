# Module 4. Modern Attention Mechanisms

*“Modern LLMs” course · Module 4 lecture · edition 2026.8*

> **What this module is about.** Attention is not only a formula for mixing tokens. It is also a storage format, a memory-access pattern, and—once context becomes long—a routing decision. This lecture follows two resource bills created by autoregressive attention. The first is the KV state retained from previous tokens. The second is the movement of scores and activations through the GPU memory hierarchy. MQA, GQA, and MLA reduce the state carried by each position. FlashAttention preserves dense attention while changing its IO schedule. Windowed and sparse designs reduce the positions that participate in a full interaction. The final goal is an attention passport: enough information to estimate serving behavior from architecture rather than from context length alone.
>
> **Prerequisites.** Softmax, matrix multiplication, causal masking, and RoPE from Modules 3 and 18. GPU memory terminology is introduced when it becomes useful.

---

## 1. Motivation: two bills hidden behind one decode step

A decoder has already processed the prompt. It now receives one new token and must compare its query with every allowed key. Recomputing past key and value projections would waste work, so the model retains them in a **KV cache—Key–Value cache**.

For an ordinary attention layout, one token adds

$$
b_{KV}=2L H_{kv}d_hb
$$

bytes, where $L$ is the number of cache-bearing layers, $H_{kv}$ the number of key/value heads, $d_h$ the head dimension, and $b$ the bytes per scalar. The factor of two accounts for both K and V.

This formula is deliberately local: it says what one token would cost if all listed layers retained that position. A real model may mix full-context and local layers, so multiplying it by the advertised context length can be wrong.

The open gpt-oss-120b implementation is a useful example. With 36 layers, eight KV heads, head dimension 64, and bf16 state, the all-full equivalent is

$$
2\cdot36\cdot8\cdot64\cdot2
=73{,}728\ \text{bytes}
=72\ \text{KiB/token}.
$$

At 131,072 tokens, that hypothetical layout is 9 GiB. The actual model alternates 18 full layers with 18 window-128 layers. Per layer, one retained position costs 2,048 bytes; therefore

$$
18\cdot131{,}072\cdot2{,}048=4.5\ \text{GiB},
$$

while the local half contributes only

$$
18\cdot128\cdot2{,}048=4.5\ \text{MiB}.
$$

The hybrid total is approximately **4.504 GiB**. That factor-of-two difference comes from the layer schedule, not from a new data type.

The second bill appears during dense attention itself. A single fp32 score matrix of shape $8192\times8192$ occupies 256 MiB per head. If an implementation writes scores and probabilities to high-bandwidth memory and reads them back, data movement can dominate the arithmetic. That is the problem FlashAttention addresses.

The module therefore asks two separate questions: **what state must survive between steps, and how should the current attention operation travel through memory?**

## 2. The historical split: compress the cache or reorganize the kernel

![VIZ m4/01 — two attention lineages](assets/modern-llms/en/module-04/m4_01_attention_timeline.svg)

The original Transformer used **MHA—Multi-Head Attention**: every query head had its own key and value projections. This maximizes head-specific state and also maximizes the cache.

**MQA—Multi-Query Attention** retained one key/value head shared by all query heads. It gave decoder inference a much smaller cache, but some experiments observed a quality cost from sharing too aggressively.

**GQA—Grouped-Query Attention** made the number of KV heads a continuum. Let $H_q$ be query heads, $H_{kv}$ KV heads, and

$$
g=H_q/H_{kv}
$$

be the query heads served by one KV head. MHA sits at $g=1$; MQA sits at $H_{kv}=1$. The GQA paper showed that its uptrained models achieved quality close to MHA with speed comparable to MQA in the reported setup. That finding motivates the design; it does not establish a universal best group size. [Primary paper](https://arxiv.org/abs/2305.13245).

**MLA—Multi-head Latent Attention** changed the cached object. DeepSeek-V2 stored a compact joint KV latent plus a small positional component rather than full per-head K/V. The authors reported a 93.3% KV-cache reduction relative to DeepSeek 67B and up to 5.76× maximum generation throughput in their system comparison. [Technical report](https://arxiv.org/abs/2405.04434).

A parallel history concerned execution. **FlashAttention** tiled dense attention into on-chip memory and maintained softmax statistics online. It did not select fewer keys; it stopped materializing the full score/probability matrices in high-bandwidth memory. [Primary paper](https://arxiv.org/abs/2205.14135).

A third family eventually changed connectivity itself. Sliding-window attention chooses positions geometrically. StreamingLLM preserves initial real tokens because they act as sinks. MoBA, NSA, and DSA learn or compute content-dependent sparse selections. Those methods answer a different question from GQA or MLA: not how many coordinates to store per token, but which tokens deserve an expensive interaction.

## 3. Three useful classical views—and where they stop

### Attention as kernel regression

Nadaraya–Watson regression computes

$$
\hat f(q)=\frac{\sum_i K(q,k_i)v_i}{\sum_i K(q,k_i)}.
$$

Softmax attention has the same structural shape: positive similarities are normalized and used to average values. The analogy is valuable, but attention is not a fixed nonparametric estimator. Q, K, and V are jointly learned; heads share a residual stream; and later layers transform the result. We should therefore read this as a lens, not a definition.

### Attention as a Gibbs distribution

For scores $s_i$,

$$
p_i\propto e^{s_i/T}.
$$

If we speak in the language of energy, then $E_i=-s_i$, because the standard Gibbs form is $e^{-E_i/T}$. The entropy of the distribution measures how concentrated the head is, but its behavior also depends on the learned score geometry.

### Attention as retained state

During decode, the model needs enough information about the past to form future attention outputs. MHA stores every head's K/V. GQA stores fewer shared heads. MLA stores a lower-dimensional latent. Recurrent or state-space mechanisms may replace position-indexed cache entries with a fixed-dimensional state. The cache is thus an architectural statement about what the model treats as sufficient history.

These views prepare the two engineering themes ahead: controlling the scale of logits, and controlling the size and placement of retained state.

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

A causal mask permits $j\le i$. A sliding mask further limits the distance between the query and key. Cross-attention may have different query and key lengths.

The $1/\sqrt{d_h}$ factor comes from a variance calculation. Under the idealized assumption that query and key components are independent, zero-mean, and unit-variance,

$$
\operatorname{Var}(q^\top k)=d_h.
$$

Without scaling, the typical logit magnitude grows as $\sqrt{d_h}$ and softmax becomes increasingly saturated. The assumptions are not a permanent description of trained representations; they explain why the original parameterization starts from a dimension-independent logit scale.

A fully masked row has no probability distribution. Stable softmax implementations usually subtract the row maximum; for all-$-\infty$ input, that already creates $-\infty-(-\infty)$ and can propagate NaNs. An API must choose a policy: reject the mask, return a defined zero output for known padding rows, or supply a finite sink score. The notebook tests all three cases explicitly.

### NumPy → PyTorch · B04 — SDPA line by line

The NumPy path exposes scores, masking, softmax, and the V reduction; the public PyTorch API may select an optimized kernel.

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
    return np.matmul(weights, v)
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
    return weights @ v
```

The mathematics is unchanged. `Tensor.transpose`, `dim=-1`, and `masked_fill` express the same chain while preserving device/dtype and supporting autograd.

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

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

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
np.testing.assert_array_equal(np_empty[..., 0, :], 0.0)
torch.testing.assert_close(t_empty[..., 0, :], torch.zeros_like(t_empty[..., 0, :]))

explicit_out.sum().backward()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
print("B04 explicit NumPy / explicit PyTorch / optimized API: PASS")
```

</details>

Complete executable file: [`m04_attention_bridges.py`](../assets/m04_attention_bridges.py)

Official contract: [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 5. Worked storage arithmetic: four head layouts and one hybrid model

Keep the depth and head dimension fixed at 36 and 64, respectively, and vary only the number of KV heads:

| Layout | $H_{kv}$ | Group size $g$ | All-full bf16 cache per token |
|---|---:|---:|---:|
| MHA | 64 | 1 | 576 KiB |
| GQA | 8 | 8 | 72 KiB |
| MQA | 1 | 64 | 9 KiB |

At 131,072 tokens these become 72 GiB, 9 GiB, and 1.125 GiB. The first and third rows are hypothetical versions of the same depth and head size; they are not claims about released gpt-oss checkpoints.

The actual gpt-oss schedule cuts the GQA figure in roughly half because only 18 layers retain the full sequence. Its 18 local layers retain 128 positions. The total is 4.5043945 GiB.

MLA needs a separate formula. In the educational DeepSeek-like geometry, 61 layers retain a 512-dimensional latent and a 64-dimensional decoupled RoPE component:

$$
b_{\text{MLA}}=(512+64)\cdot61\cdot2=68.625\ \text{KiB/token}.
$$

A selected standard MHA baseline with 128 heads of width 128 requires

$$
b_{\text{MHA}}=2\cdot61\cdot128\cdot128\cdot2
=3.8125\ \text{MiB/token}.
$$

The raw storage ratio is 56.8889×. This number should not be confused with the DeepSeek-V2 paper's 93.3% system-level cache reduction against DeepSeek 67B; the baselines are different.

![VIZ m4/02 — cache footprints](assets/modern-llms/en/module-04/m4_02_kv_per_token.png)

## 6. GQA is a grouping invariant, not a quality guarantee

For $H_q=64$ and $H_{kv}=8$, one KV head serves eight query heads. An implementation may expose this by reshaping Q to

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

The architectural tradeoff is straightforward. Fewer KV heads reduce cache and memory bandwidth. More KV heads provide more distinct key/value subspaces. GQA offers a continuum between those endpoints; it does not make group size eight a theorem.

### NumPy → PyTorch · B05 — GQA: repeating K/V and `enable_gqa`

The NumPy reference repeats K/V heads explicitly. PyTorch can express the same contract through `enable_gqa=True` when the backend supports it.

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

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
rng = np.random.default_rng(5)
q_np = rng.standard_normal((1, 4, 3, 8))
k_np = rng.standard_normal((1, 2, 3, 8))
v_np = rng.standard_normal((1, 2, 3, 7))
np_out = numpy_gqa_explicit(q_np, k_np, v_np)
q, k, v = (torch.tensor(a, dtype=torch.float64) for a in (q_np, k_np, v_np))
explicit = torch_gqa_explicit(q, k, v)
api = torch_gqa_optimized(q, k, v, dropout_p=0.0)
torch.testing.assert_close(explicit, torch.from_numpy(np_out), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
print("B05 explicit repetition / enable_gqa: PASS")
```

</details>

Complete executable file: [`m04_attention_bridges.py`](../assets/m04_attention_bridges.py)

Official contract: [PyTorch SDPA GQA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

## 7. MLA stores a joint latent and absorbs linear maps

MLA begins with a down-projection

$$
c_t^{KV}=W^{DKV}h_t.
$$

Head-specific key and value components can then be derived from this latent. At serving time, the model retains the compact $c_t^{KV}$ rather than a full K/V vector for every head.

The important implementation trick is **matrix absorption**. Linear up-projections can be composed with neighboring query and output projections, avoiding explicit materialization of reconstructed K/V tensors. This does not make the work free: latent projections and matrix multiplications remain. It changes which intermediate representation has to be stored.

RoPE complicates the algebra. For a fixed position it is a linear orthogonal transform, but a position-dependent rotation $R_m$ generally fails to commute with an arbitrary up-projection:

$$
R_mW^{UK}\ne W^{UK}R_m.
$$

DeepSeek therefore separates a small RoPE key component and stores it alongside the latent. The obstacle is not nonlinearity; it is that one constant matrix cannot absorb every position-dependent rotation.

The paper's reported quality and throughput results are evidence for the complete trained design. The raw 56.9× element ratio is only a storage calculation and cannot predict quality on its own.

![VIZ m4/04 — latent KV state](assets/modern-llms/en/module-04/m4_04_mla_latent.svg)

## 8. FlashAttention: dense attention without the quadratic HBM object

A materializing implementation often writes the score matrix, reads it for softmax, writes probabilities, and reads them again for the value product. The arithmetic is not the only cost; the memory path is expensive.

FlashAttention tiles Q, K, and V so that the active blocks fit in on-chip **SRAM—Static Random-Access Memory**. Each query row carries three online-softmax quantities across key tiles:

- the running maximum $m$;
- the running normalizer $\ell$;
- the running value numerator or output accumulator $n$.

When a later tile raises the maximum, earlier state is multiplied by $e^{m_{old}-m_{new}}$. The final output is $n/\ell$.

![VIZ m4/05 — online softmax](assets/modern-llms/en/module-04/m4_05_online_softmax.svg)

It is important to state the complexity precisely.

| Property | Materializing dense attention | FlashAttention |
|---|---|---|
| Dense pairwise arithmetic | $O(N^2d)$ | $O(N^2d)$ |
| Full $N\times N$ object in HBM | yes | no |
| Additional stored memory | quadratic | linear in sequence length |
| HBM traffic | high | substantially reduced; depends on SRAM/tile geometry |

The IO analysis is parameterized by fast-memory capacity $M$; it is not a universal reduction to $O(Nd)$ traffic. FlashAttention is called exact because it computes every dense interaction rather than selecting a subset. Floating-point order and kernel approximations may still cause small numerical differences from a naive reference.

FlashAttention-4 reports up to 1,613 TFLOP/s in BF16 on B200, approximately 71% of peak, with up to 1.3× speedup over cuDNN 9.13 and 2.7× over Triton in the authors' configurations. [Primary paper](https://arxiv.org/abs/2603.05451).

![VIZ m4/06 — tiled IO](assets/modern-llms/en/module-04/m4_06_tiling_memory.svg)

## 9. Windowed attention and layer schedules

A sliding-window layer allows each query to see only the most recent $W$ keys. Its decode cache stops growing after $W$ positions. A stack of local layers can propagate information farther than one window, but the approximate reach $L(W-1)$ is only a path-length bound. It does not guarantee that a fact survives many transformations or match a direct global edge.

Pure windowed decoder stacks do exist. Mistral 7B is a documented example. Hybrid models add global layers to shorten long-range paths.

| Model family | Published layout |
|---|---|
| Mistral 7B | sliding-window attention throughout the decoder |
| gpt-oss | 18 window-128 and 18 full layers, alternating |
| Gemma 2 | local 4096 and global 8192 layers interleaved 1:1 |
| Gemma 4 | hybrid local/global attention; final layer global |

This is why a cache estimate must be layer-aware. A global `sliding_window` field or a single context length does not reveal how many layers retain which positions.

![VIZ m4/03 — hybrid schedules](assets/modern-llms/en/module-04/m4_03_layer_alternation.svg)

## 10. “Attention sink” refers to two different objects

StreamingLLM observed that trained models often place large attention mass on initial real tokens even when those tokens are not semantically important. Dropping their K/V state from a rolling window can destabilize the model. Keeping a few initial tokens alongside the recent window restores behavior. This is an empirical property of existing attention maps. [StreamingLLM](https://arxiv.org/abs/2309.17453).

gpt-oss implements a different mechanism. Each query head owns a learned scalar $s_h$. The scalar is appended as an extra score column before softmax. After normalization, that column is removed before multiplying by V, so it receives probability mass but contributes no value vector.

For 128 equal token scores,

$$
p_{sink}=\frac{e^{s_h}}{128+e^{s_h}}.
$$

At $s_h=0,2,5$, the sink receives roughly 0.78%, 5.46%, and 53.69% of the mass.

A learned constant score is not equivalent to an ordinary fixed key. A key would produce $q^\top k_{sink}$ and vary with the query. The exact abstraction is an extra score slot with zero value. It could be represented through dot-product attention only by extending the space with a constant coordinate or by adding kernel support.

![VIZ m4/08 — two sinks](assets/modern-llms/en/module-04/m4_08_sinks.svg)

## 11. Sparse attention selects positions rather than shrinking each position

GQA and MLA reduce state per token. Windowing chooses a fixed geometric subset. Content-based sparse attention spends additional work to select tokens or blocks for each query.

| Method | Selection unit | Selection mechanism | Special training? | Evidence boundary |
|---|---|---|---|---|
| block-sparse patterns | blocks | predetermined layout | sometimes no | mature systems technique |
| MoBA—Mixture of Block Attention | blocks | learned block routing | yes | paper and open code |
| NSA—Native Sparse Attention | compressed/block/token branches | learned hierarchical selection | yes | author experiments |
| DSA—DeepSeek Sparse Attention | tokens | lightning indexer + top-$k$ core attention | continued training | official report, weights, kernels |
| SubQ | claimed sparse architecture | limited public detail | not fully disclosed | developer claim |

MoBA applies a mixture-of-experts style choice to attention blocks. NSA combines compressed global summaries, selective token access, and local context in a hardware-aware trainable design. Their speed and quality numbers belong to the reported experimental protocols. [MoBA](https://arxiv.org/abs/2502.13189), [NSA](https://arxiv.org/abs/2502.11089).

DeepSeek introduced DSA in DeepSeek-V3.2-Exp on 29 September 2025, after continued training from V3.1-Terminus. The release includes an indexer, sparse core attention, weights, a technical report, and key kernels. [Official release](https://api-docs.deepseek.com/news/news250929).

For partially disclosed systems, context length and company-provided asymptotic claims are not a substitute for an implementable architecture. The passport should mark such fields as developer-reported or undisclosed.

## 12. Level-one code: transparent SDPA, masks, and GQA

The NumPy implementation below is designed to be inspected and tested. It exposes shape contracts, grouping order, fully masked rows, initial real-token anchors, and a separate learned score sink. It is not a GPU kernel.

```python
from typing import Literal

import numpy as np

EmptyRowPolicy = Literal["error", "zero"]

def make_mask(n: int, window: int | None = None, n_sink: int = 0) -> np.ndarray:
    """Additive causal mask [n, n]: 0 allows a pair and -inf forbids it."""
    if n <= 0 or (window is not None and window <= 0):
        raise ValueError("n and window must be positive")
    if not 0 <= n_sink <= n or (n_sink and window is None):
        raise ValueError("sink anchors require a finite window")

    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) < window
        allowed |= (j < n_sink) & (j <= i)
    return np.where(allowed, 0.0, -np.inf)

def sdpa(Q, K, V, mask=None, *, sink=None, empty_row: EmptyRowPolicy = "error"):
    """Q [..., Tq, dk], K [..., Tk, dk], V [..., Tk, dv] -> [..., Tq, dv].

    A row with no legal key does not define a softmax. ``error`` exposes a broken mask;
    ``zero`` explicitly returns a zero output for a padding row. A finite sink logit
    provides legal support of its own while carrying a zero value vector.
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")

    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if min(Q.ndim, K.ndim, V.ndim) < 2:
        raise ValueError("Q, K, and V must each have at least two dimensions")
    if Q.shape[-1] != K.shape[-1] or K.shape[-2] != V.shape[-2]:
        raise ValueError("incompatible Q/K/V shapes")
    try:
        np.broadcast_shapes(Q.shape[:-2], K.shape[:-2], V.shape[:-2])
    except ValueError as exc:
        raise ValueError("Q/K/V batch dimensions are not broadcast-compatible") from exc
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q, K, and V must contain only finite values")

    # Promote float16 inputs before matmul; widening only the destination would be too late
    # if the dot product had already overflowed in float16.
    dtype = np.result_type(Q.dtype, K.dtype, V.dtype, np.float32)
    Q, K, V = (x.astype(dtype, copy=False) for x in (Q, K, V))
    scale = np.asarray(1.0 / np.sqrt(Q.shape[-1]), dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    if not np.isfinite(scores).all():
        raise FloatingPointError("Q @ K.T overflowed; rescale the inputs")

    if mask is not None:
        mask = np.asarray(mask, dtype=dtype)
        if np.isnan(mask).any() or np.isposinf(mask).any():
            raise ValueError("mask may contain finite bias and -inf, but not NaN/+inf")
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask shape is not broadcast-compatible with scores") from exc
        with np.errstate(over="ignore", invalid="ignore"):
            scores = scores + mask
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise FloatingPointError("the finite mask bias overflowed; reduce its magnitude")

    has_token = np.isfinite(scores).any(axis=-1, keepdims=True)
    token_max = scores.max(axis=-1, keepdims=True)
    if sink is None:
        if empty_row == "error" and not has_token.all():
            raise ValueError("fully masked attention row")
        row_max = np.where(has_token, token_max, np.zeros_like(token_max))
        sink_logits = None
    else:
        sink_logits = np.asarray(sink, dtype=dtype)
        if not np.isfinite(sink_logits).all():
            raise ValueError("sink logits must be finite")
        try:
            sink_logits = np.broadcast_to(sink_logits, token_max.shape)
        except ValueError as exc:
            raise ValueError("sink shape is not broadcast-compatible with query rows") from exc
        row_max = np.maximum(token_max, sink_logits)

    token_exp = np.where(
        np.isfinite(scores), np.exp(scores - row_max), np.zeros_like(scores)
    )
    denominator = token_exp.sum(axis=-1, keepdims=True)
    if sink_logits is not None:
        denominator += np.exp(sink_logits - row_max)  # normalized, never added to V

    weights = np.divide(
        token_exp, denominator, out=np.zeros_like(token_exp), where=denominator > 0
    )
    return weights @ V

def gqa(Q, K, V, n_heads: int, n_kv_heads: int, mask=None):
    """Educational GQA for 2D projections; returns [T, H*dv]."""
    Q, K, V = (np.asarray(x) for x in (Q, K, V))
    if any(x.ndim != 2 for x in (Q, K, V)):
        raise ValueError("GQA expects 2D projected sequences")
    if not (Q.shape[0] == K.shape[0] == V.shape[0]):
        raise ValueError("Q, K, and V must have the same sequence length T")
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("n_heads must be a positive multiple of n_kv_heads")
    if Q.shape[1] % n_heads or K.shape[1] % n_kv_heads or V.shape[1] % n_kv_heads:
        raise ValueError("Q/K/V widths must divide their corresponding head counts")

    T = Q.shape[0]
    d_k = Q.shape[1] // n_heads
    if K.shape[1] // n_kv_heads != d_k:
        raise ValueError("query and key head dimensions must match")
    d_v = V.shape[1] // n_kv_heads
    q = Q.reshape(T, n_heads, d_k).transpose(1, 0, 2)
    k = K.reshape(T, n_kv_heads, d_k).transpose(1, 0, 2)
    v = V.reshape(T, n_kv_heads, d_v).transpose(1, 0, 2)
    repeats = n_heads // n_kv_heads
    # repeat gives [kv0, kv0, ..., kv1, kv1, ...], preserving each group.
    k = np.repeat(k, repeats, axis=0)
    v = np.repeat(v, repeats, axis=0)
    z = sdpa(q, k, v, mask)
    return z.transpose(1, 0, 2).reshape(T, n_heads * d_v)
```

In this code, `n_sink` preserves initial **real** positions in a windowed mask. The `sink` argument to `sdpa` is a different object: a learned score without a value contribution.

## 14. Reading the gpt-oss attention implementation

The open reference code turns abstract fields into tensor operations.

**GQA.** Queries are reshaped to `[tokens, H_kv, H_q/H_kv, head_dim]`; K and V use `[tokens, H_kv, head_dim]`. The group ratio is therefore explicit in the layout.

**Layer schedule.** Even-indexed layers apply a window of 128, while alternating layers use full attention. That schedule produces 18 local and 18 global layers.

**Learned sinks.** A scalar per query head is concatenated to the score matrix. Softmax includes it, but the final column is removed before the V product.

**RoPE.** Query and key are rotated before attention, using the positional geometry studied in Module 3.

![VIZ m4/09 — reading code and config](assets/modern-llms/en/module-04/m4_09_config_attention.svg)

## 15. A contemporary attention landscape with evidence labels

The useful comparison is not a leaderboard of mechanisms. It is a record of what is stored, what is visible, and how we know.

| Family or work | Per-position state | Visible positions | Source type |
|---|---|---|---|
| Llama 3.x | GQA | full attention | open configuration/implementation |
| Mistral 7B | GQA | sliding window | paper and open configuration |
| DeepSeek-V2/V3 | MLA | full attention before later sparse variants | technical reports |
| gpt-oss | GQA | alternating full/window-128 + learned sink | open code and configuration |
| Gemma 2 | GQA | 1:1 local/global interleaving | official documentation |
| Gemma 4 | hybrid state | local/global, final global | official model card/configuration |
| MoBA | compatible KV geometry | selected blocks plus local context | paper and open code |
| NSA | architecture-specific | compressed/selective/window branches | paper, author results |
| DeepSeek-V3.2-Exp | MLA latent cache | DSA-selected tokens | official report, weights, kernels |
| closed API model | undisclosed unless documented | undisclosed unless documented | context length is insufficient evidence |

Four axes remain independent:

1. the number and shape of cached coordinates;
2. the layer schedule;
3. the rule that selects positions;
4. the kernel that executes the rule.

An advertised million-token context does not determine any of these by itself.

## 16. Multimodal attention: where the visual state is paid for

Early-fusion systems insert visual embeddings into the language sequence. Those positions then inherit the decoder's cache geometry. In a hypothetical all-full GQA decoder costing 72 KiB per token, 4,096 visual tokens would add

$$
4096\cdot72\ \text{KiB}=288\ \text{MiB}
$$

of KV state. This is a neutral teaching scenario. Released gpt-oss models are text-only; there is no official `gpt-oss-multimodal` checkpoint behind the calculation.

Cross-attention systems encode vision separately and let language layers query the resulting features. The visual encoder need not run again for every language layer. Still, each cross-attention layer may have its own K/V projections or cache, so the actual memory contract is architecture-dependent.

The connector therefore decides more than dimensional compatibility: it decides where visual tokens enter the attention state. Module 14 develops this design space in detail.

## 19. Key takeaways

![VIZ m4/10 — attention on one page](assets/modern-llms/en/module-04/m4_10_cheatsheet.svg)

- **Cache arithmetic is layer-specific.** $2LH_{kv}d_hb$ is an all-full formula; hybrid layouts must be summed by layer type.
- **MQA and GQA reduce KV heads.** A common group ratio is not a universal quality optimum.
- **MLA changes the cached object.** A raw element ratio and a paper's end-to-end cache reduction are different comparisons.
- **FlashAttention does not linearize dense attention.** It eliminates the quadratic HBM intermediate and reduces IO.
- **Windowed layers cap local state.** Depth creates possible long paths, not guaranteed global recall.
- **Attention sink has two meanings.** Initial real-token sinks and learned score slots must be distinguished.
- **Sparse attention selects positions.** Selection cost, training requirements, and evidence status matter as much as asymptotics.
- **Context length is not an attention passport.** State, connectivity, dtype, workload, and implementation determine cost.

## 20. Notebook and primary sources

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
- gpt-oss — [official repository](https://github.com/openai/gpt-oss)

**Further course connections.** KV paging, offloading, and prefill/decode disaggregation are developed in Module 9. Multimodal attention placement is developed in Module 14.

---

*Landscape verified: 5 August 2026. Memory figures are reproducible calculations for stated layouts; external performance numbers retain their authors' hardware and protocol.*
