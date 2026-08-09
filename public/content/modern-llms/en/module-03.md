# Module 3. Positional Representations: From Sinusoids to Multidimensional Time

*Modern LLMs · Module 3 lecture · 2026.8 edition*

> **What this module does.** Attention can compare token content, but content alone does not say which token came first, how far apart two tokens are, or whether a visual token belongs above, below, before, or after another one. We will follow the positional signal as it moves through the Transformer: first as an additive sinusoidal vector, then as a rotation of queries and keys, then as a frequency spectrum that can be modified for longer context, and finally as several coordinate axes for images, video, and audio. Along the way we will separate mathematical identities from engineering heuristics and configuration fields from claims about training history.
>
> **Prerequisites.** You only need the basic attention picture: queries and keys produce attention logits, values carry the retrieved content. Every positional formula used below is derived in the module.

---

## 1. Motivation: content attention is not an ordering mechanism

Consider bidirectional self-attention applied to a sequence matrix $X$. If the rows are permuted by a permutation matrix $P$, the result obeys

$$
\operatorname{Attn}(PX)=P\operatorname{Attn}(X).
$$

This is **permutation equivariance**. The output is not invariant—the rows move—but the computation has no independent way to decide that one arrangement represents “the cat chased the dog” and another represents “the dog chased the cat.” It only follows the rows it receives.

A causal mask changes the situation, because token $i$ can only attend to an allowed prefix. That asymmetry already carries some order information. It still does not provide a rich coordinate system for distance. A decoder may know that a key is in the past without having a stable representation of whether it is one, one hundred, or one hundred thousand tokens away.

A useful positional mechanism therefore has to support several distinct jobs:

- distinguish order;
- expose relative distance or direction;
- remain usable when sequence length changes;
- extend to more than one axis when the input is an image, video, or synchronized stream.

The history of positional encoding is largely a history of deciding where this information should enter the model: added to embeddings, inserted as a bias, or built into the geometry of the query-key product.

![VIZ m3/01 — from permutation symmetry to positional structure](assets/modern-llms/en/module-03/m3_01_permutation.svg)

## 2. The historical arc: one problem, several architectural answers

![VIZ m3/02 — major branches of positional representation](assets/modern-llms/en/module-03/m3_02_pe_timeline.svg)

The original Transformer used deterministic sinusoidal positional encoding. Each position was mapped to a geometric ladder of sine and cosine frequencies. The representation contained no learned positional parameters and could be evaluated beyond the training table.

Learned absolute embeddings offered an even simpler interface: position $m$ selected row $m$ of a parameter matrix. They work well inside the trained range but tie the architecture to a finite table unless an explicit extension procedure is introduced.

Relative methods moved the positional signal closer to attention. Shaw et al. added learned relative representations. Transformer-XL designed a relative formulation compatible with recurrent segment memory. ALiBi later showed that a head-specific linear distance penalty could provide a surprisingly strong extrapolation prior without rotating or adding embeddings.

RoPE—Rotary Position Embedding—became a particularly influential decoder design. It represents absolute position as a rotation but makes the query-key dot product depend on relative phase. The operation changes neither tensor width nor KV-cache layout.

Long-context work created another branch. Position Interpolation compresses new positions into the old phase range. Community-developed NTK-aware methods change the frequency ladder non-uniformly. YaRN blends original and interpolated frequencies through a ramp. LongRoPE searches dimension-specific extension factors. Llama3-style scaling, partial RoPE, RoPE/NoPE hybrids, local attention, and attention-temperature tuning solve related problems with different assumptions.

Multimodal models then changed the question itself. An image needs height and width. Video adds time. A streaming audio-video model needs time IDs that mean the same thing across modalities. M-RoPE and TMRoPE turn a one-dimensional token index into a small coordinate system.

## 3. Fourier intuition, complex rotations, and what is exact

A sinusoidal encoding maps a position $m$ to periodic coordinates such as

$$
(\sin m\theta_i,\cos m\theta_i).
$$

For the pure positional vectors, the dot product of one sine-cosine pair is exactly relative:

$$
\sin(m\theta)\sin(n\theta)+
\cos(m\theta)\cos(n\theta)
=
\cos((m-n)\theta).
$$

That identity is sometimes obscured by a broader claim that RoPE “discovers” relative position while sinusoidal encoding is purely absolute. The real distinction is where the signal enters. In additive positional encoding, content and position are summed before learned projections. Cross-terms appear, so the final attention score need not be a pure function of $m-n$.

RoPE instead applies the phase after the hidden states have been projected into queries and keys. In complex notation, a coordinate pair $z=x_1+ix_2$ is multiplied by

$$
e^{im\theta}.
$$

The common phase cancels in the inner product, leaving a rotation by the relative displacement. This is a direct algebraic property of the query-key geometry.

The Fourier-features analogy remains valuable, but it should not be read too literally. Random Fourier Features sample frequencies to approximate a chosen kernel. Transformer sinusoidal and rotary frequencies are deterministic and geometrically spaced; they are an architectural basis, not a random kernel estimator.

## 4. Formalism: the frequency ladder and the rotary identity

For an even head dimension $D$, define $D/2$ inverse frequencies

$$
\theta_i=\mathrm{base}^{-2i/D},
\qquad i=0,\ldots,D/2-1.
$$

Their wavelengths in token positions are

$$
\lambda_i=\frac{2\pi}{\theta_i}.
$$

Classic sinusoidal encoding uses

$$
PE(m,2i)=\sin(m\theta_i),
\qquad
PE(m,2i+1)=\cos(m\theta_i).
$$

RoPE turns the same phase into a rotation matrix:

$$
R_{m,i}=
\begin{pmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{pmatrix}.
$$

For a query at position $m$ and a key at position $n$,

$$
q'_m=R_mq_m,
\qquad
k'_n=R_nk_n,
$$

and

$$
(q'_m)^Tk'_n
=q_m^TR_{n-m}k_n.
$$

RoPE therefore uses absolute position IDs to construct rotations, while the positional contribution to the bilinear score enters through the relative displacement $n-m$. The full score still depends on the content carried by $q_m$ and $k_n$.

There are two common coordinate layouts. Pedagogical derivations often rotate adjacent pairs $(x_0,x_1)$, $(x_2,x_3)$, and so on. Llama-style `rotate_half` pairs the first half of the vector with the second. They are equivalent under a fixed permutation, not interchangeable on the same trained weights.

The rotation preserves pairwise norms and leaves the tensor shape unchanged. It does not commute with an arbitrary learned matrix. The actual model order is simple: project the residual stream into $Q$ and $K$, then rotate those projected vectors.

### NumPy → PyTorch · B02 — RoPE: rotating coordinate pairs

NumPy keeps the pairwise rotation visible; the PyTorch form applies the same operator while preserving device, dtype, and autograd.

```python
import numpy as np
import torch
```

#### 1. Explicit NumPy implementation

```python
def numpy_rope_explicit(
    x: np.ndarray,
    position_ids: np.ndarray,
    *,
    base: float = 10_000.0,
) -> np.ndarray:
    dim = x.shape[-1]
    inv_freq = base ** (-np.arange(0, dim, 2) / dim)
    angles = position_ids[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + angles.shape
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.result_type(x.dtype, np.float64))
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out
```

NumPy exposes inverse frequencies, absolute-position angles, and both coordinates of every 2D rotation, making the positional contribution visible.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_rope_explicit(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
    angles = position_ids.to(device=x.device, dtype=x.dtype)[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + tuple(angles.shape)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
```

The PyTorch formula is identical, but frequencies and positions live on the same device, the operations participate in autograd, and tensor dtype becomes part of the contract.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Frequencies | `np.arange` on a CPU array | `torch.arange(..., device=x.device)` |
| Trigonometry | `np.cos`, `np.sin` | `Tensor.cos`, `Tensor.sin` |
| Device | CPU | CPU/CUDA/MPS |
| Gradients | none | autograd through the input |
| Absolute position | explicit `position_ids` | must match the KV-cache offset |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
rng = np.random.default_rng(2)
x_np = rng.standard_normal((2, 5, 8))
pos_np = np.arange(5, dtype=np.int64) + 11
np_out = numpy_rope_explicit(x_np, pos_np)
x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
t_out = torch_rope_explicit(x, torch.tensor(pos_np))
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-12, atol=1e-12)
t_out.square().sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("B02 explicit NumPy / PyTorch + gradient: PASS")
```

</details>

Complete executable file: [`m03_rope_bridges.py`](../assets/m03_rope_bridges.py)

## 5. Worked example: three placements, one relative displacement

Let

$$
q=(1,0,0.5,0.5),
\qquad
k=(0,1,1,-0.5),
$$

with head dimension four and base 10,000. The inverse frequencies are

$$
\theta_0=1,
\qquad
\theta_1=0.01.
$$

Place the query at position 3 and the key at position 1. Rotating the two pairs and taking their dot product gives

$$
\langle R_3q,R_1k\rangle\approx1.144248.
$$

Now shift both tokens by four positions. The pair $(7,5)$ has the same displacement as $(3,1)$, so the result is unchanged. The pair $(2,0)$ gives the same value as well.

| query position | key position | $m-n$ | rotary logit |
|---:|---:|---:|---:|
| 3 | 1 | 2 | 1.144248 |
| 7 | 5 | 2 | 1.144248 |
| 2 | 0 | 2 | 1.144248 |

![VIZ m3/05 — a four-dimensional RoPE trace](assets/modern-llms/en/module-03/m3_05_worked_example.svg)

The identity is exact for fixed $q$ and $k$. In a network, those vectors still depend on content and layer state, so attention does not reduce to a distance-only kernel.

## 6. Reading the RoPE spectrum without inventing a hard horizon

The base parameter controls a geometric spectrum. The fastest pair is always

$$
\theta_0=1,
\qquad
\lambda_0=2\pi,
$$

regardless of base. Increasing `rope_theta` therefore does not remove the fastest local phase. It mainly slows the higher-index pairs, allocating more of the spectrum to long wavelengths.

For $D=64$:

| base | minimum wavelength | maximum wavelength | pairs with ≥1 turn by 4096 |
|---:|---:|---:|---:|
| 10,000 | 6.28 | 47,117 | 23 of 32 |
| 150,000 | 6.28 | 649,409 | 18 of 32 |
| 500,000 | 6.28 | 2,084,765 | 16 of 32 |

![VIZ m3/06 — the wavelength spectrum](assets/modern-llms/en/module-03/m3_06_wavelength_spectrum.png)

Each coordinate pair is periodic. The complete representation combines many different periods. One pair returning to a similar angle does not make two full positional vectors identical. Consequently, the largest wavelength of one pair is not a hard context limit, and a small base does not imply that every position beyond that wavelength becomes indistinguishable.

Turn counts still convey useful information. Fast pairs sweep through many phases during training; slow pairs may cover only a small arc. Extrapolating those slow coordinates sends the model into new phase regions. That is an out-of-distribution geometry, not a mathematical prohibition. Whether the model uses it successfully is empirical.

## 7. Context extension changes a spectrum, not just one number

Suppose a model has a reference length $L_0$ and we want to serve $sL_0$ tokens.

### Position Interpolation

Position Interpolation maps $m$ to $m/s$. Equivalently,

$$
\theta'_i=\frac{\theta_i}{s}.
$$

The new range stays inside familiar angular values. The cost is that all local phases are compressed as well. Fine-tuning is normally used to help the model adapt to this altered geometry.

### Static NTK-aware scaling

A widely used static heuristic changes the base to

$$
\mathrm{base}'
=
\mathrm{base}\,s^{D/(D-2)}.
$$

This leaves the fastest end of the spectrum less changed and slows the low-frequency end more aggressively. “NTK-aware” describes the motivation, not a general theorem guaranteeing safe zero-shot extension.

### YaRN

YaRN builds an original spectrum and a fully interpolated spectrum, then blends them through a frequency-dependent ramp. A diagnostic number of rotations within the reference context determines the ramp boundaries.

For the module scenario

```text
head_dim = 64
base = 150000
reference length = 4096
factor = 32
beta_fast = 32
beta_slow = 1
```

the original gpt-oss implementation uses the continuous bounds

$$
\mathrm{low}\approx8.0928,
\qquad
\mathrm{high}\approx17.3980.
$$

Those values still divide the integer-indexed pairs into three groups:

- 9 pairs remain at their original frequencies;
- 9 pairs occupy the transition ramp;
- 14 pairs are fully interpolated.

The generic Hugging Face YaRN utility also offers `truncate=True`, which rounds the boundaries to `low=8`, `high=18`. `GptOssConfig` sets `truncate=False`, so the continuous values are the faithful default for this model.

In this particular configuration, the fully interpolated set coincides with pairs that complete less than one turn in the reference context. That is a property of the chosen boundary formula and hyperparameters, not a universal rule that one full turn makes extrapolation “safe.”

![VIZ m3/07 — diagnostic YaRN zones](assets/modern-llms/en/module-03/m3_07_rotations_zones.png)

The original gpt-oss implementation and its Hugging Face adaptation also use an `attention_factor`. For $s>1$, both checked paths use

$$
a=1+0.1\ln s.
$$

At $s=32$, $a=1.346574$. Both checked implementations apply this factor to computed rotary cosines and sines. If both rotated queries and keys therefore receive the same factor, their dot product is scaled by

$$
a^2=1.813260.
$$

Another runtime may move the equivalent scaling into queries, keys, or the attention score. A configuration field only becomes unambiguous when read with the implementation that consumes it.

### Beyond one scaling recipe

LongRoPE searches per-dimension factors and can use different regimes for different lengths. LongRoPE2 incorporates training-aware criteria. Llama3-style scaling divides the spectrum into wavelength bands. Other models use partial RoPE, interleaved RoPE/NoPE, local attention, or length-dependent attention temperature.

![VIZ m3/08 — three spectrum transformations](assets/modern-llms/en/module-03/m3_08_scaling_methods.png)

## 8. ALiBi: a distance prior in the logits

ALiBi—Attention with Linear Biases—leaves queries and keys untouched. Head $h$ receives a slope $m_h$, and causal attention uses

$$
S_{ij}^{(h)}=q_i^Tk_j-m_h(i-j),
\qquad j\le i.
$$

Heads with steeper slopes favor nearby context more strongly. If the content score were fixed, softmax would turn the linear penalty into an exponential distance prior:

$$
\exp[-m_h(i-j)].
$$

The complete attention pattern is not a stationary kernel by itself; it still depends on content and normalization across available keys.

ALiBi's strength is structural simplicity and a bias that continues to any distance. Its limitation is equally clear: the form of the distance prior is fixed in advance. Good perplexity extrapolation does not automatically imply accurate retrieval from the middle of a very long document or robust multi-step reasoning over distant evidence.

## 9. NoPE and hybrid position mechanisms

NoPE—No Positional Encoding—removes an explicit positional term. A causal model is not deprived of all order information: the mask, depth, and recursively constructed hidden states expose asymmetries tied to prefix length.

Research shows that substantial positional behavior can emerge under these conditions. That does not make explicit position universally unnecessary. It means the model can infer part of the structure from other components of the computation.

Two cases should be separated:

- **pure NoPE**, where no layer receives an explicit positional encoding;
- **hybrid designs**, where some layers use RoPE and others NoPE, or where positional mechanisms are combined with different attention patterns.

Llama 4 is a concrete hybrid. Its open configuration supports a NoPE layer at a regular interval, chunked attention, and length-dependent temperature tuning. It should not be summarized as “RoPE plus YaRN.”

## 10. Reading gpt-oss positional fields as evidence—not biography

The original gpt-oss-120b configuration exposes

```json
{
  "head_dim": 64,
  "initial_context_length": 4096,
  "rope_theta": 150000,
  "rope_scaling_factor": 32.0,
  "rope_ntk_alpha": 1,
  "rope_ntk_beta": 32
}
```

This is enough to reconstruct a positional geometry:

- 32 rotary pairs per head;
- base 150,000;
- a reference length of 4,096;
- a 32-fold extension factor;
- correction-boundary parameters corresponding to 1 and 32 rotations.

The module's transparent reconstruction therefore gives the continuous bounds `low≈8.0928`, `high≈17.3980`, together with the default attention factor 1.346574. The integer pair `8/18` belongs to the optional `truncate=True` mode, not to the original gpt-oss path.

What the file does not provide is a complete training chronology. `initial_context_length` tells us the reference length used by the scaling contract. It does not, by itself, prove that all pretraining happened at 4K and that one particular extension phase followed. Such claims require a technical report or training recipe.

Converted Hugging Face configurations may express the same information through a more general RoPE parameter dictionary. Field names and scaling placement can change across representations, so the config should be read together with the model code and library revision.

![VIZ m3/09 — interpreting a RoPE configuration](assets/modern-llms/en/module-03/m3_09_config_rope.svg)

## 11. Level-one code: explicit shapes and explicit scaling placement

The following NumPy code is designed for inspection rather than fused execution. It validates head width, positions, frequency shape, and the YaRN ramp. The pairing convention is the split-half `rotate_half` layout used by Llama-like implementations.

```python
import math
import numpy as np

def rope_inv_freq(head_dim: int, base: float = 150_000.0) -> np.ndarray:
    """Return theta_i = base**(-2i/D) for D/2 rotary pairs."""
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise TypeError("head_dim must be an integer")
    if head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be positive and even")
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("base must be finite and greater than 1")
    i = np.arange(head_dim // 2, dtype=np.float64)
    return base ** (-2.0 * i / head_dim)

def rotate_half(x: np.ndarray) -> np.ndarray:
    """Quarter-turn the split-half coordinate pairs."""
    if x.ndim < 2 or x.shape[-1] % 2:
        raise ValueError("x must have shape [..., T, D] with even D")
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)

def apply_rope(q, k, positions, inv_freq):
    """Apply split-half RoPE to q/k of shape [..., T, D]."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    positions = np.asarray(positions)
    inv = np.asarray(inv_freq, dtype=np.float64)
    if q.shape != k.shape or q.ndim < 2 or q.shape[-1] % 2:
        raise ValueError("q and k must share shape [..., T, D], with even D")
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError("positions must be an integer vector [T]")
    if positions.size != q.shape[-2] or np.any(positions < 0):
        raise ValueError("positions do not match the sequence axis")
    if inv.shape != (q.shape[-1] // 2,) or np.any(inv <= 0):
        raise ValueError("inv_freq must be positive and have shape [D/2]")

    angles = positions.astype(np.float64)[:, None] * inv[None, :]
    angles = angles.reshape((1,) * (q.ndim - 2) + angles.shape)
    full = np.concatenate((angles, angles), axis=-1)
    cos, sin = np.cos(full), np.sin(full)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def yarn_inv_freq(head_dim, base, factor, original_context,
                  beta_fast=32.0, beta_slow=1.0, *, truncate=False):
    """Blend original and interpolated frequencies with a YaRN-style ramp."""
    inv = rope_inv_freq(head_dim, base)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("factor must be at least 1")
    if original_context <= 0 or beta_fast <= beta_slow or beta_slow <= 0:
        raise ValueError("invalid context or beta parameters")
    if not isinstance(truncate, bool):
        raise TypeError("truncate must be bool")

    def correction_dim(turns: float) -> float:
        return head_dim * math.log(
            original_context / (turns * 2 * math.pi)
        ) / (2 * math.log(base))

    pair_count = head_dim // 2
    low_raw = correction_dim(beta_fast)
    high_raw = correction_dim(beta_slow)
    low = math.floor(low_raw) if truncate else low_raw
    high = math.ceil(high_raw) if truncate else high_raw
    low = min(max(float(low), 0.0), float(pair_count - 1))
    high = min(max(float(high), 0.0), float(pair_count - 1))
    if high <= low:
        raise ValueError("the YaRN ramp collapsed")

    ramp = np.clip(
        (np.arange(pair_count, dtype=np.float64) - low) / (high - low),
        0.0,
        1.0,
    )
    scaled = inv * (1.0 - ramp) + (inv / factor) * ramp
    attention_factor = 1.0 if factor <= 1 else 1.0 + 0.1 * math.log(factor)
    return scaled, attention_factor, attention_factor**2, low, high
```

The last function returns both the rotary factor and the score factor implied when the same scaling enters queries and keys. Keeping them separate prevents a common ambiguity in configuration discussions.

## 13. M-RoPE and time-aligned multimodal position

M-RoPE—Multimodal Rotary Position Embedding—assigns different sections of rotary channels to temporal, height, and width coordinates. A single attention operation can therefore represent text and visual geometry without switching to a separate positional mechanism.

Text uses the same ID on all three axes:

$$
(m,m,m).
$$

Every frequency pair therefore receives the ordinary one-dimensional position $m$, regardless of which section it belongs to. The text path is equivalent to standard 1D RoPE.

A still image holds temporal position constant and varies its two spatial coordinates:

$$
(t_0,h,w).
$$

A $2\times2$ grid can be represented as

$$
(0,0,0),
(0,0,1),
(0,1,0),
(0,1,1).
$$

Video changes all three axes. Qwen2.5-VL can space temporal IDs according to real time between grid elements rather than treating frame number as the only clock.

TMRoPE—Time-aligned Multimodal Rotary Position Embedding—was documented in the Qwen2.5-Omni technical report in March 2025. It aligns audio and video timestamps. In that model, one temporal ID corresponds to 40 ms. This is a model-specific time grid, not a universal definition of TMRoPE.

![VIZ m3/10 — M-RoPE and TMRoPE](assets/modern-llms/en/module-03/m3_10_mrope.svg)

## 14. The 2026 landscape is a branching architecture map

As of 5 August 2026, position handling in leading open systems is not one YaRN monoculture. Published evidence supports several different contracts.

| Family or method | Positional mechanism | Purpose or context | Evidence |
|---|---|---|---|
| Transformer | deterministic sinusoidal PE | original sequence architecture | paper |
| RoFormer / RoPE | rotation in $Q/K$ | relative phase in attention | paper |
| ALiBi | linear logit bias | length extrapolation without a position table | paper |
| Position Interpolation | uniform position compression plus tuning | RoPE context extension | paper |
| YaRN | original/interpolated spectral ramp plus scaling | long-context extension | paper and runtime implementations |
| LongRoPE / LongRoPE2 | per-dimension and length-dependent factors | very long context | papers |
| gpt-oss | base 150K, reference length 4096, factor 32, correction parameters | up to 128K positional configuration | official config; config alone is not a training schedule |
| Llama 4 | interleaved RoPE/NoPE, chunked attention, temperature tuning | hybrid long-context stack | open configuration and implementation |
| Qwen2-VL / Qwen2.5-VL | M-RoPE | text, image, and video coordinates | paper and code |
| Qwen2.5-Omni | TMRoPE | aligned audio-video time | technical report |
| Inkling | relative positional embeddings with sliding/global layers | 1M context | official architecture announcement |
| Closed API models | may be undisclosed | advertised window does not reveal the formula | provider documentation only where available |

A context-window number does not identify the positional mechanism, and it does not establish reliable use of every position inside the window. Operational context quality is an evaluation problem, addressed later in Module 17.

## 17. Key takeaways

![VIZ m3/11 — positional mechanisms at a glance](assets/modern-llms/en/module-03/m3_11_cheatsheet.svg)

- Bidirectional self-attention without position is permutation-equivariant. A causal mask supplies access order, not a complete distance geometry.
- Pure sinusoidal pairs already have an exact relative dot product. RoPE inserts relative phase directly into the projected query-key score.
- `rope_theta` redistributes a frequency ladder. A larger base introduces slower components without removing the fastest pair.
- A single wavelength or turn count is not a hard context boundary.
- PI compresses every frequency; static NTK-aware scaling is a heuristic; YaRN blends original and interpolated spectra; other long-context families make different trade-offs.
- `attention_factor` must be interpreted at the point where the runtime applies it.
- ALiBi is a linear positional bias. NoPE can be pure or part of a hybrid stack.
- M-RoPE repeats IDs for text and uses multiple axes for visual tokens. TMRoPE aligns multimodal time.
- A configuration describes architectural geometry; it is not a substitute for a training report.

## 18. Notebook and primary sources

**Primary sources:**

- Vaswani et al., *Attention Is All You Need* — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Shaw et al., *Self-Attention with Relative Position Representations* — [arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)
- Dai et al., *Transformer-XL* — [arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)
- Su et al., *RoFormer / RoPE* — [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- Press et al., *ALiBi* — [arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)
- Haviv et al., positional behavior without explicit PE — [arxiv.org/abs/2203.16634](https://arxiv.org/abs/2203.16634)
- Chen et al., *Position Interpolation* — [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595)
- Peng et al., *YaRN* — [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- Ding et al., *LongRoPE* — [arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)
- Shang et al., *LongRoPE2* — [arxiv.org/abs/2502.20082](https://arxiv.org/abs/2502.20082)
- Bai et al., *Qwen2-VL* — [arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191)
- Xu et al., *Qwen2.5-Omni* — [arxiv.org/abs/2503.20215](https://arxiv.org/abs/2503.20215)
- [Hugging Face RoPE utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)
- [gpt-oss original configuration](https://huggingface.co/openai/gpt-oss-120b/blob/main/original/config.json)
- [gpt-oss reference RoPE implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)
- [Hugging Face GptOssConfig](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/configuration_gpt_oss.py)
- [Llama 4 model documentation](https://huggingface.co/docs/transformers/model_doc/llama4)
- [Inkling architecture announcement](https://thinkingmachines.ai/news/introducing-inkling/)

**Further study.** Fourier features, kernels, and information geometry are developed in *Information Theory for ML*. Recurrent and state-space alternatives to explicit positional encoding appear in later *Modern LLMs* modules.

---

*Landscape verified: 5 August 2026. Spectrum and YaRN numbers belong to an explicitly stated teaching scenario. For closed models, an undisclosed positional mechanism is not inferred from the advertised context window.*
