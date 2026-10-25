# Module 4. Positional representations: from sinusoids to multiple axes

*Modern LLMs · Module 4 lecture*

Module 3 followed a query as it compared itself with keys and gathered a weighted mixture of values. We now isolate one more part of that computation. Keep the content of the query and key fixed, but move their positions. What should happen if both tokens move together? What changes if the gap between them grows? Holding content steady lets us see exactly what the positional mechanism contributes.

We will first answer those questions using sinusoidal coordinates and RoPE rotations. Then we will ask why being able to compute a rotation at a large position index is not the same as knowing how to use a long context. That distinction leads to frequency scaling and, eventually, to the spatial and temporal coordinates of images, video, and audio.

You need dot products and the attention mechanism from M3. We will introduce rotation using a pair of coordinates; complex numbers provide an optional shorthand for the same operation.

**Reading routes**

**On a first reading**, follow §§1–10 in order. Use §2 as a map; the historical detail and the implementation in §4 can be opened later. The main argument runs through the geometry of rotation, the numerical example in §5, and the frequency changes in §§6–7. For YaRN, first follow which parts of the spectrum are preserved and which are slowed, then work through the transition boundaries. Finish with multimodal coordinates in §13 and the takeaways in §17.

**To reproduce the computation**, return to the example in §5, the ramp in §7, and the code in §§4 and 11. The module's existing [notebooks](/courses/modern-llms/module-04/notebooks/) include cache and position-precision checks. The separate [exercises](/courses/modern-llms/module-04/exercises/) and [tests](/courses/modern-llms/module-04/tests/) provide independent practice and review.

**For later reference**, use §4 for shapes and coordinate layouts, §6 for the spectrum, §7 for scaling, §10 for configuration fields, and §13 for multiple axes. The implementation map remains in §14. Module 5 then turns to attention architecture and execution.

## 1. Motivation: content attention is not an ordering mechanism

Consider bidirectional self-attention applied to a sequence matrix $X$, without positional information or a restrictive mask and with dropout disabled. If the rows are permuted by a permutation matrix $P$, the result obeys

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

![VIZ m4/01 — from permutation symmetry to positional structure](assets/modern-llms/en/module-04/m4_01_permutation.svg)
*Bidirectional self-attention is permutation-equivariant without a positional signal. A causal mask breaks full symmetry, but it does not by itself provide a rich distance geometry.*

## 2. A map of where position enters

**Positional encoding (PE)** introduces token coordinates into the computation. **RoPE (Rotary Position Embedding)** does so by rotating pairs of coordinates.

Recall the computation from M3. A token starts with an input vector; hidden states are projected into queries and keys; their comparison produces a logit before softmax. Position can enter at any of these points. It can be added to the input, used to transform the projected vectors, or introduced as a separate bias in the score. The table distinguishes these entry points. It is a map of design choices, not a ranking of successive generations.

| Entry point | Operation | Example and section |
|---|---|---|
| Input representation | add a vector indexed by position | sinusoidal PE, §§3–4 |
| Projected query and key | rotate coordinate pairs by position-dependent angles | RoPE, §§4–7 |
| Attention logit | add a distance-dependent bias | ALiBi, §8 |
| No separate positional term | learn order-sensitive behavior from the causal mask and layers | NoPE, §9 |

We will begin with one route through this map: how sinusoidal coordinates lead to rotating queries and keys in RoPE. Once the rotation itself is clear, choosing its frequencies becomes the next question. PI, NTK-aware scaling, and YaRN address that question in §7. M-RoPE returns to the same operation with several coordinate axes in §13. For now, the names mark questions we will reach, rather than formulas to learn in advance.

![VIZ m4/02 — branches of positional representation](assets/modern-llms/en/module-04/m4_02_pe_timeline.svg)
*The branches differ in where and how they introduce position. A later method does not replace every earlier one.*

**Historical connections — optional reading**

The original Transformer used deterministic sinusoidal positional encoding. Each position was mapped to a geometric ladder of sine and cosine frequencies. The representation contained no learned positional parameters and could be evaluated beyond the training table.

Learned absolute embeddings offered an even simpler interface: position $m$ selected row $m$ of a parameter matrix. They work well inside the trained range but tie the architecture to a finite table unless an explicit extension procedure is introduced.

Relative methods moved the positional signal closer to attention. Shaw et al. added learned relative representations. Transformer-XL designed a relative formulation compatible with recurrent segment memory. ALiBi later showed that a head-specific linear distance penalty could provide a surprisingly strong extrapolation prior without rotating or adding embeddings.

RoPE—Rotary Position Embedding—became a particularly influential decoder design. It represents absolute position as a rotation but makes the query-key dot product depend on relative phase. The operation changes neither tensor width nor KV-cache layout.

Long-context work created another branch. Position Interpolation compresses new positions into the old phase range. Community-developed NTK-aware methods change the frequency ladder non-uniformly. YaRN blends original and interpolated frequencies through a ramp. LongRoPE searches dimension-specific extension factors. Llama3-style scaling, partial RoPE, RoPE/NoPE hybrids, local attention, and attention-temperature tuning solve related problems with different assumptions.

Multimodal models then changed the question itself. An image needs height and width. Video adds time. A streaming audio-video model needs time IDs that mean the same thing across modalities. M-RoPE and TMRoPE turn a one-dimensional token index into a small coordinate system.

## 3. Fourier intuition, complex rotations, and what is exact

Let $m$ be an integer position counted from zero. Pair $i$ has angular frequency $\theta_i$, measured in radians per position step. Its **phase** is the angle $m\theta_i$. Frequency tells us how much that angle changes when we advance by one position.

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

The identity shows that absolute sinusoidal coordinates can be compared in a way that depends on relative displacement. So far, however, we have compared only positional vectors. Additive positional encoding mixes them with content embeddings before the learned projections. The resulting dot product contains content–position cross-terms, so the final attention score need not be a pure function of $m-n$. Choosing sines and cosines is only part of the design; where they enter the computation matters just as much.

RoPE changes that point of entry. It first projects the hidden states into queries and keys, then rotates their coordinate pairs. Picture each pair as an arrow in a plane. Turning both arrows through the same angle preserves their lengths and the angle between them, hence their dot product. What matters to the comparison is the difference between the applied rotations, not either arrow's orientation relative to a fixed external axis. This geometric observation is the basis of the notation that follows.

**Optional complex notation.** Write the pair as $z=x_1+ix_2$, with $i$ now denoting the imaginary unit. Rotation is multiplication by

$$
e^{im\theta}.
$$

The common phase cancels in the inner product, leaving a rotation by the relative displacement. This is a direct algebraic property of the query-key geometry.

The Fourier-features analogy remains valuable, but it should not be read too literally. Random Fourier Features sample frequencies to approximate a chosen kernel. Transformer sinusoidal and rotary frequencies are deterministic and geometrically spaced; they are an architectural basis, not a random kernel estimator.

## 4. Formalism: the frequency ladder and the rotary identity

The geometry is now clear for one coordinate pair. A query or key contains many coordinates, though, and we need a rotation for every pair. RoPE does not turn them all at the same rate: some angles change quickly with position, others slowly. We will now specify that set of frequencies and connect it to the sinusoidal encoding already introduced.

Here $D$ is an even number of rotated coordinates, not the residual width $d_{model}$. Full-head RoPE has $D=d_k$; additive PE uses the same frequency construction at the input-vector width. Code often calls these values `inv_freq`, because they are reciprocals of the phase denominators. A base greater than one controls how quickly frequency falls across pairs.

For an even rotated width $D$, define $D/2$ inverse frequencies

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

![VIZ m4/03 — sinusoidal PE](assets/modern-llms/en/module-04/m4_03_sinusoidal_heatmap.png)
*Rows are positional coordinates. Adjacent rows form sine–cosine pairs; higher-index pairs vary more slowly.*

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

![VIZ m4/04 — RoPE inside attention](assets/modern-llms/en/module-04/m4_04_rope_mechanics.svg)
*Project to Q/K/V, rotate Q and K, form scaled scores, then mix V. The version shown leaves V unrotated.*

**Partial RoPE.** Some models rotate only D coordinates of a wider head, leaving the remaining channels unchanged. Frequencies use the rotated width D; attention normalization uses the full key width $d_k$. Scaling just the rotary channels does not multiply the entire dot product by one common factor.

**NumPy and PyTorch implementation — second pass**

### NumPy → PyTorch · B02 — RoPE: rotating coordinate pairs

Implementation route. Assume a validated, nonempty floating tensor `x[..., T, D]`, even D, nonnegative integer IDs `[T]`, and a finite base greater than one. Adjacent coordinates form pairs. The existing [module notebook](/courses/modern-llms/module-04/notebooks/) adds input validation.

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
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv_freq = base ** (-torch.arange(0, dim, 2, device=x.device, dtype=work_dtype) / dim)
    angles = position_ids.to(device=x.device, dtype=work_dtype)[:, None] * inv_freq[None, :]
    shape = (1,) * (x.ndim - 2) + tuple(angles.shape)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    even, odd = x[..., 0::2].to(work_dtype), x[..., 1::2].to(work_dtype)
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)
```

Phase construction and rotation use at least float32 before conversion back to the input dtype. This avoids early FP16/BF16 position rounding. Float64 remains the reference for large indices: keeping IDs as integers does not remove error after conversion to a working float type. Gradients flow through the input vectors.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Frequencies | `np.arange` on a CPU array | `torch.arange(..., device=x.device)` |
| Trigonometry | `np.cos`, `np.sin` | `Tensor.cos`, `Tensor.sin` |
| Device | CPU | CPU/CUDA/MPS |
| Gradients | none | autograd through the input |
| Precision | float64 calculation and output | float32/64 phases; output keeps input dtype |
| Absolute position | explicit `position_ids` | must agree with cached-key positions |

**Executable parity and edge-case check**

```python
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
```

Complete executable file: `m04_rope_bridges.py`

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

Place the query at position 3 and the key at position 1. Using **adjacent coordinate pairs**, their rotated dot product before attention scaling is

$$
\langle R_3q,R_1k\rangle\approx1.144248.
$$

Now shift both tokens by four positions. The pair $(7,5)$ has the same displacement as $(3,1)$, so the result is unchanged. The pair $(2,0)$ gives the same value as well.

| query position | key position | $m-n$ | unscaled dot product |
|---:|---:|---:|---:|
| 3 | 1 | 2 | 1.144248 |
| 7 | 5 | 2 | 1.144248 |
| 2 | 0 | 2 | 1.144248 |

![VIZ m4/05 — a four-dimensional RoPE trace](assets/modern-llms/en/module-04/m4_05_worked_example.svg)
*Absolute angles change across placements, but a shared relative displacement preserves the same unscaled dot product.*

To obtain the standard attention logit from M3, divide by $\sqrt{d_k}=\sqrt4=2$, giving approximately **0.572124**. Masking and softmax follow. This constant scaling preserves the common-shift identity.

The identity is exact for fixed $q$ and $k$. In a network, those vectors still depend on content and layer state, so attention does not reduce to a distance-only kernel.

## 6. Reading the RoPE frequency spectrum

The worked example showed that a common shift preserves the dot product. We now examine a different property: how quickly each pair's angle changes as distance increases. That collection of rates is the frequency spectrum. Fast pairs vary noticeably over short distances; slow pairs change over longer scales. The base parameter sets a geometric ladder of frequencies. The fastest pair is always

$$
\theta_0=1,
\qquad
\lambda_0=2\pi,
$$

regardless of base. Increasing `rope_theta` therefore does not slow every pair equally. The fastest pair stays fixed, while higher-index pairs slow more strongly and acquire longer wavelengths. The number of pairs does not change: we redistribute the scales represented by existing coordinates rather than introduce new ones.

To relate this spectrum to context length, ask how many turns each pair completes over the interval of interest. The diagnostic is $L\theta_i/(2\pi)$: turns across a **span of L position steps**, following the YaRN convention. A sequence indexed from 0 to L−1 has endpoint separation L−1. These quantities must be kept distinct. Wavelengths below are measured in position steps.

For $D=64$:

| base | minimum wavelength | maximum wavelength | pairs with ≥1 turn by 4096 |
|---:|---:|---:|---:|
| 10,000 | 6.28 | 47,117 | 23 of 32 |
| 150,000 | 6.28 | 649,409 | 18 of 32 |
| 500,000 | 6.28 | 2,084,765 | 16 of 32 |

![VIZ m4/06 — the wavelength spectrum](assets/modern-llms/en/module-04/m4_06_wavelength_spectrum.png)
*Wavelengths show how a base redistributes phases. Context-length guide lines are diagnostics, not a global period for the full positional vector.*

The table shows how one setting changes a whole collection of scales. Each quantity nevertheless answers a limited question. An individual coordinate pair is periodic, but the full RoPE representation combines many different periods. One pair returning to a similar angle does not make the entire positional code repeat. The largest wavelength is therefore not a hard context limit, and a turn count is not a guarantee that particular positions will be distinguishable to the model.

The picture is still useful for understanding training. Within the training range, fast pairs may sweep through many turns while slow pairs cover only a small arc. At longer distances, those slow components reach angles absent from that range. The formula remains computable, but the model encounters different combinations of positional features. Whether those combinations are useful depends on the spectrum as a whole, the training, the attention pattern, and the task. This is where context extension becomes a modeling problem rather than merely an evaluation of sine and cosine.

## 7. Context extension changes a spectrum, not just one number

Computing a rotation at a large index is straightforward. Preserving a useful relationship between content and distance is harder. We can keep the frequencies and move beyond familiar phase ranges. Or we can slow the rotations so that a longer interval fits inside the old range, at the cost of changing short-distance geometry as well. The following methods make different choices within this trade-off.

Let $L_0$ be the reference length used to set the frequencies, and let $s\ge1$ be an extension factor. We want to process $sL_0$ tokens. In a research experiment, $L_0$ is often the training length; a configuration field alone does not establish the full training history (§10).

### Position Interpolation

The most direct choice is to compress the entire position scale. Position Interpolation maps $m$ to $m/s$. The token retains its place in the sequence, but receives a smaller rotation angle. Equivalently,

$$
\theta'_i=\frac{\theta_i}{s}.
$$

The longer interval now fits inside the previous angular range. No tokens have been removed and the sequence has not become shorter; only the scale used to construct rotations has changed. Neighboring positions are closer in phase in every rotary pair. We gain range by altering local geometry too. The original method uses fine-tuning to help the model adapt to that change.

### Static NTK-aware scaling

Uniform compression raises a natural question: must fast pairs be changed as strongly as slow ones? Section 6 showed that increasing the base affects them differently. A widely used static NTK-aware heuristic exploits that property by changing the base to

$$
\mathrm{base}'
=
\mathrm{base}\,s^{D/(D-2)}.
$$

Instead of one reduction factor for every frequency, we obtain a nonuniform change: fast components are affected less and slow components more. This preserves more of the original local geometry than uniform PI. The name “NTK-aware” describes the motivation, not a theorem guaranteeing safe zero-shot extension. The result still depends on the model and the chosen length and must be evaluated.

Assume $s\ge1$ and $D>2$. Substituting the new base gives $\theta_i^\prime/\theta_i=s^{-2i/(D-2)}$: pair zero is unchanged, while the final pair slows by exactly s. The expression is undefined at D=2. NTK stands for Neural Tangent Kernel, a theoretical description of wide networks; this calculation uses only the stated base-change rule.

### YaRN

YaRN makes the distinction between frequency bands explicit. Some pairs should keep their original frequency, others should use full interpolation, and pairs between them should change gradually. The method therefore constructs both the original spectrum and the fully interpolated one, then blends them according to frequency. The number of rotations within the reference interval determines where the transition begins and ends.

To decide where each pair belongs, we return to the turn count from §6. Define $r_i=L_0\theta_i/(2\pi)$, the number of turns over the reference span. Two thresholds, $\beta_{fast}>\beta_{slow}>0$, select the transition: fast pairs retain their original frequencies, while slow pairs use interpolation. To apply this rule to a frequency vector, we need boundaries in pair indices rather than in turns. Solving $r_i=\beta$ gives a fractional index:

$$
b(\beta)=\frac{D\ln[L_0/(2\pi\beta)]}{2\ln(\mathrm{base})},\qquad
\mathrm{low}=b(\beta_{fast}),\quad \mathrm{high}=b(\beta_{slow}).
$$

We can now assign each pair index a degree of movement from its original frequency toward the interpolated one. A **ramp** makes that transition linear between the boundaries and clips it to the interval [0,1]:

$$
w_i=\operatorname{clip}\!\left(\frac{i-\mathrm{low}}{\mathrm{high}-\mathrm{low}},0,1\right),\qquad
\theta_i^\prime=(1-w_i)\theta_i+w_i\frac{\theta_i}{s}.
$$

Zero selects the original frequency; one selects full interpolation. Intermediate weights blend the two values. The coordinate pairs themselves are unchanged—what changes is the rotation rate assigned to each. Our example has distinct boundaries inside the spectrum; the teaching implementation rejects a collapsed range.

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

Once the ramp is understood, its implementation convention can be considered separately. Boundaries may be fractional even though pair indices are integers: the formula simply evaluates a transition weight at each integer index. The generic Hugging Face YaRN utility also offers `truncate=True`, which rounds the boundaries to `low=8`, `high=18`. `GptOssConfig` sets `truncate=False`, so the continuous values are the faithful default for this model. Both conventions express the same transition idea, but their intermediate weights differ.

In this particular configuration, the fully interpolated set coincides with pairs that complete less than one turn in the reference context. That is a property of the chosen boundary formula and hyperparameters, not a universal rule that one full turn makes extrapolation “safe.”

![VIZ m4/07 — diagnostic YaRN zones](assets/modern-llms/en/module-04/m4_07_rotations_zones.png)
*Turn counts help define the teaching YaRN ramp. The zones mean original, transition, and interpolated—not mathematically allowed and forbidden.*

So far we have changed angles: how far each coordinate pair turns. YaRN also has a separate adjustment to the scale of the scores entering softmax. Recall **temperature** from M3: it divides the logits, and a lower temperature makes a fixed set of scores produce a more concentrated distribution. An `attention_factor` can implement such a change by scaling the rotated vectors. The original gpt-oss implementation and its Hugging Face adaptation take this approach. For $s>1$, both checked paths use

$$
a=1+0.1\ln s.
$$

At $s=32$, $a=1.346574$. To understand what this number does, follow it through the computation. Both checked implementations multiply the rotary cosines and sines by it. The operation is now a rotation with an additional scale, not a pure rotation alone. If queries and keys both receive that factor, it enters their dot product twice, giving the score multiplier

$$
a^2=1.813260.
$$

The whole-score factor assumes that **every** query and key coordinate receives a. With partial RoPE, only the rotary contribution is scaled unless the remaining channels are adjusted separately. The fitted rule for a is empirical, not a consequence of rotation algebra.

Another runtime may move the equivalent scaling into queries, keys, or the attention score. A configuration field only becomes unambiguous when read with the implementation that consumes it.

### Beyond one scaling recipe

PI, the static NTK-aware rule, and YaRN illustrate three ways to reshape one starting spectrum. They do not exhaust the choices. Scaling factors can be selected more flexibly, and positional adjustments can be combined with different attention patterns. The methods below continue that exploration of trade-offs rather than form a single ladder of mandatory upgrades.

LongRoPE searches per-dimension factors and can use different regimes for different lengths. LongRoPE2 incorporates training-aware criteria. Llama3-style scaling divides the spectrum into wavelength bands. Other models use partial RoPE, interleaved RoPE/NoPE, local attention, or length-dependent attention temperature.

![VIZ m4/08 — three spectrum transformations](assets/modern-llms/en/module-04/m4_08_scaling_methods.png)
*PI, static NTK-aware scaling, and YaRN reshape the same starting spectrum differently. The figure compares geometry, not guaranteed downstream quality.*

## 8. ALiBi: a distance prior in the logits

Until now, position has entered the comparison by rotating vectors. Return to the map in §2: distance can also enter later, after the content comparison, as a direct change to the logit. That route needs no frequency spectrum. It is an alternative answer to the original positional question, not another RoPE scaling recipe.

ALiBi—Attention with Linear Biases—leaves queries and keys untouched. Head $h$ receives a slope $m_h$, and causal attention uses

$$
S_{ij}^{(h)}=\frac{q_i^Tk_j}{\sqrt{d_k}}-m_h(i-j),
\qquad j\le i.
$$

The slope $m_h>0$ is fixed per head and measured in logit units per position step. A separate causal mask excludes future keys; a finite ALiBi penalty cannot replace it.

Heads with steeper slopes favor nearby context more strongly. If the content score were fixed, softmax would turn the linear penalty into an exponential distance prior:

$$
\exp[-m_h(i-j)].
$$

The complete attention pattern is not a stationary kernel by itself; it still depends on content and normalization across available keys.

ALiBi's strength is structural simplicity and a bias that continues to any distance. Its limitation is equally clear: the form of the distance prior is fixed in advance.

One common measure of next-token prediction is **perplexity**, $\mathrm{PPL}=\exp(\overline{\mathrm{NLL}})$, using M3's mean negative log-likelihood in nats per evaluated token. On the same data, with the same tokenizer and evaluation mask, lower perplexity means that the model assigns higher probability, on average, to the observed continuations. Maintaining this score at longer lengths does not establish reliable retrieval from the middle of a document or reasoning across distant evidence. Those require separate evaluations.

## 9. NoPE and hybrid position mechanisms

We can take the question one step further. What order information remains if neither rotations nor positional biases are supplied? The answer depends on the attention structure itself. At the start of the chapter, we distinguished unmasked bidirectional attention from a causal model whose access to the future is restricted. That distinction now becomes central.

NoPE—No Positional Encoding—removes an explicit positional term. A causal model is not deprived of all order information: the mask, depth, and recursively constructed hidden states expose asymmetries tied to prefix length.

Research shows that substantial positional behavior can emerge under these conditions. That does not make explicit position universally unnecessary. It means the model can infer part of the structure from other components of the computation.

Two cases should be separated:

- **pure NoPE**, where no layer receives an explicit positional encoding;
- **hybrid designs**, where some layers use RoPE and others NoPE, or where positional mechanisms are combined with different attention patterns.

Llama 4 is a concrete hybrid. Its open configuration supports a NoPE layer at a regular interval, chunked attention, and length-dependent temperature tuning. In the [Llama 4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py), enabling `attn_temperature_tuning` scales queries in NoPE layers. This is separate from the YaRN frequency ramp.

## 10. Reading the gpt-oss positional configuration

This example uses the [original gpt-oss-120b configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/original/config.json) at revision `b5c939de…`. The module includes its bytes and checksum, so later edits to `main` cannot change the exercise. We checked it against OpenAI revision `7b583341…` and Transformers revision `536ecc007…` on 10 October 2026.

The relevant fields are

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

![VIZ m4/09 — interpreting a RoPE configuration](assets/modern-llms/en/module-04/m4_09_config_rope.svg)
*The gpt-oss fields support a reconstruction of the positional spectrum. They do not constitute a complete training schedule.*

## 11. Level-one code: explicit shapes and explicit scaling placement

We can now bring frequencies, positions, and rotations together in one computation. The following NumPy code is designed for inspection rather than fused execution. It validates head width, positions, frequency shape, and the YaRN ramp.

Notice the coordinate-layout convention before reading the code. Section 4 and the worked example in §5 used adjacent pairs. Here we use the other layout already discussed: split-half `rotate_half`, as found in Llama-like implementations. The first half of the vector is paired with the second. The rotation mathematics is unchanged, but comparing both layouts requires the corresponding coordinate permutation.

The compact `apply_rope` below requires equal Q/K shapes and one shared position vector. It represents a full sequence block; the notebook rotates queries and cached keys separately when their lengths differ. This core calculates in float64 and returns the YaRN amplitude for the caller to apply. It covers the declared scenario, not every library variant.

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
    if x.ndim < 2 or x.size == 0 or x.shape[-1] % 2:
        raise ValueError("x must have shape [..., T, D] with even D")
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)

def apply_rope(q, k, positions, inv_freq):
    """Apply split-half RoPE to q/k of shape [..., T, D]."""
    if any(np.iscomplexobj(x) for x in (q, k, inv_freq)):
        raise TypeError("q, k and inv_freq must be real")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    positions = np.asarray(positions)
    inv = np.asarray(inv_freq, dtype=np.float64)
    if q.shape != k.shape or q.ndim < 2 or q.size == 0 or q.shape[-1] % 2:
        raise ValueError("q and k must share shape [..., T, D], with even D")
    if not np.isfinite(q).all() or not np.isfinite(k).all():
        raise ValueError("q and k must be finite")
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError("positions must be an integer vector [T]")
    if positions.size != q.shape[-2] or np.any(positions < 0):
        raise ValueError("positions do not match the sequence axis")
    if inv.shape != (q.shape[-1] // 2,) or not np.isfinite(inv).all() or np.any(inv <= 0):
        raise ValueError("inv_freq must be positive and have shape [D/2]")

    angles = positions.astype(np.float64)[:, None] * inv[None, :]
    angles = angles.reshape((1,) * (q.ndim - 2) + angles.shape)
    if not np.isfinite(angles).all():
        raise FloatingPointError("phase overflow")
    full = np.concatenate((angles, angles), axis=-1)
    cos, sin = np.cos(full), np.sin(full)
    qr, kr = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    if not np.isfinite(qr).all() or not np.isfinite(kr).all():
        raise FloatingPointError("rotation overflow")
    return qr, kr

def yarn_inv_freq(head_dim, base, factor, original_context,
                  beta_fast=32.0, beta_slow=1.0, *, truncate=False):
    """Blend original and interpolated frequencies with a YaRN-style ramp."""
    inv = rope_inv_freq(head_dim, base)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("factor must be at least 1")
    if isinstance(original_context, bool) or not isinstance(original_context, int):
        raise TypeError("original_context must be an integer")
    if not math.isfinite(beta_fast) or not math.isfinite(beta_slow):
        raise ValueError("beta parameters must be finite")
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

So far, a position has been one number: the token's place in a text sequence. An image needs more. Its elements may be processed in a one-dimensional order, but their spatial locations have both a row and a column. Video adds time. We need to preserve those relationships without confusing an element's place in the processing stream with its coordinates in the original data.

The familiar rotation can accommodate this distinction. Rather than give every coordinate pair the same text index, we can assign different groups of pairs to different axes. An angle still comes from a position and a frequency; what changes is which position each group uses.

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

The small grid shows why separate coordinates are useful: moving along a row and moving along a column change different spatial axes. Video adds change along a third axis, time. Qwen2.5-VL can space temporal IDs according to real time between grid elements rather than treating frame number as the only clock. When audio accompanies the video, the time coordinate must also have a consistent meaning across the two streams.

TMRoPE—Time-aligned Multimodal Rotary Position Embedding—was documented in the Qwen2.5-Omni technical report in March 2025. It aligns audio and video timestamps. In that model, one temporal ID corresponds to 40 ms. This is a model-specific time grid, not a universal definition of TMRoPE.

![VIZ m4/10 — M-RoPE and TMRoPE](assets/modern-llms/en/module-04/m4_10_mrope.svg)
*Text repeats one position ID across all axes; images vary height and width at fixed time; video and synchronized audio use a temporal coordinate.*

### Interleaved axes in Qwen3-VL

Sectioned M-RoPE gives time, height, and width separate frequency groups. [Qwen3-VL, §2.1](https://arxiv.org/html/2511.21631v1#S2.SS1) instead interleaves axis assignments across frequency pairs, giving each axis access to faster and slower bands. This is **Interleaved-MRoPE**. Axis allocation is separate from the adjacent-versus-split-half layout of the two coordinates inside each pair.

The report also describes explicit textual video timestamps (§2.3). The 40 ms grid from Qwen2.5-Omni is therefore not a family-wide rule. Our 2×2 example implements sectioned M-RoPE, not a complete Qwen3-VL position system. Mixed text and visual sequences also need a convention for coordinate offsets and the text that follows; the toy grid starts at zero for clarity.

## 14. The 2026 landscape is a branching architecture map

This selective implementation map was checked on 10 October 2026. Use it as a reference after learning the mechanisms: each row points to evidence for a particular positional design, not a ranking or an exhaustive catalogue.

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
| Qwen3-VL | Interleaved-MRoPE and textual timestamps | frequency allocation across axes and video event timing | [report, §2](https://arxiv.org/html/2511.21631v1#S2) and open implementation |
| Inkling | relative positional embeddings with sliding/global layers | 1M context | official architecture announcement |
| Closed API models | may be undisclosed | advertised window does not reveal the formula | provider documentation only where available |

A context-window number does not identify the positional mechanism, and it does not establish reliable use of every position inside the window. Operational context quality is an evaluation problem, addressed later in Module 19.

## 17. Key takeaways

![VIZ m4/11 — positional mechanisms at a glance](assets/modern-llms/en/module-04/m4_11_cheatsheet.svg)
*Permutation symmetry, rotary relativity, spectral diagnostics, context-scaling families, ALiBi/NoPE, and multimodal axes on one page.*

- Bidirectional self-attention without position is permutation-equivariant. A causal mask supplies access order, not a complete distance geometry.
- Pure sinusoidal pairs already have an exact relative dot product. RoPE inserts relative phase directly into the projected query-key score.
- `rope_theta` redistributes a frequency ladder. A larger base slows some existing components while preserving the fastest pair and the number of coordinates.
- A single wavelength or turn count is not a hard context boundary.
- PI compresses every frequency; static NTK-aware scaling is a heuristic; YaRN blends original and interpolated spectra; other long-context families make different trade-offs.
- `attention_factor` must be interpreted at the point where the runtime applies it.
- ALiBi is a linear positional bias. NoPE can be pure or part of a hybrid stack.
- M-RoPE repeats IDs for text and uses multiple axes for visual tokens. TMRoPE aligns multimodal time.
- A configuration describes architectural geometry; it is not a substitute for a training report.

**Next:** [Module 5, attention mechanisms](module-05.md). With position accounted for, we can examine how heads are organized and how attention is computed without unnecessary memory use.

## 18. primary sources

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
- [Hugging Face RoPE utilities](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/modeling_rope_utils.py)
- [gpt-oss original configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/original/config.json)
- [gpt-oss reference RoPE implementation](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py)
- [Hugging Face GptOssConfig](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/gpt_oss/configuration_gpt_oss.py)
- [Llama 4 model documentation](https://huggingface.co/docs/transformers/model_doc/llama4)
- Qwen3-VL — [report, §2](https://arxiv.org/html/2511.21631v1#S2); [implementation](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Inkling architecture announcement](https://thinkingmachines.ai/news/introducing-inkling/)

**Further study.** Fourier features, kernels, and information geometry are developed in *Information Theory for ML*. Recurrent and state-space alternatives to explicit positional encoding appear in later *Modern LLMs* modules.

---

*Selected implementations checked: 10 October 2026. Spectrum and YaRN numbers belong to an explicitly stated teaching scenario. For closed models, an undisclosed positional mechanism is not inferred from the advertised context window.*
