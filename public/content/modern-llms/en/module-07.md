# Module 7. Beyond the Transformer: SSMs, Mamba, and Hybrid Sequence Models

*“Modern LLMs” course · Module 7 lecture · edition 2026.8*

> **What this module is about.** Softmax attention keeps the past as an addressable KV cache. That makes exact lookup possible, but the cache grows with every token. A **State Space Model (SSM)** takes the opposite bargain: after each token, update a fixed-size state. We will start from a continuous linear system and exact Zero-Order Hold discretization, then move through Mamba selectivity, Mamba-2’s State Space Duality, delta-rule memories, and current hybrid architectures. The recurring question is not whether recurrence “beats” attention. It is **which parts of the past can be compressed into a fixed state, and which still require an addressable store**.
>
> **Prerequisites.** Module 4 introduced KV-cache geometry; Module 5 covered the residual block; Module 6 is useful for the MoE hybrids later in the lecture. Everything needed for the argument is restated here.

---

## 1. Two costs of a long context

A Transformer pays for length in two related but different places. During dense prefill, attention considers token pairs and its central arithmetic is quadratic in sequence length. During autoregressive decode, each new query reads the accumulated keys and values, so both per-step work and cache size grow with the preceding context.

An SSM stores history in another form:

$$
h_t=f(h_{t-1},x_t),
$$

where $h_t$ has a fixed shape. Processing a sequence remains linear in length, and the recurrent cache does not expand as the conversation continues.

That resource win is not free. A KV cache resembles an indexed collection: a later query can point back to a specific record. A recurrent state is closer to a continuously rewritten summary. It can preserve the statistics that training makes useful, but independent facts may interfere, and arbitrary exact retrieval does not come for free.

Our first memory calculation already shows why the alternative matters. For a 32-layer Mamba-style stack with $d_{inner}=8192$, $d_{state}=16$, a convolution width of four, and bf16 state, the main SSM cache is 8 MiB and the causal-convolution cache another 2 MiB. The complete fixed cache is therefore 10 MiB. A reference Transformer cache at 128 KiB per token reaches the same size after only 80 tokens.

For a Mamba-2-like teaching geometry with $d_{state}=128$, the complete cache is 66.0625 MiB. The exact crossover is 528.5 tokens; token 529 is the first integer position beyond it. These values are not universal model constants. They demonstrate why the cache contract must include every persistent state, not only the tensor named `ssm_state`.

## 2. From control theory to hybrid language models

![VIZ m7/01 — the second architectural branch](assets/modern-llms/en/module-07/m7_01_ssm_timeline.svg)

State-space systems long predate neural language models. In control and filtering, a compact hidden state is designed to retain what is needed for future prediction. **S4 — Structured State Spaces for Sequence Modeling** brought that idea into deep sequence models and used structured dynamics to make very long convolutions tractable.

The weakness of early learned SSMs was not simply memory length. Their update law changed too little with content. **Mamba** made the input map, output map, and effective step size depend on the current token. The model could now decide that one symbol should be recorded strongly while another should pass with little effect. A hardware-aware selective scan made this input-dependent recurrence practical on GPUs. The paper reports up to a fivefold throughput improvement over its compared Transformer configurations; that is a result of the authors’ implementation and protocol, not a generic multiplier for every stack.

**Mamba-2** introduced **State Space Duality (SSD)**. For a structured class of transitions, the same sequence operator can be expressed as a recurrence or as a semiseparable masked matrix closely related to linear attention. This opened a path to dense matrix work inside chunks while carrying only compact state across chunk boundaries. The authors report a 2–8× core-layer speedup over Mamba-1 in their settings.

In parallel, linear-attention research revisited how memory should be written. **DeltaNet**, published in June 2024, used the delta rule to correct an existing association rather than merely adding another outer product. **Gated DeltaNet**, published later in December 2024, combined that mechanism with adaptive forgetting.

Large systems then converged on hybrids rather than one universal replacement. Jamba interleaved Transformer, Mamba, and MoE components. Falcon-H1 ran Mamba-2 and attention in parallel. Qwen3.5 used three Gated DeltaNet layers followed by one full-attention layer in each repeated unit. Kimi K3 combined 69 KDA layers with 24 Gated MLA layers. The recurring design lesson is more durable than any one ratio: **fixed state is attractive as a low-cost carrier, while addressable mechanisms are retained where compression is not enough**.

## 3. Three readings of the same recurrence

### A dynamical system

A continuous linear SSM is

$$
\dot h(t)=Ah(t)+Bx(t),
\qquad
y(t)=Ch(t)+Dx(t).
$$

$A$ controls internal evolution, $B$ writes the input, $C$ reads the state, and $D$ provides a direct input-to-output path.

### A bank of temporal filters

Stable modes decay at different rates. Fast coordinates react to local changes; slow ones carry longer summaries. Seen this way, an SSM is a learned spectrum of time scales.

### An associative memory

A linear-attention memory is often written as

$$
S_t=S_{t-1}+v_tk_t^\top,
\qquad y_t=S_tq_t.
$$

The key selects a write direction, the value provides the content, and the query reads from the resulting matrix. Unlike an explicit KV table, the matrix has fixed size, so independent associations eventually interfere.

These views are complementary. Dynamics explains stability and forgetting; filters explain time scales; associative memory explains writing and retrieval. Mamba-2 makes the connection more than rhetorical for its structured class of operators.

## 4. Discretization with Zero-Order Hold

Language arrives at discrete positions, so a continuous system must be discretized. Under **Zero-Order Hold (ZOH)**, the input is treated as constant for an interval of duration $\Delta$:

$$
h_t=\bar A h_{t-1}+\bar Bx_t,
$$

$$
\bar A=e^{\Delta A},
\qquad
\bar B=A^{-1}(e^{\Delta A}-I)B.
$$

For diagonal $A$, the calculation is elementwise. When one diagonal entry is zero, the expression for $\bar B$ looks singular, but the limit is well defined:

$$
\lim_{a\to0}\frac{e^{\Delta a}-1}{a}=\Delta.
$$

The module code handles that limit explicitly so a rare zero mode cannot create a hidden `NaN`.

The exact ZOH formula and a production Mamba parameterization are not the same object. In the open Mamba-2 implementation, the input update uses a term of the form $\Delta Bx$ inside its efficient formulation rather than materializing the exact ZOH input matrix above. The continuous model supplies intuition; the architecture chooses a trainable discrete update that can be implemented efficiently.

## 5. Worked example: the cost of replacing exact ZOH by $\Delta B$

Take one stable scalar mode:

$$
a=-1,
\qquad B=1.
$$

The exact input coefficient is

$$
\bar B_{ZOH}=1-e^{-\Delta},
$$

while the small-step approximation is $\Delta$. Its relative error is

$$
\varepsilon(\Delta)=\frac{\Delta}{1-e^{-\Delta}}-1.
$$

| $\Delta$ | exact | approximation | relative error |
|---:|---:|---:|---:|
| 0.1 | 0.09516 | 0.1 | 5.08% |
| 0.5 | 0.39347 | 0.5 | 27.07% |
| 1.0 | 0.63212 | 1.0 | 58.20% |

Solving

$$
\frac{\Delta}{1-e^{-\Delta}}-1=0.1
$$

gives

$$
\boxed{\Delta\approx0.193747558}.
$$

The point is not that every selective SSM must implement full ZOH. It is that $\Delta B$ and the exact held-input coefficient are distinct claims; a lecture or codebase should say which one it uses.

![VIZ m7/03 — discretization by hand](assets/modern-llms/en/module-07/m7_03_zoh_trace.svg)

## 6. What selectivity changes in Mamba

A time-invariant linear system applies the same write and forgetting law to every symbol. Language needs content-dependent control. Mamba predicts several quantities from the current input:

$$
B_t=B(x_t),
\qquad C_t=C(x_t),
\qquad \Delta_t=\operatorname{softplus}(W_\Delta x_t).
$$

$B_t$ controls writing, $C_t$ controls reading, and $\Delta_t$ changes the effective rate at which the state moves. A token can therefore preserve the previous state, overwrite part of it, or expose a different projection to the next block.

A short depthwise causal convolution appears before the selective recurrence. It is best understood as an explicit local inductive bias, not as evidence that recurrence cannot represent order. The convolution handles nearby patterns cheaply; the SSM carries longer dynamics.

That local path also owns persistent state at decode time. A serving implementation must keep both the SSM tensor and the convolution tail. Any memory estimate that counts only the first is incomplete.

![VIZ m7/04 — a selective Mamba block](assets/modern-llms/en/module-07/m7_04_mamba_block.svg)

## 7. Mamba-2 and the scope of State Space Duality

Mamba-2 focuses on a structured class of state transitions whose sequence matrix is semiseparable. The same operator can then be evaluated as:

1. a sequential recurrence;
2. an associative parallel scan;
3. a chunked matrix computation resembling masked linear attention.

![VIZ m7/02 — the scope of SSD](assets/modern-llms/en/module-07/m7_02_duality.svg)

The distinction matters. The slogan “Transformers are SSMs” does not mean any arbitrary continuous dynamical system is interchangeable with standard softmax attention. The theorem relies on a specific structured matrix class.

Nor should numerical equivalence be confused with bitwise equality. A recurrence, a scan, and a GEMM-based chunked implementation can evaluate the same mathematical operator while accumulating floating-point error in different orders.

The systems advantage is phase-dependent. Chunked dense work is attractive during training; recurrence is attractive during decode. The cited 2–8× improvement concerns the Mamba-2 core layer in the paper’s measured settings, not the total latency of every model built around it.

## 8. Delta-rule memory and adaptive forgetting

An additive memory writes

$$
S_t=S_{t-1}+v_tk_t^\top.
$$

Reusing a key adds the new value to the old one. Delta rule instead corrects the current prediction:

$$
S_t=S_{t-1}
+
\beta_t\bigl(v_t-S_{t-1}k_t\bigr)k_t^\top.
$$

For a normalized key and $\beta_t=1$, the readout along that direction becomes exactly $v_t$. A smaller $\beta_t$ performs a partial update.

Let

$$
k=(1,0)^\top,
\quad v_1=(1,2)^\top,
\quad v_2=(3,4)^\top.
$$

After writing $v_1$, a second additive write would return $(4,6)$ for the same key. Delta update returns $(2,3)$ at $\beta=0.5$ and exactly $(3,4)$ at $\beta=1$.

DeltaNet develops this form of targeted correction and a parallel training algorithm. Gated DeltaNet adds a separate forgetting gate. The mechanisms solve different problems: delta update repairs a selected association; gating can erase broader old content.

![VIZ m7/05 — editable associative memory](assets/modern-llms/en/module-07/m7_05_delta_trace.svg)

## 9. What a fixed state can and cannot preserve

Consider a deliberately simple associative memory:

$$
S=\sum_{i=1}^{m}v_i k_i^\top,
\qquad \hat v_i=Sk_i.
$$

Random normalized keys are not mutually orthogonal, so cross terms accumulate. In the frozen scenario with $d_k=d_v=32$, mean squared error rises from roughly 0.14 at four stored pairs to about 4.14 at 128.

![VIZ m7/06 — the capacity of a fixed matrix memory](assets/modern-llms/en/module-07/m7_06_recall.png)

A high-temperature softmax control nearly recovers the explicit values in this synthetic setup because it retains the whole key/value table. That does not prove that every Transformer retrieves perfectly; it isolates the difference between an addressable table and fixed-size compression.

The defensible conclusion is:

> fixed state creates an information bottleneck; retrieval quality depends on state size, state structure, the update rule, training, and the task.

Newer models such as Mamba-3 increase recurrent expressivity and improve state tracking in their experiments. They can shift the frontier without turning finite state into an unbounded exact database.

## 10. Memory accounting and core-only arithmetic

### Complete recurrent cache

For the reference Transformer:

$$
M_{KV/token}=2LH_{kv}d_hb=128\text{ KiB/token}.
$$

For the Mamba-1-like stack:

$$
M_{SSM}=8\text{ MiB},
\qquad M_{conv}=2\text{ MiB},
$$

$$
M_{recurrent}=10\text{ MiB},
\qquad N^*=80.
$$

For the Mamba-2-like teaching geometry, the main state is 64 MiB. The open implementation may cache x plus B/C channels in the convolution path, giving another 2.0625 MiB:

$$
M_{recurrent}=66.0625\text{ MiB}.
$$

The exact crossover is 528.5 tokens, and the first integer position after it is 529.

A 32-layer hybrid with four attention layers at 256K context retains 4 GiB of KV cache plus 8.75 MiB of recurrent state in the other 28 layers. The memory remains $O(N)$ because some layers still keep token-addressable records; the coefficient is simply much smaller.

### Central operator arithmetic

The teaching comparison

$$
4Nd_{model}
\quad\text{versus}\quad
3d_{inner}d_{state}
$$

produces ratios of about 42.7× at 1K, 1365× at 32K, and 43,691× at 1M. These are **core-operator arithmetic ratios**, not model speedups. Projections, convolution, gating, FFN, weight traffic, communication, and kernel efficiency are outside the denominator.

![VIZ m7/07 — state memory and core arithmetic](assets/modern-llms/en/module-07/m7_07_crossover.png)

## 11. Three hybrid patterns

### Interleaved layers

Jamba interleaves Transformer, Mamba, and MoE components. Qwen3.5 documents an exact repeating pattern:

$$
10\times
\left[
3\times(\text{Gated DeltaNet}\to\text{MoE})
+
1\times(\text{Gated Attention}\to\text{MoE})
\right].
$$

Seventy-five percent of the layers therefore carry recurrent or linear state, while one quarter retains a full-attention cache.

### Parallel branches

Falcon-H1 runs Mamba-2 and attention within the same block. This lets addressable and compressed representations interact at every depth, but it also executes the attention branch more frequently. Ablations of a trained checkpoint can reveal strong specialization between branches, yet they do not establish a universal role assignment for every hybrid architecture.

### Specialized mixtures

Kimi K3 has 93 layers: 69 Kimi Delta Attention layers and 24 Gated MLA layers. Attention Residuals is a separate depth-mixing mechanism, not the name of the MLA branch. Nemotron 3 Super and Ultra combine Mamba, attention, LatentMoE, and multi-token prediction at different scales—120B/12B and 550B/55B.

MiniMax demonstrates why family names are not architecture passports. MiniMax-01 and M1 used Lightning Attention. MiniMax M2 moved to full attention after the company reported inadequate long-context quality from the hybrid alternative in its experiments. MiniMax M3 later introduced MiniMax Sparse Attention, a sparse-attention branch rather than a fixed recurrent state.

![VIZ m7/08 — three ways to combine memory mechanisms](assets/modern-llms/en/module-07/m7_08_hybrid_layouts.svg)

## 12. Level-1 code: exact discretization, delta update, and scan composition

```python
import numpy as np

def zoh_diagonal(a, b, delta):
    """Exact ZOH for a diagonal A, including the A -> 0 limit."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    transition = np.exp(delta * a)
    factor = np.where(np.abs(a) > 1e-14, np.expm1(delta * a) / a, delta)
    return transition, factor[:, None] * b

def delta_step(state, key, value, beta, decay=1.0):
    """One gated delta update for a normalized key."""
    state = decay * state
    prediction = state @ key
    return state + beta * np.outer(value - prediction, key)

def compose_affine(later, earlier):
    """Compose h -> a*h+b transforms."""
    a2, b2 = later
    a1, b1 = earlier
    return a2 * a1, a2 * b1 + b2
```

The functions represent three different claims. The first is exact discretization of a stated continuous system. The second is an editable matrix memory. The third supplies the associative operator required for a parallel scan.

A production kernel adds batched shapes, input-dependent parameters, fused causal convolution, Tensor Core layouts, and precision engineering. The teaching implementation is intentionally transparent rather than fast.

## 14. Reading the open Mamba implementation

A Mamba-1 inference cache commonly contains:

- `conv_state` with a shape like `[batch, d_inner, d_conv]`;
- `ssm_state` with shape `[batch, d_inner, d_state]`.

In Mamba-2, convolution channels may include x plus B/C components. A memory estimate therefore needs the actual implementation’s cache shapes, not just the headline `d_state`.

The source code also separates mathematical inspiration from executed update. Mamba-2’s efficient input branch uses a $\Delta Bx$ term. A source audit should therefore ask both:

1. which continuous-time picture motivates the model;
2. which discrete parameterization and kernel the checkpoint actually executes.

![VIZ m7/09 — reading recurrent-cache geometry](assets/modern-llms/en/module-07/m7_09_config_hybrid.svg)

## 15. A current architecture passport

| Family | Sequence mechanism | Published contract | Context | Evidence |
|---|---|---|---:|---|
| Mamba | selective SSM | attention-free backbone in the original work | up to 1M in author experiments | paper + code |
| Mamba-2 | SSD/SSM | recurrence and chunked matrix form | checkpoint-dependent | paper + code |
| Jamba | Mamba + attention + MoE | interleaved hybrid | up to 256K | paper/release |
| Falcon-H1 | Mamba-2 ∥ attention | parallel hybrid, 0.5B–34B | up to 256K | technical report |
| Qwen3.5-35B-A3B | 3 Gated DeltaNet : 1 full attention | 40 layers, 35B/3B, MoE | 262K native, ~1.01M extension | official card/config |
| Kimi K3 | 69 KDA + 24 Gated MLA | AttnRes is separate, 2.8T/104B | 1M | report/repository |
| Nemotron 3 Super | Mamba–Attention + LatentMoE + MTP | 120B/12B | up to 1M | official report |
| Nemotron 3 Ultra | Mamba–Attention + LatentMoE + MTP | 550B/55B | up to 1M | official report |
| MiniMax M2 | full attention | not a Lightning continuation | about 196K in open config | official note/config |
| MiniMax M3 | sparse attention | MSA, not fixed recurrent state | 1M report setting | paper/developer result |
| Mamba-3 | expressive recurrent SSM | complex/data-dependent state, MIMO | research setting | paper |
| SISA | SSM-derived term inside SDPA | retains the KV cache | research setting | paper |

The evidence column is part of the architecture description. For a closed model or incomplete report, “not disclosed” is more accurate than inferring an internal mechanism from a context-window number.

## 16. Audio, video, and other long streams

Recurrent carriers are attractive when a modality creates many sequential time steps: audio, video, sensor logs, genomics, or long agent trajectories. Fixed state lets the model keep processing without a cache that grows linearly with elapsed time.

Those modalities are not necessarily uniform, and they may still need exact recall of a particular event. A single recurrent state makes arbitrary random access difficult. Systems can respond with occasional attention, external retrieval, hierarchical memories, event indexes, or parallel recurrent and addressable branches.

A multimodal serving passport should therefore state both the recurrent state size and the mechanism used to revisit individual past events.

## 19. Key takeaways

![VIZ m7/10 — SSM and hybrid cheat sheet](assets/modern-llms/en/module-07/m7_10_cheatsheet.svg)

- An SSM replaces an expanding list of past tokens with a fixed-size state.
- Mamba makes writing, reading, and forgetting input-dependent.
- A serving cache includes both SSM and causal-convolution state.
- SSD provides multiple execution forms for a structured class of SSMs.
- Delta rule edits an association; gating adds adaptive forgetting.
- Fixed state saves memory but creates an information bottleneck.
- Hybrids restore addressability while retaining length-dependent KV cache in some layers.
- Core-operator FLOPs are not end-to-end latency.
- Product-family names do not guarantee architectural continuity.
- A model should be read through layer types, cache shapes, state geometry, and source quality.

## 20. Notebook and primary sources

**Primary sources:**

- Gu and Dao, [Mamba](https://arxiv.org/abs/2312.00752)
- Dao and Gu, [Mamba-2 / State Space Duality](https://arxiv.org/abs/2405.21060)
- Yang et al., [DeltaNet](https://arxiv.org/abs/2406.06484)
- Yang et al., [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
- Lieber et al., [Jamba](https://arxiv.org/abs/2403.19887)
- [Open Mamba implementation](https://github.com/state-spaces/mamba)
- [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Kimi K3 repository and report](https://github.com/MoonshotAI/Kimi-K3)
- [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/)
- [Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
- [Why MiniMax M2 uses full attention](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)
- [MiniMax M3](https://arxiv.org/abs/2606.13392)
- [Falcon-H1](https://arxiv.org/abs/2507.22448)
- [Mamba-3](https://arxiv.org/abs/2603.15569)
- [SISA](https://arxiv.org/abs/2606.02332)

**Next:** Module 7b moves from architecture to hardware: rooflines, arithmetic intensity, bandwidth, and why identical asymptotics can still produce very different measured speed.

*Landscape verified: 6 August 2026; architecture claims were checked against papers, official cards, configurations, or open code, while developer-reported results retain their original protocols.*
---

*Volatile architecture facts were checked on 6 August 2026. Developer-reported measurements retain their original protocols; the memory and arithmetic values in this module are reproducible teaching scenarios.*
