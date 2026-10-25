# Module 8. Beyond the Transformer: SSMs, Mamba, and Hybrid Sequence Models

*“Modern LLMs” course · Module 8 lecture · edition 2026.10.25*

> **What this module is about.** Softmax attention keeps the past as an addressable KV cache. That preserves separately addressable positions, but the cache grows with every token. A **State Space Model (SSM)** takes the opposite bargain: after each token, update a fixed-size state. We will start from a continuous linear system and exact Zero-Order Hold discretization, then move through Mamba selectivity, Mamba-2’s State Space Duality, delta-rule memories, and current hybrid architectures. The recurring question is not whether recurrence “beats” attention. It is **which parts of the past can be compressed into a fixed state, and which still require an addressable store**.
>
> **Prerequisites.** Module 5 introduced KV-cache geometry; Module 6 covered the residual block; Module 7 is useful for the MoE hybrids later in the lecture. Everything needed for the argument is restated here.

---

### Reading routes

**First pass:** §1 → §§3–4 → §§6–11 → §19, followed by the [existing module tests](/courses/modern-llms/module-08/tests/). In §4, first understand the discrete recurrence and its matrices; save the exact-coefficient derivation and [exercise E1](/courses/modern-llms/module-08/exercises/) for the second pass. Exercises E2–E3 on the same page provide the first memory calculations. Follow the change from a list of past tokens to a fixed state, then study writing, forgetting, and hybrids. Return to the opening memory figures when you reach §10.

**Second pass:** §§4–5, §7, and §§12 and 14 with the [module notebook](/courses/modern-llms/module-08/notebooks/). Derive the discrete update and test composition, chunk boundaries, and numerical limits.

**Reference pass:** §2, §15, and §20 provide history, model specifications, and sources. They introduce names as pointers; the mechanisms are developed in §§3–11.

## 1. Two costs of a long context

A Transformer pays for length in two related but different places. During dense prefill, attention considers token pairs and its central arithmetic is quadratic in sequence length. During autoregressive decode, each new query reads the accumulated keys and values, so both per-step work and cache size grow with the preceding context.

In the preceding modules, we usually followed a token representation through the depth of the network. Each block read the residual stream and added an update. Here we look in a different direction: hold one layer fixed and ask what it carries from one token to the next. This state across sequence positions is distinct from the representation passed between layers.

An SSM stores history in another form:

$$
h_t=f(h_{t-1},x_t),
$$

where $h_t$ has a fixed shape. Each new token changes the contents of this state instead of appending a separately stored record. With the layer dimensions held fixed, the sequence takes linear work to process, and the recurrent cache does not expand as the conversation continues. This is an inference-state claim; retaining intermediate computations for training is a separate memory question.

That resource win is not free. A KV cache resembles an indexed collection: a later query can point back to a specific record. A recurrent state is closer to a continuously rewritten summary. It can preserve the statistics that training makes useful, but independent facts may interfere, and arbitrary exact retrieval does not come for free.

Our first memory calculation already shows why the alternative matters. For a 32-layer Mamba-style stack with $d_{inner}=8192$, $d_{state}=16$, a convolution width of four, and bf16 state, the main SSM cache is 8 MiB and the causal-convolution cache another 2 MiB. The complete fixed cache is therefore 10 MiB. A reference Transformer cache at 128 KiB per token reaches the same size after only 80 tokens.

For a Mamba-2-like teaching geometry with $d_{state}=128$, the complete cache is 66.0625 MiB. The exact crossover is 528.5 tokens; token 529 is the first integer position beyond it. These values are not universal model constants. They demonstrate why the cache contract must include every persistent state, not only the tensor named `ssm_state`.

## 2. From control theory to hybrid language models

![VIZ m8/01 — the second architectural branch](assets/modern-llms/en/module-08/m8_01_ssm_timeline.svg)
*Classical state-space systems → HiPPO/S4 → selective Mamba → Mamba-2/SSD and DeltaNet → Jamba, Falcon-H1, Qwen3.5, and Kimi hybrids → new recurrent and sparse branches in 2026.*

State-space systems long predate neural language models. In control and filtering, a compact hidden state is designed to retain what is needed for future prediction. **S4 — Structured State Spaces for Sequence Modeling** brought that idea into deep sequence models and used structured dynamics to make very long convolutions tractable.

The weakness of early learned SSMs was not simply memory length. Their update law changed too little with content. **Mamba** made the input map, output map, and effective step size depend on the current token. The model could now decide that one symbol should be recorded strongly while another should pass with little effect. A hardware-aware selective scan made this input-dependent recurrence practical on GPUs. The paper reports up to a fivefold throughput improvement over its compared Transformer configurations; that is a result of the authors’ implementation and protocol, not a generic multiplier for every stack.

**Mamba-2** introduced **State Space Duality (SSD)**. For a structured class of transitions, the same sequence operator can be expressed as a recurrence or as a semiseparable masked matrix closely related to linear attention. This opened a path to dense matrix work inside chunks while carrying only compact state across chunk boundaries. The authors report a 2–8× core-layer speedup over Mamba-1 in their settings.

In parallel, linear-attention research revisited how memory should be written. The June 2024 **DeltaNet** paper developed efficient parallel training for a memory that uses the delta rule to correct an existing association rather than merely adding another outer product. **Gated DeltaNet**, published later in December 2024, combined that mechanism with adaptive forgetting.

Large systems then converged on hybrids rather than one universal replacement. Jamba interleaved Transformer, Mamba, and MoE components. Falcon-H1 ran Mamba-2 and attention in parallel. Qwen3.5 used three Gated DeltaNet layers followed by one full-attention layer in each repeated unit. Kimi K3 combined 69 KDA layers with 24 Gated MLA layers. The recurring design lesson is more durable than any one ratio: **fixed state is attractive as a low-cost carrier, while addressable mechanisms are retained where compression is not enough**.

## 3. Three readings of the same recurrence

There are three useful ways to examine a compact memory. We can specify how its state evolves, ask which time scales that evolution preserves, or organize the state to store associations. The first two views describe the same linear system. The third introduces a related matrix-memory construction; SSD will later tell us when a precise correspondence is available.

### A dynamical system

Continuous time may seem an odd starting point for a stream of tokens. Its value is that it gives us a language for rates of change and decay. We can specify those rates first, then ask what they do over one discrete step. This is a model of internal dynamics, not an assumption that the gaps between written words are physical time intervals.

A continuous linear SSM is

$$
\dot h(t)=Ah(t)+Bx(t),
\qquad
y(t)=Ch(t)+Dx(t).
$$

$A$ controls internal evolution, $B$ writes the input, $C$ reads the state, and $D$ provides a direct input-to-output path.

Updating the memory and reading an output from it are separate operations. Previously stored information can evolve even when no new input arrives. The current input contributes to that evolution, while the readout determines what part of the resulting state is exposed. The direct path also lets the current input affect the output without first being stored.

The state $h\in\mathbb R^n$, input $x\in\mathbb R^p$, and output $y\in\mathbb R^r$ fix the matrix shapes: $A:n\times n$, $B:n\times p$, $C:r\times n$, and $D:r\times p$. The dot denotes a continuous-time derivative; tokens will instead index discrete updates.

### A bank of temporal filters

If every eigenvalue of $A$ has a negative real part, the unforced state tends to zero over time: the system is asymptotically stable. This does not require each coordinate to decrease monotonically; a general matrix can produce transient growth. In the diagonal examples below, individual negative coefficients directly set different decay times. The system can then be read as a bank of fast and slow temporal filters.

The filter view is not an additional mechanism attached to the dynamics. It describes the same computation in terms of how long an input continues to influence the output. Different decay rates let a single state respond to recent changes while retaining traces of a longer history.

### An associative memory

Now ask a more specific memory question: how can a state retain a relationship between a key and a value? A matrix provides one answer. Instead of allocating a new row in an expanding KV table, each write changes the same array that already represents earlier associations.

A linear-attention memory is often written as

$$
S_t=S_{t-1}+v_tk_t^\top,
\qquad y_t=S_tq_t.
$$

The key selects a write direction, the value provides the content, and the query reads from the resulting matrix. That read combines contributions stored in a shared object rather than retrieving an isolated row. Non-orthogonal key directions can therefore interfere with one another. Section 9 will examine this interference explicitly.

For the associative form, $k_t,q_t\in\mathbb R^{d_k}$, $v_t\in\mathbb R^{d_v}$, and $S_t\in\mathbb R^{d_v\times d_k}$. The outer product writes one association; multiplication by the query reads a value-sized vector. This is unnormalized linear attention, with no softmax.

These views are complementary. Dynamics explains stability and forgetting; filters explain time scales; associative memory explains writing and retrieval. Mamba-2 makes the connection more than rhetorical for its structured class of operators.

## 4. Discretization with Zero-Order Hold

Having chosen a continuous description, we need to turn it into an operation performed once per input element. We could have defined a discrete recurrence directly; deriving one from a continuous system instead makes its coefficients interpretable. The question is what happens to the existing state, and how much the new input contributes, during a single step. Under **Zero-Order Hold (ZOH)**, the input is treated as constant for an interval of duration $\Delta$.

This duration is a step in the model’s internal dynamics. Holding the input fixed does not freeze the state: the state keeps evolving throughout the interval, under both its previous dynamics and the sustained input. At the end of the interval, the update separates into two terms:

$$
h_t=\bar A h_{t-1}+\bar Bx_t,
$$

The bars distinguish one-step discrete matrices from the continuous system: $\bar A$ propagates the previous state, while $\bar B$ adds the current input. The first term carries the past; the second accounts for the present input. But that input is not applied only at the final instant. A contribution introduced early in the interval also evolves before the interval ends. The exact write coefficient must collect all of those contributions. The recurrence is enough for a first reading of Mamba; the following equations explain where its coefficients come from:

$$
\bar A=e^{\Delta A},
\qquad
\bar B=\int_0^\Delta e^{uA}B\,du.
\quad (\text{any }A)
$$

For invertible $A$, the integral equals $A^{-1}(e^{\Delta A}-I)B$. The integral also covers singular matrices. The numerical examples set the direct path $D$ to zero.

The matrix exponential propagates the existing state across the whole interval. The integral accumulates the held input, allowing each contribution its remaining time to evolve. Thus the write coefficient depends on the dynamics as well as the step length; it is not merely the input map multiplied by a duration.

For diagonal $A$, the calculation is elementwise. When one diagonal entry is zero, the expression for $\bar B$ looks singular, but the limit is well defined:

$$
\lim_{a\to0}\frac{e^{\Delta a}-1}{a}=\Delta.
$$

The module code handles that limit explicitly so a rare zero mode cannot create a hidden `NaN`.

The exact ZOH formula and a production Mamba parameterization are not the same object. In the open Mamba-2 implementation, the input update uses a term of the form $\Delta Bx$ inside its efficient formulation rather than materializing the exact ZOH input matrix above. The continuous model supplies intuition; the architecture chooses a trainable discrete update that can be implemented efficiently.

Accuracy relative to a prescribed continuous system is therefore a different question from the quality of a trained language model. A model with another discrete update is trained with that update in place. The next example isolates the first question: how far two input coefficients differ when the original continuous dynamics and step are held fixed.

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

![VIZ m8/03 — discretization by hand](assets/modern-llms/en/module-08/m8_03_zoh_trace.svg)
*An exact two-mode ZOH trace, the zero-eigenvalue limit, and the growth of the $\Delta B$ approximation error.*

## 6. What selectivity changes in Mamba

A time-invariant linear system already responds to content: different input vectors produce different updates. What stays fixed is the rule that writes the input, carries the old state forward, and reads the result. Selectivity lets the content influence those rules themselves. A name, a closing bracket, and a frequent article may call for different treatment of memory, not just different vectors passed through the same treatment. Mamba predicts write and read maps and a positive step from the current input. Here $\operatorname{softplus}(u)=\log(1+e^u)$ ensures a positive step; the expression is schematic:

$$
B_t=B(x_t),
\qquad C_t=C(x_t),
\qquad \Delta_t=\operatorname{softplus}(W_\Delta x_t).
$$

$B_t$ controls writing, $C_t$ controls reading, and $\Delta_t$ sets how far the system advances through its internal dynamics. For the stable diagonal modes considered here, a larger step attenuates the previous state more strongly; a smaller one leaves it closer to its earlier value. The size of the final change also depends on the new input, so a larger step need not mean a larger update norm. Changing the read map can expose different features of the same memory without first rewriting all of it.

The three roles from the linear model—carry, write, and read—are still recognizable, but their use is now content-dependent. The recurrence is only the central operation of the neural block, however: its input must also be prepared and its output passed onward.

A short depthwise causal convolution appears before the selective recurrence. It is best understood as an explicit local inductive bias, not as evidence that recurrence cannot represent order. The convolution handles nearby patterns cheaply; the SSM carries longer dynamics.

That local path also owns persistent state at decode time. A serving implementation must keep both the SSM tensor and the convolution tail. Any memory estimate that counts only the first is incomplete.

![VIZ m8/04 — a selective Mamba block](assets/modern-llms/en/module-08/m8_04_mamba_block.svg)
*Input projection splits a local path and a gate; convolution feeds the selective SSM, whose output is multiplied by the gate. The two persistent caches are shown separately.*

## 7. Mamba-2 and the scope of State Space Duality

During generation, recurrence fits the workload: a new token arrives, the state is updated, and an output is read. During training, the layer already has input representations for the whole sequence. Must its states still be evaluated one at a time, or can the same dependence on the past be organized differently?

Mamba-1 already uses a parallel selective scan. Mamba-2 does not introduce parallel recurrence from scratch; it connects a particular transition structure to an especially useful matrix computation. Its sequence matrix is semiseparable, a property we will interpret by following how one input reaches a later output. The same structured operator can be evaluated as:

1. a sequential recurrence;
2. an associative parallel scan;
3. a chunked matrix computation resembling masked linear attention.

The apparent obstacle is that each state needs the preceding one. But a state and the description of a transformation are different objects. Once transition coefficients have been computed from the input sequence, two neighboring transformations can be composed without knowing the state that will enter them. Their composition is another transformation, with exactly the effect of applying the first and then the second.

A *scan* computes every prefix state, not just the final state. Represent the affine step $h\mapsto ah+b$ by $(a,b)$. Composition gives
$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,\;a_2b_1+b_2).
$$
Associativity permits regrouping, not reordering: $(2,1)$ followed by $(3,4)$ yields $(6,7)$, whereas the reverse order yields $(6,9)$.

We can therefore combine neighboring pairs, then larger stretches, to obtain the transformation associated with every prefix. Applying each prefix transformation to the initial state gives the corresponding state. The dependence on the past has not disappeared. What changed is the schedule for composing transformations, allowing parallel work while preserving token order.

A second view asks how much each input contributes to each later output. Expanding the recurrence produces a matrix of connections between sequence positions. For this calculation, treat the coefficients for the current sequence as already given. This does not make the complete selective model linear in the original input: the coefficients themselves may be input-dependent.

For this representation, take scalar input $x_t$ and output $y_t$, state $h_t\in\mathbb R^n$, and vectors $B_t,C_t\in\mathbb R^n$. Readout is $y_t=C_t^\top h_t$; the vector $C_t$ represents the readout row used in §3. With transition $a_tI$ and zero initial state, the coefficient mapping input $j$ to output $i\ge j$ is
$$
M_{ij}=C_i^\top B_j\prod_{r=j+1}^{i}a_r.
$$
Read the equation as the journey of a single contribution. The write vector at the earlier position puts the scalar input into the state. Each intervening transition carries or attenuates it. The read vector at the later position determines what reaches the output. The product of transitions accounts for that entire journey between writing and reading.

Now place a boundary between an earlier stretch and a later one. Everything the earlier inputs can contribute to later outputs passes through the state at that boundary. Those inputs need not cross as separate records. Consequently, the corresponding block below the diagonal factors through a fixed-dimensional space. This bounded rank of cross-boundary blocks is the useful meaning of *semiseparable* here; it does not require the entire sequence matrix to have low rank.

This suggests the chunked computation. Dense matrix operations handle interactions inside a short stretch. Contributions from previous stretches arrive through a boundary state, and the current stretch prepares a state for those that follow. Different parts of the work use different computational forms, without changing the operator or materializing an arbitrary full interaction matrix.

![VIZ m8/02 — the scope of SSD](assets/modern-llms/en/module-08/m8_02_duality.svg)
*One structured operator, three execution forms. SSD is a duality for a defined class, not an identity between every SSM and every linear-attention method.*

The distinction matters. The slogan “Transformers are SSMs” does not mean any arbitrary continuous dynamical system is interchangeable with standard softmax attention. The theorem relies on a specific structured matrix class.

Nor should numerical equivalence be confused with bitwise equality. A recurrence, a scan, and a GEMM-based chunked implementation can evaluate the same mathematical operator while accumulating floating-point error in different orders.

The systems advantage is phase-dependent. Chunked dense work is attractive during training; recurrence is attractive during decode. The cited 2–8× improvement concerns the Mamba-2 core layer in the paper’s measured settings, not the total latency of every model built around it.

## 8. Delta-rule memory and adaptive forgetting

So far we have asked how to carry a state through the sequence and compute the result efficiently. Now return to the write operation itself. What should happen when new information corrects an existing association? A faster implementation of the old computation cannot answer that question; the memory needs an appropriate update rule.

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

For a normalized key and $\beta_t=1$, the readout along that direction becomes exactly $v_t$. A $\beta_t$ between zero and one performs a partial update. Earlier work had already used the delta rule for associative memory.

Let

$$
k=(1,0)^\top,
\quad v_1=(1,2)^\top,
\quad v_2=(3,4)^\top.
$$

After writing $v_1$, a second additive write would return $(4,6)$ for the same key. Delta update returns $(2,3)$ at $\beta=0.5$ and exactly $(3,4)$ at $\beta=1$.

DeltaNet develops this form of targeted correction and a parallel training algorithm. Gated DeltaNet adds a separate forgetting gate. The mechanisms solve different problems: delta update repairs a selected association; gating can erase broader old content.

Let $\gamma_t\in[0,1]$ control global forgetting. Decay precedes the targeted correction:
$$
S_t=\gamma_t S_{t-1}+\beta_t(v_t-\gamma_t S_{t-1}k_t)k_t^\top.
$$
For a unit key the readout is $(1-\beta_t)\gamma_t S_{t-1}k_t+\beta_t v_t$, so $\beta_t=1$ replaces it exactly even when $\gamma_t<1$. Other non-orthogonal key directions can change too; a targeted update does not make all associations independent.

![VIZ m8/05 — editable associative memory](assets/modern-llms/en/module-08/m8_05_delta_trace.svg)
*Additive write, partial correction, exact replacement, and a gated update for one key direction.*

## 9. What a fixed state can and cannot preserve

The delta rule showed that the write policy matters. To isolate interference in a shared state, we now return deliberately to the simplest additive memory. The next experiment is not a benchmark of the DeltaNet update just described. It examines one source of error: overlapping associations in a single matrix:

$$
S=\sum_{i=1}^{m}v_i k_i^\top,
\qquad \hat v_i=Sk_i.
$$

Random normalized keys are not mutually orthogonal, so cross terms accumulate. In the frozen scenario with $d_k=d_v=32$, mean squared error rises from roughly 0.14 at four stored pairs to about 4.14 at 128.

![VIZ m8/06 — the capacity of a fixed matrix memory](assets/modern-llms/en/module-08/m8_06_recall.png)
*Interference in a toy additive linear associative memory. This is not a Mamba benchmark and not a theorem about every SSM.*

A low-temperature softmax control (temperature 0.05, or a logit multiplier of 20) nearly recovers the explicit values in this synthetic setup because it retains the whole key/value table. That does not prove that every Transformer retrieves perfectly; it isolates the difference between an addressable table and fixed-size compression.

The defensible conclusion is:

> at fixed numerical precision, fixed state creates an information bottleneck; retrieval quality depends on state size, state structure, the update rule, training, and the task.

Newer models such as Mamba-3 increase recurrent expressivity and improve state tracking in their experiments. They can shift the frontier without turning finite state into an unbounded exact database.

The plotted methods share the same random keys and values at each load. For independent isotropic unit keys in $d_k$ dimensions and independent zero-mean, unit-variance value coordinates, expected additive-memory MSE per coordinate is $(m-1)/d_k$. The dashed line shows that expectation, not a claim that one seeded sample must equal it.

The dependence has a direct interpretation. When one pair is read, all the other pairs contribute cross terms. More stored pairs create more of these terms; a wider random-key space weakens their average overlap. The formula describes that mechanism under the stated assumptions, rather than a universal quality curve for recurrent models.

## 10. Memory accounting and core-only arithmetic

### Complete recurrent cache

We can now return to the opening memory figures with a clearer inventory of what must survive between tokens. The comparison is between explicitly chosen cache geometries, not between models already known to deliver equal quality.

All sizes below cover one sequence and the stated layer stack, excluding model weights and workspaces. Let $L$ count layers, $H_{kv}$ count KV heads, $d_h$ be head width, and $b$ be bytes per value. Use $L=32$, $H_{kv}=8$, $d_h=128$, and $b=2$. The recurrent geometry uses $d_{inner}=8192$ channels, $d_{state}$ state coordinates per channel, and a stored convolution width $d_{conv}=4$. Here K=1024 and M=1024² tokens.

For the reference Transformer:

$$
M_{KV/token}=2LH_{kv}d_hb=128\text{ KiB/token}.
$$

For the Mamba-1-like stack:

$$
M_{SSM}=L d_{inner}d_{state}b=8\text{ MiB},
\qquad M_{conv}=L d_{inner}d_{conv}b=2\text{ MiB},
$$

$$
M_{recurrent}=10\text{ MiB},
\qquad N^*=80.
$$

For the Mamba-2-like geometry, use $d_{state}=128$: $L d_{inner}d_{state}b=64$ MiB. The open implementation can cache x plus B/C channels. With one B/C group, this gives $8192+2\cdot128=8448$ convolution channels and $L\cdot8448\cdot d_{conv}\cdot b=2.0625$ MiB:

$$
M_{recurrent}=66.0625\text{ MiB}.
$$

The difference between these recurrent examples comes from state geometry, not from how much text has already been read. The second state is wider, and its convolution cache includes additional channels. Fixed size means independent of sequence length; it does not mean that every SSM uses the same number of bytes.

The exact crossover is 528.5 tokens, and the first integer position after it is 529.

A 32-layer hybrid with four attention layers at 256K context retains 4 GiB of KV cache plus 8.75 MiB of recurrent state in the other 28 layers. The memory remains $O(N)$ because some layers still keep token-addressable records; the coefficient is simply much smaller.

### Central operator arithmetic

So far we have counted what remains in memory between steps. We now change the question to the arithmetic needed for one new input, specifying dimensions and the boundary of the counted operation separately.

Count one layer at one decode step. Set $d_{model}=4096$, $d_{inner}=8192$, and $d_{state}=16$, returning to the Mamba-1-like state width for this separate arithmetic example. Let $N$ be the number of visible keys. Count a multiplication and an addition as two operations, even if hardware fuses them.

Attention performs QK dot products and a weighted sum of V, each costing approximately $2Nd_{model}$ operations. For a diagonal SSM with its coefficients already available, $h^+=a\odot h+b\odot x$ costs two multiplications and one addition per state coordinate. The readout $y=C h^+$ contributes approximately two more operations per coordinate. Thus the leading estimates are

$$
4Nd_{model}
\quad\text{versus}\quad
5d_{inner}d_{state}
$$

giving ratios of 25.6× at 1K, 819.2× at 32K, and 26,214.4× at 1M. An exact scalar count uses one fewer readout addition per channel; the leading estimate drops that small correction. These are **restricted arithmetic comparisons**. They exclude SSM coefficient preparation, softmax, projections, convolution, gates, FFNs, weight movement, and execution efficiency, so they do not predict whole-model speed.

![VIZ m8/07 — state memory and core arithmetic](assets/modern-llms/en/module-08/m8_07_crossover.png)
*Top: complete recurrent cache versus a growing KV cache. Bottom: central token-mixing arithmetic only, not end-to-end latency.*

The Mamba-2 example assumes one B/C group and all 8192 channels assigned to the SSM. More generally, convolution has $d_{ssm}+2g d_{state}$ channels for $g$ groups. SSM and convolution states may use different dtypes; using bf16 for both is a scenario assumption.

## 11. Three hybrid patterns

We can now see both sides of the trade-off. Compact state makes continuation cheaper, but does not replace individually addressable records for every task. A hybrid keeps both mechanisms. Its design depends not only on their proportions, but also on where they meet: at different depths or inside the same block.

### Interleaved layers

Jamba interleaves Transformer, Mamba, and MoE components. Qwen3.5-35B-A3B documents an exact repeating pattern:

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

![VIZ m8/08 — three ways to combine memory mechanisms](assets/modern-llms/en/module-08/m8_08_hybrid_layouts.svg)
*Interleaved stacks, parallel branches, and specialized hybrids. Layer ratios are checkpoint properties, not universal standards.*

## 12. Level-1 code: exact discretization, delta update, and scan composition

```python
import numpy as np

def zoh_diagonal(a, b, delta):
    """Diagonal A: [n], B: [n,p], positive scalar step; caller validates shapes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    transition = np.exp(delta * a)
    factor = np.full_like(a, float(delta))
    np.divide(np.expm1(delta * a), a, out=factor, where=a != 0)
    return transition, factor[:, None] * b

def delta_step(state, key, value, beta, decay=1.0):
    """State [dv,dk], unit key [dk], value [dv]; decay before correction."""
    base = decay * state
    prediction = base @ key
    return base + beta * np.outer(value - prediction, key)

def compose_affine_pair(later, earlier):
    """Elementwise transforms; returns later(earlier(h)), not the reverse."""
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

Mamba-2 stores SSM state as `[batch, nheads, headdim, d_state]`, where `nheads * headdim = d_ssm`. Convolution channels can include x plus B/C components, with width `d_ssm + 2 * ngroups * d_state`. These shapes and the number of groups determine the cache; the headline `d_state` alone is insufficient.

The source code also separates mathematical inspiration from executed update. Mamba-2’s efficient input branch uses a $\Delta Bx$ term. A source audit should therefore ask both:

1. which continuous-time picture motivates the model;
2. which discrete parameterization and kernel the checkpoint actually executes.

![VIZ m8/09 — reading recurrent-cache geometry](assets/modern-llms/en/module-08/m8_09_config_hybrid.svg)
*A recurrent serving passport: layer types, SSM state, convolution state, channel counts, state width, and dtype.*

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

![VIZ m8/10 — SSM and hybrid cheat sheet](assets/modern-llms/en/module-08/m8_10_cheatsheet.svg)
*Exact discretization, selectivity, two cache states, SSD, delta rule, fixed-state bottlenecks, and hybrid layouts on one page.*

- An SSM replaces an expanding list of past tokens with a fixed-size state.
- Mamba makes writing, reading, and forgetting input-dependent.
- A serving cache includes both SSM and causal-convolution state.
- SSD provides multiple execution forms for a structured class of SSMs.
- Delta rule edits an association; gating adds adaptive forgetting.
- Fixed state saves memory but creates an information bottleneck.
- Hybrids with full-attention layers retain length-dependent KV caches; fixed local windows give a different memory bound.
- Core-operator FLOPs are not end-to-end latency.
- Product-family names do not guarantee architectural continuity.
- A model should be read through layer types, cache shapes, state geometry, and source quality.

## 20. primary sources

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

**Next:** Module 9 turns the architecture into a trained model: data selection, compute allocation, and optimization. Module 10 will then connect serving performance to memory bandwidth and roofline reasoning.

*Sources rechecked on 11 October 2026. Memory figures are teaching scenarios; author-reported measurements retain their original conditions.*
