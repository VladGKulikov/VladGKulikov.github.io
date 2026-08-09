# Module 7b. Diffusion Language Models

*Modern LLMs · Module 7b lecture · 2026.8 edition*

> **What this module changes.** Earlier modules altered the internals of a language model: attention, recurrence, MoE routing, normalization, and cache geometry. Here the architecture can remain recognizably Transformer-like while the **generation protocol** changes. An autoregressive (**AR**) model commits one token at a time. A diffusion large language model (**dLLM**) starts from a corrupted draft and repeatedly revises many positions. We will derive the absorbing-mask objective, execute a transparent denoising trace, examine reveal and remasking policies, and build a deliberately generic 8B roofline scenario. Only after that will we read the actual DiffusionGemma passport, keeping a teaching bound separate from a released system measurement.
>
> **Prerequisites.** Module 5 for Transformer blocks, Module 6 for sparse MoE, and Module 7 for the difference between a growing KV cache and fixed recurrent state. The chapter is otherwise self-contained.

**Contents**

1. [The sequential unit of work](#m7b-01)
2. [How the field arrived here](#m7b-02)
3. [Three intellectual ancestors](#m7b-03)
4. [A two-axis map](#m7b-04)
5. [The absorbing-mask forward process](#m7b-05)
6. [Worked example A: a mask schedule](#m7b-06)
7. [Training objective and likelihood](#m7b-07)
8. [Any-order training and its limits](#m7b-08)
9. [Sampling and worked example B](#m7b-09)
10. [A generic roofline calculation](#m7b-10)
11. [Steps, batching, and weight precision](#m7b-11)
12. [Level-1 code](#m7b-12)
13. [What the notebook verifies](#m7b-13)
14. [Block diffusion, length, and caching](#m7b-14)
15. [Reading the DiffusionGemma passport](#m7b-15)
16. [The dLLM landscape on 6 August 2026](#m7b-16)
17. [Multimodal diffusion systems](#m7b-17)
18. [Self-check questions](#m7b-18)
19. [Exercises](#m7b-19)
20. [Takeaways and sources](#m7b-20)

---

<a id="m7b-01"></a>

## 1. The sequential unit of work

The bottleneck in batch-one autoregressive decoding is often not a shortage of arithmetic. The problem is that a very large set of weights is streamed for one incremental token. The model then waits for that token before the next decode step can begin.

Take a deliberately stripped-down scenario: an 8-billion-parameter dense model, 16 GB of bf16 weights, 3.35 TB/s of memory bandwidth, and a 990-TFLOP/s arithmetic peak. A weight-stream lower bound for one decode step is

$$
 t_{\mathrm{mem}}
 =\frac{16\cdot10^9}{3.35\cdot10^{12}}
 \approx4.776\ \mathrm{ms}.
$$

The corresponding ideal $2P$ matrix arithmetic would take only

$$
 t_{\mathrm{compute}}
 =\frac{16\cdot10^9}{990\cdot10^{12}}
 \approx16.16\ \mu\mathrm{s}.
$$

The ratio, 0.338%, is not a measured GPU-utilization number. It is the share of this simplified step lower bound occupied by ideal arithmetic. Under these assumptions the batch-one ceiling is 209.375 tokens/s.

A diffusion sampler changes the granularity. One network invocation scores an entire canvas of $N$ positions. The price is $S$ denoising calls rather than one causal call per token. The rough arithmetic becomes

$$
2PNS.
$$

For $N=1024$, one ideal pass costs 16.549 ms of compute and is no longer limited by the 4.776-ms weight stream. At $S=64$, the arithmetic ceiling is 966.8 tokens/s—4.62 times the batch-one AR ceiling, despite roughly 64 times more FLOPs.

This apparent paradox is the core engineering proposition of diffusion decoding: **do more total work, but cross fewer sequential barriers**. It is only a proposition, not a guaranteed result. Bidirectional attention over the canvas, activation traffic, MoE expert union, kernel quality, runtime overhead, and the number of useful denoising steps determine whether the trade pays off.

There is also a behavioral distinction. A masked-diffusion sampler can reopen a weak position and revise it in the next iteration. A standard ancestral AR decode does not revise its committed prefix. AR systems can of course add a verifier, a second editing pass, Fill-in-the-Middle (**FIM**), or an agent loop; diffusion places revision inside the sampling algorithm itself.

<a id="m7b-02"></a>

## 2. How the field arrived here

![VIZ m7b/01 — the diffusion-LM timeline](assets/modern-llms/en/module-07b/m7b_01_timeline.svg)

The first problem was mathematical: what does “add noise” mean when the object is a token ID? **D3PM — Discrete Denoising Diffusion Probabilistic Models** (2021) answered with transition matrices over discrete states. One option was an absorbing state: a token may become `[MASK]` and remains masked in the forward process.

**Diffusion-LM** (2022) explored a continuous alternative. It corrupted token embeddings with Gaussian noise and later rounded the resulting representation back to the vocabulary. The route offered controllability, but the continuous-to-discrete boundary added its own difficulty.

By 2024, discrete objectives became much more competitive. **SEDD — Score Entropy Discrete Diffusion** developed a suitable discrete score objective. **MDLM — Masked Diffusion Language Models** showed that an absorbing-mask construction admits a simple likelihood-oriented loss and efficient samplers, including semi-autoregressive variable-length generation.

Scale became the next question. **LLaDA 8B** was trained from scratch on 2.3T tokens. **BD3-LM** introduced block diffusion, letting completed blocks form a causal prefix while the active block is denoised bidirectionally. **LLaDA-MoE**, Dream, Seed Diffusion, Mercury, and Gemini Diffusion explored sparse models, AR-checkpoint conversion, and specialized runtimes.

The 2025–2026 wave broadened both routes. **LLaDA2.0** converted 16B and 100B AR-MoE checkpoints through a Warmup–Stable–Decay (**WSD**) schedule. **LLaDA2.1** added Token-to-Token correction so already revealed values could be replaced. **iLLaDA** returned to from-scratch training at 8B scale, using 12T pretraining tokens, fully bidirectional attention, and variable-length generation.

Product releases developed on a separate evidence track. Mercury 2, Gemini Diffusion, Seed Diffusion, and DiffusionGemma reported very high generation rates, but on different accelerators, workloads, precision formats, and latency boundaries. Their numbers are useful only when those conditions remain attached.

The outcome is not one canonical dLLM. We now have full-canvas masked models, block-autoregressive models, token-editing samplers, insertion-based flexible-length methods, and proprietary systems whose exact objectives are not public.

<a id="m7b-03"></a>

## 3. Three intellectual ancestors

Diffusion language modeling inherits three different ideas.

The first is continuous diffusion: define a known corruption path and learn its reversal. Tokens prevent a literal copy of Gaussian pixel noise, which is why discrete transition kernels matter.

The second is masked language modeling (**MLM**). Bidirectional prediction from partially observed text already exists in BERT-like encoders. Diffusion contributes a time variable, a family of corruption levels, and a generative trajectory connecting a heavily corrupted sample to a clean one.

The third is variable-order factorization. A causal model commits to

$$
p(x)=\prod_{i=1}^{N}p(x_i\mid x_{<i}).
$$

A masked objective exposes the model to many observed/hidden subsets. Some objectives can be interpreted as averaging autoregressive factorizations over token orders. That interpretation belongs to the training objective; it does not make every dLLM computationally identical to an ensemble of all possible AR models.

![VIZ m7b/02 — a paradigm map](assets/modern-llms/en/module-07b/m7b_02_axes.svg)

<a id="m7b-04"></a>

## 4. A two-axis map

It helps to ask two questions separately:

1. Which positions may condition one another inside a network call?
2. How much of the output becomes stable before the next call?

| Regime | Stable state before a call | Progress made by a call | Cache implication |
|---|---|---|---|
| AR | causal prefix | one new token | exact prefix KV cache |
| full masked diffusion | fixed condition plus mutable canvas | many canvas positions | mutable canvas is recomputed |
| block diffusion | completed blocks | one current block | completed prefix is cacheable |
| token editing | a partially stable draft | selected replacements | depends on which regions remain stable |
| flexible masked diffusion | current variable-length draft | unmasking and/or insertion | method-specific |

The map prevents two common errors. A diffusion model is not necessarily restricted to a fixed final length. Conversely, block diffusion does not become ordinary AR merely because blocks are ordered causally: the active block is still modeled with bidirectional, any-order refinement.

<a id="m7b-05"></a>

## 5. The absorbing-mask forward process

The main formalism in this module is **absorbing masked diffusion**. It is a particular discrete kernel, not a definition of all diffusion language models. D3PM also permits random substitutions and more general transition matrices.

Let $x_0=(x_0^1,\ldots,x_0^N)$ be a clean sequence. Under a linear schedule, each eligible position is retained with probability $1-t$ and replaced by a special mask state $m$ with probability $t$:

$$
q(x_t^i\mid x_0^i)
=
(1-t)\delta_{x_0^i}(x_t^i)
+t\delta_m(x_t^i),
\qquad t\in[0,1].
$$

“Absorbing” describes the forward chain: after entering the mask state, the position does not spontaneously return to content. The learned reverse model predicts clean tokens from the partially observed sequence.

For conditional generation, a prompt can be marked ineligible for corruption. Only the answer region follows the mask process. This separation becomes important in systems such as DiffusionGemma, where the prompt is encoded and cached while the generation canvas is repeatedly recomputed.

![VIZ m7b/03 — the forward process](assets/modern-llms/en/module-07b/m7b_03_forward.svg)

<a id="m7b-06"></a>

## 6. Worked example A: a mask schedule

With eight positions and four model calls, use the target counts

$$
[8,6,4,2,0].
$$

One explicit schedule is

$$
m_s=\left\lceil N\frac{S-s}{S}\right\rceil,
\qquad s=0,\ldots,S.
$$

Exactly two positions become stable after each call. When $S>N$, duplicate counts can appear; a model call may then perform work without revealing a new token.

For a continuous linear masking process, the slice weight is $1/t$. At $t=0.25,0.5,0.75$, the values are $4$, $2$, and $4/3$. Early-time slices contain fewer corrupted positions, so each receives greater weight. This may increase estimator variance and motivates alternative schedules or time-sampling strategies in practical training.

<a id="m7b-07"></a>

## 7. Training objective and likelihood

For the linear absorbing process, a continuous-time ELBO-derived slice reduces to weighted cross-entropy over corrupted positions:

$$
L_t
=
\frac1t
\sum_{i\in M_t}
-\log p_\theta(x_0^i\mid x_t,t).
$$

The module uses **sum reduction**. A trainer that averages over $|M_t|$ will report a different number even with identical logits.

Suppose four masked target tokens receive probabilities

$$
[0.9,0.7,0.6,0.8]
$$

at $t=0.5$. Then

$$
L_{0.5}
=
2[-\log0.9-\log0.7-\log0.6-\log0.8]
=2.392009269.
$$

![VIZ m7b/04 — the masked-diffusion objective](assets/modern-llms/en/module-07b/m7b_04_loss.svg)

The value $t=0$ has zero measure under a continuous sampler but remains a software edge case because $1/t$ is undefined.

Likelihood language also needs care. AR provides a direct left-to-right **NLL — Negative Log-Likelihood**. Many diffusion models optimize an **ELBO — Evidence Lower Bound** or report **NELBO — Negative Evidence Lower Bound**. Yet masked diffusion can be designed as a likelihood-oriented language model, as MDLM demonstrates. A fair comparison records the estimator, time discretization, and number of Monte Carlo samples instead of treating every reported “perplexity” as the same object.

<a id="m7b-08"></a>

## 8. Any-order training and its limits

Random corruption turns one document into many conditional prediction tasks. A person’s name may be observed while the relation is hidden in one sample, and the reverse may occur in another. This offers a more symmetric training signal than a single left-to-right order.

For a uniform-order objective, masked diffusion can be related to autoregressive factorizations averaged over permutations. The Any-Order GPT study also finds that many orders are less informative than natural left-to-right structure. The result is a design trade-off, not a contradiction: order diversity is useful, but language still has asymmetric sequential regularities.

The same caution applies to the reversal curse. LLaDA reported a striking reversal-poem result. Later work, DiffER, found that dLLMs can still fail reversed relations because of fragmented entities, asymmetric training data, or entirely missing reverse facts. Bidirectional conditioning removes one architectural asymmetry; it does not invent absent evidence.

A related empirical result appears in data-constrained training. When a finite corpus is reused for many epochs, masked diffusion exposes the model to varied conditional tasks. *Diffusion Beats Autoregressive in Data-Constrained Settings* reports an advantage under the studied data/compute regimes. The claim should remain local to those regimes rather than becoming a universal statement about data efficiency.

<a id="m7b-09"></a>

## 9. Sampling and worked example B

The reverse model produces distributions for the entire canvas, but a sampler must decide which positions become stable.

A generic iteration performs five operations:

1. evaluate the model once over the current canvas;
2. select token candidates;
3. assign a confidence or entropy score to positions;
4. reveal enough positions to meet the next mask count;
5. optionally remask weak committed positions.

Possible ordering rules include random reveal, maximum candidate probability, minimum entropy, a learned policy, block-specific schedules, and task-dependent heuristics.

The frozen toy example reveals two positions at each step. High-confidence anchors are opened early, and the final sequence matches six of eight target tokens. The example is intentionally synthetic: it demonstrates how reveal order changes an outcome without pretending to benchmark real checkpoints.

![VIZ m7b/05 — a denoising trace](assets/modern-llms/en/module-07b/m7b_05_demask_trace.svg)

Remasking can revisit a poor early choice after more context becomes available. That revision is native to the sampler, but it consumes further calls. The useful metric is therefore not “tokens predicted per pass” alone; it is final committed output per pass at a target quality.

DiffusionGemma illustrates a more sophisticated policy. Its recommended **EB — Entropy Bound** sampler selects low-entropy positions subject to a mutual-information bound, renoises unselected positions, and stops only when entropy and prediction stability criteria are satisfied. The sampler is part of the released system contract, not an interchangeable afterthought.

<a id="m7b-10"></a>

## 10. A generic roofline calculation

Return to the hypothetical dense 8B model. Nothing in this section describes DiffusionGemma’s MoE traffic.

For a 1024-position canvas, one ideal pass performs

$$
2PN
=2\cdot8\cdot10^9\cdot1024
=16.384\ \mathrm{TFLOP}.
$$

At 990 TFLOP/s, the compute floor is

$$
16.549\ \mathrm{ms},
$$

which is larger than the 4.776-ms weight-stream floor. For $S$ calls,

$$
Q_{\mathrm{dLLM}}
\le
\frac{1024}{S\cdot16.549\ \mathrm{ms}}.
$$

| Calls $S$ | Arithmetic ceiling |
|---:|---:|
| 16 | 3867.2 tok/s |
| 64 | 966.8 tok/s |
| 128 | 483.4 tok/s |

The continuous parity point with the batch-one AR ceiling is approximately $S=295.5$. Integer call counts give 295 as the largest value still above that ceiling.

![VIZ m7b/06 — the teaching roofline](assets/modern-llms/en/module-07b/m7b_06_roofline.png)

The calculation is useful precisely because it is narrow. It excludes attention’s sequence-length cost, KV traffic, activations, communication, routing, batching, kernel overhead, and actual utilization.

<a id="m7b-11"></a>

## 11. Steps, batching, and weight precision

Autoregressive batching also amortizes weight reads. Under a crude linear batch-throughput approximation, the $S=64$ diffusion ceiling equals AR at

$$
B^*\approx\frac{966.8}{209.4}=4.62.
$$

This is not a claim that the fifth concurrent request always erases the benefit. Actual batching changes KV traffic, tensor shapes, occupancy, and latency objectives. The calculation identifies the direction: low-batch interactive use is the most favorable regime for parallel denoising.

The same dense model gives a second teaching comparison at a 256-token block. The compute floor is 4.137 ms. With 16 GB of bf16 weights, the 4.776-ms memory floor still dominates. If every parameter were ideally stored in four bits, the payload would be 4 GB and the memory floor 1.194 ms; the block would become compute-bound. At 16 calls, the arithmetic ceiling is again 3867.2 tok/s.

![VIZ m7b/07 — calls versus throughput](assets/modern-llms/en/module-07b/m7b_07_tps.png)

A sparse MoE invalidates a naive substitution of “active parameters” for bytes per call. Different canvas positions can select different experts, so the union of touched expert weights may be much larger than a single token’s active count. This is the key reason to keep the following released-model passport separate.

<a id="m7b-12"></a>

## 12. Level-1 code

The minimal implementation must make two contracts visible: the loss only includes corrupted positions, and one sampler iteration makes exactly one network call before revealing the scheduled number of tokens.

```python
import math
import numpy as np

def linear_mask_schedule(length: int, steps: int) -> tuple[int, ...]:
    """Target counts from fully masked to fully revealed."""
    if length <= 0 or steps <= 0:
        raise ValueError("length and steps must be positive")
    return tuple(
        math.ceil(length * remaining / steps)
        for remaining in range(steps, -1, -1)
    )

def dllm_loss(logits, target_ids, loss_mask, time: float) -> float:
    """(1/t) times the summed NLL on explicitly masked positions."""
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    loss_mask = np.asarray(loss_mask)
    if logits.ndim != 2 or target_ids.shape != (logits.shape[0],):
        raise ValueError("expected logits [T,V] and target_ids [T]")
    if loss_mask.shape != target_ids.shape or loss_mask.dtype.kind != "b":
        raise ValueError("loss_mask must be boolean [T]")
    if not (0.0 < time <= 1.0) or not loss_mask.any():
        raise ValueError("time must be in (0,1] and loss_mask must be non-empty")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(target_ids.size), target_ids]
    return float(nll[loss_mask].sum() / time)

def sample_dllm(predict, length: int, steps: int, mask_id: int):
    """Greedy confidence reveal with one model invocation per iteration."""
    schedule = linear_mask_schedule(length, steps)
    tokens = np.full(length, mask_id, dtype=np.int64)
    trace = []
    for step, target_masked in enumerate(schedule[1:], start=1):
        logits = np.asarray(predict(tokens.copy(), step), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != length or not np.isfinite(logits).all():
            raise ValueError("predict must return finite logits [T,V]")
        if 0 <= mask_id < logits.shape[1]:
            raise ValueError("mask_id must lie outside the vocabulary")
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=-1, keepdims=True)
        candidates = probs.argmax(axis=-1)
        confidence = probs[np.arange(length), candidates]
        masked = np.flatnonzero(tokens == mask_id)
        open_count = masked.size - target_masked
        order = np.lexsort((masked, -confidence[masked]))
        opened = masked[order[:open_count]]
        tokens[opened] = candidates[opened]
        trace.append((tuple(opened), tokens.copy()))
    return tokens, trace
```

A production sampler adds categorical draws, entropy policies, adaptive stopping, remasking, protected prompt regions, and fused kernels. The visible invariants remain useful in every implementation.

<a id="m7b-13"></a>

## 14. Block diffusion, length, and caching

A mutable bidirectional canvas cannot use an ordinary causal KV cache internally: changing one position can change representations at every other position on the next call.

Block diffusion introduces a stable boundary. Completed blocks form a causal prefix. The active block attends to that prefix and bidirectionally to itself. Prefix K/V can therefore be reused across denoising calls, while active-block states are recomputed.

![VIZ m7b/08 — block-diffusion spectrum](assets/modern-llms/en/module-07b/m7b_08_block_spectrum.svg)

As $L_b$ changes:

- $L_b=1$ approaches AR;
- larger blocks expose more parallel work;
- completed blocks become cacheable;
- the active block remains bidirectional;
- output length can grow block by block.

BD3-LM formalizes this class. MDLM also presents semi-autoregressive variable-length sampling, FlexMDM inserts mask states, and iLLaDA reports variable-length generation. Block diffusion is therefore one strong design, not a monopoly on flexible length.

LLaDA2.0 uses block size as a conversion curriculum. During warmup:

$$
1\rightarrow4\rightarrow32\rightarrow64\rightarrow4096.
$$

The stable phase trains full-sequence masked diffusion; decay returns to smaller blocks for efficient inference. LLaDA2.1 adds Token-to-Token editing so a revealed value may later be replaced. It should not be described as a published `INSERT/DELETE` length-editing model.

<a id="m7b-15"></a>

## 15. Reading the DiffusionGemma passport

DiffusionGemma is a useful case because its model card exposes architecture, sampler, and hardware conditions rather than only a headline TPS number.

![VIZ m7b/09 — DiffusionGemma passport](assets/modern-llms/en/module-07b/m7b_09_gemma_passport.svg)

| Field | Released contract |
|---|---|
| architecture | encoder–decoder, block-autoregressive multi-canvas diffusion |
| total / active parameters | 25.2B / 3.8B |
| layers | 30 |
| experts | 128 routed, top-8, plus one shared |
| canvas | 256 tokens |
| context | up to 256K |
| sliding window | 1024 |
| input | text, images, and video represented as frames |
| output | text |
| sampler | Entropy-Bounded Denoising, adaptive stopping, at most 48 calls |
| committed progress | typically 15–20 final tokens per forward pass |
| reported speed | over 1100 tok/s, H100, FP8, low batch |
| native low-bit path | NVFP4 on NVIDIA Blackwell |

The prompt is processed by an autoregressive encoder and stored as cacheable context. A bidirectional decoder denoises the active canvas through cross-attention to the prefix. After completion, the canvas is encoded and appended to the prefix cache.

Two idealized four-bit payloads are easy to compute:

$$
25.2\mathrm{B}\times\frac4{8}=12.6\ \mathrm{GB}=11.735\ \mathrm{GiB},
$$

$$
3.8\mathrm{B}\times\frac4{8}=1.9\ \mathrm{GB}=1.770\ \mathrm{GiB}.
$$

Neither is measured bytes per diffusion call. The full bank includes heterogeneous tensors; a 256-position canvas can route tokens to different experts; actual traffic depends on the expert union, precision, metadata, caches, and kernels.

The quality side matters as much as speed. The model card reports lower scores than AR Gemma 4 on many benchmarks, with some exceptions. DiffusionGemma is therefore a point on a quality–latency curve, not proof that a generation paradigm dominates every task.

<a id="m7b-16"></a>

## 16. The dLLM landscape on 6 August 2026

The following rows deliberately include the measurement context. Tokens per second without hardware, sampler, and overhead is not a comparable metric.

| System | Access and branch | Published signal | Evidence boundary |
|---|---|---|---|
| LLaDA 8B | open weights, from scratch | 2.3T pretraining tokens | scale result; reversal result belongs to one task |
| LLaDA-MoE 7B-A1B | open weights | sparse diffusion MoE, 1.4B active | active count does not determine runtime speed |
| LLaDA2.0 16B / 100B | open weights, AR conversion | WSD and block diffusion | throughput depends on CAP/dInfer, threshold, and hardware |
| LLaDA2.1 | open weights | Token-to-Token editing | value revision, not insertion-based length editing |
| iLLaDA 8B | open weights, from scratch | 12T pretraining tokens, variable length | foundation-model result rather than a universal TPS claim |
| Mercury 2 | commercial API | 1009 tok/s on Blackwell, 128K | developer-reported product configuration |
| Gemini Diffusion | experimental demo | 1479 tok/s sampling, 0.84 s overhead excluded | sampling rate is not end-to-end latency |
| Seed Diffusion Preview | experimental, code-focused | 2146 tok/s on H20 | developer result on coding workloads |
| DiffusionGemma | open weights | >1100 tok/s, H100 FP8; 15–20 committed tokens/pass | most detailed open system passport in this list |

Mercury 2 adds reasoning controls, tool use, and schema-constrained JSON to a commercial dLLM API. Gemini Diffusion makes the measurement boundary unusually explicit: 1479 tok/s excludes 0.84 seconds of fixed overhead. Seed Diffusion targets code and reports H20 throughput. None of those numbers can be ranked fairly without rerunning a shared workload.

iLLaDA highlights a different frontier: improving a from-scratch bidirectional model rather than solely accelerating a converted checkpoint. Its 12T-token pretraining run and variable-length generation show that the research space includes foundation-model quality, not only sampler speed.

<a id="m7b-17"></a>

## 17. Multimodal diffusion systems

“Multimodal diffusion” can refer to different components.

**LLaDA-V** applies diffusion language modeling to vision-language understanding: a visual representation conditions a diffusion-generated text response.

**LLaDA2.0-Uni** discretizes visual inputs through SigLIP-VQ, uses a MoE dLLM backbone over semantic tokens, and employs a separate diffusion decoder to reconstruct images. Text-token diffusion and pixel/latent generation are distinct parts of the stack.

**DiffusionGemma** accepts interleaved text, images, and video frames but produces text. Its vision encoder does not imply that the response is generated by the same image diffusion decoder used in a text-to-image system.

The useful passport question is:

> Which object is diffused—language tokens, visual semantic tokens, a continuous image latent, or several of these at different stages?

Without that distinction, two systems bearing the same label may have entirely different compute and memory profiles.

<a id="m7b-18"></a>

## 20. Takeaways and sources

![VIZ m7b/10 — diffusion language models at a glance](assets/modern-llms/en/module-07b/m7b_10_cheatsheet.svg)

**Generation protocol.** AR commits one token. Masked diffusion revises a canvas. Block diffusion makes completed blocks causal and cacheable.

**Forward process.** The chapter uses an absorbing mask; discrete diffusion supports other kernels.

**Objective.** The 2.392009269 example uses time weighting and sum reduction over corrupted positions. Likelihood claims require an explicit estimator.

**Sampling.** Reveal order, remasking, entropy bounds, and adaptive stopping are part of the algorithm, not cosmetic decoding flags.

**Roofline.** The 8B/H100 figures are a generic teaching floor. The 0.338% quantity is not measured utilization.

**DiffusionGemma.** 25.2B total / 3.8B active, 128 routed experts with top-8 plus one shared, 256-token canvas, 15–20 committed tokens per pass, over 1100 tok/s on H100 FP8, and a native NVFP4 Blackwell path. Four-bit parameter arithmetic does not equal measured per-pass traffic.

**Evidence.** Mercury, Gemini, Seed, and DiffusionGemma report throughput under different protocols. Always keep hardware, precision, sampler, batch, overhead, and quality attached.

**Primary sources**

- [D3PM](https://arxiv.org/abs/2107.03006)
- [Diffusion-LM](https://arxiv.org/abs/2205.14217)
- [SEDD](https://arxiv.org/abs/2310.16834)
- [MDLM](https://arxiv.org/abs/2406.07524)
- [LLaDA](https://arxiv.org/abs/2502.09992)
- [BD3-LM](https://arxiv.org/abs/2503.09573)
- [Any-Order GPT as Masked Diffusion](https://arxiv.org/abs/2506.19935)
- [Data-Constrained Diffusion](https://arxiv.org/abs/2507.15857)
- [LLaDA2.0](https://arxiv.org/abs/2512.15745)
- [DiffER](https://arxiv.org/abs/2601.07347)
- [LLaDA2.1](https://arxiv.org/abs/2602.08676)
- [LLaDA2.0-Uni](https://arxiv.org/abs/2604.20796)
- [iLLaDA](https://arxiv.org/abs/2606.25331)
- [DiffusionGemma model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [DiffusionGemma announcement](https://deepmind.google/models/gemma/diffusiongemma/)
- [Mercury 2](https://www.inceptionlabs.ai/blog/introducing-mercury-2)
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Seed Diffusion](https://seed.bytedance.com/en/public_papers/seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed-inference)

**Next:** Module 8 returns to pretraining. Its treatment of data reuse and compute budgets will provide a broader context for the data-constrained result discussed here.

---

*Landscape verified: 6 August 2026. Product throughput figures are developer-reported results under different hardware and sampling protocols. Sections 10–11 describe a separate generic dense-8B teaching scenario.*
