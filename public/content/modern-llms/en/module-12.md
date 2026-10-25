# Module 12. Diffusion Language Models

*Modern LLMs · Module 12 lecture · 11 October 2026 edition*

> **What this module changes.** In Module 11, speculative decoding accelerated sampling while preserving the distribution of a given autoregressive model. We now change the modeling problem itself. The network may still be recognizably Transformer-like, but its training objective and **generation protocol** must support reconstruction from a partially hidden sequence. This is not simply another execution mode for an unchanged AR checkpoint.
>
> An autoregressive (**AR**) sampler extends a committed prefix. Our main example of a diffusion large language model (**dLLM**) instead begins with masks and predicts several missing positions at once. Its working sequence is a **canvas**; recovering the hidden tokens is **denoising**. We will connect the corruption process to the learning objective, examine how a sampler turns predictions into text, and ask why fewer sequential calls can require more total computation. A generic 8B roofline scenario will then provide a comparison point for the actual DiffusionGemma passport—not a substitute for its measurements.
>
> **Prerequisites.** We build on autoregressive generation from Module 3, pretraining from Module 9, the serving cost model from Module 10, and speculative decoding from Module 11. Together they make a fair comparison of generation protocols possible. The sparse-model discussion also uses MoE from Module 7.

**First pass:** read Sections 1 and 4–9 for the mechanism, Section 10 for one cost calculation, and Section 14 for block generation. In Section 7, focus on what the loss measures and the worked example; return to the expectation formula later. Use the [self-check questions (Section 18)](/courses/modern-llms/module-12/tests/) to test your understanding.

**Second pass:** work through the objective in Section 7, batching in Section 11, and the implementation in Section 12 before doing the [notebook (Section 13)](/courses/modern-llms/module-12/notebooks/) and [exercises (Section 19)](/courses/modern-llms/module-12/exercises/). **Reference pass:** Sections 2–3 explain the origins of the ideas; Sections 15–17 are a dated guide to released systems.

**Contents**

1. [The sequential unit of work](#1-the-sequential-unit-of-work)
2. [How the field arrived here](#2-how-the-field-arrived-here)
3. [Three intellectual ancestors](#3-three-intellectual-ancestors)
4. [A two-axis map](#4-a-two-axis-map)
5. [The absorbing-mask forward process](#5-the-absorbing-mask-forward-process)
6. [Worked example A: a mask schedule](#6-worked-example-a-a-mask-schedule)
7. [Training objective and likelihood](#7-training-objective-and-likelihood)
8. [Any-order training and its limits](#8-any-order-training-and-its-limits)
9. [Sampling and worked example B](#9-sampling-and-worked-example-b)
10. [A generic roofline calculation](#10-a-generic-roofline-calculation)
11. [Steps, batching, and weight precision](#11-steps-batching-and-weight-precision)
12. [Level-1 code](#12-level-1-code)
13. [What the notebook verifies](/courses/modern-llms/module-12/notebooks/)
14. [Block diffusion, length, and caching](#14-block-diffusion-length-and-caching)
15. [Reading the DiffusionGemma passport](#15-reading-the-diffusiongemma-passport)
16. [The dLLM landscape on 11 October 2026](#16-the-dllm-landscape-on-11-october-2026)
17. [Multimodal diffusion systems](#17-multimodal-diffusion-systems)
18. [Self-check questions](/courses/modern-llms/module-12/tests/)
19. [Exercises](/courses/modern-llms/module-12/exercises/)
20. [Takeaways and sources](#20-takeaways-and-sources)

---

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

The ratio $t_{\mathrm{compute}}/t_{\mathrm{mem}}$, 0.338%, is not a measured GPU-utilization number. It is the share of this simplified step lower bound occupied by ideal arithmetic. Under these assumptions the batch-one ceiling is 209.375 tokens/s.

A diffusion sampler changes the granularity. One network invocation scores an entire canvas of $N$ positions. The price is $S$ denoising calls rather than one causal call per token. The rough arithmetic becomes

$$
2PNS.
$$

For $N=1024$, one ideal pass costs 16.549 ms of compute and is no longer limited by the 4.776-ms weight stream. At $S=64$, the arithmetic ceiling is 966.8 tokens/s—4.62 times the batch-one AR ceiling, despite roughly 64 times more FLOPs.

This apparent paradox is the core engineering proposition of diffusion decoding: **do more total work, but cross fewer sequential barriers**. It is only a proposition, not a guaranteed result. Bidirectional attention over the canvas, activation traffic, MoE expert union, kernel quality, runtime overhead, and the number of useful denoising steps determine whether the trade pays off.

There is also a behavioral reason to use a canvas: missing positions can be predicted from available context on both sides. Some samplers go further and reopen a weak choice after its neighbors have become more informative. Revision is an option supplied by the sampling policy, not a necessary action in every diffusion pass; a monotone unmasking sampler keeps the tokens it has already revealed. Standard ancestral AR decoding likewise leaves its committed prefix unchanged. AR systems can add a verifier, a second editing pass, Fill-in-the-Middle (**FIM**), or an agent loop; a diffusion system can incorporate revision directly into its reconstruction procedure.

## 2. How the field arrived here

![VIZ m12/01 — the diffusion-LM timeline](assets/modern-llms/en/module-12/m12_01_timeline.svg)
*The line from discrete transition kernels to masked likelihood models, block diffusion, open-weight systems, and commercial APIs. Each milestone addresses a different obstacle rather than replacing all earlier methods.*

The first problem was mathematical: what does “add noise” mean when the object is a token ID? **D3PM — Discrete Denoising Diffusion Probabilistic Models** (2021) answered with transition matrices over discrete states. One option was an absorbing state: a token may become `[MASK]` and remains masked in the forward process.

**Diffusion-LM** (2022) explored a continuous alternative. It corrupted token embeddings with Gaussian noise and later rounded the resulting representation back to the vocabulary. The route offered controllability, but the continuous-to-discrete boundary added its own difficulty.

By 2024, discrete objectives became much more competitive. **SEDD — Score Entropy Discrete Diffusion** developed a suitable discrete score objective. **MDLM — Masked Diffusion Language Models** showed that an absorbing-mask construction admits a simple likelihood-oriented loss and efficient samplers, including semi-autoregressive variable-length generation.

Scale became the next question. **LLaDA 8B** was trained from scratch on 2.3T tokens. **BD3-LM** introduced block diffusion, letting completed blocks form a causal prefix while the active block is denoised bidirectionally. **LLaDA-MoE**, Dream, Seed Diffusion, Mercury, and Gemini Diffusion explored sparse models, AR-checkpoint conversion, and specialized runtimes.

The 2025–2026 wave broadened both routes. **LLaDA2.0** converted 16B and 100B AR-MoE checkpoints through a Warmup–Stable–Decay (**WSD**) schedule. **LLaDA2.1** added Token-to-Token correction so already revealed values could be replaced. **iLLaDA** returned to from-scratch training at 8B scale, using 12T pretraining tokens, fully bidirectional attention, and variable-length generation.

Product releases developed on a separate evidence track. Mercury 2, Gemini Diffusion, Seed Diffusion, and DiffusionGemma reported very high generation rates, but on different accelerators, workloads, precision formats, and latency boundaries. Their numbers are useful only when those conditions remain attached.

The outcome is not one canonical dLLM. We now have full-canvas masked models, block-autoregressive models, token-editing samplers, insertion-based flexible-length methods, and proprietary systems whose exact objectives are not public.

## 3. Three intellectual ancestors

Diffusion language modeling inherits three different ideas.

The first is continuous diffusion: define a known corruption path and learn its reversal. Tokens prevent a literal copy of Gaussian pixel noise, which is why discrete transition kernels matter.

The second is masked language modeling (**MLM**). Bidirectional prediction from partially observed text already exists in BERT-like encoders. Diffusion contributes a time variable, a family of corruption levels, and a generative trajectory connecting a heavily corrupted sample to a clean one.

The third is variable-order factorization. A causal model commits to

$$
p(x)=\prod_{i=1}^{N}p(x_i\mid x_{<i}).
$$

A masked objective exposes the model to many observed/hidden subsets. Some objectives can be interpreted as averaging autoregressive factorizations over token orders. That interpretation belongs to the training objective; it does not make every dLLM computationally identical to an ensemble of all possible AR models.

![VIZ m12/02 — a paradigm map](assets/modern-llms/en/module-12/m12_02_axes.svg)
*Two independent design axes: causal versus bidirectional conditioning, and token versus block versus canvas as the unit of sequential progress.*

## 4. A two-axis map

It helps to ask two questions separately:

1. Which positions may condition one another inside a network call?
2. How much of the output becomes stable before the next call?

| Regime | Stable state before a call | Positions updated by a call | Cache implication |
|---|---|---|---|
| AR | causal prefix | one new token | exact prefix KV cache |
| full masked diffusion | fixed condition plus mutable canvas | many canvas positions | mutable canvas is recomputed |
| block diffusion | completed blocks | one current block | completed prefix is cacheable |
| token editing | a partially stable draft | selected replacements | depends on which regions remain stable |
| flexible masked diffusion | current variable-length draft | unmasking and/or insertion | method-specific |

A call can refine positions in the current block without completing that block. Several calls may be needed before its tokens become stable. The map prevents two common errors. A diffusion model is not necessarily restricted to a fixed final length. Conversely, block diffusion does not become ordinary AR merely because blocks are ordered causally: the active block is still modeled with bidirectional, any-order refinement.

## 5. The absorbing-mask forward process

The main formalism in this module is **absorbing masked diffusion**. It is a particular discrete kernel, not a definition of all diffusion language models. D3PM also permits random substitutions and more general transition matrices.

The construction starts with a text we already know, not an answer we are trying to generate. We hide some of its tokens while retaining their original values as training targets. Corruption is a rule we specify; the network must learn to recover the content that this rule removes.

Let $x_0=(x_0^1,\ldots,x_0^N)$ be a clean sequence. Under a linear schedule, each eligible position is retained with probability $1-t$ and replaced by a special mask state $m$ with probability $t$:

$$
q(x_t^i\mid x_0^i)
=
(1-t)\delta_{x_0^i}(x_t^i)
+t\delta_m(x_t^i),
\qquad t\in[0,1].
$$

The symbol $\delta_a(z)$ is 1 for $z=a$ and 0 otherwise. This equation specifies a marginal at a single time: it describes what a corrupted position looks like at the selected noise level. Connecting several such observations into one history is a separate matter.

A monotone sample path can be constructed by drawing one threshold $u_i\sim U(0,1)$ per position and masking that position whenever $t\ge u_i$. Resampling fresh Bernoulli variables at every time gives the right marginals but does not trace one absorbing path. For training, that distinction is useful rather than troublesome: a corrupted example at a chosen time can be drawn directly, without simulating every earlier state.

“Absorbing” describes the forward chain: after entering the mask state, the position does not spontaneously return to content. The learned reverse model predicts clean tokens from the partially observed sequence.

For conditional generation, a prompt can be marked ineligible for corruption. Only the answer region follows the mask process. This separation becomes important in systems such as DiffusionGemma, where the prompt is encoded and cached while the generation canvas is repeatedly recomputed.

![VIZ m12/03 — the forward process](assets/modern-llms/en/module-12/m12_03_forward.svg)
*Absorbing masking is the chapter’s reference kernel. Other discrete diffusion models may use substitution noise or a general transition matrix.*

## 6. Worked example A: a mask schedule

For a moment, look ahead from corruption to generation. The forward process hides known content; a sampler must fill positions whose values are not yet known. We can specify how quickly it should reveal them before deciding which tokens to choose. With eight positions and four model calls, use the target counts

$$
[8,6,4,2,0].
$$

One explicit schedule is

$$
m_s=\left\lceil N\frac{S-s}{S}\right\rceil,
\qquad s=0,\ldots,S.
$$

Exactly two positions become stable after each call. When $S>N$, duplicate counts can appear; a model call may then perform work without revealing a new token.

Here $s$ is a discrete iteration index, whereas $t$ in the forward process is a continuous corruption time. The deterministic count schedule is not a literal reversal of every random masking path. It tells the sampler how much to reveal, not what belongs in the revealed positions. That second question takes us back to training.

## 7. Training objective and likelihood

Consider what happens to one training example. The clean text is available. We choose a corruption level, draw a set of masked positions, and give the damaged sequence to the network. Its predictions at those positions are compared with the original tokens. Unlike generation, this procedure does not have to discover the correct answer through a complete sequence of guesses: the targets were saved before corruption.

One sampled time slice is enough to supply such a learning signal. Running a full denoising trajectory for every example is not required by the objective below. The remaining question is how to weight prediction errors across corruption levels so that these local reconstruction tasks correspond to a probabilistic model of the whole text.

### Why likelihood leads to a variational bound

For an autoregressive model, scoring a known sequence follows a single familiar route: read a prefix, score the next observed token, and continue. A diffusion model can produce the same final text through different intermediate states and different reveal times. The probability of the text includes **all** those routes, not merely one plausible reconstruction.

For this part of the argument, choose a finite time grid and let $z$ denote the entire intermediate trajectory. The reverse model assigns a joint probability $p_\theta(x_0,z)$ to taking that route and ending at the observed text. Obtaining $p_\theta(x_0)$ requires summing over the hidden trajectories. Even when each transition is simple, that sum can be prohibitively large.

We do have an inexpensive way to obtain trajectories: apply the prescribed masking process to the known text. This gives a distribution $q(z\mid x_0)$. We can estimate averages under it instead of enumerating every path. But the average log probability of sampled paths is not automatically the log probability of the text: taking a logarithm after summation is different from averaging logarithms.

The **evidence lower bound (ELBO)** makes the connection precise:

$$
\operatorname{ELBO}(x_0)
=\mathbb E_{q(z\mid x_0)}
\left[\log\frac{p_\theta(x_0,z)}{q(z\mid x_0)}\right]
\le \log p_\theta(x_0).
$$

The denominator accounts for the distribution under which we chose the paths. The inequality follows from the concavity of the logarithm. It gives us an estimable objective tied to the probability of the whole text without requiring a direct sum over every way of generating it.

The gap also has an exact interpretation, under the usual support conditions:

$$
\log p_\theta(x_0)-\operatorname{ELBO}(x_0)
=D_{\mathrm{KL}}\!\left(q(z\mid x_0)\,\|\,p_\theta(z\mid x_0)\right)
\ge 0.
$$

The two distributions on the right concern **paths conditioned on an already observed final text**. One comes from our corruption procedure; the other is induced by the learned generative model. They need not coincide. That is why the objective is a bound rather than automatically the exact log-likelihood. Negating it gives **NELBO**, an upper bound on negative log-likelihood that we minimize.

The expression still seems to require reasoning about whole trajectories. The Markov structure makes it local: a path probability factors over transitions, and the logarithm turns the product into a sum. Rearranging NELBO produces comparisons between the known reverse transitions of the corruption process and the model's transitions. During training, the known transitions can use the original text; at generation time, that unavailable information must be supplied by the network. We can now inspect one such comparison for an absorbing mask.

### Where the reveal probability comes from

Return to the threshold construction in Section 5. Each position has one random time at which it becomes masked. Suppose position $i$ is masked at time $t$. Its threshold must lie between zero and the current time; conditional on that event, it is uniform within that interval.

Take a backward step of length $\Delta t$, with $0<\Delta t<t$. The position was visible at the earlier time only if its masking event occurred **during the final interval**. That interval has the length of our step, while the full range of possible thresholds has the length of the current time. Hence

$$
\Pr\!\left(t-\Delta t<u_i\le t\mid u_i\le t\right)
=\frac{\Delta t}{t}.
$$

This is neither network confidence nor the probability of guessing the correct token. It is the probability that reversing the **specified corruption process** requires revealing the position on this step. The schedule supplies it. What remains unknown is the content that should replace the mask.

Training supplies the missing answer: the original token. The known reverse transition, conditioned on the clean text, therefore has only two possible outcomes—keep the mask or restore that token. The learned transition uses the same reveal probability, but distributes the revealed mass over possible tokens according to the network's prediction from the current canvas. If a position is already visible, both transitions simply copy it.

This separates two jobs: **the schedule controls whether a reveal occurs; the network predicts the content if it does**. The network can therefore learn a clean-token distribution rather than an unconstrained transition matrix between every pair of vocabulary states.

### Why the transition loss becomes cross-entropy

Fix the current canvas and one masked position. Call its known reverse distribution $Q_i$ and its learned reverse distribution $P_i$. Both keep the mask with probability $1-\Delta t/t$ and allocate mass $\Delta t/t$ to a reveal. The known transition puts all revealed mass on the correct token; the model spreads it over the vocabulary.

KL divergence compares the probabilities of **the same outcomes**. The mask-retention probabilities agree, so that outcome contributes zero. For the correct token, the model's transition probability contains an extra factor: the network's probability for that token. Other content tokens have zero mass under the known transition and contribute no additional terms to the KL sum. Thus

$$
\begin{aligned}
D_{\mathrm{KL}}(Q_i\|P_i)
&=\frac{\Delta t}{t}
\log\frac{\Delta t/t}
{(\Delta t/t)\,p_\theta(x_0^i\mid x_t,t)}\\
&=\frac{\Delta t}{t}
\bigl[-\log p_\theta(x_0^i\mid x_t,t)\bigr].
\end{aligned}
$$

The common reveal probability cancels inside the logarithm but remains outside it. Both the **ordinary token-prediction loss** and its **time-dependent weight** have emerged from the same comparison. They are not two unrelated choices added to the training recipe.

An already visible position has identical deterministic transitions and contributes zero. Copying is imposed by the transition construction; it does not mean that an untrained network already predicts the visible token with probability one. Under the chosen position-factorized transition, the local KL terms add. This does not assume that language tokens are independent: every network prediction uses the available canvas. The restriction concerns one transition, and Section 9 will examine what happens when several positions are revealed together.

Now sum over time intervals. Each term contains the interval width. As the grid becomes finer, the sum approaches an integral, leaving inverse time as the coefficient per unit time. A different corruption schedule changes this coefficient; its simple form here comes from the linear masking probability.

The full variational bound also has endpoint terms. In this idealized construction, the forward process ends with every position masked, and generation starts from that same all-mask sequence. At the clean endpoint, observed values are copied. Under these assumptions, the starting distribution and final copying add no separate learned penalty. Different endpoint conditions, or a clean-token predictor allowed to emit the mask, require revisiting the simplification. These are the parameterization conditions used in the [MDLM derivation, Sections 3.2–3.5 and Appendix B](https://arxiv.org/html/2406.07524v2#S3).

### Estimating the objective from one training slice

For a linear absorbing-mask process, a clean-token parameterization that copies observed tokens and excludes mask from clean predictions now has a concrete interpretation through those local transitions. One sampled slice contributes:

$$
L_t
=
\frac1t
\sum_{i\in M_t}
-\log p_\theta(x_0^i\mid x_t,t).
$$

Here $M_t$ denotes the masked positions, and natural logarithms give the loss in nats. There are two sources of randomness: the selected time and the particular positions corrupted at that time. A loss observed on one damaged example is therefore a contribution to the objective, not the entire objective. Because the time interval has length one, integration over it is an expectation under uniform time: estimating the objective does not require visiting every time slice for each text. Averaging over uniform time and the corruption process gives

$$
\mathcal L(x_0)=\mathbb E_{t\sim U(0,1)}\mathbb E_{q(x_t\mid x_0)}L_t,
\qquad -\log p_\theta(x_0)\le\mathcal L(x_0).
$$

The bound applies to this complete objective, not to every sampled slice individually. A particular mask may make a reconstruction unusually easy or difficult; the likelihood statement concerns the prescribed average.

With retention probability $\alpha(t)=1-t$, the derivation gives inverse time. For a general smooth decreasing schedule, the same reasoning produces $-\alpha'(t)/(1-\alpha(t))$: the rate at which visible mass disappears, divided by the mass already hidden. Its linear-schedule value is $1/t$.

There is also a useful training interpretation. A position is less likely to be masked at a low-noise time, but receives a larger multiplier when it is masked. This compensates for the rarity of those contributions. It is consistent with the transition derivation, not a replacement for it: a wish to balance frequencies would not by itself establish a likelihood bound. Nor does the weighting make the reconstruction tasks equally difficult or require a fixed mask count in every example.

For a continuous linear masking process, the slice weight is $1/t$. At $t=0.25,0.5,0.75$, the values are $4$, $2$, and $4/3$. The average is specified by the objective, but the larger weights on rare events can make individual estimates fluctuate substantially. This may increase estimator variance and motivates alternative schedules or time-sampling strategies in practical training.

### Interpreting the number returned by the loss

Reduction is part of the objective, not merely a formatting choice for the reported loss. We sum the contributions of the positions that happened to be masked. Dividing by a fixed sequence length changes a constant scale. Dividing by the random mask count instead changes the relative weight of examples and requires a correction if the same estimator is intended.

A corruption sample with no masks contributes zero to the sum. It is a valid draw from the prescribed process, not a missing example. Discarding it, or forcing one position to be masked, changes the sampling law.

Suppose four masked target tokens receive probabilities

$$
[0.9,0.7,0.6,0.8]
$$

at $t=0.5$. Then

$$
L_{0.5}
=
2[-\log0.9-\log0.7-\log0.6-\log0.8]
\approx2.392009269.
$$

![VIZ m12/04 — the masked-diffusion objective](assets/modern-llms/en/module-12/m12_04_loss.svg)
*The loss contract: masked positions, time weighting, and an explicit reduction convention. Likelihood interpretation depends on the forward process and estimator.*

The value $t=0$ has zero measure under a continuous sampler but remains a software edge case because $1/t$ is undefined.

Likelihood language also needs care. AR provides a directly computable left-to-right **NLL — Negative Log-Likelihood** for an observed sequence. In the diffusion construction discussed here, **NELBO — Negative Evidence Lower Bound** supplies an upper bound on that quantity.

The bound concerns the reverse-process model for which it was derived. Replacing that process at generation time with greedy confidence-based reveal or another heuristic does not automatically preserve its distribution. The training objective, its numerical estimate, and a chosen fast sampler are related, but they are not interchangeable objects.

Being likelihood-oriented does not mean avoiding a variational bound. In [MDLM](https://arxiv.org/abs/2406.07524), the **ELBO — Evidence Lower Bound** is precisely what connects the reconstruction objective to likelihood. Perplexity derived from NELBO should be read as a bound-based quantity, not automatically as exact model perplexity. A fair comparison records what is estimated, how the averaging is performed, the time discretization, and the number of Monte Carlo samples.

## 8. Any-order training and its limits

Random corruption turns one document into many conditional prediction tasks. A person’s name may be observed while the relation is hidden in one sample, and the reverse may occur in another. This offers a more symmetric training signal than a single left-to-right order.

For a uniform-order objective, masked diffusion can be related to autoregressive factorizations averaged over permutations. The Any-Order GPT study also finds that many orders are less informative than natural left-to-right structure. The result is a design trade-off, not a contradiction: order diversity is useful, but language still has asymmetric sequential regularities.

The same caution applies to the **reversal curse**. Consider the fictional fact “Anna Vetrova wrote *The Quiet Harbour*.” A model may complete the author’s name with the book title yet fail to name the author when asked who wrote that book. Reversing the question is not the same as claiming that authorship is a symmetric relation. LLaDA reported a striking reversal-poem result. Later work, DiffER, found that dLLMs can still fail reversed relations because of fragmented entities, asymmetric training data, or entirely missing reverse facts. Bidirectional conditioning removes one architectural asymmetry; it does not invent absent evidence.

A related empirical result appears in data-constrained training. When a finite corpus is reused for many epochs, masked diffusion exposes the model to varied conditional tasks. *Diffusion Beats Autoregressive in Data-Constrained Settings* reports an advantage under the studied data/compute regimes. The claim should remain local to those regimes rather than becoming a universal statement about data efficiency.

Training on many observed/hidden subsets still leaves a decision to make at generation time. It does not guarantee that every reveal order produces the same result. Any-order prediction provides a range of conditional tasks; a sampler must choose a particular route through them.

## 9. Sampling and worked example B

A denoising network returns token distributions for the current canvas. It does not, by that output alone, specify which candidates to select, which positions to reveal first, or whether an earlier choice may be revised. Those decisions belong to the sampler.

Section 7 separated two decisions: the schedule supplies the reveal probability, and the network predicts the content of a revealed position. We can now use both during generation. For the linear absorbing process, one standard reverse transition from time $t$ to $\tau<t$ leaves a masked position masked with probability $\tau/t$ and reveals it with probability $(t-\tau)/t$, using the predicted clean-token distribution. Visible positions are copied. This connects the forward kernel to a reverse transition. Deployed policies may alter the schedule or position choices to trade quality for speed, but those alterations should be distinguished from the transition itself.

Our teaching trace returns to the fixed mask-count schedule from Section 6 and uses greedy token choices. It makes the operations inspectable but does not establish exact sampling from the data distribution. Confidence-based reveal has no universal distribution-correction guarantee analogous to the speculative algorithm in Module 11.

The following sequence describes a generic iteration with optional revision. The minimal code in Section 12 uses monotone reveal; a remasking variant additionally performs the first operation:

1. optionally remask open positions using scores saved from earlier predictions;
2. call the model once on that actual corrupted canvas;
3. choose candidates and rank masked positions;
4. reveal enough positions to meet the next mask count;
5. save the result and confidence scores for the next iteration.

Possible ordering rules include random reveal, maximum candidate probability, minimum entropy, a learned policy, block-specific schedules, and task-dependent heuristics.

Choosing a token and choosing a position serve different purposes. Within a position, we need a candidate value; across positions, we decide which choices should become context for the next network call. Two common scores answer this second question differently.

Candidate confidence is the single probability $p_i(v_i)$, whereas entropy $H(p_i)=-\sum_v p_i(v)\log p_i(v)$ measures uncertainty across the whole positional distribution. Low entropy means that probability is concentrated among fewer alternatives. The rankings can differ, especially when a candidate is sampled rather than chosen by argmax.

The first frozen trace uses fixed logits and opens two positions per step. Six of eight final tokens match the targets, regardless of the order in which all greedy candidates are eventually revealed. It tests scheduling, not context sensitivity. A second notebook experiment makes predictions depend on previously revealed anchors; there the order can change the final answer.

![VIZ m12/05 — a denoising trace](assets/modern-llms/en/module-12/m12_05_demask_trace.svg)
*One call per iteration, candidate tokens for every position, a confidence ranking, and an exact target mask count.*

A two-token example explains why extra passes can matter. Suppose the source emits only $AA$ or $BB$, each with probability $1/2$. With both positions masked, each marginal assigns $1/2$ to A and B. Sampling them independently in one pass produces $AA,AB,BA,BB$ with probability $1/4$ each; half the outputs violate the source's dependency. Reveal one token and recompute the other conditional, and the dependency can be recovered. Parallel predictions may be individually well calibrated while their product is a poor joint model.

The extra call is useful because the conditioning changes after a choice has been made. It is not merely another attempt to improve already-correct marginals. This is why the number of positions revealed together affects the consistency of a result as well as the speed at which the mask count falls.

A monotone sampler keeps revealed values. Remasking adds the option to revisit a poor early choice after more context becomes available. It is a deliberate sampling policy rather than a requirement that every diffusion pass must edit every visible token.

Revision spends extra calls or reduces net progress at a fixed call budget. In a remasking iteration, selected positions are hidden before the network is invoked; logits computed while a token was still visible are not treated as predictions from a masked input. The useful metric is therefore not “tokens predicted per pass” alone; it is final committed output per pass at a target quality. Repeatedly revising one position does not create several new output tokens.

DiffusionGemma illustrates a more sophisticated policy. Its recommended **EB — Entropy Bound** sampler selects low-entropy positions subject to a bound on dependence between jointly revealed positions (a mutual-information bound), renoises unselected positions, and stops only when entropy and prediction stability criteria are satisfied. The sampler is part of the released system contract, not an interchangeable afterthought.

## 10. A generic roofline calculation

Return to the hypothetical dense 8B model and count the cost of a finished output. Scoring many positions in one call is not the same as delivering that many final tokens: the same canvas may need to pass through the network repeatedly. Nothing in this section describes DiffusionGemma’s MoE traffic.

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

The numerator counts the completed text once; the denominator pays for every included pass over it. Intermediate predictions increase the work but not the final output length. This is a canvas-completion calculation, not by itself a prediction of when the first readable fragment reaches the user.

| Calls $S$ | Arithmetic ceiling |
|---:|---:|
| 16 | 3867.2 tok/s |
| 64 | 966.8 tok/s |
| 128 | 483.4 tok/s |

The continuous parity point with the batch-one AR ceiling is approximately $S=295.5$. Integer call counts give 295 as the largest value still above that ceiling.

![VIZ m12/06 — the teaching roofline](assets/modern-llms/en/module-12/m12_06_roofline.png)
*Generic dense-8B roofline. The 0.338% figure is ideal compute time divided by the simplified step floor, not measured accelerator utilization.*

The calculation is useful precisely because it is narrow. It excludes attention’s sequence-length cost, KV traffic, activations, communication, routing, batching, kernel overhead, and actual utilization.

## 11. Steps, batching, and weight precision

Autoregressive batching also amortizes weight reads. Under a crude linear batch-throughput approximation, the $S=64$ diffusion ceiling equals AR at

$$
B^*\approx\frac{966.8}{209.4}=4.62.
$$

The comparison now concerns aggregate throughput across several AR requests versus the completion rate of the diffusion canvas in this scenario. Equal aggregate rates do not imply equal latency for one user: the AR batch shares weight traffic among several sequences.

This is not a claim that the fifth concurrent request always erases the benefit. Actual batching changes KV traffic, tensor shapes, occupancy, and latency objectives. The calculation identifies why low-batch interactive use is a natural setting to investigate parallel denoising; it does not establish an advantage over an already saturated serving system.

The same dense model gives a second teaching comparison at a 256-token block. The compute floor is 4.137 ms. With 16 GB of bf16 weights, the 4.776-ms memory floor still dominates. If every parameter were ideally stored in four bits, the payload would be 4 GB and the memory floor 1.194 ms; the block would become compute-bound. At 16 calls, the arithmetic ceiling is again 3867.2 tok/s.

![VIZ m12/07 — calls versus throughput](assets/modern-llms/en/module-12/m12_07_tps.png)
*Throughput versus call count for the generic dense-8B scenario. The four-bit line is an arithmetic payload hypothesis, not a DiffusionGemma checkpoint measurement.*

A sparse MoE invalidates a naive substitution of “active parameters” for bytes per call. Different canvas positions can select different experts, so the union of touched expert weights may be much larger than a single token’s active count. This is the key reason to keep the following released-model passport separate.

## 12. Level-1 code

The minimal implementation must make two contracts visible: the loss only includes corrupted positions, and one sampler iteration makes exactly one network call before revealing the scheduled number of tokens.

```python
"""M12: summed masked loss and deterministic reveal schedule."""
import numpy as np

def linear_mask_schedule(length: int, steps: int) -> tuple[int, ...]:
    """Target counts from fully masked to fully revealed."""
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, np.integer)) or v <= 0
           for v in (length, steps)):
        raise ValueError("length and steps must be positive")
    return tuple(
        (int(length) * remaining + int(steps) - 1) // int(steps)
        for remaining in range(steps, -1, -1)
    )

def dllm_loss(logits, target_ids, loss_mask, time: float) -> float:
    """(1/t) times the summed NLL on explicitly masked positions."""
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    loss_mask = np.asarray(loss_mask)
    if logits.ndim != 2 or target_ids.shape != (logits.shape[0],):
        raise ValueError("expected logits [T,V] and target_ids [T]")
    if logits.shape[1] == 0 or target_ids.dtype.kind not in "iu":
        raise ValueError("target_ids must be integers and vocabulary non-empty")
    if np.any(target_ids < 0) or np.any(target_ids >= logits.shape[1]):
        raise ValueError("target IDs must be in [0, V)")
    if loss_mask.shape != target_ids.shape or loss_mask.dtype.kind != "b":
        raise ValueError("loss_mask must be boolean [T]")
    if isinstance(time, (bool, np.bool_)) or not (0.0 < time <= 1.0):
        raise ValueError("time must be in (0,1]")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(target_ids.size), target_ids]
    return float(nll[loss_mask].sum() / time)

def sample_dllm(predict, length: int, steps: int, mask_id: int):
    """Greedy confidence reveal with one model invocation per iteration."""
    schedule = linear_mask_schedule(length, steps)
    if isinstance(mask_id, (bool, np.bool_)) or not isinstance(mask_id, (int, np.integer)):
        raise ValueError("mask_id must be an integer")
    vocabulary = None
    tokens = np.full(length, mask_id, dtype=np.int64)
    trace = []
    for step, target_masked in enumerate(schedule[1:], start=1):
        logits = np.asarray(predict(tokens.copy(), step), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != length or logits.shape[1] == 0 or not np.isfinite(logits).all():
            raise ValueError("predict must return finite logits [T,V]")
        if vocabulary is not None and logits.shape[1] != vocabulary:
            raise ValueError("predict changed vocabulary size")
        vocabulary = logits.shape[1]
        if 0 <= mask_id < vocabulary:
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

Read the loop as an execution record: one copied input goes into the network, the stored output supplies both candidates and scores, and only then does the state change. Equal scores choose the lower position index. A schedule with more calls than tokens may include calls that reveal nothing.

Using −1 outside the output vocabulary is a teaching interface, not a universal model requirement. Real models often include a special mask token in their vocabulary and exclude it from final clean-token predictions through parameterization or filtering. The notebook adds categorical draws, entropy ranking, and remasking. Adaptive production samplers need not follow our fixed mask counts.

## 14. Block diffusion, length, and caching

The prefix cache in Module 10 worked because appending a continuation did not change the retained past. A mutable bidirectional canvas has a different dependency structure: changing one position can change representations at other positions on the next call. Even an unchanged token ID can acquire a different representation when its context changes. Token stability alone is therefore insufficient to justify reusing K/V, and the ordinary causal prefix-cache argument does not apply inside the mutable canvas.

Block diffusion separates completed history from a mutable block of length $L_b$. Cache validity requires the history's states to be independent of future blocks. Causality is needed between blocks, not necessarily between every pair of tokens within a completed block. Bidirectional attention inside an already completed block is compatible with block causality as long as it cannot read future blocks. The [Section 19 exercise](/courses/modern-llms/module-12/exercises/) uses the simpler special case of a token-causal prefix.

The active block reads that history and all of its own positions. After the sampler makes its final choices, there is one more distinction to respect: **finished token values are not necessarily finished cache states**. The last denoising call computed representations from its input, and only afterward were its predictions used to update the canvas. That input may still have contained masks or earlier values.

The resulting K/V cannot automatically become completed history. The cache must represent the clean block that was actually selected. BD3-LM runs the clean block to obtain cache states; DiffusionGemma uses its encoder for this step. That pass and prompt processing contribute to end-to-end latency, though they are omitted from the narrow arithmetic in Sections 10–11.

![VIZ m12/08 — block-diffusion spectrum](assets/modern-llms/en/module-12/m12_08_block_spectrum.svg)
*The teaching attention mask makes the completed prefix independent of the active block. Cache states must correspond to the finalized history.*

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

The stable phase trains full-sequence masked diffusion; decay returns to smaller blocks for efficient inference. LLaDA2.1 adds Token-to-Token editing so a revealed value may later be replaced. The published mechanism revises values while keeping the current positions.

## 15. Reading the DiffusionGemma passport

DiffusionGemma is a useful case because its model card exposes architecture, sampler, and hardware conditions rather than only a headline TPS number.

![VIZ m12/09 — DiffusionGemma passport](assets/modern-llms/en/module-12/m12_09_gemma_passport.svg)
*An autoregressive prompt encoder, cacheable context, a bidirectional diffusion canvas, sparse MoE routing, and separate H100 FP8 and Blackwell NVFP4 paths.*

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

The joint vLLM release report gives 1008 tokens/s on H100 FP8 in its own experiment. That is a separate protocol record from the model card’s “over 1100,” not a number to average into it.

The quality side matters as much as speed. The model card reports lower scores than AR Gemma 4 on many benchmarks, with some exceptions. DiffusionGemma is therefore a point on a quality–latency curve, not proof that a generation paradigm dominates every task.

## 16. The dLLM landscape on 11 October 2026

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
| DiffusionGemma | open weights | >1100 tok/s, H100 FP8; 15–20 committed tokens/pass | detailed open passport with stated measurement conditions |

Mercury 2 adds reasoning controls, tool use, and schema-constrained JSON to a commercial dLLM API. Gemini Diffusion makes the measurement boundary unusually explicit: 1479 tok/s excludes 0.84 seconds of fixed overhead. Seed Diffusion targets code and reports H20 throughput. None of those numbers can be ranked fairly without rerunning a shared workload.

iLLaDA highlights a different frontier: improving a from-scratch bidirectional model rather than solely accelerating a converted checkpoint. Its 12T-token pretraining run and variable-length generation show that the research space includes foundation-model quality, not only sampler speed.

## 17. Multimodal diffusion systems

“Multimodal diffusion” can refer to different components.

**LLaDA-V** applies diffusion language modeling to vision-language understanding: a visual representation conditions a diffusion-generated text response.

**LLaDA2.0-Uni** discretizes visual inputs through SigLIP-VQ, uses a MoE dLLM backbone over semantic tokens, and employs a separate diffusion decoder to reconstruct images. Text-token diffusion and pixel/latent generation are distinct parts of the stack.

**DiffusionGemma** accepts interleaved text, images, and video frames but produces text. Its vision encoder does not imply that the response is generated by the same image diffusion decoder used in a text-to-image system.

The useful passport question is:

> Which object is diffused—language tokens, visual semantic tokens, a continuous image latent, or several of these at different stages?

Without that distinction, two systems bearing the same label may have entirely different compute and memory profiles.

## 20. Takeaways and sources

![VIZ m12/10 — diffusion language models at a glance](assets/modern-llms/en/module-12/m12_10_cheatsheet.svg)
*The forward kernel, loss, sampler, teaching roofline, block spectrum, and a released-system evidence passport on one page.*

**Generation protocol.** AR commits one token. Masked diffusion revises a canvas. Block diffusion makes completed history independent of future blocks and cacheable.

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
- [FlexMDM](https://arxiv.org/abs/2509.01025)
- [DiffusionGemma in vLLM](https://vllm-project.github.io/2026/06/10/diffusion-gemma)
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

**Next:** Module 13 returns to the autoregressive path for supervised fine-tuning: preparing demonstrations and teaching a pretrained model the response patterns expected of an assistant.

---

*Landscape verified: 11 October 2026. Product throughput figures are developer-reported results under different hardware and sampling protocols. Sections 10–11 describe a separate generic dense-8B teaching scenario.*
