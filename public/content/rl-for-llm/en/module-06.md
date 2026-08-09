# Module 6. The LLM as a policy: tokens, log-probabilities, and KL

> **Material version:** 2026.11  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Core practice:** level A — browser and CPU; level B — free Colab T4  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 8–10 hours, excluding the optional real-model run

The first five modules developed the general RL toolkit. We now connect it to the concrete objects of an autoregressive model: tokens, logits, masks, log-probabilities, and generation settings. This is where errors hidden by a compact PPO or GRPO formula usually appear: a one-token shift, a wrong response mask, a log-probability from the wrong policy, or a confusion between natural termination and a length limit.

In this module, KL denotes Kullback–Leibler divergence. The deeper information-theoretic development is in *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” (`it_ml.module_03_cross_entropy_kl`). The present module remains self-contained: it derives and measures KL in the token-policy setting, while the cross-course pointer supplies the broader entropy and coding perspective.

**Learning objectives.** By the end of the module, you will be able to:

1. define a token-level Markov decision process (MDP) for ordinary generation and state its boundary precisely;
2. distinguish natural response termination, rule-based stopping, and technical truncation;
3. gather selected-token log-probabilities from a logits tensor without an indexing error;
4. compute response log-probability, mean log-probability, and perplexity;
5. explain the chain rule for KL between autoregressive policies;
6. distinguish exact conditional KL, a sampled log-ratio, and the k1, k2, and k3 estimators;
7. reconstruct the actual behavior policy after temperature, top-k, top-p, and other logits transformations;
8. test whether sampling, stored log-probabilities, and the denominator of a probability ratio refer to the same policy;
9. describe a minimal rollout-batch contract for training an LLM with verifiable reward;
10. reproduce these measurements on an open 0.6-billion-parameter model in Colab.

The English computational route is the executable practice, and the claim/source registry is `Module_6_Sources.md`.

---

> **Practice route.** Graded tasks, worked solutions revealed after an attempt, and executable practice follow the lecture. The explanation remains self-contained.

## Lesson 6.1. A token-level MDP without hidden assumptions

> **Lesson 6.1. A token-level MDP without hidden assumptions**
>
> 1. Why say “the LLM is a policy” again?
> 2. Formal model for an ordinary text response
> 3. EOS, stop tokens, stop strings, and length limits
> 4. Probability of a variable-length response
> 5. Where reward lives in a token trajectory
>
> then 3 assessment steps

### Step 6.1.1 — Why say “the LLM is a policy” again?

Module 1 established the exact correspondence

$$
\pi_\theta(y_t\mid x,y_{<t}),
$$

which defines a distribution over the next token. There it introduced the language of RL. Here the same equation becomes the interface between four parts of a system:

1. the model computes logits;
2. the generation procedure turns those logits into an actual action distribution;
3. the rollout engine selects a token and records data;
4. the trainer recomputes probabilities and builds a loss.

An error in any transition changes the mathematical policy even when the program continues to run. The goal is therefore not to repeat a slogan, but to answer an engineering question:

> **Which object in the tensors and logs is the policy that actually generated the tokens?**

![Token-level MDP](assets/rl-for-llm/en/module-06/M6_token_mdp_EN.png)

The figure separates two arrows. The transformer emits base logits, but the action is selected only after every generation transform has been applied. With no transform, the behavior policy is simply the softmax of the base logits. In a real system, that equality must be verified rather than assumed.

---

### Step 6.1.2 — Formal model for an ordinary text response

Let $x$ be the prepared prompt, including the system message and chat template, and let $y_{<t}=(y_1,\ldots,y_{t-1})$ be the generated response prefix. We can define

$$
s_t=(x,y_{<t}),
\qquad
a_t=y_t.
$$

For pure autoregressive generation, the transition is deterministic:

$$
s_{t+1}=s_t\oplus a_t,
$$

where $\oplus$ appends the token to the prefix. The stochasticity lies in the next-token policy, not in concatenation.

Under this abstraction:

- the action space is the set of admissible token identifiers;
- the initial-state distribution comes from the prompt distribution and prompt preparation rules;
- intermediate rewards are often zero;
- a terminal reward is assigned to the completed response;
- the horizon is random when the policy may select a termination token.

The word *admissible* matters. The tokenizer vocabulary and the actual action set need not coincide. A mask may ban some tokens; EOS may be forbidden before a minimum length; a grammar-constrained decoder may allow only tokens compatible with the current prefix.

**Model boundary.** The state $(x,y_{<t})$ is sufficient only when no hidden external state changes between tokens. Tool calls, search results, user responses, and persistent memory must be included as observations or state variables. Module 11 develops that setting.

> **Architecture extension:** *Modern LLMs*, Module 2 “Tokenization and Embeddings” (`modern_llms.module_02_tokenization`) covers tokenization, special tokens, and chat templates. This module remains self-contained by treating $x$ as an already-tokenized initial sequence.

---

### Step 6.1.3 — EOS, stop tokens, stop strings, and length limits

The end-of-sequence token (EOS) can be an ordinary policy action. If the task contract treats model-selected EOS as the natural end of an answer, the transition after that action is terminal. That useful convention should not be generalized to every reason an inference engine stops emitting tokens.

| Stop event | What happened in the system | RL interpretation |
|---|---|---|
| the model selected EOS | the policy produced the boundary action | usually task termination when EOS belongs to the task contract |
| an allowed stop token or multi-token stop string matched | a task or generation rule recognized a boundary | may be termination, but the contract—not the word `stop`—decides |
| `max_new_tokens` was reached | the technical generation budget was exhausted | usually truncation: the process could have continued |
| a timeout, safety filter, or manual cancellation fired | an external component interrupted generation | keep a separate reason; terminal/truncation semantics are task-specific |

![Stop reason versus episode semantics](assets/rl-for-llm/en/module-06/M6_stopping_semantics_EN.png)

Two metadata layers should therefore remain separate.

1. `stop_reason` records **why the engine stopped**: `eos`, `stop_token`, `stop_string`, `length`, `timeout`, `safety`, and so on.
2. `terminated` and `truncated` record **how the RL task interprets that boundary**. `terminated=True` means there is no continuation in the task; `truncated=True` means collection ended because of an external limit even though the process could have continued.

A hard-coded rule such as “every stop string is terminal” is unsafe. A task-level `</answer>` marker may define completion, whereas a transport-layer delimiter may merely stop streaming. Likewise, a safety interruption can end a product request while still being an external truncation in the environment being studied.

Returned length cannot recover causality. A response may end naturally exactly at the limit, be cut by the limit, finish on a multi-token pattern, or lose a service EOS during detokenization. Store the machine-readable stop reason independently of the episode-boundary flags.

### Step 6.1.4 — Probability of a variable-length response

For $y=(y_1,\ldots,y_T)$, the autoregressive factorization is

$$
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

If a sequence is complete only after EOS, EOS belongs in the product as the final action. A generation engine may remove it from the displayed string, but its token identifier still belongs in the machine record whenever its probability contributes to the training objective.

Taking logarithms turns the product into a sum:

$$
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_t\mid x,y_{<t}).
$$

The sum usually decreases with length because each factor is at most one and each log-probability is non-positive. Sequence log-probabilities therefore cannot be compared across different lengths without a convention. We will normalize by the number of active tokens later, but that does not remove every length or tokenization effect.

A stop string creates another boundary case: the output can end after a multi-token pattern rather than one designated action. A strict probabilistic model must either include the stopping rule in the environment or retain the original tokens before the stop string is stripped.

---

### Step 6.1.5 — Where reward lives in a token trajectory

Tasks with verifiable outcomes often use sparse terminal reward:

$$
r_{t+1}=0\quad (t<T-1),
\qquad
r_T=R(x,y).
$$

For example, $R$ may indicate whether generated code passed tests or whether a numeric answer matched a reference. With $\gamma=1$, every token in the response receives the same return $R(x,y)$.

That does not prove that all tokens were equally useful. A shared scalar is a property of the estimator, not a causal decomposition. Module 4 showed how a full-response policy gradient applies one coefficient to a sum of token log-probabilities. Module 9 returns to group estimates, length normalization, and token attribution.

A system may also use local terms:

- a cost per generated token;
- a penalty for invalid format;
- an intermediate tool outcome;
- token-level KL regularization;
- a process-reward score.

Each addition changes the task. The log should therefore separate raw task reward from shaping terms and regularizers.

---

## Lesson 6.2. Logits, masks, and log-probabilities

> **Lesson 6.2. Logits, masks, and log-probabilities**
>
> 1. From logits to the selected-token probability
> 2. The one-token shift and the response mask
> 3. Sum, mean, and perplexity
> 4. Three roles for log-probabilities
> 5. Numeric precision and a reproducible protocol
>
> then 4 assessment steps

### Step 6.2.1 — From logits to the selected-token probability

Suppose the model returns logits $z_t\in\mathbb{R}^{V}$ at position $t$, with vocabulary size $V$. The base distribution is

$$
\pi_\theta(v\mid s_t)
=
\frac{\exp z_{t,v}}
{\sum_{u=1}^{V}\exp z_{t,u}}.
$$

In practice, we usually need the log-probability

$$
\log \pi_\theta(v\mid s_t)
=
z_{t,v}-\operatorname{LSE}(z_t),
$$

where

$$
\operatorname{LSE}(z_t)
=
m_t+\log\sum_u\exp(z_{t,u}-m_t),
\qquad
m_t=\max_u z_{t,u}.
$$

Subtracting the maximum leaves the distribution unchanged and prevents exponential overflow. In PyTorch, `torch.log_softmax` computes the normalization and `gather` selects the column of the action actually taken.

Keep three tensors distinct:

1. all-token logits with shape $[B,L,V]$;
2. all-token log-probabilities with the same shape;
3. selected-token log-probabilities with shape $[B,L]$.

PPO or GRPO usually needs the third tensor plus a response mask. Full distributions are required for quantities such as exact conditional KL over the vocabulary.

---

### Step 6.2.2 — The one-token shift and the response mask

At position $i$, a causal model predicts the **next** token $u_{i+1}$. For

$$
(u_0,u_1,\ldots,u_{L-1}),
$$

the aligned pairs are

$$
\text{logits}[:,i,:]
\longleftrightarrow
\text{input\_ids}[:,i+1].
$$

![Logit shift and response mask](assets/rl-for-llm/en/module-06/M6_tensor_alignment_EN.png)

A typical implementation is

```python
shifted_logits = logits[:, :-1, :]
shifted_tokens = input_ids[:, 1:]
logp_all = torch.log_softmax(shifted_logits.float(), dim=-1)
selected_logp = logp_all.gather(-1, shifted_tokens[..., None]).squeeze(-1)
```

A mask is then applied. It answers a semantic question: **which actions belong to the policy optimized by this objective?**

For ordinary SFT or response-level RL:

- system and prompt tokens do not contribute to the policy loss;
- response tokens do;
- padding does not;
- EOS contributes when it was an action of the model and is retained in the sequence;
- positions after termination do not.

An off-by-one error is dangerous because every tensor may still have a plausible shape while each action is paired with the neighboring token’s probability. A good test therefore uses a tiny hand-checkable example rather than only shape assertions.

---

### Step 6.2.3 — Sum, mean, and perplexity

Let $m_t\in\{0,1\}$ be the active-token mask and let $\ell_t$ be a selected log-probability. Define

$$
L_{\mathrm{seq}}
=
\sum_t m_t\ell_t,
$$

$$
\bar\ell
=
\frac{\sum_t m_t\ell_t}{\sum_t m_t},
$$

and perplexity (PPL) as

$$
\operatorname{PPL}=\exp(-\bar\ell).
$$

With natural logarithms, PPL is the exponential of mean cross-entropy in nats per token. If every active token had probability $1/K$, then PPL equals $K$. This motivates the “effective number of equally likely choices” intuition, but it is not a literal count for a general distribution.

Three limitations are essential.

1. **PPL depends on the tokenizer.** Different tokenizations produce different token counts, so PPL values are not a universal physical scale.
2. **PPL depends on the mask and template.** Response-only, full-dialogue, and content-token PPL are different quantities.
3. **PPL is not response utility.** A model can assign high probability to a bland or incorrect continuation.

> **Mathematical extension:** *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” (`it_ml.module_03_cross_entropy_kl`) develops cross-entropy, KL, and the coding-length interpretation.

---

### Step 6.2.4 — Three roles for log-probabilities

Post-training systems often keep several models or snapshots of one model side by side.

![Roles of log-probabilities](assets/rl-for-llm/en/module-06/M6_logprob_roles_EN.png)

- $\log\pi_{\mathrm{old}}$ is the probability under the policy that generated the rollout. It belongs in the denominator of a PPO ratio.
- $\log\pi_\theta$ is the probability under the current trainable parameters. The trainer recomputes it and differentiates through it.
- $\log\pi_{\mathrm{ref}}$ is the probability under a frozen reference policy. It anchors a KL term.

The three are not interchangeable even when all models started from the same checkpoint.

There are also two computation modes.

1. **Sampling.** A token is drawn from the actual behavior policy and its log-probability is recorded during rollout.
2. **Rescoring a known sequence.** The whole sequence is passed through the model under teacher forcing, and selected-token probabilities are gathered in parallel.

In an ideal synchronous system, the rescored $\log\pi_{\mathrm{old}}$ equals the stored value. Distributed systems may disagree because of a a different version of the weights, processor order, numeric precision, or generation-engine behavior. A small discrepancy is not automatically a bug, but it must be measured and explained.

---

### Step 6.2.5 — Numeric precision and a reproducible protocol

Log-probabilities react to small logit changes, especially for rare tokens and long accumulated sums. A reproducible protocol records:

- the resolved model commit or revision;
- the tokenizer revision;
- the exact chat-template mode;
- every logits transform and its order;
- the dtype used for `log_softmax` and accumulation;
- generation parameters, random seed, and stop metadata;
- generation-engine, Transformers/TRL, and trainer versions.

A useful safeguard is to normalize the softmax and accumulate long sums in at least FP32 even when the matrix operations use BF16. That does not force byte-identical outputs across kernels and devices, but it removes one common source of avoidable error.

Inference APIs expose different slices of the distribution. Current vLLM documentation states that a non-`None` `logprobs` result includes the chosen token even when it is outside the requested top alternatives, and that `-1` requests all vocabulary log-probabilities subject to engine limits. Full-vocabulary output is useful for exact checks but can be expensive in memory and bandwidth, so the API version and engine cap belong in the experiment record.

Hugging Face Transformers can recover the selected generation scores with `compute_transition_scores`. The course notebook requests `normalize_logits=True` and independently teacher-forces the same sequence. Agreement between those two paths is a stronger invariant than a single plausible-looking score table.

> **Engineering extension:** KV caching, batching, and efficient autoregressive inference are covered in *Modern LLMs*, Module 9 “KV Caches and Efficient Inference” (`modern_llms.module_09_inference`).

## Lesson 6.3. KL to a reference policy

> **Lesson 6.3. KL to a reference policy**
>
> 1. A price for policy drift, not a certificate of quality
> 2. The chain rule for sequence KL
> 3. Exact conditional KL and a sampled log-ratio
> 4. The k1, k2, and k3 estimators
> 5. Estimating a number and differentiating the right objective are different tasks
>
> then 4 assessment steps

### Step 6.3.1 — A price for policy drift, not a certificate of quality

For a fixed prompt $x$, consider

$$
J_x(\theta)
=
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]
-
\beta
D_{\mathrm{KL}}
\!\left(
\pi_\theta(\cdot\mid x)
\middle\|
\pi_{\mathrm{ref}}(\cdot\mid x)
\right).
$$

The first term favors reward. The second makes departure from the reference policy costly. The coefficient $\beta>0$ controls the trade-off.

A KL anchor can:

- limit the rate at which the response distribution changes;
- preserve some behavior of the starting model;
- reduce the incentive to exploit extreme regions where a reward estimator is unreliable;
- provide a drift diagnostic.

Small KL does not prove truthfulness, safety, or utility. The reference policy can be wrong, and a small probability shift can still alter a rare but important behavior. Nor is this the same as offline-RL pessimism from Module 5: that mechanism responds to uncertain values outside data coverage, whereas KL imposes a geometric cost on distributional change.

The full variational derivation of a KL-regularized optimum appears in *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” (`it_ml.module_08_maxent_kl_policy`). Here we focus on how the measurement is computed for an autoregressive LLM.

---

### Step 6.3.2 — The chain rule for sequence KL

For two autoregressive policies $\pi$ and $\pi_{\mathrm{ref}}$,

$$
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\sum_{t=1}^{T}
\log
\frac{\pi(y_t\mid x,y_{<t})}
{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
$$

Taking expectation under $y\sim\pi$ gives sequence KL:

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}\mid x)
=
\mathbb{E}_{y\sim\pi}
\left[
\sum_{t=1}^{T}
\log
\frac{\pi(y_t\mid s_t)}
{\pi_{\mathrm{ref}}(y_t\mid s_t)}
\right].
$$

Equivalently,

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}\mid x)
=
\mathbb{E}_{y\sim\pi}
\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}
\bigl(
\pi(\cdot\mid s_t)
\|\
\pi_{\mathrm{ref}}(\cdot\mid s_t)
\bigr)
\right].
$$

![Chain rule for KL](assets/rl-for-llm/en/module-06/M6_kl_chain_rule_EN.png)

Read the equation carefully. An early token changes not only its local log-ratio but also the future prefix and therefore every later conditional KL. Sequence regularization is a property of trajectory distributions, not a sum of independent preassigned positions.

With random length, one may model EOS as a transition to an absorbing terminal state or sum only over the active mask. Either convention requires compatible stopping rules for the two compared distributions, or an explicit representation of those rules in the process.

---

### Step 6.3.3 — Exact conditional KL and a sampled log-ratio

At a visited prefix $s_t$, exact conditional KL is

$$
D_t
=
\sum_{v=1}^{V}
\pi(v\mid s_t)
\log\frac{\pi(v\mid s_t)}{\pi_{\mathrm{ref}}(v\mid s_t)}.
$$

It requires both full vocabulary distributions and is always non-negative.

If $a_t$ is sampled from $\pi(\cdot\mid s_t)$, one can instead compute

$$
\widehat D_t^{(1)}
=
\log\pi(a_t\mid s_t)
-
\log\pi_{\mathrm{ref}}(a_t\mid s_t).
$$

Its conditional expectation is $D_t$, but an individual value can be negative. That is not “negative KL”; it is a signed contribution from one random action.

Along a trajectory,

$$
\sum_t\widehat D_t^{(1)}
=
\log\pi(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)
$$

is an unbiased estimator of sequence KL **when the trajectory itself is sampled from $\pi$**. If a different behavior policy $\mu$ produced the data, the uncorrected mean estimates a different object.

This distinction—exact over actions at visited prefixes versus sampled over both actions and prefixes—recurs throughout later algorithms.

---

### Step 6.3.4 — The k1, k2, and k3 estimators

Let $x\sim q$ and suppose the target quantity is

$$
D_{\mathrm{KL}}(q\|p)
=
\mathbb{E}_{x\sim q}
\left[
\log\frac{q(x)}{p(x)}
\right],
$$

with

$$
r(x)=\frac{p(x)}{q(x)}
$$

on the support of $q$. Three common sample values are

$$
k_1=-\log r,
\qquad
k_2=\frac12(\log r)^2,
\qquad
k_3=(r-1)-\log r.
$$

![KL estimators](assets/rl-for-llm/en/module-06/M6_kl_estimators_EN.png)

| Estimator | Non-negative per sample | Expectation status | Practical qualification |
|---|---:|---|---|
| $k_1$ | no | unbiased for $D_{\mathrm{KL}}(q\|p)$ when the quantity is finite | individual samples can have either sign |
| $k_2$ | yes | generally biased; a local quadratic approximation | useful as a near-policy diagnostic, not an exact identity |
| $k_3$ | yes | unbiased under the compatible-support conditions below | the control variate does not guarantee small variance in heavy ratio tails |

The pointwise inequality $\log r\le r-1$ proves that $k_3\ge0$. Its unbiasedness needs more care than the phrase “support condition” often suggests. Finite $k_1$ under samples from $q$ requires

$$
q\ll p:
\qquad q(x)>0\Rightarrow p(x)>0.
$$

The extra identity

$$
\mathbb{E}_{q}[r]
=
\sum_{x:q(x)>0}q(x)\frac{p(x)}{q(x)}
=1
$$

also requires $p$ to have no mass outside the support of $q$, i.e. $p\ll q$. Thus exact unbiasedness of $k_3$ in this form assumes compatible common support—mutual absolute continuity in the general statement. If $p$ places mass where $q=0$, the control variate cannot see that mass; if $q>0$ where $p=0$, the logarithm diverges.

Finite-logit softmax distributions share full support. Hard top-k or top-p truncation creates zeros, so the proof must be rechecked after the generation transform rather than inherited from the estimator’s name.

![When k3 helps and when it becomes noisy](assets/rl-for-llm/en/module-06/M6_kl_variance_stress_EN.png)

A finite synthetic example shows both regimes. For two nearby four-point distributions, $k_3$ has a small standard deviation. With $q=(0.999,0.001)$ and $p=(0.9,0.1)$, a rare event creates a heavy ratio tail: the exact KL remains finite, but the standard deviation of $k_3$ is about $2.98$, compared with about $0.15$ for $k_1$. These are exact moments of two finite distributions, not an external benchmark. A control variate can reduce variance locally without making $k_3$ universally superior.

### Step 6.3.5 — Estimating a number and differentiating the right objective are different tasks

The k1–k3 formulas first answer a question about estimating a **KL value**. It does not follow that placing any one of them in an autograd graph produces the gradient of the intended sequence-level KL objective.

There are two distinct reasons.

1. Samples depend on the policy parameters, but differentiating through an already selected discrete token does not by itself account for the derivative of the sampling distribution.
2. In an autoregressive policy, an early token changes the distribution over every later prefix and therefore every later KL contribution.

Tang and Munos analyze implementations in which a scalar estimate has the desired expectation while direct differentiation produces a different gradient or drops the sequential dependency. Later analyses of KL regularization likewise separate three properties: unbiased value estimation, finite variance, and a correct optimization gradient. In this module k1–k3 are **stop-gradient diagnostics and numerical estimators**. A training loss should be derived from the desired regularized policy gradient, not from the cosmetic appeal of a non-negative scalar.

> **Snapshot 2026-08-05.** Current TRL PPO Trainer documentation describes `k1` as the default logging estimator and offers `k3` as a non-negative alternative that the documentation characterizes as lower-variance. That is an implementation description, not a universal theorem: heavy probability-ratio tails can make $k_3$ far noisier. The full placement of old, current, and reference policies in PPO, DPO, RLOO, and GRPO is covered in *Modern LLMs*, Module 12 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” (`modern_llms.module_12_preference_optimization`).

## Lesson 6.4. The actual behavior policy during generation

> **Lesson 6.4. The actual behavior policy during generation**
>
> 1. Base softmax versus post-decoding policy
> 2. Temperature preserves full support when positive
> 3. Top-k and top-p are truncation followed by renormalization
> 4. Log-probability mismatch and the support condition
> 5. A practical consistency rule
>
> then 6 assessment steps

### Step 6.4.1 — Base softmax versus post-decoding policy

Let a transformer emit logits $z_\theta(s)$. The base distribution

$$
\pi_\theta(\cdot\mid s)=\operatorname{softmax}(z_\theta(s))
$$

may pass through:

- temperature scaling;
- repetition penalties;
- token bans;
- grammar constraints;
- top-k, top-p, or min-p truncation;
- forced or forbidden EOS.

Let $\psi$ denote all generation settings and rules. The actual behavior policy is

$$
\mu_{\theta,\psi}(\cdot\mid s)
=
\mathcal{G}_\psi[z_\theta(s)],
$$

which need not equal $\pi_\theta$.

![From logits to the behavior policy](assets/rl-for-llm/en/module-06/M6_decoding_policy_EN.png)

Hugging Face Transformers separates logits processors from stopping criteria, while vLLM exposes temperature, top-p, top-k, min-p, stop-token, and stop-string controls. The operation order is part of the engine specification. “Sampled at top-p 0.9” is therefore incomplete without the version and the rest of the configuration.

---

### Step 6.4.2 — Temperature preserves full support when positive

For $\tau>0$,

$$
\pi_{\theta,\tau}(v\mid s)
=
\frac{\exp(z_v/\tau)}{\sum_u\exp(z_u/\tau)}.
$$

For fixed finite logits:

- $\tau<1$ sharpens the distribution;
- $\tau>1$ flattens it;
- every token retains positive probability.

The last point matters for probability ratios: positive temperature changes mass but not support. Temperature zero is a different rule, typically greedy `argmax`, not an ordinary member of the positive-temperature softmax family.

A log-probability recorded under temperature $\tau$ is

$$
\log\pi_{\theta,\tau}(v\mid s)
=
\frac{z_v}{\tau}
-
\operatorname{LSE}(z/\tau).
$$

Storing the temperature while recomputing log-probabilities from unscaled logits creates the wrong denominator even though the selected token is perfectly valid under both distributions.

---

### Step 6.4.3 — Top-k and top-p are truncation followed by renormalization

Top-k retains the $k$ most probable tokens:

$$
\widetilde p(v)
=
\frac{p(v)\mathbf 1\{v\in S_k\}}
{\sum_{u\in S_k}p(u)}.
$$

Top-p, or nucleus sampling, sorts tokens by probability and chooses the smallest prefix $S_p$ whose mass reaches a threshold $p_0$:

$$
\sum_{v\in S_p}p(v)\ge p_0.
$$

The distribution is then renormalized over $S_p$. Nucleus sampling was introduced as a dynamic way to remove an unreliable tail; the support size changes from one position to the next.

A reproducible implementation must specify:

- how ties are broken;
- whether the token that crosses the threshold is included;
- the order of top-k and top-p when both are used;
- whether truncation happens before or after temperature scaling.

The exercises use a stable descending sort with lower token identifiers breaking ties, followed by temperature, then top-k, then top-p.

---

### Step 6.4.4 — Log-probability mismatch and the support condition

Let $S$ be the retained token set and

$$
Z_S=\sum_{v\in S}p(v).
$$

For a retained token,

$$
\log\widetilde p(v)
=
\log p(v)-\log Z_S.
$$

If a token was sampled from $\widetilde p$ but the log recorded was $\log p(v)$, the denominator is wrong by $-\log Z_S$. Tokens outside $S$ satisfy

$$
\widetilde p(v)=0,
$$

although base softmax assigns them finite probability.

This is not merely a rounding issue. Evaluation from data generated by a behavior policy $\mu$ requires a support condition: every target-policy event to be assessed through probability ratios must lie where $\mu>0$. Hard truncation can violate that condition.

For example, suppose the behavior policy retains tokens carrying base mass $0.85$. Every retained token’s correct log-probability is higher than its base log-probability by

$$
-\log0.85\approx0.1625.
$$

The error repeats at every position and exponentiates in a sequence probability ratio.

---

### Step 6.4.5 — A practical consistency rule

There is no universal commandment that “top-p is always forbidden during training.” The more precise rule is:

> **The data, stored log-probabilities, and mathematical objective must refer to the same behavior policy.**

The simplest regime uses full support, a fixed positive temperature, and no hard truncation. Then a ratio is defined for every sampled token and base and behavior supports coincide.

When truncation is used, the system must:

1. record the log-probability after all transforms;
2. store transform parameters and order;
3. define whether the base or transformed policy is being optimized;
4. verify the support condition for the intended estimator;
5. account for the discrete dependence of top-k/top-p boundaries on logits.

Greedy decoding and beam search are search procedures rather than ordinary sampling from a full stochastic policy. They are useful at inference time but require a separate formalization for standard REINFORCE or PPO. Best-of-N and verifier-guided search are covered in Module 10.

---

## Lesson 6.5. A rollout contract for RL with verifiable rewards

> **Lesson 6.5. A rollout contract for RL with verifiable rewards**
>
> 1. A five-component system
> 2. A minimal batch schema
> 3. Where deterministic concatenation stops being enough
> 4. Level B practice: real logits from a 0.6B model
> 5. Module map
>
> then 7 assessment steps

### Step 6.5.1 — A five-component system

A clean reinforcement learning with verifiable rewards (RLVR) setup can be decomposed into five roles.

1. **Prompt sampler** chooses initial states.
2. **Rollout engine** generates responses from the actual behavior policy.
3. **Verifier or reward model** computes feedback.
4. **Trainer** recomputes current and reference log-probabilities and updates parameters.
5. **Weight synchronization** sends a new policy version back to the engine.

![RLVR rollout contract](assets/rl-for-llm/en/module-06/M6_rollout_contract_EN.png)

This decomposition exposes three kinds of delay:

- prompts may be selected in advance;
- reward may be evaluated asynchronously;
- the engine may keep generating with an older version of the weights.

Module 12 covers distributed infrastructure. The immediate goal is to identify the data without which the training objective cannot even be defined.

---

### Step 6.5.2 — A minimal batch schema

A rollout item commonly needs the following fields.

| Field | Why it is needed |
|---|---|
| `prompt_id` and raw prompt | reproducibility and coverage analysis |
| `prompt_ids` | exact tokenized initial context |
| `response_ids` | policy actions, including special tokens under the chosen convention |
| `attention_mask` | valid input positions |
| `response_mask` | positions included in the policy loss |
| `old_logprobs` | denominator for the actual behavior policy |
| `ref_logprobs` or reference-policy version | KL computation and rescoring |
| `reward_raw` | raw task reward |
| additional reward terms | length, format, shaping, and KL terms kept separate |
| `terminated`, `truncated`, `stop_reason` | correct episode-boundary handling |
| `sampling_config` | temperature, truncation, stopping rules, and seed |
| `policy_version` | weights that generated the response |
| verifier metadata | version, result, and diagnostics |

Some fields can be reconstructed, but doing so increases dependence on external versions. Retokenizing a string after a template update can change the sequence, so storing token identifiers is safer.

Token-level log-probabilities are preferable to only one sequence sum because PPO, masking, length analysis, and local diagnostics all require positional information.

---

### Step 6.5.3 — Where deterministic concatenation stops being enough

For a plain response without tools, the environment really can implement

$$
s_{t+1}=s_t\oplus a_t.
$$

A verifier assigns terminal reward after the trajectory. Under this approximation, most response stochasticity comes from the policy and prompt selection.

Once an action invokes a tool, however,

$$
P(s_{t+1}\mid s_t,a_t)
$$

may depend on a browser, file system, network, user, or simulator randomness. Tool output is an observation, not a token freely selected by the model. It must not be included in `response_mask` as a policy action merely because it appears inside one serialized transcript.

This distinction prepares a central topic of Module 11: the gradient mask must separate agent actions from environment observations.

---

### Step 6.5.4 — Level B practice: real logits from a 0.6B model

The module now ships two separate notebooks:

They share the computational kernel but have independently written explanations and labels. Three profiles are available: `smoke`, `full`, and `real-model`. The canonical release executes only the local synthetic profiles; weight download is disabled by default (`RUN_REAL_MODEL=False`). This keeps the release gate offline and reproducible, and it avoids presenting an unexecuted model branch as a validated result.

The optional `real-model` profile uses the public open-weight `Qwen/Qwen3-0.6B` entry configured in `configs/models.yaml`. Its official model card reports a 0.6B causal language model with 28 layers and a 32,768-token context window, and notes that Transformers versions earlier than 4.51.0 do not recognize `qwen3`. Those are model-card properties, not a claim that the model is best or suitable for every task.

The optional run:

1. applies the chat template and stores exact `prompt_ids`;
2. generates a short response with `return_dict_in_generate=True` and `output_scores=True`;
3. obtains selected-token generation scores through `compute_transition_scores(..., normalize_logits=True)`;
4. independently teacher-forces the same sequence and verifies the one-token shift;
5. computes sum, mean, and PPL over `response_mask` only;
6. compares exact conditional KL between two temperature-scaled distributions at visited prefixes;
7. measures top-p support and its renormalization correction;
8. assembles a minimal JSON rollout record with resolved software and generation metadata.

A direct `transformers.generate()` call reliably returns sequences and scores, but it does not expose one production-grade `stop_reason` contract equivalent to a serving-engine finish reason. The notebook therefore labels the reason as **inferred** from EOS and generated length. When an explicit finish reason is required, use a serving API that provides one and record that API version.

The real profile records the resolved model and tokenizer revisions, chat-template mode, library versions, device, dtype, generation settings, and random seed. Outputs may change with any of them; the assertions target alignment and probability identities rather than a preselected response string.

### Step 6.5.5 — Module map

The core data path is now

$$
\text{logits}
\longrightarrow
\text{actual generation policy}
\longrightarrow
\text{tokens and old log-probabilities}
\longrightarrow
\text{reward}
\longrightarrow
\text{rescoring by current and reference policies}.
$$

Seven questions are enough to audit an implementation.

1. What are the state and action?
2. What counts as termination and what counts as truncation?
3. Which logits position supplied each token probability?
4. Which positions belong to `response_mask`?
5. Which policy actually sampled the actions after all transforms?
6. Does each stored `old_logprob` belong to that policy?
7. Were the model version, tokenizer version, template, and stop reason recorded?

Three equations should be recoverable without prompting:

$$
\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t}),
$$

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=
\mathbb{E}_{y\sim\pi}
\left[
\log\pi(y)-\log\pi_{\mathrm{ref}}(y)
\right],
$$

$$
\log\widetilde p(v)=\log p(v)-\log Z_S
\quad\text{for }v\in S.
$$

Module 7 fills in the next system block: where reward comes from, how preference data are constructed, how the Bradley–Terry model works, and why reward-model quality cannot be reduced to one held-out accuracy number.

---

## Practice route

The graded core set contains 20 tasks: 8 conceptual questions, 6 analytical problems, and 6 programming microtasks. Twelve additional platform tasks cover stopping semantics, EOS accounting, processor order, estimator support, and masking of tool observations. Every prompt is paired with a worked solution revealed after an attempt.

---

## Sources and further reading

The items below support specific mathematical or implementation claims. `Module_6_Sources.md` contains the full claim registry, allowed inference, and limitations.

1. [Hugging Face Transformers: Generation](https://huggingface.co/docs/transformers/main_classes/text_generation) — `compute_transition_scores`, generation score structures, and `normalize_logits`; used for the selected-token cross-check.
2. [Hugging Face Transformers: internal generation utilities](https://huggingface.co/docs/transformers/internal/generation_utils) — the separation between logits processors and stopping criteria. The API does not decide the RL semantics of a boundary for you.
3. [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) — temperature, top-k/top-p, stop rules, and `logprobs`; the sampled token is included, and `-1` requests full-vocabulary log-probabilities subject to engine limits.
4. [John Schulman, “Approximating KL Divergence”](https://joschu.net/blog/kl-approx.html) — the k1, k2, and k3 value estimators. Support and tail behavior still need an explicit check.
5. [Tang & Munos, “On a few pitfalls in KL divergence gradient estimation for RL”](https://arxiv.org/abs/2506.09477) — why a correct scalar expectation does not automatically produce the intended KL gradient.
6. [“Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization”](https://arxiv.org/abs/2510.01555) — additional support, variance, and value-versus-gradient analysis.
7. [Holtzman et al., “The Curious Case of Neural Text Degeneration”](https://arxiv.org/abs/1904.09751) — the primary nucleus-sampling paper; it defines the mechanism but does not make one top-p setting universally appropriate for RL.
8. [Official Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the optional open-weight model’s architecture metadata and Transformers requirement.

For deeper links within the course series:

- *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” (`it_ml.module_03_cross_entropy_kl`) — entropy, cross-entropy, KL, and perplexity;
- *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” (`it_ml.module_08_maxent_kl_policy`) — the variational logic of a KL anchor;
- *Modern LLMs*, Module 2 “Tokenization and Embeddings” (`modern_llms.module_02_tokenization`) — tokenizers, special tokens, and templates;
- *Modern LLMs*, Module 9 “KV Caches and Efficient Inference” (`modern_llms.module_09_inference`) — efficient generation mechanics;
- *Modern LLMs*, Module 12 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” (`modern_llms.module_12_preference_optimization`) — the roles of old, current, and reference policies in the full training loop.
