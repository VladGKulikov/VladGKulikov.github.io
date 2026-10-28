# Module 9. RLVR and GRPO: verifiable rewards, group-relative estimates, and stability

> **Material version:** 2026.21  
> **Explanatory prose revised:** 2026-10-27  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Core practice:** level A — browser and central processing unit (CPU); level B — free Colab T4 with Low-Rank Adaptation (LoRA)  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 10–13 hours, excluding optional model training

[Module 8 “RLHF and direct preference optimization”](module-08.md) compared RLHF with policy optimization on fixed preference pairs. We now examine a fresh-data loop in detail: for each prompt it produces several responses, a checking procedure assigns rewards, and the next training update changes the distribution of future responses.

This regime is called reinforcement learning with verifiable rewards (RLVR). Group Relative Policy Optimization (GRPO) is often used for it, but the two names are not synonyms:

- **RLVR** describes the feedback source: a program, formal system, test suite, or environment can check the outcome;
- **GRPO** describes one way to estimate relative advantages and update a policy without a learned critic.

The practical problem of this module is:

> **Build a reproducible RLVR loop in which the reward actually checks the intended property and the chosen GRPO variant does not hide normalization bias, policy mismatch, or loss of diversity.**

After completing the module, you will be able to:

1. distinguish RLVR from GRPO, PPO, direct preference optimization, and inference-time search;
2. explain the roles of current, old, and reference policies in a group update;
3. derive group-relative advantages and the canonical clipped GRPO objective;
4. explain why “critic-free” does not mean “baseline-free”;
5. design a verifier and distinguish false-positive from false-negative rewards;
6. compute the probability of an informative group and the cost of dynamic resampling;
7. compare original GRPO, Dr. GRPO, DAPO, and GSPO by the exact unit of normalization and importance correction;
8. diagnose entropy loss, length drift, truncation, and generation–training mismatch;
9. discuss the “elicitation versus discovery” question without overstating pass@k evidence;
10. run deterministic CPU exercises and prepare an optional experiment with a roughly 0.6-billion-parameter model.

Practice remains separate from the lecture: [exercises](/courses/rl-for-llm/module-09/exercises/), [assessment questions](/courses/rl-for-llm/module-09/tests/), and the [English notebook](/downloads/rl-for-llm/notebooks/module-09-rlvr-grpo.en.ipynb). Primary sources and the limits of the research claims are listed at the end of the chapter.

---

> **Practice route.** Graded tasks, worked solutions revealed after an attempt, and executable practice follow the lecture. The explanation remains self-contained.

## Lesson 9.1. RLVR as a feedback contract

Begin before the optimizer. Learning from a check requires knowing what it tests, how it can fail, and when a response group offers a comparison at all. Only then does a number in the log become a defensible basis for an update.

### Step 9.1.1 — What makes a reward verifiable

An RL reward may come from a human, a learned model, an environment, or a physical measurement. RLVR identifies a narrower regime: there is a reproducible procedure that checks a response or trajectory.

Examples include:

- comparing a number or symbolic expression with a canonicalized reference;
- running code inside a sandbox against tests;
- checking a proof with a trusted formal kernel;
- evaluating a tool call by the resulting environment state;
- scoring a game or simulator trajectory under explicit rules.

Verifiable does not mean binary. A reward can be real-valued, multi-component, or partially graded. The score must come from a declared checking procedure, and that procedure must test something sufficiently close to the intended property. A deterministic checker returns the same result for the same complete input and version. With randomized tests or a stochastic environment, the test set, environment state, and random seed—or an explicitly defined distribution of trials—also belong to the experiment. Repeating the response text alone need not repeat the checking conditions. The numerical example below uses a deterministic rule.

The loop is

$$
q
\longrightarrow
\{o_i\}_{i=1}^{G}
\longrightarrow
\{r_i\}_{i=1}^{G}
\longrightarrow
\{\widehat A_i\}_{i=1}^{G}
\longrightarrow
\text{policy update}.
$$

Here $q$ is a prompt, $o_i$ are responses sampled by the old policy, $r_i$ are checker outputs, and $\widehat A_i$ are relative estimates within the group.

![The complete RLVR loop](assets/rl-for-llm/en/module-09/M9_rlvr_loop_EN.png)

**Boundary.** Best-of-N with the same checker remains inference-time search when model parameters do not change. DPO remains fixed-data preference optimization when no fresh responses are collected inside the stage. RLVR begins when a verifiable outcome enters a policy update.

---

### Step 9.1.2 — DeepSeek-R1-Zero as a historical case, not a universal recipe

The DeepSeek-R1 technical report presents DeepSeek-R1-Zero as an RL stage started directly from DeepSeek-V3-Base, without a preceding supervised stage containing labeled reasoning traces. The authors report that, during training, responses became longer and more often included intermediate checking, backtracking, and alternative approaches.

The report also documents serious limitations: poor readability, language mixing, and a narrow concentration on formally verifiable tasks. Final DeepSeek-R1 therefore used a multi-stage pipeline with cold-start data, further supervised fine-tuning, rejection-sampled data, and later RL stages. The historical lesson comes from both parts: the pure run shows that a verifiable signal can strongly reshape the response distribution, while the final pipeline shows that this stage alone is not a complete recipe for a generally useful model.

The R1-Zero reward is described through two rule-based components:

$$
R_{\mathrm{rule}}
=
R_{\mathrm{accuracy}}
+
R_{\mathrm{format}}.
$$

Mathematics used final-answer checking, code used compilation and tests, and a separate term encouraged the required response structure.

The compact GRPO formula displayed in the report includes a KL term with coefficient $\beta$ and a full-response probability ratio. The report does not, however, provide a machine-complete run passport from which every reference-policy schedule, rollout-reuse detail, and estimator choice can be reconstructed. The careful statement is therefore: **the reported objective allows KL regularization; the actual coefficient and remaining engineering details belong to the versioned implementation or run configuration**. “R1-Zero was definitionally GRPO with no KL whatsoever” and “every RLVR system should copy the R1 configuration” both go beyond the available evidence.

> **Epistemic status:** pipeline design, reward types, and observed behavior are claims from the model authors’ technical report. They demonstrate one successful large-scale process, not that the absence of preliminary SFT, the same reward, or the same hyperparameters is optimal for another model or task distribution.

### Step 9.1.3 — A verifier is a stack, not a single Boolean

“Reward equals correctness” hides a sequence of decisions:

$$
\text{response text}
\rightarrow
\text{extraction}
\rightarrow
\text{canonicalization}
\rightarrow
\text{checking}
\rightarrow
\text{reward}.
$$

![The verifier contract](assets/rl-for-llm/en/module-09/M9_verifier_contract_EN.png)

A numerical task must specify where the answer appears, how fractions are interpreted, whether rounding is accepted, and how multiple final answers are handled. Code adds language version, dependencies, time and memory limits, network and file-system access, randomness, and hidden tests.

Two errors have different consequences:

- **false-positive reward:** an incorrect response passes, giving the policy a direct incentive to exploit the checker;
- **false-negative reward:** a correct response fails, suppressing a useful trajectory.

False positives are especially dangerous under repeated optimization because a rare loophole can become the dominant strategy. False negatives make the signal sparser and reduce the effective number of successes.

A minimum checker passport records:

1. parser and reference-answer versions;
2. canonicalization rules;
3. accepted formats;
4. execution restrictions;
5. positive, negative, and adversarial tests;
6. failure categories: no answer, parse error, timeout, exception, wrong result;
7. a configuration hash for later replay.

The verifier is part of the task specification, not an auxiliary implementation detail.

---

### Step 9.1.4 — A numerical verifier: useful core, explicit limits

The teaching implementation uses a narrow, testable contract:

1. take the contents of the last `\\boxed{...}` when present;
2. otherwise use the last number or fraction;
3. remove thousands separators;
4. parse fractions exactly with `Fraction`;
5. compare with absolute and relative tolerance.

```python
from module_9_reference import verify_numeric_answer

assert verify_numeric_answer(r"Final: \\boxed{3/7}", "0.428571") == 1
```

This is appropriate for an arithmetic exercise but is not a general mathematical verifier. It does not automatically know that $\sqrt{8}=2\sqrt{2}$, handle sets and intervals, track units, or verify conditions of existence.

Before connecting any rule to RL, test at least:

- correct answers in every allowed format;
- plausible wrong answers;
- a gold number mentioned inside an incorrect derivation;
- several `\\boxed{}` blocks;
- zero denominators;
- `NaN`, infinities, and overflow;
- very long responses;
- attempts to inject control sequences or code.

**Engineering rule.** Run checker tests first, collect rewards without updating weights second, inspect failures third, and only then enable learning. Otherwise the policy explores undocumented parser behavior instead of the intended task.

---

### Step 9.1.5 — When does a group carry a learning signal?

Suppose each of $G$ conditionally independent responses is correct with probability $p$, and reward is binary. A group is informative for a centered group estimate when it contains at least one success and one failure:

$$
P_{\mathrm{info}}(p,G)
=
1-p^G-(1-p)^G.
$$

![Probability of an informative group](assets/rl-for-llm/en/module-09/M9_informative_groups_EN.png)

When $p\approx0$, almost every group is all wrong; when $p\approx1$, almost every group is all correct. The probability of a nondegenerate group is highest near $p=1/2$. Here “informative” means that centering leaves a relative reward signal. It says nothing yet about verifier correctness, the direction of the full update, or generalization. A group can distinguish responses according to the wrong criterion, and nonzero coefficients need not produce a nonzero vector sum. The calculation concerns availability of this particular signal, not overall learning value per unit of compute.

For $G=8$:

| $p$ | $P_{\mathrm{info}}$ |
|---:|---:|
| 0.01 | 0.0773 |
| 0.10 | 0.5695 |
| 0.50 | 0.9922 |
| 0.90 | 0.5695 |
| 0.99 | 0.0773 |

To obtain $B$ informative prompts at an approximately stationary rate,

$$
\mathbb E[N_{\mathrm{draw}}]
=
\frac{B}{P_{\mathrm{info}}}.
$$

This motivates dynamic resampling and a moving difficulty curriculum. It also changes the training distribution: easy and nearly impossible prompts stop contributing gradients. Keep the original stream, rejection reason, and evaluation on the complete held-out distribution.

Under a draw-once-then-accept-or-reject protocol, prompts whose groups more often pass the filter appear more often in the accepted batch. Generation for rejected groups has still been paid for. The accepted task distribution and the cost per accepted prompt are therefore two consequences of the same selection rule. Retrying one prompt until it yields a mixed group would be a different protocol; the cost expression must be interpreted with the unit of a draw held fixed.

The formula assumes independent draws conditional on one fixed prompt and generation protocol. Low temperature, a fixed template, and independently repeated prefixes can reduce entropy while preserving independence. Reusing a sampled prefix or a search branch across candidates can couple draws. Across prompts, different success probabilities also matter: average the per-prompt formula rather than substituting a pooled success rate. Duplicate answers alone do not identify dependence.

---

## Lesson 9.2. Canonical GRPO

The checked responses are now available. We can follow their rewards into fixed coefficients, bring stored probabilities together with current ones, and trace the differentiable path to the parameters. The central object is one batch that is collected first and optimized afterwards.

### Step 9.2.1 — Three policies and one data source

For prompt $q$, the old policy $\pi_{\theta_{\mathrm{old}}}$ samples a group

$$
\{o_i\}_{i=1}^{G}
\sim
\pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
$$

Three roles must then remain distinct:

1. **old policy** — the actual source of stored actions and the denominator of importance ratios;
2. **current policy** $\pi_\theta$ — the model whose gradient is being computed;
3. **reference policy** $\pi_{\mathrm{ref}}$ — an optional KL anchor.

![Three policy roles in one update](assets/rl-for-llm/en/module-09/M9_policy_roles_EN.png)

Old and reference policies may start from identical weights, but they are different mathematical objects. The old policy is frozen for a rollout batch and replaced when a new batch is collected. The reference may remain frozen, be replaced under an explicit schedule, or be omitted. The current policy changes while the batch is optimized.

For token $o_{i,t}$, define

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}.
$$

The denominator must describe the **actual behavior policy** after temperature, masks, top-k/top-p, and any other logit processing. If generation used a transformed distribution, raw-softmax log-probabilities cannot be substituted merely because they are easier to obtain. Stored old log-probabilities, decoding parameters, and the generator version belong in the rollout record.

The numerator needs an equally explicit definition: is it the current model's raw softmax, or its policy after the same generation transforms? When we equate old and current policies before an update, we mean these distributions, not merely their checkpoint files. A numerator and denominator using different decoding rules can disagree before the optimizer has taken a single step. That requires the separate interpretation developed in Module 6.

### Step 9.2.2 — Group baseline and advantage estimate

For rewards $r_1,\ldots,r_G$, canonical GRPO uses

$$
\bar r
=
\frac1G\sum_{i=1}^{G}r_i,
$$

$$
\widehat A_i
=
\frac{r_i-\bar r}{s_r+\varepsilon},
$$

where $s_r$ is the within-group standard deviation.

Each response is first compared with other attempts on the same prompt, then measured in units of that group's reward spread. The result is a relative training coefficient, not a correctness probability, an absolute utility, or the exact $A^\pi(s_t,a_t)$ at every prefix. When the batch is reused, rewards and their group statistics are held fixed. Backpropagation does not differentiate the checking procedure or the earlier random draw of responses.

The same $\widehat A_i$ is usually applied to every token in response $o_i$. This removes the learned value model, but it does not solve token-level credit assignment: every token in a successful response receives the same sign.

“GRPO has no baseline” is false. The group mean is a baseline. What GRPO removes is a **learned critic** that predicts state values.

If every reward is equal, the centered reward contribution vanishes. A separate KL term, entropy regularizer, or other auxiliary objective may still produce an update. A positive numerical $\varepsilon$ makes zero spread well defined; it does not manufacture reward information.

Now take a binary group of $G>1$ responses with exactly one success. With the population convention,

$$
\bar r=\frac1G,
\qquad
s_r^2
=\frac1G\left[
\left(1-\frac1G\right)^2
+(G-1)\left(0-\frac1G\right)^2
\right]
=\frac{G-1}{G^2}.
$$

Thus $s_r=\sqrt{G-1}/G$. Dividing the two possible deviations from $1/G$ by this scale gives

$$
\widehat A_{\mathrm{success}}
=
\sqrt{G-1},
$$

and every failure has

$$
\widehat A_{\mathrm{failure}}
=
-\frac1{\sqrt{G-1}},
$$

ignoring the numerical $\varepsilon$. The group estimates sum to zero, but their scale depends on group composition.

A zero coefficient sum does not switch learning off. Each coefficient multiplies a different response's gradient vector. Those vectors need not cancel when combined with positive and negative weights. Centering defines a comparison; the response-dependent derivatives turn it into a redistribution of probability, just as in the group-baseline discussion in Module 4.

The numbers above use divisor $G$ for the reward variance. A sample standard deviation with divisor $G-1$ changes the scale. Also keep the normalization constant $\varepsilon$ distinct from the clipping parameter $\epsilon$: the similar symbols have different jobs.

---

### Step 9.2.3 — The clipped objective

Follow one response after the group has been checked. Its tokens are now fixed, and its old log-probabilities come from the rollout record. The current model rescores the same actions on the same prefixes. Each token receives a ratio $\rho_{i,t}$, a fixed response coefficient $\widehat A_i$, and a clipped-surrogate contribution. The contributions are then reduced. Verification determines which changes are desirable; current log-probabilities supply the differentiable route to the parameters.

An implementation that minimizes a loss takes the negative of the objective $J$ below. This sign convention does not reverse the meaning of an advantage: a positive coefficient locally favors the selected action, while a negative coefficient disfavors it, subject to clipping and shared parameters.

The original DeepSeekMath presentation uses token-level ratios and averages token terms within each response before averaging responses:

$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb E
\left[
\frac1G
\sum_{i=1}^{G}
\frac1{|o_i|}
\sum_{t=1}^{|o_i|}
\left(
\min\left[
\rho_{i,t}\widehat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\widehat A_i
\right]
-
\beta K_{i,t}
\right)
\right].
$$

![From a response group to the clipped objective](assets/rl-for-llm/en/module-09/M9_grpo_objective_EN.png)

$K_{i,t}$ is the chosen estimator of deviation from the reference. Its exact form matters: a sampled log-ratio, a non-negative sample form such as $k_3$, and the exact categorical KL have different variance, bias, and gradient contracts.

Choosing a scalar to log is not yet choosing its optimization gradient. Module 6 showed why one must specify the sampling distribution of actions and prefixes, the stop-gradient operations, and whether regularization enters the loss directly or has already been included in reward. Swapping diagnostic estimators without that derivation can change the update. Even an exact vocabulary sum on old prefixes is not automatically the exact gradient of current-policy sequence KL.

Clipping behaves differently for the two advantage signs:

- when $\widehat A_i>0$, the upper branch stops further growth of the surrogate contribution after the ratio has increased too far;
- when $\widehat A_i<0$, the lower branch limits the apparent gain from decreasing probability too aggressively;
- clipping does not project the policy into a trust region, force every $\rho$ into the interval, or impose a strict KL bound.

Notation differs across papers and libraries. DeepSeekMath uses token ratios; the DeepSeek-R1 report displays a compact sequence-level expression; current libraries expose several ratio levels and reduction schemes. The word GRPO therefore does not reconstruct an implementation. At minimum, specify the ratio unit, reduction denominator, epochs per rollout batch, truncation handling, and KL contract.

### Step 9.2.4 — Why the first update may not need a nontrivial ratio

When rollouts were generated by the current policy, rescoring uses the same distributional transforms, and the parameters have not yet changed,

$$
\pi_\theta
=
\pi_{\theta_{\mathrm{old}}}
$$

at the evaluation point, so $\rho_{i,t}=1$. The denominator remains a recorded value or a frozen-policy result. Only the numerator varies with the trainable parameters:

$$
\begin{aligned}
\nabla_\theta\rho_{i,t}(\theta)
&=\rho_{i,t}(\theta)\,
\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}),\\[4pt]
\left.\nabla_\theta\rho_{i,t}\right|_{\theta=\theta_{\mathrm{old}}}
&=\left.\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\right|_{\theta=\theta_{\mathrm{old}}}.
\end{aligned}
$$

A value of one is therefore compatible with a nonzero derivative. Replacing the ratio by the constant `1` in the computational graph would lose that derivative. With positive clipping widths, one is inside the interval, so the limiting branch is inactive at this initial evaluation.

Ratios matter after repeated mini-batches, several optimization iterations on the same completions, or asynchronous generator lag. The data then becomes progressively less on-policy.

Count actual parameter updates rather than backward calls alone: gradient accumulation across several pieces without an optimizer step does not yet change the policy. Conversely, after the first parameter update, the rest of the saved rollout belongs to an earlier distribution. A small reuse count is not itself a bound on policy distance; ratios and the other diagnostics must establish whether the reuse remains modest.

A precise classification is:

- fresh responses and one update: effectively on-policy;
- several iterations over one rollout batch: near/on-policy with bounded reuse;
- large asynchronous lag or a long-lived buffer: off-policy, requiring separate analysis.

Log `num_iterations`, rollout age, generator checkpoint, and the ratio distribution. “We used GRPO” conveys none of these details.

---

### Step 9.2.5 — Relationship to REINFORCE, RLOO, and PPO

Remove KL and examine the initial point where the policies coincide and clipping is inactive. For the fixed group, the derivative of the objective in 9.2.3 is

$$
\widehat g_{\mathrm{mean}}
=\frac1G\sum_{i=1}^{G}
\frac{\widehat A_i}{|o_i|}
\left.\nabla_\theta\log\pi_\theta(o_i\mid q)\right|_{\theta=\theta_{\mathrm{old}}}.
$$

Summing token log-probabilities has produced a full-response log-probability, but the response-length denominator remains. Removing clipping or reuse does not remove that reduction.

Make one additional choice: sum token contributions rather than averaging them within each response. This gives the full-response REINFORCE-like expression:

$$
\widehat g
=
\frac1G
\sum_{i=1}^{G}
\widehat A_i
\nabla_\theta
\log\pi_\theta(o_i\mid q).
$$

With equal response lengths, these reductions differ by one common multiplier. With unequal lengths, they reweight examples relative to one another. “REINFORCE-like” describes the form, not an unbiasedness guarantee for arbitrary $\widehat A_i$. The common mean includes the response's own reward, and the random $s_r$ depends on the entire group. Neither dependence disappears by renaming the estimator.

REINFORCE Leave-One-Out (RLOO) instead uses the mean of the **other** rewards for response $i$:

$$
b_{-i}
=
\frac1{G-1}\sum_{j\ne i}r_j.
$$

Under conditionally independent sampling, this baseline is independent of the current action and avoids the finite-sample $(G-1)/G$ scaling induced by the common sample mean. GRPO generally retains the common group mean and adds normalization, clipping, and sometimes KL.

The $(G-1)/G$ statement belongs to the unstandardized group-baseline comparison from Module 4 before random standardization, clipping, or other reweighting operations are introduced. After those operations, the complete estimator needs its own analysis.

PPO uses a learned critic and stepwise advantage estimates. GRPO replaces the critic with comparisons among responses to the same prompt. That saves model memory while increasing dependence on rollout count, verifier quality, and within-group diversity.

**Compact map:** RLVR specifies where reward comes from; REINFORCE, RLOO, GRPO, and PPO specify how the gradient is estimated and applied.

---

## Lesson 9.3. Normalization and algorithm variants

The GRPO formula is assembled, but its averages are not merely implementation details. Different denominators change the coefficients assigned to prompts, responses, and tokens. We will separate those choices before reading algorithm names as combinations of specific operations.

### Step 9.3.1 — Dividing by group standard deviation reweights prompts

The normalization

$$
\widehat A_i
=
\frac{r_i-\bar r}{s_r+\varepsilon}
$$

makes the spread of each nondegenerate group approximately unit scale. That is convenient numerically, but it also changes the relative weight of prompts.

Consider centered rewards

$$
(-0.75,0.25,0.25,0.25)
$$

and

$$
(-0.075,0.025,0.025,0.025).
$$

The second group has ten times smaller differences. Ignoring the numerical $\varepsilon$, division by each group\'s own standard deviation makes the normalized advantages identical. A barely differentiated partial signal receives the same scale as a large reward gap.

That example concerns reward scale, not yet binary task difficulty. For binary rewards, the connection becomes explicit: with empirical success fraction $\widehat p$, the population deviation is $s_r=\sqrt{\widehat p(1-\widehat p)}$. The scale now depends on the group's success composition. A rare success among failures gets a large positive coefficient, as the one-success calculation showed. If no success occurs at all, however, the centered signal is still zero. A large coefficient conditional on observing an event and a small probability of observing it must be considered together.

Standardization is not merely a guaranteed variance reduction for unchanged REINFORCE. Its random denominator is computed from the same rewards as the numerator. It changes the estimator and the relative emphasis on prompts; preserving the original expected gradient requires an argument about the whole construction.

*Understanding R1-Zero-Like Training: A Critical Perspective* calls this question-level difficulty bias and proposes Dr. GRPO with

$$
\widehat A_i^{\mathrm{Dr}}
=
r_i-\bar r
$$

and no within-group standard-deviation scaling.

The correction removes the reweighting mechanism identified in that analysis. “Unbiased” should be read relative to the authors' particular objective and assumptions, not as a claim that every estimator used in a Dr. GRPO implementation is free of every source of bias.

This is not free. Without normalization, update scale depends directly on reward units and batch composition. Components in $[0,1]$ and $[-100,100]$ cannot be combined without explicit calibration.

---

### Step 9.3.2 — Length normalization decides who receives gradient

Original GRPO averages each response over its own tokens:

$$
\frac1{|o_i|}
\sum_t \ell_{i,t}.
$$

Each response therefore has a length-independent sum of arithmetic coefficients, while each token term carries a coefficient proportional to $1/|o_i|$. All else equal, a long failure gives each token less negative weight than a short failure. This is a statement about reduction coefficients, not equal gradient norms: token derivatives differ in direction and magnitude and can partly cancel.

Dr. GRPO replaces the response-specific denominator with a fixed constant $L$, such as the maximum completion length:

$$
J_{\mathrm{Dr}}
=
\mathbb E
\left[
\frac1{GL}
\sum_i\sum_t\ell_{i,t}
\right].
$$

DAPO normalizes by the total number of active tokens:

$$
J_{\mathrm{DAPO-token}}
=
\mathbb E
\left[
\frac1{\sum_i|o_i|}
\sum_i\sum_t\ell_{i,t}
\right].
$$

![How normalization changes response weights](assets/rl-for-llm/en/module-09/M9_normalization_bias_EN.png)

Both remove the per-response length denominator, but they are not identical. Dr. GRPO uses a denominator independent of random batch lengths; DAPO’s denominator changes with the active-token count. This can matter across distributed layouts and gradient accumulation schemes.

Here $\ell_{i,t}$ denotes a token contribution to the chosen surrogate, not necessarily a bare log-probability. The summation domain must be explicit as well. DAPO's denominator counts all active tokens in the declared reduction batch. Averaging independently normalized sub-batches generally differs from dividing their combined numerator by their combined token count; matching the latter requires weighting the pieces accordingly. The denominator is therefore a mathematical choice before it becomes a distributed-systems detail in Module 12.

Neither objective proves that longer responses are useful. It only allocates gradient budget among tokens.

---

### Step 9.3.3 — DAPO: four related but independent changes

Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) combines four techniques.

1. **Clip-Higher.** It separates the lower and upper ranges:

$$
\operatorname{clip}\!\left(
\rho,\,1-\epsilon_{\mathrm{low}},\,
1+\epsilon_{\mathrm{high}}\right),
\qquad
\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}.
$$

The authors argue that an early upper clip can suppress growth of rare but useful tokens. This is an empirical mechanism and stabilization choice, not a guarantee of diversity.

These are boundaries of the clipping operation, not constraints that keep $\rho$ inside the interval. A ratio can cross a boundary while the surrogate changes branches. Clip-Higher should therefore be interpreted together with the positive contributions that actually reach the upper plateau, not merely with the configured width.

2. **Dynamic sampling.** All-equal reward groups are rejected and generation continues until enough informative prompts have been collected. The costs are additional tokens and a changed prompt distribution.

3. **Token-level policy-gradient loss.** All active tokens in the batch share one normalizer instead of averaging each response first.

4. **Overlong reward shaping.** The paper first evaluates masking truncated responses, then introduces the soft penalty

$$
R_{\mathrm{length}}(y)
=
\begin{cases}
0,
& |y|\le L_{\max}-L_{\mathrm{cache}},\\[4pt]
\dfrac{L_{\max}-L_{\mathrm{cache}}-|y|}{L_{\mathrm{cache}}},
& L_{\max}-L_{\mathrm{cache}}<|y|\le L_{\max},\\[8pt]
-1,
& |y|>L_{\max}.
\end{cases}
$$

DAPO therefore cannot be summarized as “always drop truncated responses.” Masking was one investigated variant; the named package includes a gradual length reward near the boundary.

---

### Step 9.3.4 — GSPO: a normalized response-level ratio

Group Sequence Policy Optimization (GSPO) changes the unit of correction. It uses one **length-normalized** response-level ratio:

$$
s_i(\theta)
=
\left(
\frac{\pi_\theta(o_i\mid q)}
{\pi_{\theta_{\mathrm{old}}}(o_i\mid q)}
\right)^{1/|o_i|}
=
\exp\left[
\frac1{|o_i|}
\sum_t
\log\rho_{i,t}
\right].
$$

The sequence-level term is clipped:

$$
J_{\mathrm{GSPO}}
=
\mathbb E
\left[
\frac1G\sum_i
\min\left(
 s_i\widehat A_i,
 \operatorname{clip}(s_i,1-\epsilon,1+\epsilon)\widehat A_i
\right)
\right].
$$

This is the geometric mean of token ratios, not the unnormalized product of full-sequence probabilities.

Taking the root is part of the surrogate's design. Exact importance sampling of an expectation over whole responses uses the full likelihood ratio, not its root. Consequently, $s_i$ is not a universal unbiased correction for arbitrary sequence-level expectations. A more manageable clipping scale is a choice about the update, not an unchanged importance-sampling identity. The GSPO authors argue that the unit of correction and clipping should match the sequence-level reward. They report improved stability in the long-response and mixture-of-experts settings they studied. These are reported experiments, not a theorem of universal superiority.

Differentiate one $s_i$ while holding the recorded response length and old-policy denominator fixed:

$$
\nabla_\theta s_i(\theta)
=\frac{s_i(\theta)}{|o_i|}
\sum_{t=1}^{|o_i|}
\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

On the unclipped branch, this derivative is multiplied by $\widehat A_i$. All tokens share one sequence-level multiplier and one decision about whether the contribution is on a plateau. When the constant clipped branch is selected, this response's reward contribution through that term has zero derivative. Its probabilities may still change through other responses or auxiliary terms because the parameters are shared.

Initially $s_i=1$. With identical fixed advantages, matching policies, no KL, and inactive clipping, this gradient equals $\widehat g_{\mathrm{mean}}$ from 9.2.5. The important difference emerges after departure from that point: a shared reweighting and clipping decision replaces tokenwise decisions. Token log-ratio changes can compensate inside the average, so a sequence-level plateau is not a bound on each token ratio. GSPO clip ranges are not numerically interchangeable with token-level GRPO ranges because the ratios live on different scales.

---

### Step 9.3.5 — A method name is not a full specification

![Map of group-policy variants](assets/rl-for-llm/en/module-09/M9_algorithm_map_EN.png)

| Variant | Reward scaling | Token-term normalization | Ratio level | Additional choices |
|---|---|---|---|---|
| Original GRPO | group centering and standard deviation | per response length | token | symmetric clip, optional KL |
| Dr. GRPO | centering only | fixed constant | token | removes the two mechanisms identified by its authors |
| DAPO | usually group scaling | active tokens in batch | token | Clip-Higher, dynamic sampling, length shaping |
| GSPO | group-relative estimate | sequence objective | normalized response ratio | geometric-mean ratio and sequence clip |

Production code often creates hybrids: DAPO token normalization without standard-deviation scaling, asymmetric clipping with Dr. GRPO, or a sequence-level ratio with another reduction. Such a run cannot be described honestly by one acronym.

A minimum algorithm record includes:

- behavior-policy source and version;
- completions per prompt;
- reward-component aggregation;
- reward centering and scaling;
- probability-ratio level;
- clipping function;
- token, response, and device normalization;
- rollout-reuse iterations;
- KL estimator and coefficient;
- truncation masks and stop reasons.

**Software snapshot on 2026-08-05.** TRL 1.9.2 exposes these axes independently. Its `GRPOConfig` documentation lists, among other defaults:

```text
loss_type="dapo"
scale_rewards="group"
importance_sampling_level="token"
beta=0.0
mask_truncated_completions=False
```

Other modes include `loss_type="dr_grpo"`, batch-level reward scaling, sequence-level ratios, and explicit masking of truncated completions. These defaults describe one library release, not the mathematical definition of GRPO. A reproducible run stores the actual arguments instead of relying on remembered defaults from another version.

## Lesson 9.4. Training dynamics and diagnosis

A completely consistent loss can still produce an unhelpful result. We now compare what the optimizer rewards with what the updated policy actually generates. Diagnosis should explain why one metric improves while another deteriorates, rather than simply record their movement.

### Step 9.4.1 — Entropy: specify what is measured

For prefix $s_t$, token entropy is

$$
H_t
=
-\sum_{v\in\mathcal V}
\pi(v\mid s_t)
\log\pi(v\mid s_t).
$$

Logs typically average $H_t$ over active tokens and completions. A single mean hides structure: opening moves, operation choices, formatting tokens, and final-answer tokens may behave differently.

Distinguish:

- entropy of the base softmax;
- entropy of the actual behavior policy after sampling transforms;
- full-response diversity;
- semantic strategy diversity;
- held-out pass@k.

These quantities are related but not identical. High token entropy can be spent on punctuation; low entropy can encode a stable correct algorithm.

[*The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*](https://arxiv.org/abs/2505.22617) studies rapid entropy decline and relates entropy changes to a covariance property of updates; its interventions target tokens with a high corresponding contribution. A separate paper, [*Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning*](https://arxiv.org/abs/2506.01939), investigates updates restricted to a small fraction of high-entropy positions. High distributional entropy and a high covariance contribution are not interchangeable properties, so the findings should not be merged.

The 2026 preprint [*Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization*](https://arxiv.org/abs/2605.11491) studies balancing entropy-increasing and entropy-decreasing updates. It supplies a further research hypothesis and results from the authors' experiments, not a universal prescription. None of these studies defines one transferable “healthy entropy” threshold for all models, rewards, and response lengths.

---

### Step 9.4.2 — Loss of diversity and interventions

Warning signs include:

1. entropy falls without stabilizing;
2. responses within a group become near duplicates;
3. informative-group rate falls faster than accuracy alone explains;
4. pass@1 rises while the pass@k curve stops improving;
5. upper clip fraction and heavy-tailed ratios grow;
6. different random seeds converge to the same response template.

Read these signals together. With binary reward, fewer mixed groups can mean near-universal success or near-universal failure. Track all-success and all-failure groups separately: one aggregate cannot distinguish task mastery from loss of access to a positive learning signal.

Possible interventions are:

- increase diversity of the actual behavior policy;
- widen the positive upper clip;
- add cautious entropy regularization;
- increase $G$ or prompt diversity;
- move the difficulty band;
- reduce rollout-reuse iterations;
- stop before severe collapse;
- separate reward components when one dominates.

Every intervention has a cost. More sampling temperature can lower correctness and raise generation expense. An entropy bonus rewards useful and useless uncertainty. Larger $G$ costs more generation. Clip-Higher relaxes one constraint but cannot repair a wrong reward.

An intervention should match a mechanism that is actually active. On the first fresh evaluation with $\rho=1$, widening the upper bound does not change the reward contribution because clipping is not yet active. It may matter later during reuse. A larger group gives more chances to observe differences when $0<p<1$; it does not guarantee a new solution strategy. Diagnose the operating mechanism before changing its setting.

---

### Step 9.4.3 — Length, EOS, and technical truncation

A growing mean response length may indicate:

- useful additional reasoning steps;
- weak per-token penalties for long failures;
- repetition encouraged by format or reward;
- decreasing probability of the end-of-sequence (EOS) token;
- a rising fraction of responses cut by the length limit.

Log length together with:

- reward or correctness by length bucket;
- natural termination and truncation rates;
- EOS position;
- length after removing control tags;
- repetition and parse-failure rates;
- generation cost.

A truncated response is not necessarily wrong: the checker did not observe its ending. Unconditionally ignoring every truncated response is also unsafe because the policy may learn to avoid negative feedback by never stopping. Choose and record a contract: masking, a soft penalty, a separate label, or a larger budget.

The target is not “longer” or “shorter” responses. It is better quality under a declared budget and an interpretable stopping process.

---

### Step 9.4.4 — Elicitation of reachable behavior versus new effective strategies

A large pass@1 increase after RL does not by itself explain the source of improvement.

One hypothesis is that the policy reweighted successful trajectories that the base model could already produce under a sufficiently large sampling budget. This is **elicitation of practically reachable behavior**. Another is that training made strategies practically reachable that were essentially absent under the original budget. This is **new effective strategy discovery**.

Use “support” carefully. A standard softmax gives most tokens nonzero probability. The operational object is effective support under fixed temperature, maximum length, and sampling budget.

A minimum comparison protocol is:

1. plot base and RL pass@k curves under identical decoding;
2. compare strategy clusters, not only final answers;
3. use fresh tasks and contamination controls;
4. repeat across random seeds;
5. separate RL effects from later SFT or distillation.

Pass@k curves describe access to correct outcomes under a sampling budget, not the origin of a particular strategy. The same answer can arise through different paths; failure to observe a strategy in a finite base-model sample does not prove that the base could never generate it. Trajectory analysis is therefore not decoration added to the metric—it addresses a different question.

A 2025 study concludes that, for the models and tasks it examined, much of the gain could be explained by reweighting trajectories already reachable from the base model. This is an important counterweight to strong “reasoning creation” claims, not a final answer for all scales and domains.

The full treatment of Best-of-N, self-consistency, and search is in [Module 10 “Search, verification, and test-time compute”](module-10.md).

---

### Step 9.4.5 — A minimum dashboard

![A minimum RLVR dashboard](assets/rl-for-llm/en/module-09/M9_diagnostics_EN.png)

A group-based RLVR dashboard needs several independent axes.

**Data and verification**

- every reward component separately;
- parse errors, timeouts, and exceptions;
- informative-group fraction;
- reward distributions by source and difficulty.

**Policy and optimization**

- average and position-wise entropy;
- KL to the declared reference, when computed;
- token- or sequence-ratio distribution;
- positive and negative clip fractions separately;
- rollout age and reuse count.

**Responses and quality**

- pass@1 and several pass@k points on held-out data;
- mean, median, and tail length;
- EOS, truncation, and parse-failure rates;
- string and semantic-strategy diversity;
- performance on control tasks outside the training domain.

**System**

- generated tokens per informative prompt;
- generation and optimization time;
- peak memory;
- stale or reused rollout fraction.

Define stopping criteria before training, such as no independent-quality improvement while proxy reward keeps rising, a sharp pass@k decline, or a growing checker-error rate.

---

## Lesson 9.5. From formulas to a reproducible run

Turn the preceding conditions into a reproducible run. A run record matters because it preserves the choices that determined rewards, group coefficients, and gradients. The small training example then shows why local improvement is not sufficient evidence of broader success.

### Step 9.5.1 — Data and verifier passport

Before the first update, record:

- prompt provenance and train/evaluation split rules;
- reference-answer provenance and license;
- verifier version;
- false-positive and false-negative rates on a manually reviewed sample;
- initial pass-rate distribution;
- answer format and chat template;
- stopping rules;
- length and execution limits;
- handling of ambiguous or damaged examples.

Code tasks additionally require container image, compiler, dependencies, network policy, time, memory, and test determinism. Formal mathematics needs kernel version and allowed axioms. Numerical tasks need canonicalization and tolerances.

Dynamic sampling must not erase the original distribution. Keep accepted and rejected prompts with rejection reasons.

---

### Step 9.5.2 — Rollout and update passport

A rollout batch should reproduce the loss without regenerating outputs.

Module 6 established the tokens, masks, and probability record. Group learning adds another essential structure: which responses belong together and which rewards produced their statistics. Regrouping the same responses can change their coefficients. Group identity and the precise normalization convention therefore belong to the training example, not just incidental metadata.

The record includes:

- prompt identifier and tokens;
- completion tokens and action mask;
- log-probabilities under the actual old policy;
- generator checkpoint/version;
- decoding parameters and random seed;
- reward components;
- parse result and stop reason;
- length before and after truncation;
- reference log-probabilities when needed;
- group identifier and completions per prompt.

The update passport adds:

- reward centering and scaling;
- `loss_type` and exact denominator;
- importance-ratio level;
- clip ranges;
- iterations per batch;
- KL and entropy coefficients;
- global batch size and gradient accumulation.

Generator and trainer distributions may differ even with the same checkpoint identifier because of library versions, precision, logit processors, and templates. Measure this mismatch rather than assuming it is zero.

---

### Step 9.5.3 — Resource profiles and the absence of a universal recipe

**Level A: CPU.** NumPy reference functions, verifier tests, synthetic groups, comparison of normalization, clipping, and sequence ratios. These local checks do not require a neural network. The CPU route also contains the small GRU training loop described below: it has its own model, but requires no pretrained LLM download.

**Level B: free Colab T4.** A roughly 0.6B model, LoRA, short completions, few generations, and an explicit smoke configuration. The goal is to observe the data path, not reproduce a large technical report.

**Level C: optional full run.** A larger model, long completions, a separate rollout engine, multiple GPUs, and reproducible infrastructure. Cost depends on completion length, group size, reuse, and verification expense.

Do not copy $G$, $\epsilon$, $\beta$, temperature, or learning rate as an “R1 recipe.” They interact with model size, reward scale, sequence length, batch size, optimizer, and loss normalization. Changing the denominator changes effective gradient scale.

---

### Step 9.5.4 — Practice: an autoregressive CPU loop and an optional real model

Separate Russian and English executable routes accompany the lecture.

The core route runs on CPU and uses a small but genuine autoregressive model: an embedding layer, a GRU, and a language-model head. Responses are generated token by token under a constrained grammar. The versioned verifier returns correctness, intermediate consistency, format status, and total reward separately. Each prompt receives a group sampled by the frozen old policy; the current policy is then optimized with a clipped group-relative objective. A separate reference policy is retained for teacher-forced KL diagnostics.

The loop is small in scale but realistic in contract. It exposes:

1. the actual behavior distribution after the allowed-token mask;
2. reproduction of stored old log-probabilities under the frozen old policy;
3. group advantages and the zero signal of an all-equal group;
4. multiple update epochs over one rollout batch;
5. entropy, clipping, length, KL, and informative-group rate;
6. fresh-generation evaluation rather than training loss alone;
7. held-out arithmetic prompts and prompt-level coverage.

The course’s fixed smoke profile produced:

| Metric | Before RL, train | After RL, train | Before RL, held-out | After RL, held-out |
|---|---:|---:|---:|---:|
| accuracy | 0.313 | 0.408 | 0.033 | 0.000 |
| mean reward | 0.341 | 0.448 | 0.037 | 0.000 |
| mean length | 7.03 | 8.38 | 6.92 | 9.24 |
| token entropy | 0.292 | 0.148 | 0.382 | 0.285 |
| teacher-forced KL to reference | 0.000 | 0.219 | 0.000 | 0.195 |
| pass@4 | 0.583 | 0.450 | 0.111 | 0.000 |

![What improved and what deteriorated at the same time](assets/rl-for-llm/en/module-09/M9_training_results_EN.png)

This is a **course-generated synthetic result**, not an external benchmark. Mean training accuracy rose, yet entropy and pass@4 fell, successes concentrated on fewer training prompts, and transfer to held-out combinations vanished. One mean reward is therefore not sufficient evidence of a good update. The small experiment does not claim that GRPO must collapse; it reproducibly demonstrates a failure pattern that a serious evaluation stack must be able to detect.

Higher mean accuracy with lower pass@4 is not a contradiction. More successes can concentrate on already accessible tasks while other prompts become nearly unsolvable. The probability of at least one success is nonlinear in a particular prompt's success rate, so mean pass@4 across heterogeneous prompts cannot be recovered from pooled accuracy alone. Prompt-level coverage is needed to interpret the two averages.

The KL row has a specific convention. Evaluation first samples a response from the policy under test. On each visited prefix, it computes exact categorical forward KL to the reference over grammar-allowed actions, averages those values within the response, and then averages responses. Teacher forcing means rescoring the selected sequence, not supplying a correct gold solution. This length-normalized diagnostic is not the sum defining sequence KL. The evaluated policies also visit different prefix distributions. The published numbers retain that protocol.

The optional real-model branch is disabled by default:

```python
RUN_REAL_MODEL = False
SMOKE_MODE = True
```

Its configuration skeleton targets `Qwen/Qwen3-0.6B`, LoRA, and `GRPOTrainer`. The snapshot pins TRL 1.9.2, Transformers 5.14.1, Datasets 5.0.1, PEFT 0.20.0, Accelerate 1.14.0, and bitsandbytes 0.50.0. **This branch was not executed in the release gate**: no runtime, memory use, or quality result is claimed for it. Before a Colab run, re-check the current CUDA image, model, package compatibility, and every `GRPOConfig` argument.

---

### Step 9.5.5 — Conclusions and boundaries

The main conclusions are:

1. RLVR is a feedback regime; GRPO is a family of policy updates. Neither defines the other.
2. A verifiable reward remains a proxy. Parsing, canonicalization, tests, and execution environment belong to the training contract.
3. Canonical GRPO replaces a learned critic with a group baseline but does not solve within-response credit assignment.
4. Old, current, and reference policies have different roles. The ratio denominator must match the actual behavior policy.
5. Group-standard-deviation and per-response-length normalization are not neutral: they allocate weight across prompts and tokens.
6. Dr. GRPO, DAPO, and GSPO modify different parts of the protocol; comparison requires the complete objective and reduction.
7. Entropy, response diversity, pass@k, and informative-group rate are distinct diagnostics.
8. Longer output does not prove better reasoning, and truncation is not identical to an incorrect answer.
9. Elicitation-versus-discovery claims require matched sampling budgets and analysis of the trajectories themselves.
10. A smoke run proves that the loop executes; it does not reproduce a large-scale paper result.

A group can now be read as a connected computation: checking produces rewards, within-prompt comparison produces coefficients, reduction assigns weights to tokens and responses, and current probabilities provide the differentiable route to parameters. Changing one operation changes a particular part of the algorithm. The next module retains checking and multiple candidates but removes the weight update: the problem becomes selecting an answer under an inference budget.

Continuation within this course:

- [Module 10 “Search, verification, and test-time compute”](module-10.md) develops Best-of-N, self-consistency, and search;
- [Module 12 “RL infrastructure for LLMs: memory, rollouts, and asynchrony”](module-12.md) covers distributed generation, policy versions, and rollout economics;
- [Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety”](module-13.md) builds the statistical and safety evaluation stack.

Extensions in the companion courses:

- *Modern LLMs*, [Module 14 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO”](/courses/modern-llms/module-14/) provides the broader post-training context;
- *Modern LLMs*, [Module 15 “Reasoning Models and Test-Time Compute”](/courses/modern-llms/module-15/) connects RLVR to reasoning-model architectures and inference budgets;
- *Information Theory for ML*, [Module 2 “Entropy”](/courses/information-theory-for-ml/module-02/) develops entropy formally;
- *Information Theory for ML*, [Module 3 “Cross-Entropy and KL Divergence”](/courses/information-theory-for-ml/module-03/) treats KL as a divergence between distributions;
- *Information Theory for ML*, [Module 8 “Maximum entropy, exponential models, and KL regularization”](/courses/information-theory-for-ml/module-08/) gives the variational view of KL-regularized policy updates.

These references deepen selected topics, but Module 9 remains self-contained: every definition and equation required by its exercises and practice is introduced here.

## Sources and reading route

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — the original GRPO presentation and token-level clipped objective.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — R1-Zero, rule-based rewards, and the final multi-stage pipeline.
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) — the normalization, response-length, and question-difficulty analysis behind Dr. GRPO.
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — Clip-Higher, dynamic sampling, token-level reduction, and overlong reward shaping.
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) — a response-level unit for the ratio and clipping decision.
- [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) — empirical and mechanistic entropy analysis in the authors’ studied runs.

- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939) — a separate study of updates at high-entropy positions, not the covariance criterion of the preceding work.
- [Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization](https://arxiv.org/abs/2605.11491) — a 2026 preprint on balancing entropy changes; preliminary evidence for a particular method and experimental suite.
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) — evidence on eliciting practically reachable trajectories; its conclusions are not universal across models and regimes.
- [TRL GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer) — the current engineering interface, not a primary source of mathematical guarantees.

The status of the course experiments is stated alongside their results: the published CPU numbers belong to a fixed teaching protocol, while the optional real-model branch is not presented as an executed experiment.
