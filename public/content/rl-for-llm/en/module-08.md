# Module 8. RLHF and direct preference optimization

> **Material version:** 2026.20  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Primary practice:** level A — a reproducible CPU workflow; level B — an optional open-model run with Low-Rank Adaptation (LoRA) on a suitable accelerator  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 9–11 hours excluding optional model training

Module 7 “Reward models and preference data” (`rl_llm.module_07_reward_models`) took us from pairwise labels to reward models. The next problem is operational: **how do we change a policy so that preferred responses become more likely without discarding useful behavior from the starting model or learning to exploit defects in the feedback signal?**

The classical answer is reinforcement learning from human feedback (RLHF): train a reward model, sample fresh responses, and optimize expected reward while limiting deviation from a reference policy. A more direct answer is direct preference optimization (DPO): convert fixed preference pairs into a policy loss without fitting an explicit reward model or collecting new rollouts inside that stage.

The practical question of the module is:

> **How should we choose between RLHF and direct preference optimization once data, compute, hidden assumptions, and observable failure modes are made explicit?**

After completing the module, you will be able to:

1. write the RLHF objective regularized by Kullback–Leibler (KL) divergence and explain the roles of policy, reference, reward model, and critic;
2. separate four logical roles from four mandatory full LLM copies;
3. derive the exact exponentially tilted optimum;
4. derive DPO from that optimum and the Bradley–Terry model;
5. implement the DPO loss, gradient weight, and label smoothing;
6. explain why a growing relative margin does not guarantee a growing chosen-response probability;
7. audit lengths, masks, chat templates, stop reasons, and cached reference log-probabilities;
8. distinguish DPO, IPO, SimPO, KTO, and ORPO by their data contracts;
9. choose among fixed pairs, iterative preference collection, and online RL;
10. run a synthetic CPU experiment and prepare a DPO smoke test on an open 0.6B-class model.

The English practical route is the executable practice; dated claims, primary sources, and inference limits are recorded in `Module_8_Sources.md`.

---

> **Practice route.** Graded tasks, worked solutions revealed after an attempt, and executable practice follow the lecture. The explanation remains self-contained.

## Lesson 8.1. From preferences to a KL-regularized policy

> **Lesson 8.1. From preferences to a KL-regularized policy**
>
> 1. What classical RLHF optimizes
> 2. Four logical roles, not four mandatory copies
> 3. Turning sequence KL into token rewards
> 4. Exact optimum: exponential tilting of the reference
> 5. A numerical reward–deviation frontier
>
> then 5 assessment steps

### Step 8.1.1 — What classical RLHF optimizes

Let $x$ be a prompt, $y$ a full completion, $\pi_\theta(y\mid x)$ the trainable policy, $r_\phi(x,y)$ a reward model, and $\pi_{\mathrm{ref}}(y\mid x)$ a frozen reference policy. A standard objective is

$$
\max_\theta\;
\mathbb E_{x\sim\mathcal D,\;y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)]
-
\beta\,
\mathbb E_{x\sim\mathcal D}
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\middle\|
\pi_{\mathrm{ref}}(\cdot\mid x)
\right),
$$

with $\beta>0$ pricing departure from the reference.

The reward term moves mass toward responses favored by the reward model. The KL term charges for redistribution relative to the reference. With a fixed reward, a larger $\beta$ makes movement more expensive; a smaller value allows the reward to dominate more strongly.

A related constrained form is

$$
\max_\pi\;\mathbb E_\pi[r]
\quad\text{subject to}\quad
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})\le\delta.
$$

In an ideal convex problem, the penalty is connected to the constraint through a Lagrange multiplier. In finite neural training, the selected $\beta$ does not predetermine the realized KL because parameterization, optimization noise, and training duration matter.

**Boundary of the result.** KL controls departure in one geometry. It does not validate the reward model, prevent every harmful change, or preserve every capability.

---

### Step 8.1.2 — Four logical roles, not four mandatory copies

A Proximal Policy Optimization (PPO)-like RLHF loop usually separates:

1. a trainable policy $\pi_\theta$;
2. a frozen reference policy $\pi_{\mathrm{ref}}$;
3. a reward model $r_\phi$;
4. a critic $V_\psi$ used to estimate advantages.

![Logical roles in RLHF](assets/rl-for-llm/en/module-08/M8_rlhf_roles_EN.png)

These are mathematical and computational roles. An implementation need not retain four independent full-size parameter sets. It may share a policy/value backbone, use adapters, quantize frozen components, cache reference log-probabilities, offload models to CPU, or separate generation and optimization across processes.

The precise statement is:

> RLHF contains several logical models and computational roles; physical placement and parameter sharing are implementation choices.

Fresh generation, reward and reference forward passes, value estimation, rollout storage, and policy backpropagation remain costly even after memory engineering. DPO shortens this loop, but it does not eliminate the cost of obtaining reliable preference data.

---

### Step 8.1.3 — Turning sequence KL into token rewards

Autoregressive factorization gives

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\sum_{t=1}^{T}
\left[
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
\right].
$$

A token-level RL implementation can therefore add

$$
r_t^{\mathrm{KL}}
=
-\beta
\left[
\log\pi_\theta(y_t\mid s_t)
-
\log\pi_{\mathrm{ref}}(y_t\mid s_t)
\right]
$$

to every active response token and attach the terminal score $R(x,y)$ to the final active token. The sum is exactly

$$
R(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

A sampled log-ratio can be negative. It is not a negative KL divergence: its expectation under the current policy equals forward KL. Module 6 “The LLM as a policy: tokens, log-probabilities, and KL” (`rl_llm.module_06_llm_policy`) develops this distinction in detail.

The response mask and stop reason are part of the objective. Prompt tokens, padding, and environment observations must not accidentally receive policy reward.

---

### Step 8.1.4 — Exact optimum: exponential tilting of the reference

Fix one prompt and a finite response space. Optimize the full distribution:

$$
\max_\pi
\left\{
\sum_y\pi(y)r(y)
-
\beta\sum_y\pi(y)
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
\right\}.
$$

Introducing a multiplier for normalization and differentiating gives

$$
\boxed{
\pi^*(y)
=
\frac{
\pi_{\mathrm{ref}}(y)\exp(r(y)/\beta)
}{Z}
},
$$

where

$$
Z
=
\sum_{y'}
\pi_{\mathrm{ref}}(y')
\exp(r(y')/\beta).
$$

![Exponential tilting of the reference policy](assets/rl-for-llm/en/module-08/M8_reference_tilt_EN.png)

Reward reweights mass already available under the reference. If $\pi_{\mathrm{ref}}(y)=0$, then $\pi^*(y)=0$ for finite reward. Adding a constant to every reward leaves the optimum unchanged because the common factor cancels in $Z$.

As $\beta\to\infty$, the optimum approaches the reference. As $\beta\to0^+$, it concentrates on reward maxima inside reference support.

> **Mathematical extension:** *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” (`it_ml.module_08_maxent_kl_policy`) develops variational forms, partition functions, and maximum-entropy connections.

---

### Step 8.1.5 — A numerical reward–deviation frontier

Take

$$
\pi_{\mathrm{ref}}=(0.55,0.30,0.15),
\qquad
r=(0,1,-0.5).
$$

The exact solution is:

| $\beta$ | $\pi^*$ | $\mathbb E[r]$ | $D_{\mathrm{KL}}(\pi^*\|\pi_{\mathrm{ref}})$ |
|---:|---|---:|---:|
| 4.0 | $(0.515,0.361,0.124)$ | 0.299 | 0.009 |
| 2.0 | $(0.474,0.426,0.101)$ | 0.376 | 0.038 |
| 1.0 | $(0.378,0.560,0.062)$ | 0.529 | 0.153 |
| 0.5 | $(0.195,0.786,0.020)$ | 0.776 | 0.514 |
| 0.25 | $(0.032,0.966,0.001)$ | 0.966 | 1.033 |

![Reward–KL frontier](assets/rl-for-llm/en/module-08/M8_reward_kl_frontier_EN.png)

The values are reproduced by `reward_kl_frontier` in `module_8_reference.py`. Lower $\beta$ raises reward and KL for this exact optimum. It does not imply monotonic behavior for every neural training step.

A useful report therefore presents a frontier, not one proxy-reward number. Reward without deviation and deviation without independent quality are both incomplete.

#### Constraints: utility and cost are different signals

KL keeps a policy near its reference, but it does not itself encode a safety requirement. Let $R$ denote response utility and let $C_j$ denote separate costs, such as the probability of a prohibited action, an access-policy violation, or a dangerous side effect. A natural constrained objective is

$$
\max_\theta\;
\mathbb E_{\pi_\theta}[R]
\quad\text{subject to}\quad
\mathbb E_{\pi_\theta}[C_j]\le d_j,
\qquad j=1,\ldots,m.
$$

Its Lagrangian is

$$
\mathcal L(\theta,\lambda)
=
\mathbb E[R]
-
\sum_{j=1}^{m}
\lambda_j\bigl(\mathbb E[C_j]-d_j\bigr),
\qquad
\lambda_j\ge0.
$$

The policy maximizes $\mathcal L$ under the current penalties, while a dual variable rises when its constraint is violated:

$$
\lambda_j
\leftarrow
\left[
\lambda_j
+
\eta_\lambda
\bigl(\widehat{\mathbb E}[C_j]-d_j\bigr)
\right]_+.
$$

A **reward model** and a **cost model** are therefore different statistical components. High utility should not automatically compensate for a critical violation. A fixed sum $R-\alpha C$ is a scalarization with a preselected trade-off; an explicit constraint states an admissible cost budget and allows the multiplier to adapt to the observed violation.

Consider a two-action example. The safe action has reward $0.6$ and cost $0$, while the risky action has reward $1.0$ and cost $1$. If the policy selects the risky action with probability $q$, then

$$
\mathbb E[C]=q,
\qquad
\mathbb E[R]=0.6+0.4q.
$$

Under the budget $d=0.25$, the optimum is

$$
q^*=0.25,
\qquad
\mathbb E[R]=0.7.
$$

The functions `two_action_constrained_optimum` and `dual_ascent_step` reproduce the calculation.

**Where the guarantee ends.** An average constraint $\mathbb E[C]\le d$ does not guarantee safety on every prompt. Critical actions require additional mechanisms such as hard filters, safe action sets, refusal or abort paths, and scenario-level checks. A learned cost model can also be wrong and can itself be optimized against. Training-time constraints and the release controls in Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” (`rl_llm.module_13_evaluation`) therefore complement rather than replace one another.

Constrained Policy Optimization (CPO) is a classical policy-optimization line for expected-cost constraints. Safe RLHF adapts the reward-versus-cost separation to LLM alignment. These sources provide mathematical and empirical grounding; neither turns an average constraint into an absolute safety guarantee.

---

## Lesson 8.2. DPO: derivation, loss, and gradient

> **Lesson 8.2. DPO: derivation, loss, and gradient**
>
> 1. Preference pairs as probabilistic observations
> 2. Implicit reward and cancellation of the normalizer
> 3. The DPO loss
> 4. Gradient weight and the role of β
> 5. DPO assumptions and the boundary with classical offline RL
>
> then 3 assessment steps

### Step 8.2.1 — Preference pairs as probabilistic observations

For one prompt, let $y^+$ be chosen and $y^-$ rejected. The Bradley–Terry (BT) model writes

$$
P(y^+\succ y^-\mid x)
=
\sigma\left(r(x,y^+)-r(x,y^-)\right).
$$

Module 7 “Reward models and preference data” (`rl_llm.module_07_reward_models`) used this likelihood to fit an explicit reward model. DPO instead expresses reward through a policy and substitutes that expression directly into the preference likelihood.

The label remains conditional on a prompt, two candidates, a rubric, and a protocol. It does not turn the chosen response into globally correct ground truth or identify an absolute reward scale across prompts.

A sound dataset preserves prompt identity, both full completions, candidate provenance, rubric version, tie/abstention policy, symmetric preprocessing, and prompt-level train/evaluation splits.

---

### Step 8.2.2 — Implicit reward and cancellation of the normalizer

The exact optimum implies

$$
r(x,y)
=
\beta
\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x).
$$

For two responses to one prompt, the prompt-dependent $\beta\log Z(x)$ cancels:

$$
r(x,y^+)-r(x,y^-)
=
\beta
\left[
\log\frac{\pi^*(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}
-
\log\frac{\pi^*(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right].
$$

Replacing the unknown optimum with a trainable policy produces a preference logit. No explicit reward-model training is required during this stage: the policy-to-reference log-ratio acts as an implicit reward.

![From KL-regularized reward to DPO](assets/rl-for-llm/en/module-08/M8_dpo_derivation_EN.png)

The cancellation is exact inside the model. It does not prove that human decisions are generated by one scalar reward and logistic noise.

---

### Step 8.2.3 — The DPO loss

Define the sequence margin

$$
h_\theta
=
\left[
\log\pi_\theta(y^+\mid x)
-
\log\pi_{\mathrm{ref}}(y^+\mid x)
\right]
-
\left[
\log\pi_\theta(y^-\mid x)
-
\log\pi_{\mathrm{ref}}(y^-\mid x)
\right].
$$

Then $z_\theta=\beta h_\theta$ and

$$
\boxed{
\mathcal L_{\mathrm{DPO}}
=-\log\sigma(z_\theta)
=
\operatorname{softplus}(-z_\theta)
}.
$$

The policy is asked to improve the chosen response relative to the reference more than it improves the rejected response. DPO does not independently maximize raw chosen log-probability.

Sequence log-probabilities sum over response tokens. Prompt and padding tokens are masked. Both completions must use the same tokenizer, chat template, and truncation rule.

---

### Step 8.2.4 — Gradient weight and the role of β

The derivative with respect to the margin is

$$
\frac{\partial\mathcal L}{\partial h_\theta}
=
-\beta\sigma(-\beta h_\theta),
$$

so the positive pair-weight magnitude is

$$
w(h)=\beta\sigma(-\beta h).
$$

![DPO gradient weight](assets/rl-for-llm/en/module-08/M8_dpo_gradient_EN.png)

Negative-margin pairs receive strong pressure. At a large positive margin the weight decays. Because $\beta$ both scales the logit and multiplies the derivative, “larger $\beta$ always means a stronger update” is not a valid statement without the current margin, learning rate, and exact implementation.

In the variational objective, $\beta$ prices KL. In the DPO code path it scales a classifier logit. The derivation connects those roles, but realized KL still depends on data, optimization, and training duration.

---

### Step 8.2.5 — DPO assumptions and the boundary with classical offline RL

DPO is easiest to interpret when:

- both completions share one prompt;
- BT is a reasonable preference model;
- the reference is fixed and evaluated with the same template;
- both completions lie in reference support;
- labels represent the intended criterion;
- the fixed data cover future usage reasonably well.

DPO uses no explicit Q-function, Bellman target, or bootstrapping. Calling it “exactly classical offline RL” is therefore too strong. A better statement is:

> DPO is direct policy optimization on fixed preference pairs, derived from a KL-regularized problem; it is related to offline learning but not identical to value-based offline RL.

If BT is misspecified, DPO remains a well-defined pairwise loss. Its interpretation changes from exact implicit-reward recovery to a discriminative approximation.

---

## Lesson 8.3. Where the simple picture breaks

> **Lesson 8.3. Where the simple picture breaks**
>
> 1. Relative margin is not absolute probability
> 2. An exact likelihood-displacement toy
> 3. Length, masks, and chat templates
> 4. Noisy pairs and conservative DPO (cDPO)
> 5. Fixed pairs grow stale
>
> then 4 assessment steps

### Step 8.3.1 — Relative margin is not absolute probability

A growing $h_\theta$ does not imply that $\log\pi_\theta(y^+\mid x)$ grows. The chosen probability may rise while the rejected falls, both may rise at different rates, or both may fall while the rejected falls faster.

The third regime is invisible if a dashboard contains only DPO loss and pair accuracy. Softmax normalization couples the observed pair to all other sequences; an unlabelled completion may absorb the probability mass.

At minimum, log chosen and rejected sequence probabilities, their separate reference ratios, margin, gradient weight, response lengths, KL, and independent generation quality.

---

### Step 8.3.2 — An exact likelihood-displacement toy

Consider three completions with logits

$$
z(t)=(-t,-3t,4t).
$$

The first is chosen, the second rejected, and the third absent from the pair. Their pairwise log-margin is

$$
\log p_1(t)-\log p_2(t)=2t,
$$

which grows strictly, while probabilities are:

| $t$ | $p_1$ | $p_2$ | $p_3$ | pair margin |
|---:|---:|---:|---:|---:|
| 0 | 0.3333 | 0.3333 | 0.3333 | 0.0 |
| 0.25 | 0.1962 | 0.1190 | 0.6848 | 0.5 |
| 0.5 | 0.0738 | 0.0271 | 0.8991 | 1.0 |
| 1.0 | 0.0067 | 0.0009 | 0.9924 | 2.0 |

![Likelihood displacement](assets/rl-for-llm/en/module-08/M8_likelihood_displacement_EN.png)

The pair is increasingly well separated, yet the chosen probability falls. The toy does not claim that every DPO run deteriorates. It proves the narrower point that pair margin does not determine absolute chosen probability.

---

### Step 8.3.3 — Length, masks, and chat templates

Sequence log-probability is a sum of negative token contributions, so length affects scale directly. If chosen responses are longer, length may become a label-correlated shortcut. Dividing by length is not a universal repair: it changes the objective and can introduce a short-response bias.

Audit chosen/rejected length distributions, symmetric truncation, truncation rate, EOS and stop reasons, response-only masks, identical chat templates, tokenizer/reference versions, and hidden winner metadata.

Cached reference log-probabilities become stale after changing tokenization, template, truncation, or checkpoint. Visually identical text is not necessarily the same action sequence.

![Sequence-level DPO pair contract](assets/rl-for-llm/en/module-08/M8_sequence_contract_EN.png)

Tokenizer revision, chat template, truncation rule, reference checkpoint, and completion mask are part of the pair's mathematical contract.

> **Engineering extension:** *Modern LLMs*, Module 11 “Supervised Fine-Tuning and Data Work” (`modern_llms.module_11_sft`) covers dialogue-data and chat-template engineering in more depth.

---

### Step 8.3.4 — Noisy pairs and conservative DPO (cDPO)

Conservative DPO (cDPO) smooths the binary target. With a possible label-flip rate, for $\varepsilon\in[0,1/2)$,

$$
\mathcal L_{\mathrm{cDPO}}
=
(1-\varepsilon)\operatorname{softplus}(-z)
+
\varepsilon\operatorname{softplus}(z).
$$

At $\varepsilon=0$ this is ordinary DPO. Positive smoothing avoids treating every observed label as perfectly certain. In this form, cDPO is a practical label-smoothing heuristic; it should not be conflated with Robust DPO, whose correction is derived under a specified symmetric label-flip model and retains that model's assumptions.

Neither variant repairs a systematically wrong rubric, position leakage, generator bias, or arbitrary distribution shift. Monitor held-out pair accuracy, chosen/rejected log-probabilities, margin distribution, KL, lengths, domain slices, and independent fresh-generation evaluations. Training loss can keep improving after useful quality plateaus.

---

### Step 8.3.5 — Fixed pairs grow stale

DPO commonly trains on responses produced by an earlier policy. After updating the model, new responses may occupy a different region. The original pairs then cover the current policy less well.

Three regimes are useful to distinguish:

1. one fixed dataset — easiest to reproduce, but never refreshed;
2. iterative preference optimization — regenerate candidates and recollect pairs;
3. online RL — score current-policy rollouts and update directly.

Iterative collection reduces policy–data mismatch, but requires versioning of generators, rubrics, judges, and rounds. It cannot repair a systematically wrong feedback source.

Store `policy_version`, `reference_version`, decoding settings, date, label source, and round identifier for every collection cycle.

---

## Lesson 8.4. Variants of direct preference optimization

> **Lesson 8.4. Variants of direct preference optimization**
>
> 1. IPO: a finite target margin
> 2. SimPO: mean log-probability and a target gap
> 3. KTO: unpaired binary feedback
> 4. ORPO: supervised fine-tuning and odds ratio in one objective
> 5. Method map: data first, name second
>
> then 3 assessment steps

### Step 8.4.1 — IPO: a finite target margin

Identity Preference Optimization (IPO) retains a pairwise reference-corrected margin but gives it a finite target. In the convention used here,

$$
\mathcal L_{\mathrm{IPO}}
=
\left(
 h_\theta-
 \frac{1}{2\tau}
\right)^2.
$$

Ordinary DPO keeps decreasing as a positive margin grows. IPO penalizes both undershooting and overshooting its target. This is useful when unbounded margin growth is not the desired behavior.

A finite target does not repair bad pairs, stale templates, or weak coverage. Its $\tau$ is not numerically comparable to DPO's $\beta$ without mapping the exact objectives.

---

### Step 8.4.2 — SimPO: mean log-probability and a target gap

Simple Preference Optimization (SimPO) uses a reference-free response score

$$
r_{\mathrm{SimPO}}(x,y)
=
\frac{\beta}{|y|}
\log\pi_\theta(y\mid x),
$$

and forms a pair logit with a target margin $\gamma$:

$$
z
=
r_{\mathrm{SimPO}}(x,y^+)
-
r_{\mathrm{SimPO}}(x,y^-)
-
\gamma.
$$

It avoids a separate reference forward pass and explicitly normalizes by length. The trade-off is a different mathematical contract: no reference anchor remains, and mean log-probability has its own biases.

SimPO and DPO should be compared under matched data, compute, and independent evaluation. Length normalization is not a proof that length effects are solved.

---

### Step 8.4.3 — KTO: unpaired binary feedback

Kahneman–Tversky Optimization (KTO) targets datasets where an individual completion is marked desirable or undesirable, without an explicit paired completion for the same prompt.

Its appeal is a different data contract: collecting isolated binary judgments may be cheaper than forming balanced pairs. The key question is whether those labels have a consistent semantic meaning.

Do not manufacture DPO pairs by joining unrelated prompts. Audit class balance, label provenance, domain balance, and the baseline against which desirability is interpreted.

---

### Step 8.4.4 — ORPO: supervised fine-tuning and odds ratio in one objective

Odds Ratio Preference Optimization (ORPO) combines a supervised fine-tuning (SFT) term on the chosen completion with a pairwise odds-ratio term:

$$
\mathcal L_{\mathrm{ORPO}}
=
\mathcal L_{\mathrm{SFT}}(y^+)
+
\lambda\mathcal L_{\mathrm{odds}}(y^+,y^-).
$$

No separate reference model is required during training. The SFT component supports chosen likelihood, while the pair component separates chosen from rejected.

ORPO is not literally DPO with one model removed; the loss, scale, and anchoring mechanism differ. Report the chosen-response negative log-likelihood (NLL) and the pairwise preference loss separately so one cannot hide degradation in the other.

---

### Step 8.4.5 — Method map: data first, name second

![Map of direct preference methods](assets/rl-for-llm/en/module-08/M8_method_map_EN.png)

| Method | Primary data | Reference in loss | Characteristic mechanism |
|---|---|---|---|
| DPO | same-prompt pairs | yes | logistic loss on relative log-ratios |
| cDPO | possibly noisy pairs | yes | label smoothing |
| IPO | pairs | yes | finite margin target |
| SimPO | pairs | no | mean log-probability and target gap |
| KTO | unpaired binary labels | usually yes | utility of desirable/undesirable examples |
| ORPO | pairs plus SFT anchor | no | NLL plus odds ratio |

No method is uniformly best. Ask whether the data are paired, whether an explicit reference anchor is needed, how lengths differ, how much label noise is expected, whether data will be refreshed, and whether independent generation evaluation is available.

> **Survey extension:** *Modern LLMs*, Module 12 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” (`modern_llms.module_12_preference_optimization`) provides a broader post-training map. This module focuses on the RL interpretation, derivations, and diagnostics.

---

## Lesson 8.5. From equations to a reproducible run

> **Lesson 8.5. From equations to a reproducible run**
>
> 1. Minimal DPO data flow
> 2. Practice A: exact CPU experiments
> 3. Practice B: DPO on Qwen3-0.6B
> 4. What to measure after training
> 5. Decision rule: DPO, RLHF, or online RL
>
> then 6 assessment steps

### Step 8.5.1 — Minimal DPO data flow

A reproducible pipeline contains:

1. a prompt and two completions;
2. one chat template;
3. tokenization with symmetric truncation;
4. response-token masks;
5. current-policy sequence log-probabilities;
6. reference sequence log-probabilities;
7. the loss plus complete diagnostics.

Retain prompt identity, candidate provenance, generator and reference versions, rubric, label source, decoding settings, and stop reason. Split by prompt, not by pair row.

Libraries may optimize reference storage when the initial policy and reference coincide and LoRA is used. The logical reference nevertheless remains part of the DPO definition.

---

### Step 8.5.2 — Practice A: exact CPU experiments

Two files form the required CPU route. `module_8_reference.py` covers exponential tilting, the reward–KL frontier, token-level KL shaping, DPO loss and weights, cDPO, IPO, SimPO, ORPO, and the exact likelihood-displacement toy. `module_8_training.py` adds a small but complete autoregressive system with a frozen reference, causal shifting, completion masks, sequence log-probabilities, exact teacher-forced token KL, and free greedy generation.

This practice is optional for formal course completion but essential for genuine understanding: it separates mathematical mistakes from failures in a large software stack.

![Moderate DPO and overoptimization](assets/rl-for-llm/en/module-08/M8_toy_dpo_training_EN.png)

Under the pinned `smoke` protocol, both DPO regimes reach pair-ranking accuracy 1.0. Moderate training retains exact greedy-generation matches on all eight templates, whereas the high-pressure regime retains only 0.125. Mean chosen-response log-probability changes from roughly $-0.015$ under moderate training to $-1.765$ under high pressure. This is a synthetic course result, not an external benchmark; its purpose is to show why pair accuracy and DPO loss are insufficient without absolute likelihoods and free generation.

```bash
python -B modules/module-08/test_module_8_reference.py
python -B modules/module-08/test_module_8_training.py
python -B modules/module-08/module_8_training.py
```

The graded practice provides tasks and worked solutions revealed after an attempt; the executable practice provides the connected computational route.

---

### Step 8.5.3 — Practice B: DPO on Qwen3-0.6B

- a synthetic CPU path without model downloads;
- a real path using `Qwen/Qwen3-0.6B`, `trl-lib/ultrafeedback_binarized`, TRL `DPOTrainer`, and LoRA.

`RUN_REAL_MODEL=False` by default, so the release route downloads nothing and needs no accelerator. The real branch is a current engineering template for a suitable GPU environment; memory, runtime, and stability depend on the runtime image, sequence length, batch size, numerical precision, and library versions. The course does not promise that every free T4 profile will fit the selected configuration.

The optional branch snapshot is pinned to TRL 1.9.2, Transformers 5.14.1, PEFT 0.20.0, Accelerate 1.14.0, and Datasets 5.0.1 because APIs are part of the experiment.

**Release boundary:** the Qwen3/TRL branch was not executed for this edition. The course therefore makes no measured claim about Qwen3 runtime, GPU memory, convergence, or trained-model quality. The NumPy mathematics and the complete autoregressive CPU route with a small GRU policy were executed.

---

### Step 8.5.4 — What to measure after training

A minimal report includes DPO loss, pair accuracy, chosen and rejected log-probabilities, margin and gradient weight, KL under a consistent protocol, lengths and stop reasons, domain/generator slices, fresh-generation evaluation, and checks for retained base capabilities.

Use at least:

1. held-out pairs from the same source;
2. new prompts from the same domain;
3. shifted prompts or responses from the updated policy.

High pair accuracy does not prove strong generation. The model may learn style shortcuts or improve relative ordering while chosen absolute likelihood falls.

---

### Step 8.5.5 — Decision rule: DPO, RLHF, or online RL

A useful sequence is:

1. Is the feedback reliable? If not, repair the rubric and data first.
2. Are there fixed same-prompt pairs? DPO and its variants provide a strong baseline.
3. Must data follow the current policy? Use iterative collection or online RL.
4. Is reward programmatically verifiable? This naturally leads to reinforcement learning with verifiable rewards (RLVR), covered in Module 9 “RLVR and GRPO: verifiable rewards, group-relative estimates, and stability” (`rl_llm.module_09_rlvr_grpo`).
5. Is open-ended reward and token-level credit assignment required? PPO or REINFORCE Leave-One-Out (RLOO) loops remain relevant despite their cost.
6. Is a simple one-stage path required? Consider ORPO or SimPO with their own assumptions.

The final map is deliberately less dramatic than “DPO replaced RLHF”:

> DPO replaces a particular expensive loop when fixed pairs and its assumptions fit the task. Online RL remains necessary when data must track the current policy or interaction and verifiable reward are part of the objective.

---

## Module summary

- RLHF maximizes reward under a KL price for departing from a reference.
- Four RLHF roles do not imply four mandatory full model copies.
- The exact KL-regularized optimum is an exponential tilt of the reference.
- DPO follows by inserting implicit reward into Bradley–Terry and cancelling prompt normalization.
- DPO controls a relative margin; absolute likelihoods must be monitored separately.
- Length, masks, templates, truncation, and reference version belong to the mathematical contract.
- IPO, SimPO, KTO, and ORPO solve different problems under different data assumptions.
- Fixed pairs grow stale as the policy changes; iterative collection and online RL remain distinct tools.

## Compact glossary

- **RLHF** — policy training from human feedback through a reward and RL loop.
- **DPO** — a direct pairwise policy loss relative to a reference.
- **KL anchor** — a penalty or constraint on policy departure from the reference.
- **Implicit reward** — $\beta\log(\pi/\pi_{\mathrm{ref}})$ up to a prompt-dependent offset.
- **cDPO** — DPO with label smoothing.
- **IPO** — a pairwise objective with a finite desired margin.
- **SimPO** — a reference-free pairwise objective based on mean sequence log-probability.
- **KTO** — an objective for unpaired desirable and undesirable examples.
- **ORPO** — an SFT term combined with a pairwise odds-ratio term.
- **Likelihood displacement** — a regime where pair margin grows while chosen probability may fall.

## Sources and further-study route

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) — the canonical LLM sequence SFT → reward model → PPO; it does not prescribe one mandatory memory layout.
- [Rafailov et al. (2023), *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) — the core DPO derivation from a KL-regularized objective and a pairwise preference model.
- [Azar et al. (2023), *A General Theoretical Paradigm to Understand Learning from Human Preferences*](https://arxiv.org/abs/2310.12036) — analysis of pairwise objectives and the motivation for IPO.
- [Meng, Xia, and Chen (2024), *SimPO*](https://arxiv.org/abs/2405.14734), [Ethayarajh et al. (2024), *KTO*](https://arxiv.org/abs/2402.01306), and [Hong, Lee, and Thorne (2024), *ORPO*](https://arxiv.org/abs/2403.07691) — primary sources for alternative direct-optimization contracts.
- [Chowdhury et al. (2024), *Provably Robust DPO*](https://arxiv.org/abs/2403.00409) — a theoretical correction under a specified noise model, not a synonym for ordinary cDPO label smoothing.
- [Razin et al. (2024), *Unintentional Unalignment: Likelihood Displacement in DPO*](https://arxiv.org/abs/2410.08847) — motivation for tracking chosen and rejected absolute likelihoods separately.
- [Current TRL DPOTrainer documentation](https://huggingface.co/docs/trl/dpo_trainer) — a dated implementation interface, not a mathematical guarantee.

The line-by-line registry of claims, limitations, and reproduced values is in `Module_8_Sources.md`. For the variational derivation, use *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” (`it_ml.module_08_maxent_kl_policy`); for SFT data engineering, use *Modern LLMs*, Module 11 “Supervised Fine-Tuning and Data Work” (`modern_llms.module_11_sft`); for the broader post-training map, use *Modern LLMs*, Module 12 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” (`modern_llms.module_12_preference_optimization`).
