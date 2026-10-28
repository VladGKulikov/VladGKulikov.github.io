# Module 8. RLHF and direct preference optimization

> **Material version:** 2026.20  
> **Factual snapshot:** 2026-08-05  
> **Explanatory-text revision:** 2026-10-27  
> **Language:** EN  
> **Primary practice:** level A — a reproducible CPU workflow; level B — an optional open-model run with Low-Rank Adaptation (LoRA) on a suitable accelerator  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 9–11 hours excluding optional model training

Module 7 “Reward models and preference data” ([open module](module-07.md)) took us from pairwise labels to reward models. The next problem is operational: **how do we change a policy so that preferred responses become more likely without discarding useful behavior from the starting model or learning to exploit defects in the feedback signal?**

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
10. distinguish reward and cost models, formulate expected-cost constraints, and interpret the multiplier update;
11. run a synthetic CPU experiment and prepare a DPO smoke test on an open 0.6B-class model.

The English executable practice accompanies the lecture. Primary works and the limits of their use are identified in the [sources section](#sources-and-further-study-route).

---

> **Practice route.** [Exercises and solutions](/courses/rl-for-llm/module-08/exercises/), [assessment questions](/courses/rl-for-llm/module-08/tests/), and [executable practice](/courses/rl-for-llm/module-08/notebooks/) are separate resources. The lecture remains self-contained.

## Lesson 8.1. From preferences to a KL-regularized policy

Module 7 turned comparisons into a numerical scoring function. We now ask how that score should reshape a distribution over responses. An explicit reward gives us a useful starting point: a finite optimization problem that we can solve exactly. Its solution will then let us express preference learning without a separately trained reward model.

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

A sampled log-ratio can be negative. It is not a negative KL divergence: its expectation under the current policy equals forward KL. Module 6 “The LLM as a policy: tokens, log-probabilities, and KL” ([open module](module-06.md)) develops this distinction in detail.

The response mask and stop reason are part of the objective. Prompt tokens, padding, and environment observations must not accidentally receive policy reward.

The equation describes an objective, not yet the lifetime of the tensors in an optimizer. At the start of an iteration, a behavior snapshot $\pi_{\mathrm{old}}$ is fixed. Its sampled completions receive reward-model scores and token-level penalties relative to the reference. Advantages are then constructed from these quantities. In a conventional PPO update with several epochs over one batch, those rewards and advantages remain fixed numerical inputs. The trainable policy is rescored for the surrogate loss, while the denominator still refers to the policy that collected the batch.

At collection time, the current and old policies coincide; after an update they need not. The reference has a different job and does not become the behavior snapshot merely because both models were once initialized from the same checkpoint. Reusing the frozen batch is a local optimization procedure, not an exact reevaluation of the original objective at arbitrarily changed parameters. A new iteration refreshes the data and learning signal. An implementation with a separate differentiable KL loss must specify its derivative path instead. Module 6 explains why an identity between scalar values alone cannot establish that two gradient implementations optimize the same objective.

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

For now, the optimization variables are the probabilities themselves, not the parameters of a particular neural network. Assume $\beta>0$ and finite rewards, and restrict attention to reference support: putting positive mass outside it would incur infinite KL. Form weights $\pi_{\mathrm{ref}}(y)\exp(r(y)/\beta)$, let $Z$ be their sum, and normalize them to obtain a distribution $\widetilde\pi$. The reference, rewards, and $\beta$ determine $Z$; the trial policy $\pi$ does not.

Writing the original objective as $\mathcal J(\pi)$, we can combine its reward and penalty terms:

$$
\begin{aligned}
\mathcal J(\pi)
&=\beta\sum_y\pi(y)
  \log\frac{\pi_{\mathrm{ref}}(y)\exp(r(y)/\beta)}{\pi(y)}\\
&=\beta\log Z-\beta D_{\mathrm{KL}}(\pi\|\widetilde\pi).
\end{aligned}
$$

The first term is constant with respect to our choice of policy. The second is nonpositive and vanishes exactly at $\pi=\widetilde\pi$. This identifies a global maximum of the finite problem, not merely a stationary candidate:

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

This is an exact result for an unrestricted finite distribution. A parameterized LLM can express only a particular family of distributions and is fitted with finite data and finite optimization. The formula identifies the target of the idealized problem; it does not guarantee the endpoint of a neural training run. We will first inspect the exact solution numerically, then return to a trainable parameterization in Lesson 8.2.

> **Mathematical extension:** *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” ([open module](/courses/information-theory-for-ml/module-08/)) develops variational forms, partition functions, and maximum-entropy connections.

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

**Where the guarantee ends.** An average constraint $\mathbb E[C]\le d$ does not guarantee safety on every prompt. Critical actions require additional mechanisms such as hard filters, safe action sets, refusal or abort paths, and scenario-level checks. A learned cost model can also be wrong and can itself be optimized against. Training-time constraints and the release controls in Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” ([open module](module-13.md)) therefore complement rather than replace one another.

Constrained Policy Optimization (CPO) is a classical policy-optimization line for expected-cost constraints. Safe RLHF adapts the reward-versus-cost separation to LLM alignment. These sources provide mathematical and empirical grounding; neither turns an average constraint into an absolute safety guarantee.

---

## Lesson 8.2. DPO: derivation, loss, and gradient

The optimum expresses a policy in terms of a reward. Reading that relationship backwards is the key to DPO. We will substitute a policy-based expression for the reward in the preference likelihood, then turn the resulting likelihood into a loss for a trainable generator.

### Step 8.2.1 — Preference pairs as probabilistic observations

For one prompt, let $y^+$ be chosen and $y^-$ rejected. The Bradley–Terry (BT) model writes

$$
P(y^+\succ y^-\mid x)
=
\sigma\left(r(x,y^+)-r(x,y^-)\right).
$$

Module 7 “Reward models and preference data” ([open module](module-07.md)) used this likelihood to fit an explicit reward model. DPO instead expresses reward through a policy and substitutes that expression directly into the preference likelihood.

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

So far, the equations relate a reward to its corresponding ideal policy. Neither the true reward nor that completed ideal policy has been supplied to us. We therefore do not train by copying an already known optimum. Instead, we make the preference probability a function of a trainable policy and fit that function to the observed comparisons.

For a positive policy on common support, $r_\theta(x,y)=\beta\log[\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]$ is a valid representative of an implicit reward. Substituting that particular reward into the variational solution returns $\pi_\theta$ after normalization. This is a property of the parameterization, not evidence that the current model already optimizes human preferences. Learning must still find policy parameters whose induced preference probabilities explain the data.

Replacing $\pi^*$ with $\pi_\theta$ thus changes what is parameterized. Module 7 fitted a separate scoring function; DPO fits the generator, using its log-ratio to a fixed reference as the score in the preference likelihood. The empirical comparisons determine which relative probabilities are encouraged.

Cancellation of $Z(x)$ is also what makes the construction computationally useful. Evaluating one pair does not require enumerating all possible completions. We need the likelihoods of the two recorded responses under the policy and the reference, obtainable as sums of autoregressive token log-probabilities. The separate reward-model fit and RL loop disappear from this stage, but the need for informative preference data does not.

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

The weight is a scalar; the update acts on the parameters of the whole model. With the reference held fixed, the chain rule gives

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}}
=-w(h_\theta)
\left[
\nabla_\theta\log\pi_\theta(y^+\mid x)
-\nabla_\theta\log\pi_\theta(y^-\mid x)
\right].
$$

Gradient descent reverses the leading sign and moves along a difference of two sensitivity vectors. It does not operate two independent probability controls. Both responses share parameters with each other and with unlabelled completions, which will matter in Lesson 8.3.

At initialization, a policy identical to the reference has $h_\theta=0$ and $w(0)=\beta/2$. Unit-valued likelihood ratios do not imply zero derivatives: the reference is constant while the policy branch is trainable. The difference between the response gradients supplies the learning signal. A large scalar weight is therefore not, by itself, a claim about the norm of the complete parameter update; the vectors may be close, or contributions from other pairs may counteract them.

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

The reference ratio also serves a different purpose from the current-to-old ratio in PPO. PPO uses its ratio in a construction for reusing sampled trajectories. In DPO, the log-ratio parameterizes a preference probability on a fixed pair; it is not an importance weight that turns the recorded dataset into samples from the updated policy. Candidate generators need not therefore be identical to the reference. Their provenance still matters for coverage and transfer, but the standard DPO loss does not acquire an extra behavior-policy denominator simply because candidates came from another model.

---

## Lesson 8.3. Where the simple picture breaks

A convenient loss does not specify every aspect of a model’s behavior. We now examine what a pairwise margin controls, what it leaves unconstrained, and why the exact tokens and preprocessing of a preference pair belong to the mathematical problem rather than merely to data engineering.

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

which grows strictly.

For a direct comparison with DPO, take the uniform distribution at $t=0$ as the reference. Its pairwise log-probability difference is zero, so $h_\theta=2t$. A different fixed reference subtracts a constant without changing the growth of the margin. The parameter $t$ controls all three logits together; the model cannot independently raise the first probability while holding the others unchanged. The resulting probabilities are:

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

> **Engineering extension:** *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)) covers dialogue-data and chat-template engineering in more depth.

---

### Step 8.3.4 — Noisy pairs and conservative DPO (cDPO)

Conservative DPO (cDPO) smooths the binary target. Rather than treating each recorded preference as certain, it uses a soft target $1-\varepsilon$, with $\varepsilon\in[0,1/2)$:

$$
\mathcal L_{\mathrm{cDPO}}
=
(1-\varepsilon)\operatorname{softplus}(-z)
+
\varepsilon\operatorname{softplus}(z).
$$

At $\varepsilon=0$, the ordinary DPO loss is recovered. The effect of positive smoothing is visible by differentiating with respect to one scalar logit:

$$
\frac{\partial\mathcal L_{\mathrm{cDPO}}}{\partial z}
=\sigma(z)-(1-\varepsilon),
\qquad
z^*=\log\frac{1-\varepsilon}{\varepsilon}
\quad(0<\varepsilon<1/2).
$$

For an independently adjustable pair, the minimum predicts probability $1-\varepsilon$. Below that probability the gradient increases the margin; above it, the gradient reverses. When $\varepsilon=0$, no finite scalar minimizer remains. A shared neural model couples many pairs, so this calculation explains the local learning signal, not a guarantee that every training row attains its individual optimum.

Smoothing deliberately limits confidence in a label. It is not a general procedure for recovering an unknown clean preference distribution. In particular, it should not be conflated with Robust DPO, whose correction is derived for a specified symmetric label-flip model. A common motivation—noisy feedback—does not make the two losses interchangeable.

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

The following methods change different parts of the construction: the loss shape, response normalization, the reference, or the feedback format. They are alternatives with distinct assumptions, not successive upgrades that every training pipeline must adopt.

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

Here $\tau>0$. There is a useful connection to cDPO, but not an identity: smoothing specifies a soft preference probability, whereas IPO places a quadratic target directly on $h_\theta$. Even when hyperparameters produce the same isolated minimizer, the losses can have different curvature and different gradients away from it. Their finite targets do not make their update rules equivalent.

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

The symbol $\gamma$ denotes a preference margin here, not an RL discount factor. Applying the usual logistic loss $\operatorname{softplus}(-z)$ clarifies its role. When the score difference reaches $\gamma$, the logit is zero and the predicted preference probability is $1/2$; the gradient has not vanished. A larger difference still reduces the loss. SimPO’s target margin therefore shifts the comparison threshold rather than creating IPO’s finite scalar minimizer.

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

The odds term needs one additional definition. The original construction starts with the mean log-likelihood of the active response tokens. Let $q_\theta(x,y)$ denote its exponential, keeping it distinct from the probability of the whole sequence:

$$
\begin{aligned}
q_\theta(x,y)&=\exp\!\left(\frac{1}{|y|}\log\pi_\theta(y\mid x)\right),\\
o_\theta(x,y)&=\frac{q_\theta(x,y)}{1-q_\theta(x,y)},\\
\mathcal L_{\mathrm{odds}}
&=-\log\sigma\!\left(\log\frac{o_\theta(x,y^+)}{o_\theta(x,y^-)}\right).
\end{aligned}
$$

For $0<q_\theta<1$, the odds transformation is nonlinear and becomes particularly sensitive near the upper boundary. Here $q_\theta$ is a geometric mean of token probabilities, not a normalized probability over complete responses. Its complement should therefore not be read literally as the probability of every other possible text. Length normalization is part of the definition; inserting a raw sequence probability would produce a different loss.

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

> **Survey extension:** *Modern LLMs*, Module 14 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” ([open module](/courses/modern-llms/module-14/)) provides a broader post-training map. This module focuses on the RL interpretation, derivations, and diagnostics.

---

## Lesson 8.5. From equations to a reproducible run

We can now follow a recorded pair through one update and ask how to evaluate the resulting generator. Computing a preference loss does not require sampling a new response at each optimization step. Establishing that the model has become more useful, however, requires evidence beyond that same loss.

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

Follow one pair through an update. Both completions already exist in the dataset. The policy and reference receive the same prompt and the recorded prefixes of each completion. Causal rescoring supplies the probabilities of the tokens actually present. Four sequence log-probabilities—two completions under two models—produce $h_\theta$, the preference logit, and the loss. Backpropagation traverses both trainable policy branches; the reference stays fixed.

A generative model can therefore be trained without sampling a new completion at every optimization step. The training objects are given, while their likelihoods change with the parameters. Fresh generation is still needed to assess the learned behavior, and iterative preference learning may introduce separate rounds that collect new pairs.

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

The executable practice provides two routes:

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

The metric definition matters. In the teaching DPO implementation, `pair_accuracy` is the fraction of pairs with $h_\theta>0$; a zero margin is not counted as a win. It measures the order of implicit reference-relative rewards, not the raw condition $\pi_\theta(y^+\mid x)>\pi_\theta(y^-\mid x)$ and not a match between free generation and the recorded completion. These measurements may therefore disagree without contradiction. Similarly, exact vocabulary KL evaluated at the prefixes of fixed pairs must be labelled as such: it is not automatically sequence KL averaged over fresh trajectories from the current policy.

---

### Step 8.5.5 — Decision rule: DPO, RLHF, or online RL

A useful sequence is:

1. **Is the feedback reliable?** If not, clarify the rubric and data first. A different loss cannot resolve an underspecified criterion.
2. **Are there same-prompt preference pairs?** Fixed pairs make DPO a useful starting point, with its assumptions and diagnostics checked explicitly.
3. **Must the data track the current policy?** Choose how to refresh them: separate rounds of candidate generation and pair annotation, or a direct RL loop using fresh trajectories.
4. **Can reward be checked programmatically?** This is useful for reinforcement learning with verifiable rewards (RLVR), covered in Module 9 “RLVR and GRPO: verifiable rewards, group-relative estimates, and stability” ([open module](module-09.md)). The same verifier can also filter demonstrations or construct preference pairs. A feedback source does not, by itself, choose an update algorithm.
5. **Should new responses be optimized directly against reward?** PPO and REINFORCE Leave-One-Out (RLOO) provide such update mechanisms. Basic RLOO uses a whole-response signal, not local causal labels for tokens. A PPO critic likewise does not magically identify each token’s contribution from a sparse terminal reward.
6. **Which part should be simplified?** Removing a reference model and combining SFT with preference learning are different choices. ORPO combines the two losses; SimPO removes the reference but does not, by that fact alone, remove a preceding SFT or instruction-tuning stage.

The final map is deliberately less dramatic than “DPO replaced RLHF”:

> DPO avoids a separate reward-model fit and RL stage when the pairwise data and preference model suit the task. A fresh RL loop is useful when updates should directly use the consequences of current behavior. Iterative pair collection is another way to keep data relevant; changing policies or using a verifier does not rule it out.

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

The works above support the corresponding methods; they do not replace the protocols of the course’s own experiments. For the variational derivation, use *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” ([open module](/courses/information-theory-for-ml/module-08/)); for SFT data engineering, use *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)); for the broader post-training map, use *Modern LLMs*, Module 14 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” ([open module](/courses/modern-llms/module-14/)).
