# Module 7. Reward models and preference data

> **Material version:** 2026.19  
> **Text revision:** 2026-10-26  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Core practice:** level A — a reproducible CPU workflow; level B — an optional open-model run with Low-Rank Adaptation (LoRA) in a runtime with a suitable accelerator  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 8–10 hours without optional model training

Module 6 “The LLM as a policy: tokens, log-probabilities, and KL” ([open module](/courses/rl-for-llm/module-06/)) made the policy, token log-probabilities, and rollout record explicit. The next question is unavoidable: **who assigns the number that the policy will later optimize?** Unit tests can do this for some coding tasks, but usefulness, clarity, tone, and safety rarely come with a complete executable checker. We therefore translate a criterion into preference data and train a reward model (RM).

The practical question of this module is:

> **How do we build a training signal that represents the intended criterion, transfers beyond the training sample, and does not turn a convenient proxy into a false objective?**

A reward model does not automatically discover “true answer quality.” It approximates decisions produced by a particular protocol: a rubric, a candidate set, human annotators or a model judge, tie rules, and an aggregation procedure. This is not a weakness of the definition. It is what makes an RM an auditable engineering component.

**By the end of the module, you will be able to:**

1. distinguish scalar ratings, binary labels, pairwise preferences, rankings, ties, and abstentions;
2. design a preference-collection protocol that does not leak answer position or generator identity into the label;
3. derive the Bradley–Terry and Plackett–Luce likelihoods;
4. implement a numerically stable pairwise loss and interpret its gradient;
5. explain what within-prompt comparisons do not identify;
6. evaluate an RM with ranking, probabilistic, calibration, and distribution-shift diagnostics;
7. separate annotator disagreement from model error instead of assuming a universal “70–80% ceiling”;
8. split data by prompt and test transfer to new sources, generators, languages, and domains;
9. reproduce a toy overoptimization curve in which proxy reward rises while an independent objective eventually falls;
10. distinguish scalar RMs, executable verifiers, LLM judges, generative RMs, and multi-criterion rubrics;
11. prepare a smoke run for training an RM on an open 0.6-billion-parameter model.

The English executable practice is on its own page. Primary works are listed in the [sources section](#sources-and-further-reading), with assumptions and inference boundaries stated alongside the relevant explanations.

---

> **Practice route.** [Exercises](/courses/rl-for-llm/module-07/exercises/), [assessment questions](/courses/rl-for-llm/module-07/tests/), and [executable practice](/courses/rl-for-llm/module-07/notebooks/) are on separate pages. The lecture is self-contained.

## Lesson 7.1. Feedback is data, not ready-made truth

Before training a scorer, establish what its observations mean. We start with an evaluator’s decision, identify the protocol that produced it, and then build a probabilistic model of comparison.

### Step 7.1.1 — Define the criterion before constructing the reward

Suppose a policy produces two answers to the same prompt and a person selects the first. What exactly did we learn?

We did not obtain a universal scalar called “the quality of answer one.” We observed that, for this prompt, this pair of candidates, this annotation instruction, and this evaluator, the first answer was preferred.

At least four ingredients determine that observation:

1. **Criterion.** Was the evaluator judging factual correctness, usefulness, brevity, safety, or a trade-off among them?
2. **Candidate set.** Choosing between a clearly good and clearly bad answer is different from choosing between two strong but differently compromised answers.
3. **Decision source.** A person, a panel, a program, an environment outcome, and another LLM may disagree systematically.
4. **Protocol.** Display order, visible model names, the rubric, permission to tie, and access to external sources can all change labels.

A better mental model is therefore a contract:

$$
(x,\text{candidates},\text{rubric},\text{protocol})
\longrightarrow
\text{observed feedback}.
$$

![From a criterion to a training signal](assets/rl-for-llm/en/module-07/M7_feedback_contract_EN.png)

The RM learns to predict this observed signal. A larger model cannot resolve a logically underspecified contract. One evaluator may treat detail as helpful; another may treat the same detail as unnecessary verbosity. Both can be internally consistent while optimizing different objectives.

**Boundary of the claim.** In this course, “target reward” means an operationally specified criterion or an independent check. It does not mean a philosophically final measure of human values. Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” ([open module](/courses/rl-for-llm/module-13/)) returns to broader evaluation and proxy-exploitation questions.

---

### Step 7.1.2 — Five feedback formats

The same criterion can be encoded in several ways.

| Format | Example | What it provides | Main cost |
|---|---|---|---|
| scalar rating | “7 out of 10” | an absolute-looking signal and direct regression target | annotators use scales differently |
| binary label | “acceptable / unacceptable” | a cheap threshold decision | degree of quality is lost |
| pairwise comparison | “A is better than B” | a local improvement direction | no absolute level |
| listwise ranking | $A\succ C\succ B$ | more comparative structure per prompt | greater cognitive load |
| tie or abstention | “equivalent” / “insufficient evidence” | preserves ambiguity | needs explicit modeling and aggregation |

Pairwise annotation is often convenient because an evaluator need not maintain a global 1–10 scale. That does not make pairs universally superior. A hard safety threshold may naturally produce binary labels. A ranking can be efficient when several candidates are close. Forced choice is harmful when the difference is immaterial or evidence is insufficient.

![Preference-data formats](assets/rl-for-llm/en/module-07/M7_preference_models_EN.png)

The main derivations below use pairs because they expose the mechanism cleanly. The broader map still matters.

**Practical rule.** Do not silently replace a tie with a random winner. A soft target of $1/2$ in a logistic loss can be a useful approximation, but it is not a full probabilistic model of ties. Models such as the Davidson extension represent ties explicitly.

---

### Step 7.1.3 — Prevent nuisance signals from becoming labels

A good preference dataset begins with an experimental protocol, not a file schema. For each pair, record at least:

- the exact prompt and chat template;
- both raw answers before manual editing;
- generator identity, revision, and decoding settings in metadata, while hiding them from evaluators when appropriate;
- the rubric and its version;
- randomized A/B order;
- preference, tie, or abstention;
- annotator identity and qualification, or judge model and judge prompt;
- time, language, domain, and available external evidence;
- example provenance and license.

Three common mistakes look deceptively harmless.

**Showing the model name.** The label can capture reputation rather than answer content.

**Always placing the stronger answer first.** A human evaluator or an LLM judge that sees the whole pair may learn to favor a position rather than the content. The resulting labels can carry that bias into training. A scalar RM that scores each answer separately with the same function does not automatically see which branch it occupies: storing the preferred text in a field named `chosen` is not itself a positional feature. Direct leakage requires a branch marker, a positional cue, or another label-correlated feature in the scorer's actual input. The annotation procedure and the model input therefore need separate checks.

**Splitting different responses to the same prompt across train and test.** The model can memorize prompt-specific cues and appear to generalize.

The minimum protocol therefore includes generator blinding, position randomization, prompt-level splitting, and provenance tracking. Multiple independent judgments on important examples reveal ambiguity rather than hiding it behind one label.

**Cross-course link.** Dialogue data and chat templates are covered in *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)). Here the focus is narrower: how an annotation protocol defines the statistical target of the RM.

---

### Step 7.1.4 — The Bradley–Terry model

Let $x$ be a prompt, $y^+$ the preferred answer, and $y^-$ the rejected answer. The RM outputs a scalar

$$
r_\theta(x,y)\in\mathbb{R}.
$$

The scalar $r_\theta(x,y)$ is not yet a probability or a known quality label. Its purpose is to make the observed comparisons predictable. The data tell us which answer was preferred; they do not assign an independently correct score to each answer.

The Bradley–Terry model (BT) supplies a particular link between scores and decisions. Give each answer a positive relative worth $w_\theta$, and assume that its chance of being selected is proportional to its share of the two candidates' total worth. An exponential parameterization lets the underlying score range over the real line:

$$
w_\theta(x,y)=\exp r_\theta(x,y),
\qquad
P(y^+\succ y^-\mid x)
=\frac{w_\theta(x,y^+)}{w_\theta(x,y^+)+w_\theta(x,y^-)}.
$$

Dividing by the first answer's worth turns this expression into a sigmoid of the score difference:

$$
P(y^+\succ y^-\mid x)
=
\sigma\!\left(r_\theta(x,y^+)-r_\theta(x,y^-)\right),
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Define

$$
\Delta_\theta=r_\theta(x,y^+)-r_\theta(x,y^-).
$$

The negative log-likelihood of one pair is

$$
\mathcal{L}_{\mathrm{BT}}(\theta)
=-\log\sigma(\Delta_\theta)
=\operatorname{softplus}(-\Delta_\theta).
$$

The `softplus` form is numerically stable even when the preferred item is assigned a very low score. Its derivative is

$$
\frac{\partial \mathcal{L}_{\mathrm{BT}}}{\partial \Delta_\theta}
=
\sigma(\Delta_\theta)-1.
$$

A confidently wrong pair therefore produces a large correction, while a confidently correct pair produces a small one. This is binary logistic classification whose logit is the difference between two outputs of the same scoring function.

The sigmoid is a modeling assumption, not a unique consequence of having pairwise data. Its probability concerns a preference under the stated evaluation protocol, not the factual correctness of an answer. The superscripts in $y^+$ and $y^-$ identify the observed winner and loser; the model may still assign them the opposite order.

---

### Step 7.1.5 — From pairs to rankings: the Plackett–Luce model

Suppose an evaluator ranks $m$ answers as

$$
y_{\pi_1}\succ y_{\pi_2}\succ\cdots\succ y_{\pi_m}.
$$

One can turn the ranking into pairs and train with a sum of pairwise losses. The comparisons still come from a single listwise decision, however; they are not that many independent annotation events. A pairwise training objective can be useful without its product of probabilities being the full likelihood of the original ranking. The Plackett–Luce model (PL) specifies that likelihood directly:

$$
P(\pi\mid x)
=
\prod_{k=1}^{m-1}
\frac{\exp r_\theta(x,y_{\pi_k})}
{\sum_{j=k}^{m}\exp r_\theta(x,y_{\pi_j})}.
$$

The model first selects the best item among all candidates, then the best among the remaining items, and so on. For $m=2$, it reduces to BT.

Consider the earlier ordering $A\succ C\succ B$. The first factor selects A from all three answers, so all three worths enter its denominator. The next factor selects C from the remaining C and B. Once C has been chosen, B is the only item left and contributes a conditional probability of one. This is why the product ends at $m-1$. The factors describe one complete ranking, not a fresh collection of independently labeled pairs.

PL uses more of the listwise structure, but it also assumes a sequential-choice mechanism and an independence-of-irrelevant-alternatives property. If adding a near-duplicate candidate changes human preferences in a richer way, PL is misspecified.

> **The feedback format and the probabilistic model should match the decision the evaluator actually made.**

No format is universally correct. The choice depends on the criterion, annotation cost, ambiguity, and the assumptions one is willing to defend.

---

## Lesson 7.2. Training a scalar reward model

The preference likelihood now gives scores a training role. Next we follow it into a shared scoring function, inspect the text that function actually sees, and determine which aspects of its scale the data leave unresolved.

### Step 7.2.1 — A common architecture, not the definition of an RM

A typical text RM contains:

1. a tokenizer and chat template;
2. a transformer that processes the prompt and answer;
3. a rule for selecting a sequence representation;
4. a scalar head.

For a decoder-only model, one can use the hidden state of the last non-padding token,

$$
h_\theta(x,y)\in\mathbb{R}^d,
$$

and compute

$$
r_\theta(x,y)=w^\top h_\theta(x,y)+b.
$$

![Training a scalar reward model](assets/rl-for-llm/en/module-07/M7_reward_training_EN.png)

The same RM scores $y^+$ and $y^-$, and the BT loss is applied to their difference. Shared parameters force one scoring function to explain both sides of the comparison.

Follow one update through this construction. Each answer is paired with the prompt, the transformer produces a sequence representation, and the scalar head gives two scores. Their difference determines the likelihood of the observed preference. In a causal transformer, the last non-padding token can attend to the preceding text; choosing its state is a representation rule, not a claim that the last word alone determines quality.

Let $\theta$ denote all trainable scorer parameters. Applying the chain rule to Step 7.1.4 gives

$$
\nabla_\theta\mathcal L_{\mathrm{BT}}
=
\bigl[\sigma(\Delta_\theta)-1\bigr]
\left[
\nabla_\theta r_\theta(x,y^+)
-
\nabla_\theta r_\theta(x,y^-)
\right].
$$

The two scalar outputs have opposite loss derivatives, but they are not independently adjustable entries: they share parameters. An update combines evidence from comparisons rather than assigning a predetermined score to each answer. There is no need to label the winner 10 and the loser 0; the dataset supplies no such absolute targets. Shared parameters also prevent a guarantee that every individual score moves in the desired direction after every update.

At this stage we train the scorer, not the policy that produced the answers. Initializing from a language model does not make this another token-level SFT objective: supervision comes from a comparison between two sequences. Even the common head bias $b$ cancels in the difference, so pairwise loss alone supplies no gradient for it. This is an early sign of the free score origin examined in Step 7.2.3.

A transformer with a scalar head is a common implementation, not the definition of a reward model. The evaluator could instead be an encoder, an ensemble, executable code, a generative critique model, or an environment outcome. The defining role is to transform available evidence into a signal used for selection or policy updates.

**Engineering detail.** When LoRA is applied to a causal model, the newly introduced scalar head must be trained and saved along with the adapters. Otherwise a seemingly successful run can leave the scoring head random or fail to restore it.

---

### Step 7.2.2 — What exactly enters the model

The preferred and rejected branches must be processed symmetrically:

$$
(x,y^+)\quad\text{and}\quad(x,y^-).
$$

Symmetry is easy to break:

- one response is truncated while the other is complete;
- different chat templates are used;
- an end-of-sequence token (EOS) is retained on one branch but not the other;
- answer length correlates with the label and the RM learns verbosity;
- generator identity leaks through a special token.

When the maximum sequence length is $L$, the truncation policy must be explicit. Truncating the end may remove the final conclusion or refusal that determined the preference. Truncating the beginning may remove the prompt. Filtering all long pairs changes the data distribution. No choice is neutral.

The training record should therefore retain:

- raw prompt and answers;
- templated strings;
- lengths before and after tokenization;
- exclusion or truncation reason;
- tokenizer and template revision;
- padding mask and the rule used to select the scalar-head state.

Module 6 established the same principle for policy losses: a loss has meaning only when the tokens on which it was built are known exactly.

---

### Step 7.2.3 — Shift invariance and an unidentified score origin

BT depends only on differences. For any prompt-dependent function $c(x)$,

$$
r_\theta(x,y)\longmapsto r_\theta(x,y)+c(x)
$$

leaves within-prompt probabilities unchanged:

$$
[r(x,y^+)+c(x)]-[r(x,y^-)+c(x)]
=r(x,y^+)-r(x,y^-).
$$

This is an additive gauge freedom. A tabular implementation can impose zero-mean scores; a neural RM can use a centering regularizer. Such a convention chooses one representative among equivalent solutions. It does not recover an absent absolute quality level.

A second boundary matters. If all observations compare answers only within the same prompt, the pairwise likelihood alone does not identify absolute score comparability across prompts. Shared transformer parameters and regularization couple examples in practice, so the model still emits a common numeric scale. That scale comes from modeling assumptions, not from within-prompt pairs as a theorem.

Scale is also meaningful only relative to the logistic noise model. Multiplying all scores by a positive constant preserves ranking but changes predicted probabilities. Two RMs can therefore have identical pair accuracy and different calibration.

Origin and scale are therefore not interchangeable ambiguities. With the sigmoid $\sigma(\Delta)$ fixed, rescaling scores generally changes the data likelihood; adding the same offset within a prompt does not. Learning a separate preference temperature would create an additional scale–temperature question. The BT specification on this page has already fixed that temperature.

Can a score with an arbitrary zero still serve as reward? For a fixed prompt and one terminal score per complete response,

$$
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)+c(x)]
=
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]+c(x).
$$

The added term is independent of the answer choice. With a fixed prompt distribution it is also independent of policy parameters, so it leaves the exact expected gradient and the optimum unchanged, including when the same KL penalty is retained. This is the action-independent-baseline logic from Module 4. Finite-sample noise and implementation-level normalization can still change. The claim concerns one terminal response reward: adding a constant at every token of a variable-length answer would change the total objective.

Rescaling has a different effect. Multiplying reward by $a>0$ while keeping the KL coefficient at $\beta$ changes the trade-off; in the exact objective it is equivalent to a KL price of $\beta/a$ relative to the original reward. Handing a new RM to an optimizer therefore requires a scale convention, not merely a ranking check. Module 8 derives the regularized policy in detail. Here the key distinction is that a free origin does not make reward unusable, while normalization is not always cosmetic.

---

### Step 7.2.4 — Ties, soft labels, and inconsistent cycles

A simple approximation for a reported tie uses a target probability

$$
t=\frac12
$$

and binary cross-entropy

$$
\mathcal{L}(\Delta,t)
=
\operatorname{softplus}(\Delta)-t\Delta.
$$

Its gradient is

$$
\frac{\partial\mathcal{L}}{\partial\Delta}
=
\sigma(\Delta)-t,
$$

so $t=1/2$ is minimized at $\Delta=0$.

This is convenient, but it does not model tie probability as a separate outcome. If ties are part of the data-generating process, use a model or head that represents them explicitly.

Abstention says something else. “Equivalent under the rubric” is a comparative judgment; “I lack the evidence to compare them” is a statement about missing information. Mapping both to $t=1/2$ silently turns missing evidence into a claim of equality. Keep the outcomes separate and specify which ones contribute to the chosen loss.

Preferences may also form cycles:

$$
A\succ B,
\qquad B\succ C,
\qquad C\succ A.
$$

No scalar ordering can satisfy all three strict relations at once. For this isolated triangle with equal weights on the observations, the BT optimum assigns equal scores up to a common shift and probability $1/2$ to each pair. That is a compromise between incompatible observed directions, not a numerical failure.

An observed cycle does not by itself refute the probabilistic model. Finite BT scores allow either outcome in each comparison, so a few random decisions can form a cycle even when the underlying scores are scalar. Persistent frequencies assigning probability above one half to every edge of the cycle are different: one BT scale cannot represent that pattern. Equal fitted scores on the small balanced dataset do not establish that the answers are truly equivalent either.

Cycles can arise from noise, ambiguity, differing criteria, or heterogeneous user groups. Deleting them automatically can erase evidence that one scalar axis is insufficient.

---

### Step 7.2.5 — Practice: from NumPy to a 0.6B model

The practice has four resource profiles.

**Level A — CPU.** `module_7_reference.py` implements a stable BT loss, its gradient, PL likelihood, tabular score fitting, calibration diagnostics, and a toy overoptimization experiment. This route needs only NumPy.

**Level B — optional training of a 0.6B model.** The following resource plan concerns a neural RM with LoRA, not the NumPy-only route above.

| Parameter | Smoke mode | Full learning mode |
|---|---:|---:|
| training pairs | 512–1,000 | 10,000–50,000 |
| maximum length | 512 | 1,024 |
| adaptation | LoRA for sequence classification | LoRA |
| device | T4 16 GB | T4 16 GB or a graphics processing unit (GPU) with 24 GB |
| goal | validate the complete data path | observe stable train/evaluation separation and calibration |

Runtime depends on the Colab instance, library revisions, and sequence lengths. These are resource plans, not benchmark numbers reproduced in this release.

**Level B+ — application programming interface (API).** Instead of training a local RM, a paid judge can generate preference labels. Record the provider, exact dated model identifier, rubric, answer order, temperature, repeated calls, token use, and cost.

**Level C — optional full run.** A 1.5B model, long responses, and the full dataset may require 24–48 GB of GPU memory or multiple GPUs. This belongs to the capstone and does not affect module completion.

> A successful 512-pair run proves that the code and data format agree. It does not prove that the RM is safe or useful for policy optimization.

---

## Lesson 7.3. Evaluation: ordering, probability, transfer, and uncertainty

A low training loss does not establish readiness for use. We will check ordering, probabilistic predictions, and transfer separately, with particular attention to the responses an updated policy will produce.

### Step 7.3.1 — Why pair accuracy is not enough

An RM is correct on a held-out pair if

$$
r_\theta(x,y^+)>r_\theta(x,y^-).
$$

Pair accuracy is intuitive, but it ignores confidence. Predictions of $0.51$ and $0.99$ count equally when correct, although a wrong $0.99$ should be penalized more heavily.

Measure at least three families of quantities:

1. **Ordering:** pair accuracy, tie-aware accuracy, rank correlation for lists.
2. **Probabilistic quality:** mean BT loss or binary log loss.
3. **Calibration:** whether predicted preference probabilities match empirical frequencies.

For probabilistic evaluation, fix an orientation before looking at the label: answers A and B to the same prompt. Let $p_i$ predict that A is preferred, with $z_i=1$ when the protocol selects A and $z_i=0$ when it selects B. Prediction and observation must refer to the same event. “Winner” and “loser” were convenient names in the training loss; calibration should not quietly redefine its event after seeing the outcome. Ties and abstentions follow a separately declared rule.

If about four fifths of decisions favor A among pairs assigned probability near 0.8, the forecast agrees with the frequency of **preference under that rubric**. This does not mean four fifths of those answers are factually flawless. Factual correctness may be only one part of the criterion, or may not have been checked at all.

The Brier score for probabilities $p_i$ and labels $z_i\in\{0,1\}$ is

$$
\operatorname{BS}
=
\frac1n\sum_{i=1}^{n}(p_i-z_i)^2.
$$

A reliability diagram bins predictions and compares confidence with win frequency. Expected calibration error (ECE) summarizes these gaps, but it depends on binning and is not a uniquely correct calibration measure.

Brier score assesses probabilistic predictions as a whole rather than calibration alone; distinguishing more predictable pairs from less predictable ones matters too. The summaries are related, but they do not become interchangeable merely because both use probabilities.

![Reward-model evaluation stack](assets/rl-for-llm/en/module-07/M7_evaluation_stack_EN.png)

Accuracy asks whether ordering is correct. Log loss asks whether the probability is sensible. Calibration asks whether stated confidence matches observed frequency. None alone predicts behavior under strong policy optimization.

---

### Step 7.3.2 — The split should imitate the future shift

Randomly splitting rows is often too easy. If answers A/B to one prompt appear in training and C/D to the same prompt appear in evaluation, the RM has already seen the topic and prompt-specific cues.

The minimum rule is:

> all comparisons for one prompt belong to one split.

Then add stricter slices when they match the deployment question:

- unseen response generators;
- new prompt sources;
- new languages and domains;
- different length ranges;
- stronger policies;
- adversarial examples;
- responses collected after a policy update.

The last slice is crucial for reinforcement learning from human feedback (RLHF). An RM is trained on responses from some initial policies. The optimized policy then searches for high-scoring responses, moving the distribution toward regions where RM errors can be profitable.

A useful evaluation report therefore does not stop at `accuracy=0.78`. It reports source, generator, language, length, criterion, and novelty slices. A deployment regime that matters deserves its own held-out set.

---

### Step 7.3.3 — RewardBench 2 is a benchmark, not a certificate

RewardBench evaluates preferences on curated response pairs. [RewardBench 2](https://arxiv.org/abs/2506.01937) changes the main selection task: the evaluator must identify one preferred response among four candidates, with three distractors. A separate *Ties* subset tests how it treats multiple equivalent correct answers while separating them from incorrect ones. The second benchmark is therefore not simply a collection of harder pairs. Its score must be read together with the changed candidate set and the special scoring rule for ties.

A shared benchmark remains useful for comparing models under one protocol and locating systematic weaknesses.

The dated RewardBench 2 report makes several empirical claims:

- the evaluated contemporary models scored about twenty points lower on average than on the first benchmark;
- new prompts were collected from people rather than only inherited from existing downstream evaluations;
- in the authors' experiments, benchmark performance correlated with RM use in Best-of-N and Proximal Policy Optimization (PPO).

These are findings for a particular benchmark and experimental suite. A high score does not certify resistance to overoptimization or transfer to a new language, rubric, policy, or attack family. A fixed benchmark also cannot cover all regions that an optimizing policy may discover.

Use it as one layer:

1. a shared benchmark for external comparability;
2. a task-specific held-out set;
3. slices for suspected proxy features;
4. evaluation on responses from the updated policy;
5. independent checks during optimization.

---

### Step 7.3.4 — Annotator disagreement is not a universal ceiling

There is no universal “70–80% agreement” constant or RM ceiling. Agreement varies with domain, rubric, pair difficulty, evaluator population, aggregation, and metric definition.

A narrow model does yield an exact ceiling. Suppose each pair has a hidden strict order and the observed label is independently flipped with probability $\eta$. A model that always recovers the hidden order agrees with **one noisy label** with probability

$$
1-\eta.
$$

For $\eta=0.2$, the value is $0.8$. This is a consequence of an independent-flip model, not a universal law of preference annotation.

Real datasets may contain genuine ambiguity, multiple defensible value systems, unequal expertise, correlated errors, order effects, ties, and a flawed rubric. Aggregating several labels can be more reproducible than predicting one noisy label. A personalized model can predict a particular group better than a single population-wide ordering. Conversely, high accuracy on easy consensus pairs can hide failure on the contested cases that matter most.

Report separately:

1. agreement among feedback sources;
2. RM quality relative to a chosen aggregation;
3. uncertainty in the criterion itself.

---

### Step 7.3.5 — An ensemble is an uncertainty signal, not a magic shield

A single RM can be confidently wrong. Comparing several scorers offers a natural way to find answers whose evaluation is model-dependent. Before interpreting a spread of numbers, however, recall Step 7.2.3: are the numbers on comparable scales?

Adding a different prompt-dependent offset $c_k(x)$ to each model leaves every one of its within-prompt preferences unchanged. It can nevertheless change the raw-score dispersion for a single answer. A large standard deviation may reflect different origins rather than different judgments. Agreement about rankings likewise does not establish agreement about probability scale.

Raw-score aggregation therefore needs a declared origin and scale convention. One can, for example, anchor scores to the same control answers or use a fixed within-prompt centering rule, and check the scale on a common calibration set. Apply the same procedure to the candidates being compared and fix it before making the selection. Subtracting one global mean does not remove arbitrary prompt-dependent offsets. Normalization does not create missing absolute-quality labels either.

Suppose the $K$ models now return scores $r_1,\ldots,r_K$ for one response **under that shared convention**. Compute

$$
\bar r=\frac1K\sum_{k=1}^{K}r_k,
$$

and

$$
s=\sqrt{\frac1K\sum_{k=1}^{K}(r_k-\bar r)^2}.
$$

A simple conservative heuristic is

$$
r_{\mathrm{cons}}=\bar r-\lambda s,
\qquad \lambda\ge 0.
$$

![Ensemble disagreement and uncertainty](assets/rl-for-llm/en/module-07/M7_ensemble_uncertainty_EN.png)

Once the scales are aligned, a large spread indicates disagreement about this answer's score. To check whether the choice between A and B changes, compare margins for the same pair:

$$
\Delta_k(x;y_A,y_B)=r_k(x,y_A)-r_k(x,y_B).
$$

The offset $c_k(x)$ cancels. Opposite signs indicate different rankings; different magnitudes may indicate different confidence despite the same ranking. Probabilities $\sigma(\Delta_k)$ must still be interpreted through each model's calibration. This separates comparative disagreement from a free score origin without introducing a new ensemble-training method.

Substantive disagreement can indicate epistemic uncertainty. It does **not** automatically turn $\bar r-\lambda s$ into a statistically valid lower confidence bound; such an interpretation needs assumptions about dependence, calibration, and error distributions.

Shared bias is the harder failure mode. Every ensemble member can learn the same spurious feature. The RM-ensemble study [Helping or Herding?](https://arxiv.org/abs/2312.09244) found that diversity in pretraining helped more than merely changing fine-tuning seeds, while shared failures still remained.

The practical conclusion is modest:

- disagreement helps prioritize new labels;
- conservative aggregation can suppress isolated outliers;
- an independent criterion and adversarial evaluation are still necessary.

---

## Lesson 7.4. Proxy-reward overoptimization

Until now, evaluation examples arrived from outside the system. Once the system starts searching for high scores, the test changes: a rare ordinary error can become an attractive optimization target.

### Step 7.4.1 — Why a higher RM score can produce a worse answer

Let

$$
r_{\mathrm{proxy}}(x,y)
$$

be the RM score and

$$
r_{\mathrm{goal}}(x,y)
$$

an independent target criterion. They may correlate well on the training distribution. Optimization, however, does not select a random answer. It searches the tail in which the proxy is unusually large.

If

$$
\varepsilon(x,y)=r_{\mathrm{proxy}}(x,y)-r_{\mathrm{goal}}(x,y)
$$

varies across candidates, maximizing the proxy can favor favorable errors as well as quality. The usual upward-selection argument assumes, for example, independent additive noise independent of candidate utilities; variation or zero unconditional mean alone is insufficient. Increasing search or policy-optimization pressure can then amplify exploitation of the error.

This resembles maximization bias from Module 2 “State, Value, and Bellman Equations” ([open module](/courses/rl-for-llm/module-02/)), but here the policy changes the data distribution and pushes the RM beyond familiar examples.

Common proxy features include length, confident tone, restating rubric language, displaying plausible reasoning without correctness, familiar formatting, and over-refusal. No finite checklist is complete. The RM must therefore be evaluated under optimization pressure, not only on a random held-out sample.

---

### Step 7.4.2 — A reproducible toy experiment

Give each candidate two independent features:

$$
g\sim\mathcal N(0,1)
\quad\text{(useful feature)},
$$

$$
h\sim\mathcal N(0,1)
\quad\text{(nuisance feature)}.
$$

Define

$$
r_{\mathrm{proxy}}=g+h,
$$

and

$$
r_{\mathrm{goal}}=g-\max(h-1,0)^2.
$$

For moderate $h$, proxy and goal agree. Large positive $h$ raises the proxy but becomes harmful beyond the threshold. For each pool size $N$, draw $N$ candidates and select the proxy maximum.

No generator or RM weights are updated in this experiment. The candidate distribution and both scoring functions remain fixed; only the number of candidates inspected before selection changes. This isolates selection pressure. A system can search increasingly hard for proxy–goal disagreement without taking a single gradient step, connecting the example to Best-of-N in Module 3.

![A toy overoptimization curve](assets/rl-for-llm/en/module-07/M7_overoptimization_EN.png)

With `seed=7` and 4,000 independent trials, the module code produces:

| $N$ | mean proxy reward | mean target score |
|---:|---:|---:|
| 1 | -0.037 | -0.082 |
| 4 | 1.427 | 0.480 |
| 16 | 2.480 | 0.735 |
| 64 | 3.323 | 0.711 |
| 256 | 4.020 | 0.484 |
| 512 | 4.303 | 0.293 |

The proxy rises monotonically. The target initially improves, peaks around $N=16$ for this protocol, then falls. Larger pools find both higher $g$ and more extreme $h$; eventually the nuisance term dominates.

**Boundary.** This is a teaching construction, not a realistic RM or a universal scaling law. The transferable mechanism is selection on proxy error, not the particular numbers or peak location.

---

### Step 7.4.3 — What the reward-overoptimization scaling study showed

*Scaling Laws for Reward Model Overoptimization* studied a related question in a controlled setting. The authors used a fixed “gold” RM in place of people, trained proxy RMs from its labels, and increased optimization pressure using RL or Best-of-N.

In their experiments, the gold score could decline while the proxy score continued to rise. They also reported different functional behavior for RL and Best-of-N, with coefficients changing across RM size, dataset size, policy size, and KL penalty.

Keep two limitations next to the result:

1. a gold RM is a convenient synthetic stand-in, not true human utility;
2. the fitted forms belong to the studied models and protocols, not every RLHF system.

The study's durable contribution is methodological: evaluate a trajectory

$$
\text{optimization strength}
\longmapsto
(r_{\mathrm{proxy}},r_{\mathrm{independent}})
$$

rather than reporting only one final point.

---

### Step 7.4.4 — A defense portfolio, not one magic formula

Overoptimization is reduced by several independent controls.

**Limit optimization pressure.** A KL budget, fewer epochs, a smaller Best-of-N pool, or early stopping keeps the policy closer to the distribution on which the RM was evaluated. Small KL does not prove reward correctness; it narrows the extrapolation region.

**Track an independent signal along the path.** Periodic human review, a separately trained RM, executable subchecks, or manual audits at several optimization levels are needed.

**Collect fresh data.** Label responses from the current policy, especially high-scoring, high-disagreement, or adversarial cases, then retrain or recalibrate the RM.

These operations solve different problems. Positive temperature scaling of margins can correct excessive confidence without changing the sign of any preference. It cannot swap a good and a bad answer or remove a spurious feature by itself. Correcting substantive ordering requires suitable new evidence and changes to the scorer; matching probabilities to observed frequencies requires calibration checks.

**Use diversity and uncertainty.** Ensembles, different pretraining sources, rubrics, and adversarial judges reduce dependence on one error but cannot eliminate shared spurious features.

**Decompose the criterion.** Preserve executable properties—tests, types, citations, formatting constraints—separately from learned judgments. An incomplete verifier can also be exploited.

**Predefine stopping conditions.** For example, independent usefulness must not fall on two consecutive checks; ensemble disagreement must remain below a threshold; critical violations must stay at zero.

“Independent” describes the role of the check, not merely a different model name. A second scorer trained on the same flawed labels and features can repeat the first scorer's failure. Updating the RM on fresh data is not itself evidence that the intended outcome improved; that conclusion needs a separate check. A control set repeatedly used to choose settings also becomes part of the optimization process, so retain a final assessment outside that selection loop.

> **Evaluate the RM on the updated policy's responses, refresh its training data when needed, and retain a check that does not reduce to the optimized proxy. Retraining does not replace that check.**

---

### Step 7.4.5 — The iterative data loop

A practical process rarely ends after one RM fit.

1. Freeze a rubric and an initial prompt set.
2. Generate diverse candidates using several policies and decoding regimes.
3. Collect preferences, ties, and decision reasons.
4. Train the RM and measure ordering, log loss, calibration, and slices.
5. Use the RM moderately for Best-of-N, filtering, or a short policy update.
6. Select new high-scoring, disputed, adversarial, and distribution-shifted responses.
7. Obtain independent labels.
8. Compare proxy and independent metrics before continuing.

The loop resembles DAgger from Module 5 “Learning from fixed data: imitation, deep Q-methods, and offline RL” ([open module](/courses/rl-for-llm/module-05/)) because new data are collected where the updated policy goes. The label is different: here it is a preference over responses rather than an expert action. The analogy concerns distribution shift, not identical statistical objectives.

Version every iteration:

- candidate-generating policy;
- RM and its training data;
- rubric;
- aggregation rule;
- independent control set;
- optimization settings.

Without these versions, one cannot tell whether the system improved or the measuring instrument changed.

---

## Lesson 7.5. Modern reward systems

A scalar model is only one source of feedback. Comparing the roles of verifiers, judges, and rubrics will help us combine their outputs without losing the origin or meaning of the final reward.

### Step 7.5.1 — Five roles that should not be conflated

The word “reward” may refer to different mechanisms.

| Mechanism | Output | Strength | Main boundary |
|---|---|---|---|
| scalar RM | one number | cheap large-scale scoring after training | extrapolation and spurious features |
| executable verifier | test or rule result | precise for a formalized property | incomplete specification |
| LLM judge | rating, pair, or rubric | flexible open-text evaluation | bias, attacks, cost, and model-version drift |
| generative RM | critique or reasoning plus decision | can spend more computation and expose a rationale | a persuasive rationale need not be correct |
| environment outcome | task success, return, or state | closer to real consequences | delay, noise, and experiment safety |

One component can serve multiple roles, but the record should preserve components separately. For example,

$$
R=R_{\mathrm{tests}}+0.2R_{\mathrm{style}}-2R_{\mathrm{unsafe}}.
$$

If only the sum is stored, one cannot tell which component drove improvement.

A verifiable reward is not automatically a secure reward. Unit tests cover only written cases; a format checker can accept meaningless text; an environment can contain exploitable bugs. Formalization reduces ambiguity by moving the quality requirement into the verifier specification.

---

### Step 7.5.2 — For an LLM judge, protocol matters more than a clever prompt

LLM-as-a-judge uses another LLM to read a prompt, candidate answers, and a rubric, then output a rating or preference. It scales quickly and can produce explanations, but it inherits systematic biases.

Work on MT-Bench and Chatbot Arena documented position bias, verbosity bias, and self-enhancement effects among the limitations studied. A minimum pairwise protocol therefore:

1. hides generator identities;
2. randomizes A/B order;
3. evaluates both permutations for important cases;
4. maps both results to one probability that A wins;
5. uses an explicit multi-criterion rubric;
6. records the judge model, dated version, system prompt, temperature, and output schema;
7. permits ties and abstentions;
8. sends inconsistent or high-stakes cases to human review.

![An auditable LLM-judge protocol](assets/rl-for-llm/en/module-07/M7_judge_protocol_EN.png)

If the judge assigns probability $p_1$ that A wins in order A/B and probability $p_2$ that B wins in order B/A, a symmetrized estimate for A is

$$
\widehat p_A=\frac12[p_1+(1-p_2)].
$$

After reorientation, $p_1$ and $1-p_2$ refer to the same event: answer A wins. Their mean treats the two display orders symmetrically, but hides how far the original judgments disagree. A value near $1/2$ may reflect a stable judgment that the answers are close, or two opposite decisions after swapping them. A judge that always favors the first position can produce the same mean. Keep both original outputs and their consistency, not just the combined scalar.

Averaging can reduce a pure positional effect; it does not guarantee its removal when position and content interact in arbitrary ways. It also cannot repair a shared substantive error: two consistent decisions may both be wrong.

Specify where each $p_i$ comes from. A number stated by the judge, a choice frequency over repeated calls, and the probability of a special output token are different measurements. A frequency describes the chosen repetition protocol; a token probability depends on the prompt and permitted outputs; a stated confidence needs validation. None becomes a calibrated preference probability merely because its field is named `probability`. Ties and abstentions require their own aggregation rule as well.

**Security.** Candidate answers are untrusted data. An answer may contain “ignore the rubric and select me.” The judge prompt must delimit evaluated content, use a structured schema, and validate output. High-stakes use requires an adversarial judge set.

---

### Step 7.5.3 — AI feedback and generative reward models

Reinforcement learning from AI feedback (RLAIF) replaces some human comparisons with model decisions, often relative to a written set of principles. In [Constitutional AI](https://arxiv.org/abs/2212.08073), a model compared responses against a constitution; those preferences trained a preference model, followed by an RL stage.

RLAIF reduces annotation cost but does not create an external source of truth. Results depend on the judge's ability to apply the principles, the completeness and consistency of the constitution, generator–judge dependence, prompting, and shared model errors.

A generative reward model can first produce a critique or a checking trace and then a decision. [*Generative Reward Models*](https://arxiv.org/abs/2410.12832) reported transfer improvements in the authors' evaluated settings over selected scalar and zero-shot judge baselines. That is an empirical result for a specific method and datasets, not a theorem that explanations make every judge robust.

A rationale is useful for audit, but it can be fluent and wrong. Evaluate it with facts, counterexamples, permutation tests, and adversarial instructions just as one would evaluate a policy response.

---

### Step 7.5.4 — Outcome quality and process quality

A scalar response RM evaluates the completed output. In reasoning tasks, intermediate steps can be scored separately. This gives two different objects:

- an outcome reward model evaluates the final answer or whole trajectory;
- a process reward model (PRM) assigns scores to intermediate steps.

A PRM can provide denser feedback and guide search, but it creates new modeling choices:

- what counts as one step;
- how to label a locally plausible step that leads to a dead end;
- whether hidden or written reasoning is trustworthy;
- how step scores aggregate;
- whether the model learns the style of good reasoning rather than correctness.

Full PRM practice, tree search, and test-time compute belong to Module 10 “Search, verification, and test-time compute” ([open module](/courses/rl-for-llm/module-10/)). Here the essential point is that whole-answer reward and step-level reward are not interchangeable.

---

### Step 7.5.5 — A rubric is a vector and a bridge to Module 8

Open-ended answers rarely lie on one natural axis. Let a judge or a collection of models return

$$
q(x,y)=
(q_{\mathrm{correct}},q_{\mathrm{helpful}},q_{\mathrm{clear}},q_{\mathrm{safe}}).
$$

The simplest scalarization is

$$
R_w(x,y)=w^\top q(x,y).
$$

The weight vector does not discover the correct trade-off. It **defines** which trade-off is optimized. If component scales differ, a numerically wide component can dominate regardless of intended weights. Calibrate or normalize components and log them separately.

Some requirements should not be traded for usefulness. A safety condition such as

$$
q_{\mathrm{safe}}(x,y)\ge\tau
$$

may be a constraint rather than a small penalty that style can compensate.

A requirement for every response is different from a constraint on average risk across a distribution. Low expected cost can coexist with individual violations; it is not automatically a formulation of “this must never happen.” A threshold on a cost model's prediction must also be connected to the property actually being checked. Satisfying the numeric constraint exactly does not remove error in the predictor. Module 8 will therefore describe an expected-cost constraint as such, not as a guarantee about every individual outcome.

It is useful to separate two more objects. A **reward model** $r_\phi(x,y)$ estimates what the optimizer should increase. A **cost model** $c_\psi(x,y)$ estimates a violation, risk, or unwanted side effect that should remain below a limit. Writing $c=-r$ is usually too strong: usefulness and risk may depend on different features, data, and thresholds. Log and validate the two signals separately. Module 8 “RLHF and direct preference optimization” ([open module](/courses/rl-for-llm/module-08/)) writes this distinction as an expected-cost constraint, while Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” ([open module](/courses/rl-for-llm/module-13/)) turns it into an independent pre-release safety check.

Before handing an RM to an optimizer, complete a model card:

- criterion and rubric version;
- who or what produced labels;
- represented prompts and policies;
- tie handling;
- probabilistic model and loss;
- score-gauge convention;
- splits and distribution-shift slices;
- probability calibration;
- ensemble disagreement;
- independent overoptimization control;
- separately logged reward components;
- stopping conditions.

Module 8 “RLHF and direct preference optimization” ([open module](/courses/rl-for-llm/module-08/)) uses this component in two distinct ways: an explicit RM becomes the reward in RLHF, while fixed preference pairs lead to Direct Preference Optimization (DPO) without separately training an RM. The optimization formulas change, but the result still begins with the criterion and data quality established here.

---

## Module summary

A reward model is not a neural oracle. It is a statistical interface between an operational criterion and an optimization algorithm.

- BT maps scalar-score differences to pairwise preference probabilities.
- PL extends the idea to sequential listwise rankings.
- Within-prompt pairwise likelihood does not identify an absolute cross-prompt score level without additional assumptions.
- Pair accuracy should be complemented with log loss, calibration, slices, and transfer to new prompts and policies.
- Feedback-source disagreement should be measured rather than converted into a universal numerical ceiling.
- Optimization actively searches for proxy errors, so the RM must be evaluated along the optimization path and updated on fresh data.
- Ensemble disagreement is useful after score conventions or comparable margins have been defined, but does not remove shared bias.
- LLM judges, generative RMs, executable verifiers, and environment outcomes have different failure modes.
- Rubric weights are part of the objective, not neutral metadata.

The next module asks the natural follow-up: **how do preference pairs and an RM update the policy through RLHF, DPO, and related methods?**

## Sources and further reading

- [Bradley and Terry (1952), *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*](https://doi.org/10.2307/2334029) — the foundational pairwise probability model.
- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) — the canonical LLM pipeline with demonstrations, comparisons, a reward model, and PPO; it does not prescribe one universal annotation protocol.
- [Gao, Schulman, and Hilton (2022), *Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) — a controlled study in which proxy and independent target scores diverge under optimization pressure.
- [Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685) — empirical judge limitations and motivation for order-swap diagnostics.

- [Turner et al., *Introduction to PlackettLuce*](https://hturner.github.io/PlackettLuce/articles/Overview.html) — the authors' development of relative worths, sequential choice, and tie extensions; supports Steps 7.1.4–7.1.5 without making pairs from one ranking independent observations.
- [Rafailov et al., *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) — prompt-dependent reward shifts and their equivalence in a KL-regularized policy objective. Only that connection is used here; the algorithm's derivation remains in Module 8.
- [Malik et al., *RewardBench 2: Advancing Reward Model Evaluation*](https://arxiv.org/abs/2506.01937) — the main best-of-4 task, the separate Ties subset, and the scope of the reported findings. This chapter does not update the later model leaderboard.
- [Eisenstein et al., *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*](https://arxiv.org/abs/2312.09244) — ensemble diversity and shared failure under optimization pressure; disagreement is not automatically a confidence bound.
- [Bai et al., *Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073) — AI-generated preferences, preference-model training, and the subsequent RL stage in the reported protocol.
- [Mahan et al., *Generative Reward Models*](https://arxiv.org/abs/2410.12832) — trained generative evaluators and transfer in the authors' tested settings; a rationale is not by itself a correctness guarantee.

These references support specific claims; they do not replace checking the assumptions under which a claim applies. Dialogue-data engineering is extended in *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)); policy optimization begins in Module 8 “RLHF and direct preference optimization” ([open module](/courses/rl-for-llm/module-08/)); overoptimization and safety evaluation continue in Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” ([open module](/courses/rl-for-llm/module-13/)).
