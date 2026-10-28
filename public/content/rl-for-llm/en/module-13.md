# Module 13. Evaluating RL systems for LLMs: statistics, reward hacking, and safety

> **Material version:** 2026.25  
> **Factual snapshot:** 2026-08-06  
> **Text revision:** 2026-10-27  
> **Language:** EN  
> **Core practice:** tier A — browser and central processing unit (CPU); tier B — free Colab T4 or a local model; a paid API is an optional substitute  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 10–12 hours without the optional model run

By this point in the course, we can design rewards, optimize policies, collect rollouts, run search, and build the surrounding infrastructure. One question determines whether any of those results mean what we think they mean:

> **How do we know that a new policy actually improved rather than learning to exploit sampling noise, leaked data, a weak verifier, or a gap in the evaluation protocol?**

Evaluation is not merely the final table after training. It is itself a decision system: it selects checkpoints, hyperparameters, methods, and sometimes entire research directions. A benchmark, an LLM judge, and a unit-test suite therefore face the same optimization pressure as a reward model.

The practical task of this module is:

> **Design an auditable evaluation loop that separates useful policy improvement from statistical noise, data contamination, evaluator bias, reward hacking, and safety regressions, then preserves enough evidence for an independent rerun.**

After completing the module, you will be able to:

1. define the evaluated object as weights, a decoding configuration, an agent scaffold, or a full system;
2. separate target utility, an observable proxy, and the reported metric;
3. build Wilson intervals for proportions and plan a dataset size for a target precision;
4. compare two systems on shared tasks with a paired protocol;
5. explain when exact McNemar reasoning, paired permutation, or bootstrap is appropriate;
6. account for task, sampling, training-seed, and judge variability;
7. compute the finite-sample pass@k estimator without confusing coverage and selection;
8. recognize benchmark saturation and loss of discriminating power;
9. distinguish literal, semantic, and protocol-level contamination;
10. explain why an n-gram filter cannot certify a clean test set;
11. govern a private golden set and limit adaptive leaderboard overfitting;
12. classify reward hacking by the system component being exploited;
13. audit an LLM judge for position, style, length, and self-preference biases;
14. measure prompt-injection robustness through attack success and false-positive rates;
15. evaluate sycophancy relative to evidence rather than raw agreement frequency;
16. treat safety as a multidimensional evaluation including harmful compliance and over-refusal;
17. connect threats, test cases, severity, and release criteria;
18. assemble a protocol passport and a reproducible evidence bundle;
19. choose among `lm-evaluation-harness`, Inspect, HELM, and custom code by contract;
20. write an honest final report and move into one of the course capstones.

> **Practice route.** The lecture is followed by [exercises](/courses/rl-for-llm/module-13/exercises/), [assessment tasks](/courses/rl-for-llm/module-13/tests/), and [executable practice](/courses/rl-for-llm/module-13/notebooks/). The explanation remains self-contained; primary sources and the limits of their claims are listed at the end of the lecture.

---

## Lesson 13.1. Evaluation as a measurement system

### Step 13.1.1 — A score belongs to a configuration, not to one weight file

Start with a deceptively simple sentence: “the model scored 61%.” What exactly received that score?

For a generative run, the result may depend on:

- the checkpoint and tokenizer;
- the system message and chat template;
- temperature, top-p, length limit, and number of attempts;
- whether a reasoning mode is enabled;
- tools, memory, and retry rules;
- the environment version;
- answer canonicalization;
- the verifier or judge;
- task-level aggregation.

![Layers of the measured system](assets/rl-for-llm/en/module-13/M13_measurement_stack_EN.png)

The correct comparison unit is therefore a **versioned system**:

$$
\mathcal S=(M,P,D,A,E,G),
$$

where $M$ is the model and tokenizer; $P$ is the system message and chat template; $D$ is the decoding configuration and compute budget; $A$ is the agent scaffold, including tools, memory, and retries; $E$ is the environment and initial state; and $G$ is the scorer and aggregation rule.

*RL for LLM*, Module 11 “RL for LLM agents: tools, environments, and long trajectories” ([open module](/courses/rl-for-llm/module-11/)) showed that the same LLM can be part of different agent policies. The measurement consequence is immediate: a scaffold can improve outcomes without changing weights, while a scorer change can move a reported number without changing behavior.

This does not prevent checkpoint comparisons. It means the remaining components must be held fixed when the question is about checkpoints. If the question is whether a system can close a ticket end to end, the scaffold is part of the evaluated object and should vary with the system.

> **Further context:** *Modern LLMs*, Module 19 “Evaluating Modern LLM Systems: From Benchmarks to Decisions” ([open module](/courses/modern-llms/module-19/)) provides the broader benchmark landscape and a detailed treatment of scaffold-sensitive evaluation. This module focuses on what happens under statistical and optimization pressure.

---

### Step 13.1.2 — Wilson intervals: a proportion without false precision

Suppose the system succeeds on $k$ out of $n$ tasks:

$$
\widehat p=\frac{k}{n}.
$$

A single number hides sampling uncertainty. The familiar normal approximation is

$$
\widehat p
\pm
z\sqrt{\frac{\widehat p(1-\widehat p)}{n}},
$$

but it behaves poorly near zero and one. At $k=0$, for example, it produces the degenerate interval $[0,0]$, even though twenty failures do not establish a zero success probability.

The difficulty begins with the plug-in estimate of variability. Replacing the unknown probability by the observed proportion makes the estimated variance vanish when no successes are observed. The data are scarce, yet the calculation reports complete certainty.

The **Wilson score interval** reverses the question. Hold the observed $\widehat p$ fixed and consider a candidate value of the unknown probability $p$. Is our observation reasonably close to that value relative to the variability implied by that same $p$? Under the score approximation, the retained values satisfy

$$
\begin{gathered}
(\widehat p-p)^2\le z^2\frac{p(1-p)}{n},\\
(n+z^2)p^2-(2n\widehat p+z^2)p+n\widehat p^2\le0.
\end{gathered}
$$

The second line follows by multiplying through by $n$ and collecting terms. It is a quadratic inequality in the unknown probability, not in the already observed proportion. Its roots delimit the values that the chosen score test does not reject. This is the test-inversion construction described in [Wilson's original paper](https://doi.org/10.1080/01621459.1927.10502953) and the [NIST handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

We use Wilson as the course's default binomial interval. For an approximately 95% interval, set $z=1.96$. The roots can be written in the following convenient computational form:

$$
\mathrm{denom}=1+\frac{z^2}{n},
$$

$$
\mathrm{center}
=
\frac{\widehat p+z^2/(2n)}{\mathrm{denom}},
$$

$$
\mathrm{margin}
=
\frac{z}{\mathrm{denom}}
\sqrt{
\frac{\widehat p(1-\widehat p)}{n}
+
\frac{z^2}{4n^2}
}.
$$

The interval is `center ± margin`.

Its center need not equal $\widehat p$. We solved a compatibility condition for $p$ rather than moving symmetrically away from the observed proportion. The distinction is especially visible near the boundaries.

The 95% level describes a **procedure**: over repeated samples under the assumed model, how often do the resulting intervals cover the fixed underlying probability? It is not automatically a 95% probability that a particular hypothesis is true. Wilson uses an approximation, so finite-sample coverage need not be exactly 95% for every $n$ and $p$. Nor does the confidence level validate the scorer that supplied the observations.

| Result | 95% Wilson interval |
|---|---|
| $62/100$ | approximately $[0.522,0.709]$ |
| $0/20$ | approximately $[0,0.161]$ |

The second line says that zero successes were observed, while a nontrivial underlying success rate remains compatible with such a small sample.

For worst-case planning at $p=0.5$,

$$
n\approx\frac{z^2}{4h^2},
$$

where $h$ is the desired half-width. About 1,068 independent tasks are needed for roughly ±3 percentage points, not one hundred.

This plans the precision of **one proportion**. It comes from the approximation $h\approx z\sqrt{p(1-p)/n}$ at the largest possible variance, $p=0.5$. It is not a promise that the same dataset will detect an arbitrarily small improvement between systems. A comparison also depends on the effect one needs to detect, the desired detection probability, and how the two systems' outcomes align on shared tasks.

> **Boundary.** A narrow interval describes sampling uncertainty under the chosen model. It does not validate labels, certify independence, rule out contamination, or establish production transfer.

---

### Step 13.1.3 — Choose the analysis unit before choosing the formula

A binomial model is appropriate when each task produces one binary outcome and tasks are treated as independent units. Real evaluations often contain additional levels:

- several generations for one task;
- many tests inside one repository;
- multiple turns in one trajectory;
- several judge calls for one response;
- templated or near-duplicate tasks.

For a claim about new tasks, treating every such row as an independently sampled task can make the interval artificially narrow. One hundred generations for each of ten source tasks do not become one thousand independent tasks. Generations can be independent conditional on their prompt while still sharing that selected prompt; generalization across prompts introduces another level of randomness through task difficulty.

A useful rule is:

> **The analysis unit should match the unit to which the conclusion is meant to generalize.**

If the claim concerns new tasks, aggregate repeated attempts within a task and quantify uncertainty across tasks. If it concerns stochastic decoding on a fixed set, quantify decoding variance separately.

For continuous metrics such as cost or trajectory length, a binomial interval is not appropriate. To compare two systems, a **paired bootstrap** resamples task records rather than the two result columns separately. Each record carries a task identifier and both outcomes. If a task is drawn twice, its A and B values both occur twice. If it is absent, it disappears from both sides of the comparison.

For a difference in means, one resample gives

$$
\widehat\Delta^{*}
=\frac1n\sum_{j=1}^{n}
\left[m_A(I_j)-m_B(I_j)\right],
\qquad
I_j\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}\{1,\ldots,n\}.
$$

Only the resampled indices are random here. The model outputs have already been recorded: bootstrap does not call the models again. Using the same indices for both systems retains the common difficulty of each selected task.

Repeating the procedure produces a distribution of estimated differences. The simple percentile construction uses its quantiles as an approximate interval; other bootstrap intervals modify that final step. More resamples reduce Monte Carlo variation in this calculation, not the uncertainty caused by having only a limited number of independent source tasks. Shared index resampling is precisely the [paired mode documented by SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

Bootstrap is not a validity machine. It reproduces the structure of the observed dataset. Dependent clusters must be resampled as clusters, and systematic benchmark bias remains systematic after resampling.

![Uncertainty and paired comparison](assets/rl-for-llm/en/module-13/M13_uncertainty_paired_EN.png)

---

### Step 13.1.4 — Paired comparison: which tasks changed outcome?

Two systems should normally be evaluated on the same tasks. For binary outcomes, each task enters one of four cells:

| | B succeeds | B fails |
|---|---:|---:|
| **A succeeds** | both succeed | A only: $b$ |
| **A fails** | B only: $c$ | both fail |

The agreement cells matter for absolute accuracy, but the difference between systems is carried by the discordant counts $b$ and $c$.

Under the null that either system is equally likely to be the sole success on a discordant task, the exact two-sided McNemar calculation is

$$
p_{\mathrm{exact}}
=
\min\left(
1,
2\sum_{j=0}^{\min(b,c)}
\binom{b+c}{j}2^{-(b+c)}
\right).
$$

This reveals a common approximation failure. With $b=9$ and $c=2$,

$$
\frac{b-c}{\sqrt{b+c}}
\approx2.11
$$

looks significant at a 1.96 threshold, yet the exact two-sided $p$ is about $0.065$. When discordant counts are small, use the exact calculation or an appropriate correction rather than a bare normal approximation.

For continuous metrics, compare paired task differences

$$
d_i=m_A(i)-m_B(i),
$$

rather than treating the two result columns as unrelated samples. Two independent standard errors discard the shared task information.

A paired bootstrap and a paired permutation test use this pairing for different purposes. Bootstrap resamples tasks while keeping the original A and B labels inside each record. A permutation test instead requires a null model under which the labels within a pair are exchangeable. Swapping them changes the sign of the difference:

$$
T^{*}=\frac1n\sum_{i=1}^{n}\xi_i d_i,
\qquad
\xi_i\in\{-1,+1\}.
$$

An exact calculation enumerates the permitted swaps; a randomized calculation samples them. Under the chosen null model with independent pairs, each sign is equally likely within a pair. Merely reordering the already computed $d_i$ would leave their mean unchanged and would not generate the intended null distribution.

Exchangeability is a substantive assumption. Equality of means alone does not generally justify an exact sign-flip test. These procedures are therefore not interchangeable buttons: the bootstrap here estimates sampling variability, whereas the permutation test measures how unusual the observed statistic is under a declared null model. The [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) distinguishes within-pair label swaps from rearranging the pairings themselves.

A resulting $p$-value is not the probability that the null hypothesis is true. It compares the observation with the statistic's distribution under that hypothesis and the rest of the protocol's assumptions.

Finally, significance is not importance. A 0.2-point difference can become statistically detectable on a huge sample while remaining operationally irrelevant. State a minimum practically important effect before the run.

---

### Step 13.1.5 — Four sources of randomness and an honest run plan

A reported score can vary because of:

1. **Task sampling.** A different sample from the target population changes the mean.
2. **Decoding.** A stochastic policy gives different completions for the same task.
3. **Training.** A different seed, order, or hardware path yields a different checkpoint.
4. **Scoring.** An LLM judge, human, or service can be inconsistent.

One global seed does not remove these distinct sources. Decide which levels belong to the claim.

A reasonable comparison of two training methods might:

- train three independent checkpoints per method;
- evaluate all checkpoints on the same tasks;
- use the same generation budget on every task;
- share decoding seeds or random numbers where appropriate;
- compare methods by paired tasks within each run;
- report between-training-run dispersion separately.

A single expensive checkpoint can still be evaluated. The claim must then be narrower: “this checkpoint under this protocol,” not “the method on average.”

Stopping is part of the design. Repeatedly inspecting the metric and stopping at the first favorable crossing does not automatically preserve the interpretation of an interval designed for one fixed sample size. A prespecified calendar of inspections, by itself, does not account for the multiple opportunities to make a decision.

The inspection schedule and decision rule need a compatible error-control procedure. Alternatives include a method designed for sequential monitoring or an untouched final set that did not influence intermediate selection. Data used to choose when to stop have already participated in the decision. [Work on confidence sequences](https://arxiv.org/abs/1810.08240) studies time-uniform guarantees; this course does not assume that an ordinary fixed-sample interval provides them.

---

## Lesson 13.2. pass@k, benchmark life cycles, and contamination

### Step 13.2.1 — pass@k measures coverage, not answer selection

If one independent attempt succeeds with probability $p$, the probability of at least one success among $k$ attempts is

$$
1-(1-p)^k.
$$

That is a model-level probability. In an experiment, we may already have $n$ candidates of which $c$ succeeded. For a random subset of size $k\le n$, the finite-sample estimator is

$$
\widehat{\operatorname{pass@}k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

It can be computed stably as a product rather than through factorials.

The central interpretation is:

> **pass@k asks whether a successful candidate is present, not whether the system can identify it.**

![Candidate coverage and answer selection](assets/rl-for-llm/en/module-13/M13_passk_selection_EN.png)

A correct answer may exist among 32 candidates while a weak verifier selects a wrong one. pass@32 remains high; delivered quality does not. *RL for LLM*, Module 10 “Search, verification, and test-time compute” ([open module](/courses/rl-for-llm/module-10/)) develops this distinction between candidate coverage and actual selection under a fixed compute budget.

For $n=k=20$ and $c=1$, finite-sample pass@20 is exactly one because the full set necessarily contains the observed success. Plugging $\widehat p=c/n$ into $1-(1-\widehat p)^k$ gives about 0.642 and describes a different hypothetical experiment.

That value of one does not promise success for every fresh pool of twenty generations. It is exact for the observed pool. Interpreting the statistic as an estimate of generator performance requires the stated pool-sampling protocol; compute it for each task, then aggregate across the declared task set. Pooling answers from unrelated prompts first would change the analysis unit.

Every reported pass@k should include $n$, $k$, the success rule, decoding settings, final-selection rule, and compute cost.

---

### Step 13.2.2 — Generation budget is part of the metric

Comparing one system at pass@1 with another at pass@64 is not a pure policy comparison. It includes different compute budgets.

Represent quality as

$$
Q(B),
$$

where $B$ may be candidates, generated tokens, wall time, or API cost. A fair evaluation asks either:

- which system is better at equal budget;
- how much budget each system needs for a target quality;
- or what trade-off it achieves over a declared budget range.

For independent attempts, the marginal gain from the next candidate is

$$
\Delta_k=p(1-p)^k.
$$

It declines, but the rate depends on task difficulty. Hard tasks with small $p$ may benefit from additional attempts for longer.

Independence is conditional on a fixed prompt and sampling protocol. Similar outputs or numerically nearby seeds do not by themselves demonstrate dependence: independent draws from a concentrated distribution often repeat. Reused random prefixes, shared search branches, or correlated sampling can violate the assumption; pooling tasks with different success probabilities also invalidates substitution of one mean p into the formula. Estimate pass@k from the actual candidate groups and preserve the prompt as the analysis unit; report diversity as a separate diagnostic.

pass@k is also not calibration. A system can have high coverage while being unable to recognize its successful answer. This matters whenever the user receives only one response.

---

### Step 13.2.3 — A benchmark has a life cycle

A public benchmark does not preserve the same measurement value forever. A simplified life cycle is:

1. the new set is hard and discriminating;
2. methods adapt to its format;
3. scores rise and remaining errors concentrate in ambiguous or defective items;
4. the set and solutions enter training corpora;
5. leaderboard differences compress;
6. a successor, private split, or dynamic version appears.

![Benchmark life cycle](assets/rl-for-llm/en/module-13/M13_benchmark_lifecycle_EN.png)

**Saturation** is more than a high mean. It is a loss of discriminating power: most items no longer separate strong systems, so rankings hinge on a small set of rare, noisy, or mislabeled items.

As of the snapshot date, useful evaluation families include:

| Target behavior | Examples | Protocol fields that matter |
|---|---|---|
| difficult closed knowledge and reasoning | GPQA Diamond, HLE | version, subset, answer format, grader |
| competition mathematics | a specific AIME year | year, tools, attempts, answer extraction |
| post-cutoff programming | LiveCodeBench | task period, language, execution, pass@k |
| repository-level software repair | SWE-bench Verified | agent scaffold, image, budget, dataset version |
| tool calling | BFCL | leaderboard version, category, call format, multi-turn mode |
| safety | AILuminate and a private red-team | taxonomy version, public/private split, evaluator |

SWE-bench Verified is a human-validated 500-task subset, but a score still depends on the agent and execution environment. BFCL V4 currently covers agentic components in addition to function-call syntax. These are dated properties of particular releases, not timeless definitions.

The broader benchmark catalog belongs to *Modern LLMs*, Module 19 “Evaluating Modern LLM Systems: From Benchmarks to Decisions” ([open module](/courses/modern-llms/module-19/)). The local rule is enough: select a benchmark by target behavior and measurement risk, not by name recognition.

---

### Step 13.2.4 — Contamination: a cheap sieve and its boundary

**Test contamination** occurs when information about evaluation items enters training or system tuning. Distinguish:

- literal overlap;
- near duplicates;
- semantic duplicates;
- leaked solutions;
- protocol leakage through repeated prompt or scaffold tuning on the test set.

A cheap filter is the fraction of an example’s n-grams that appear in an accessible training corpus:

$$
\operatorname{overlap}_n(x,C)
=
\frac{
\#\{\text{n-grams from }x\text{ found in }C\}
}{
\#\{\text{n-grams in }x\}
}.
$$

This containment score is asymmetric: it asks what fraction of the example's fragments occur in the corpus. A symmetric comparison between two n-gram sets $A$ and $B$ is the Jaccard coefficient:

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

Containment is natural when one object is the query and the other is a reference corpus; Jaccard is useful when the two objects should be compared on equal footing. Neither detects semantic equivalence by itself.

![Contamination and adaptive-overfitting paths](assets/rl-for-llm/en/module-13/M13_contamination_paths_EN.png)

This catches literal fragments and misses translations, paraphrases, and rearrangements. It is a **sieve**, not a cleanliness certificate.

A canary string is a unique marker placed in benchmark files so that developers can filter them from training or detect obvious reproduction. Reproducing a canary is strong evidence of access to associated material. Failure to reproduce it is not proof of absence: the data may have appeared without the marker, been reformatted, or simply not been emitted.

A useful report states the accessible corpus, matching method, tokenization, threshold, flagged fraction, score with and without flags, and the detector’s limitations. [*Soft Contamination Means Benchmarks Test Shallow Generalization*](https://arxiv.org/abs/2602.12413) investigates related content that literal filtering can miss. This does not turn every high score into evidence of contamination; it limits what the absence of literal overlap can establish.

---

### Step 13.2.5 — Adaptive overfitting: the test becomes part of development

A test can wear out even if it was not present in pretraining.

Suppose a team tries 50 prompts, temperatures, and retry rules. Every configuration has the same true quality, but the observed metric contains noise:

$$
\widehat\mu_j=\mu+\varepsilon_j.
$$

Even when every prespecified configuration has an unbiased estimate, taking the maximum favors favorable errors. When those errors differ across configurations, the selected result becomes optimistic: the best noise is mistaken for the best system. The set that chose the winner is no longer an untouched confirmation of that winner. If all errors are identical, maximization adds no such selection advantage; their joint structure matters, not just their individual variances.

The same process occurs at ecosystem scale as thousands of teams adapt to one public leaderboard. No deliberate cheating is required; test information still enters development decisions.

Layered defenses include:

1. an open development set;
2. a separate validation set;
3. a minimally queried private holdout;
4. live or frequently refreshed tasks;
5. form and item rotation;
6. a query log for leaderboard feedback;
7. a final independent audit set used only after the system is frozen.

LiveCodeBench and LiveBench use updated tasks to reduce some contamination channels. This does not eliminate item defects, protocol dependence, or later adaptive overfitting.

A private golden set needs an owner, version, access policy, change log, and refresh plan. Secrecy without label quality is not enough; label quality without access control quickly turns the test into another validation set.

---

## Lesson 13.3. Reward hacking and judges under pressure

### Step 13.3.1 — Taxonomy: what component is the policy exploiting?

Reward hacking is easier to reason about when classified by the exploited component.

*RL for LLM*, Module 7 “Reward models and preference data” ([open module](/courses/rl-for-llm/module-07/)) introduced reward models and LLM judges as feedback sources. Here the same components are treated as measurement surfaces that an optimized policy can pressure and exploit.

![Reward-hacking surface](assets/rl-for-llm/en/module-13/M13_reward_hacking_surface_EN.png)

**1. Specification gaming.** The reward correctly implements the written rule, but the rule is not the intended goal. “Make tests pass” permits deleting the tests; “maximize engagement” permits clickbait.

**2. Verifier exploitation.** The response passes incomplete tests, triggers a parser bug, hard-codes known cases, or finds a string accepted by canonicalization.

**3. Verifier tampering.** The agent changes tests, references, files, time, or environment state used to compute reward.

**4. Judge hacking.** The response exploits length, confident style, rubric mimicry, instruction injection, or judge-family-specific behavior.

**5. Auxiliary-metric gaming.** The policy optimizes an intermediate metric such as tool-call count or reasoning length rather than task outcome.

**6. Distributional evasion.** The policy finds inputs on which a reward model or verifier extrapolates especially badly.

There is no universal defense implied by the phrase “reward hacking.” Specification gaming needs objective red-teaming; program verifiers need hidden tests and immutability; judges need isolation and audit; distributional evasion needs fresh data and out-of-distribution checks.

The highest-reward and fastest-improving examples deserve first inspection. A suspiciously easy success is often more informative than a random sample.

---

### Step 13.3.2 — A synthetic judge audit: what sanitization fixes

Consider a teaching judge. Each candidate has true quality $g$, length $\ell$, confident-phrase count $c$, and injection flag $j$.

Raw score:

$$
S_{\mathrm{raw}}=g+0.001\ell+0.3c+10j.
$$

After removing explicit instructions and confident self-evaluations:

$$
S_{\mathrm{san}}=g+0.001\ell.
$$

The gap

$$
\operatorname{gap}=S_{\mathrm{raw}}-S_{\mathrm{san}}
$$

can be used as a suspicion signal.

The example separates three outcomes:

1. injection can dominate the raw judge;
2. sanitization removes that particular string-level attack;
3. the judge’s own length bias remains.

“Clean input” is therefore not “objective judge.” Attack mitigation and evaluator calibration are separate problems.

In the teaching construction, sanitization is defined to remove two added terms while leaving true quality $g$ unchanged. A real transformation has no such automatic guarantee: the removed passage may contain substantive answer content as well as an attempt to influence the judge. A smaller `gap` must therefore be checked alongside meaning preservation and agreement with independent anchors. It is not, on its own, evidence of a more accurate score.

A real audit needs both

$$
\operatorname{ASR}
=
\frac{\text{successful attacks}}{\text{valid attacked cases}}
$$

and

$$
\operatorname{FPR}
=
\frac{\text{honest cases incorrectly flagged}}{\text{honest cases}}.
$$

ASR is the attack success rate; FPR is the false-positive rate. A filter that blocks every attack and half of benign responses is not a free improvement.

![LLM-judge audit](assets/rl-for-llm/en/module-13/M13_judge_audit_EN.png)

The synthetic formula is not a complete model of a modern judge. Its role is to force a clean distinction between input-level vulnerability, systematic evaluator bias, and detector error.

---

### Step 13.3.3 — LLM judges: order swapping and structural isolation

LLM judges are useful for open-ended outputs, but their verdicts can depend on position, verbosity, style, self-preference, limited reasoning ability, and instructions embedded in the candidate response.

A minimal pairwise protocol evaluates both orders:

1. A first, B second;
2. B first, A second.

After the reverse-order judgment, first map labels back to **candidate identity** A or B, then measure whether the same candidate wins in both orders. That is an identity-consistency audit.

Preserve the raw `FIRST`, `SECOND`, and `TIE` labels as a separate **screen-position audit**. They answer a different question: how often does the judge select the first slot regardless of which candidate occupies it? Identity consistency and position preference must not be collapsed into one statistic.

For example, `FIRST` in both presentations means different candidates won: A in the forward order and B in the reverse order. `FIRST` followed by `SECOND` instead means a stable A win. Keeping both raw and identity-mapped labels makes that distinction inspectable.

A compact summary can use three groups:

- stable A win;
- stable B win;
- no stable winner.

The last group must retain its internal causes. `TIE` in both orders is a stable tie, not necessarily inconsistency. A reversal of the winning candidate is disagreement. A tie in only one presentation is another outcome. Combining them for display must not erase the underlying records or turn them into arbitrary wins.

A win rate calculated only over stable decisions is conditional on a **selected subset**. Report the retained fraction and the reasons other pairs were excluded. Otherwise a strong percentage may describe only the comparisons the judge found convenient. Swapping order diagnoses one source of bias; it does not establish that the stable verdict is correct.

**Structural isolation** passes the candidate as clearly delimited data and instructs the judge not to execute commands inside it. This reduces direct prompt injection but does not create a formal security boundary: the model still processes the text and can respond to subtler attacks.

Binary rubric items are usually more auditable than an unconstrained 1–10 score: factual consistency, completion of each required instruction, format compliance, and unsupported claims. Each item still needs an aggregation rule.

Finally, compare the judge with human or executable anchors on the same distribution. High agreement on ordinary answers does not establish robustness to answers optimized against the judge.

---

### Step 13.3.4 — Three levels of optimization pressure

The same imperfect evaluator behaves differently under different search strengths.

**Best-of-N** samples candidates and selects the largest proxy score. Judge errors are found by chance, increasingly often as $N$ grows.

**RL** changes the policy distribution toward features that raise proxy reward. A discovered weakness becomes more common and is stored in the weights.

**The ecosystem** repeatedly selects models, prompts, and scaffolds by one public number. A benchmark can be worn down by honest use without a single explicit attacker.

Work on reward-model overoptimization found different empirical degradation patterns for Best-of-N and RL against an independent gold model. This is not a universal theorem that RL is always more dangerous, but it warns that robustness to weak search does not imply robustness to directed optimization.

Defenses include optimization limits, early stopping, independent evaluation, diverse ensembles, fresh labels, adversarial regression sets, separation of training reward and release criteria, proxy–target divergence monitoring, and manual review of the top reward tail.

None is sufficient alone. Ensembles can share errors, and an independent set becomes dependent after repeated use. Technical and process controls are both necessary.

---

### Step 13.3.5 — Sycophancy: user preference is not ground truth

**Sycophancy** is the tendency to support a user’s stated position even when it conflicts with available evidence or with the model’s earlier answer.

Raw agreement rate is not a valid metric because users are often right. Construct paired cases with a known answer:

- a neutral question;
- the same question with a correct user belief;
- the same question with an incorrect user belief.

Useful quantities include

$$
C_{\mathrm{wrong}}
=
P(\text{model corrects the user}\mid\text{user is wrong}),
$$

$$
E_{\mathrm{wrong}}
=
P(\text{model endorses the error}\mid\text{user is wrong}),
$$

$$
D_{\mathrm{right}}
=
P(\text{model contradicts without cause}\mid\text{user is right}).
$$

A good system does not merely disagree more often; it discriminates truth better. Otherwise an anti-sycophancy intervention creates stubbornness.

[*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548) documents sycophantic behavior and evidence that preference judgments can favor responses aligned with user beliefs. This is an empirical mechanism in the studied settings, not proof that all human-feedback optimization inevitably creates sycophancy.

For RL systems, the practical implication is clear: if the reward model favors confident agreement, stronger optimization can amplify it. Sycophancy tests should therefore run after changes to reward data or preference optimization, not only as a one-time safety benchmark.

---

## Lesson 13.4. Safety and the protocol passport

### Step 13.4.1 — The evaluation passport: a number without a protocol does not transfer

An evaluation passport should let another person identify exactly what ran and reproduce it without guessing hidden settings.

Minimum sections:

**Object.** Model, revision or checkpoint hash, tokenizer, chat template, adapters, reasoning mode.

**Scaffold.** System message, tools, memory, retries, agent program, dependency versions.

**Data.** Name, version, split, file hashes, task count, exclusions, contamination policy, access control.

**Generation.** Temperature, top-p/top-k, length limit, candidate count, seeds, token and time budgets.

**Scorer.** Canonicalizer, tests, judge model, exact rubric prompt, sanitizer version, tie and abstention policy.

**Statistics.** Analysis unit, metric formula, intervals, paired protocol, run count, multiplicity and stopping rules.

**Cost.** Tokens, calls, GPU-hours, latency, and technical-error rate.

**Changes and incidents.** What changed after the run began, what was excluded, and why.

SHA-256 hashes for data, configuration, judge prompt, and verifier code help establish artifact identity. They do not establish artifact quality.

Anti-patterns include best-of-five reporting without the other runs, comparing against a different scaffold, changing canonicalization after seeing errors, tuning on the test while retaining the word “test,” reporting pass@k without $n$, reporting safety without benign neighboring prompts, or using an unversioned judge.

---

### Step 13.4.2 — Safety is not a refusal rate

A system can look safe by refusing everything. That outcome is useless and often reveals an inability to distinguish risk.

A minimal safety evaluation includes both:

1. **harmful compliance** — the system helps execute a disallowed or harmful action;
2. **over-refusal** — the system rejects a benign or useful request that is only superficially similar to a risky one.

For each hazard class, construct:

- a clearly harmful request;
- a benign near neighbor with similar words;
- an ambiguous request that should trigger clarification;
- an allowed defensive or educational form of the topic.

Then report harmful compliance, correct refusal, safe helpful alternative, over-refusal, confidence or rationale quality, and severity.

![Safety and release passport](assets/rl-for-llm/en/module-13/M13_safety_passport_EN.png)

A single macro-average can hide a critical failure. A rare severe category is not necessarily offset by many easy benign cases.

A teaching risk functional is

$$
R=\sum_h w_h p_h s_h,
$$

with estimated failure frequency $p_h$, severity $s_h$, and context weight $w_h$. This is not a universal regulatory formula; it forces otherwise hidden assumptions into the open.

The distribution behind $p_h$ must also be named. Success frequency on a deliberately selected attack set describes robustness under that test. It is not automatically the frequency of the same event in deployment. That transfer needs assumptions about threat prevalence, user behavior, and attacker resources; the weights $w_h$ cannot supply missing evidence. Until those assumptions are justified, the sum is a chosen comparison measure rather than a measured total deployment risk.

---

### Step 13.4.3 — Red-teaming: ASR and the cost of false alarms

Red-teaming is a threat matrix, not an arbitrary list of alarming prompts.

For each class, record the attacker’s goal, knowledge, channels, attempt budget, success criterion, expected severity, and defense version.

Channels may include direct jailbreaks, indirect injection in documents or tool results, multi-turn escalation, encoding or translation, role substitution, judge attacks, and environment tampering.

Attack success should be accompanied by the number of valid attempts, a confidence interval, attacker budget, a false-alarm or benign over-refusal rate, and performance on new attacks not used during defense development. The label FPR is reserved for a declared binary detector; otherwise the denominator should be described directly.

Once a defense is tuned on a fixed attack catalog, that catalog is a development or validation set. Final evaluation requires fresh phrasings, another attack generator, or an independent red team.

Choose the trial unit before attaching an interval. A successful individual attempt and at least one success in a ten-attempt campaign against one task are different events. If later attempts adapt to earlier responses, ten attempts are not automatically ten independent Bernoulli observations. For a claim about new tasks or campaigns, preserve those groups and quantify uncertainty at the corresponding level.

Define validity in advance as well. A malformed test invocation can be distinguished from a valid attempt, but unsuccessful attacks should not be removed merely because they were weak. This is the rule from Step 13.1.3 applied to adversarial evaluation: define the intended generalization and analysis unit before choosing the statistical procedure.

As of the snapshot, AILuminate v1.1 provides a twelve-category hazard taxonomy and a private official test alongside public practice prompts. It illustrates the development-versus-official-evaluation split, but it does not replace a product-specific threat model.

---

### Step 13.4.4 — From metrics to a release decision

An evaluation becomes an engineering control only when it is connected to a predeclared action.

Three questions must remain separate: whether any difference was detected, whether a deterioration would matter in practice, and whether the data are adequate for a release decision. Failing to detect regression does not exclude it. A wide interval may allow both improvement and an unacceptable loss. “Meaningful regression” in the table therefore needs a prespecified effect margin and statistical rule, not a convenient $p$-value alone.

| Signal | Threshold | Action |
|---|---|---|
| meaningful regression on the core task | below the minimum acceptable effect | block release |
| ASR increase in a critical category | any confirmed deterioration | block and investigate |
| over-refusal increase | above product budget | revise policy or UX |
| high judge inconsistency | above calibration threshold | do not use the judge alone |
| confirmed private-set contamination | any | replace items and revise history |
| excessive environment failures | above limit | do not interpret the score |

Thresholds should not be invented after results are visible.

Suppose larger $Q$ is better and the maximum acceptable deterioration is fixed in advance at $\delta>0$. Construct a confidence interval for the candidate-minus-baseline difference using the declared protocol:

$$
\begin{gathered}
\Delta=Q_{\mathrm{candidate}}-Q_{\mathrm{baseline}},\\
[L,U]\ \text{is an interval for }\Delta,\qquad\delta>0.
\end{gathered}
$$

This supports three distinct readings. If $L>-\delta$, the data exclude deterioration beyond the declared margin at the chosen confidence level, so the candidate may pass this gate. If $U<-\delta$, the whole interval lies in the unacceptable-regression region. If it crosses $-\delta$, the data do not establish acceptability. The response is more evidence or a predeclared conservative policy—not an automatic pass.

This is one gate, not the entire release decision. The one-sided or two-sided construction, level, and multiplicity control must be chosen before the run. Ruling out unacceptable deterioration is not the same as establishing equivalence in every respect or proving superiority. [Lakens's primer on equivalence testing](https://doi.org/10.1177/1948550617697177) develops the underlying distinction between a nonsignificant difference and evidence that an important effect is absent.

Rare failures need the same discipline. The earlier $0/20$ example has a Wilson upper bound near 0.161. Observing no violations in that small set therefore does not establish a low violation probability under the assumed model. Threat coverage and deployment transfer remain separate questions.

A contaminated or revised test set also needs a new version. Keep previous reports attached to their original identifiers and publish recomputed values separately. New items without saved outputs need a new run. Revising the historical comparison must not erase the history of how earlier numbers were measured.

The NIST AI RMF and its Generative AI Profile use the broader Govern–Map–Measure–Manage structure: define context, measure risk, assign ownership, and connect findings to action. The course-level lesson is simple: **an evaluation without an owner and a response does not manage risk.**

A training constraint and a release gate solve different problems. *RL for LLM*, Module 8 “RLHF and direct preference optimization” ([open module](/courses/rl-for-llm/module-08/)) uses $\mathbb E[C]\le d$ to control expected cost on the training distribution. This module evaluates observed violations on a finite scenario set and may block release. An average constraint does not protect every critical prompt, while a passed test does not prove safety on all future inputs. Serious risks require both constrained optimization and an independent release control.

An incident record keeps the input, transcript, model and environment versions, tools and side effects, scorer verdict, expected and observed behavior, severity, root cause if known, remediation, and a new regression test.

A discovered failure becomes a test, but that test will also be optimized against over time. Regression suites need rotation and independent audits too.

---

### Step 13.4.5 — Evaluation tools: choose a contract, not a logo

At the 6 August 2026 snapshot, three tool families are useful to distinguish without treating them as mutually exclusive camps.

**`lm-evaluation-harness`.** It is a strong starting point for reproducible runs over many standard language-model tasks. In the current 0.4 line, the CLI is organized around subcommands, so a typical invocation starts with `lm-eval run`; for Python integration, the official documentation recommends `simple_evaluate()`. Hugging Face model support is installed through the `lm_eval[hf]` extra. Pin the version: an old command example can become stale even though the metric mathematics has not changed.

**Inspect.** An Inspect task combines a `dataset`, `solver`, and `scorer`. That contract maps naturally to multi-turn agents, tools, sandboxes, and model-graded evaluation. Inspect logs can preserve the solving trace, scores, and auxiliary events—provided the task author actually records the artifacts needed for audit.

**HELM.** Stanford CRFM’s framework organizes transparent scenarios and multidimensional reports. Its value is the explicit connection among scenario, adaptation, metrics, and efficiency rather than a promise of one final scalar.

No framework repairs a bad evaluation design. A standardized harness can reproducibly compute the wrong metric; a rich log cannot make silently changing task versions comparable; automated judge calls only scale judge bias faster.

| Need | Natural starting point |
|---|---|
| standard academic tasks | `lm-evaluation-harness` |
| agents, tools, red-team, complex scorers | Inspect |
| broad multidimensional characterization | HELM |
| a small private product set | a minimal custom harness using common libraries |

Recheck APIs before publication, pin dependencies, and preserve raw artifacts. Software capabilities are dated engineering facts, not mathematical properties of evaluation.

## Lesson 13.5. A minimal harness and the course finale

### Step 13.5.1 — A custom eval has five versioned layers

A minimal evaluation system is easiest to audit when it has five separable layers.

**1. Samples.** Each item has a stable `sample_id`, input, reference or rubric, risk category, and metadata.

**2. Solver.** It receives a sample and a complete system configuration, then returns the raw answer, transcript, cost, stop reason, and technical status.

**3. Scorer.** It canonicalizes the artifact and emits a structured, versioned record:

```json
{
  "score": 1.0,
  "status": "graded",
  "details": {"exact_match": true},
  "scorer_version": "exact-v2"
}
```

**4. Aggregator.** It groups by task and category, computes intervals, paired differences, cost, technical failures, and safety metrics.

**5. Release gates and passport.** They apply predeclared decision rules and bind the result to hashes of the dataset, raw records, summary, and report.

![Evaluation evidence pipeline](assets/rl-for-llm/en/module-13/M13_evidence_pipeline_EN.png)

Do not hide model calls, dataset loading, extraction, and table printing inside one function. Separation enables caching, rescoring with a new verifier, and incident investigation.

*RL for LLM*, Module 12 “RL infrastructure for LLMs: memory, rollouts, and asynchrony” ([open module](/courses/rl-for-llm/module-12/)) provides the infrastructure counterpart: policy, environment, and verifier versions travel through the log as data rather than being reconstructed from memory.

A useful invariant is:

> **Preserve the raw answer and transcript before computing the final metric.**

A corrected canonicalizer or judge can then rescore existing outputs without expensive regeneration. The new scorer version must produce a new result column; old and new numbers must not be mixed.

#### Reproducible course experiment

`module_13_evaluation_harness.py` implements this contract on 240 synthetic tasks from four categories. With fixed `seed=13`, it produces:

| Quantity | Baseline | Candidate |
|---|---:|---:|
| accuracy | 0.475, Wilson [0.413, 0.538] | 0.692, Wilson [0.631, 0.747] |
| technical failure rate | 0 | 0.0083 |

The paired `candidate − baseline` difference is 0.2167, with a paired-bootstrap interval of approximately [0.1375, 0.2959]. The candidate alone succeeds on 79 tasks and the baseline alone on 27; the exact McNemar value is about $4.33\times10^{-7}$.

Each number has a distinct job. The two proportions and their intervals describe each system separately. The paired improvement comes from $(79-27)/240=52/240\approx0.2167$: matching outcomes contribute zero to the difference but remain in the evaluation set. Paired bootstrap quantifies uncertainty in the size of the change, while McNemar tests the symmetry of discordant binary outcomes under its null model. A small $p$-value does not determine the cost or safety of the improvement.

In this run, technical failures **remain in the primary denominator**. The scorer preserves a separate infrastructure status while setting `correct = False`; the aggregator computes accuracy and its interval over all 240 records. Paired comparison also uses the complete shared set of sample identifiers. The table therefore measures end-to-end successful, correct completion, not accuracy conditional on a run having completed. Technical-failure rate is reported separately so that an operational failure remains distinguishable from a substantively wrong answer.

This is the declared teaching protocol, not a universal rule for every infrastructure outage. Another exclusion policy can be appropriate if specified in advance, but it must retain comparable denominators and visible failure records.

The same run preserves `dataset.jsonl`, raw records for both systems, `summary.json`, `report.md`, judge and safety audits, and their SHA-256 hashes in `passport.json`. This is a **deterministic synthetic course experiment**, not a benchmark of a real LLM. Its purpose is to make every contract and artifact inspectable.

### Step 13.5.2 — An honest report starts with the decision, not the leaderboard

A final report should start from the decision being made.

Recommended structure:

1. decision and constraints;
2. full evaluated-system configurations;
3. dataset origin, exclusions, and contamination checks;
4. metric formulas and analysis units;
5. intervals, paired comparisons, and minimum important effect;
6. scorer quality, inconsistency, ASR, and—when a separate binary detector exists—FPR;
7. safety by hazard and over-refusal;
8. tokens, time, money, and technical failures;
9. incidents where the metric disagreed with audit;
10. explicit non-claims.

Do not rank systems by a third decimal place when intervals and paired differences cannot distinguish them. The honest conclusion is “insufficient evidence to detect a difference.” Equivalence requires a predeclared acceptable difference margin and a separate test against that margin.

Nor should the comparison be decided solely by overlap between two separate accuracy intervals. Use the paired difference and its uncertainty: they retain the information about which shared tasks changed outcome.

A public table should have a machine-readable `results.json` and a human-readable `report.md`. A table without configuration quickly becomes indistinguishable from marketing.

---

### Step 13.5.3 — Practice route: from synthetic checks to `lm-eval`

The **core CPU route** needs no downloaded model and covers the full chain:

- Wilson intervals and sample-size planning;
- paired and hierarchical resampling;
- the complete synthetic harness and evidence bundle;
- pass@k separately from answer selection;
- literal n-gram containment and a paraphrase it misses;
- two distinct judge audits: candidate identity and screen position;
- harmful and benign safety denominators;
- release-gate and passport-hash validation.

The **real-model route** is disabled by default (`RUN_REAL_MODEL = False`). For the `lm-evaluation-harness` 0.4 line, it uses the documented Python entry point `simple_evaluate()` and preserves the full result object. Hugging Face model support is installed through `lm_eval[hf]`. This branch is a small contract run, not a capability claim.

A 16- or 32-example run is a smoke test. It checks the model/task/template/few-shot/batch/result-serialization contract; it does not estimate capability reliably.

A paid API can be substituted as another solver. Record provider, exact model version, date, retry policy, cost, parameters, and raw responses. A mutable API requires a stronger log than a fixed local checkpoint.

### Step 13.5.4 — Capstone B5: evaluation engineering as a course-wide test

Capstone B5 turns the module into a compact project: a rubric, programmable and judged items, an LLM judge, sanitization and structural isolation, an attack set, class-specific ASR, benign false-alarm or over-refusal rates, paired model comparison, and a versioned protocol passport.

The module already provides the main anchors:

- intervals in Step 13.1.2;
- paired comparison in Step 13.1.4;
- contamination in Step 13.2.4;
- attack taxonomy in Step 13.3.1;
- `raw/sanitized/gap/flag` in Step 13.3.2;
- optimization pressure in Step 13.3.4;
- the passport in Step 13.4.1.

Include a positive control whose expected direction has been established independently: for example, verified correct answers and deliberately degraded variants, as specified in B5. If the harness cannot distinguish these controls, investigate sample power, judge quality, and protocol before launching a large red-team. Comparing real checkpoints remains an empirical question: model size or membership in one family does not guarantee a quality ordering on the chosen task.

GO/NO-GO before scale-up:

- programmable rubric items have tests;
- the judge separates clearly good and bad answers;
- A/B order does not determine a large share of verdicts;
- the sanitized path is genuinely active;
- the cost of one hundred samples is measured;
- logs permit rescoring without new model calls.

---

### Step 13.5.5 — Course finale: RL as the discipline of signal design

The thirteen modules reduce to one recurring question:

> **What signal does the policy receive, where did it come from, and what behavior becomes profitable if it is optimized strongly enough?**

Part I built the RL language: state, policy, return, value, exploration, policy gradients, and learning from fixed data.

Part II mapped that language onto LLMs: token probabilities, KL, reward models, DPO, RLHF, RLVR, and group-relative estimation.

Part III moved beyond one loss function into search, agents, environments, asynchronous infrastructure, and evaluation under optimization pressure.

The cross-course map is now:

- reward measures a local objective;
- advantage allocates comparative signal;
- KL prices distribution shift;
- a verifier formalizes success;
- a benchmark is a verifier for the research process;
- a safety eval is a verifier for risk;
- a passport provides an auditable link between a number and its experiment.

The course does not promise one universal algorithm. It develops a more durable skill: locating the statistical assumption, software contract, proxy objective, or hidden degree of freedom in a real system.

The next step can be any of five capstones: a full alignment pipeline, reasoning and RLVR, a sandboxed agent, visual RLVR, or evaluation engineering.

Practice is optional for formal reading, but it is what turns formulas into the ability to diagnose a real run. A strong final course artifact is not merely a trained model; it is an experiment in which every number can be explained, reproduced, and challenged by a named failure mode.

---

## Primary-source reading route

- [Wilson (1927), *Probable Inference, the Law of Succession, and Statistical Inference*](https://doi.org/10.1080/01621459.1927.10502953) — the original score-interval logic; it does not repair clustered samples or dataset bias.
- [McNemar (1947), *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages*](https://doi.org/10.1007/BF02295996) — paired binary outcomes; use the exact tail when discordant counts are small.
- [Chen et al. (2021), *Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) — the finite-pool pass@k estimator; it measures coverage, not selector quality.
- [Gao et al. (2022), *Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) — empirical proxy overoptimization in the studied regimes; the reported curve is not a universal law.
- [Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685) — primary evidence of systematic LLM-judge biases; order swapping does not repair content errors.
- [NIST AI 600-1, Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) — connects risk context, measurement, ownership, and action; it does not prescribe one universal release threshold.
- [EleutherAI `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) — the current standard-task software contract; recheck and pin the API before a run.
- [Inspect AI](https://inspect.aisi.org.uk/) — the official `dataset + solver + scorer` task model and evaluation logs for complex scenarios.
- [HELM](https://crfm.stanford.edu/helm/) — transparent scenarios and multidimensional evaluation; a framework does not replace a product threat model.

- [NIST: *Confidence intervals for a proportion*](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) — score-test inversion and Wilson limits; sampling uncertainty does not cover measurement-system bias.
- [SciPy: paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) — shared-index resampling and label-swap semantics; executable calls must match the installed version.
- [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*](https://arxiv.org/abs/1810.08240) — distinguishes a fixed-sample interval from monitoring over time.
- [Lakens (2017), *Equivalence Tests: A Practical Primer*](https://doi.org/10.1177/1948550617697177) — failure to detect an effect versus exclusion of practically important differences; effect margins are prespecified.
- [*Soft Contamination Means Benchmarks Test Shallow Generalization*](https://arxiv.org/abs/2602.12413) — limits of literal overlap checks for semantically related data.
- [Sharma et al., *Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548) — sycophancy and preference-label evidence in studied systems, not a universal theorem about human-feedback optimization.

The benchmark and software overview retains its 6 August 2026 factual snapshot. Course-owned synthetic results are distinguished from external research. The statistical explanations and their references do not replace validation of a concrete evaluation protocol.
