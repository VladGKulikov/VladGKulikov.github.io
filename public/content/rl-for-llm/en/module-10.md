# Module 10. Search, verification, and test-time compute

> **Material version:** 2026.22  
> **Explanatory text revised:** 2026-10-27  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Core practice:** level A — browser and CPU; level B — free Colab T4 or an optional paid API route  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 9–11 hours without the optional model run

Module 9 “RLVR and GRPO: verifiable rewards, group-relative estimates, and stability” ([open module](/courses/rl-for-llm/module-09/)) changed the policy parameters using fresh rollouts. This module freezes the weights and asks a different question: **how can we spend more compute on one prompt, obtain a better answer, and avoid confusing more attempts with a better selected output?**

Additional compute can produce independent candidates, lengthen one trajectory, expand a search tree, or be allocated adaptively across prompts. Every design contains the same pipeline:

$$
\text{generator}
\longrightarrow
\text{candidates or search states}
\longrightarrow
\text{selection rule}
\longrightarrow
\text{final answer}.
$$

The generator determines which solutions are reachable. The selector determines whether a good solution can be recognized. The budget determines how many opportunities can be inspected. If any one of these components is weak, more tokens or candidates may fail to help and can even reduce deployed quality.

The practical question of the module is:

> **Design and validate a test-time scaling system while measuring coverage, selection quality, cost, and robustness to verifier error separately.**

After completing the module, you will be able to:

1. distinguish parallel, sequential, search-based, and adaptive test-time scaling;
2. derive pass@N and state its assumptions;
3. compute the finite-sample pass@k estimator;
4. explain when self-consistency helps and when it amplifies a systematic error;
5. separate the existence of a correct candidate from the ability to select it;
6. analyze Best-of-N with exact and imperfect verification;
7. derive the limit of a fully specified binary-verifier protocol;
8. recognize proxy overoptimization as the candidate pool grows;
9. distinguish outcome and process scoring and compare step aggregations;
10. design beam search, best-first search, and Monte Carlo tree search without assigning them universal guarantees;
11. allocate a finite budget by marginal gain;
12. compare fixed training cost with higher per-query inference cost;
13. run deterministic CPU experiments and prepare an optional small open-model demonstration.

> **Practice route.** [Exercises](/courses/rl-for-llm/module-10/exercises/), [assessment questions](/courses/rl-for-llm/module-10/tests/), and [executable practice](/downloads/rl-for-llm/notebooks/module-10-test-time-search.en.ipynb) remain separate. The lecture can be read continuously without opening them.

---

## Lesson 10.1. What scales at generation time

### Step 10.1.1 — Frozen weights, a different answer algorithm

**Test-time compute** is extra computation performed after training for a particular prompt. The model parameters may remain fixed while the algorithm that produces the answer becomes more elaborate than one greedy decoding pass.

Four mechanisms should be kept distinct.

1. **Parallel compute:** generate several candidates and aggregate or select one.
2. **Sequential compute:** allow one trajectory more tokens, checks, or revisions.
3. **Search compute:** branch, score intermediate states, and revisit promising branches.
4. **Adaptive compute:** assign different budgets to different prompts.

![Four test-time compute mechanisms](assets/rl-for-llm/en/module-10/M10_compute_axes_EN.png)

The mechanisms can be combined. A system may generate four long trajectories, score each prefix, and continue only the two most promising branches.

More compute is not an objective by itself. A valid experiment must first define quality, the measurement procedure, and the acceptable price.

---

### Step 10.1.2 — Units and definitions of compute budget

“Four times more compute” is not a reproducible specification. A budget can mean:

- number of generated candidates;
- prompt and completion tokens;
- generator forward passes;
- verifier or judge calls;
- expanded search nodes;
- time to first token and time to final answer;
- graphics processing unit (GPU) seconds;
- peak memory;
- external API cost.

Two methods with the same candidate count can have very different cost. One may produce short answers and use cheap executable tests; another may generate long trajectories and call a separate judge model.

A minimal report records

$$
(\text{quality},\ \text{tokens},\ \text{verifier calls},\ \text{latency},\ \text{monetary cost}).
$$

Interactive systems also need latency quantiles rather than a mean alone. Rare multi-minute responses may be unacceptable in chat yet reasonable in batch code analysis.

---

### Step 10.1.3 — Parallel and sequential scaling are different interventions

Parallel scaling changes sample width:

$$
y_1,\ldots,y_N
\sim
\pi(\cdot\mid x).
$$

Sequential scaling changes the length or internal structure of one trajectory:

$$
y=(y_1,\ldots,y_T),
\qquad
T\ \text{may depend on the prompt and intermediate signals}.
$$

Parallel gains depend on candidate diversity and the selector. Sequential gains depend on whether the policy can use additional steps to inspect and repair its reasoning rather than repeat the same error.

Budget forcing controls reasoning length by preventing an early stop or truncating at a fixed limit. It is a useful diagnostic, not a monotonicity theorem. A longer trajectory can loop, overwrite a correct conclusion, or spend tokens on irrelevant detail.

Frozen parameters do not imply an unchanged distribution of delivered responses. Preventing EOS changes which continuations are allowed; adding an instruction or a checker result changes the context of the next call. Module 6 distinguished network parameters from the actual generation policy. The same distinction matters here: drawing more independent responses with unchanged decoding and forcing one trajectory to continue are not interchangeable interventions in a sampling formula.

---

### Step 10.1.4 — Search adds states and a backup rule

Search operates on intermediate states rather than completed answers alone:

$$
s
\longrightarrow
\{s'_1,\ldots,s'_b\}
\longrightarrow
\text{score}
\longrightarrow
\text{next expansion}.
$$

At least four components are required:

1. a proposal rule;
2. an intermediate-state score;
3. a data structure for open candidates;
4. a stopping and final-selection rule.

Search can use feedback before an answer is complete. It also creates a new failure mode: a wrong early score may prune the branch that would have produced the correct solution. Search therefore adds a controller with its own policy and errors; it is not merely “more thinking time.”

---

### Step 10.1.5 — Write the evaluation contract first

Before the first run, fix:

- prompts and the averaging unit;
- answer extraction and canonicalization;
- generator, tokenizer, chat template, and decoding parameters;
- maximum budgets;
- aggregation or selection rule;
- verifier version;
- quality and cost metrics;
- handling of ties, abstentions, timeouts, and infrastructure errors;
- an independent set not used for tuning.

Otherwise different tasks can be compared under one budget label. pass@32 measures whether a success exists. self-consistency@32 measures a voting rule. Best-of-32 measures a particular ranker. The shared number 32 does not make the metrics interchangeable.

> **Extension:** *Modern LLMs*, Module 15 “Reasoning Models and Test-Time Compute” ([open module](/courses/modern-llms/module-15/)), provides the broader history of reasoning models and product interfaces. This module owns the exact selection and cost mechanics.

---

## Lesson 10.2. Repeated sampling and aggregation

### Step 10.2.1 — pass@N and its assumptions

Suppose each attempt independently solves the prompt with probability $p$. The probability that all $N$ attempts fail is $(1-p)^N$, hence

$$
\boxed{
\operatorname{pass@}N
=
1-(1-p)^N
}.
$$

![Pass@N coverage curves](assets/rl-for-llm/en/module-10/M10_pass_coverage_EN.png)

This is **coverage**: the probability that at least one correct candidate exists. It does not say whether the system can identify that candidate and return it to the user.

The assumptions belong next to the formula:

- attempts are conditionally independent for a fixed prompt;
- every attempt has the same success probability;
- correctness is defined before candidates are inspected;
- $p$ refers to the same decoding policy used at larger $N$.

When generations are positively dependent and repeat one failure mode, the independence formula usually overstates the gain. “Correlation” alone does not define a universal correction, however: dependence may be clustered, driven by latent prompt difficulty, or altered by temperature. Empirical `pass@k` should therefore be measured from candidate groups with prompt-level records.

Here $p$ belongs to one prompt. For a dataset, compute coverage for each $p_i$ and then average using the declared prompt weights. In general, the average of $1-(1-p_i)^N$ is not $1-(1-\bar p)^N$. A pooled single-sample accuracy therefore does not determine the coverage curve: the same mean can describe broadly distributed success or success concentrated on a few easy prompts.

If $p=0$, sampling cannot create support that the generator lacks. A small positive $p$ can still produce high coverage while a practical selector remains unreliable.

### Step 10.2.2 — Marginal gain and diminishing returns

The gain from one additional attempt is

$$
\Delta_N
=
\operatorname{pass@}(N+1)-\operatorname{pass@}N
=
p(1-p)^N.
$$

It is non-increasing in $N$. For $p=0.2$:

| Current budget $N$ | Gain from the next candidate |
|---:|---:|
| 0 | 0.2000 |
| 1 | 0.1600 |
| 3 | 0.1024 |
| 7 | 0.0419 |
| 15 | 0.0070 |

A large $N$ can still be worthwhile when a rare success is extremely valuable. The formula only says that decisions should be based on marginal rather than average return.

---

### Step 10.2.3 — The finite-sample pass@k estimator

The underlying $p$ is usually unknown. A common protocol generates $n$ candidates and observes $c$ verified successes. Take integer counts with $0\le c\le n$ and $1\le k\le n$. If $k$ candidates are conceptually sampled without replacement, the probability of selecting no success is

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

Therefore

$$
\boxed{
\widehat{\operatorname{pass@}k}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
}.
$$

When $n-c<k$, the value is one. With $n=200$, $c=10$, and $k=5$:

$$
\widehat{\operatorname{pass@}5}
\approx0.228284.
$$

Two kinds of randomness are involved. Once the pool is fixed, the formula gives the exact fraction of its size-$k$ subsets containing a success. That combinatorial statement needs no additional model call. Interpreting the number as an estimate of future generation quality requires an assumption about how the original pool was obtained. If its $n$ responses were independent draws from the same policy for this prompt, averaging the expression over newly collected pools gives $1-(1-p)^k$. This is the sense in which the estimator is unbiased.

A hand-curated collection, dependent tree-search results, or a mixture of decoding regimes does not inherit that conclusion automatically. For multiple prompts, calculate the expression within each prompt's group and aggregate afterward. Counts collected under one generation protocol cannot silently become a claim about another.

---

### Step 10.2.4 — Self-consistency votes on canonical answers

**Self-consistency** samples several reasoning trajectories, extracts a canonical final answer from each, and returns the most frequent answer. It votes on answers, not on the wording of the reasoning trace.

It helps when the correct canonical answer is the largest single mode while errors are dispersed across many alternatives. Under independent sampling, empirical frequencies converge to their probabilities and the correct mode wins more reliably.

This is a plurality rule, not a requirement that the winner receive more than half the votes. The correct answer may have probability below $1/2$ and still be the unique largest category. Canonicalization matters because it determines which strings contribute to the same category rather than competing separately.

Voting does not know which mode is correct. If one systematic error is more likely than the correct answer, additional votes reinforce the error.

![Two voting regimes](assets/rl-for-llm/en/module-10/M10_majority_modes_EN.png)

For a small number of categories, voting accuracy can be computed exactly by summing multinomial probability over all count vectors and sharing tie mass uniformly. Two teaching configurations give:

- probabilities $(0.40,0.15,0.15,0.15,0.15)$ yield accuracy about $0.79422$ at $k=16$;
- probabilities $(0.30,0.45,0.25)$ yield accuracy about $0.07707$ at $k=64$, although `pass@64` for the correct answer is essentially one.

The correct answer can be present in almost every pool while the selector almost always returns another mode.

Monte Carlo remains useful for larger answer spaces, but reports should include the trial count, random seed, and sampling uncertainty. Deterministic `argmax` tie-breaking is not neutral: it grants a positional advantage to the lowest index.

### Step 10.2.5 — Canonicalization, ties, and abstention

Voting is not defined until the function

$$
C(y)=\text{canonical answer}
$$

has been specified. Numerical representations can be normalized. Executing a program against tests, however, primarily checks a property; it does not by itself assign a unique canonical program. A protocol may group programs by their output vectors on a fixed input set, but those observed vectors then define the voting categories. Agreement on finitely many inputs does not establish complete program equivalence. Free-form text creates another difficulty: equivalent answers may be phrased differently, while similar strings may contradict one another.

The protocol must define:

- what is extracted;
- how parser failure is handled;
- how ties are resolved;
- whether abstention is allowed;
- whether confidence or an external score is used;
- whether repeated answers retain all their occurrences or are subject to an explicit deduplication rule.

A deterministic lowest-index tie rule is useful for code tests but creates a positional preference. Random tie breaking, an additional verification stage, or abstention is usually more defensible in a deployed system.

Independence is a property of the sampling procedure, not a label assigned while counting votes. Independent generations can produce identical strings. Ordinary self-consistency counts every occurrence because frequency is precisely what it estimates. Keeping only one copy of each distinct string defines a different aggregation rule. That may suit another task, but it is no longer the same estimate of the original answer distribution's mode.

---

## Lesson 10.3. Verifiers and Best-of-N

### Step 10.3.1 — Candidate coverage versus selection

**Best-of-N** (BoN) has two stages:

1. generate $N$ candidates;
2. choose one with a verifier or ranker.

With perfect verification, final accuracy equals pass@N. With an imperfect selector it does not. A useful decomposition is

$$
P(\text{output error})
=
P(\text{no correct candidate})
+
P(\text{a correct candidate exists but is not selected}).
$$

This decomposition assumes that the system returns a member of the original pool, with abstention counted as failure. The two events on the right are disjoint: either a correct choice was unavailable, or selection lost an available opportunity. A fresh candidate generated afterward enlarges the pool and must be treated explicitly.

The first term concerns generator support and budget. The second concerns selection. Increasing $N$ may reduce the first while making the second harder because a larger pool contains more convincing false positives.

A verifier may be an exact program, tests, a reward model, a process reward model (PRM), an LLM judge, or a hybrid. These mechanisms should not be collapsed into one word without an error model.

---

### Step 10.3.2 — An exact binary-verifier model

Consider a fully specified teaching protocol.

- A candidate is correct with probability $p$.
- A correct candidate is accepted with true-positive rate $\alpha$.
- An incorrect candidate is accepted with false-positive rate $\beta$.
- If any candidate is accepted, select uniformly among accepted candidates.

Assume that the $N$ candidate–verdict pairs are independent for this fixed prompt and protocol. This is what permits the power $(1-q)^N$. The marginal rates $p$, $\alpha$, and $\beta$ alone do not rule out a shared cause of checker errors. If such dependence exists, they do not determine the probability that every candidate is rejected.

The single-candidate acceptance probability is

$$
q=p\alpha+(1-p)\beta.
$$

Correctness among accepted candidates is

$$
\rho_{\mathrm{acc}}
=
\frac{p\alpha}{q},
\qquad q>0,
$$

while correctness among rejected candidates is

$$
\rho_{\mathrm{rej}}
=
\frac{p(1-\alpha)}{1-q},
\qquad q<1.
$$

These are conditional probabilities: each numerator is the probability of correctness together with the relevant verdict, and each denominator is the probability of that verdict. Once rejection is known, the appropriate success probability is no longer the original $p$. Under independence, learning the verdicts on the other candidates does not further change the correctness distribution of a selected rejected candidate.

The second conditional probability matters when no candidate is accepted. At least two fallback protocols are possible.

**Fallback to the same rejected pool:** if none is accepted, select uniformly from the original rejected candidates.

$$
\boxed{
S_N^{\mathrm{rej}}
=
\bigl[1-(1-q)^N\bigr]\rho_{\mathrm{acc}}
+
(1-q)^N\rho_{\mathrm{rej}}
}.
$$

**Draw a fresh independent candidate without verification:** if none is accepted, make one additional draw from the same generator rather than returning to the checked pool.

$$
\boxed{
S_N^{\mathrm{fresh}}
=
\bigl[1-(1-q)^N\bigr]\rho_{\mathrm{acc}}
+
(1-q)^Np
}.
$$

![Binary-verifier protocol](assets/rl-for-llm/en/module-10/M10_verifier_selection_EN.png)

With $p=0.2$, $\alpha=0.9$, and $\beta=0.1$:

$$
q=0.26,
\qquad
\rho_{\mathrm{acc}}=\frac9{13}\approx0.69231,
\qquad
\rho_{\mathrm{rej}}=\frac{0.02}{0.74}\approx0.02703.
$$

At $N=8$:

$$
S_8^{\mathrm{rej}}\approx0.63249,
\qquad
S_8^{\mathrm{fresh}}\approx0.64804.
$$

For $q>0$, the no-accept event vanishes as $N\to\infty$, so both protocols converge to $\rho_{\mathrm{acc}}$. Their finite-$N$ values differ. The term $p$ is wrong when fallback selects among candidates already known to have been rejected.

A larger pool makes it increasingly likely that at least one candidate is accepted. It does not give uniform selection an additional ability to distinguish genuine successes from false positives inside that accepted set. The vanishing no-accept event, rather than a steadily improving accepted-set composition, explains the limit.

The fresh-fallback protocol also spends more compute. It makes an extra generation with probability $(1-q)^N$, giving an expected total of $N+(1-q)^N$ generations rather than $N$. There are still exactly $N$ checks because the extra answer is unverified. At $N=8$ in this example, the expected addition is about $0.0899$ generations. The accuracy difference is therefore not a free gain under an identical strict candidate budget. At $q=0$ or $q=1$, reason directly about the reachable branch instead of evaluating an undefined conditional probability in an impossible branch.

**Boundary.** This limit belongs to binary acceptance with uniform selection among accepted candidates. It is not a universal ceiling for continuous-score ranking, pairwise tournaments, or exact verification.

### Step 10.3.3 — Ranking and the winner's error

Let candidate $i$ have true utility $u_i$ and verifier score

$$
s_i=u_i+\varepsilon_i.
$$

BoN selects

$$
\hat i=\arg\max_i s_i.
$$

Zero unconditional mean of $\varepsilon_i$ alone does not imply positive mean error for the selected candidate: the joint distribution of utility and error matters. With independent, nondegenerate additive evaluator noise independent of candidate utilities, maximization favors favorable noise as well as utility. This winner's-curse mechanism can cause proxy overoptimization; without those assumptions, the sign is not guaranteed.

The synthetic experiment uses the same two-feature construction introduced in Module 7, with a separately fixed sampling protocol. Each candidate has independent useful and nuisance features:

$$
g,h\overset{\mathrm{ind}}{\sim}\mathcal N(0,1),
\qquad
r_{\mathrm{proxy}}=g+h,
\qquad
r_{\mathrm{goal}}=g-\max(h-1,0)^2.
$$

Increasing $g$ helps both scores. Increasing $h$ always helps the proxy, but beyond one it incurs a quadratic penalty in the goal. This construction is not an instance of the independent additive-error assumption discussed above: its proxy error is structurally related to candidate features. It exhibits a specified, different route by which selection separates proxy improvement from actual improvement.

The function `proxy_overoptimization_curve` uses 10,000 trials with `seed=202610`. Each trial draws one pool of 512 candidates. For budget $N$, the selector takes the first $N$ entries and returns the proxy maximum. Trials are independent, while the points along a trial's curve use nested subsets. Thus the selected proxy maximum cannot decrease as that subset grows. No model weights are updated. Among the budgets evaluated below, the mean independent goal peaks at $N=32$ and then declines.

| $N$ | Proxy score | Independent goal |
|---:|---:|---:|
| 1 | -0.004 | -0.075 |
| 8 | 2.005 | 0.653 |
| 32 | 2.931 | 0.723 |
| 128 | 3.670 | 0.604 |
| 512 | 4.306 | 0.253 |

![Proxy overoptimization under Best-of-N](assets/rl-for-llm/en/module-10/M10_proxy_overoptimization_EN.png)

This is not a universal LLM curve. It isolates the mechanism: stronger optimization against an imperfect score requires stronger independent evaluation.

---

### Step 10.3.4 — Five selection protocols

The same candidate pool can be processed in different ways.

| Protocol | Requirement | Main risk |
|---|---|---|
| Exact verification | executable criterion | incomplete tests or unsafe environment |
| Threshold and abstention | calibrated score | too many refusals or false accepts |
| Maximum score | comparable scale | extreme-value proxy error |
| Pairwise tournament | reliable comparisons | non-transitivity and bracket order |
| Voting | canonical answer | systematic mode |

There is no universal winner. Executable tests are usually stronger than stylistic model judgments for code, but incomplete tests may accept a solution that violates an unstated requirement. Open-ended text often lacks exact verification; calibration, abstention, candidate-order swaps, and independent rubrics then matter more.

---

### Step 10.3.5 — Verification is part of the threat model

At large $N$, the generator effectively searches for cases on which the verifier fails. A BoN design therefore includes:

- sandboxed code execution;
- time, memory, file-system, and network limits;
- separation of training and hidden tests;
- versioned parsers and verifier environments;
- recorded accept/reject reasons;
- false-positive audits;
- an independent control criterion;
- protection of judge instructions and rubrics.

The lesson is the same as in RLVR: reproducible reward remains a proxy. Selection pressure can be strong even when model weights never change.

---

## Lesson 10.4. Process scores and structured search

### Step 10.4.1 — Outcome versus process feedback

An outcome evaluator scores a complete answer:

$$
R(x,y).
$$

A **PRM** scores intermediate prefixes or steps:

$$
r_t
=
R_{\mathrm{process}}(x,y_{\le t}).
$$

Process feedback is useful when the final signal is sparse and search must decide which branch to continue. The semantics of $r_t$ depend on labeling.

- “This local step is valid” scores a local property.
- “This prefix often leads to success” estimates success under a particular continuation policy.
- “An expert prefers this prefix” defines another object.

Calling all three a state value without qualification is incorrect.

---

### Step 10.4.2 — Monte Carlo labels for a prefix

From prefix $s$, generate $K$ independent continuations under policy $\mu$. Let $Z_j\in\{0,1\}$ indicate final success. Then

$$
\widehat V_K^{\mu}(s)
=
\frac1K\sum_{j=1}^{K}Z_j
$$

estimates

$$
V^{\mu}(s)
=
P_{\mu}(\text{success}\mid s).
$$

This connects directly to Monte Carlo evaluation in Module 2 “State, Value, and Bellman Equations” ([open module](/courses/rl-for-llm/module-02/)). It also fixes the scope: the label depends on continuation policy $\mu$, decoding, budget, and final criterion. It may become stale when the policy changes.

A zero label from a small $K$ does not prove impossibility. If the true success probability is $0.05$, observing no success in ten continuations has probability

$$
(1-0.05)^{10}\approx0.599.
$$

Even six successes out of eight give point estimate $0.75$ but a wide approximate 95% Wilson interval, about $(0.409,0.929)$. For a process model, two nearby raw means may therefore fail to define a reliable ordering.

![Uncertainty in a Monte Carlo prefix label](assets/rl-for-llm/en/module-10/M10_prefix_uncertainty_EN.png)

A label passport should record continuation count, policy and decoding parameters, final-verifier version, and an uncertainty or smoothing rule. Equal weighting of raw means also needs justification when prefixes receive different numbers of continuations.

There is a further boundary when the estimate guides search. A low $V^\mu(s)$ says that this continuation policy seldom succeeds from this prefix under this budget. It does not prove that the prefix is logically invalid or that a stronger search procedure could not finish it. Conversely, a high predicted success rate does not certify every step already written. Decide whether branch selection needs a continuation forecast, a local-validity check, or both as distinct signals.

### Step 10.4.3 — Aggregating step scores

Given $q_1,\ldots,q_T\in[0,1]$, common rules include

$$
S_{\min}=\min_tq_t,
\qquad
S_{\mathrm{mean}}=\frac1T\sum_tq_t,
$$

$$
S_{\mathrm{prod}}=\prod_tq_t,
\qquad
S_{\mathrm{last}}=q_T.
$$

For $(0.95,0.95,0.30,0.95)$:

| Rule | Value | Emphasis |
|---|---:|---|
| minimum | 0.3000 | weakest step |
| mean | 0.7875 | average level |
| product | 0.2572 | joint chain under strong assumptions |
| last | 0.9500 | final prefix only |

A product contains an implicit length dependence. A constant score $q=0.9$ gives $0.9^{10}\approx0.349$ and $0.9^{50}\approx0.005$. This can be meaningful when scores are compatible conditional probabilities in one factorization. Otherwise it is a hidden length penalty.

Return to the meaning of $q_t$ before selecting a row of the table. With genuinely local labels, the minimum highlights a suspect step, the mean can hide a fatal error, and $q_T$ does not combine earlier local checks. If the last prediction evaluates the entire prefix it has read, however, it can take earlier violations into account. “Last prediction” is not synonymous with “a judgment of the final sentence alone.”

The product requires an equally careful reading. Compatible conditional probabilities of step validity can factorize the probability that a chain is valid; the chain rule does not require every step to be independent. But successive forecasts of the same eventual success, made at nested prefixes, are not separate factors of that event. Multiplying them mechanically reuses overlapping information and need not produce the probability claimed for it.

This distinction is visible in the literature. In the setting of [Snell and colleagues](https://arxiv.org/html/2408.03314v1), the last prediction performed best among the tested aggregators. The authors related the difference from results using local labels to their soft Monte Carlo continuation labels. That finding supports checking label semantics, not adopting $S_{\mathrm{last}}$ universally.

The table therefore compares ways to use the given numbers; it does not declare all four outputs to be probabilities of one event. Choose an aggregator from the meaning of the labels and validate it against an independent outcome, rather than inheriting a library default.

---

### Step 10.4.4 — Beam search, best-first search, and MCTS

A prefix score is not yet a rule for spending the next unit of compute. One question makes the algorithms' differences concrete: **what happens to an alternative that we do not pursue now?**

**Beam search** keeps the best $B$ current prefixes, expands them, and prunes back to $B$. Candidates at the next level compete for a limited set of surviving slots. A pruned branch no longer participates, even if its poor early score was misleading. This makes the method simple and vectorizable, but also makes beam width a limit on remembered alternatives. In this solution-search setting, “best” refers to the declared process score, not necessarily ordinary text log-likelihood.

**Best-first search** maintains a global queue of states that remain available for expansion. Expanding one state inserts its continuations; alternatives need not vanish merely because another branch was selected. One line may grow deep while another waits near the root. This flexibility requires comparable scores across depths and explicit handling of memory, repeated states, and the total budget. Best-first alone does not guarantee an optimal path under an arbitrary heuristic.

**Monte Carlo tree search** (MCTS) stores statistics from earlier investigations of branches, not just open candidates. Its usual cycle is selection, expansion, evaluation or simulation, and backup to ancestors. For a visited child, it needs at least a visit count and a mean of the values returned through that branch. Unlike a fixed prefix score, those statistics change as search proceeds.

Classical UCT (Upper Confidence bounds applied to Trees) assigns a visited child an index of the form

$$
\operatorname{UCT}(j)
=
\overline X_j
+
c\sqrt{\frac{\ln N}{n_j}},
$$

where $\overline X_j$ is the child's mean return, $N$ is the parent's visit count, $n_j$ is the child's count, and $c$ controls exploration. Unvisited children normally receive a separate expansion priority.

The relation to UCB in Module 3 “Bandits, exploration, and one-step decisions” ([open module](/courses/rl-for-llm/module-03/)) is conceptually useful, but it does not import guarantees automatically. Text trees have enormous branching, string-distinct semantic duplicates, biased evaluation, and variable-length actions. Modern systems often use priors and PUCT-like variants; the exact formula and cost unit should be named.

Consider one iteration in the reasoning-maze setting used in the next step. A node records a position and the path that reached it; a continuation is a legal move. Start at the root and traverse the already expanded part of the tree. At each branch, compare accumulated value with an exploration bonus. A child with a lower current mean can still be selected when it has received far fewer investigations. The mean summarizes what previous investigations returned; the bonus expresses why limited evidence is not yet a sufficient reason to abandon an alternative.

At an unexpanded state, request legal continuations. This consumes one proposal-budget unit in the teaching environment. The new investigation then needs a value: either simulate onward to a checkable outcome or evaluate an unfinished prefix. These supply different kinds of evidence. An observed success is an outcome; a process-model prediction remains a prediction even after it enters a tree statistic.

The existing PUCT-like lab creates children and stores their priors from the frozen proposer. Among the new children, it first evaluates the one favored by the process score. A reached goal returns 1, a terminal state without a goal returns 0, and an unfinished state receives its process value. A long random rollout is not mandatory in this implementation: the supplied evaluator plays the leaf-evaluation role.

Let the returned value be $v$. Every node on the traversed path receives one additional visit and the same value in its accumulated total. With previous count $n_j$ and total $W_j$,

$$
n_j^{\mathrm{new}}=n_j+1,
\qquad
W_j^{\mathrm{new}}=W_j+v,
\qquad
\overline X_j^{\mathrm{new}}
=\frac{W_j+v}{n_j+1}.
$$

This is the single-player task's convention: the same investigation value travels to its ancestors without alternating player signs or adding edge rewards. Other tasks must account for their own rewards and discounting. Nor should the mean in an adaptive search be identified automatically with $V^\mu$ for one fixed continuation policy from Step 10.4.2: the procedure exploring a subtree can itself change.

The next iteration starts at the root again, but its evidence is different. The visited branch has a revised mean; its own exploration bonus generally falls with additional visits, while a larger parent count also changes other children's priorities. One investigation therefore changes what is worth exploring next. A systematically biased evaluator can make this same mechanism reinforce an unhelpful direction.

**Backing up a tree value is not backpropagating a neural-network gradient.** Counts and value totals change in the search algorithm's memory. The proposer weights and evaluator parameters remain frozen. Search accumulates information within a prompt without thereby performing another RL update of the network.

The lab uses a PUCT-like priority rather than the classical UCT expression above. Its child-selection score is

$$
\overline X_j+
 c\,p_j\frac{\sqrt{\max(N,1)}}{1+n_j},
$$

where $p_j$ is the proposer's stored prior for that move. An unvisited child's mean is initialized to zero in this implementation. The prior directs exploration toward moves the generator finds plausible; it is not a probability that the branch is correct. This specifies the small teaching implementation, not a universal formula for every PUCT system.

Finally, separate exploration from final selection. A large bonus can justify checking a branch next without justifying its immediate return as the answer. In the maze implementation, a discovered, exactly verified goal is returned as success. If the budget ends without a goal, the implementation retains a selected intermediate state but reports failure: its process score does not substitute for reaching the goal. A text system likewise needs a separate rule for returning a completed candidate or abstaining when no sufficiently checked solution exists.

![Process scoring and search](assets/rl-for-llm/en/module-10/M10_prm_search_EN.png)

### Step 10.4.5 — Search failure modes

Structured search can lose to repeated sampling when:

- prefix scoring is worse than final verification;
- the correct branch looks weak early;
- proposals lack diversity;
- width and depth are mismatched to the task;
- scores across depths are incomparable;
- the local verifier can be exploited;
- tree-management overhead consumes the gain;
- several branches are paraphrases of one error.

The notebook includes a small “reasoning maze.” A frozen proposal policy chooses the next step, an exact goal check supplies outcome verification, and a process score guides beam, best-first, and PUCT-like search. One proposal call is one budget unit.

![Search under a good and a biased process scorer](assets/rl-for-llm/en/module-10/M10_search_benchmark_EN.png)

In the fixed synthetic protocol, a good geometric scorer lets best-first search find the route earlier than repeated sampling. A biased scorer delays best-first search and worsens early beam pruning. With enough budget, the small environment becomes easy for almost every method. These are course-code results, not an LLM benchmark or a theorem that one search algorithm dominates.

Every comparison should therefore include cost-matched simple baselines: one answer, repeated sampling, self-consistency, and BoN. Search is justified by a better independent quality–cost frontier, not by architectural complexity. Reports should also retain tree coverage, proposal calls, scorer version, and examples of incorrectly pruned branches.

---

## Lesson 10.5. Adaptive budgets and economics

### Step 10.5.1 — An objective with compute cost

One formulation is

$$
\max_{m\in\mathcal M}
\left\{
Q(m)-\lambda C(m)
\right\},
$$

where $m$ is a generation mode, $Q$ expected quality, $C$ cost, and $\lambda$ converts cost into utility units.

A constrained form is

$$
\max_m Q(m)
\quad\text{subject to}\quad
C(m)\le B.
$$

Both force the cost definition to be explicit. GPU-seconds may dominate batch processing; end-to-end latency may dominate chat. Paid judge calls belong in $C$ just like generator tokens.

The non-dominated operating points form a Pareto frontier. Selecting one point is a product decision, not an intrinsic property of an algorithm.

---

### Step 10.5.2 — Marginal gain and adaptive allocation

Suppose prompt $i$ has independent single-attempt success probability $p_i$. Give all attempts the same unit cost and all prompts equal weight, and allocate an integer budget in advance. Its coverage—the probability of at least one correct candidate—after $n_i$ attempts is

$$
f_i(n_i)=1-(1-p_i)^{n_i},
$$

with next-sample gain

$$
\Delta_i(n_i)=p_i(1-p_i)^{n_i}.
$$

These gains are non-increasing. Under this exact model, an integer budget can be allocated greedily by repeatedly assigning the next sample to the prompt with the largest current $\Delta_i$. An exchange argument shows this is optimal **among allocations that satisfy the declared per-prompt minimum**.

The exchange argument explains why the rule works. Each prompt offers a sequence of additions with diminishing gains; taking a later addition requires taking the earlier ones. Consider the first point where another feasible allocation differs from the greedy choice. Move the last non-mandatory attempt from an overallocated prompt to the next available addition chosen by the greedy rule. The receiving gain is no smaller: greedy selected the largest available gain, whereas a later attempt on the donor prompt cannot be more valuable than its earlier available attempt. The exchange respects the minimum counts and does not reduce total coverage. Repeating such exchanges produces the greedy allocation with no worse objective.

That minimum is a separate product decision, not a consequence of the formula. `allocate_budget_greedy` defaults to `minimum_per_prompt = 1`: every prompt receives at least one attempt and the remainder is allocated greedily. It is a reasonable service guarantee — no prompt left without an answer — but it lowers achievable coverage.

For

$$
p=(0.02,0.08,0.20,0.50)
$$

and a total budget of 16, uniform allocation $(4,4,4,4)$ yields mean coverage $0.4723$. The greedy rule with `minimum_per_prompt = 1` produces

$$
(1,5,6,4)
$$

and mean coverage $0.5091$. Without the floor ($\texttt{minimum\_per\_prompt} = 0$) the same rule produces

$$
(0,6,6,4)
$$

and mean coverage $0.5173$ — the global optimum, confirmed by exhaustive search. The gap of $0.0082$ is the price of the "at least one attempt per prompt" guarantee: a near-hopeless prompt with $p=0.02$ consumes an attempt worth more on the prompt with $p=0.08$.

![Adaptive budget allocation](assets/rl-for-llm/en/module-10/M10_adaptive_budget_EN.png)

The result is exact for the stated coverage objective with known $p_i$, independent equally priced attempts, and declared minima. It is not automatically optimal for the accuracy of the answer actually returned. Perfect verification makes the two qualities coincide; an imperfect ranker can instead be overoptimized as the pool grows, as in Step 10.3.3. Final-answer quality may then lack diminishing positive gains. If completion lengths or verification costs differ, even the common budget unit disappears. Simply replacing “attempts” with “tokens” in the description does not preserve the optimality argument.

Real systems must estimate difficulty, and that estimator can be miscalibrated. Adaptivity introduces another component that needs validation. So far, the allocation adapts planned work to differences between prompts. Stopping an ongoing search in response to evidence already obtained is a separate layer of control.

---

### Step 10.5.3 — Stopping rules

A system may stop when:

- exact verification succeeds;
- a voting lead is mathematically irreversible under the remaining budget;
- marginal gain is small;
- several independent evaluators agree;
- time, token, or monetary budget is exhausted;
- proxy-overoptimization risk becomes unacceptable;
- uncertainty remains high enough to justify abstention.

The decision now depends on what has happened, not only on the prompt's estimated difficulty. Exact verification may establish a solution before the planned budget is spent; accumulated votes may make further counting unnecessary. This differs from assigning an attempt count in advance using an estimated $p_i$.

For self-consistency, suppose $R$ votes remain before a fixed maximum and the leader's margin over the strongest competitor is $d$. If $d>R$, even a tie is impossible, provided each remaining candidate contributes at most one vote and category definitions and vote weights stay fixed. At $d=R$, a tie remains possible and its handling matters. Stopping at an irreversible lead preserves the winner of the full vote up to the declared maximum. It saves samples, but does not establish that the leader is correct or repair a systematic wrong mode.

For continuous-score BoN there is no universal “good enough” score. A calibrated probability of success and the cost of another candidate are more useful than the raw maximum score.

---

### Step 10.5.4 — Train better or infer more expensively

Suppose additional training costs $C_{\mathrm{train}}$ and reduces per-query cost from $c_{\mathrm{old}}$ to $c_{\mathrm{new}}$ at acceptable quality. The cost-only break-even volume is

$$
Q^*
=
\frac{C_{\mathrm{train}}}
{c_{\mathrm{old}}-c_{\mathrm{new}}},
\qquad
c_{\mathrm{old}}>c_{\mathrm{new}}.
$$

With $C_{\mathrm{train}}=12000$, $c_{\mathrm{old}}=0.18$, and $c_{\mathrm{new}}=0.06$:

$$
Q^*=100000\ \text{queries}.
$$

If the new system is not cheaper per query, there is no cost-only break-even point. Even a finite point is insufficient: quality, latency, risk, and system lifetime must also be compared.

Expensive inference is often faster to deploy for small or changing traffic. At large stable volume, training, distillation, or a specialized ranker can amortize a fixed cost.

---

### Step 10.5.5 — Release protocol for a search system

Record four layers before publication.

**Generator**

- model and tokenizer versions;
- chat template;
- temperature, top-k, top-p, and length limits;
- seed and candidate count;
- sequential budget rules.

**Selection and search**

- answer canonicalizer;
- verifier or PRM;
- step aggregation;
- search algorithm and hyperparameters;
- ties, abstention, and stopping.

**Cost**

- generator and judge tokens;
- expanded nodes;
- latency and GPU-seconds;
- API price;
- retries and environment failures.

**Independent quality**

- pass@1 and pass@k;
- accuracy of the actually selected answer;
- calibration and abstention rate;
- quality by difficulty slice;
- robustness to rephrasing and verifier changes;
- audit cases where proxy and independent goal disagree.

This module stayed within one prompt and a controlled reasoning tree. Module 11 “RL for LLM agents: tools, environments, and long trajectories” ([open module](/courses/rl-for-llm/module-11/)) adds external state, tools, hidden variables, and multi-turn episodes. Module 12 “RL infrastructure for LLMs: memory, rollouts, and asynchrony” ([open module](/courses/rl-for-llm/module-12/)) covers distributed trajectory collection and policy versioning. Module 13 “Evaluating RL systems for LLMs: statistics, reward hacking, and safety” ([open module](/courses/rl-for-llm/module-13/)) develops independent evaluation and metric gaming in depth.

---

## Sources and reading route

The following primary sources support the module’s main external claims. The corresponding lecture steps explain the course calculations and synthetic experiments; their implementations are available through the course’s executable practice.

- [Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171) — the original multiple-trajectory and answer-aggregation protocol. It reports gains on selected tasks, not a guarantee under a dominant systematic error.
- [Chen et al., *Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374) — HumanEval and the finite-sample `pass@k` estimator. The expression belongs to a defined sampling and verification protocol.
- [Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*](https://arxiv.org/abs/2407.21787) — repeated-sampling scale and the distinction between candidate coverage and recognizing a success.
- [Lightman et al., *Let’s Verify Step by Step*](https://arxiv.org/abs/2305.20050) — process supervision, PRM800K, and the paper’s selection experiments. It does not make every process score a calibrated value function.
- [Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*](https://arxiv.org/abs/2305.10601) — branching over intermediate states, evaluation, lookahead, and backtracking.
- [Kocsis and Szepesvári, *Bandit Based Monte-Carlo Planning*](https://doi.org/10.1007/11871842_29) — classical UCT. Its guarantees do not transfer automatically to language trees with learned evaluators.
- [Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*](https://arxiv.org/abs/2408.03314) — prompt-dependent test-time scaling and adaptive compute in the studied setup.
- [Muennighoff et al., *s1: Simple test-time scaling*](https://arxiv.org/abs/2501.19393) — `budget forcing` through forced continuation or early termination; an empirical method, not a theorem that longer reasoning is always better.
- [Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) — reward-model overoptimization under stronger selection pressure, including Best-of-N.
- [Official Transformers generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation) and the [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — used only by the optional engineering branch; the core notebook runs offline on CPU.

---

## Cross-references

- Module 2 “State, Value, and Bellman Equations” ([open module](/courses/rl-for-llm/module-02/)): Monte Carlo prefix values and value backup along trajectories.
- Module 3 “Bandits, exploration, and one-step decisions” ([open module](/courses/rl-for-llm/module-03/)): UCB, exploration, and constrained budget allocation.
- Module 6 “The LLM as a policy: tokens, log-probabilities, and KL” ([open module](/courses/rl-for-llm/module-06/)): the actual generation distribution and decoding parameters.
- Module 7 “Reward models and preference data” ([open module](/courses/rl-for-llm/module-07/)): calibration, winner's error, and proxy overoptimization.
- Module 9 “RLVR and GRPO: verifiable rewards, group-relative estimates, and stability” ([open module](/courses/rl-for-llm/module-09/)): verifier contracts, process feedback, and rollout cost.
- *Information Theory for ML*, Module 2 “Entropy” ([open module](/courses/information-theory-for-ml/module-02/)): candidate-distribution entropy and limits of diversity metrics.
- *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” ([open module](/courses/information-theory-for-ml/module-03/)): logarithmic scores, KL, and distribution comparison.
- *Modern LLMs*, Module 10 “KV Caches and Efficient Inference” ([open module](/courses/modern-llms/module-10/)): engineering decoding modes, latency, and serving.
- *Modern LLMs*, Module 15 “Reasoning Models and Test-Time Compute” ([open module](/courses/modern-llms/module-15/)): the broader landscape of reasoning models and current interfaces.

## Key takeaways

1. pass@N measures the presence of a success under explicit assumptions, not selector quality.
2. Self-consistency removes dispersed error and amplifies the most likely systematic mode.
3. An imperfect verifier defines its own quality frontier and can be overoptimized at large $N$.
4. A PRM is useful only after its labels and step aggregation have been defined.
5. Search enables revisiting decisions but adds early pruning and new proxy objectives.
6. Budgets should be allocated by marginal gain and measured in tokens, latency, compute, and money.
7. Training versus expensive inference depends on traffic, lifetime, quality, and risk—not on one per-query price alone.
