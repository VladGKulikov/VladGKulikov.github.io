# Module 19. Evaluating Modern LLM Systems: From Benchmarks to Decisions

*“Modern LLMs” course · Module 19 lecture · edition 2026.10.25*

> **What this module is about.** The earlier modules built the objects we now have to choose between: pretrained models, quantized serving stacks, retrieval systems, post-trained policies, multimodal models, and agents. Evaluation is the discipline that turns those artifacts into a release decision. It is not merely a leaderboard lookup. A defensible result joins a decision, a task distribution, an execution scaffold, a scoring rule, and a statement of uncertainty. We will follow a single practical question throughout the lecture: which of two candidate systems should a team deploy for a real product?
>
> Public benchmarks will provide orientation, but the decision will eventually depend on paired outcomes, a private golden set, judge calibration, long-context diagnostics, operational cost, and continuous regression testing. Six numerical scenarios make those ideas concrete: assumed probabilities 95.5% and 95.0% at N=500, finite-sample `pass@k`, a position-biased judge, confidence and selective answering, a synthetic working-context curve, and a costed private evaluation run.
>
> **Prerequisites.** Module 18 introduced agent scaffolds and trajectory logging; Modules 10 and 17 covered long context and retrieval; Module 15 introduced verifiers and test-time sampling. The necessary definitions, formulas, and assumptions are introduced where they are used, so the module can also be read on its own.

---

### Reading routes

On a first pass, read Sections 1–6, 13, and 15. They connect a release decision to an inspectable report. In Section 6, work through the integer-outcome table before returning to the formulas; the small harness in Section 13 runs on its own.

On a second pass, examine how measurements fail: multiple attempts in Section 8, judges in Section 10, and context position in Section 12. Use the [Section 14 notebooks](/courses/modern-llms/module-19/notebooks/) as a map of these experiments. Run both notebooks, change one assumption, and explain the resulting change in interpretation. Sections 7, 9, 11, 16, and 17 provide a reference for benchmark families and experiment passports. In the [Section 19 exercises](/courses/modern-llms/module-19/exercises/), E1 completes the first route; E2–E3 use Sections 8, 10, and 12 and complete the second.

## 1. Evaluation is an instrument for a decision

Suppose a support organization is considering two candidate assistants. System A sits slightly higher on a public leaderboard. System B is cheaper, responds faster, and already integrates with the company’s refund tools. Which one should be released?

The question becomes difficult as soon as “better” must be operationalized. The deployed assistant must retrieve the right policy, avoid inventing rules, preserve state across turns, call tools with valid arguments, recover from tool failures, and finish the customer’s task under latency and cost limits. It must also handle rare but expensive mistakes, such as issuing an unauthorized refund.

No single public dataset represents that product contract. A knowledge benchmark may reveal academic breadth while saying little about tool state. A coding benchmark may test execution without testing customer communication. A human-preference arena may reward pleasant prose while missing a policy violation.

It helps to separate three objects.

- A **model** is a set of weights under a decoding configuration.
- A **system** adds prompts, retrieval, tools, retries, memory, and reasoning budget.
- A **product** places the system inside a real request distribution with business, safety, latency, and cost constraints.

Most benchmark numbers describe one particular system configuration. Release decisions concern the product.

This distinction yields the first governing rule of the module:

> **Begin with the decision that the evaluation must support.**

A model-selection study, a prompt-regression test, and a safety gate need different evidence. Even when they use the same examples, they may require different primary metrics and different stopping rules.

The second rule is that averages hide structure. Two systems can have identical overall accuracy while failing on different slices. One may be uniformly mediocre; another may excel on common requests and fail catastrophically on a rare financial workflow. Those outcomes are not interchangeable.

The third rule concerns uncertainty. A half-point difference can represent a real improvement or two examples that happened to flip. Any score intended to support a decision should arrive with the sample size, the protocol, per-example outcomes, and an uncertainty estimate.

By the end of the module, the public leaderboard will have become only the first page of a much larger evidence package.

## 2. The historical arc: every evaluation regime repaired a failure

![VIZ m19/01 — the evolution of LLM evaluation](assets/modern-llms/en/module-19/m19_01_timeline.svg)

Modern LLM evaluation did not replace one canonical benchmark with another. It accumulated layers as older instruments stopped answering new questions.

**GLUE — General Language Understanding Evaluation** standardized a basket of language-understanding tasks in 2018. SuperGLUE increased the challenge soon afterward. The attraction was obvious: one common suite made research comparisons possible.

**MMLU — Massive Multitask Language Understanding** broadened the idea to 57 academic subjects. Its aggregate score became a compact proxy for general knowledge and problem solving. For the models of its era, it was discriminative and easy to report.

The first major limitation was dimensionality. One mean score could conceal radically different capability profiles. **BIG-bench — Beyond the Imitation Game Benchmark** assembled 204 diverse tasks, while **HELM — Holistic Evaluation of Language Models** made the evaluation matrix explicit: scenarios crossed with accuracy, calibration, robustness, efficiency, fairness, and other desiderata.

The next limitation was open-ended quality. A dialogue response rarely has one exact reference. **MT-Bench — Multi-Turn Benchmark** explored strong LLMs as evaluators, and Chatbot Arena collected pairwise human preferences. These approaches made conversational quality measurable at scale, but introduced judge bias, population dependence, and ranking uncertainty.

Static public test sets then encountered contamination and saturation. **LiveCodeBench** continuously sources new programming problems. **LiveBench** refreshes objective, automatically scored tasks from recent information sources. Freshness narrows the window for memorization, although no published set remains protected indefinitely.

A further shift occurred when models became agents. **SWE-bench — Software Engineering Benchmark** asks a system to modify a real repository. **OSWorld** evaluates work in a computer environment. **$\tau$-bench** evaluates conversations in which an agent must obey policies and manipulate a database through tools. The object of evaluation is now a trajectory and its final state, not a short answer.

Finally, harder benchmarks restored headroom at the frontier. MMLU-Pro, Humanity’s Last Exam, FrontierMath, and **ARC-AGI — Abstraction and Reasoning Corpus for Artificial General Intelligence**, including ARC-AGI-2 and the interactive ARC-AGI-3 create difficulty through different mechanisms: expert knowledge, unpublished mathematics, novel abstraction, or interaction with a new environment.

The useful historical lesson is not “newer is always better.” Each benchmark family answers a different question and fails in a different way. A contemporary evaluation stack therefore combines broad public orientation, capability-specific probes, trajectory tasks, and private product evidence.

## 3. Classical measurement theory already knows most of the problems

Three older disciplines provide a more reliable vocabulary than leaderboard folklore.

### Psychometrics: difficulty and discrimination

In test theory, an individual question is an **item**. Items differ not only in difficulty but also in how well they separate stronger from weaker test takers. An item that nearly everyone answers correctly may remain pedagogically valid while contributing almost no information near the top of the ability range.

That is the ceiling effect. It does not begin at one universal percentage. The useful range depends on the number of items, annotation noise, item heterogeneity, and correlation between systems’ mistakes. The practical implication is simple: a thousand easy questions need not distinguish frontier systems as well as a much smaller, carefully constructed set of difficult items.

**IRT — Item Response Theory** formalizes the relationship between latent ability and item response probabilities. We will not fit an IRT model here, but its perspective is valuable: benchmark composition matters as much as benchmark size.

### Experimental design: exploit pairing

Two systems are normally evaluated on the same items. Their outcomes are therefore paired. The strongest evidence lies in disagreements: items solved only by A versus items solved only by B.

McNemar’s test uses precisely those discordant counts. Two aggregate accuracies cannot recover that information because they discard covariance. This is why an evaluation harness should retain per-example results rather than only the final mean.

Good experimental practice also includes pre-registering the primary metric, freezing the protocol before inspecting results, and keeping a held-out set for final decisions.

### Software testing: evaluate at multiple levels

A benchmark suite resembles a software test hierarchy.

- A small isolated question behaves like a unit test.
- A prompt plus retrieval or a tool call resembles an integration test.
- Completing a user workflow in an environment is an end-to-end test.
- A private suite run on every change becomes a regression test.

“Percentage of tests passed” is useful, but an engineer still needs logs, failure categories, slices, and reproducible environments. LLM systems require the same discipline, with the added complication that outputs may be stochastic and external judges may themselves be fallible.

Throughout this module, raw records and trajectories come first. Aggregates are derived artifacts.

## 4. How a benchmark loses measurement power

![VIZ m19/02 — the benchmark life cycle](assets/modern-llms/en/module-19/m19_02_paradigm.svg)

A new benchmark usually begins with ample headroom. Current systems fail in diverse ways, and score differences are large enough to be informative. Success changes the benchmark.

First, developers optimize the **protocol**: prompts, few-shot examples, answer extraction, retries, tools, and reasoning budget. Such tuning is legitimate when disclosed and applied consistently. Yet the score increasingly describes “model plus scaffold” rather than the weights alone.

Second, benchmark items spread through repositories, papers, tutorials, synthetic datasets, and training mixtures. **Contamination** becomes plausible. A high score can reflect the intended capability, recall of the exact item, or familiarity with a close variant. The aggregate result alone cannot distinguish those mechanisms.

Third, systems converge near the item ceiling. The leaderboard may still look precise, but rankings hinge on a small number of ambiguous or noisy questions. At this stage, item auditing and paired outcomes matter more than another decimal place.

The field responds in several ways.

1. **Harder successors.** MMLU-Pro expands answer choices, removes some trivial or noisy questions, and emphasizes reasoning.
2. **Living benchmarks.** LiveBench and LiveCodeBench add fresh, objectively scored material over time.
3. **Private hold-outs.** Organizers reveal the task family while retaining evaluation items; ARC-AGI uses private evaluation.
4. **Unpublished expert problems.** FrontierMath was constructed from original problems with automated verification.
5. **Counterfactual or mutated items.** Surface details change while the target skill remains testable.
6. **Trajectory evaluation.** The system must change an environment rather than emit one answer.

Hardness does not guarantee item quality. Humanity’s Last Exam contains 2,500 expert-created multimodal questions, yet later HLE-Verified work investigated ambiguity, answer errors, and rationale mismatches. A difficult item can still be defective.

The benchmark passport should therefore document provenance, update policy, held-out structure, annotation review, and the handling of disputed items—not merely the observed score range.

## 5. Build an evaluation map before choosing benchmark names

![VIZ m19/03 — an evaluation evidence map](assets/modern-llms/en/module-19/m19_03_map.svg)

Return to the support-assistant decision. A useful plan separates four layers.

### Layer 1: capabilities

This layer contains knowledge, reasoning, code, vision, long-context use, and tool interaction. Representative instruments include:

- MMLU-Pro, **GPQA — Graduate-Level Google-Proof Q&A**, and Humanity’s Last Exam for broad or expert academic tasks;
- FrontierMath and ARC-AGI-2 for specialized forms of difficult reasoning;
- LiveCodeBench for fresh programming problems;
- SWE-bench, OSWorld, and $\tau$-bench for actions in environments;
- **RULER — What’s the Real Context Size of Your Long-Context Language Models?**, NoCha, and LongBench v2 for different long-context requirements;
- **VHELM — A Holistic Evaluation of Vision Language Models** for a multidimensional view of vision-language systems.

These are not interchangeable rungs on one intelligence ladder. They differ in task object, interaction model, tools, and evaluation cost.

### Layer 2: behavior, safety, and honesty

Instruction following, refusal behavior, robustness under attack, truthfulness, calibration, and honesty can move independently. HarmBench studies red-teaming and robust refusal. TruthfulQA probes imitation of common falsehoods. **MASK — Model Alignment between Statements and Knowledge** distinguishes accuracy of belief from honesty of reporting.

A single capability average should not absorb these axes.

### Layer 3: operational properties

A user experiences time to first token, inter-token latency, cost, tool failure rate, repeatability, and task-completion time. Those metrics belong to the deployed system and can change when prompts, retrieval, retries, or reasoning effort change.

### Layer 4: evidence protocol

The observed result depends on the system prompt, decoding parameters, permitted tools, retry policy, scaffold version, answer extraction, judge rubric, and reasoning budget. A report that omits them is not reproducible.

A pragmatic model-selection package contains four kinds of evidence:

1. public capability benchmarks relevant to the use case;
2. a private golden set sampled from product traffic;
3. operational measurements on the target serving stack;
4. a separate safety and critical-edge-case evaluation.

The output is a decision dossier rather than one universal score.

### Confidence, calibration, and the option to abstain

The evaluation map tells us what to measure when comparing the two assistants. Deployment introduces a more local decision: which responses may the system deliver on its own, and which should go to review? Overall accuracy cannot make that distinction. We need a signal that separates cases within the same stream of requests, rather than another average over the stream.

A document assistant needs more than an average accuracy figure. It may be useful to send a difficult question to a person rather than return an incorrect instruction. To design that choice, first specify what confidence refers to. The token probability from Module 3 describes the next piece of text given its context. A familiar sequence can be highly probable and factually wrong. Likewise, “I am certain” is a generated statement until its reliability has been tested.

Let $q_i\in[0,1]$ estimate the probability that the **complete response** to item $i$ will pass a declared correctness check. The check supplies an outcome $y_i\in\{0,1\}$. A verifier, agreement among attempts, or a trained predictor might supply $q_i$; none is automatically a trustworthy probability estimator. Assess the estimates on data kept out of their construction and tuning.

**Calibration** asks whether stated probabilities agree with success frequencies. Among responses assigned confidence near 0.8, roughly 80% should be correct. The population condition is $\Pr(Y=1\mid Q=q)=q$. A finite evaluation estimates this relationship by grouping nearby scores under a declared binning rule. Report group sizes: three successes in three trials cannot establish certainty. This interpretation follows the calibration framework discussed by [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).

Consider ten authored response records, each carrying confidence 0.8, of which six are correct. The observed success rate is 0.6. The group suggests overconfidence, subject to sampling uncertainty. The **Brier score** evaluates probability estimates through squared error:

$$
\operatorname{BS}=\frac1N\sum_{i=1}^N(q_i-y_i)^2.
$$

Here the score is $(6\cdot0.2^2+4\cdot0.8^2)/10=0.28$. Assigning 0.6 to every record would give 0.24 on these same outcomes. That retrospective calculation illustrates the loss; it does not validate a calibrator fitted to the test set. If the true success probability is $p$, the identity $\mathbb E[(q-Y)^2]=(q-p)^2+p(1-p)$ shows why the expected score is minimized at $q=p$. Brier score measures overall probability-prediction quality, not calibration alone. A constant base-rate prediction can be calibrated yet fail to distinguish easy items from hard ones.

We can now define a decision rule: accept a response when $q_i\ge t$, and otherwise abstain or request review. This is **selective prediction**. For the accepted set $A_t$, define **coverage** and **selective risk**, the error rate among accepted responses, as

$$
\operatorname{coverage}(t)=\frac{|A_t|}{N},\qquad
\operatorname{risk}(t)=\frac{\sum_{i\in A_t}(1-y_i)}{|A_t|}.
$$

Coverage here is the fraction of tasks answered; it is distinct from the probability of finding at least one correct candidate in Module 15. An empty accepted set has undefined risk, not zero risk. Raising the threshold cannot increase coverage, but it need not lower risk if confidence ranks errors poorly. [Geifman and El-Yaniv (2017)](https://arxiv.org/abs/1705.08500) study prediction with this option to reject part of the input population.

The notebook uses 20 authored records: four successes out of four at confidence 0.95, six out of ten at 0.8, and three out of six at 0.6. Overall accuracy is 13/20 and Brier score is 0.2185. Threshold 0.8 accepts 14 responses, including four errors: coverage is 70% and risk is $4/14\approx28.6\%$. Threshold 0.9 accepts four correct responses. Its observed risk is zero, but four cases establish little about future failures, and 80% of requests now need another route. Choose a threshold on separate development data, accounting for both errors and handoffs; evaluate it on held-out data and critical slices. Calibration and abstention also leave the authorization checks from Module 18 fully necessary.

Notice what the higher threshold achieved. None of the stored responses was rewritten, and none of the model’s mistakes was repaired. What changed was **which responses the system allowed through to the user**. That can be a valuable system improvement, but it is not evidence that the underlying model became more accurate.

The risk figure must therefore travel with coverage. An assistant that answers almost every request and one that answers only a small selected subset are not being scored on the same set of cases. Compare operating points under a declared coverage or service policy, and account for what happens to the remaining requests: who handles them, how long that takes, and how often that route fails. A handoff continues the service; it does not remove the user’s task from the evaluation.

![Confidence and selective answering on an authored dataset](assets/modern-llms/en/module-19/m19_09_calibration.png)

## 6. Worked example A: what does half a percentage point mean?

There are four questions to keep apart. How much variation should we expect before an experiment has run? What changes when both systems answer the same items? How many items should we collect for a desired precision? And how should we report one observed success rate from a small sample? We will follow them in that order. Planning an experiment and interpreting its recorded outcomes are related tasks, not interchangeable calculations.

Start with a planning assumption, not an observed leaderboard result. Suppose success probabilities are $p_1=0.955$ and $p_2=0.950$, and we plan $N=500$ binary tasks per system. These are valid probabilities, but 95.5% cannot be the accuracy of a single 500-item binary run: the resolution is $1/500=0.002$, or 0.2 percentage points. It would require 477.5 successes. Actual counts of 478 and 475 would give 95.6% and 95.0%, a 0.6-point difference.

In a simple model, each independently sampled task produces a binary outcome $X_i$: one for success and zero for failure. Assume that these trials share the same success probability. Accuracy is their average. A single outcome has variance $p(1-p)$; independence lets us add those variances, while averaging divides the variance of their sum by the square of the sample size:

$$
\operatorname{Var}(\hat p)
=\operatorname{Var}\!\left(\frac1N\sum_{i=1}^N X_i\right)
=\frac1{N^2}\sum_{i=1}^N p(1-p)
=\frac{p(1-p)}{N}.
$$

Taking the square root puts the spread back on the scale of accuracy itself. A **standard error (SE)** describes that typical spread of the estimate across repeated samples. For independent binary outcomes, the standard error of a proportion is $\sqrt{p(1-p)/N}$. Planning uses an assumed probability $p$; analysis often substitutes the observed proportion $\hat p$.

This explains the pace at which more data helps. At the same success probability, four times as many independent items halve the standard error. The improvement concerns sampling variation; averaging does not remove incorrect labels or make an unrepresentative task set representative.

For two independent samples, variances add:

$$
SE_\Delta=\sqrt{\frac{p_1(1-p_1)}{N}+\frac{p_2(1-p_2)}{N}}
=0.013452.
$$

That is 1.345 percentage points. The assumed difference $\Delta=0.005$ is only

$$
z=\frac{\Delta}{SE_\Delta}=0.3717
$$

standard errors. A normal 95% confidence interval uses an estimate plus or minus $1.96SE$. Under this planning scenario, its approximate half-width is 2.64 points, much larger than the assumed 0.5-point effect. An actual interval must use observed data. The 95% coverage refers to repeated applications of the procedure to fresh samples; it is not a probability assigned to the fixed parameter after this particular interval has been computed.

![VIZ m19/04 — precision and sample size](assets/modern-llms/en/module-19/m19_04_saturation.png)

### Keep the task pairs

Now change the experimental setup: both systems answer the same tasks. An easy item may be easy for both; a particular obstacle may defeat both. Two aggregate accuracies do not retain that relationship, and the independent-sample formula does not account for it. Store the counts for both correct, A only, B only, and both wrong.

| Paired scenario, N=200 | Both correct | A only | B only | Both wrong | Accuracy A / B |
|---|---:|---:|---:|---:|---:|
| I | 150 | 20 | 10 | 20 | 85% / 80% |
| II | 160 | 10 | 0 | 30 | 85% / 80% |

Read each row as a comparison on the same task. A shared success raises both accuracies; a shared failure lowers both. Neither favors one system over the other. The difference comes from disagreements: in the first scenario, A’s twenty gains are accompanied by ten losses relative to B; in the second, its ten gains have no corresponding loss. Shared outcomes still matter for absolute quality, but the conditional test of which system has the advantage focuses on the discordant rows.

The **null hypothesis** of exact McNemar inference gives equal probability to the two directions of disagreement. Scenario I has 30 disagreements, 20 favoring A. Conditioning on that disagreement count, we ask how unusual the imbalance of directions would be under equal odds. At most ten outcomes favoring A is as far from balance as at least twenty, so the two-sided value is

$$
p_{\mathrm{test}}=2P[\operatorname{Bin}(30,0.5)\le10]\approx0.098737.
$$

Here $\operatorname{Bin}(n,p)$ counts successes in $n$ independent trials with probability $p$. A p-value describes how likely an equally or more extreme deviation is under the null; it is not the probability that the null is true. In scenario II all ten disagreements favor A, giving $p_{\mathrm{test}}=2/2^{10}=0.001953$. The aggregate accuracies are identical, yet the evidence differs. At this fixed difference, fewer discordant outcomes provide stronger evidence. Failing to reject the first null does not establish equivalence.

A **paired bootstrap** can estimate an interval for the difference: repeatedly sample whole rows, each containing both outcomes, with replacement; recompute the difference; take, for example, the 2.5th and 97.5th percentiles. The project notebook does this from saved responses. When multiple rows share a document or session, resample independent groups rather than pretending every row is independent.

Keeping each pair together preserves the observed relationship between the two outcomes. Resampling the A and B columns separately could create combinations that never belonged to the same task and discard the pairing. This bootstrap makes no new model calls: it studies variation in the estimate using the saved sample. It does not, by itself, test a future change in traffic, API revision, or external-tool behavior.

### Plan the precision you need

We now return to a decision made before collecting data: how large should the experiment be to achieve a chosen precision? For a transparent planning calculation, return to independent samples. That simplification estimates the cost of precision; it does not replace the paired protocol when both systems will face the same items.

For two independent proportions near $p$, targeting an interval half-width $\delta$ gives

$$
N\approx2p(1-p)\left(\frac{1.96}{\delta}\right)^2.
$$

At $p=0.95$, rounding up gives 146, 913, 3650, and 14,599 tasks **per system** for half-widths of 5, 2, 1, and 0.5 points. This does not guarantee 80% or 90% power to detect an effect of that size. **Power** is the probability of rejecting the null under a specified true alternative; planning it is a separate calculation that must also account for pairing. The useful lesson here is the precision cost: $N$ grows as $1/\delta^2$.

One narrower question remains: how should we report **one system’s success rate** when observations are scarce? We are no longer comparing two accuracies. A simple symmetric interval around the observed rate can extend beyond the allowed probability range. If no errors were observed, plugging that rate into its estimated variance gives zero and an unjustified appearance of certainty.

For a single observed proportion on a small dataset, we use a Wilson interval. Rather than assigning all plausible probabilities the variance estimated at the observed rate, it asks which candidate probabilities remain compatible with the observation under the corresponding test. With $s$ successes out of $N$, $\hat p=s/N$, and $z=1.96$, the condition for a candidate probability $p_0$ is

$$
(\hat p-p_0)^2\le z^2\frac{p_0(1-p_0)}{N},
\qquad 0\le p_0\le1.
$$

The left side measures the observed discrepancy; the right side supplies its permitted scale under the probability being tested. Solving for that probability gives an interval that need not be centered on the observed proportion. This construction, described in the [NIST handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm), is an inversion of the test. The resulting endpoints are

$$
\frac{\hat p+z^2/(2N)\ \pm\ z\sqrt{\hat p(1-\hat p)/N+z^2/(4N^2)}}{1+z^2/N}.
$$

Seven successes out of eight give 87.5% accuracy but an interval of approximately [52.9%, 97.8%]. That breadth is part of the result. Even zero observed errors does not prove perfect reliability. No interval can repair an unrepresentative sample or an incorrect independence assumption; statistical analysis comes after experimental design.

## 7. Difficulty has several sources: HLE, FrontierMath, and ARC-AGI

A benchmark can be difficult because it demands expert knowledge, because its problems are novel, because the system must infer an abstract rule, or because it must learn through interaction. Those forms of difficulty should not be collapsed.

**HLE — Humanity’s Last Exam** contains 2,500 multimodal questions contributed by subject-matter experts across many fields. Its challenge comes from broad, high-level expertise under automatically gradable answer formats.

**FrontierMath** contains hundreds of original, previously unpublished mathematical problems. Some require hours or days of work by specialists, and answers are constructed for automatic verification. Novelty and research depth are the main barriers.

**ARC-AGI-2 — Abstraction and Reasoning Corpus for Artificial General Intelligence** asks systems to infer transformations from a small set of demonstrations and applies private evaluation under efficiency constraints. Its ambition is skill acquisition on unfamiliar abstract tasks rather than recall of domain knowledge.

**ARC-AGI-3** changes the object again. A system explores interactive environments, discovers goals, forms a world model, and acts. Difficulty is expressed through a trajectory.

These benchmarks can produce different system rankings without contradiction. A system with mathematical tools may excel on one; a system with stronger adaptation may excel on another.

Hard benchmarks also introduce measurement costs:

- small samples yield wide uncertainty;
- expert labels are difficult to audit;
- private items reduce transparency;
- tools and inference budget can dominate the weights;
- rapid progress begins another saturation cycle.

A serious report pairs the score with item count, private/public split, verification procedure, allowed tools, cost limits, and retry policy.

## 8. `pass@k`: finding one solution is not the same as being reliable

The number of generations we store and the attempt budget we want to evaluate have different jobs. One determines how much evidence we collected; the other defines the operating regime under study. A stored pool can support estimates at several budgets, but the entire pool is not a single user-facing answer obtained without generation or selection cost.

When tasks are automatically verifiable, a system may generate several candidates. Given $n$ stored samples of which $c$ are correct, the finite-sample unbiased estimator for at least one success among $k$ draws is

$$
\operatorname{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

The formula has two connected interpretations. Conditional on a fixed pool of $n$ candidates, the ratio is the chance of selecting $k$ failures without replacement. If that pool was itself generated by independent draws from a fixed sampling policy, the estimator has expectation $1-(1-p)^k$, where $p$ is the unknown single-attempt success probability. That is the sense in which it is unbiased. Copying one generation, selecting favorable candidates before counting, or changing temperature within the pool breaks this justification.

Compute the estimate separately for each task, then average with declared weights. Pooling successes and samples from heterogeneous tasks into a single $(n,c)$ pair changes the estimand. The implementation checks $n\ge k$ and $0\le c\le n$, including no successes and fewer than $k$ failures.

For $n=20$ and $c=4$:

| $k$ | unbiased `pass@k` | plug-in $1-(1-c/n)^k$ |
|---:|---:|---:|
| 1 | 0.2000 | 0.2000 |
| 5 | 0.7183 | 0.6723 |
| 10 | 0.9567 | 0.8926 |

The formulas agree at one sample. At ten, the plug-in estimate is lower by roughly 6.4 percentage points because it treats the noisy observed rate $c/n$ as the true probability.

A large `pass@1` to `pass@10` gap indicates that the model can often produce a correct candidate somewhere in the sample set. Turning that headroom into a product improvement requires a verifier that can identify the correct candidate without access to the hidden reference.

Do not confuse `pass@k` with $\operatorname{pass}^k$, a reliability-oriented metric used in agent evaluations such as $\tau$-bench. `pass@k` asks whether at least one of $k$ attempts succeeds; it rises with $k$. $\operatorname{pass}^k$ asks for consistent success across repeated trials and normally falls as $k$ grows.

A complete `pass@k` report includes the number of generated samples, decoding parameters, verifier, aggregation rule, estimator, and generation cost.

## 9. Human arenas and composite indices answer someone else’s question

Open-ended assistant quality is naturally comparative. Arena (formerly LMArena and Chatbot Arena) presents two anonymous outputs and asks a user which one they prefer. A Bradley–Terry model converts many pairwise votes into a ranking.

The platform supplies valuable evidence because prompts come from real users and no exact reference answer is required. Yet the resulting ranking describes the arena population and protocol. It is not automatically the utility function of your organization.

Rankings depend on topic and language mix, model exposure, user heterogeneity, response length, presentation, model revisions under stable names, and statistical robustness of the aggregation.

This does not make arenas useless. It makes their scope explicit: they are broad human-preference instruments.

Bradley–Terry assigns each system a strength parameter $\beta_i$ and models $P(A\succ B)=\sigma(\beta_A-\beta_B)$, with $\sigma(z)=1/(1+e^{-z})$. Only differences matter: adding the same constant to all strengths changes no probability. Ties and heterogeneous preferences require protocol rules or extensions. Rating parameters are therefore not task-success percentages.

Composite indices introduce another layer. If

$$
I=\sum_j w_j s_j,
$$

then changing the weights $w_j$ can change the winner without changing any component score. The index answers the designer’s weighted question.

A composite is helpful for generating a shortlist when components and weights are visible. It is weak evidence for a specific deployment until the relevant axes are unpacked.

A defensible workflow is:

1. use arenas and composites for landscape navigation;
2. inspect component tasks and protocols;
3. select evidence relevant to the product;
4. make the final decision on private data and the target system stack.

## 10. LLM judges: a synthetic 30-point illusion

Using an LLM as a judge makes rubric-based evaluation inexpensive and scalable. The judge is still a model. MT-Bench documented position, verbosity, and self-enhancement biases, and subsequent work has studied those effects systematically.

Consider an intentionally simple scenario. Candidate systems are equally good, but the judge favors whichever response appears first by 15 percentage points. The first position therefore wins 65% of comparisons.

If A is always shown first, the report says A wins 65% to 35%—a false 30-point advantage. When the order is reversed, A wins only 35%. Averaging both orderings restores 50%.

![VIZ m19/05 — an auditable judge protocol](assets/modern-llms/en/module-19/m19_05_judge.png)

So far we have balanced the **average bias** in a deliberately specified judge model. The return to 50% is an expectation; a finite run can still differ. Balanced totals do not tell us whether the judge held the same view of any particular pair. To answer that question, we must move from aggregate votes to two linked records for each comparison.

A robust swap protocol maps positional labels back to candidate identity.

- If the same candidate wins in both orders, the preference is consistent.
- If the judge always chooses the first position, the identity-level verdict conflicts and the pair becomes an abstention.
- If only one ordering produces a tie, the protocol used here also assigns `abstain` and retains both original verdicts.

Two ties produce `tie`; conflicting verdicts or a tie in only one order produce `abstain`. Report these separately. The aggregate counts 650 and 350 cannot reconstruct agreement for individual pairs. Returning to 50% is an expectation under this pure position-bias model, not a general cure.

The denominator now matters. A win rate computed only over consistent winner verdicts describes that selected subset, not every comparison submitted to the judge. Excluded pairs may be the most ambiguous ones rather than a random slice. Report the total number of pairs, the shares of consistent decisions, ties, and abstentions, and the rule for counting ties in the final score. Otherwise a more cautious evaluator can make a victory look more decisive simply by declining the difficult comparisons.

This is not the abstention from Section 5. There, the evaluated assistant decided whether to deliver a response; here, the measuring system decides whether its own judgments are consistent enough to retain. Consistency is useful but does not establish correctness: the judge can favor verbosity in both orders or miss the same factual error twice.

Swapping diagnoses dependence on position. A stronger protocol also includes:

- an explicit rubric and anchor examples;
- hidden model identities;
- length control or a separate verbosity analysis;
- calibration against human judgments;
- multiple judge families for high-stakes decisions;
- reported disagreement and abstention rates;
- explicit cost.

Three judges evaluated in two orders create six model calls per pair. A 500-pair golden set therefore requires 3,000 judge calls. Judge quality and judge cost belong in the result passport.

## 11. Multimodal capability, safety, truthfulness, and honesty are not one axis

A composite “intelligence” score can hide properties that move in opposite directions.

### Multimodal evaluation

A vision-language system should be evaluated separately for perception, knowledge, reasoning, robustness, multilinguality, and safety. **VHELM** standardizes inference and spans nine aspects. The multidimensional view matters because chart reasoning can improve while robustness to a small visual change remains poor.

### Safety evaluation

HarmBench provides a standardized framework for automated red teaming and robust refusal. Safety is not equivalent to refusal rate. A useful system must reject genuinely harmful requests while continuing to answer benign ones. Agent safety also includes actions: a harmless-looking final message does not excuse a dangerous intermediate tool call.

### Truthfulness, honesty, and calibration

TruthfulQA tests whether models reproduce common human misconceptions. That behavior depends on knowledge and response policy. **MASK** makes a different distinction: whether a model’s belief is correct and whether it reports what it believes are separate questions.

MASK elicits a model’s answers in one setting and compares them with responses under pressure. This is an operational measure based on observable behavior across prompts, not direct access to internal beliefs or distributions.

A product evaluation may therefore need independent measures for factual correctness, calibration, willingness to admit missing evidence, robustness to false premises, tool-result faithfulness, and action safety.

The lesson is not to maximize the number of benchmarks. It is to prevent an average from erasing the failure mode that matters most.

Calibration in Section 5 asks whether numerical confidence matches observed correctness frequencies. It is distinct from both the truth of an individual claim and an honest expression of uncertainty. A decision to abstain requires recorded confidence estimates, checked outcomes, and separate validation of the threshold.

## 12. Long context: accepted length versus usable evidence

A context-window specification tells us that a system accepts an input. It does not tell us how uniformly the model retrieves, aggregates, or reasons over that input.

**NIAH — Needle-in-a-Haystack** is a useful basic addressability test, but retrieving one distinctive string is a weak proxy for real document work. **RULER** adds multiple needles, tracing, and aggregation. **NoCha** uses long narratives and global understanding. **LongBench v2** includes realistic tasks over documents, dialogue history, repositories, and structured data.

Position also matters. *Lost in the Middle* found that relevant evidence placed near the beginning or end was often used more effectively than evidence in the middle.

The module’s numerical example is a synthetic profile, not a measurement of a named model. Let $x$ be the normalized location of the evidence, with zero at the start of the context and one at the end. Hold the task and input length fixed and define

$$
a(x)=0.95-0.35\sin^2(\pi x),\qquad x\in[0,1].
$$

Accuracy is 0.95 at both edges and 0.60 in the center. An average over positions also needs a distribution over positions: this example weights every location equally. The mean of the squared sine on the unit interval is one half. Using $\sin^2(\pi x)=(1-\cos(2\pi x))/2$, the cosine term integrates to zero over its full period, giving

$$
\int_0^1\sin^2(\pi x)\,dx=\frac12,
\qquad 0.95-0.35\cdot\frac12=0.775.
$$

Mean accuracy is

$$
\bar a=0.775.
$$

Define the above-threshold position share as the share of positions satisfying $a(x)\ge0.9$. This asks a different question from mean accuracy: where is the requirement met? Rearranging the inequality gives the boundary of the first interval:

$$
\begin{aligned}
\sin^2(\pi x)&\le\frac{0.95-0.9}{0.35}=\frac17,\\
x_*&=\frac1\pi\arcsin\!\left(\frac1{\sqrt7}\right),
\qquad \text{working share}=2x_*.
\end{aligned}
$$

The sine rises from zero toward the center, then falls symmetrically. The condition therefore holds near both edges, with each admissible interval as long as the boundary value just computed. We add those lengths, not the accuracy values within them. Then

$$
\text{working share}=0.246752.
$$

The inequality selects two intervals, $x\le0.123376$ and $x\ge0.876624$. Under a uniform position distribution, their combined measure corresponds to approximately 246,752 positions in a hypothetical million-token window. These are two disjoint edge regions, not a contiguous usable window of that length.

![VIZ m19/06 — nominal and working context](assets/modern-llms/en/module-19/m19_06_lost_middle.png)

There is no task-independent “real context size.” The answer depends on retrieval versus aggregation, number and position of evidence pieces, document format, answer length, retrieval assistance, prompt design, reasoning budget, and the chosen quality threshold.

A good long-context report plots performance across lengths, positions, and task types instead of reporting only the maximum accepted input.

## 13. A minimal evaluation harness: one call, one observed output

An **evaluation harness** connects a model, examples, and metrics. Its most important elementary contract is:

> **For a fixed system revision, each example and trial ID identifies one saved response.**

The small example uses one trial. A `pass@k` evaluation needs multiple explicit trials, all counted in its budget. The output is stored and every metric is computed from that same observation. Calling a stochastic model separately for exact match, a judge, and a length metric would evaluate different generations while pretending they were one answer.

Here `exact_match` ignores case and leading or trailing whitespace; it is not byte-for-byte equality. Internal whitespace and punctuation still matter. The notebook also defines `normalized_exact_match`, which applies Unicode NFKC, collapses internal whitespace, and strips specified terminal punctuation. Choose the scoring policy before comparing systems.

```python
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: str
    expected: str
    slice_name: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.example_id, self.prompt, self.expected, self.slice_name)):
            raise ValueError("example fields must be non-empty strings")

def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip().casefold() == expected.strip().casefold())

def evaluate(
    model_fn: Callable[[str], str],
    dataset: Sequence[Example],
    metrics: Mapping[str, Callable[[str, str], float]],
) -> dict:
    dataset = tuple(dataset)
    if not callable(model_fn) or not metrics or any(not isinstance(k, str) or not k or not callable(v) for k, v in metrics.items()):
        raise ValueError("supply a callable model and named callable metrics")
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if any(not isinstance(ex, Example) for ex in dataset):
        raise TypeError("expected Example records")
    if len({ex.example_id for ex in dataset}) != len(dataset):
        raise ValueError("example_id values must be unique")

    records = []
    for ex in dataset:
        prediction = model_fn(ex.prompt)  # the only model call for this item
        if not isinstance(prediction, str):
            raise TypeError("model response must be text")
        scores = {
            name: metric(prediction, ex.expected)
            for name, metric in metrics.items()
        }
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("metric scores must be finite numbers in [0, 1]")
        records.append({
            "id": ex.example_id,
            "slice": ex.slice_name,
            "expected": ex.expected,
            "prediction": prediction,
            "scores": scores,
        })

    aggregate = {
        name: sum(row["scores"][name] for row in records) / len(records)
        for name in metrics
    }
    return {
        "records": records,
        "aggregate": aggregate,
        "model_calls": len(records),
    }
```

This is not yet a production platform, but it already preserves unique item IDs, raw predictions, slices, multiple metrics over one output, and the call count. This minimal kernel stops on a model or metric error. The full notebook distinguishes a declared model-failure score of zero from a metric implementation error, which invalidates the aggregate (`None`). An evaluator bug must not silently become a model-quality result.

A technical failure or a missing required result is not the same as an appropriate refusal. If an item tests whether the system respects a prohibition, refusing the action may be the successful outcome. The task criterion determines the label, not the mere presence of a refusal phrase.

The aggregate has a separate limitation. This function weights every row equally. Recording a `slice` field does not change that average. If rare cases were deliberately oversampled, the result describes the constructed test set; estimating ordinary traffic requires combining slice results with their traffic proportions. Section 15 returns to that reporting decision.

A mature harness adds model and prompt versions, sampling parameters, latency, cost, failures, traces, and budgets.

### Planning the golden-set size

For an expected success rate $p=0.8$ and desired 95% half-width $m$, a normal planning approximation is

$$
N\approx p(1-p)\left(\frac{1.96}{m}\right)^2.
$$

It gives 683 examples for a three-point half-width and 246 for five points. These numbers describe the overall rate, not every rare slice. Critical low-frequency cases should be deliberately oversampled and reported separately.

### What a canary can tell us

A **canary** is a marker placed in protected data and searched for where it should not appear. For one exact independent guess, a uniformly random 32-character hexadecimal canary has chance-match probability

$$
16^{-32}=2^{-128}\approx2.94\cdot10^{-39}.
$$

Exact reproduction is strong evidence that the system encountered the string through some channel: training, logs, a cache, prompt leakage, or another data path. It does not by itself identify the channel or prove pretraining contamination. Canaries complement provenance and access controls.

For $M$ possible matches, the union bound is $\min(1,M/2^{128})$. A string derived from a public seed or known text has no such secrecy. The notebook uses a public deterministic marker with negative and planted positive controls; it tests the detector, not data privacy.

### Interpreting conditional compute accounting

For 700 examples averaging 400 input and 150 output tokens, total token volume is 385,000. A dense $2P$-per-token estimate for an 8.03B model is

$$
C=6.1831\cdot10^{15}\ \text{FLOP}.
$$

At an assumed effective 495 TFLOP/s, the conditional arithmetic equivalent is 12.49 seconds. Real wall-clock time includes autoregressive decode, batching, network calls, tools, scheduling, and judges. This is neither a strict latency bound nor evidence that frequent runs are affordable. It assumes one effective throughput for prompt processing and decode, although their execution differs. Measure duration and monetary cost on the target stack.

## 15. A private golden set in seven deliberate steps

![VIZ m19/07 — building a product golden set](assets/modern-llms/en/module-19/m19_07_golden_pipeline.svg)

Public benchmarks position a system in the general landscape. A private **golden set** determines whether it serves your product.

### Step 1: state the decision and unit of analysis

Is the result selecting a model, approving a prompt revision, gating a release, or tuning a router? The unit may be a response, a conversation, an agent trajectory, a document, or a user session.

For the support assistant, distinguish demonstrating a capability from justifying a replacement of the current system. If the goal is to resolve a customer request, several messages from one conversation are not several independent successful customer outcomes. The unit of analysis should match the result the deployment decision is meant to improve.

### Step 2: sample the product distribution

Include common requests and deliberately oversample rare critical cases. Report both natural-weighted performance and slice results. A set made only of adversarial examples is valuable for red teaming but does not estimate normal user experience.

Oversampling buys a clearer view of rare cases; it should not silently redefine the product population. Estimate performance within each declared slice, then use the corresponding traffic proportions when forming an overall traffic-weighted result. Keep critical slices visible in their own right. If those proportions are unknown, state the chosen weighting scheme rather than presenting the unweighted row average as measured user experience.

### Step 3: choose a metric that matches the task

Classification can use accuracy or F1. Structured outputs need schema and semantic-field checks. Code should be executed against hidden tests in a sandbox. Open-ended responses need human pairwise comparison or a calibrated judge. Agents need final environment state, trajectory quality, failures, cost, and repeatability.

### Step 4: freeze the protocol before seeing the outcome

Record the model revision, system prompt, tools, temperature, retry policy, reasoning budget, judge rubric, and failure handling. Otherwise protocol tuning can silently choose the winner.

Record what will count as sufficient evidence for the choice, too. Statistical persuasiveness and practical value are different questions. A small but well-established gain on routine answers need not compensate for a refund-policy violation or unacceptable latency. Conversely, an inconclusive comparison does not establish equal quality and automatically justify the cheaper candidate: the uncertainty interval may still allow a deterioration the product cannot accept.

### Step 5: protect the held-out data

Keep the decisive split out of training corpora, public examples, and logs reused for training. Canaries detect some leakage paths, but provenance, access control, and data-retention policy remain primary.

Repeatedly inspected regression cases gradually become development data even if they never enter weight training. Prompt selection based on their errors is still adaptation. Separate development cases, regression tests, and a rarely opened final audit split. Repeated testing also raises the chance of a spurious win; predeclare primary metrics and release criteria.

### Step 6: retain trajectories and failure causes

For agents, final success is too coarse. Save tool calls, intermediate state, retries, errors, and recovery. A system may reach the goal through an unsafe action, or fail because the environment—not the model—was defective.

### Step 7: turn evaluation into continuous regression testing

**CI/CD — Continuous Integration and Continuous Delivery** can run a fast subset on every change and a full suite before release. Prompt edits, retrieval-index changes, tool updates, and dependency upgrades can regress behavior independently of the weights.

The platform is secondary to the contracts. `lm-evaluation-harness` is widely used for academic tasks; Inspect AI supports configurable and safety-oriented evaluations; OpenAI Evals provides programmable eval patterns; LangSmith, Langfuse, and Phoenix support trajectory observability. Whatever the stack, preserve versions, raw records, and reproducible reports.

We can now return to the two assistants from the opening. The first system’s public rank does not settle the choice; the second system’s lower price does not settle it either. Bring together paired outcomes on relevant requests, the size and uncertainty of the difference, critical-slice failures, and measured service cost. If the policy uses the threshold from Section 5, include the handoff rate and the performance of that route. These observations should describe one declared configuration, not a convenient collection of the best numbers from different operating modes.

A release decision follows when that evidence is compared with the conditions set in advance. The study may support a change, reveal why a candidate is unacceptable, or leave a specific uncertainty that warrants a more informative check. The last outcome is not an evaluation failure. It means the measurement did not claim more certainty than the evidence permits. The chain from requests to saved responses, metrics, and intervals ends in an engineering decision—not an obligation to name a winner.

## 16. Eight directions shaping evaluation as of 10 October 2026

Rather than memorizing a current list of “best benchmarks,” track the directions changing measurement practice.

### 1. Living and refreshed datasets

LiveBench, LiveCodeBench, and domain-specific variants reduce the contamination window. Scores must be tied to a benchmark version and date because successive releases are not identical experiments.

### 2. Private and centrally executed evaluation

Private ARC-AGI splits and some professional suites retain evaluation material or run submissions centrally. This reduces direct tuning pressure while increasing the need to trust and document the organizer’s infrastructure.

### 3. Controlled mutation and counterfactual items

Benchmark mutation changes surface form while preserving a verifiable target skill. It can separate memorization from generalization, but the mutation procedure itself needs validation.

### 4. Trajectories replace terminal answers

SWE-bench, OSWorld, and $\tau$-bench evaluate changes to an environment. Newer variants such as SWE-bench Pro and OSWorld 2.0 extend horizon and professional complexity. Longer trajectories raise the importance of repeatability, environment state, and complete logs.

OSWorld 2 provides a concrete example: the checked `osworld-v2.1` release of 16 September 2026 pins coordinated task, asset, website, and environment-image revisions. Mixing releases changes the experiment even if the agent name stays fixed. [Source: OSWorld-V2 release history](https://github.com/xlang-ai/OSWorld-V2).

### 5. Cost and efficiency enter the metric

ARC-AGI includes efficiency constraints, and agent evaluations increasingly report cost, latency, and trial count. A system that succeeds after hundreds of expensive attempts is not equivalent to one that succeeds reliably on the first run.

### 6. Judges receive their own evaluations

Position, verbosity, style, shortcut, and self-preference biases have made judge calibration a separate discipline. Judge reports increasingly include human agreement, swap consistency, rubric version, and failure analysis.

### 7. Rankings become personalized or pluralistic

Aggregate human preference hides heterogeneous users. Personalized and pluralistic leaderboards recognize that “best” depends on audience, language, and task distribution. That framing is closer to product selection than a universal ordering.

### 8. Evaluation cards and evidence passports

A result is increasingly packaged with benchmark family, version, split, metric, scaffold, cost, and uncertainty. The unit of trust is moving from a leaderboard cell to a structured evidence object.

Across all eight directions, confidence moves away from the benchmark name and toward protocol completeness and inspectable artifacts.

## 17. Read two passports: the benchmark and the reported run

Before using a score, fill in two linked records.

### Benchmark passport

1. What capability or behavior is intended to be measured?
2. How many items and which slices are included?
3. How and when were the items created?
4. How were labels verified and disputes handled?
5. Is the test public, private, refreshed, or centrally executed?
6. What does the metric aggregate and conceal?
7. Is the benchmark still discriminative for the systems being compared?

### Result passport

1. exact model revision;
2. system prompt and message format;
3. tools, retrieval, and scaffold;
4. decoding parameters, trials, and `pass@k`;
5. reasoning effort and context budget;
6. answer extraction;
7. judge, rubric, and presentation order;
8. item-level outcomes, intervals, and paired tests;
9. cost, latency, and evaluation date;
10. harness code and dependency versions.

This mirrors Module 18. There, the scaffold transformed a model into an agent. Here, the harness transforms tasks into evidence about a system. In both cases, the model name alone is insufficient.

## 20. Key takeaways, course finale, and sources

![VIZ m19/08 — evaluation in one page](assets/modern-llms/en/module-19/m19_08_cheatsheet.svg)

Modern LLM evaluation is a chain of evidence, not a hunt for one authoritative number.

- **Start from the decision.** Desired behavior and acceptable risk determine the task set.
- **Separate model, system, and product.** A public score normally describes one configuration, not every deployment.
- **Read the benchmark life cycle.** Protocol adaptation, contamination, saturation, and item defects reduce discriminating power.
- **Keep item-level outcomes.** Paired evidence is more informative than two independent means.
- **Report uncertainty.** In the teaching scenario, assumed probabilities 95.5% and 95.0% at N=500 give $z=0.372$, not a decisive victory.
- **Use finite-sample `pass@k`.** It measures the chance of finding a solution, not repeatable agent reliability.
- **Audit LLM judges.** Swapping, rubrics, human calibration, and disagreement reporting belong to the result.
- **Distinguish nominal and working context.** Performance depends on length, position, task, and threshold.
- **Build a private golden set.** Public benchmarks map the field; private evidence makes the product decision.
- **Run evaluation continuously.** Prompts, retrieval, tools, serving changes, and dependencies can regress independently of model weights.

### Primary sources

- GLUE — [General Language Understanding Evaluation](https://arxiv.org/abs/1804.07461)
- MMLU — [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- BIG-bench — [Beyond the Imitation Game](https://arxiv.org/abs/2206.04615)
- HELM — [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- MMLU-Pro — [A More Robust and Challenging Multi-Task Benchmark](https://arxiv.org/abs/2406.01574)
- LiveBench — [A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314)
- LiveCodeBench — [Holistic and Contamination-Free Code Evaluation](https://arxiv.org/abs/2403.07974)
- Chatbot Arena — [An Open Platform for Human Preference Evaluation](https://arxiv.org/abs/2403.04132)
- MT-Bench / LLM-as-a-judge — [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- Humanity’s Last Exam — [arxiv.org/abs/2501.14249](https://arxiv.org/abs/2501.14249)
- HLE-Verified — [arxiv.org/abs/2602.13964](https://arxiv.org/abs/2602.13964)
- FrontierMath — [arxiv.org/abs/2411.04872](https://arxiv.org/abs/2411.04872)
- ARC-AGI-2 — [official technical report](https://arcprize.org/blog/arc-agi-2-technical-report)
- ARC-AGI-3 — [official description](https://arcprize.org/arc-agi/3)
- SWE-bench — [Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- SWE-bench Pro — [arxiv.org/abs/2509.16941](https://arxiv.org/abs/2509.16941)
- OSWorld — [arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)
- $\tau$-bench — [arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)
- `pass@k` — [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- Lost in the Middle — [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
- RULER — [arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654)
- NoCha — [arxiv.org/abs/2406.16264](https://arxiv.org/abs/2406.16264)
- LongBench v2 — [arxiv.org/abs/2412.15204](https://arxiv.org/abs/2412.15204)
- VHELM — [arxiv.org/abs/2410.07112](https://arxiv.org/abs/2410.07112)
- HarmBench — [arxiv.org/abs/2402.04249](https://arxiv.org/abs/2402.04249)
- TruthfulQA — [arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958)
- MASK — [arxiv.org/abs/2503.03750](https://arxiv.org/abs/2503.03750)
- lm-evaluation-harness — [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Inspect AI — [github.com/UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai)
- OpenAI Evals — [github.com/openai/evals](https://github.com/openai/evals)

- Arena naming — [official announcement](https://arena.ai/blog/lmarena-is-now-arena)
- Exact McNemar inference — [statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)

### From evaluation to the full mathematical model

Nineteen modules have covered the entire stack: tokenization, attention, MoE, data, pretraining, serving, post-training, multimodality, retrieval, agents, and finally evidence about system behavior. The course’s recurring idea is causal integration: architecture, data, optimization, runtime, and evaluation must be understood together. A change is an improvement only when its consequences are measured elsewhere in the stack.

The additional [Module 20](module-20.md) now assembles the model computation: tokens, a forward distribution, a loss, a weight update, and generation. Its numerical example checks the pieces together, while comparisons with SSMs and diffusion establish the scope of the final expression. [Module 3](module-03.md) remains the place to revisit Transformer foundations.

---

*Landscape verified: 10 October 2026. The lecture intentionally omits a current frontier leaderboard: those values change faster than a stable course and depend on scaffold version, inference budget, and evaluation date. All numerical examples in Sections 6, 8, 10, 12, and 13 belong to the frozen educational contract and are not measurements of named external models.*
