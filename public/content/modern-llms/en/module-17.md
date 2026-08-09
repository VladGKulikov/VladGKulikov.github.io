# Module 17. Evaluating Modern LLM Systems: From Benchmarks to Decisions

*“Modern LLMs” course · Module 17 lecture · edition 2026.8*

> **What this module is about.** The earlier modules built the objects we now have to choose between: pretrained models, quantized serving stacks, retrieval systems, post-trained policies, multimodal models, and agents. Evaluation is the discipline that turns those artifacts into a release decision. It is not merely a leaderboard lookup. A defensible result joins a decision, a task distribution, an execution scaffold, a scoring rule, and a statement of uncertainty. We will follow a single practical question throughout the lecture: which of two candidate systems should a team deploy for a real product?
>
> Public benchmarks will provide orientation, but the decision will eventually depend on paired outcomes, a private golden set, judge calibration, long-context diagnostics, operational cost, and continuous regression testing. Five frozen numerical scenarios make those ideas concrete: 95.5% versus 95.0% on 500 items, finite-sample `pass@k`, a position-biased judge, a synthetic working-context curve, and a costed private evaluation run.
>
> **Prerequisites.** Module 16 introduced agent scaffolds and trajectory logging; Modules 9 and 15 covered long context and retrieval; Module 13 introduced verifiers and test-time sampling. Every formula needed here is derived again, so the module can also be read on its own.

---

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

![VIZ m17/01 — the evolution of LLM evaluation](assets/modern-llms/en/module-17/m17_01_timeline.svg)

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

![VIZ m17/02 — the benchmark life cycle](assets/modern-llms/en/module-17/m17_02_paradigm.svg)

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

![VIZ m17/03 — an evaluation evidence map](assets/modern-llms/en/module-17/m17_03_map.svg)

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

## 6. Worked example A: are 95.5% and 95.0% distinguishable?

Consider two anonymous systems evaluated on 500 binary items. They score 95.5% and 95.0%. This is an illustrative near-ceiling scenario, not a claim about a particular leaderboard.

If only aggregate rates are available, a rough independent-rate standard error is

$$
SE_\Delta=
\sqrt{\frac{p_1(1-p_1)}{N}+\frac{p_2(1-p_2)}{N}}.
$$

With $p_1=0.955$, $p_2=0.950$, and $N=500$,

$$
SE_\Delta=0.013452.
$$

The observed difference is 0.005, giving

$$
z=\frac{0.005}{0.013452}=0.372.
$$

A rough 95% interval for the difference is approximately

$$
0.5\pm2.64\ \text{percentage points},
$$

which clearly includes zero.

![VIZ m17/04 — sample size and discriminating power](assets/modern-llms/en/module-17/m17_04_saturation.png)

The calculation is deliberately incomplete because both systems saw the same tasks. Correct inference should be paired. Suppose a 200-item comparison yields:

- both correct: 150;
- only A correct: 20;
- only B correct: 10;
- both wrong: 20.

The accuracies are 85% and 80%, but the exact two-sided McNemar $p$-value is about 0.099. A five-point aggregate gap is not automatically strong evidence on a small paired set.

For planning intuition, an independent two-rate approximation near success rate $p$ is

$$
N\approx2p(1-p)\left(\frac{1.96}{\delta}\right)^2.
$$

At $p=0.95$, it gives 146 items for a five-point difference, 913 for two points, 3,650 for one point, and 14,599 for half a point. Pairing and the pattern of disagreements can alter the requirement substantially, but the $1/\delta^2$ growth remains a useful warning.

The engineering lesson is straightforward: inspect $N$, retain item-level outcomes, and report uncertainty before debating leaderboard rank.

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

When tasks are automatically verifiable, a system may generate several candidates. Given $n$ stored samples of which $c$ are correct, the finite-sample unbiased estimator for at least one success among $k$ draws is

$$
\operatorname{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

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

Open-ended assistant quality is naturally comparative. LMArena presents two anonymous outputs and asks a user which one they prefer. A Bradley–Terry model converts many pairwise votes into a ranking.

The platform supplies valuable evidence because prompts come from real users and no exact reference answer is required. Yet the resulting ranking describes the arena population and protocol. It is not automatically the utility function of your organization.

Rankings depend on topic and language mix, model exposure, user heterogeneity, response length, presentation, model revisions under stable names, and statistical robustness of the aggregation. Recent work also shows that top Bradley–Terry rankings can be sensitive to removing a small adversarially chosen subset of votes.

This does not make arenas useless. It makes their scope explicit: they are broad human-preference instruments.

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

![VIZ m17/05 — an auditable judge protocol](assets/modern-llms/en/module-17/m17_05_judge.png)

A robust swap protocol maps positional labels back to candidate identity.

- If the same candidate wins in both orders, the preference is consistent.
- If the judge always chooses the first position, the identity-level verdict conflicts and the pair becomes a tie or abstention.
- If one ordering produces a tie, the protocol preserves both outcomes and applies a predeclared rule.

Swapping addresses position bias only. A stronger protocol also includes:

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

### Truthfulness and honesty

TruthfulQA tests whether models reproduce common human misconceptions. That behavior depends on knowledge and response policy. **MASK** makes a different distinction: whether a model’s belief is correct and whether it reports what it believes are separate questions.

A product evaluation may therefore need independent measures for factual correctness, calibration, willingness to admit missing evidence, robustness to false premises, tool-result faithfulness, and action safety.

The lesson is not to maximize the number of benchmarks. It is to prevent an average from erasing the failure mode that matters most.

## 12. Long context: accepted length versus usable evidence

A context-window specification tells us that a system accepts an input. It does not tell us how uniformly the model retrieves, aggregates, or reasons over that input.

**NIAH — Needle-in-a-Haystack** is a useful basic addressability test, but retrieving one distinctive string is a weak proxy for real document work. **RULER** adds multiple needles, tracing, and aggregation. **NoCha** uses long narratives and global understanding. **LongBench v2** includes realistic tasks over documents, dialogue history, repositories, and structured data.

Position also matters. *Lost in the Middle* found that relevant evidence placed near the beginning or end was often used more effectively than evidence in the middle.

The module’s numerical example is a synthetic profile, not a measurement of a named model:

$$
a(x)=0.95-0.35\sin^2(\pi x),\qquad x\in[0,1].
$$

Accuracy is 0.95 at both edges and 0.60 in the center. Mean accuracy is

$$
\bar a=0.775.
$$

Define working context as the share of positions satisfying $a(x)\ge0.9$. Then

$$
\text{working share}=0.246752.
$$

For a nominal one-million-token window, the corresponding length is approximately 246,752 positions.

![VIZ m17/06 — nominal and working context](assets/modern-llms/en/module-17/m17_06_lost_middle.png)

There is no task-independent “real context size.” The answer depends on retrieval versus aggregation, number and position of evidence pieces, document format, answer length, retrieval assistance, prompt design, reasoning budget, and the chosen quality threshold.

A good long-context report plots performance across lengths, positions, and task types instead of reporting only the maximum accepted input.

## 13. A minimal evaluation harness: one call, one observed output

An **evaluation harness** connects a model, examples, and metrics. Its most important elementary contract is:

> **Each example invokes the model exactly once.**

The output is stored and every metric is computed from that same observation. Calling a stochastic model separately for exact match, a judge, and a length metric would evaluate different generations while pretending they were one answer.

```python
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: str
    expected: str
    slice_name: str

def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip().casefold() == expected.strip().casefold())

def evaluate(
    model_fn: Callable[[str], str],
    dataset: Sequence[Example],
    metrics: Mapping[str, Callable[[str, str], float]],
) -> dict:
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if len({ex.example_id for ex in dataset}) != len(dataset):
        raise ValueError("example_id values must be unique")

    records = []
    for ex in dataset:
        prediction = model_fn(ex.prompt)  # the only model call for this item
        scores = {
            name: metric(prediction, ex.expected)
            for name, metric in metrics.items()
        }
        records.append({
            "id": ex.example_id,
            "slice": ex.slice_name,
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

This is not yet a production platform, but it already preserves unique item IDs, raw predictions, slices, multiple metrics over one output, and the call count. A mature harness adds model and prompt versions, sampling parameters, latency, cost, failures, traces, and budgets.

### Planning the golden-set size

For an expected success rate $p=0.8$ and desired 95% half-width $m$, a normal planning approximation is

$$
N\approx p(1-p)\left(\frac{1.96}{m}\right)^2.
$$

It gives 683 examples for a three-point half-width and 246 for five points. These numbers describe the overall rate, not every rare slice. Critical low-frequency cases should be deliberately oversampled and reported separately.

### What a canary can tell us

A uniformly random 32-character hexadecimal canary has chance-match probability

$$
16^{-32}=2^{-128}\approx2.94\cdot10^{-39}.
$$

Exact reproduction is strong evidence that the system encountered the string through some channel: training, logs, a cache, prompt leakage, or another data path. It does not by itself identify the channel or prove pretraining contamination. Canaries complement provenance and access controls.

### Interpreting an arithmetic cost floor

For 700 examples averaging 400 input and 150 output tokens, total token volume is 385,000. A dense $2P$-per-token estimate for an 8.03B model is

$$
C=6.1831\cdot10^{15}\ \text{FLOP}.
$$

At an assumed effective 495 TFLOP/s, the arithmetic floor is 12.49 seconds. Real wall-clock time includes autoregressive decode, batching, network calls, tools, scheduling, and judges. The estimate shows that frequent evaluation may be affordable; only measurement on the target stack determines the actual duration.

## 15. A private golden set in seven deliberate steps

![VIZ m17/07 — building a product golden set](assets/modern-llms/en/module-17/m17_07_golden_pipeline.svg)

Public benchmarks position a system in the general landscape. A private **golden set** determines whether it serves your product.

### Step 1: state the decision and unit of analysis

Is the result selecting a model, approving a prompt revision, gating a release, or tuning a router? The unit may be a response, a conversation, an agent trajectory, a document, or a user session.

### Step 2: sample the product distribution

Include common requests and deliberately oversample rare critical cases. Report both natural-weighted performance and slice results. A set made only of adversarial examples is valuable for red teaming but does not estimate normal user experience.

### Step 3: choose a metric that matches the task

Classification can use accuracy or F1. Structured outputs need schema and semantic-field checks. Code should be executed against hidden tests in a sandbox. Open-ended responses need human pairwise comparison or a calibrated judge. Agents need final environment state, trajectory quality, failures, cost, and repeatability.

### Step 4: freeze the protocol before seeing the outcome

Record the model revision, system prompt, tools, temperature, retry policy, reasoning budget, judge rubric, and failure handling. Otherwise protocol tuning can silently choose the winner.

### Step 5: protect the held-out data

Keep the decisive split out of training corpora, public examples, and logs reused for training. Canaries detect some leakage paths, but provenance, access control, and data-retention policy remain primary.

### Step 6: retain trajectories and failure causes

For agents, final success is too coarse. Save tool calls, intermediate state, retries, errors, and recovery. A system may reach the goal through an unsafe action, or fail because the environment—not the model—was defective.

### Step 7: turn evaluation into continuous regression testing

**CI/CD — Continuous Integration and Continuous Delivery** can run a fast subset on every change and a full suite before release. Prompt edits, retrieval-index changes, tool updates, and dependency upgrades can regress behavior independently of the weights.

The platform is secondary to the contracts. `lm-evaluation-harness` is widely used for academic tasks; Inspect AI supports configurable and safety-oriented evaluations; OpenAI Evals provides programmable eval patterns; LangSmith, Langfuse, and Phoenix support trajectory observability. Whatever the stack, preserve versions, raw records, and reproducible reports.

## 16. Eight directions shaping evaluation as of August 2026

Rather than memorizing a current list of “best benchmarks,” track the directions changing measurement practice.

### 1. Living and refreshed datasets

LiveBench, LiveCodeBench, and domain-specific variants reduce the contamination window. Scores must be tied to a benchmark version and date because successive releases are not identical experiments.

### 2. Private and centrally executed evaluation

ARC-AGI, NoCha, and professional suites retain evaluation material or run submissions centrally. This reduces direct tuning pressure while increasing the need to trust and document the organizer’s infrastructure.

### 3. Controlled mutation and counterfactual items

Benchmark mutation changes surface form while preserving a verifiable target skill. It can separate memorization from generalization, but the mutation procedure itself needs validation.

### 4. Trajectories replace terminal answers

SWE-bench, OSWorld, and $\tau$-bench evaluate changes to an environment. Newer variants such as SWE-bench Pro and OSWorld 2.0 extend horizon and professional complexity. Longer trajectories raise the importance of repeatability, environment state, and complete logs.

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

This mirrors Module 16. There, the scaffold transformed a model into an agent. Here, the harness transforms tasks into evidence about a system. In both cases, the model name alone is insufficient.

## 20. Key takeaways, course finale, and sources

![VIZ m17/08 — evaluation in one page](assets/modern-llms/en/module-17/m17_08_cheatsheet.svg)

Modern LLM evaluation is a chain of evidence, not a hunt for one authoritative number.

- **Start from the decision.** Desired behavior and acceptable risk determine the task set.
- **Separate model, system, and product.** A public score normally describes one configuration, not every deployment.
- **Read the benchmark life cycle.** Protocol adaptation, contamination, saturation, and item defects reduce discriminating power.
- **Keep item-level outcomes.** Paired evidence is more informative than two independent means.
- **Report uncertainty.** In the teaching scenario, 95.5% versus 95.0% on 500 items gives $z=0.372$, not a decisive victory.
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

### End of the main route

Seventeen modules have covered the entire stack: tokenization, attention, MoE, data, pretraining, serving, post-training, multimodality, retrieval, agents, and finally evidence about system behavior. The course’s recurring idea is causal integration: architecture, data, optimization, runtime, and evaluation must be understood together. A change is an improvement only when its consequences are measured elsewhere in the stack.

[Module 18](../../module-18/en/module_18_lecture_EN.md) is a reference lesson on the basic Transformer. A newcomer may read it before Module 1; after completing the course, it serves as a compact map of equations, tensor shapes, and implementation invariants.

---

*Landscape verified: 5 August 2026. The lecture intentionally omits a current frontier leaderboard: those values change faster than a stable course and depend on scaffold version, inference budget, and evaluation date. All numerical examples in Sections 6, 8, 10, 12, and 13 belong to the frozen educational contract and are not measurements of named external models.*
