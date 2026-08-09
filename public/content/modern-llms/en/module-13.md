# Module 13. Reasoning Models and Test-Time Compute

> **What this module is about.** The previous modules improved a model before deployment: pretraining built the base policy, supervised fine-tuning shaped its interface, and feedback-based learning changed which outputs it prefers. This module studies a different lever. Once the weights are fixed, a system can still spend more computation on a particular request—by extending one solution path, sampling several alternatives, checking them, or routing difficult prompts to a more expensive procedure.
>
> These mechanisms are often grouped under *test-time compute*, but they buy different things. A longer trajectory deepens a single search. Parallel candidates widen the search distribution. A verifier turns candidate generation into selection. An adaptive controller decides which kind of work is worth paying for on each prompt.
>
> We will treat quality and systems cost together. The module therefore follows reasoning tokens all the way down to decode latency, FLOPs, KV memory, and concurrency. Numerical serving figures are explicit teaching scenarios, not performance claims about a particular vendor or accelerator.

> **Further study.** Everything required for this module is developed here. The companion course *RL for LLM* goes deeper into RLVR, GRPO, verifier-driven training, and interactive environments, especially in Modules 9–10. *Information Theory for ML* provides a broader treatment of ensembles, uncertainty, and the value of additional observations.

---

## 1. A second scaling axis

A large language model is usually evaluated as though it had one chance: one prompt enters, one completion leaves. That convention hides an important design choice. We can also ask how well the same trained model performs when it is allowed to spend more computation on the instance in front of it.

There are three broad ways to spend that budget.

- **Go deeper.** Continue one trajectory, revise an intermediate step, run a longer plan, or interleave reasoning with tools.
- **Go wider.** Produce independent candidates, perhaps from one model or from several agents, and then aggregate or select.
- **Allocate adaptively.** Give easy inputs a short path and reserve expensive search for requests that appear difficult or uncertain.

All three need a decision rule. A long chain needs a stopping condition. A candidate pool needs a selector. An adaptive system needs a signal that says whether more work is likely to pay off.

This is why inference-time scaling is not merely “turning up intelligence.” It is an algorithm for allocating a finite resource. Snell et al. showed that the most effective allocation varied strongly with prompt difficulty. Under their experimental setup, compute-optimal routing used test-time computation more than four times as efficiently as a simple best-of-N baseline, and a smaller model could beat a model fourteen times larger on a subset of tasks under FLOP-matched evaluation. Those are study-specific results, but the underlying lesson is general: **the allocation policy is part of the method**. [Snell et al., 2024](https://arxiv.org/abs/2408.03314)

![VIZ m13/01 — the second scaling axis](assets/modern-llms/en/module-13/m13_01_timeline.svg)

---

## 2. From longer traces to explicit controls

Chain-of-thought prompting, self-consistency, and generate-and-test workflows predate the current generation of reasoning models. The 2024–2026 shift was more specific: model training and product interfaces began to treat inference computation as a resource that could be deliberately scaled.

OpenAI's September 2024 o1 announcement described performance improving along two axes: more reinforcement-learning compute during training and more time spent reasoning at test time. The public article did not disclose a full recipe, but it made inference-time scaling a first-class claim rather than a prompting trick. [OpenAI, 2024](https://openai.com/index/learning-to-reason-with-llms/)

DeepSeek-R1 supplied a much more inspectable training story in January 2025. R1-Zero used large-scale reinforcement learning on verifiable signals without a preliminary supervised phase. The model developed longer solutions, self-checking behavior, and course corrections, while also suffering from readability and language-mixing problems. The full R1 pipeline then combined cold-start data, reasoning RL, rejection sampling, renewed SFT, and broader RL objectives. [DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948)

By **4 August 2026**, inference controls had become a regular part of model APIs, but the contracts were not standardized:

- GPT-5.6 exposes an effort ladder from `none` through `max`, while single-model reasoning and multi-agent coordination remain distinct mechanisms;
- newer Claude models use adaptive thinking and effort controls, whereas some older models accepted a manual thinking-token target;
- Gemini 3 uses `thinkingLevel`, while Gemini 2.5 models use `thinkingBudget`, including dynamic behavior on supported variants;
- Kimi K3 always reasons and supports `low`, `high`, and `max` effort, with `max` as the documented default;
- Inkling was trained for controllable reasoning effort.

The labels are local to each API. A `high` setting does not imply a common token count, latency target, search algorithm, or billing rule. Documentation tells us what control surface exists; it rarely reveals the complete internal computation. [OpenAI API](https://developers.openai.com/api/docs/guides/latest-model) · [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models) · [Gemini API](https://ai.google.dev/gemini-api/docs/generate-content/thinking) · [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) · [Inkling](https://thinkingmachines.ai/news/introducing-inkling/)

The important historical change, then, is not that models began emitting intermediate text. It is that **reasoning depth, search breadth, and routing became deployable system parameters**.

---

## 3. Three older ideas hiding inside test-time compute

Modern reasoning systems recombine several classical patterns.

### Anytime computation

An anytime algorithm returns a usable answer early and improves it while resources remain. Search, planning, and numerical optimization all have examples. A well-designed reasoning model should exhibit the same property: extra time should be optional, and the system should be able to stop with a coherent result.

### Repeated stochastic search

If one attempt has a non-zero chance of success, independent restarts increase the chance that a successful path appears. This is the same intuition behind ensembles and randomized search. Yet a pool containing a correct answer is useful only if the system can identify it.

### Generate and test

Program synthesis, theorem proving, and planning often separate proposal from validation. Generate a candidate; run a compiler, unit tests, a proof kernel, or a simulator; keep what passes. This arrangement is especially attractive when generation is uncertain but checking is cheap and exact.

The mapping is straightforward:

- sequential reasoning is a form of deeper search;
- multiple completions form a stochastic ensemble;
- a verifier supplies the acceptance or ranking rule;
- an adaptive controller is a meta-policy over those choices.

![VIZ m13/02 — depth, breadth, and adaptation](assets/modern-llms/en/module-13/m13_02_modes.svg)

---

## 4. What counts as a reasoning model?

The phrase can easily be overinterpreted. A visible chain of text is not guaranteed to be a faithful transcript of internal computation, and a hidden chain is not automatically more rigorous. We will use a functional definition:

> A **reasoning model** is trained or deployed so that additional computation before the final answer can be used productively.

That property has several separable dimensions.

**Representation.** Intermediate work may be exposed, hidden, summarized, placed in a dedicated message field, or represented by opaque signatures needed for continuity. A `<think>` token in a chat template establishes a format, not a capability guarantee.

**Control.** The user may specify a token target, select an effort level, enable an adaptive mode, or receive no direct control at all.

**Topology.** More compute can mean a longer path from one model, a wider pool of candidates, or orchestration across subagents. GPT-5.6 documentation, for example, distinguishes additional work by one model from multi-agent coordination. Muse Spark 1.1 similarly combines planning and delegation, which is a different resource from merely extending a single completion. [OpenAI API](https://developers.openai.com/api/docs/guides/latest-model) · [Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/)

**Accounting.** Reasoning tokens can consume output limits and be billed even when only a summary is visible. Gemini reports thought-token usage separately in metadata while charging for the underlying generated thinking tokens. [Gemini API](https://ai.google.dev/gemini-api/docs/generate-content/thinking)

**State across turns.** Some APIs require the application to return complete assistant messages, thought signatures, or reasoning items after tool calls. Dropping those fields may discard useful state or violate the protocol.

A proper comparison therefore needs more than an effort label. It needs the depth of one trajectory, the number of parallel attempts, the selection rule, the visibility contract, and the accounting model.

---

## 5. What DeepSeek-R1 teaches us about training reasoning

DeepSeek-R1 is useful because its report separates an exploratory RL result from a deployable multi-stage system.

### R1-Zero as a controlled experiment

R1-Zero did not receive a supervised warm-up that taught it how a long solution should look. It was optimized with reinforcement learning on verifiable rewards and formatting constraints. During training, solutions became longer and exhibited behaviors such as checking previous work and trying another route.

The defensible conclusion is not that a language model acquired human cognition. It is narrower: **a policy can discover token-level computational strategies when exploration is coupled to an informative, verifiable reward**.

The experiment also exposed the cost of optimizing correctness in isolation. Outputs could be hard to read, switch languages, or use an inconvenient format. A useful assistant needs both problem-solving behavior and a stable human-facing interface.

### The full R1 pipeline

The complete recipe adds several stages:

1. a small cold-start SFT set for readable reasoning behavior;
2. reasoning-focused reinforcement learning;
3. selection of successful trajectories followed by another supervised phase on a broader mixture;
4. additional reinforcement learning for general helpfulness, safety, and instruction following.

This is a good antidote to the false choice between SFT and RL. Supervised data establish a usable format; RL explores under reward; selected trajectories become new data; broader post-training restores the full interaction contract.

![VIZ m13/03 — the DeepSeek-R1 recipe](assets/modern-llms/en/module-13/m13_03_r1_pipeline.svg)

---

## 6. Parallel scaling, worked example A: voting can get worse first

Assume that one independent completion is correct with probability

$$
p=0.4.
$$

When it is wrong, the answer is uniformly distributed across six distinct distractors. We draw $N$ completions and return the most frequent normalized answer.

The procedure is still underspecified. What happens when several answers share the top count? We compare two exact policies:

- **tie means failure**: the correct answer must be the unique mode;
- **random top-mode selection**: choose uniformly among all modes tied for first place.

Enumerating the multinomial outcomes gives:

| $N$ | unique-mode success | random tie-breaking |
|---:|---:|---:|
| 1 | 0.400 | 0.400 |
| 3 | 0.352 | 0.472 |
| 5 | 0.509 | 0.596 |
| 9 | 0.671 | 0.737 |
| 17 | 0.857 | 0.889 |

At $N=3$, a top-mode tie occurs with probability 0.48. Under the strict rule, three samples are worse than one. The result is not a paradox; it is a reminder that tie handling is part of the algorithm.

A reproducible self-consistency protocol must also define answer normalization. Are `0.5`, `1/2`, and `50%` equivalent? Are algebraically equivalent expressions canonicalized? Are code outputs compared by text or by execution? Small implementation choices can dominate the apparent gain from extra sampling.

![VIZ m13/04 — voting and verification](assets/modern-llms/en/module-13/m13_04_maj_vs_verifier.png)

The *Large Language Monkeys* study showed that repeated sampling can improve coverage over very large ranges of sample count. In automatically verifiable domains, coverage can translate directly into solved tasks. Where selection relies on majority vote or imperfect reward models, the selector can plateau even while the generator continues to produce additional correct candidates. [Brown et al., 2024](https://arxiv.org/abs/2407.21787)

---

## 7. Worked example B: coverage is not selection accuracy

Voting uses frequency as a proxy for correctness. A verifier uses evidence about the candidate itself. Before modeling an imperfect verifier, we need an upper bound.

For independent attempts with success probability $p$, the probability that at least one of $N$ candidates is correct is

$$
P_{\mathrm{coverage}}(N)=1-(1-p)^N.
$$

With $p=0.4$:

| $N$ | oracle coverage |
|---:|---:|
| 1 | 0.400 |
| 3 | 0.784 |
| 5 | 0.922 |
| 9 | 0.990 |
| 17 | 0.9998 |

This curve tells us what an oracle *could* recover. It says nothing about whether an actual ranking model will find the correct item.

Now consider a simple accept/reject verifier. A correct candidate is accepted with true-positive rate 0.90; an incorrect candidate is accepted with false-positive rate 0.05. Selection is uniform among accepted candidates, with a random-candidate fallback if none is accepted. The exact success probabilities are:

| $N$ | noisy-verifier accuracy |
|---:|---:|
| 1 | 0.400 |
| 3 | 0.728 |
| 5 | 0.851 |
| 9 | 0.913 |
| 17 | 0.923 |

The early gains are substantial, but the curve eventually becomes verifier-limited. More candidates also create more opportunities for a wrong solution to trigger a false positive. In a richer setting, search can actively discover outputs that exploit the judge—the inference-time counterpart of reward hacking.

A sensible verification hierarchy is:

1. a formal checker or deterministic executable rule;
2. unit tests and explicit constraints;
3. a specialized process or outcome model;
4. a general LLM judge with a documented rubric;
5. voting when no stronger signal is available.

An LLM judge should not replace a compiler or proof kernel merely because it is convenient.

---

## 8. Sequential scaling and budget forcing

Parallel inference restarts the search. Sequential inference gives one trajectory more room to evolve.

The s1 paper provides a clean demonstration. The authors curated 1,000 difficult, diverse, high-quality reasoning examples, fine-tuned Qwen2.5-32B-Instruct, and then controlled the length of its reasoning without another weight update. Their **budget forcing** procedure could:

- extend a trace by appending `Wait` when the model attempted to finish;
- shorten a trace by forcing the transition to the final answer.

Under the reported AIME 2024 setup, increasing the budget raised s1-32B from 50% to 57%. That is evidence for one model and evaluation protocol, not a law that every task improves with every extra token. [Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)

The mechanism changes decoding, not knowledge. It can help when the model already contains the relevant operations but would otherwise stop before applying them. It cannot manufacture a missing theorem, API fact, or representation.

Sequential compute can support several policies: continue the current derivation, insert explicit checks, revisit a branch point, alternate planning with tool execution, or stop early once evidence is sufficient. A mature system needs the last operation as much as the first. Scaling inference includes learning when *not* to continue.

---

## 9. Failure modes of longer reasoning

Extra tokens are useful only if they change the computation in a productive direction.

**Overthinking.** A model may reach the right answer quickly, then revise it into a wrong one. Empirical work has found non-monotonic relationships between chain length and accuracy, with shorter selections sometimes outperforming longer traces. The effect depends on the model, task, and selection protocol. [Chen et al., 2025](https://arxiv.org/abs/2505.17813)

**Redundant continuation.** The model may restate the same hypothesis without creating new search state.

**Error accumulation.** An early false premise becomes part of the context and shapes every later step. Fluency can increase even as correctness falls.

**Verifier exploitation.** A search process can optimize whatever signal it sees. If the verifier rewards persuasive form, more search may find more persuasive errors.

The module notebook contains a deliberately synthetic illustration: difficult tasks improve with additional budget, while easy-task accuracy eventually declines. The curves demonstrate a possible mechanism; they are not measurements of a deployed model.

![VIZ m13/05 — test-time compute paradigms](assets/modern-llms/en/module-13/m13_05_paradigms.svg)

![VIZ m13/06 — limits and failure modes](assets/modern-llms/en/module-13/m13_06_pitfalls.svg)

---

## 10. Worked example C: an illustrative budget ladder

Provider effort labels do not map to a universal token count. To reason about systems cost, we therefore use an explicit course scenario rather than pretending to reconstruct an API.

Assumptions:

- a dense $8.03\times10^9$-parameter decoder;
- 209.4 sequential decode tokens per second, fixed as a course assumption;
- 512 tokens for a short baseline answer;
- illustrative budgets of 1K, 4K, 16K, and 64K generated tokens.

A simple dense-decode FLOP floor is

$$
C_{\text{decode}}\approx 2N_pT,
$$

and unbatched latency is

$$
t\approx\frac{T}{209.4}.
$$

| illustrative tier | tokens | latency | dense-decode FLOP floor | vs. 512 tokens |
|---|---:|---:|---:|---:|
| short baseline | 512 | 2.45 s | $8.22\times10^{12}$ | 1× |
| small | 1,000 | 4.78 s | $1.61\times10^{13}$ | 1.95× |
| medium | 4,000 | 19.10 s | $6.42\times10^{13}$ | 7.81× |
| large | 16,000 | 76.41 s | $2.57\times10^{14}$ | 31.25× |
| very large | 64,000 | 305.64 s | $1.03\times10^{15}$ | 125× |

The arithmetic scales linearly with generated length. The user experience does not. Five seconds may still be interactive; five minutes implies a different product and scheduling regime.

Extending a trajectory from 2K to 8K tokens adds

$$
\Delta t=\frac{8000-2000}{209.4}\approx28.65\text{ s}.
$$

Whether that is worthwhile cannot be decided from the cost equation. It requires an empirical quality curve on the target workload.

---

## 11. Worked example D: the KV cost of a thought

Every generated token also leaves K and V vectors in every attention layer.

Take an illustrative model with 32 layers, KV width 1,024 per layer, and bf16 storage. The KV bytes per token are

$$
M_{\text{token}}
=2\;(K,V)\times32\times1024\times2
=131072\text{ bytes}=128\text{ KiB}.
$$

| reasoning length | KV per request |
|---:|---:|
| 512 | 0.0625 GiB |
| 1K | 0.122 GiB |
| 4K | 0.488 GiB |
| 16K | 1.953 GiB |
| 64K | 7.8125 GiB |

The bf16 weights of an 8.03B model occupy about 14.96 GiB. If an 80 GiB device devoted every remaining byte to KV, the resulting upper bounds would be 1,040 concurrent 512-token requests, 532 at 1K, 133 at 4K, 33 at 16K, and only 8 at 64K.

These are not throughput forecasts. They omit activations, fragmentation, scheduler state, batching, and runtime reserve. Reserving 8 GiB lowers the 1K–64K figures to 467, 116, 29, and 7. The calculation still exposes the order of magnitude: moving from 512 to 64K tokens reduces a KV-limited concurrency bound by roughly 130×.

This is why the serving techniques from Module 9—paged allocation, KV quantization, offload, and tiered storage—become central for reasoning workloads rather than optional optimizations.

![VIZ m13/07 — the economics of long reasoning](assets/modern-llms/en/module-13/m13_07_economics.png)

---

## 12. Worked example E: why adaptive allocation matters

Suppose a workload has three classes:

- 70% easy requests needing about 500 tokens;
- 20% medium requests needing about 4,000;
- 10% hard requests needing about 32,000.

Assigning the maximum budget to every request costs 32,000 tokens on average. A perfectly routed mixture costs

$$
\mathbb{E}[T]
=0.7(500)+0.2(4000)+0.1(32000)
=4350.
$$

The ratio is

$$
\frac{32000}{4350}\approx7.36.
$$

This is mixture arithmetic, not a measured router. It explains why a single fixed budget is structurally inefficient when prompt difficulty varies.

Possible routing signals include a cheap difficulty classifier, uncertainty after a short first pass, agreement between samples, a verifier threshold, or a signal from the environment such as passing tests.

The notebook includes a synthetic threshold-stopping experiment. Candidates arrive one at a time; generation stops when a score exceeds 0.82 or after 16 attempts. On the fixed simulated workload, the adaptive policy uses 4.02 candidates on average and reaches 0.886 accuracy, compared with 0.632 for one shot. Fixed best-of-16 reaches 0.940 but always pays for all sixteen candidates. The values illustrate a mechanism, not a real verifier benchmark.

Adaptive inference is therefore a classification-and-control problem under noise. A router can waste compute on easy inputs or prematurely stop hard ones. Its error costs belong in the evaluation.

---

## 13. Level-1 code: make selection and cost assumptions explicit

The first function computes oracle coverage. It deliberately does not claim to model a selector.

```python
from __future__ import annotations

def oracle_coverage(p_correct: float, n: int) -> float:
    """Probability that at least one of n independent candidates is correct."""
    if not 0.0 <= p_correct <= 1.0:
        raise ValueError("p_correct must lie in [0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - (1.0 - p_correct) ** n

for n in (1, 3, 5, 9, 17):
    print(n, f"{oracle_coverage(0.4, n):.6f}")
```

The second block records the serving assumptions in a dataclass rather than burying them in a benchmark-like number.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ReasoningCost:
    parameters: float = 8.03e9
    tokens_per_second: float = 209.4
    layers: int = 32
    kv_width: int = 1024
    bytes_per_element: int = 2

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_width * self.bytes_per_element

    def estimate(self, tokens: int) -> dict[str, float]:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        return {
            "latency_s": tokens / self.tokens_per_second,
            "decode_flops_floor": 2.0 * self.parameters * tokens,
            "kv_gib": self.kv_bytes_per_token * tokens / 2**30,
        }

scenario = ReasoningCost()
for budget in (512, 1_000, 4_000, 16_000, 64_000):
    print(budget, scenario.estimate(budget))
```

The code is intentionally a floor model. Its value lies in making units and assumptions inspectable before real measurements are added.

---

## 15. Verifier families

A verifier is not a single architectural object. Different tasks support different evidence.

### Exact executable checks

A compiler, unit-test suite, symbolic algebra system, proof kernel, or environment simulator can give a transparent signal. Such checks may still be incomplete—tests do not cover every behavior—but their semantics are usually clearer than a learned judge.

### Outcome reward models

An outcome model scores a complete solution. Labels are relatively cheap, but the feedback is sparse: a poor score does not identify the failing step.

### Process reward models

A PRM scores intermediate steps. *Let's Verify Step by Step* released PRM800K, containing 800,000 human step-level labels, and reported process supervision outperforming outcome supervision in its mathematical setting. The denser signal supports search and early pruning, but the annotation is expensive and domain-specific. [Lightman et al., 2023](https://arxiv.org/abs/2305.20050)

### Generative verifiers

GenRM casts verification as next-token prediction. The model may produce a critique, while the probability of a verdict token supplies a score. This makes chain-of-thought and repeated sampling available to the verifier itself. The published experiments improved Best-of-N selection over several discriminative baselines under the paper's protocols; that result should not be universalized to every domain. [Zhang et al., 2024](https://arxiv.org/abs/2408.15240)

### Reasoning reward models

RM-R1 treats judging as a reasoning task. It generates rubrics or analyses before the verdict and uses a two-stage recipe involving reasoning-trace distillation and RLVR. It is a research family with released artifacts, not a mandatory endpoint for all reward modeling. [Chen et al., 2025](https://arxiv.org/abs/2505.02387)

![VIZ m13/08 — verifier families](assets/modern-llms/en/module-13/m13_08_verifier_stack.svg)

Reasoning RL is most straightforward when rewards are objectively checkable. Creative writing, advice, and aesthetics do not have a unique ground truth; there, a learned verifier represents a preference or rubric rather than mathematical correctness.

---

## 16. Distilling reasoning traces

DeepSeek released six dense R1-Distill models ranging from 1.5B to 70B, trained using data generated by R1. The release made a practical route visible: successful long-form solutions from a strong teacher can become supervised data for a smaller student. [DeepSeek-R1, 2025](https://arxiv.org/abs/2501.12948)

A student can absorb:

- the surface format of decomposition;
- recurring intermediate operations;
- habits of checking and revision;
- length and style distributions;
- domain heuristics represented in the traces.

It does not receive the teacher's hidden parameters or full capability distribution. Distillation teaches observable trajectories. If the student's base model lacks the relevant knowledge or representations, a longer imitation will not create them.

It helps to separate three claims:

1. **format transfer**: the model produces an organized reasoning trace;
2. **procedure transfer**: it applies patterns present in the training data;
3. **out-of-distribution generalization**: this still depends on base capacity, data diversity, and often further reinforcement learning.

A “reasoning distillate” should therefore be evaluated as its own model, including quality, token efficiency, calibration, and failure modes.

![VIZ m13/09 — reasoning distillation](assets/modern-llms/en/module-13/m13_09_distill_2026.svg)

---

## 17. Reading a model's reasoning passport

As of 4 August 2026, “supports reasoning” is not enough information for deployment. A useful passport answers at least seven questions.

1. **How is reasoning enabled?** Is it always on, adaptive, selected by effort level, or controlled by a token target? Can it be disabled?
2. **What kind of compute scales?** One longer trajectory, several candidates, subagents, or a combination?
3. **What is exposed?** Full traces, summaries, a separate field, opaque signatures, or only the final answer?
4. **What is billed and limited?** Do reasoning tokens count as output and consume `max_tokens`? Are they reported in usage metadata?
5. **How does it interact with tools?** Must the client preserve complete assistant messages, signatures, or reasoning items across calls?
6. **What are the systems consequences?** Measure median and tail latency, reasoning length, KV footprint, and concurrency on the actual workload.
7. **How was quality compared?** Equal cost, equal wall-clock time, equal generated tokens, or unrestricted settings? Were tools and multiple agents allowed?

This passport separates model capability from serving policy. It also prevents accidental comparisons between single-trajectory depth and multi-agent breadth.

---

## 20. Key takeaways and sources

- Test-time compute is a real scaling axis, but not a free substitute for a capable base model.
- Sequential depth, parallel breadth, and adaptive allocation address different failure modes.
- Repeated sampling raises coverage; selection quality determines how much of that coverage becomes accuracy.
- Tie handling, answer normalization, and sample dependence are part of the self-consistency algorithm.
- A verifier can become the bottleneck—and can itself be exploited.
- Decode time and KV memory grow roughly linearly with generated reasoning length, while the product impact on latency and concurrency can be severe.
- Adaptive budgets are valuable because workload difficulty is heterogeneous and quality need not be monotonic in chain length.
- Distillation transfers visible procedures and traces, not the teacher's full capability.
- Comparing reasoning systems requires a passport covering controls, depth versus breadth, visibility, billing, tool continuity, systems cost, and evaluation protocol.

**Primary sources:**

- [OpenAI — Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787)
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Generative Verifiers](https://arxiv.org/abs/2408.15240)
- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)
- [OpenAI API model guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [Anthropic thinking documentation](https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models)
- [Gemini API thinking documentation](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [Kimi K3 quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Thinking Machines — Inkling](https://thinkingmachines.ai/news/introducing-inkling/)
- [Meta — Muse Spark 1.1](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/)

**Snapshot for current API details:** 4 August 2026.

---

*Landscape verified: August 4, 2026. Current API controls were checked against official documentation and research results against primary publications. Teaching estimates for latency, memory, and cost are reproduced by local calculations and apply only under the stated assumptions.*
