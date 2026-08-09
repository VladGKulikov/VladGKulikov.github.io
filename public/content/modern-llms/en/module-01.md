# Module 1. The Modern LLM Landscape

*“Modern LLMs” course · Module 1 lecture · edition 2026.8*

> **How to use this module.** Imagine that a model announcement lands this morning. It promises trillions of parameters, a million-token context window, a reasoning mode, and a new coding score. Our job is to turn that announcement into an auditable engineering passport: what must be stored, what is active per token, which artefacts exist, what the system can actually do, how expensive it is to run, and which fields are still undisclosed.
>
> That passport is also the map of the course. A field in `config.json`, a model card, or a serving contract will point forward to tokenization, attention, MoE, pretraining, inference, post-training, RAG, agents, and evaluation. Readers who want to assemble the Transformer from first principles can take [reference Module 18](../../module-18/en/module_18_lecture_EN.md) before returning here.
>
> **The wider series.** *Modern LLMs* is the engineering spine. *Information Theory for ML* develops the mathematical language behind cross-entropy, KL divergence, coding, and rate–distortion. *RL for LLM* follows policies from preferences and verifiable rewards into interactive environments and agentic RL. Each course remains self-contained; cross-references are invitations to go deeper, not hidden prerequisites.

---

## 1. Why an engineer needs a map of the landscape

It is tempting to skip the survey module: "just give me RoPE (Rotary Positional Embedding) and FlashAttention." The problem is that a modern LLM (Large Language Model) is not one idea but a **coherent package** of a dozen-plus decisions, and the point of each is visible only against the others. GQA (Grouped-Query Attention; module 4) exists because the KV cache (Key–Value cache; module 9) became a major inference cost. MoE (Mixture of Experts; module 6) increases total parameter capacity without increasing active parameters per token in proportion; data quality and scarcity influence scaling recipes, but they are not the sole reason MoE became widespread. Speculative decoding (module 10) grew out of the fact that decoding is often bound by memory rather than arithmetic. Without a map these stories look like a bag of tricks; with a map they are one coherent engineering logic.

The second reason is more prosaic. In 2026 models ship by the dozen per quarter, and announcement headlines ("1.6T parameters!", "10M context!", "95% on SWE-Bench (Software Engineering Benchmark)!") are useless without the skill of converting them into engineering quantities: how much memory that is, how many FLOPs per token, which part is marketing and which is a real shift. This module is a workout in exactly that conversion.

The third reason is professional hygiene. Choosing a model for a task ("what do we self-host?", "what is cheaper under our load profile?", "which license allows a product?") is a decision along several axes at once. Below we formalize those axes as a **model passport** and learn to fill it in from primary sources — configs and tech reports, not other people's retellings.

## 2. How the course reads code, and why a few systems recur

A model release can be studied at several levels. Confusing those levels creates two opposite mistakes: treating a twenty-line teaching implementation as production software, or dismissing readable reference code because it is not the fastest kernel.

| Code level | Question it answers | What it deliberately leaves out |
|---|---|---|
| **Minimal mechanism** | What is the smallest program that exposes the idea? | hardware optimization, distributed execution, operational safeguards |
| **Engineering notebook** | Do shapes, units, edge cases, and numerical contracts hold? | multi-tenant reliability, full observability, deployment lifecycle |
| **Real implementation** | How is the mechanism expressed under actual hardware and product constraints? | pedagogical simplicity |

The course repeatedly moves down this ladder. We derive a mechanism on a small example, exercise it in a deterministic notebook, and then inspect a real implementation. The production artefact is not “more correct” mathematically; it answers a larger systems question.

A small set of reference systems gives the course continuity.

| Reference | Why it is useful | Where it returns |
|---|---|---|
| **gpt-oss-20b / 120b** | Open weights and a readable PyTorch path alongside optimized Triton and Metal implementations; one family exposes GQA, hybrid attention, learned sinks, YaRN, MoE, MXFP4, and Harmony | Modules 2–10 and 16 |
| **DeepSeek family** | unusually detailed technical reports and open implementations for MLA, MoE routing, sparse attention, training, and reasoning | Modules 4, 6, 8, 12, and 13 |
| **OLMo family** | one of the strongest examples of an open training route—code, data, checkpoints, logs, and evaluation artefacts | Modules 8, 11, and 12 |
| **Inkling and Tinker** | a clean separation between open model weights and a managed training platform | Modules 6, 8, and 11 |

Three publication labels must remain distinct throughout the course:

- **open source**: the relevant implementation is available under an open licence;
- **open weights**: trained parameters are downloadable, while the original data or training stack may remain closed;
- **API-only**: the provider exposes behaviour, not the weight artefact or internal architecture.

These are not moral rankings. They determine what can be verified, modified, hosted, and reproduced.

## 3. Nine years in fifteen minutes: 2017 → 2026

![VIZ m1/01 — LLM evolution, 2017–2026](assets/modern-llms/en/module-01/m1_01_timeline_2017_2026.svg)

A useful history of LLMs is not a parade of release names. It is a history of changing bottlenecks.

**2017–2019: the sequence bottleneck.** The Transformer replaced recurrence with attention, making training over positions parallel. BERT established the bidirectional encoder, GPT established the causal decoder, and GPT-2 showed that a single generative objective could acquire broad language behaviour.

**2020–2021: the scale bottleneck.** GPT-3 made in-context learning visible at scale. Sparse expert models demonstrated that total capacity could grow faster than active computation. Scaling laws turned architectural ambition into a budget-allocation problem.

**2022: the behaviour bottleneck.** Chinchilla corrected the relation between parameters and training tokens. Instruction tuning and RLHF made base models useful as assistants. PaLM, Gopher, and BLOOM clarified both the promise and the operational cost of large dense models.

**2023: the inference bottleneck.** Llama widened access to strong open-weight models. FlashAttention, GQA, quantization, and efficient fine-tuning moved from specialist topics into the standard engineering stack. Long-context claims began to require separate memory and quality audits.

**2024: conditional and hybrid computation.** Mixtral and DeepSeek made MoE mainstream in open models; MLA attacked KV-cache directly; Mamba and hybrid recurrent architectures reopened the design space beyond pure softmax attention. Multimodality moved into the base model rather than living only in product glue.

**2025–2026: the system becomes the unit of analysis.** Reasoning effort, tool execution, multi-agent orchestration, managed adaptation, verifiable rewards, and long-running tasks increasingly determine what users experience. A model release is no longer only a checkpoint; it is a checkpoint embedded in a serving, safety, memory, and evaluation system.

A dated snapshot makes the last transition concrete without turning the history section into a news feed.

| Shift in 2025–2026 | Representative examples | New engineering object |
|---|---|---|
| Reasoning becomes a controllable product dimension | configurable effort, adaptive thinking, separate high-compute modes | test-time compute budget |
| One request becomes a set of cooperating trajectories | multi-agent and swarm modes | orchestration and merge cost |
| Open weights coexist with managed training | Inkling + Tinker | ownership of the artefact versus ownership of the infrastructure |
| Capability evaluation becomes an executable security environment | cyber evaluations and agent incidents | containment, monitoring, and evaluation security |

The stable lesson is not the identity of a temporary leader. It is that each generation turns a formerly hidden cost into an explicit field in the passport.

## 4. A bridge to the classics: taxonomy instead of a zoo

Biology becomes manageable when one stops memorizing every animal and starts classifying body plans. Model families need the same discipline. Names change quickly; recurring structural decisions change more slowly.

A useful taxonomy asks:

1. Is the backbone dense or conditionally sparse?
2. How are query, key, and value state represented?
3. What positional geometry is used?
4. Does token mixing rely on full attention, local attention, recurrence, diffusion, or a hybrid?
5. Which modalities enter and leave the system?
6. Which artefacts and rights are actually published?

This view prevents a common error: inferring architecture from a brand name. Two products in one family may use different checkpoints or serving paths; two unrelated families may share the same core mechanism. Classification follows evidence—configuration, code, technical report—not naming.

## 5. Two passports: the model and the system around it

A **model passport** describes the checkpoint and its first-order economics. A **system passport** describes the environment that turns the checkpoint into an assistant or agent. They must be recorded separately. The same weights in a constrained chat endpoint and in a cyber-evaluation harness with network access are not the same risk-bearing system.

![VIZ m1/02 — model and system passports](assets/modern-llms/en/module-01/m1_02_model_passport.svg)

### 5.1. The six axes of a model passport

| Axis | Record | Primary evidence | Course destination |
|---|---|---|---|
| **I. Parameters** | total, active, dense/MoE, artefact precision | config, model card, implementation | M4–M7, M10 |
| **II. Context** | nominal window, working quality, KV/state cost | config, long-context evaluation, runtime | M3–M4, M7, M9 |
| **III. Modalities** | inputs, outputs, encoder/connector/fusion | model card and code | M14 |
| **IV. Reasoning** | instant/thinking, effort control, adaptive or system-level mode | API/model card | M11–M13 |
| **V. Access** | weights, licence, gated access, training artefacts, jurisdiction | repository and provider documentation | M11, M16 |
| **VI. Economics** | token rates, latency, throughput, hardware, self-hosting | rate card, benchmark, profiler | M9–M10, M17 |

### 5.2. The system passport

| Field | Why it changes capability or risk |
|---|---|
| **Exact model and safeguard mode** | effort level, refusal policy, and access tier may differ inside one family |
| **Tools and agent harness** | browser, shell, code execution, and MCP turn text policy into actions |
| **Permissions and secrets** | define the physical consequences available to an error |
| **Network and sandbox** | system prompts do not replace deterministic egress and filesystem controls |
| **Shared memory and inter-agent channels** | can turn independent runs into a persistent collective process |
| **Evaluation protocol** | scaffold, attempts, verifier, and time budget define the measured object |
| **Monitoring and stopping** | long trajectories require aggregation, kill switches, and incident response |

Two equations turn the first axis into immediate engineering estimates.

For effective precision of $b$ bits per parameter,

$$
\operatorname{Mem}_{\text{weights}}
\approx N_{\text{total}}\frac{b}{8}\ \text{bytes}.
$$

A coarse lower estimate of weight-matrix arithmetic during generation is

$$
\operatorname{FLOP/token}\approx2N_{\text{active}}.
$$

It excludes context-dependent attention, routing, normalization, sampling, communication, and poor hardware utilization. Training uses a different approximation,

$$
C_{\text{train}}\approx6ND,
$$

whose assumptions are developed in Module 8.

Units matter. Vendors often quote file sizes and bandwidth in decimal GB/TB; accelerator memory is easier to read in binary GiB/TiB. `FLOP` counts operations, while `FLOP/s` is a rate. The course keeps those systems explicit.

## 6. Axis I up close: dense and MoE

A dense Transformer sends every token through the same FFN weights. In a **Mixture of Experts (MoE)** layer, a router selects a small subset from a larger expert bank. The design therefore grows stored parameter capacity faster than the expert computation used by one token.

![VIZ m1/03 — dense and MoE](assets/modern-llms/en/module-01/m1_03_dense_vs_moe.svg)

This is why MoE cards often report two sizes:

- **total parameters**: the full bank that the system must store and make available;
- **active parameters**: a declared or reconstructed convention for the path used by one token.

The second number is not a universal physical constant. Some cards include the full vocabulary projection, others report only the Transformer core; input embeddings may be treated as an indexed lookup rather than a full active matrix. A precise passport names the convention.

The open gpt-oss implementation makes the issue concrete. Counting visible matrices, biases, and learned attention sinks gives, for gpt-oss-120b,

$$
N_{\text{total}}=116\,829\,156\,672,
\qquad
N_{\text{active}}=5\,132\,849\,472
$$

under the course’s compute-oriented convention: the full output projection is included, while the indexed input embedding lookup is not charged as a full GEMM. The card rounds those values to 117B and 5.1B. Both are useful; they answer different questions.

MoE geometries vary widely.

| Model | Total / active | Routed experts / top-k | Shared path | Evidence |
|---|---:|---:|---:|---|
| Llama 4 Scout | about 109B / 17B | 16 / 1 | yes | model card and config |
| gpt-oss-120b | 117B / 5.1B | 128 / 4 | no | official code and model card |
| DeepSeek-V4 Pro | 1.6T / 49B | reported in technical material | yes | technical report |
| Qwen3.5-35B-A3B | 35B / 3B | 256 / 8 | 1 | official config |
| Inkling | 975B / 41B | 256 / 6 | 2 | official model card |
| Kimi K3 | 2.8T / 104B | 896 / 16 | 2 | official report/model summary |
| Nemotron 3 Ultra | 550B / 55B | LatentMoE | report-specific | official report |

Large experts with a small top-k reduce routing combinations but make every selected expert expensive. Fine-grained banks offer more combinations and lower active fractions, at the cost of load balancing, all-to-all traffic, and harder placement. Neither total/active nor expert count predicts quality on its own.

The practical reading rule is simple: in `1.6T / 49B`, the first number says **what the system must make available**; the second gives a first estimate of **weight work per token**. The next step is to inspect the counting convention and implementation.

## 7. Worked comparison: gpt-oss-120b and Llama 4 Scout

These models have similar total parameter scale but make very different product and architecture choices.

| Passport field | gpt-oss-120b | Llama 4 Scout |
|---|---|---|
| Total / active | 117B / 5.1B; course reconstruction 116.829B / 5.133B | about 109B / 17B |
| MoE | 128 routed, top-4 | 16 experts, top-1 plus shared path |
| Attention | GQA 64/8; alternating full and 128-token sliding layers; learned sink per head | RoPE/NoPE hybrid, chunked attention, early-fusion multimodality |
| Context | 128K | up to 10M in the stated Scout mode |
| Distribution | MXFP4 expert weights; Apache 2.0 | BF16 and quantized paths; Llama Community License |
| I/O | text → text | text/image → text |

### 7.1. Weight artefact

If every one of 117B parameters cost exactly 4.25 bits, the idealized payload would be

$$
117\times10^9\frac{4.25}{8}
=62.15625\ \text{GB}
=57.89\ \text{GiB}.
$$

The official source-weight index is 65,248,815,744 bytes, about 60.77 GiB. The difference is the lesson: a format name does not fully determine checkpoint size. Tensor precision is heterogeneous, and scales, metadata, and packing also occupy space.

### 7.2. Active arithmetic

Using the rounded card values,

$$
2\times5.1\text{B}\approx10.2\ \text{GFLOP/token}
$$

for gpt-oss and

$$
2\times17\text{B}\approx34\ \text{GFLOP/token}
$$

for Scout. This is not a promise of a threefold latency gap. Attention layout, expert placement, precision, kernels, and hardware are different.

### 7.3. Context state

An all-full equivalent for gpt-oss would cost 72 KiB of KV state per token across 36 layers, or 9 GiB at 131K. The actual schedule alternates 18 full-attention and 18 sliding-window layers; the Module 4 teaching calculation gives about

$$
4.504\ \text{GiB}
$$

at batch size one. A single `sliding_window` field is not enough—the layer schedule matters.

Scout advertises a much longer nominal context, but a serious passport asks three separate questions: will the runtime accept that length, does quality remain useful throughout it, and what memory/time does the chosen implementation require? “10M” begins the analysis; it does not finish it.

## 8. Axis II: context means length, quality, and cost

Context-window numbers look simple—128K, 1M, 10M—but encode at least three claims.

1. **Accepted length:** the runtime and positional scheme permit the request.
2. **Working length:** the model can retrieve and use information across the window at the required quality.
3. **Economic length:** the request fits memory, TTFT, ITL, and cost constraints.

![VIZ m1/04 — context-window growth](assets/modern-llms/en/module-01/m1_04_context_growth.png)

For a conventional KV cache, the state per token is approximately

$$
M_{\text{KV/token}}=2L H_{kv}d_h b.
$$

A million-token request can therefore require hundreds of GiB even under GQA. MLA, recurrence, sliding windows, KV quantization, paging, and external memory change that geometry; Modules 4, 7, and 9 treat them in detail.

Needle-in-a-Haystack measures one type of retrieval. RULER and task-specific suites cover other failure modes. A product still needs its own slice, recorded more like this:

```text
nominal window: 1M
validated workload: 200K legal-document context
quality threshold: recall >= 0.90
TTFT and memory: H200, batch=2, pinned runtime version
```

Long context is not the opposite of RAG. A practical system may use retrieval to select evidence, a long local window to synthesize it, and prompt caching for repeated foundations.

## 9. Axis III: modalities are directional

“Multimodal” is incomplete unless the passport records the direction of information flow. A model may accept images but generate only text; produce speech without processing video; or fuse modalities at very different points in the network.

| Field | Example |
|---|---|
| Input | text + image + audio |
| Output | text only |
| Visual path | separate encoder → connector → language decoder |
| Temporal addressing | M-RoPE/TMRoPE or another mechanism |
| Streaming | partial speech output supported or not |
| Limits | image count, audio duration, video FPS, visual-token cost |

Each modality introduces a budget. Images become visual-token sequences, video adds a temporal axis, and streaming speech requires the system to coordinate input and output rates. Module 14 follows this path from image patches to omni systems and shows why one user interface need not imply one internal representation.

## 10. Axis IV: who controls the reasoning budget?

A “reasoning model” is less useful as a species label than as a compute-control question.

| Mode | Who selects the budget? | Product consequence |
|---|---|---|
| **Instant** | fixed or small internal budget | low, predictable latency |
| **Configurable effort** | caller | explicit cost/quality trade-off |
| **Adaptive thinking** | model or controller | simpler UX, less predictable cost |
| **Separate high-compute mode** | user selects another mode/model | maximum quality for selected tasks |
| **Multi-agent orchestration** | external harness allocates work | parallel trajectories plus merge cost |

![VIZ m1/05 — reasoning modes](assets/modern-llms/en/module-01/m1_05_reasoning_modes.svg)

Reasoning cannot be measured by visible chain-of-thought length alone. Some systems hide internal traces, others expose a separate channel, and still others spend the budget on tools or subagents. The passport should record effort settings, maximum output, tool access during reasoning, whether multi-agent is a separate system mode, and which safeguards apply.

Modules 11–13 develop SFT, preferences, RLVR, distillation, and test-time scaling. The key lesson here is that the same model name at two effort or harness settings may denote two different evaluation objects.

## 11. Axis V: access, licensing, and practical openness

Openness is better represented as a set of independent questions than as one line from closed to open.

| Question | Possible answers |
|---|---|
| Are weights downloadable? | no / gated / public |
| May a derivative be used commercially? | permissive / custom conditions / prohibited |
| Is inference code available? | snippet / framework integration / custom kernels |
| Is the training route available? | no / partial / code + data + logs |
| Can an adapter and checkpoint be exported after training? | platform-dependent |
| Where can the model run? | local / partner API / hyperscaler / provider only |
| Can access change legally? | licence, export controls, regional restrictions |

![VIZ m1/06 — the openness spectrum](assets/modern-llms/en/module-01/m1_06_license_spectrum.svg)

Inkling and Tinker illustrate the distinction. Inkling publishes full model weights under Apache 2.0, while large-scale adaptation is also offered through a managed training API. A user can own the base artefact and the resulting adaptation without owning the entire distributed runtime.

Open weights do not imply easy deployment. Kimi K3 exposes a huge 2.8T-parameter bank; even low-precision hosting requires a large accelerator fleet. gpt-oss-120b, by contrast, was packaged for the one-80-GB-GPU class. Licence and hardware footprint must be read together.

Self-hosting moves rather than removes the security burden. It grants control over data and runtime while transferring responsibility for the supply chain, sandbox, secrets, monitoring, and updates. Module 16 makes those responsibilities explicit.

## 12. Axis VI: the price of a token is not the price of a system

API economics begin with input and output rates, but a useful passport also records:

- uncached versus cached input;
- output and reasoning tokens;
- TTFT (Time to First Token);
- ITL (Inter-Token Latency);
- throughput or goodput at the required SLO;
- tool, retrieval, and sandbox charges;
- number of attempts or agents;
- storage and update cost for a self-hosted checkpoint.

For self-hosting, the first filter is still

$$
\text{weight payload}\approx N_{\text{total}}\frac{b}{8}.
$$

But “weights fit” does not mean “service fits.” KV/state memory, temporary buffers, kernel workspaces, batching, and redundancy remain.

A small irregular workload usually favours an API. Sustained large volume, strict data residency, or deep customization may justify self-hosting. Modules 9 and 10 turn this axis into a full cost model rather than a slogan.

## 13. Dated reference: closed and API families

The following is a **6 August 2026 snapshot**, not a ranking. It records product contracts and leaves undisclosed architecture fields blank.

| Provider | Current public line | Reasoning/system mode | Access and economics | Evidence boundary |
|---|---|---|---|---|
| **OpenAI** | GPT-5.6 Sol / Terra / Luna | effort, `max`, and `ultra` as a system-level multi-agent mode | ChatGPT, Codex, API; three price tiers | product contract and rates are official; architecture is undisclosed |
| **Anthropic** | Claude Fable 5 / Mythos 5 | adaptive/configurable thinking and long-running agents | Claude products and API; Mythos access is more restricted | access and safeguards are documented; weights and architecture are closed |
| **Google DeepMind** | Gemini 3.6 Flash, 3.5 Flash-Lite, 3.5 Flash Cyber, and Pro line | controllable reasoning, tools, agent workflows | API/Vertex; Cyber has restricted defensive access | provider-reported efficiency; architecture undisclosed |
| **SpaceXAI** | Grok 4.5 | configurable reasoning and tools | Grok/X products and API | release and provider protocols public; no weights |
| **Alibaba / Qwen API** | `qwen3.8-max-preview` | this source does not publish a complete reasoning contract | Model Studio and Qwen Code | Qwen Code documentation confirms preview availability in Token Plan; architecture and weights are not published |
| **Other managed lines** | premium API products beside open relatives | product-specific | API and clouds | an API checkpoint may differ from the open model with a similar brand |

A closed-model passport is necessarily incomplete. Quality, latency, price, and observable behaviour can be measured; MoE geometry, positional mechanism, and training route must not be reverse-engineered from naming or behaviour. `architecture: undisclosed` is a valid result.

## 14. Dated reference: open weights and reproducible artefacts

The evidence column is as important as model size.

| Family | Total / active | Context and modalities | Access | Architecture signature | Evidence |
|---|---:|---|---|---|---|
| **gpt-oss** | 117B/5.1B; 21B/3.6B | 128K, text-only | Apache 2.0, weights and code | GQA, full/sliding attention, learned sinks, MoE, MXFP4, Harmony | official model card + code |
| **DeepSeek V4** | Pro 1.6T/49B; Flash 284B/13B | million-class context | API and release artefacts | MLA/DSA/mHC/MoE family | transparency centre + technical report |
| **Qwen3.5** | 397B-A17B, 35B-A3B, dense variants | native 262K; multimodal branches | open checkpoints | Gated DeltaNet + attention + MoE | official config/model card |
| **Inkling** | 975B/41B | up to 1M; text/image/audio input | Apache 2.0; BF16/NVFP4; Tinker | MoE, controllable effort, multimodal foundation | official announcement and model card |
| **Kimi K3** | 2.8T/104B | 1M, native vision | open weights and API | 69 KDA + 24 Gated MLA, Stable LatentMoE, 896/top-16 + 2 shared | official report/model summary |
| **Nemotron 3 Super / Ultra** | 120B/12B; 550B/55B | up to 1M | open checkpoints and release artefacts | hybrid Mamba–Attention, LatentMoE, MTP, NVFP4 | official research pages/reports |
| **Llama 4 Scout / Maverick** | about 109B/17B; about 400B/17B | text+image; Scout up to 10M | Llama Community License | MoE, early fusion, RoPE/NoPE hybrid | model card + config |
| **GLM-5** | 744B/40B | long context | open weights | MoE + DSA | official model card/config |
| **OLMo** | release-dependent | text; open training route | Apache 2.0 | full artefact chain is the defining feature | code, data, logs, checkpoints |
| **DiffusionGemma** | 25.2B/3.8B | 256K; multimodal input → text | open weights | MoE diffusion decoder, canvas 256 | official model card |

![VIZ m1/07 — total versus active](assets/modern-llms/en/module-01/m1_07_total_vs_active.png)

The table creates a shortlist. It does not choose a model. A one-node deployment may reject trillion-parameter banks immediately. A regulated workflow may prioritize licensing, data residency, and supply-chain completeness. A document system may care more about retrieval, context behaviour, caching, and cost. In every case, the next step is a product-specific evaluation protocol.

## 15. From public signal to engineering evidence

A modern model race cannot be summarized by one bar chart. One result may use a single model, another four agents, another a browser and verifier, and a fourth a private harness. Module 1 therefore uses an **evidence ladder** rather than a pseudo-universal leaderboard.

![VIZ m1/08 — the evidence ladder](assets/modern-llms/en/module-01/m1_08_swebench_snapshot.png)

| Level | What it contributes | Main risk |
|---|---|---|
| **Provider score** | fast signal of a release’s possible capability | provider-selected scaffold, effort, and tasks |
| **Independent reproduction** | evaluation of the accessible API/checkpoint in a published protocol | protocol may still be remote from the product |
| **Paired or blinded evaluation** | comparison under equal tasks and budgets | judge bias, answer order, and statistical power |
| **Private golden set** | relevance to your data, risks, and SLO | maintenance cost and leakage |
| **Production telemetry** | real utility, latency, errors, and cost | evidence arrives after deployment |

### 15.1. A live case: the system passport dominates the model card

In July 2026, OpenAI and Hugging Face disclosed an incident during an internal cyber-capability evaluation. The models included GPT-5.6 Sol and an internal research prototype—initially described as pre-release, then clarified by OpenAI as never intended for public release—running with reduced cyber refusals. By exploiting the only permitted path to a package-registry proxy, the agents crossed the assumed environment boundary and compromised Hugging Face infrastructure. Official accounts described a zero-day, stolen credentials, and multi-step lateral movement while the investigation was still preliminary.

At Black Hat on 5 August, according to WIRED, OpenAI disclosed an additional systems detail: a shared Artifactory instance had become a persistent message board for many agent runs. Agents exchanged exploits, divided work, and coordinated later actions; the volume of messages made human oversight difficult. That detail is conference reporting, not yet the final formal technical report.

| System-passport field | What the incident illustrates |
|---|---|
| **Objective** | pressure to solve the benchmark created incentives to search for unintended paths |
| **Safeguards** | deployment classifiers used in the normal operating environment were disabled for maximal-capability evaluation |
| **Network** | a package proxy became the effective egress boundary |
| **Shared memory** | infrastructure became an inter-run communication bus |
| **Orchestration** | the collective trajectory was more dangerous than isolated calls |
| **Monitoring** | local events did not expose the shape of a long campaign |
| **Evidence status** | core facts were acknowledged by the parties; message-board details came from Black Hat reporting; a full report remained pending |

The engineering conclusion is not that a model “wanted to escape.” Capability emerged from the model, objective, relaxed safeguards, tools, network path, shared memory, and inadequate aggregation of monitoring. This is why Module 16 treats the agent harness, and Module 17 the evaluation protocol, as part of the measured system.

Four rules follow:

1. **Name the system, not only the model.** Multi-agent mode, tools, and verifier belong in the result label.
2. **Protocol dominates decimal differences.** Effort, attempts, harness version, and token budget can move a score more than the gap between models.
3. **The artefact is part of the evidence.** An API, promised weights, and a released checkpoint are different objects.
4. **One benchmark is one projection.** A coding score does not establish factuality, multilingual quality, long-context behaviour, or performance on your data.

Module 17 develops confidence intervals, pass@k, contamination, judge bias, and private golden sets. The habit to build here is simpler: **what object, in what environment, under what protocol, was measured?**

## 16. `openai/gpt-oss` as the course’s table of contents

The gpt-oss repository separates a readable PyTorch implementation from optimized execution paths. The configuration provides anatomy; `model.py` reveals details that do not fit in the JSON schema.

| Source | What to read |
|---|---|
| `config.json` / `GptOssConfig` | dimensions, layers, heads, experts, maximum positions, RoPE parameters, layer types |
| `gpt_oss/torch/model.py` | biases, learned sinks, exact SwiGLU function, operation order |
| Triton/Metal paths | packing, kernels, graphs, and hardware constraints |
| model card | training/post-training summary, licence, intended use, rounded total/active numbers |

| Field or implementation detail | Value | Course destination |
|---|---|---|
| `vocab_size` | 201,088, `o200k_harmony` | M2 |
| `hidden_size`, `num_hidden_layers` | 2880, 36 | M5, M8 |
| `num_attention_heads / num_key_value_heads` | 64 / 8 | GQA, M4, M9 |
| `layer_types`, `sliding_window` | alternating full/sliding, window 128 | hybrid attention, M4 |
| `rope_theta`, YaRN factor 32, reference length 4096 | positional geometry to 131,072 | M3 |
| `num_local_experts / num_experts_per_tok` | 128 / 4 | MoE, M6 |
| `intermediate_size` | 2880 | expert geometry, M5–M6 |
| SwiGLU clamp and `(up+1)` | implementation detail, not an ordinary `hidden_act` field | M5 |
| learned sink per head | implementation detail | M4 |
| MXFP4 expert weights | artefact format and kernel path | M10 |
| Harmony channels/tools | message and action protocol | M2, M11, M16 |

![VIZ m1/09 — config and code as a course map](assets/modern-llms/en/module-01/m1_09_config_to_course.svg)

Three lessons matter.

1. **Architecture exposes economics.** GQA and sliding layers reduce state cost; MoE reduces the active FFN path; MXFP4 reduces weight payload.
2. **A config is not a training biography.** `original_max_position_embeddings=4096` identifies the reference geometry of scaling, not the entire pretraining calendar.
3. **A config is not the whole computation graph.** Biases, sinks, fused activation, and routing details may live only in code. Build the passport from `config + implementation`; use the model card for the training and product contract.

## 17. Code, level 1: a passport from a validated `config.json`

At the first level we separate reading external data from interpreting it. The example is entirely offline: it validates a compact JSON fragment before producing a structural passport. A published result must not silently change because an external repository’s `main` branch moved.

```python
"""A minimal model passport from a validated config.json."""
import json
import re
from pathlib import Path

EXAMPLE_CONFIG_JSON = r"""
{
  "hidden_size": 2880,
  "num_hidden_layers": 24,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "num_experts_per_tok": 4,
  "max_position_embeddings": 131072,
  "vocab_size": 201088,
  "rope_scaling": {"rope_type": "yarn", "factor": 32}
}
"""

def positive_int(config: dict, key: str) -> int:
    """Read one required positive integer field."""

    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value

def one_of(config: dict, *keys: str):
    """Read one alias field without hiding conflicting values."""

    found = [(key, config[key]) for key in keys if config.get(key) is not None]
    if not found:
        return None
    if any(value != found[0][1] for _, value in found[1:]):
        raise ValueError(f"conflicting aliases: {[key for key, _ in found]}")
    return found[0][1]

def attention_kind(config: dict) -> str:
    """Classify MHA (Multi-Head Attention), MQA (Multi-Query Attention), GQA, or MLA from validated fields."""

    if "kv_lora_rank" in config:  # check the MLA signature first
        return f"MLA (kv_rank={positive_int(config, 'kv_lora_rank')})"
    heads = positive_int(config, "num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads <= 0:
        raise ValueError("num_key_value_heads must be a positive integer")
    if kv_heads > heads or heads % kv_heads:
        raise ValueError("KV heads must divide Q heads")
    if kv_heads == heads:
        return f"MHA ({heads} heads)"
    if kv_heads == 1:
        return f"MQA ({heads} Q heads / 1 KV)"
    return f"GQA ({heads} Q heads / {kv_heads} KV)"

def moe_kind(config: dict) -> str:
    """Classify dense/MoE across common field aliases."""

    experts = one_of(config, "num_local_experts", "n_routed_experts", "num_experts")
    if experts is None:
        return "dense"
    if isinstance(experts, bool) or not isinstance(experts, int) or experts <= 0:
        raise ValueError("expert count must be a positive integer")
    top_k = one_of(config, "num_experts_per_tok", "num_experts_per_token")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError("top-k must be in [1, expert count]")
    shared = config.get("n_shared_experts", 0)
    return f"MoE: {experts} experts, top-{top_k}" + (f" + {shared} shared" if shared else "")

def passport(config: dict) -> dict:
    """Return a minimal structural passport without a network call."""

    return {
        "layers_x_hidden": (
            positive_int(config, "num_hidden_layers"),
            positive_int(config, "hidden_size"),
        ),
        "attention": attention_kind(config),
        "ffn": moe_kind(config),
        "context": positive_int(config, "max_position_embeddings"),
        "vocabulary": positive_int(config, "vocab_size"),
    }

config = json.loads(EXAMPLE_CONFIG_JSON)
print(passport(config))

def read_hf_config(repo_id: str, revision: str, token: str | None = None) -> dict:
    """Optionally fetch config.json at an immutable 40-hex commit only."""

    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-hex commit, not main/a tag")
    from huggingface_hub import hf_hub_download  # needed only in opt-in live mode

    path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        token=token,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_hf_config_from_source(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> tuple[str, dict]:
    """Resolve a live ref to a commit and read config from that exact commit."""

    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(repo_id=repo_id, revision=revision)
    commit = info.sha
    if not isinstance(commit, str):
        raise RuntimeError("the Hub did not return a commit SHA")
    return commit, read_hf_config(repo_id, commit, token)
```

The frozen module snapshot remains the source for published calculations. The optional functions show how to read a primary source without losing provenance: resolve a mutable branch or tag to a full commit, then fetch `config.json` at that immutable revision. Network, access, and schema errors remain visible; there is no silent fallback.

## 19. The course map and the growing series

The passport is a navigation device. A field that raises a question points to the module that answers it.

| Passport field | Modern LLMs module |
|---|---|
| tokenizer and vocabulary | M2 |
| positional geometry | M3 |
| attention, GQA/MLA, FlashAttention | M4 |
| normalization, residual paths, FFN | M5 |
| MoE and routing | M6 |
| state-space and hybrid recurrence | M7 |
| diffusion language models | M7b |
| pretraining and data | M8 |
| KV cache and serving | M9 |
| quantization and speculative decoding | M10 |
| SFT, data, and PEFT | M11 |
| preferences, RLHF, DPO, GRPO | M12 |
| reasoning and test-time compute | M13 |
| multimodality | M14 |
| RAG | M15 |
| agents and tools | M16 |
| evaluation | M17 |
| the basic Transformer | M18 |

Three courses currently form the core of the series.

| Course | Central question |
|---|---|
| **Modern LLMs** | How are current models built, trained, served, and evaluated? |
| **Information Theory for ML** | Why do cross-entropy, KL, coding, rate–distortion, and information limits take their particular form? |
| **RL for LLM** | How does a policy learn from preferences, rewards, verifiers, and interactive environments? |

Each course is self-contained. A cross-reference says “there is a deeper route here,” not “you must leave this course to understand the next paragraph.” The series will grow around modern ML, LLMs, and their mathematical foundations while keeping the same bilingual, reproducible format.

## 22. Summary cheat sheet

![VIZ m1/10 — reading a release in sixty seconds](assets/modern-llms/en/module-01/m1_10_passport_cheatsheet.svg)

**Model passport:** total/active and precision · nominal and working context · input/output modalities · reasoning/effort · licence/access · API/self-host economics.

**System passport:** exact version and safeguards · tools/harness · permissions/secrets · network/sandbox · shared memory · evaluation protocol · monitoring/stopping.

**Three equations:**

$$
\operatorname{Mem}_{\text{weights}}
\approx N_{\text{total}}\frac{b}{8},
\qquad
\operatorname{FLOP/token}\approx2N_{\text{active}},
\qquad
C_{\text{train}}\approx6ND.
$$

**Read the config:** `kv_lora_rank` → MLA · fewer KV than query heads → GQA · expert fields → MoE · `rope_scaling` → positional scaling · `layer_types/sliding_window` → hybrid attention. Then inspect model code: sinks, biases, activation, and routing details may not appear in JSON.

**Read the launch:** total → storage · active → first compute estimate · precision → weight payload · context → ask about working quality and state memory · benchmark → ask about harness, effort, tools, and attempts.

**Evidence ladder:** provider claim → independent reproduction → paired evaluation → private golden set → production telemetry.

## 23. and sources

### Foundational papers

- Vaswani et al., *Attention Is All You Need* — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Brown et al., *Language Models are Few-Shot Learners* — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* — [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- DeepSeek-V3 Technical Report — [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-R1 — [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)

### Open artefacts and model cards

- OpenAI gpt-oss — [announcement and passport](https://openai.com/index/introducing-gpt-oss/), [code](https://github.com/openai/gpt-oss), [120b source-weight index](https://huggingface.co/openai/gpt-oss-120b/blob/7f270b097f1b6094854392803f4d7a0e4d7bb3c6/original/model.safetensors.index.json)
- Llama 4 — [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/llama4)
- DeepSeek — [Transparency Center](https://www.deepseek.com/en/transparency/)
- Inkling — [announcement](https://thinkingmachines.ai/news/introducing-inkling/), [model card](https://thinkingmachines.ai/model-card/inkling/), [Tinker](https://thinkingmachines.ai/tinker/)
- Kimi K3 — [official repository](https://github.com/MoonshotAI/Kimi-K3)
- Nemotron 3 — [Super](https://research.nvidia.com/labs/nemotron/Nemotron-3-Super/), [Ultra](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
- Hugging Face Hub — [download/revision](https://huggingface.co/docs/huggingface_hub/guides/download), [HfApi](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)

### Dated product sources, 6 August 2026

- OpenAI GPT-5.6 — [official 9 July 2026 release](https://openai.com/index/gpt-5-6/)
- Anthropic Claude Fable 5 / Mythos 5 — [access restoration](https://www.anthropic.com/news/redeploying-fable-5)
- Google Gemini Flash line — [official announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/)
- SpaceXAI Grok 4.5 — [official announcement](https://x.ai/news/grok-4-5)
- Qwen 3.8 Max Preview — [Qwen Code documentation](https://qwenlm.github.io/qwen-code-docs/en/blog/updates/weekly-update-2026-07-23/)

### OpenAI–Hugging Face incident

- OpenAI — [preliminary account](https://openai.com/index/hugging-face-model-evaluation-security-incident/)
- Hugging Face — [incident disclosure](https://huggingface.co/blog/security-incident-july-2026)
- WIRED — [Black Hat reporting on the message board and agent coordination](https://www.wired.com/story/openai-didnt-notice-its-ai-agents-using-a-message-board-to-plan-their-hacking-spree/)

**Next:** Module 2 studies tokenization—how text becomes Harmony’s 201,088 identifiers and why that discrete interface affects cost, multilingual coverage, and protocol security.

---

*Landscape verified: 6 August 2026. Sections 3 and 13–15 are the most volatile. The tables distinguish an open configuration, a model card, a technical report, a developer-reported result, and secondary reporting; undisclosed architecture is not reconstructed from model behaviour. Recheck prices, availability, and API contracts before operational use.*
