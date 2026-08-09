# Module 12. RL infrastructure for LLMs: memory, rollouts, and asynchrony

> **Material version:** 2026.24  
> **Factual snapshot:** 2026-08-06  
> **Language:** EN  
> **Core practice:** profile A — browser and central processing unit (CPU); profile B — a free Colab T4; profile C — an optional multi-GPU run  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 10–12 hours without the optional model run

Modules 8–11 developed objectives, group rollouts, verifiable rewards, and long agent trajectories. The last topic is treated directly in *RL for LLM*, Module 11 “RL for LLM Agents: Tools, Environments, and Long Trajectories” (`rl_llm.module_11_agentic_rl`). The engineering question is now unavoidable: **how do we turn these formulas into a system that collects data, updates a model, and preserves the meaning of the behavior policy while information moves between programs?**

Follow one rollout through the machine. It begins as a scheduler record, becomes an inference request, accumulates tokens and KV-cache state, passes through a verifier, waits in a queue, enters a training batch, contributes to a gradient, and must finally be attributable to a published weight version. Losing a mask, policy version, actual behavior log-probability, or reward provenance at any boundary changes the statistical object, not merely the software quality.

The same GRPO objective can run as a sequential one-GPU script, a pipeline with separate rollout and training pools, or a fully asynchronous queue. The algorithmic mechanics are developed in *RL for LLM*, Module 9 “RLVR and GRPO: Verifiable Rewards, Group-Relative Estimates, and Stability” (`rl_llm.module_09_rlvr_grpo`). This module studies the systems layer: different executions create different memory peaks, lag distributions, and risks of stale or incorrectly attributed data.

The practical task of this module is:

> **Design and measure an RL loop in which generation, training, weight synchronization, rollout logging, and failure recovery form one auditable system with controlled memory, throughput, and staleness.**

By the end of the module, you will be able to:

1. represent an RL loop as a dataflow among a policy, rollout engine, verifier, trainer, and storage;
2. distinguish colocated from disaggregated placement;
3. separate algorithmic synchrony from physical process placement;
4. define a minimal rollout contract with a behavior-policy version;
5. account for parameters, gradients, and Adam state without assuming one universal byte count per parameter;
6. explain what Fully Sharded Data Parallel (FSDP) and Zero Redundancy Optimizer (ZeRO) shard and which peaks remain;
7. estimate key–value cache (KV-cache) memory from architecture and retained tokens;
8. distinguish training memory, generation memory, and the time-dependent peak of colocation;
9. compute sequential and overlapped pipeline time;
10. measure loss of utilization from variable completion length and stragglers;
11. state a first-order queue-stability condition and introduce backpressure;
12. distinguish policy lag from training–inference mismatch;
13. compute probability ratios, clipping statistics, and effective sample size (ESS);
14. choose among log-probability recomputation, importance correction, and stale-data rejection;
15. design atomic synchronization for full weights or adapters;
16. define a complete RL-run checkpoint and test recovery before failure;
17. trace a prompt through rollout, weight version, reward, batch, and optimizer step;
18. compare TRL, verl, OpenRLHF, NeMo RL, PRIME-RL, SkyRL, and AReaL along architectural axes rather than one ranking;
19. profile a synthetic RL pipeline on CPU and optionally a small open model on a Colab T4.

The English lab is the executable practice, and the claim and primary-source registry is `Module_12_Sources.md`.

---

> **Practice route.** Graded tasks, worked solutions revealed after an attempt, and executable practice follow the lecture. The explanation remains self-contained.

## Lesson 12.1. The RL loop as a dataflow

> **Lesson 12.1. The RL loop as a dataflow**
>
> 1. Why one `trainer.generate()` call is not enough
> 2. The role contract matters more than the queue format
> 3. Placement and synchrony are independent axes
> 4. The behavior policy needs an atomic rollout version
> 5. Measure a sequential baseline before building a distributed system
>
> then 2 assessment steps

### Step 12.1.1 — Why one `trainer.generate()` call is not enough

Start with the physical loop rather than a framework name. One online-RL update needs to:

1. select a prompt or environment state;
2. obtain one or more rollouts from the behavior policy;
3. compute rewards and operational status;
4. recover masks and log-probabilities;
5. assemble a training batch;
6. compute the loss and gradients;
7. update parameters;
8. make the new policy version available to data collectors.

![Dataflow in an RL loop](assets/rl-for-llm/en/module-12/M12_system_dataflow_EN.png)

A small script can perform every role in one process. At scale, the roles become separate programs or worker groups:

- a **prompt scheduler** selects tasks;
- a **rollout engine** decodes efficiently;
- an **environment or verifier** computes outcomes;
- a **trainer** runs forward and backward passes;
- a **synchronization service** publishes a new weight version;
- **storage and telemetry** preserve experience, metrics, and checkpoints.

This decomposition is not a theorem. It appears because generation and training have different computational profiles.

- Decode performs many short sequential steps, while KV-cache memory follows the number and length of active sequences.
- Training benefits from dense batches, saved activations, backward computation, and gradient collectives.

An inference engine such as vLLM or SGLang and a trainer built on FSDP, DeepSpeed, or Megatron may serve the same policy while solving different systems problems.

> **Boundary.** A training library can generate text, and an inference engine can execute a forward pass. Role separation is an engineering choice, not an impossibility result. A sequential one-GPU script is often the best control because it has fewer moving parts.

---

### Step 12.1.2 — The role contract matters more than the queue format

Passing only an answer string between two processes is insufficient for a correct policy update. A minimal rollout record must answer at least the following questions.

| Field | Why it is needed |
|---|---|
| `sample_id`, `prompt_id` | connect a prompt, its answer group, and later diagnosis |
| token IDs | reconstruct the exact policy actions |
| policy-token mask | avoid training on environment or template text |
| behavior log-probabilities | compute probability ratios or test stack agreement |
| `policy_version` | measure policy lag |
| decoding parameters | reconstruct the actual behavior distribution |
| reward components | separate task outcome from format and cost terms |
| stop reason | distinguish the end-of-sequence token (EOS), limits, task failure, and infrastructure failure |
| verifier version | avoid silently mixing reward specifications |
| timestamps | measure queueing, rollout, verification, and training |

A record may look like this:

```json
{
  "sample_id": "run42-p008-g03",
  "policy_version": 127,
  "prompt_tokens": [101, 42, 17],
  "response_tokens": [314, 159, 2],
  "policy_mask": [0, 0, 0, 1, 1, 1],
  "behavior_logprobs": [-0.31, -1.07, -0.22],
  "decoding": {"temperature": 0.8, "top_p": 1.0},
  "reward": {"correctness": 1.0, "format": 0.0},
  "stop_reason": "eos",
  "verifier_version": "arith-v4"
}
```

The physical container need not be JSON. Apache Arrow, Parquet, a message queue, or an in-memory object can all work. What matters is preserved semantics.

Module 11 introduced token provenance in a multi-turn trajectory. The systems requirement is stronger: **provenance must survive serialization, queuing, and process restart**. If a record cannot distinguish policy tokens from environment observations after reload, no trainer can recover an honest loss.

---

### Step 12.1.3 — Placement and synchrony are independent axes

Infrastructure is easier to reason about when two commonly conflated axes are separated.

**Axis 1 — computational placement.**

- **Colocation:** rollout and training alternate on the same graphics processing units (GPUs). It lowers the minimum device count and can simplify weight publication, but the roles compete for memory and cannot use the same device simultaneously.
- **Disaggregation:** rollout engines and trainers occupy different GPUs or nodes. Work can overlap and pools can scale independently, but the design adds networking, weight transfer, queues, and staleness.

**Axis 2 — data and update order.**

- **Synchronous mode:** the trainer waits for a defined rollout batch, updates, and then starts the next logical cycle.
- **Asynchronous mode:** collectors keep producing while the trainer updates; records arrive from multiple policy versions.

![Placement and synchrony are separate axes](assets/rl-for-llm/en/module-12/M12_execution_layouts_EN.png)

The axes are independent.

- Colocation is usually sequential, but CPU-side preparation and buffering may overlap.
- Disaggregated pools may remain strictly synchronous: the rollout pool creates a batch and then waits for training and synchronization.
- Disaggregated pools may be fully asynchronous with an age-bounded queue.

This map prevents the false conclusion that “a separate vLLM server automatically makes the algorithm asynchronous.” A physically separate server can still participate in a strict barrier protocol.

For a first reproducible experiment, begin in the simplest cell: one process, one batch, one policy version. Every later complication should be justified by a measured bottleneck.

---

### Step 12.1.4 — The behavior policy needs an atomic rollout version

Suppose the rollout engine uses weight version $v$, while the trainer has already reached version $u$. Write the actual behavior policy as

$$
\mu_v(y\mid x)
$$

and the current trainable policy as

$$
\pi_u(y\mid x).
$$

In a synchronous run, $u=v$ when data collection begins, although Proximal Policy Optimization (PPO) may update the current policy relative to a frozen old policy over several epochs. In an asynchronous run, the difference

$$
L=u-v
$$

is the **policy lag**.

A version should refer to a concrete probability distribution, not merely a filename. Three safe regimes are possible.

1. **One version per completion.** In-flight requests finish under the old weights, and the engine atomically switches before accepting new requests.
2. **One version per agent turn.** This can be valid for long episodes if the log stores the version and behavior log-probabilities of every action.
3. **The exact behavior policy is recorded.** If weights change in the middle of a completion, token-level versions and probabilities plus an unambiguous update order are required; analysis becomes substantially harder.

The first regime is usually simplest for one-shot completions. Long multi-turn agents may require the second.

> **Invariant:** a rollout cannot be labeled with one version if its actions were actually sampled from several versions.

Atomicity also includes decoding. Temperature, top-p, forbidden tokens, and EOS handling are part of the behavior policy. Identical weights with different decoding rules define different distributions.

---

### Step 12.1.5 — Measure a sequential baseline before building a distributed system

Before choosing Ray, Kubernetes, or a multi-process engine, measure one sequential control loop. For every batch, log:

- prompt preparation time;
- prefill time;
- decode time;
- verifier time;
- log-probability recomputation;
- forward and backward time;
- optimizer-step time;
- weight-synchronization time;
- input and output token counts;
- peak memory.

The sequential cycle is

$$
T_{\mathrm{seq}}
=
T_{\mathrm{rollout}}
+
T_{\mathrm{reward}}
+
T_{\mathrm{logp}}
+
T_{\mathrm{train}}
+
T_{\mathrm{sync}}.
$$

This is accounting, not a cluster-performance model. It answers the first question: **which stage actually dominates this configuration?**

“Generation is always the bottleneck” is too strong. It is often true for long reasoning traces and few training epochs, but can fail with short outputs, many epochs, a heavy critic, slow verification, or expensive weight transfer.

The optional notebook first constructs this synthetic profile. Its real-model branch then measures generation and a teacher-forced forward pass on a small open model. That branch teaches timers and units; it is not a benchmark of a large cluster.

---

## Lesson 12.2. Memory and throughput

> **Lesson 12.2. Memory and throughput**
>
> 1. “Sixteen bytes per parameter” is a profile, not a law
> 2. Replication, FSDP, and ZeRO
> 3. Engine memory: weights and KV cache
> 4. Colocation requires a time-dependent memory ledger
> 5. Throughput needs tokens and events, not one example count
>
> then 3 assessment steps

### Step 12.2.1 — “Sixteen bytes per parameter” is a profile, not a law

A common estimate says mixed-precision Adam training needs about 16 bytes per parameter. It is useful as one scenario and dangerous as a universal constant.

Logical model-state memory is the sum of representations that are actually stored:

$$
B_{\mathrm{state}}
=
N\left(
 b_{\mathrm{param}}
+b_{\mathrm{grad}}
+b_{m_1}
+b_{m_2}
+b_{\mathrm{master}}
\right),
$$

where $N$ is parameter count and each $b_*$ is bytes per parameter of the corresponding object.

One common profile is:

| Component | Type | Bytes/parameter |
|---|---:|---:|
| working parameters | bfloat16 | 2 |
| gradients | bfloat16 | 2 |
| first Adam moment | float32 | 4 |
| second Adam moment | float32 | 4 |
| separate master copy | float32 | 0 or 4 |
| **Total** |  | **12 or 16** |

The range exists because implementations differ.

- Some keep a separate float32 master copy; others treat another high-precision shard as the canonical state.
- Gradients may be bfloat16 or float32.
- Optimizer states may be quantized, offloaded, or replaced by another optimizer.
- Adapter training stores optimizer state only for a small trainable subset.

Model state is not the whole device footprint. Activations, collective buffers, logits, masks, KV cache, allocator fragmentation, and environment memory remain. Bytes per parameter are the first line of a budget, not the answer to “will this run fit?”

---

### Step 12.2.2 — Replication, FSDP, and ZeRO

With Distributed Data Parallel (DDP), every process holds a full model and optimizer-state replica. Each process receives different examples and gradients are reduced across replicas.

Fully Sharded Data Parallel (FSDP) and the Zero Redundancy Optimizer (ZeRO) reduce model-state replication. In their most complete forms, parameters, gradients, and optimizer state are partitioned across $D$ data-parallel participants.

A first-order persistent footprint is

$$
B_{\mathrm{rank}}
\approx
\frac{B_{\mathrm{state}}}{D}.
$$

The approximation sign is essential. During forward and backward passes, modules may be gathered temporarily, communication buffers are allocated, and large objects may not align perfectly with the shard boundary. Peak memory depends on:

- wrapping and shard boundaries;
- the largest materialized module;
- prefetching;
- activation checkpointing;
- concurrent microbatches;
- logit computation;
- the communication backend.

Therefore $B_{\mathrm{state}}/D$ is an order-of-magnitude check, not a substitute for a memory trace.

> **Extension:** general DDP, FSDP/ZeRO, tensor parallelism, and pipeline parallelism are developed in *Modern LLMs*, Module 8 “Pretraining: Data, Scaling Laws, and Optimizers” (`modern_llms.module_08_pretraining`). The additional concern here is the rollout engine and movement of one policy between training and generation layouts.

---

### Step 12.2.3 — Engine memory: weights and KV cache

A rollout engine normally stores no gradients or optimizer states. Its largest components are:

1. model weights;
2. the KV cache of active sequences;
3. temporary workspaces and graphs;
4. sometimes several adapter or weight versions.

For a transformer with $L$ layers, $H_{kv}$ key–value heads, head dimension $d_h$, and $N_{\mathrm{tok}}$ retained tokens,

$$
B_{\mathrm{KV}}
=
2L H_{kv} d_h N_{\mathrm{tok}} b,
$$

where the factor two accounts for keys and values and $b$ is bytes per element.

Consider

$$
L=32,
\qquad
H_{kv}=8,
\qquad
d_h=128,
\qquad
b=2.
$$

One retained token needs

$$
2\cdot32\cdot8\cdot128\cdot2
=
131072\ \text{bytes}
=
128\ \text{KiB}.
$$

Sixty-four sequences of 4096 tokens contain 262144 retained tokens and require exactly 32 GiB of useful KV payload in this simplified architecture.

![KV-cache memory grows with retained tokens](assets/rl-for-llm/en/module-12/M12_kv_cache_scaling_EN.png)

This explains why GRPO group size and reasoning length are infrastructure parameters: more candidates increase active sequence count, and long completions hold pages longer.

PagedAttention reduces fragmentation in vLLM; RadixAttention in SGLang can reuse common prefixes. Neither removes the linear payload of retained keys and values.

> **Extension:** multi-query, grouped-query, and multi-head latent attention, PagedAttention, prefix caching, and distributed serving are developed in *Modern LLMs*, Module 9 “KV Caches and Efficient Inference” (`modern_llms.module_09_inference`).

---

### Step 12.2.4 — Colocation requires a time-dependent memory ledger

With disaggregation, trainer and rollout engine occupy different GPUs. Their memory contributes to cluster total but not to one device peak.

With colocation, the same GPU pool alternates roles. Simply adding both phase maxima is too pessimistic because some objects can sleep, offload, or be freed between phases. The correct object is a timeline.

![Training and rollout memory by phase](assets/rl-for-llm/en/module-12/M12_memory_ledger_EN.png)

A typical sequence is:

1. **Rollout:** engine-format weights and a large KV cache are active; optimizer state may be sharded, offloaded, or put to sleep.
2. **Transition:** the KV cache is released or reduced and trainer state becomes active.
3. **Training:** parameters, gradients, optimizer state, and activations are active; most engine workspace is released.
4. **Synchronization:** the new weight version is transformed or loaded into the engine representation.

The peak is

$$
B_{\mathrm{peak}}=\max_t B(t),
$$

including transition buffers.

LoRA changes two parts of the ledger.

- Gradients and optimizer states exist only for adapters.
- Base weights still support every forward pass, and the rollout engine must receive a consistent adapter version or merged representation.

LoRA can make colocation practical, but it does not make rollout memory vanish. The exact adapter-update path remains a versioned library feature.

---

### Step 12.2.5 — Throughput needs tokens and events, not one example count

“Examples per second” is a poor comparison when prompts and completions differ in length. A minimal report separates:

- input tokens per second;
- output tokens per second;
- completed rollouts per second;
- verified outcomes per second;
- training tokens per second;
- optimizer steps per hour;
- cost per informative update.

For rollout, prefill and inter-token decode time should be separated. For training, count **non-padding policy tokens**, not only packed sequence length.

A generic throughput is

$$
\text{throughput}
=
\frac{\text{useful work}}{\text{stage time}}.
$$

But useful work depends on the question. If 90% of groups have equal reward, high output-token throughput does not imply high learning-signal throughput. A systems dashboard can therefore include

$$
\text{informative groups/s}
=
\text{groups/s}
\times
(1-\text{degenerate-group fraction}).
$$

This is not a task-quality metric. It explains why expensive hardware may produce many tokens but few non-zero relative advantages.

---

## Lesson 12.3. Pipelines, queues, and stragglers

> **Lesson 12.3. Pipelines, queues, and stragglers**
>
> 1. Sequential loop and overlapped pipeline
> 2. Mean time hides slow tails
> 3. Continuous batching and prefix reuse
> 4. Queue stability and backpressure
> 5. Profile the timeline before interpreting GPU utilization
>
> then 4 assessment steps

### Step 12.3.1 — Sequential loop and overlapped pipeline

Let one logical batch have two large stages:

- rollout plus reward takes $G$ seconds;
- training plus synchronization takes $T$ seconds.

Sequential cycle time is

$$
C_{\mathrm{seq}}=G+T.
$$

If the stages occupy independent resources and overlap, the steady-state period after filling the two-stage pipeline is

$$
C_{\mathrm{pipe}}=\max(G,T).
$$

For $n$ batches, ideal total time is

$$
T_n
=
G+T+(n-1)\max(G,T).
$$

The first batch still traverses both stages; throughput improves only over a stream.

![Sequential, overlapped, and asynchronous execution](assets/rl-for-llm/en/module-12/M12_pipeline_timeline_EN.png)

Two-stage speedup approaches two only when $G\approx T$. If rollout takes 90 seconds and training takes 10, the period falls from 100 to 90 seconds, only $1.11\times$. The fast trainer spends most of its time waiting for data.

Pool sizes should therefore follow measured stage rates rather than an equal split of GPUs. Add resources to the stage that limits the period until another bottleneck—network, verifier, or memory—appears.

---

### Step 12.3.2 — Mean time hides slow tails

A group method often waits for $G$ completions of one prompt. With parallel rollout, group time is the maximum:

$$
T_{\mathrm{group}}
=
\max_{1\le i\le G}T_i.
$$

Suppose eight rollouts take

$$
[4,4,5,5,5,6,6,20]\ \text{s}.
$$

Total work is 55 GPU-seconds, but eight workers are reserved for 20 seconds. Utilization relative to the ideal rectangle is

$$
U
=
\frac{\sum_iT_i}{G\max_iT_i}
=
\frac{55}{160}
\approx0.344.
$$

One long completion leaves most resources idle.

Long tails come from completion length, tool calls, environment timeouts, prompt size, retries, and uneven routing. Remedies include cost-aware bucketing, continuous batching, explicit turn/time budgets, streaming completed rollouts, and separating infrastructure retries from policy actions.

Truncating the tail saves wall time but changes data. If hard tasks are systematically longer, unconditional removal biases training toward easy cases. Timeout rules and truncation rates belong in the experiment report.

---

### Step 12.3.3 — Continuous batching and prefix reuse

An inference engine is valuable not merely because it “can generate,” but because it schedules many requests with different lengths.

**Continuous batching** inserts a new request when another finishes rather than waiting for a static batch barrier. This matters in RL because completion lengths have broad tails.

PagedAttention manages KV cache in blocks and reduces fragmentation. SGLang uses RadixAttention to reuse common prefixes. In GRPO, several completions share the same prompt, so the prompt portion can often be computed once or efficiently shared.

Prefix reuse has exact boundaries.

- Actual tokens and model parameters must match.
- A changed chat template, system prompt, or policy version invalidates the old cache.
- KV state computed under old weights is not KV state of the new policy.
- In a multi-turn environment, trajectories stop sharing a prefix as soon as their observations diverge.

Weight publication may therefore invalidate warm cache state. Frequent synchronization can cost more than bytes transferred because it also destroys reusable decode state.

> **Scope boundary:** PagedAttention, RadixAttention, chunked prefill, and distributed serving belong to *Modern LLMs*, Module 9 “KV Caches and Efficient Inference” (`modern_llms.module_09_inference`). Here we keep only their effect on batch boundaries and policy-version changes.

---

### Step 12.3.4 — Queue stability and backpressure

In an asynchronous system, rollouts enter a queue at rate $\lambda$ and the trainer consumes them at rate $\mu$. In consistent work units, a necessary first-order long-run condition is

$$
\lambda<\mu.
$$

When $\lambda>\mu$, average queue length grows. So do policy lag, memory or disk usage, collection-to-update delay, and the chance that a record will be discarded as too stale.

The queue cannot grow without bound. **Backpressure** slows workers, issues fewer prompts, or temporarily changes priorities when a limit is reached.

Limits may be expressed as:

- rollout count;
- token count;
- maximum age in seconds;
- maximum policy lag;
- total memory;
- a combination of these quantities.

Counting records alone is dangerous: one multi-turn episode can be hundreds of times larger than a short arithmetic completion. Token- or cost-based limits track the physical queue better.

The opposite regime also matters. If $\lambda\ll\mu$, the trainer starves. Adding rollout workers is not always the answer; the verifier, prompt length, or poor prefix reuse may be the actual constraint. Stage-level traces must show waiting time at each boundary.

![A bounded queue and observable backpressure](assets/rl-for-llm/en/module-12/M12_queue_backpressure_EN.png)

The module’s deterministic event simulator makes the trade-off visible. The 120-second smoke profile uses the same seed, eight rollout workers, 4096-token training batches, and one trainer in all three configurations. It is **not a framework benchmark**; it is a tested model of the queue contract.

| Profile | Training tokens/s | Mean lag, versions | Mean normalized ESS | Rejected rollouts | Backpressure time |
|---|---:|---:|---:|---:|---:|
| sequential | 566.64 | 0.000 | 1.0000 | 0 | 0.0 s |
| bounded async, synchronize every update | 1442.38 | 1.803 | 0.9798 | 74 | 18.6 s |
| deeper overlap, synchronize every three updates | 1800.82 | 3.550 | 0.8804 | 0 | 52.8 s |

In this specific model, deeper overlap raises throughput while increasing lag and concentrating synthetic importance weights. The table does not define a universally optimal synchronization cadence; its coefficients were chosen to expose the mechanism. The defensible conclusion is narrower: throughput and freshness must be measured together.

The admission threshold applies to **new work**. It does not cancel requests already in flight, so observed queue depth may overshoot briefly as those requests finish. An implementation that does not state this rule can make a normal transient look like a broken counter.

---

### Step 12.3.5 — Profile the timeline before interpreting GPU utilization

Average GPU utilization does not identify a bottleneck. Ninety percent may represent useful decode, weight conversion, or a stalled collective inside an active kernel.

A minimal profiling protocol is:

1. assign one `trace_id` across prompt, rollout, and training batch;
2. record start and end of every stage;
3. measure tokens as well as records;
4. separate queue wait from compute time;
5. record policy and verifier versions;
6. measure after warm-up;
7. report median and tail quantiles, not only the mean;
8. change one parameter per experiment.

Useful stage names include:

```text
queue_wait
prefill
decode
verifier
logprob_recompute
forward_backward
optimizer_step
weight_sync
checkpoint
```

The bottleneck is then identified from the critical path. If rollout is 70% of sequential time but already hidden behind training, faster decode may not change the period. If synchronization lies on every critical path, an ideal engine still cannot fix throughput.

The optional notebook builds a synthetic timeline and distinguishes total work, critical-path time, pool utilization, and steady-state throughput.

---

## Lesson 12.4. Staleness, mismatch, and weight synchronization

> **Lesson 12.4. Staleness, mismatch, and weight synchronization**
>
> 1. Three policies and two different mismatches
> 2. Probability ratios and effective sample size
> 3. Identical weights do not guarantee identical probabilities
> 4. What to do with stale data
> 5. Weight synchronization must be atomic and measurable
>
> then 4 assessment steps

### Step 12.4.1 — Three policies and two different mismatches

Infrastructure discussions benefit from distinguishing three policies.

1. The **behavior policy** $\mu_v$ actually sampled tokens in the rollout engine.
2. The **old policy** $\pi_{\mathrm{old}}$ is the frozen denominator of a PPO-like objective within an update.
3. The **current policy** $\pi_\theta$ receives the gradient.

The token-level contract—causal shift, response masks, and the distinction among old, current, and reference log-probabilities—is constructed in *RL for LLM*, Module 6 “The LLM as a Policy: Tokens, Log-Probabilities, and KL” (`rl_llm.module_06_llm_policy`). Infrastructure does not alter those definitions; it must preserve them across serialization and process boundaries.

A separate reference policy $\pi_{\mathrm{ref}}$ may provide Kullback–Leibler (KL) regularization. It does not replace the behavior policy.

Now separate two failure modes.

**Policy lag.** Let $u$ denote the current published and trainable version, and let the rollout have been generated by version $v$. The sample is stale when

$$
v<u.
$$

**Training–inference mismatch (TIM).** Engine and trainer declare the same weight version but compute different probabilities because of:

- different numerical kernels or reduction order;
- precision or quantization differences;
- tensor-parallel layout;
- template, mask, or EOS differences;
- logit transformations used by decoding.

They require different diagnostics. Lag is measured by version and age. TIM is tested at **zero lag**: immediately after synchronization, score the same sequence in the engine and trainer and compare token log-probabilities.

A ratio far from one cannot be attributed automatically to staleness. Conversely, exact stack agreement does not remove version lag.

---

### Step 12.4.2 — Probability ratios and effective sample size

For action $a$ in state $s$, the importance ratio is

$$
w
=
\frac{\pi_\theta(a\mid s)}{\mu_v(a\mid s)}
=
\exp\left[
\log\pi_\theta(a\mid s)-\log\mu_v(a\mid s)
\right].
$$

The ratio is defined only where the behavior policy assigns positive probability: $\mu_v(a\mid s)>0$. Full importance-sampling claims require support: the target policy must not assign positive probability to actions that the behavior policy could never produce. Numerical clipping cannot reconstruct missing support.

For a complete autoregressive sequence,

$$
w_{\mathrm{seq}}
=
\prod_{t=1}^{T}w_t,
$$

or, more stably,

$$
\log w_{\mathrm{seq}}
=
\sum_{t=1}^{T}
\left(
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\mu_v(y_t\mid x,y_{<t})
\right).
$$

Small token-level differences accumulate with length, so long completions are especially sensitive to lag and TIM.

For non-negative weights $w_1,\ldots,w_n$, effective sample size is

$$
\operatorname{ESS}
=
\frac{\left(\sum_iw_i\right)^2}{\sum_iw_i^2}.
$$

Equal weights give ESS $n$; one dominant record pushes it toward one. Normalized ESS, $\operatorname{ESS}/n$, compares batches of different sizes.

![Policy lag, probability ratios, and ESS](assets/rl-for-llm/en/module-12/M12_staleness_ess_EN.png)

ESS is a concentration diagnostic, not proof of an unbiased update. Log policy lag, mean and maximum $|\log w|$, clipping fraction, rejected records, gradient norm, and completion length together.

---

### Step 12.4.3 — Identical weights do not guarantee identical probabilities

Suppose trainer and engine weights are byte-identical. It still need not follow that

$$
\mu_v(y_t\mid h_t)=\pi_v(y_t\mid h_t).
$$

Floating-point addition is non-associative. Batch shape, tensor parallelism, fused kernels, quantization, and softmax implementation can change logits slightly. Low-probability tokens can turn a small logit perturbation into a substantial relative probability change.

Decoding is an even more direct source. If the engine applies top-p and renormalizes the retained set, the behavior log-probability is the probability **after** truncation. The untruncated softmax probability is not the behavior probability.

At the 2026-08-06 snapshot, official TRL documentation exposes a separate experimental Async GRPO path and vLLM integration, while verl documents corrections for rollouts produced by a different numerical stack or an older policy. These are implementation contracts, not universal rules of RL systems. Research on TIM reports that numerical mismatch can be an independent instability factor in studied configurations; this is an empirical warning, not proof that every run collapses.

Three practical regimes are:

1. **Recompute log-probabilities in the trainer.** This is expensive but uses the same stack as the gradient.
2. **Store actual engine log-probabilities and apply an explicit correction.** This requires a precise behavior policy and variance control.
3. **Unify kernels and configuration.** This reduces mismatch but can restrict available optimizations.

The choice belongs in the experiment protocol.

---

### Step 12.4.4 — What to do with stale data

Asynchrony does not make a rollout “wrong” automatically. It changes the statistical problem. Options include:

- impose a maximum lag $L_{\max}$;
- reject records older than a threshold;
- use clipped or masked importance correction;
- reduce updates per rollout;
- lower the learning rate when ESS falls;
- rebalance pools so the queue stops growing;
- periodically return to a synchronous control run.

Each choice has a cost.

- Rejection improves freshness but wastes expensive rollouts and may favor short episodes.
- Clipping controls variance but changes the estimator.
- Full sequence ratios are natural for trajectory-level correction but become high variance on long responses.
- Token-level ratios are convenient and common but are not automatically equivalent to exact sequence correction under large lag.

AReaL and other asynchronous systems report speedups in their own configurations while introducing lag controls or specialized off-policy mechanisms. These results show that asynchrony can pay off; they do not define one universal safe lag.

Group advantages, clipping, and GRPO diagnostics are developed in *RL for LLM*, Module 9 “RLVR and GRPO: Verifiable Rewards, Group-Relative Estimates, and Stability” (`rl_llm.module_09_rlvr_grpo`). The systems-specific addition is that the lag distribution and queue policy determine which behavior policy belongs in the denominator.

A sound experimental order is:

1. obtain a stable synchronous baseline;
2. enable overlap with measured lag;
3. measure ESS, clipping, and quality;
4. deepen the queue only while statistical signal remains acceptable.

---

### Step 12.4.5 — Weight synchronization must be atomic and measurable

New policy weights can reach the rollout engine through several paths.

- **Full snapshot:** gather, save or transfer, and reload the model. Simple but expensive in time and peak memory.
- **Sharded transfer:** convert an FSDP or Megatron layout directly to the engine layout and transmit shards through collectives.
- **Adapter update:** transfer only LoRA parameters when the engine and library support a consistent update path.
- **Colocation:** switch roles on the same GPUs and use a shared or convertible parameter representation.

Every path needs four properties.

1. **Version identifier.** The engine reports the version it actually serves.
2. **Atomicity.** A new request never sees a mixture of old and new tensors.
3. **Readiness barrier.** The version is not considered available until all required replicas have loaded it.
4. **Verification.** Tensor hashes or control logits confirm that the expected version is active.

Long requests should either finish under old weights or record the version of each action explicitly. Updating weights under a request and then assigning one version to the entire completion is invalid.

Current verl documentation describes checkpoint and weight transfer across training and rollout backends. OpenRLHF and NeMo RL document colocated or sleep/wake variants. Official vLLM documentation for NCCL weight transfer describes multi-node synchronization, a phased update protocol, and an asynchronous-generation example with mid-flight pause, weight synchronization, resume, and validation. These are mutable engineering contracts, not mandatory standards.

---

## Lesson 12.5. Recovery, observability, and framework choice

> **Lesson 12.5. Recovery, observability, and framework choice**
>
> 1. An RL checkpoint is wider than model weights
> 2. A trace must connect data to the update
> 3. Recovery is tested by a failure, not by a directory
> 4. Framework map at the 2026-08-06 snapshot
> 5. Infrastructure is part of the algorithm
>
> then 8 assessment steps

### Step 12.5.1 — An RL checkpoint is wider than model weights

An honest restart requires more than policy parameters. A minimal checkpoint contains:

- model and adapter parameters;
- optimizer state;
- learning-rate scheduler;
- global step and policy version;
- random-number generator states;
- dataset cursor;
- curriculum or prompt-scheduler state;
- reward and advantage normalization;
- tokenizer, template, verifier, and environment versions;
- rollout schema;
- decoding configuration;
- distributed-layout metadata.

If a queue contains stale rollouts, recovery must either preserve it together with behavior versions or discard it explicitly. The same applies to unfinished agent episodes: save environment state or restart and mark them lost.

Engine weights do not necessarily need an independent checkpoint if they are deterministically reconstructed from the trainer checkpoint and synchronization recipe. The recipe—engine version, precision, quantization, and adapter conversion—still belongs in the manifest.

A simple failure model approximates checkpoint-writing and lost-work fraction by

$$
H(I)
\approx
\frac{C}{I}
+
\frac{I}{2M},
$$

where $I$ is checkpoint interval, $C$ is write time, and $M$ is mean time between failures. Its minimum is

$$
I^*=\sqrt{2CM}.
$$

This is an orientation tool, not a complete policy. Asynchronous writes, restart cost, and correlated failures alter the model.

---

### Step 12.5.2 — A trace must connect data to the update

A dashboard says what changed; a trace says which events one record passed through.

The minimal identity chain is:

```text
prompt_id
  -> rollout_id
  -> behavior_policy_version
  -> verifier_version
  -> training_batch_id
  -> optimizer_step
  -> resulting_policy_version
```

![Checkpoint and traceability of an RL run](assets/rl-for-llm/en/module-12/M12_checkpoint_trace_EN.png)

Useful metric families include:

**Data**

- prompt and completion length;
- stop reasons;
- infrastructure-error fraction;
- degenerate-group fraction;
- reward distributions.

**Policy**

- KL to the reference;
- entropy;
- log-probabilities and ratios;
- ESS;
- policy lag.

**System**

- stage-level tokens per second;
- queue wait;
- the 50th, 95th, and 99th percentiles of rollout duration (p50/p95/p99);
- pool utilization;
- synchronization time;
- peak memory;
- retries.

**Reliability**

- age of the latest successful checkpoint;
- save and restore time;
- lost rollouts;
- metric changes across restart.

A metric without configuration provenance is often misleading. Completion length changing after `max_new_tokens` changes is not spontaneous policy dynamics.

---

### Step 12.5.3 — Recovery is tested by a failure, not by a directory

The existence of `checkpoint-1000` does not show that training can continue. Recovery needs a routine drill.

1. Run several steps and save a checkpoint.
2. Record a configuration hash and a small metric baseline.
3. Terminate one or more processes deliberately.
4. Restore trainer, rollout engine, and scheduler.
5. Restore or discard the queue according to the declared policy.
6. Execute one control batch.
7. Check finite losses, policy version, tokenization, and batch composition.
8. Compare statistics against an allowed range.

Bitwise determinism is often too expensive or impossible after changing process count. The more practical goal is to preserve experiment semantics and avoid reporting a cold optimizer, reset curriculum, or changed verifier as a continuation of the same run.

Test at least three failures:

- one rollout worker dies;
- the trainer dies after writing a checkpoint;
- the verifier fails while a batch is only partially written.

A batch should be accepted as one complete transaction or rejected. A partial batch with unknown reward provenance must not silently enter training.

![A recovery drill must continue the same declared event contract](assets/rl-for-llm/en/module-12/M12_recovery_drill_EN.png)

The notebook runs three branches of one teaching experiment. The uninterrupted branch and a full restore—RNG state, data cursor, and component versions included—produce the same event sequence. A weights-only restart diverges immediately. Exact equality is intentionally strict in this single-process example. A distributed run need not be bitwise deterministic, but it must preserve the declared experiment semantics and document acceptable sources of divergence.

---

### Step 12.5.4 — Framework map at the 2026-08-06 snapshot

The following is a dated map, not a ranking. Features change quickly, so recheck the documentation and examples of the exact version before a run.

| Stack | Main abstraction level | Training | Rollout and placement | Asynchrony and agents | Practical profile |
|---|---|---|---|---|---|
| TRL | ready-made trainers around Transformers | Accelerate, DeepSpeed, and related integrations | vLLM colocate or server modes | experimental Async GRPO and environment integrations in current docs | fast research route and small or medium runs |
| verl | programmable HybridFlow RL dataflow | FSDP/DTensor and Megatron paths | vLLM or SGLang; colocated and disaggregated layouts | fully asynchronous and one-step-off recipes, agent loop, checkpoint engine, and rollout correction | flexible large-scale system with explicit version control |
| OpenRLHF | Ray orchestration of roles | DeepSpeed ZeRO | vLLM and a hybrid engine | asynchronous training, partial rollouts, and an agent interface in the current release | distributed RLHF/RLVR recipes |
| NeMo RL | NVIDIA distributed post-training recipes | DTensor or Megatron Core | vLLM and several generation paths | documented Async GRPO with importance correction and environment integrations | large runs in the NVIDIA ecosystem |
| PRIME-RL | fully asynchronous dataflow as described by the project | FSDP2 and specialized paths | vLLM with disaggregated generation | emphasis on fully asynchronous and long-running training | modern engineering stack; project claims need local reproduction |
| SkyRL | modular full-stack system | project-supported distributed backends | vLLM and a unified training/rollout interface | agent layer and asynchronous dispatch | reasoning and multi-turn research |
| AReaL | research-oriented fully asynchronous system | its distributed trainer | streaming rollout workers | explicit lag control and off-policy mechanisms | studying benefits and risks of full asynchrony |

HybridFlow describes RLHF as a distributed dataflow in which each logical model is a program, with layout conversion between training and generation. The official verl fully asynchronous recipe reports author-measured 2.35–2.67× speedups for stated Qwen2.5-7B configurations on 128 GPUs; AReaL reports up to 2.57× over its compared synchronous systems. These are author-protocol results, not guarantees for another cluster, model, or length distribution.

Framework selection begins with requirements:

- one node or many;
- full fine-tuning or LoRA;
- one-shot or multi-turn rollouts;
- synchrony, allowed lag, and queue policy;
- FSDP, Megatron, or DeepSpeed;
- vLLM or SGLang;
- recovery and telemetry requirements;
- the team’s ability to debug the chosen abstraction level.

---

### Step 12.5.5 — Infrastructure is part of the algorithm

The module now forms one continuous chain.

1. The behavior policy samples tokens under a concrete weight version and decoding rule.
2. The rollout stores tokens, masks, log-probabilities, reward, stop reasons, and component versions.
3. The queue bounds age, volume, and cost.
4. The trainer updates relative to the declared behavior policy and monitors probability ratios.
5. Synchronization atomically publishes a new version.
6. A checkpoint preserves enough state to continue the same experiment.
7. A trace connects an individual completion to an optimizer step and back.

The central conclusion is:

> **In online RL, infrastructure determines not only compute speed but also the statistical meaning of data. Queues, weight versions, decoding, and synchronization are part of the algorithm specification.**

A conservative progression for a large run is:

- establish a one-process sequential baseline;
- validate the rollout contract and zero-lag log-probabilities;
- build a component- and phase-level memory ledger;
- measure critical paths and tail latency;
- add overlap before full asynchrony;
- log lag, TIM, ESS, and clipping fraction;
- test recovery with an injected failure;
- compare stacks on the same task and cost units.

The next and final theory module is *RL for LLM*, Module 13 “Evaluating RL Systems for LLMs: Statistics, Reward Hacking, and Safety” (`rl_llm.module_13_evaluation`). The infrastructure contract established here tells us where data came from; Module 13 asks whether conclusions drawn from those data can be trusted.

---

## Sources and further reading

The following primary papers and official documents support the module’s most consequential claims. The full claim-by-claim registry, including evidence class, access date, and limits of each inference, is in [`Module_12_Sources.md`](Module_12_Sources.md).

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — the primary account of staged model-state partitioning; it does not replace an activation and temporary-buffer trace for a concrete run.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the primary PagedAttention and vLLM paper; reported speedups belong to the paper’s configurations.
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — the source for RadixAttention and prefix reuse.
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256) and the [official fully asynchronous recipe](https://verl.readthedocs.io/en/latest/advance/fully_async.html) — distributed RL dataflow and dated author-reported asynchronous results.
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298) — a primary fully asynchronous RL system paper with explicit lag control; numerical gains are author reports.
- [Diagnosing Training–Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/abs/2605.14220) — analysis of probability mismatch between rollout and training stacks; it does not imply that every kernel difference necessarily destabilizes training.
- [TRL: Asynchronous GRPO](https://huggingface.co/docs/trl/async_grpo_trainer), [verl: rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html), [NeMo RL: Async GRPO](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html), [OpenRLHF: RL training](https://openrlhf.readthedocs.io/en/latest/agent_training.html), and [vLLM: NCCL weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/nccl/) — mutable software contracts checked at the 2026-08-06 snapshot; pin resolved versions before execution.
- [A First Order Approximation to the Optimum Checkpoint Interval](https://dl.acm.org/doi/10.1145/361147.361115) — the source of the teaching Young interval; correlated failures and tiered storage require richer models.
