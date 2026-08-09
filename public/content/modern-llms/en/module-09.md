# Module 9. KV Caches and Efficient Inference

*“Modern LLMs” course · Module 9 lecture · edition 2026.8.1*

> **What this module is about.** Module 8 ended when the training run produced a model. That is only the beginning of the system problem. A deployed model must now answer many requests at once, preserve latency targets, reuse work when possible, and fit both its weights and a growing amount of per-request state into a finite memory hierarchy.
>
> The key object is the **KV cache**. It removes the need to recompute the entire prefix at every autoregressive step, yet it also grows with context length and concurrency. We will derive its size from the attention geometry, place it inside a simple bandwidth model, locate the point at which batching stops amortizing the weights efficiently, and then follow the engineering consequences: paged allocation, prefix reuse, tiered storage, prefill/decode disaggregation, quantization, and eviction.
>
> Every number in the worked examples belongs to an explicit teaching scenario. A bandwidth ceiling is not a benchmark, a vendor result is not an independent measurement, and an API price is not a permanent property of the underlying mechanism. Keeping those categories separate is part of the lesson.
>
> **Useful background.** Module 4 introduced MHA, GQA, and MLA; Module 7b introduced roofline reasoning; Module 8 used an inference term in the lifecycle break-even calculation. The present module supplies the systems mechanics behind that term.

---

## 1. Motivation: a generated token is a memory transaction

During training, thousands of tokens can contribute to large matrix multiplications before the weights must be fetched again. Autoregressive decoding has a less favorable shape. Each active sequence asks for one new token, the model reads its weights, and attention reads the state accumulated for the preceding tokens. The amount of arithmetic per byte can therefore become very small even on an accelerator with enormous peak FLOP/s.

This is why parameter count alone is a poor predictor of serving behavior. Two models with similar active parameters may store radically different amounts of history. A conventional multi-head attention model keeps one key and one value vector for every head at every layer. GQA reduces the number of KV heads. MLA stores a compressed latent representation. Hybrid architectures may combine full-attention layers with recurrent or convolutional state. Before estimating latency or capacity, we must first ask a seemingly mundane question: **what exactly is retained after each token?**

The answer affects three quantities at once.

First, it affects **capacity**. A cache that is modest at 8K tokens may exceed the weight payload at one million tokens. Second, it affects **bandwidth**. Decoding must read the relevant state repeatedly, so longer contexts lower the memory-bound throughput ceiling. Third, it affects **scheduling**. A server that mixes long prompts, short prompts, cache hits, and growing outputs needs a memory manager rather than a single preallocated tensor.

The rest of the module follows that chain. We begin with bytes per token, turn those bytes into a speed ceiling, and then ask what a serving stack can do about the result.

## 2. The historical arc: four successive bottlenecks

![VIZ m9/01 — from a local buffer to a serving platform](assets/modern-llms/en/module-09/m9_01_timeline.svg)

The first wave changed the **architecture**. Multi-Query Attention (MQA) showed that all query heads could share a much smaller set of key/value heads. Generalized Multi-Query Attention (GQA) retained several KV groups and offered a better quality–memory compromise for many dense models. Multi-head Latent Attention (MLA) went further by storing a latent representation from which the required key/value information can be reconstructed.

The second wave changed the **scheduler**. Orca observed that sequences in a batch do not finish together and that waiting for a whole batch to complete wastes capacity. Its iteration-level scheduling allowed the active set to change after every decode step. The modern term *continuous batching* describes this family of schedulers: completed requests leave, new requests enter, and the GPU keeps doing useful work.

The third wave changed the **allocator**. Variable-length sequences make contiguous KV buffers expensive. PagedAttention borrowed the central idea of virtual memory: a logical sequence can be represented by a table of fixed-size physical blocks. This reduces fragmentation, supports copy-on-write sharing, and lets the scheduler grow sequences without reserving their maximum length in advance. vLLM made this design widely accessible; SGLang’s RadixAttention emphasized structured reuse of common prefixes.

The fourth wave changed the **scope of the cache**. Once the KV state became valuable enough, keeping it inside one process was no longer sufficient. LMCache added external storage and transfer paths. Mooncake treated CPU memory, local storage, and networked resources as a global KV pool. Prefill/decode disaggregation placed the two phases on different worker pools and turned KV transfer into an explicit systems operation. By 2026, context state had become a first-class distributed resource with its own placement, routing, and lifecycle policies.

No wave made the previous ones obsolete. A current serving system typically combines architectural reduction, iteration-level scheduling, paged allocation, prefix reuse, and a memory hierarchy. Each mechanism removes a different term from the cost.

## 3. A bridge to classical systems: pages, caches, and queues

Three older systems ideas provide a useful mental model.

**Virtual memory.** A process sees a contiguous address space even though physical pages may be scattered. PagedAttention applies the same separation to a sequence: logical token positions map to fixed-size KV blocks, and the blocks need not be adjacent. Sharing a prefix then means sharing page-table entries; copy-on-write creates a private block only when one branch diverges.

**Cache hierarchies.** HBM is fast and scarce, CPU DRAM is larger and slower, local SSD is larger again, and remote storage adds network cost. Moving KV between these levels is governed by the same questions that appear in databases and operating systems: expected reuse, transfer latency, eviction policy, and prefetch timing. The unusual feature is the object being cached: a representation whose recomputation cost grows with the prefix length.

**Queueing and phase separation.** Prefill and decode stress the accelerator differently. A long prefill can occupy compute resources for a relatively large contiguous interval, while interactive decode consists of many short latency-sensitive steps. A system may have excellent average throughput and still violate user-facing latency objectives because the two phases interfere in the tail. Goodput—the work completed while meeting the service-level objective—is therefore more informative than raw throughput alone.

These analogies are not just pedagogical decoration. They tell us which knobs to expect: block size, admission control, queue discipline, cache key, time-to-live, placement tier, prefetch, and backpressure.

## 4. Formalism and worked example A: how many bytes does one token add?

For ordinary MHA or GQA, let

- $L$ be the number of attention layers;
- $H_{kv}$ the number of KV heads per layer;
- $d_h$ the head dimension;
- $b$ the number of bytes per stored scalar.

Each token contributes one key vector and one value vector at every attention layer, so

$$
 b_{KV}=2L H_{kv}d_hb.
$$

The factor of two is not an approximation: it is the pair $K$ and $V$. What *is* architecture-dependent is the object represented by $H_{kv}d_h$. MLA, recurrent attention variants, and hybrid models may store a latent or several kinds of state instead of ordinary key/value heads; those cases require their own geometry.

For bf16, $b=2$. The teaching configurations give the following ladder:

| Architecture | Configuration | Stored state per token | Interpretation |
|---|---|---:|---|
| 70B MHA equivalent | $L=80$, $H_{kv}=64$, $d_h=128$ | **2,560 KiB** | full KV for every attention head |
| Llama-3.3-70B GQA | $L=80$, $H_{kv}=8$, $d_h=128$ | **320 KiB** | eightfold reduction from the MHA equivalent |
| Llama-3-8B GQA | $L=32$, $H_{kv}=8$, $d_h=128$ | **128 KiB** | reference geometry for the roofline example |
| DeepSeek-V3 MLA | model-specific latent + RoPE state | **68.625 KiB** | a separate MLA calculation, not the GQA formula |

The units matter. Dividing bytes by $2^{10}$ gives KiB; accelerator and operating-system tools often report binary capacities even when product names use decimal “GB”.

Now consider the 70B GQA row. It stores 327,680 bytes per token. At batch one:

- 8,192 tokens require 2.5 GiB;
- 32,768 tokens require 10 GiB;
- 131,072 tokens require 40 GiB;
- 1,000,000 tokens require 327.68 GB, or about 305.18 GiB.

A bf16 weight payload of 70 billion parameters is roughly 140 GB. The cache crosses that weight payload at

$$
N_{cross}=\frac{140\cdot 10^9}{327{,}680}\approx427{,}246\ \text{tokens}.
$$

Beyond that point, a *single* sequence carries more KV state than the model’s weights. Concurrency multiplies the cache again. This is the arithmetic behind the statement that long-context serving is a memory problem.

![VIZ m9/02 — the KV ladder and the weight crossing](assets/modern-llms/en/module-09/m9_02_kv_ladder.png)

## 5. One request, two computational regimes

An inference request has two phases, and treating them as one homogeneous workload hides the central scheduling problem.

During **prefill**, the model processes the prompt and constructs its state. Many positions participate in matrix operations at once, so the accelerator can often use its compute units efficiently. The user experiences this phase through **time to first token (TTFT)**.

During **decode**, the model adds one token per active sequence per step. The weights are read again, and attention reads the accumulated KV state. Arithmetic no longer grows in proportion to the amount of data moved, so the phase is often memory-bandwidth bound. The user experiences it through **inter-token latency (ITL)** or the corresponding stream rate after the first token.

The distinction is workload-dependent, not metaphysical. Very short prompts may not make prefill fully compute-bound; large batches and fused kernels can improve decode utilization; sparse or recurrent architectures alter the state traffic. Nevertheless, the two-regime model is a useful first approximation because it explains why one long prompt can disturb many otherwise smooth decode streams.

A service-level objective may constrain TTFT, ITL, a percentile of end-to-end latency, or all three. Interactive chat, offline batch generation, and agent rollout generation place different weights on these metrics. The scheduler should therefore maximize useful work **under the chosen constraints**, not merely display the largest aggregate token count.

![VIZ m9/03 — two regimes inside one request](assets/modern-llms/en/module-09/m9_03_two_regimes.svg)

## 6. Worked example B: the bandwidth ceiling and the KV term

Take an 8B bf16 model with a 16 GB weight payload, the 128 KiB/token geometry above, and an H100-like memory bandwidth of 3.35 TB/s. We deliberately use a simple model: one decode step reads the weights once and reads the resident KV state once. Ignoring compute and all other traffic gives the aggregate bandwidth ceiling

$$
Q(B,N)\leq\frac{B\,BW}{W+B N b_{KV}},
$$

where $B$ is the number of active sequences, $N$ the context length, $W$ the weight bytes, and $BW$ the memory bandwidth. For batch one this reduces to

$$
Q(1,N)\leq\frac{BW}{W+N b_{KV}}.
$$

The resulting ceilings are:

| Context | Cache traffic per sequence | Ceiling at batch 1 |
|---:|---:|---:|
| 0 | 0 | **209.4 tok/s** |
| 8K | 1.0 GiB | **196.2 tok/s** |
| 128K | 16 GiB | **101.0 tok/s** |
| 1M | 122.1 GiB | **22.8 tok/s** |

These are not expected wall-clock measurements. They omit kernels, synchronization, activation traffic, scheduler overhead, and the fact that not every implementation reads every byte in exactly this way. Their value is diagnostic: the equation shows why the ceiling falls as context grows even though the weights have not changed.

At 128K, the teaching calculation gives an arithmetic intensity of 0.4822 FLOP/byte. With a 990 TFLOP/s peak and 3.35 TB/s bandwidth, the roofline ridge is

$$
\frac{990}{3.35}=295.52\ \text{FLOP/byte}.
$$

The decode point lies about 612.8 times below that ridge. Peak arithmetic throughput is therefore not the limiting resource in this simplified regime.

## 7. Batch economics: where weight amortization stops being the whole story

Batching helps because several sequences can share one read of the weights. If KV traffic were zero, aggregate throughput would initially grow almost linearly with batch size. The denominator above shows why the gain eventually bends: each additional sequence brings its own context state.

Set total KV traffic equal to the weight payload:

$$
B^*=\frac{W}{N b_{KV}}.
$$

This is not a hard optimum. It marks a useful transition: before $B^*$, reading the weights dominates the denominator; after it, per-request KV traffic increasingly controls the gain.

For the teaching scenario:

- at 8K, $B^*\approx14.90$;
- at 32K, $B^*\approx3.73$;
- at 128K, $B^*\approx0.93$.

The last value is the striking one. At 128K, one sequence already brings roughly as much KV traffic as the weights. Batching remains useful, but it no longer behaves like free weight amortization.

At batch 32 and 8K, the model predicts 2,128.7 tok/s rather than the 6,700 tok/s one would obtain by multiplying the zero-KV batch-one ceiling by 32. The ratio, about 3.15×, is the “KV tax” in this particular arithmetic scenario.

![VIZ m9/04 — batch knees under growing context](assets/modern-llms/en/module-09/m9_04_roofline_batch.png)

## 8. PagedAttention: decouple logical length from physical placement

A naïve server can reserve one contiguous KV buffer per request, sized for that request’s maximum allowed context. The policy is simple and usually wasteful. Most requests finish before the limit, sequences grow at different rates, and free regions become fragmented. Worse, a request that needs one more token may be unable to extend even when enough memory exists elsewhere.

PagedAttention replaces the contiguous buffer with a table of fixed-size blocks. The logical token positions remain ordered, but their physical blocks may sit anywhere in the pool. A request receives blocks as it grows, not when it is admitted.

The internal waste is then bounded by the unused tail of the final block. With block size $P$, a sequence wastes at most $P-1$ token slots. For a 32K maximum context, a request that currently contains 2K tokens would waste

$$
1-\frac{2{,}048}{32{,}768}=93.75\%
$$

under full contiguous reservation. With 16-token blocks and a uniformly distributed final-block occupancy, the expected unused fraction is approximately

$$
\frac{(16-1)/2}{2{,}048}\approx0.37\%.
$$

Both figures belong to the stated example. They are not universal measurements of vLLM. The original PagedAttention work reported near-zero memory waste and 2–4× throughput improvements over its comparison systems, but the realized gain depends on length distribution, block size, scheduler, and hardware.

Block allocation also enables **copy-on-write**. Two branches can point to the same complete prefix blocks. A physical copy is needed only when a branch must modify a shared partial block. This matters for beam-like branching, speculative decoding, and any workflow that forks a common conversation state.

![VIZ m9/05 — pages instead of contiguous buffers](assets/modern-llms/en/module-09/m9_05_paged.svg)

## 9. Prefix caching: do not rebuild an identical beginning

Many requests begin with the same material: a system instruction, a few-shot demonstration set, a tool schema, a large document, or an earlier part of a conversation. If the token sequence and every computation-relevant setting match, the corresponding KV state can be built once and reused.

Consider the 8K prompt in the notebook scenario. The arithmetic lower bound for full prefill is 132.4 ms. Suppose 7.5K tokens are already cached and only a 512-token suffix remains. The corresponding lower bound is 8.27 ms, a 16× ratio. Real systems must also look up the entry, move or map blocks, schedule the request, and verify the exact-match conditions. The saving nevertheless has a clear source: the repeated matrix work for the common prefix disappears.

By 2026, prompt caching had become part of public API contracts as well as an internal engine optimization. Anthropic documents cache-read tokens at 0.1 times the base input-token price. OpenAI’s GPT-5.6 release of 9 July 2026 introduced explicit cache breakpoints and a minimum cache lifetime of 30 minutes; cache writes are priced at 1.25 times uncached input, while cache reads receive a 90% discount. Kimi K3’s API likewise separates cache-hit from cache-miss input pricing. Moonshot reports a cache-hit rate above 90% for the official API on its own coding workloads, a service-specific result enabled by its disaggregated Mooncake infrastructure. Prices and hit rates can change with the service and traffic mix; the reusable-prefix mechanism does not depend on either number.

Exact-prefix matching leads to a practical prompt-layout rule: place stable instructions and reusable documents first, and put request-specific material later. The rule has limits. A provider or engine may hash blocks differently, include model settings in the cache key, impose a minimum length, restrict sharing scope, or invalidate entries when images, tools, or model versions change. The relevant documentation must therefore be read as part of the serving contract.

Prefix caching offers little when the common portion is small, tokenization differs, or every request carries unique context. A process-local cache is also bounded by the process’s memory and lifetime. The next step is to make the state portable beyond one engine instance.

![VIZ m9/06 — prompt layout determines reuse](assets/modern-llms/en/module-09/m9_06_prefix.svg)

## 10. Tiered placement: when HBM is only the first level

If useful KV state does not fit in HBM, the system can recompute it later or preserve it on a cheaper tier. The second option creates a hierarchy such as

$$
\text{HBM}\rightarrow\text{CPU DRAM}\rightarrow\text{local SSD}\rightarrow\text{remote storage}.
$$

Capacity rises along the hierarchy, while access latency and transfer cost rise as well. Offloading is therefore not automatically beneficial. Keeping a block is worthwhile when its expected reuse value and recomputation cost exceed the cost of writing, storing, and restoring it.

**LMCache** separates KV storage and transfer from a particular inference engine. A deployment may use CPU memory or disk on the same node, a separate LMCache server shared by several vLLM instances, or external backends such as Redis/Valkey, Mooncake, S3-compatible storage, NIXL, or GDS. Some configurations reuse exact prefixes; CacheBlend extends the idea to selected blocks that reappear in different positions with partial recomputation. The important abstraction is that KV becomes managed data rather than a private tensor owned by one attention kernel.

**Mooncake** applies the idea at distributed-platform scale. In the published Kimi system, prefill and decode are separated, idle CPU memory and storage resources form a global KV pool, and a transfer engine moves state across the network. The scheduler considers service-level objectives and can reject requests early when the available capacity cannot meet them. In the FAST 2025 paper, the authors report 59–498% higher effective serving capacity in their trace-driven experiments while meeting the selected SLOs, operation across thousands of nodes, and more than 100 billion tokens served per day. These are results for the authors’ system and workloads, not generic factors for any cluster.

NVIDIA’s CMX announcement of 16 March 2026 describes a hardware-and-network context-memory layer built around BlueField-4 and Spectrum-X for the Vera Rubin ecosystem. NVIDIA reports up to 5× higher tokens per second and energy efficiency against a chosen conventional storage design. At the time of the course snapshot, that figure is a vendor projection for a specific forthcoming platform rather than a broadly reproduced H100 result.

The systems principle is stable even when products change: a large context state needs an explicit placement policy, a transfer path, and a decision about whether retrieval or recomputation is cheaper.

![VIZ m9/07 — a hierarchy for context state](assets/modern-llms/en/module-09/m9_07_tiered.svg)

## 11. Prefill/decode disaggregation: give each phase its own pool

Return to the two regimes from §5. If prefill and decode share one scheduler queue, a long prefill can delay every active decode stream. In the teaching scenario, an 8K prefill has a 132.396 ms arithmetic lower bound, while the weight-read lower bound for one decode step is 4.776 ms. Executed serially, they create a 137.172 ms interruption.

One local remedy is **chunked prefill**. Instead of treating the prompt as one indivisible operation, the scheduler processes a smaller chunk and lets decode steps run between chunks. With a 512-token chunk, the corresponding teaching bound becomes approximately

$$
4.776+8.275=13.051\ \text{ms}.
$$

The prefill work has not vanished; it has been divided and interleaved so that its tail-latency effect is smaller.

A more structural remedy is **prefill/decode disaggregation**. Prefill workers can be configured for large prompt matrix operations and throughput; decode workers can be configured for low ITL and memory bandwidth. The KV state is transferred between the pools after prefill. Each pool can use a different tensor-parallel degree, admission policy, and batching strategy.

The separation is not free. KV transfer joins the critical path, the scheduler must know where each block resides, and network backpressure becomes part of request admission. For small loads or short prompts, one pool with continuous batching and chunked prefill may be simpler and faster.

As of 3 August 2026, vLLM’s stable documentation still labels disaggregated prefill experimental, while exposing several KV connectors and independent TTFT/ITL tuning. NVIDIA Dynamo 1.0, released on 16 March 2026, orchestrates vLLM, SGLang, and TensorRT-LLM, uses NIXL for data movement, and routes requests with KV locality in mind. NVIDIA reports up to 7× more requests served on Blackwell in a specific SemiAnalysis InferenceX DeepSeek-R1 configuration. That number belongs to the stated stack and protocol; it is not a universal multiplier for phase separation itself.

> **Connection to RL infrastructure.** This module focuses on interactive and service inference. Large-scale rollout generation, actor/learner separation, policy staleness, and asynchronous queues are developed in *RL for LLM*, Module 12. The same memory and scheduling primitives reappear there under a different workload profile.

![VIZ m9/08 — separate pools for separate regimes](assets/modern-llms/en/module-09/m9_08_pd_disagg.svg)

## 12. Code (level 1): geometry, bandwidth, and pages

The following three primitives correspond to the three layers developed so far. `KVGeometry` makes the architectural state explicit. `decode_ceiling` evaluates the simplified bandwidth bound. `PagedKV` separates logical sequence growth from physical block allocation. This is teaching code rather than a serving engine: it deliberately omits CUDA kernels, reference counts, a scheduler, and network transport so that the assumptions remain visible.

```python
from dataclasses import dataclass, field
import math

@dataclass(frozen=True)
class KVGeometry:
    """Ordinary MHA/GQA cache geometry; the result includes K and V."""
    attention_layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int = 2

    def __post_init__(self):
        values = (self.attention_layers, self.kv_heads,
                  self.head_dim, self.bytes_per_element)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
            raise ValueError("all KV dimensions must be positive integers")

    @property
    def bytes_per_token(self):
        return 2 * self.attention_layers * self.kv_heads * self.head_dim * self.bytes_per_element

def decode_ceiling(context_tokens, weight_bytes, geometry, bandwidth_Bps,
                   batch_size=1, kv_scale=1.0):
    """A bandwidth ceiling, not a wall-clock benchmark or total device memory."""
    if context_tokens < 0 or batch_size <= 0 or kv_scale <= 0:
        raise ValueError("invalid context, batch, or scale")
    kv_bytes = geometry.bytes_per_token * context_tokens * batch_size * kv_scale
    if not math.isclose(kv_bytes, round(kv_bytes), abs_tol=1e-9):
        raise ValueError("the scaled KV payload must be an integer byte count")
    return batch_size * bandwidth_Bps / (weight_bytes + round(kv_bytes))

@dataclass
class PagedKV:
    """Minimal page table; full partial-block COW is implemented in the notebook."""
    n_blocks: int
    block_size: int = 16
    free: list[int] = field(init=False)
    tables: dict[str, list[int]] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_blocks <= 0 or self.block_size <= 0:
            raise ValueError("pool and block_size must be positive")
        self.free = list(range(self.n_blocks - 1, -1, -1))

    def append_to(self, seq_id, total_tokens):
        """Increase logical length; allocation is checked before page-table mutation."""
        if total_tokens < self.lengths.get(seq_id, 0):
            raise ValueError("append_to cannot shrink a sequence")
        table = self.tables.get(seq_id, [])
        required = math.ceil(total_tokens / self.block_size) if total_tokens else 0
        need = required - len(table)
        if need > len(self.free):
            raise MemoryError("not enough physical blocks")
        if seq_id not in self.tables:
            self.tables[seq_id] = table
        table.extend(self.free.pop() for _ in range(need))
        self.lengths[seq_id] = total_tokens

    def share_full_prefix(self, src, dst, prefix_tokens):
        """Share full blocks only; partial-block COW is demonstrated in the notebook."""
        if dst in self.tables or prefix_tokens % self.block_size:
            raise ValueError("dst must be new and prefix must align to block_size")
        if prefix_tokens > self.lengths[src]:
            raise ValueError("prefix exceeds source length")
        count = prefix_tokens // self.block_size
        self.tables[dst] = list(self.tables[src][:count])
        self.lengths[dst] = prefix_tokens
```

Read the code in the same order as the lecture. `bytes_per_token` fixes the architectural geometry. `decode_ceiling` multiplies it by context and concurrency before adding the weight traffic. The `kv_scale` argument represents an idealized storage reduction; it does not include scales, metadata, dequantization cost, or kernel efficiency and therefore is not a benchmark for a quantized cache.

`PagedKV.append_to` verifies capacity before changing the page table. `share_full_prefix` copies block identifiers rather than KV bytes. The notebook adds reference counts, copy-on-write for a partially shared tail block, and block release.

### NumPy → PyTorch · B17 — Bounded KV cache: append, window, and rollback

The cache stores K/V together with an absolute position range. A bounded window physically evicts the old prefix, while rollback truncates only the retained suffix.

```python
import numpy as np
import torch
```

#### 1. Explicit NumPy implementation

```python
class NumpyKVCacheExplicit:

    def __init__(self, *, window: int | None = None) -> None:
        self.window = window
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.start_pos = 0
        self.next_pos = 0

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        self.k = k_new.copy() if self.k is None else np.concatenate([self.k, k_new], axis=-2)
        self.v = v_new.copy() if self.v is None else np.concatenate([self.v, v_new], axis=-2)
        self.next_pos += k_new.shape[-2]
        if self.window is not None and self.k.shape[-2] > self.window:
            drop = self.k.shape[-2] - self.window
            self.k, self.v = self.k[..., drop:, :], self.v[..., drop:, :]
            self.start_pos += drop

    def truncate(self, absolute_end: int) -> None:
        keep = absolute_end - self.start_pos
        self.k, self.v = self.k[..., :keep, :], self.v[..., :keep, :]
        self.next_pos = absolute_end

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.start_pos, self.next_pos, dtype=np.int64)
```

The minimal cache exposes three pieces of state: retained K/V, the absolute position of the first retained token, and the next append position. The window physically evicts the old prefix.

#### 2. The same mathematics as explicit PyTorch operations

```python
class TorchKVCacheExplicit:

    def __init__(self, *, window: int | None = None) -> None:
        self.window = window
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.start_pos = 0
        self.next_pos = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        self.k = k_new.clone() if self.k is None else torch.cat([self.k, k_new], dim=-2)
        self.v = v_new.clone() if self.v is None else torch.cat([self.v, v_new], dim=-2)
        self.next_pos += k_new.shape[-2]
        if self.window is not None and self.k.shape[-2] > self.window:
            drop = self.k.shape[-2] - self.window
            self.k, self.v = self.k[..., drop:, :], self.v[..., drop:, :]
            self.start_pos += drop

    def truncate(self, absolute_end: int) -> None:
        keep = absolute_end - self.start_pos
        self.k, self.v = self.k[..., :keep, :], self.v[..., :keep, :]
        self.next_pos = absolute_end

    @property
    def positions(self) -> torch.Tensor:
        device = self.k.device if self.k is not None else None
        return torch.arange(self.start_pos, self.next_pos, dtype=torch.long, device=device)
```

The PyTorch class repeats the same mutable contract with `torch.cat` and slicing. Tensors stay on the input device; a production block cache avoids full concatenation at every step.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Append | `np.concatenate` | `torch.cat` |
| Physical window | slice along `T` | same slicing |
| Absolute positions | Python integers | Python integers plus tensor state |
| Rollback | truncates the retained suffix | same contract |
| Production difference | reference container | usually preallocated/block storage |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
np_cache = NumpyKVCacheExplicit(window=3); t_cache = TorchKVCacheExplicit(window=3)
for value in range(5):
    k_np = np.full((1,1,1,2), value, dtype=np.float64); v_np = k_np + 100
    np_cache.append(k_np, v_np); t_cache.append(torch.tensor(k_np), torch.tensor(v_np))
np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
np.testing.assert_array_equal(np_cache.positions, t_cache.positions.numpy())
assert (np_cache.start_pos, np_cache.next_pos) == (2,5)
np_cache.truncate(4); t_cache.truncate(4)
np.testing.assert_allclose(np_cache.k, t_cache.k.numpy())
rng = np.random.default_rng(17)
keys = rng.standard_normal((1,1,6,4)); values = rng.standard_normal((1,1,6,4)); queries = rng.standard_normal((1,1,6,4))
cache = TorchKVCacheExplicit(window=3)
for step in range(6):
    cache.append(torch.tensor(keys[..., step:step+1, :]), torch.tensor(values[..., step:step+1, :]))
    cached = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), cache.k, cache.v)
    lo = max(0, step - 2)
    full = F.scaled_dot_product_attention(torch.tensor(queries[..., step:step+1, :]), torch.tensor(keys[..., lo:step+1, :]), torch.tensor(values[..., lo:step+1, :]))
    torch.testing.assert_close(cached, full)
print("B17 bounded cache + rollback: PASS")
```

</details>

Complete executable file: [`m09_cache_bridges.py`](../assets/m09_cache_bridges.py)

## 14. Compression I: eviction changes which history exists

When full state cannot be retained even with a memory hierarchy, the server may keep only a subset of positions. This is the strongest form of compression: once a KV entry is removed, recovering it requires recomputing the relevant prefix.

**StreamingLLM** keeps a small set of initial attention-sink tokens together with a sliding recent window. The rule is cheap and predictable. **H2O** keeps recent tokens plus heavy hitters with high accumulated attention mass. **SnapKV** estimates prompt-token importance from an observation window; QUEST, Ada-KV, and PyramidKV allocate budgets in query- or layer-dependent ways.

No policy is universally best. The notebook uses a deterministic 16-position example with a budget of eight. H2O retains $\{0,1,2,5,7,8,14,15\}$, while a positional “first four plus last four” rule keeps $\{0,1,2,3,12,13,14,15\}$ and loses heavy hitters $\{5,7,8\}$. The example does not prove that H2O wins generally; it demonstrates the difference between content-aware and position-only decisions.

Results published in 2025–2026 add two cautions. Reasoning workloads can be especially sensitive to eviction, and compression may lengthen the generated reasoning chain even when final accuracy is preserved. Another line of work shows that protecting structural prompt boundaries can matter as much as the scoring rule. Evaluation must therefore include the real prompt format, long decode, and total generated-token cost—not only retrieval accuracy on a short benchmark.

## 15. Compression II: quantization, low rank, and the order of operations

Eviction reduces the number of positions. Quantization keeps the positions but represents each one with fewer bits.

**KIVI** uses asymmetric 2-bit quantization: keys are quantized per channel and values per token. The paper reports up to 2.6× lower peak memory including weights, up to 4× larger batches, and 2.35–3.47× higher throughput on the evaluated workloads. The factors are smaller than the raw 16/2 ratio because the system still stores weights, scales, metadata, and a residual full-precision window and pays kernel overhead.

**TurboQuant** treats KV compression as streaming vector quantization. PolarQuant transforms vectors into a quantization-friendly representation, while QJL corrects dot-product estimates. The paper reports a near-lossless regime around 3.5 bits per channel and small degradation at 2.5 bits; Google Research reports up to 8× KV compression and up to 6× throughput in selected settings. These are results of the reference method and implementations, not a guarantee for every model or reasoning workload.

Low-rank methods attack another redundancy: the stored vectors may lie near a lower-dimensional subspace. They can be attractive when the projection cost and reconstruction kernels are well optimized, but rank is workload- and layer-dependent.

The practical order of levers is useful:

1. **architecture** — reduce the state the model creates, through GQA, MLA, or another mechanism;
2. **quantization or low rank** — preserve positions while reducing bytes per position;
3. **placement** — move colder state to a slower tier;
4. **eviction** — remove positions only when the previous levers are insufficient.

The order is not a theorem. It is a risk ordering: later steps are more likely to make specific long-range information unavailable.

> **Optional depth.** Rate–distortion reasoning and the distinction between lossy coding and task-aware distortion are developed in *Information Theory for ML*, Modules 6–7. The present module uses only the engineering consequence: fewer bits buy capacity at the price of controlled representation error.

![VIZ m9/09 — the compression ladder](assets/modern-llms/en/module-09/m9_09_compression.svg)

## 16. Engines and cache contracts: a dated map, not a permanent ranking

Serving software changes faster than the arithmetic. A useful map therefore describes responsibilities rather than declaring a single winner.

| Layer | Representative systems | Main responsibility |
|---|---|---|
| execution engine | vLLM, SGLang, TensorRT-LLM | kernels, batching, paged KV, model-specific execution |
| reuse and transfer | LMCache, Mooncake Store, NIXL connectors | external KV storage, movement, and sharing |
| orchestration | NVIDIA Dynamo and comparable platforms | routing, locality, admission, multi-engine deployment |
| public API contract | OpenAI, Anthropic, Kimi and others | cache key semantics, lifetime, minimum size, billing |

vLLM combines paged KV management, continuous batching, chunked prefill, prefix caching, quantized KV, and experimental disaggregated prefill. SGLang emphasizes RadixAttention, structured execution, and aggressive reuse. TensorRT-LLM integrates tightly with NVIDIA kernels and deployment tooling. LMCache and Mooncake extend the cache beyond one engine process. Dynamo coordinates several engines and transports rather than replacing them.

The public API layer deserves separate attention. A provider may expose prompt caching without revealing its allocator, may bill writes and reads differently, and may change cache lifetime or model compatibility. For reproducible cost analysis, store the provider, model version, date, cache conditions, and price sheet together with the result.

## 17. Reading a model’s inference passport

Before estimating service capacity, collect a compact inference passport.

**Architecture and state**

- number of attention layers;
- attention type by layer: MHA, GQA, MLA, sparse, recurrent, or hybrid;
- KV heads, head dimension, latent dimensions, and any retained recurrent state;
- cache datatype or quantization scheme, group size, scales, and residual window.

**Execution and hardware**

- weight format and weight payload;
- target accelerator, HBM capacity, memory bandwidth, and interconnect;
- tensor-, pipeline-, and expert-parallel degrees;
- supported kernels and engine version.

**Workload and objectives**

- input/output length distributions rather than one maximum;
- concurrency and arrival process;
- prefix-hit distribution and sharing scope;
- TTFT/ITL or end-to-end SLO percentiles;
- acceptable quality loss from compression.

**Placement and lifecycle**

- block size, cache key, time-to-live, and invalidation rules;
- HBM/CPU/SSD/remote tiers and transfer bandwidth;
- behavior across engine restarts and model-version changes;
- admission control and overload policy.

Only after this passport is known does a formula such as $2L H_{kv}d_hb$ become an operational estimate. A nominal “1M context” says nothing by itself about concurrency, latency, or cost.

## 20. Key takeaways and sources

![VIZ m9/10 — efficient inference on one page](assets/modern-llms/en/module-09/m9_10_cheatsheet.svg)

**KV geometry comes first.** For ordinary attention,

$$
b_{KV}=2LH_{kv}d_hb.
$$

The teaching ladder is 2,560, 320, and 128 KiB/token for the MHA/GQA configurations and 68.625 KiB/token for the separate MLA geometry. A hybrid model requires an inventory of its actual stored state before any capacity calculation.

**Context lowers the memory-bound ceiling.** In the simplified model,

$$
Q(B,N)\leq\frac{B\,BW}{W+B N b_{KV}}.
$$

For the 8B bf16 scenario, batch-one throughput falls from 209.4 tok/s with no KV to 101.0 at 128K and 22.8 at 1M. These are arithmetic ceilings, not measurements of a particular engine.

**Batching has a changing return.** $B^*=W/(Nb_{KV})$ marks the scale at which aggregate KV traffic catches the weight payload. Longer context moves the knee toward smaller batches.

**Memory management and state reduction are different jobs.** PagedAttention reduces allocation waste and enables sharing. Prefix caching removes repeated prefill for exact common beginnings. LMCache and Mooncake extend placement and reuse beyond one process or one memory tier. Prefill/decode disaggregation separates latency-sensitive decode from long prompt work at the cost of explicit KV transport.

**Compression levers act on different quantities.** Architectural methods change the base state, quantization changes bytes per coordinate, tiering changes location, and eviction changes which positions survive. They can be combined, but their quality and operational risks differ.

### Primary sources

- MQA: [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150)
- GQA: [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245)
- iteration-level scheduling / continuous batching: [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)
- PagedAttention and vLLM: [*Efficient Memory Management for Large Language Model Serving*](https://arxiv.org/abs/2309.06180) and [vLLM stable documentation](https://docs.vllm.ai/en/stable/)
- RadixAttention and SGLang: [SGLang](https://arxiv.org/abs/2312.07104)
- external and tiered KV: [LMCache repository](https://github.com/LMCache/LMCache), [LMCache/vLLM examples](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/), and [Mooncake, FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)
- distributed orchestration: [NVIDIA Dynamo 1.0](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) and [NVIDIA CMX](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/)
- public caching contracts: [OpenAI GPT-5.6](https://openai.com/index/gpt-5-6/), [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), [Kimi K3](https://www.kimi.com/blog/kimi-k3), and [vLLM’s Kimi K3 serving note](https://vllm.ai/blog/2026-07-27-k3)
- hybrid state geometry: [DeepSeek V4 technical report](https://arxiv.org/abs/2606.19348) and [vLLM’s DeepSeek V4 serving note](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- eviction: [StreamingLLM](https://arxiv.org/abs/2309.17453), [H2O](https://arxiv.org/abs/2306.14048), [SnapKV](https://arxiv.org/abs/2404.14469), [reasoning benchmark](https://arxiv.org/abs/2512.12008), and [structural-protection study](https://arxiv.org/abs/2605.18053)
- quantization: [KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079), [TurboQuant](https://arxiv.org/abs/2504.19874), and [Google Research overview](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

**Next.** Module 10 turns from state management to token generation itself: sampling, speculative decoding, constrained generation, and the mechanisms that extract more useful work from each expensive decode step.

---

*Landscape snapshot: 3 August 2026. Engine features, cache APIs, prices, and infrastructure releases are dated facts. The H100 quantities in the worked examples belong to a fixed teaching configuration; for another accelerator, substitute its capacity, memory bandwidth, and peak FLOP/s.*

*Landscape verified: 4 August 2026; dated claims about models, optimizers, APIs, and serving systems were checked against primary sources, while pricing and availability should be rechecked before operational use.*
