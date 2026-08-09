# Module 6. Mixture of Experts

*“Modern LLMs” course · Module 6 lecture · edition 2026.8*

> **What this module is about.** The previous module showed that most parameters in a conventional Transformer block live in the feed-forward network. A **Mixture of Experts (MoE)** exploits that concentration: it stores many feed-forward experts but routes each token through only a few of them. Capacity can therefore grow much faster than per-token expert compute. That advantage is only the beginning of the story. The full expert bank still has to be stored, token assignments must be balanced, and hidden states may have to cross a cluster before a selected expert can process them. We will follow a token from router scores to top-k selection, weighted expert output, load control, all-to-all dispatch, and the newer LatentMoE design. Along the way, we will rebuild gpt-oss and DeepSeek-V3 parameter counts from explicit architectural conventions rather than relying on model names.
>
> **Prerequisites.** Modules 1, 4, and 5 provide the useful background: total versus active parameters, attention geometry, and the parameter cost of a SwiGLU feed-forward network. All notation needed for the calculations is introduced again here.

---

## 1. The bargain: capacity without activating the whole bank

A dense feed-forward network applies every one of its matrices to every token. If we enlarge that network, stored capacity and token-level arithmetic rise together. MoE changes the unit of conditional execution. A layer holds $E$ complete feed-forward experts, while a small router chooses only $k\ll E$ of them for the current hidden state.

This creates two related but non-interchangeable quantities:

- **total parameters** describe the full bank that the serving system must make available;
- **active parameters** describe the parameters charged under a stated per-token counting convention.

Neither quantity is a direct measurement of intelligence or latency. Total size does not uniquely determine knowledge, and active size omits communication, attention, the output head, memory traffic, and hardware utilization unless the convention explicitly includes them. It is nevertheless a useful first decomposition.

The range is already wide among documented open models. gpt-oss-120b publishes 117B total and 5.1B active parameters with 128 experts and top-4 routing. DeepSeek-V3 reports 671B/37B. Kimi K3 reaches 2.8T/104B while selecting 16 of 896 routed experts. Those headlines come from different architectures and sometimes different counting rules. A responsible comparison therefore asks what is common, what is routed, and which vocabulary matrices or auxiliary modules are included.

The compute bargain also creates a systems bill. The expert bank must be stored somewhere, route decisions must become efficient batches, and devices must exchange token representations. For MoE, the model architecture and the distributed runtime are inseparable parts of the design.

## 2. From local specialists to trillion-parameter sparse models

![VIZ m6/01 — the history of Mixture of Experts](assets/modern-llms/en/module-06/m6_01_moe_timeline.svg)

The statistical ancestor is the adaptive mixture of local experts introduced by Jacobs and Jordan. A gating network assigned examples to local predictors, allowing them to specialize. Modern MoE keeps that division of labor but changes both the scale and the computational shape.

Shazeer et al. brought sparsely gated experts to very large neural networks in 2017. Top-k routing made the parameter count enormous without evaluating every expert, but it also made expert collapse visible: a slightly favored expert receives more data, improves faster, and attracts still more tokens.

GShard and Switch Transformer turned sparse experts into a distributed Transformer component. Expert parallelism, capacity factors, all-to-all traffic, and token dropping became first-class concerns. Switch simplified routing to top-1; ST-MoE introduced router z-loss as one of its stabilization mechanisms.

Mixtral made top-2 MoE broadly inspectable in an open model. DeepSeekMoE then proposed finer expert segmentation and always-on shared experts. DeepSeek-V3 combined a large fine-grained bank with sigmoid scores, group preselection, and balancing biases updated outside the language-model objective.

Recent work branches rather than converging on one final router. LatentMoE compresses the routed representation and expert width. ReMoE and later differentiable methods modify top-k itself. Routing-free proposals remove a separate central router. The enduring idea is conditional expert compute; the exact routing and runtime contracts remain open design choices.

## 3. Three older ideas hiding inside modern MoE

### A mixture model

The familiar form is

$$
y(x)=\sum_{i=1}^{E}g_i(x)f_i(x).
$$

Sparse MoE makes most $g_i(x)$ exactly zero through top-k selection. The experts are large dense FFNs, so selected work still maps well to matrix multiplication hardware.

### Conditional computation

MoE is structured conditional execution. It differs from arbitrary weight sparsity because the active units are large blocks. A chosen expert can be evaluated by dense GEMM; the challenge is assembling enough tokens for each expert to make that GEMM efficient.

### A distributed dispatcher

The router is also a scheduler. If one expert receives twice the load of its peers, the step may be limited by the device hosting that expert. A good probability model is therefore not enough: capacity, placement, networking, and queue behavior all matter.

These views explain why the module needs both equations and systems arithmetic. The mixture formula explains the output; conditional computation explains total versus active parameters; the dispatcher view explains load balancing and expert parallelism.

## 4. A typed contract for routing and parameter accounting

Let $x\in\mathbb{R}^h$ be a token representation. One sigmoid router uses

$$
s_i=\sigma(w_i^\top x).
$$

With an expert-wise selection bias $b_i$,

$$
\mathcal T(x)=\operatorname{TopK}(s_i+b_i,k).
$$

A model may then normalize gates from the clean scores rather than the biased selection values:

$$
g_i=\rho\,\frac{s_i}{\sum_{j\in\mathcal T(x)}s_j},
\qquad i\in\mathcal T(x).
$$

The layer output is

$$
\operatorname{MoE}(x)
=
\sum_{i\in\mathcal T(x)}g_i\operatorname{FFN}_i(x)
+
\sum_{j=1}^{S}\operatorname{FFN}^{\text{shared}}_j(x).
$$

![VIZ m6/02 — anatomy of an MoE layer](assets/modern-llms/en/module-06/m6_02_moe_block.svg)

A SwiGLU expert with hidden width $h$ and expert intermediate width $i_{\text{exp}}$ contains

$$
3h\,i_{\text{exp}}
$$

matrix parameters. If all three projections have biases, the complete count is

$$
3h\,i_{\text{exp}}+2i_{\text{exp}}+h.
$$

No single total/active formula covers every model. A real count must state:

- how many layers are dense versus MoE;
- the number and width of routed and shared experts;
- whether embeddings are tied;
- which projections have biases;
- whether MTP or auxiliary modules are present;
- which full vocabulary matrices enter the active convention;
- the attention architecture and any extra normalization parameters.

For that reason, the numerical code uses a typed `MoEModelSpec`. It makes assumptions data rather than hiding them in algebra. In the gpt-oss examples, active parameters include one full vocabulary projection—the output head—but do not charge an indexed input embedding lookup as if every row had been multiplied.

## 5. Rebuilding both gpt-oss models from open code

The open implementation exposes details that a short configuration summary does not: QKV and output biases, router bias, expert biases, learned attention sinks, untied embeddings, and packed expert matrices.

Shared geometry for the two models:

- vocabulary size 201,088;
- hidden and expert intermediate width 2,880;
- 64 query heads, 8 KV heads, head width 64;
- two vocabulary matrices;
- biased attention projections, router, and experts.

### gpt-oss-120b

With 36 layers, 128 experts, and top-4 routing, the complete reconstruction is:

| Component | Parameters |
|---|---:|
| two vocabulary matrices | 1,158,266,880 |
| attention, including biases | 955,802,880 |
| all expert weights and biases | 114,701,598,720 |
| routers | 13,275,648 |
| normalization scales | 210,240 |
| learned sinks | 2,304 |
| **Total** | **116,829,156,672** |

Under the declared active convention:

$$
P_{\text{active}}=5,132,849,472,
$$

and

$$
P_{\text{total}}/P_{\text{active}}=22.761.
$$

The published 117B/5.1B values are appropriate rounded headlines. The course number is a reproducible count with its convention attached.

### gpt-oss-20b

For 24 layers, 32 experts, and the same top-4:

$$
P_{\text{total}}=20,914,757,184,
$$

$$
P_{\text{active}}=3,608,307,264,
$$

$$
P_{\text{total}}/P_{\text{active}}=5.796.
$$

Top-4 selects a much larger fraction of a 32-expert bank than of a 128-expert bank. The two models therefore have different sparsity factors despite sharing the same $k$.

![VIZ m6/03 — total and active parameters](assets/modern-llms/en/module-06/m6_03_total_active.png)

The rough estimate $2P_{\text{active}}$ covers part of the weight-matmul arithmetic. It does not turn the 22.76 ratio into a latency multiplier: weight placement, expert batching, attention, and communication remain.

## 6. A numerical router trace

Take one token with router logits

```text
[1.2, -0.3, 2.1, 0.4, 1.7, -1.0, 0.9, 2.5]
```

The sigmoid scores are

```text
[0.7685, 0.4256, 0.8909, 0.5987,
 0.8455, 0.2689, 0.7109, 0.9241].
```

Top-4 selects experts

```text
[7, 2, 4, 0]
```

with normalized gates

```text
[0.2695, 0.2598, 0.2466, 0.2241].
```

Now give expert 3 a selection bias of $+0.35$. The selected set changes to

```text
[3, 7, 2, 4].
```

The gate for expert 3 is still based on its clean score 0.5987, not on 0.9487. The gates become

```text
[0.1837, 0.2835, 0.2733, 0.2594].
```

![VIZ m6/04 — router trace](assets/modern-llms/en/module-06/m6_04_router_trace.svg)

DeepSeek-V3 adds a group stage. Its 256 routed experts are divided into eight groups; a token first retains four groups and then chooses eight experts from those groups. With balancing bias enabled, the group score is built from the two strongest adjusted scores in each group. Gates are then formed from clean scores and multiplied by the model's routed scaling factor.

This separation is broadly useful: a system can have one quantity for *who is eligible* and another for *how much the selected output contributes*.

### NumPy → PyTorch · B09 — Top-k MoE: expert selection and mixture weights

A selection bias may change expert choice while mixture weights still come from clean scores. NumPy makes the boundary explicit; PyTorch uses `topk` and `gather`.

```python
import numpy as np
import torch
```

#### 1. Explicit NumPy implementation

```python
def numpy_topk_router_explicit(
    logits: np.ndarray,
    top_k: int,
    *,
    selection_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    clean = 1.0 / (1.0 + np.exp(-logits))
    selection = clean if selection_bias is None else clean + selection_bias
    indices = np.argsort(-selection, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(clean, indices, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return indices, weights
```

The minimal NumPy code separates clean scores, selection scores after bias, and mixture weights gathered from the clean scores of selected experts.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_topk_router_explicit(
    logits: torch.Tensor,
    top_k: int,
    *,
    selection_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean = logits.sigmoid()
    selection = clean if selection_bias is None else clean + selection_bias
    _, indices = torch.topk(selection, top_k, dim=-1)
    weights = clean.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights
```

PyTorch expresses the same route with `sigmoid`, `topk`, and `gather`. Tie ordering from `topk` is not a portable guarantee.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Clean scores | NumPy sigmoid | `Tensor.sigmoid` |
| Selection | stable `argsort` | `torch.topk` |
| Selected weight | `take_along_axis` | `gather` |
| Bias | selection only | same contract |
| Ties | stable reference sort | ordering is not guaranteed |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
logits_np = np.array([[2.1, -0.4, 0.8, 1.3], [0.2, 1.9, -1.1, 0.7]])
bias_np = np.array([0.0, 0.08, -0.03, 0.02])
i_np, w_np = numpy_topk_router_explicit(logits_np, 2, selection_bias=bias_np)
i_t, w_t = torch_topk_router_explicit(torch.tensor(logits_np), 2, selection_bias=torch.tensor(bias_np))
np.testing.assert_array_equal(i_np, i_t.numpy())
np.testing.assert_allclose(w_np, w_t.numpy())
print("B09 explicit top-k parity away from ties: PASS")
```

</details>

Complete executable file: [`m06_moe_bridges.py`](../assets/m06_moe_bridges.py)

Tie-breaking contract: [PyTorch `topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).

## 7. Load balancing without conflating three mechanisms

A router can collapse through positive feedback. An early favorite receives more examples, learns faster, and becomes an even stronger favorite. Different mechanisms address different parts of that loop.

### Auxiliary load-balancing loss

A Switch-style term is

$$
L_{\text{aux}}=E\sum_i f_iP_i,
$$

where $f_i$ is assignment frequency and $P_i$ is mean router probability. It contributes a direct gradient that trades some task objective for a more even distribution.

### Router z-loss

ST-MoE uses

$$
L_z=\mathbb{E}\!\left[\log\sum_i e^{z_i}\right]^2
$$

to restrain router logit magnitude. It is a numerical stabilization term, not a complete load-balancing strategy.

### Auxiliary-loss-free balancing

An expert-wise bias is added before top-k and updated from recent loads outside backpropagation. The main rule in the paper is

$$
b_i\leftarrow b_i+u\,\operatorname{sign}(\bar c-c_i).
$$

This avoids a **direct interference gradient** from an auxiliary objective. It does not leave training unchanged: altered routes determine which experts receive language-model gradients.

![VIZ m6/05 — load-balancing traces](assets/modern-llms/en/module-06/m6_05_balance.png)

The frozen example starts at load CV 0.6506. After 400 updates:

- the sign rule with step 0.001 reaches 0.0393;
- the proportional teaching controller with rate 0.15 reaches 0.0193.

The smoother curve is not presented as the exact DeepSeek algorithm. It is an interpretable control-law experiment; the paper also analyzes a proportional variant in a separate setting.

Capacity control is distinct again. The course uses

$$
C_e=\left\lceil c\frac{Tk}{E}\right\rceil
$$

and an explicit FIFO overflow policy. Surviving gates are renormalized by course convention. Production implementations may reroute, allocate larger buffers, drop without renormalization, or use dropless grouped GEMM.

A token that loses every routed assignment does not vanish. Its routed branch is zero in that layer, while the residual path and other sublayers continue.

## 8. Fine-grained experts and always-on shared capacity

Suppose an architecture replaces a small number of wide experts with a larger number of narrower ones while increasing $k$ proportionally. The active expert arithmetic can remain of similar order, while the router gains more ways to combine sub-blocks.

The combinatorial contrast is striking:

$$
\binom{8}{2}=28,
\qquad
\binom{128}{4}=10,668,000.
$$

These are counts of possible unordered subsets, not counts of learned skills. Continuous gate weights, correlated routing, and group constraints matter. For DeepSeek-V3, $\binom{256}{8}$ is only an unconstrained upper bound because group preselection rules out many subsets.

A shared expert follows a different path: it runs for every token. DeepSeekMoE motivates shared capacity as a place for common information and a way to reduce redundancy across routed experts. That motivation does not identify the content of one trained shared expert. Claims such as “this expert stores grammar” need intervention evidence, not just architecture diagrams.

![VIZ m6/06 — fine-grained and shared experts](assets/modern-llms/en/module-06/m6_06_fine_grained.svg)

### DeepSeek-V3 reconstruction

Under the course's explicit convention:

$$
P_{\text{total}}=671,026,419,200,
$$

$$
P_{\text{active}}=36,625,618,432.
$$

These are consistent with the published 671B/37B headline. The Hugging Face package is about 685B when the 14B MTP module is included. A model passport should keep the main model, auxiliary modules, and active convention separate.

![VIZ m6/07 — DeepSeek-V3 budget](assets/modern-llms/en/module-06/m6_07_v3_budget.svg)

## 9. What expert specialization evidence can actually support

Routing statistics are easy to collect; semantic conclusions are harder.

Reliable observables include:

- expert frequencies by layer, language, or domain;
- co-activation matrices;
- changes after ablating or replacing an expert;
- router sensitivity to syntactic and semantic features;
- stability of a routing pattern across corpora.

A useful interpretation ladder is:

1. **routing preference:** which inputs select an expert more often;
2. **functional effect:** what changes under intervention;
3. **semantic label:** a human summary that may be incomplete.

Experts often align with narrower operations or linguistic patterns rather than clean topics such as “mathematics” or “Chinese.” Several experts may be partly substitutable. The role of a shared expert can change across depth.

The same caution applies to adaptation. Failure to find a pre-existing language-specific expert does not prove that fine-tuning will smear the language across the bank or damage old languages. Those are testable outcomes, not architectural necessities.

Serving systems care about a different pattern: hot experts under the actual product workload. Training-time balance does not guarantee balanced traffic after deployment. Per-expert utilization therefore belongs in production telemetry.

## 10. Inference: the expert bank, all-to-all, and batch union

The full bank must be available to the system, but it need not reside on one GPU or be replicated everywhere. Options include expert parallelism, CPU or NVMe tiers, weight streaming, hot-expert replication, and sharded caches.

### Expert parallelism

With EP, devices own subsets of experts. Tokens are dispatched to their owners, processed by grouped GEMMs, and returned. Ignoring metadata and padding, a round trip moves

$$
M_{\text{comm}}=2Thkb
$$

bytes. For $T=2048$, $h=7168$, $k=8$, and bf16, the idealized payload is 448 MiB.

### How many experts does a batch touch?

Under independent uniform routing,

$$
f_{\text{touched}}
=1-\left(1-\frac{k}{E}\right)^N.
$$

For 128/top-4 and 256/top-8, $k/E$ is the same, so the theoretical curve is the same: at 64 tokens, about 86.9% of experts are touched.

For Kimi K3's 896/top-16 routed bank:

- 16 tokens touch about 25.0% in expectation;
- 64 tokens touch about 68.4%.

![VIZ m6/08 — the systems side of MoE](assets/modern-llms/en/module-06/m6_08_expert_parallel.svg)

Actual routes are correlated and skewed, shared experts are always active, and grouping changes the choice set. The formula is a baseline against which telemetry can be compared.

Expert offloading is not automatically useless at realistic batch sizes. Its value depends on popularity skew, temporal locality, prefetching, and transport latency. What fails is the simplistic assumption that only the $k$ experts for one isolated token ever need to be resident.

## 11. LatentMoE and alternatives to fixed discrete top-k

LatentMoE keeps a discrete expert bank but reduces the width of the routed path. A hidden state of width $d$ is projected into a latent vector of width $\ell$:

$$
z=P_{\text{down}}x,
\qquad \ell<d.
$$

Dispatch, expert FFNs, and combination operate in that latent space. The result is projected back:

$$
y=P_{\text{up}}\operatorname{MoE}_{\ell}(z).
$$

The router can still consume the original hidden representation and choose top-k experts. LatentMoE is therefore not a continuous virtual expert assembled from basis matrices.

Kimi K3 uses hidden width 7168 and latent expert width 3584, giving a twofold idealized reduction in routed activation width. It retains 896 routed experts, top-16, and two shared experts. NVIDIA uses LatentMoE in Nemotron 3 Super and Ultra.

Several other branches change routing more directly:

| Method | Routing mechanism | Adaptive active count | Original evaluation domain |
|---|---|---:|---|
| Soft MoE (2023) | soft token-to-slot and slot-to-token mixtures | fixed slot layout | vision Transformers |
| ReMoE | differentiable ReLU routing | effective dynamic sparsity | language models |
| SoftMoE (2026) | differentiable truncated soft top-k | yes, under a global budget | LLMs |
| Routing-Free MoE | removes a separate central router | method-dependent | research models |

These methods do not form one simple succession. They trade off differentiability, predictable kernel shapes, load balance, and adaptive compute.

## 12. Level-one code: policies should be visible

The educational implementation is intentionally small enough to inspect. It makes four choices explicit:

- selection scores and clean gate scores are separate;
- ties have a deterministic lower-index rule;
- capacity overflow is not silent;
- parameter counting is driven by a typed architecture specification.

```python
import numpy as np

def route_logits(logits, top_k, selection_bias=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite [T,E]")
    _, experts = logits.shape
    if not 1 <= top_k <= experts:
        raise ValueError("top_k must satisfy 1 <= k <= E")

    scores = 1.0 / (1.0 + np.exp(-logits))
    bias = np.zeros(experts) if selection_bias is None else np.asarray(selection_bias)
    selection = scores + bias[None, :]

    expert_ids = np.arange(experts)
    chosen = np.empty((len(logits), top_k), dtype=np.int64)
    for token in range(len(logits)):
        chosen[token] = np.lexsort((expert_ids, -selection[token]))[:top_k]

    selected_scores = np.take_along_axis(scores, chosen, axis=1)
    gates = selected_scores / selected_scores.sum(axis=1, keepdims=True)
    return chosen, gates

def loss_free_bias_step(bias, counts, rate):
    """Sign update used by the main auxiliary-loss-free rule."""
    counts = np.asarray(counts, dtype=np.float64)
    fractions = counts / counts.sum()
    return np.asarray(bias) + rate * np.sign(fractions.mean() - fractions)
```

The full code adds stable extreme-logit handling, capacity dispatch, shared experts, auxiliary losses, the accounting classes, and explicit negative tests. It is not a grouped-GEMM or distributed runtime.

### NumPy → PyTorch · B10 — MoE dispatch and combine

NumPy executes a transparent assignment loop. PyTorch batches tokens by expert and combines weighted outputs with `index_add_`.

```python
import numpy as np
import torch
```

#### 1. Explicit NumPy implementation

```python
def numpy_expert_dispatch_explicit(
    hidden: np.ndarray,
    expert_indices: np.ndarray,
    gates: np.ndarray,
    w_in: np.ndarray,
    w_out: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(expert_indices.shape[1]):
            expert = int(expert_indices[token, slot])
            mid = np.tanh(hidden[token] @ w_in[expert])
            output[token] += gates[token, slot] * (mid @ w_out[expert])
    return output
```

NumPy loops over tokens and routing slots, selects expert weights, evaluates the expert FFN, and adds the gate-weighted result to the token output.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_expert_dispatch_explicit(
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    gates: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden)
    for expert in range(w_in.shape[0]):
        token_ids, slots = torch.where(expert_indices == expert)
        if token_ids.numel() == 0:
            continue
        mid = torch.tanh(hidden[token_ids] @ w_in[expert])
        expert_out = (mid @ w_out[expert]) * gates[token_ids, slots, None]
        output.index_add_(0, token_ids, expert_out)
    return output
```

PyTorch changes the execution order: assignments are batched per expert, then results are returned to tokens with `index_add_`.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Order | token → slot | expert → assigned tokens |
| Expert batch | none | indexed `hidden[token_ids]` |
| Combine | `output[token] += ...` | `output.index_add_` |
| Autograd | none | through expert matmul and combine |
| CUDA determinism | CPU reference | `index_add_` may depend on atomic-add ordering |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
rng = np.random.default_rng(10)
x_np = rng.standard_normal((6, 4))
idx_np = np.array([[0,1],[1,2],[2,0],[0,2],[1,0],[2,1]])
gates_np = np.full((6,2), 0.5)
w1_np = rng.standard_normal((3,4,5)); w2_np = rng.standard_normal((3,5,4))
np_out = numpy_expert_dispatch_explicit(x_np, idx_np, gates_np, w1_np, w2_np)
t_out = torch_expert_dispatch_explicit(torch.tensor(x_np), torch.tensor(idx_np), torch.tensor(gates_np), torch.tensor(w1_np), torch.tensor(w2_np))
np.testing.assert_allclose(np_out, t_out.numpy(), rtol=1e-10, atol=1e-10)
print("B10 token/slot loop vs expert batches: PASS")
```

</details>

Complete executable file: [`m06_moe_bridges.py`](../assets/m06_moe_bridges.py)

Accumulation contract: [PyTorch `index_add_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html).

## 14. Reading the gpt-oss implementation

The open reference code makes its routing contract unusually clear.

`GptOssTopKRouter` has a weight matrix and a bias, takes top-k router logits, and applies softmax over the selected logits. That is not the sigmoid-plus-selection-bias contract used by DeepSeek-V3.

The expert implementation stores packed gate/up weights, corresponding biases, down weights, and down biases. Ignoring those tensors was the reason an earlier matrix-only count understated the 120b model by 40,105,728 parameters.

The readable reference path loops over experts touched by the batch and accumulates outputs with indexed addition. A high-performance serving implementation replaces that loop with fused dispatch and grouped GEMMs, frequently across an expert-parallel group.

![VIZ m6/09 — gpt-oss from source](assets/modern-llms/en/module-06/m6_09_config_moe.svg)

The general reading lesson is portable: inspect the score function, selection bias, gate normalization, shared capacity, expert tensor shapes, overflow policy, and distributed execution path. “MoE” alone specifies none of them.

## 15. A source-aware snapshot of current MoE architectures

The table is dated 6 August 2026 and records evidence rather than ranking models.

| Model | Total / active | Expert contract | Routing or related design | Evidence |
|---|---:|---|---|---|
| gpt-oss-120b | 117B / 5.1B | 128 routed, top-4, no shared expert | softmax over selected logits | official announcement, config, and code |
| DeepSeek-V3 | 671B / 37B | 256 routed + 1 shared, top-8 | sigmoid, group preselection, loss-free bias | technical report and inference code |
| Qwen3.5-35B-A3B | 35B / 3B | 256 routed + 1 shared, top-8 | hybrid DeltaNet/attention, MTP | official model card and config |
| Kimi K3 | 2.8T / 104B | 896 routed + 2 shared, top-16 | Stable LatentMoE, latent width 3584 | official report and weights |
| Nemotron 3 Ultra | 550B / 55B | LatentMoE | hybrid Mamba–Attention, MTP, NVFP4 | official technical material |
| GLM-5 | 744B / 40B | 256 routed + 1 shared, top-8 | sigmoid/no-aux routing; DSA in attention | official model card and config |
| DeepSeek-V4 Pro / Flash | 1.6T / 49B; 284B / 13B | report-specific MoE | hybrid attention and MoE | technical report |

The active ratios are not directly comparable performance predictors. Models differ in dense prefixes, attention, MTP, modality towers, output heads, and counting conventions.

Qwen3.5-35B-A3B is a useful configuration-reading example. Its language model has hidden width 2048, 40 layers, 256 routed experts, top-8 plus one shared expert, expert intermediate width 512, and an MTP layer. Its block pattern interleaves Gated DeltaNet and full attention. These published fields are a better basis for an exercise than a deliberately wrong 48-layer configuration.

## 16. Multimodal MoE is not one architecture

Multimodality can meet sparse experts in several ways.

A unified residual stream may send text and visual tokens through one routed bank. That enables mixed functional specialization but can widen the expert union touched by a batch. Kimi K3 is a current documented example of a native multimodal MoE system.

Another architecture can reserve modality-specific experts, trading simpler specialization for more complex balance and placement. A third can keep a separate vision encoder and route only the projected visual tokens through the language backbone.

If every token still chooses $k$ experts, expert compute per token does not rise merely because the token is visual. What can rise is the number of distinct experts touched across the mixed batch and therefore communication diversity. A multimodal MoE must be evaluated with routing and all-to-all telemetry, not only a total/active headline.

## 19. Key takeaways

![VIZ m6/10 — Mixture of Experts cheat sheet](assets/modern-llms/en/module-06/m6_10_cheatsheet.svg)

- MoE grows stored capacity faster than per-token expert compute, but storage and communication remain.
- Total and active counts are meaningful only with a declared convention.
- A selection bias can alter top-k without becoming an expert mixture weight.
- Auxiliary-loss-free balancing removes a direct auxiliary gradient, not the effect of routing on training.
- Fine-grained banks create more routing flexibility; combinatorial subsets are not semantic skills.
- Shared experts reduce pressure to duplicate common computation, but their contents must be measured.
- Batch union can touch a large part of the bank; the uniform formula is a baseline.
- LatentMoE keeps discrete top-k while compressing the routed path.
- Current model tables should cite configs, reports, or code rather than inherit properties across families.

## 20. Notebook and primary sources

**Primary sources:**

- Jacobs et al., *Adaptive Mixtures of Local Experts* — [paper](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- Shazeer et al., *Outrageously Large Neural Networks* — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- GShard — [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Switch Transformer — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- ST-MoE — [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
- DeepSeekMoE — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Auxiliary-Loss-Free Load Balancing — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- DeepSeek-V3 — [technical report](https://arxiv.org/abs/2412.19437) and [official repository](https://github.com/deepseek-ai/DeepSeek-V3)
- gpt-oss — [official announcement](https://openai.com/index/introducing-gpt-oss/) and [open implementation](https://github.com/openai/gpt-oss)
- LatentMoE — [NVIDIA Research](https://research.nvidia.com/labs/nemotron/LatentMoE/)
- Nemotron 3 Ultra — [official page](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
- Kimi K3 — [official repository and report](https://github.com/MoonshotAI/Kimi-K3)
- Qwen3.5-35B-A3B — [official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- GLM-5 — [official model card and config](https://huggingface.co/zai-org/GLM-5)
- Soft MoE — [arXiv:2308.00951](https://arxiv.org/abs/2308.00951)
- ReMoE — [arXiv:2412.14711](https://arxiv.org/abs/2412.14711)
- SoftMoE 2026 — [arXiv:2606.17952](https://arxiv.org/abs/2606.17952)

**Next:** Module 7 changes the other large component of the block. Instead of routing among feed-forward experts, it asks whether sequence information must always move through attention, or whether recurrent state and state-space models can carry part of the load.

---

*Landscape verified: 6 August 2026. Course reconstructions are separated from rounded developer headlines, and reported research results retain their original experimental scope.*
