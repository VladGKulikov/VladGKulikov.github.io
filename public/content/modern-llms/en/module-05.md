# Module 5. Normalization, activations, and the anatomy of a Transformer block

*Modern LLMs · Module 5 lecture · edition 2026.8*

> **What this module does.** Attention moves information across positions, but a Transformer block is held together by a different set of mechanisms: the residual stream, normalization placement, and the feed-forward network. We will treat them as one system. The route starts with the optimization geometry of Post-LN and Pre-LN, moves through RMSNorm and query/key stabilization, derives the parameter cost of gated FFNs, and reconstructs every parameter of Llama-3.1-8B. The last part separates two research directions that are often conflated: element-wise replacements for normalization, such as DyT, and architectural changes to residual mixing, such as mHC.
>
> **Prerequisites.** Modules 1–4, especially projection shapes and Grouped-Query Attention. All normalization and activation definitions used below are introduced locally.

---

## 1. Why the quiet fields in a configuration matter

A model configuration contains fields that rarely appear in benchmark tables: `rms_norm_eps`, `intermediate_size`, or `swiglu_limit`. They are easy to ignore because none of them sounds like a headline architectural idea. Yet each field changes a different part of the block contract.

Normalization controls the scale at which sublayers read the residual stream. The intermediate width determines most of the parameter mass of a dense block. An activation clamp changes the function the checkpoint was trained to compute. These choices usually fail quietly: rather than producing an immediate exception, a wrong implementation changes activation statistics, gradients, or the model's exact logits.

Three numerical examples will keep the discussion grounded. A shared-weight linear stack shows how norm placement changes forward-scale dynamics. A parameter ledger separates the $13d^2$ matrix estimate from the complete $13d^2+2d$ Llama block. Finally, the public gpt-oss activation gives us a real example of a trained, clamped SwiGLU variant.

## 2. The historical problem was not “which norm wins?”

![VIZ m5/01 — evolution of the block](assets/modern-llms/en/module-05/m5_01_block_bus.svg)

Layer Normalization was introduced for sequence models in 2016 and became part of the original Transformer as Post-LN:

$$x_{l+1}=\operatorname{LN}(x_l+F_l(x_l)).$$

Pre-LN later moved normalization inside the residual branch:

$$x_{l+1}=x_l+F_l(\operatorname{LN}(x_l)).$$

The difference is not cosmetic. Xiong et al. showed that, at initialization, Post-LN can produce large expected gradients near the output layers; warmup mitigates the instability caused by applying a large learning rate there. Their Pre-LN analysis gives better-behaved initial gradients and supported training without warmup in the reported experiments. It does not imply that every Post-LN run without warmup must diverge.

RMSNorm simplified the normalizer by removing recentering. Its original experiments reported comparable quality and 7–64% runtime reductions across the tested systems. Modern fused implementations reduce the end-to-end advantage, but RMSNorm remains attractive because its contract is simple and robust.

A separate line changed the FFN. GLU variants turned the intermediate transformation into a product of two learned branches; SwiGLU became a common choice in large decoders.

The later normalization literature branches again. DeepNorm modifies residual scaling and initialization for very deep Post-LN models. Peri-LN normalizes both the sublayer input and its output contribution. OLMo2 uses post-sublayer normalization inside the residual branch, without a matching pre-normalizer. DyT removes the feature reduction entirely, while mHC changes the topology of residual mixing. These methods solve related but non-identical problems.

## 3. Three useful classical analogies

LayerNorm resembles per-example standardization, but it is not whitening: it neither estimates a full covariance matrix nor rotates the feature basis. RMSNorm retains only scale control. The preconditioning analogy is therefore local and practical—normalizers keep the numerical range seen by later matrices more predictable.

Residual updates resemble Euler steps:

$$h_{l+1}=h_l+F_l(h_l).$$

This viewpoint is useful for asking how large an update is and how errors accumulate with depth. It is not an exact continuous-time model because the functions change from layer to layer and the step size need not be small.

Gated FFNs illustrate a third idea: multiplicative interaction. Two separately learned projections are multiplied element-wise, so the input controls both what is represented and how strongly it passes. This is more expressive than applying a fixed pointwise nonlinearity to a single projection, although it costs an extra matrix.

## 4. The residual stream and what the identity path actually guarantees

A Pre-LN decoder block can be written as

$$\tilde h=h+\operatorname{Attn}(N_1(h)),$$

$$h^+=\tilde h+\operatorname{FFN}(N_2(\tilde h)).$$

The residual stream is the shared state vector of width $d_{\text{model}}$. Attention and the FFN read normalized versions of that state and contribute updates through their output projections.

For one residual step,

$$\frac{\partial h^+}{\partial h}=I+\frac{\partial F}{\partial h}.$$

The identity term is valuable: it provides a direct path for activations and gradients. It does not prove that an arbitrary trained layer is expendable. A block may learn an essential transformation, so pruning or early exit still requires measurement or special training.

This distinction matters because architecture and learned function are different claims. The graph contains a bypass; the trained model decides how much it relies on the branch.

## 5. LayerNorm and RMSNorm side by side

LayerNorm computes

$$\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,$$

whereas RMSNorm uses

$$\operatorname{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\sqrt{d^{-1}\sum_jx_j^2+\epsilon}}.$$

For $x=(2,-1,3,0)$ with unit scales, LayerNorm produces approximately

$$(0.632,-1.265,1.265,-0.632),$$

and RMSNorm produces

$$(1.069,-0.535,1.604,0).$$

The latter preserves signs and coordinate ratios before the learned scale, but it does not enforce zero mean. RMSNorm's success is an empirical statement about the tested architectures, not a proof that recentering is always redundant.

### Epsilon as a gain limit

If the RMS of the input approaches zero, the largest possible normalization multiplier is $1/\sqrt\epsilon$. At input RMS $10^{-3}$, the multipliers are about 707 for $\epsilon=10^{-6}$ and 302 for $10^{-5}$.

Choosing epsilon is not a one-line consequence of machine epsilon. It depends on reduction precision, activation scale, the kernel, and how much amplification is acceptable on near-degenerate states. Many reference implementations accumulate RMS statistics in fp32, including the public gpt-oss PyTorch code, but fused kernels may use other accumulation strategies.

### NumPy → PyTorch · B07 — RMSNorm: reduction, dtype, and autograd

NumPy exposes the mean-square reduction. PyTorch adds autograd, device/dtype semantics, and the possibility of a fused kernel.

```python
import numpy as np
import torch
import torch.nn.functional as F
```

#### 1. Explicit NumPy implementation

```python
def numpy_rms_norm(x: np.ndarray, weight: np.ndarray | None = None, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0 or not np.isfinite(x).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("x must be finite and eps positive")
    work = x.astype(np.float64, copy=False)
    y = work / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != (x.shape[-1],):
            raise ValueError("weight must have shape [D]")
        y = y * w
    return y.astype(x.dtype, copy=False) if np.issubdtype(x.dtype, np.floating) else y
```

The NumPy reference exposes the last-dimension mean square, $\epsilon$, reciprocal square root, and the learned per-coordinate scale.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_rms_norm_explicit(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    rms_inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x * rms_inv
    return y if weight is None else y * weight
```

The explicit tensor version follows the equation almost line by line. A mixed-precision model must also decide the reduction dtype.

#### 3. Optimized or library PyTorch API

```python
def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, *, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, [x.shape[-1]], weight=weight, eps=eps)
```

`F.rms_norm` moves the reduction into a library operator that may use an optimized implementation.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Reduction | `np.mean(..., axis=-1)` | `mean(dim=-1)` |
| Reciprocal root | `1 / np.sqrt` | `torch.rsqrt` |
| Weight | broadcast array `[D]` | learned parameter `[D]` |
| Accumulation | float64 reference | depends on explicit/fused path |
| Gradient | none | checked with `gradcheck` |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
rng = np.random.default_rng(7)
x_np = rng.standard_normal((3, 5)); w_np = rng.standard_normal(5)
np_out = numpy_rms_norm(x_np, w_np, eps=1e-6)
x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
explicit = torch_rms_norm_explicit(x, w, eps=1e-6)
api = torch_rms_norm(x, w, eps=1e-6)
np.testing.assert_allclose(np_out, explicit.detach().numpy(), rtol=1e-10, atol=1e-10)
torch.testing.assert_close(api, explicit, rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_rms_norm_explicit(a, b, eps=1e-6), (x, w))
print("B07 explicit NumPy / PyTorch / F.rms_norm: PASS")
```

</details>

Complete executable file: [`m05_block_bridges.py`](../assets/m05_block_bridges.py)

Official API: [PyTorch `rms_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html).

## 6. Four norm placements, four computation graphs

![VIZ m5/02 — normalization placement](assets/modern-llms/en/module-05/m5_02_norm_placements.svg)

Post-LN normalizes after residual addition:

$$x^+=N(x+F(x)).$$

Pre-LN keeps the residual highway untouched:

$$x^+=x+F(N(x)).$$

Peri-LN adds a second normalizer around the sublayer:

$$x^+=x+N_{out}(F(N_{in}(x))).$$

OLMo2 follows a different pattern:

$$x^+=x+N(F(x)).$$

Its documentation explicitly states that normalization is applied after attention and feed-forward sublayers, and that queries and keys receive their own normalization. Calling this Peri-LN would erase a real architectural distinction.

The module's depth experiment uses a deliberately small linear system with shared input and weights across modes. After 36 steps, the no-norm trajectory reaches RMS 412,442, the pre-like trajectory reaches 6.42, and the post-like trajectory is exactly 1 because it is explicitly renormalized after every step.

![VIZ m5/03 — depth and scale](assets/modern-llms/en/module-05/m5_03_depth_variance.png)

The experiment isolates placement. It says nothing by itself about backpropagation, optimizer state, or the learned scale parameters of a real Transformer.

## 7. Query/key stabilization is a family, not one universal recipe

Scaled Dot-Product Attention uses

$$s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.$$

The denominator controls the expected variance under a simple initialization model, but the learned norms of $q$ and $k$ can drift.

The original Query-Key Normalization method L2-normalizes each query and key vector and replaces the fixed scale with a learned parameter. Its authors reported an average gain of 0.928 BLEU on five low-resource translation pairs in their evaluation.

Later systems use several related mechanisms: L2 Q/K normalization, RMSNorm on Q and K, per-head or per-dimension scales, logit soft-capping, and optimizer-level clipping. These are not interchangeable. Llama 4 can enable L2 Q/K normalization on RoPE layers, with the setting depending on the checkpoint. OLMo2, Qwen3, and Gemma 3 normalize Q and K with RMSNorm. Gemma 2 instead exposes attention-logit soft-capping. Architecture tables should name the actual mechanism rather than label every stabilizer “QK-norm.”

## 8. From a two-matrix FFN to gated multiplication

The original FFN has one expanded projection and one contraction:

$$\operatorname{FFN}(x)=W_2\phi(W_1x).$$

GLU variants split the expanded representation into an up branch and a gate branch:

$$\operatorname{GLU}(x)=(W_{up}x)\odot\phi(W_{gate}x),$$

followed by $W_{down}$. Shazeer's experiments found several gated variants to outperform their non-gated counterparts under matched budgets.

The relevant structural change is multiplicative conditioning. The gate is not merely “another activation”; it lets one learned feature modulate another learned feature for each input.

GELU and a scaled sigmoid form are numerically close:

$$\operatorname{GELU}(z)\approx z\sigma(1.702z).$$

Our deterministic grid on $[-6,6]$ gives a maximum absolute error of about 0.0203.

![VIZ m5/04 — activation functions](assets/modern-llms/en/module-05/m5_04_activations.png)

## 9. SwiGLU budgets and the gpt-oss variant

A standard SwiGLU block is

$$W_{down}\left[\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right].$$

Its matrix count is $3d\,d_{ff}$. Matching the parameters of a classic $4d$ FFN gives

$$d_{ff}=\frac83d.$$

That ratio is a budget-equivalence point, not a design law. Models often select a different width for capacity or hardware reasons.

The public gpt-oss implementation changes the activation:

$$g\leftarrow\min(g,7),\qquad u\leftarrow\operatorname{clip}(u,-7,7),$$

$$y=g\sigma(1.702g)(u+1).$$

At $g=u=z$, the outputs for $z=7$ and $z=9$ are both 55.99963. The plateau is part of the learned graph.

![VIZ m5/05 — SwiGLU in gpt-oss](assets/modern-llms/en/module-05/m5_05_swiglu_gate.svg)

![VIZ m5/06 — activation clamp](assets/modern-llms/en/module-05/m5_06_clamp_curve.png)

Clamping is compatible with low-precision execution because it controls outliers, but the code alone does not prove that MXFP4 was the sole motivation. Likewise, the $+1$ shift keeps the value factor near one when $u$ is near zero; that is a mathematical effect, not a documented design history unless the authors state it.

### NumPy → PyTorch · B08 — SwiGLU and a bounded variant

Both implementations keep the gate and up branches explicit. `alpha`, optional clipping, and the up-branch shift are visible parameters.

```python
import numpy as np
import torch
```

#### 1. Explicit NumPy implementation

```python
def numpy_swiglu(
    gate: np.ndarray,
    up: np.ndarray,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> np.ndarray:
    gate, up = np.asarray(gate), np.asarray(up)
    if gate.shape != up.shape or not np.isfinite(gate).all() or not np.isfinite(up).all():
        raise ValueError("gate and up must be finite arrays of the same shape")
    g = gate.astype(np.float64, copy=False)
    u = up.astype(np.float64, copy=False)
    if gate_limit is not None:
        g = np.minimum(g, gate_limit)
    if up_limit is not None:
        u = np.clip(u, -up_limit, up_limit)
    y = g / (1.0 + np.exp(-alpha * g)) * (u + shift_up)
    return y.astype(np.result_type(gate.dtype, up.dtype), copy=False)
```

Both branches remain visible: the gate passes through a SiLU-like nonlinearity and multiplies the up branch. `alpha`, clipping, and the shift are explicit parameters.

#### 2. The same mathematics as explicit PyTorch operations

```python
def torch_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 1.0,
    gate_limit: float | None = None,
    up_limit: float | None = None,
    shift_up: float = 0.0,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    g, u = gate, up
    if gate_limit is not None:
        g = g.clamp(max=gate_limit)
    if up_limit is not None:
        u = u.clamp(min=-up_limit, max=up_limit)
    return g * torch.sigmoid(alpha * g) * (u + shift_up)
```

PyTorch replaces `np.exp` with `torch.sigmoid` and preserves the derivative graph. `clamp` introduces kinks, so gradient parity is tested away from the boundaries.

| Aspect | NumPy | PyTorch |
|---|---|---|
| Nonlinearity | explicit sigmoid | `torch.sigmoid` |
| Gate bound | `np.minimum` | `Tensor.clamp(max=...)` |
| Up bound | `np.clip` | `Tensor.clamp(min=..., max=...)` |
| Standard SwiGLU | no limits, `shift_up=0` | same contract |
| Gradients | none | autograd; clamp kinks are tested separately |

<details>
<summary><strong>Executable parity and edge-case check</strong></summary>

```python
rng = np.random.default_rng(8)
g_np = rng.uniform(-4, 4, (3, 5)); u_np = rng.uniform(-4, 4, (3, 5))
kwargs = dict(alpha=1.702, gate_limit=7.0, up_limit=7.0, shift_up=1.0)
np_out = numpy_swiglu(g_np, u_np, **kwargs)
g = torch.tensor(g_np, dtype=torch.float64, requires_grad=True); u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
t_out = torch_swiglu(g, u, **kwargs)
np.testing.assert_allclose(np_out, t_out.detach().numpy(), rtol=1e-10, atol=1e-10)
assert torch.autograd.gradcheck(lambda a, b: torch_swiglu(a, b, **kwargs), (g, u))
print("B08 forward + gradient: PASS")
```

</details>

Complete executable file: [`m05_block_bridges.py`](../assets/m05_block_bridges.py)

## 10. Reconstructing Llama-3.1-8B exactly

Take $V=128256$, $d=4096$, 32 layers, $d_{ff}=14336$, 32 query heads, 8 KV heads, and head dimension 128. The model is bias-free and uses separate input and output embeddings.

The attention matrices cost

$$P_{attn}=2.5d^2=41{,}943{,}040.$$

The SwiGLU matrices cost

$$P_{ffn}=3d\,d_{ff}=176{,}160{,}768.$$

Two RMSNorm scales add $2d=8192$. Therefore

$$P_{matrix}=13d^2=218{,}103{,}808,$$

but the complete block is

$$P_{block}=13d^2+2d=218{,}112{,}000.$$

With two embedding tables and one final norm:

$$2Vd+32P_{block}+d=8{,}030{,}261{,}248.$$

![VIZ m5/07 — exact parameter ledger](assets/modern-llms/en/module-05/m5_07_param_budget.svg)

The FFN accounts for about 80.8% of the full block. Norm parameters are tiny, but the module promised an exact count, so omitting them would still be wrong.

## 11. DyT and DyISRU: removing the feature reduction

Dynamic Tanh replaces normalization with an element-wise transform:

$$\operatorname{DyT}(x)=\gamma\odot\tanh(\alpha x)+\beta.$$

The original work reports competitive or better results across several vision and language settings. The key structural difference from RMSNorm is independence across coordinates: DyT does not compute a shared denominator.

A later mathematical analysis derives Dynamic Inverse Square Root Unit as an exact element-wise counterpart of RMSNorm after a particular decoupling:

$$\operatorname{DyISRU}(x)=\gamma\odot\frac{x}{\sqrt{1+(\alpha x)^2}}+\beta.$$

This does not make DyISRU identical to ordinary vector RMSNorm. The exactness belongs to the decoupled construction.

The notebook's $\alpha\approx1.43$ is a module-owned scalar fit on a synthetic $N(0,1)$ sample. It is not a constant reported by the DyT paper.

![VIZ m5/08 — DyT and DyISRU](assets/modern-llms/en/module-05/m5_08_dyt_family.png)

A 2026 study found a strong adverse interaction between Dynamic Erf and Muon, while DyT served as a bounded control and did not show the same penalty. That evidence argues for testing normalization and optimizer choices jointly; it does not explain away DyT itself.

## 12. mHC constrains residual mixing, not hidden-state trajectories

Hyper-Connections carry several residual streams and learn matrices that mix them around each sublayer. Unconstrained mixing can weaken the identity-mapping property that makes residual networks easy to optimize.

Manifold-Constrained Hyper-Connections project the residual mixing matrices onto a constrained set. In the published design, Sinkhorn–Knopp iterations produce a doubly stochastic mixing matrix on the Birkhoff polytope. DeepSeek-V4 keeps multiple residual streams through the block and collapses them before the final head.

The manifold therefore describes the allowed residual connection matrices. It is not a claim that hidden states themselves move on a low-dimensional representation manifold.

Peri-LN and mHC act at different levels. Peri-LN preserves the ordinary residual addition and normalizes the branch contribution. mHC changes how several residual streams are combined.

Subsequent papers have proposed alternative constraints and faster Birkhoff projections. They broaden the research line, although they are not equivalent to an independent full-scale replication of DeepSeek-V4.

## 13. Level-1 code: the pieces we can inspect directly

The following NumPy code implements a stable sigmoid, RMSNorm with higher-precision statistics, standard SwiGLU, the public gpt-oss activation, and a small Pre-LN wrapper. Attention itself remains an injected function because Module 4 already covered it.

```python
import numpy as np

def sigmoid(x):
    """Stable sigmoid without exp overflow for large negative inputs."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))

def rms_norm(x, gamma, eps=1e-6):
    """Normalize [..., d_model] on the last axis; accumulate statistics in at least float32."""
    x = np.asarray(x)
    work_dtype = np.float32 if x.dtype.itemsize < 4 else x.dtype
    x_hi = x.astype(work_dtype, copy=False)
    ms = np.mean(x_hi * x_hi, axis=-1, keepdims=True)
    y = x_hi / np.sqrt(ms + eps)
    return (y * np.asarray(gamma, dtype=work_dtype)).astype(x.dtype, copy=False)

def swiglu(x, W_gate, W_up, W_down):
    """Standard SwiGLU: SiLU(gate) ⊙ up, followed by the down projection."""
    gate, up = x @ W_gate, x @ W_up
    return (gate * sigmoid(gate) * up) @ W_down

def swiglu_oss(x, W_gate, W_up, W_down, alpha=1.702, limit=7.0):
    """Public gpt-oss variant: clipping, alpha=1.702, and the (up + 1) shift."""
    gate = np.minimum(x @ W_gate, limit)
    up = np.clip(x @ W_up, -limit, limit)
    return (gate * sigmoid(alpha * gate) * (up + 1.0)) @ W_down

class PreNormBlock:
    """Two residual branches; the attention function is injected explicitly."""
    def __init__(self, norm1, norm2, attention, ffn):
        self.norm1, self.norm2 = norm1, norm2
        self.attention, self.ffn = attention, ffn

    def __call__(self, x, mask):
        attn_update = self.attention(self.norm1(x), mask)
        h = x + attn_update
        ffn_update = self.ffn(self.norm2(h))
        return h + ffn_update
```

The code deliberately does not hide placement behind a vague “norm mode.” Peri-LN, OLMo2-style post-sublayer normalization, and mHC have different graphs and should be represented explicitly when implemented.

## 15. Reading gpt-oss from code rather than from a guessed config field

The public implementation makes three design choices visible.

First, RMSNorm accumulates in fp32 and uses $\epsilon=10^{-5}$. Second, `swiglu` implements the 1.702 coefficient, asymmetric gate/up clamps, and the $+1$ value shift. Third, both attention and MLP use Pre-LN residual updates.

The model configuration exposes sizes, experts, top-k, and the clamp limit. It does not contain a `hidden_act` field. The activation is specified in model code.

```json
{
  "hidden_size": 2880,
  "intermediate_size": 2880,
  "num_experts": 128,
  "experts_per_token": 4,
  "swiglu_limit": 7.0
}
```

![VIZ m5/09 — gpt-oss block passport](assets/modern-llms/en/module-05/m5_09_config_block.svg)

The intermediate width describes one expert. Total and active parameter accounting therefore depends on routing and belongs in Module 6.

## 16. A current block table with an evidence column

| Family | Normalization / placement | FFN | Q/K stabilization | Evidence |
|---|---|---|---|---|
| Llama 3.x | RMSNorm, Pre-LN | SwiGLU | no separate QK-norm field in the open config | official config/code |
| Llama 4 | RMSNorm, Pre-LN-style residual | gated MLP/MoE | L2 Q/K normalization on RoPE layers when `use_qk_norm=True`; checkpoint-dependent | official config/code |
| Gemma 2 | RMSNorm, sandwich-style sublayer norms | GELU-tanh gated MLP | attention-logit soft-capping | official config/code |
| Gemma 3 | RMSNorm before and after each sublayer | GELU-tanh gated MLP | RMSNorm on Q and K; standard config disables soft-capping | official config/code |
| OLMo2 | post-sublayer RMSNorm inside residual branches | SiLU-gated MLP | RMSNorm on Q and K | official docs/code |
| gpt-oss | RMSNorm, Pre-LN | clamped SwiGLU + $(u+1)$ | learned sinks rather than QK norm | OpenAI reference code |
| DeepSeek-V4 | normalization inside sublayers plus mHC residual mixing | clamped SwiGLU experts | attention-type dependent | technical report/open implementation |
| Qwen3 | RMSNorm, Pre-LN | SiLU-gated MLP | RMSNorm on Q and K over the head dimension | official config/code |

The evidence column prevents several common mistakes: transferring checkpoint-specific Llama 4 properties to Llama 3, calling OLMo2 Peri-LN, assuming all Gemma generations share the same soft-cap, or placing an optimizer safeguard in the normalization column.

For a closed model, an undisclosed field should remain undisclosed. Family resemblance is not a source.

## 17. The same accounting applies to vision blocks

A ViT encoder still consists of normalization, attention, an MLP, and residual connections. Once its hidden size, head count, MLP width, and bias policy are known, the same parameter ledger applies.

The history of Q/K normalization is worth stating carefully. QKNorm was published in 2020 for low-resource machine translation. ViT-22B later provided a prominent large-scale vision example. It is therefore inaccurate to say that the mechanism originated in vision and then migrated to language models.

Multimodal model configurations often contain separate text and vision sub-configs. They should be audited independently; the two towers may use different activations, epsilon values, widths, and norm placement.

## 20. Key takeaways

![VIZ m5/10 — block anatomy](assets/modern-llms/en/module-05/m5_10_cheatsheet.svg)

- A residual bypass helps optimization but does not make every learned branch disposable.
- RMSNorm controls scale without recentering; epsilon defines near-zero behavior.
- Post-LN, Pre-LN, Peri-LN, and OLMo2-style normalization are different graphs.
- Query/key stabilization is a family of mechanisms, not one universal QKNorm recipe.
- Gated FFNs use three matrices and input-dependent multiplicative interaction.
- The Llama-3.1-8B matrix block is $13d^2$; the complete block is $13d^2+2d$.
- gpt-oss activation behavior lives in model code, not in a `hidden_act` config field.
- DyT and DyISRU are element-wise alternatives; the notebook's $\alpha\approx1.43$ is a teaching fit.
- mHC constrains residual mixing matrices across multiple streams.

## 21. Notebook and primary sources

**Primary sources:**

- Ba, Kiros, Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Wang et al., [DeepNet](https://arxiv.org/abs/2203.00555)
- Kim et al., [Peri-LN](https://arxiv.org/abs/2502.02732)
- Henry et al., [Query-Key Normalization](https://arxiv.org/abs/2010.04245)
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Zhu et al., [Transformers without Normalization](https://arxiv.org/abs/2503.10622)
- Stollenwerk, [Layer Normalization and Dynamic Activation Functions](https://arxiv.org/abs/2503.21708)
- Xie et al., [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [OpenAI gpt-oss reference implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)
- Hugging Face documentation for [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2), [Llama 4](https://huggingface.co/docs/transformers/model_doc/llama4), [Gemma 2](https://huggingface.co/docs/transformers/model_doc/gemma2), [Gemma 3](https://huggingface.co/docs/transformers/model_doc/gemma3), [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3), and [DeepSeek-V4](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- Official Transformers implementations for [OLMo2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modular_olmo2.py), [Llama 4](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py), [Gemma 3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py), and [Qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)

**Next:** Module 6 turns the narrow gpt-oss expert into a full Mixture-of-Experts calculation: total versus active parameters, routing, and load balancing.

---

*Landscape verified: 6 August 2026. Numerical examples are reproduced by the local contract; empirical results retain source conditions; architecture properties are tied to official configurations, code, or technical reports.*
