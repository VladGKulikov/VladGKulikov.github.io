# Module 18 (reference). The Basic Transformer

*Modern LLMs · reference lecture · revision 2026.5*

> **How this module fits the course.** This is not the eighteenth step of the core sequence. It is a reference entry point placed at the end of the repository. If you can sketch a basic Transformer from memory and explain $Q$, $K$, $V$, causal masking, the FFN, residual connections, and LayerNorm, you may skip it or keep it as a refresher. If any of those items still requires guesswork, study this lesson now and then return to [Module 1](../../module-01/en/module_01_lecture_EN.md).
>
> **What you will be able to do.** You will reconstruct the computation graph of the original encoder–decoder Transformer, work through one attention example by hand, implement a minimal version in PyTorch, explain the difference between training and autoregressive generation, and identify which 2017 components modern LLMs kept or replaced.

![VIZ m18/01 — two routes through the reference module](assets/modern-llms/en/module-18/m18_01_learning_route.svg)
*Readers who already know the architecture proceed directly to Module 1. Everyone else follows four anchors: input representations, attention, the block, and training versus generation.*

## 1. Why the Transformer was needed

Start with the engineering problem rather than the equation. A model that translates a sentence, summarizes a document, or predicts the next token must combine two kinds of information at every position:

1. the local content of the current token;
2. information from other positions, where the useful positions depend on context.

In “the bank raised the rate,” the meaning of “rate” is constrained by “bank.” In code, a closing delimiter may depend on a construct opened many lines earlier. During translation, each target token must use both the generated target prefix and the complete source sequence.

Before the Transformer, sequence models were commonly recurrent: state at step $t$ was built from state at step $t-1$. That order is natural, but it creates a long computation chain and prevents all positions from being processed in parallel during training. The Transformer takes a different route. Within each layer, every position may **directly assemble context from other positions** through attention. A small nonlinear network—the FFN—then transforms every position independently with shared weights. Layers remain sequential in depth, but positions within a layer are parallel.

Order has not disappeared. Pure self-attention does not distinguish a simultaneous permutation of its inputs, so order must be supplied explicitly through a positional signal and, in autoregressive models, a causal mask.

The primary source is [Vaswani et al., “Attention Is All You Need” (2017)](https://arxiv.org/abs/1706.03762). The paper studied machine translation, so the complete architecture has an encoder and a decoder. Most current text LLMs use only a decoder stack, but the full design is the cleanest place to learn the parts.

![VIZ m18/02 — the complete encoder–decoder Transformer](assets/modern-llms/en/module-18/m18_02_transformer_overview.svg)
*Source tokens pass through the encoder to form memory. The decoder reads a shifted target prefix, applies masked self-attention, cross-attends to encoder memory, and produces a next-token distribution.*

### 1.1. Scope of the reference lesson

This module develops the **basic mechanism**. You do not need RoPE, GQA, FlashAttention, RMSNorm, SwiGLU, KV-cache paging, or MoE yet. The core course will introduce them as engineering modifications of a graph you already understand:

- [Module 2](../../module-02/en/module_02_lecture_EN.md) expands tokenization and embeddings;
- [Module 3](../../module-03/en/module_03_lecture_EN.md) develops modern positional methods;
- [Module 4](../../module-04/en/module_04_lecture_EN.md) moves from MHA to GQA, MLA, and FlashAttention;
- [Module 5](../../module-05/en/module_05_lecture_EN.md) studies the modern residual block;
- [Module 9](../../module-09/en/module_09_lecture_EN.md) turns the autoregressive forward pass into efficient serving.

The reference module remains self-contained. Those links deepen the subject but are not prerequisites for completing this lesson.

## 2. Notation and tensor shapes

The Transformer becomes much easier to reason about when every operation is paired with a tensor shape. We will use:

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $T$ | sequence length |
| $T_q$, $T_k$ | number of queries and keys; usually equal in self-attention |
| $d_{model}$ | residual-stream width |
| $h$ | number of attention heads |
| $d_k$, $d_v$ | key/query width and value width per head |
| $d_{ff}$ | hidden width of the FFN |
| $\lvert \mathcal V \rvert$ | vocabulary size |

After embedding lookup,

$$
X \in \mathbb{R}^{B \times T \times d_{model}}.
$$

For multi-head attention, it is convenient to expose the head axis:

$$
Q,K \in \mathbb{R}^{B \times h \times T \times d_k}, \qquad
V \in \mathbb{R}^{B \times h \times T \times d_v}.
$$

Attention weights have shape

$$
A \in \mathbb{R}^{B \times h \times T_q \times T_k}.
$$

We will denote the value mixture produced by each head as $Z$:

$$
Z=AV \in \mathbb{R}^{B \times h \times T_q \times d_v}.
$$

The final axis of $A$ is the **key axis**. Softmax is applied there: for a fixed batch element, head, and query, key weights must sum to one. The final axis of $Z$ instead has value width $d_v$.

That is the first useful invariant. The second is that concatenated heads and the output projection $W_O$ must return to width $d_{model}$, otherwise the result cannot be added to the residual stream.

### 2.1. The residual stream as the main route

A useful mental model is a tensor of shape $B \times T \times d_{model}$ flowing through the entire network. Attention and the FFN temporarily transform it, but every sublayer returns the same shape. This makes residual addition possible:

$$
x \leftarrow x + \operatorname{Sublayer}(x).
$$

If the last dimension unexpectedly changes inside a basic block, that is almost never a subtle Transformer property. It is usually a shape bug or a missing output projection.

## 3. Step 1: tokens, embeddings, and positions

A neural network does not receive a string directly. A tokenizer maps text to integer identifiers

$$
(t_1,t_2,\ldots,t_T), \qquad t_i \in \{0,\ldots,|\mathcal V|-1\}.
$$

An embedding matrix

$$
E \in \mathbb{R}^{|\mathcal V| \times d_{model}}
$$

acts as a learned lookup table. Row $E_{t_i}$ is the vector for the token at position $i$. Batched lookup produces a tensor of shape $B \times T \times d_{model}$.

The original architecture scaled token embeddings by $\sqrt{d_{model}}$ and added a positional encoding:

$$
x_{pos}=\sqrt{d_{model}}\,E[t_{pos}] + PE(pos).
$$

The scale factor is not a universal law of every LLM; it belongs to the 2017 recipe. The important idea is the addition: token content and position enter the same residual space.

### 3.1. Why order vanishes without positions

Consider unmasked self-attention with no positional information. If rows of $Q$, $K$, and $V$ are permuted in the same way, the output is permuted in that same way. The mechanism sees a set of vectors and relations among them, but it does not know which vector came first.

Therefore “dog bites person” and “person bites dog” cannot be distinguished from token identities alone. The model needs an extra coordinate signal.

### 3.2. Sinusoidal encoding in the original paper

Even and odd dimensions use sine/cosine pairs at different frequencies:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

Small $i$ yields rapidly varying coordinates; large $i$ yields slowly varying ones. Together they encode position across several scales: high-frequency coordinates distinguish nearby tokens, while low-frequency coordinates preserve longer-range structure.

![VIZ m18/03 — sinusoidal positional encoding](assets/modern-llms/en/module-18/m18_03_positional_encoding.png)
*Four dimensions of one encoding. The high-frequency pair changes quickly from position to position, while the low-frequency pair supplies a slowly varying background.*

Sinusoidal encoding is an excellent conceptual baseline, not a description of most modern LLMs. Module 3 develops learned absolute embeddings, ALiBi, RoPE, YaRN, and multimodal extensions.

### 3.3. A quick shape check

Let $B=2$, $T=5$, and $d_{model}=8$:

- `token_ids`: `[2, 5]`;
- `embedding(token_ids)`: `[2, 5, 8]`;
- `position_encoding[:5]`: `[5, 8]`;
- after broadcasting and addition: `[2, 5, 8]`.

The positional table does not need a batch axis. The same coordinate $pos$ is used for every item in the batch.

## 4. Step 2: scaled dot-product attention

Attention answers one precise question: **which mixture of value vectors $V_j$ should each query position $i$ receive?**

In the original paper, one head is defined **without a mask argument**:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

This is Equation (1) in *Attention Is All You Need*. Softmax is applied row-wise, over the key axis. Decoder masking is described separately: entries corresponding to illegal connections are set to $-\infty$ at the input to softmax. To keep the base operation distinct from an information-flow constraint, we first develop unmasked attention and introduce masks in §5.

It is useful to unpack the equation into three steps:

$$
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{softmax}_{\mathrm{row}}(S),
\qquad
Z=AV.
$$

Here $S$ contains scaled compatibility scores, $A$ contains attention weights, and $Z$ is the result of one head. The paper does not assign that output matrix a separate letter. This lesson uses $Z$, not $O$, to avoid confusing a head output with the multi-head output projection $W_O$.

The matrix identity $Z=AV$ means, row by row,

$$
z_i=\sum_{j=1}^{T_k} a_{ij}v_j.
$$

Every query position receives a weighted sum of value vectors. Section 5 keeps the same computation but replaces $S$ with masked scores $\widetilde S$ before softmax.

![VIZ m18/04 — the roles of query, key, and value](assets/modern-llms/en/module-18/m18_04_qkv_roles.svg)
*$Q$, $K$, and $V$ are not three kinds of tokens. They are three learned roles assigned to representations. A table-lookup analogy helps as long as we remember that the query, index, and payload co-adapt during training.*

### 4.1. Where $Q$, $K$, and $V$ come from

In self-attention, all three tensors are linear projections of the same input $X$:

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V.
$$

For one head, we may write

$$
W_Q,W_K \in \mathbb{R}^{d_{model}\times d_k}, \qquad
W_V \in \mathbb{R}^{d_{model}\times d_v}.
$$

The projections let the model learn different features for “what I am looking for,” “how I should be found,” and “what I transmit.” A token may have a key that is useful for a syntactic match while its value carries semantic or positional content.

In cross-attention, the sources differ:

$$
Q=X_{decoder}W_Q, \qquad K=X_{encoder}W_K, \qquad V=X_{encoder}W_V.
$$

The decoder asks; encoder memory answers.

### 4.2. Why a dot product

For query $q_i$ and key $k_j$,

$$
s_{ij}=q_i^\top k_j.
$$

A large positive value indicates alignment in the learned space; a negative value indicates incompatibility. This is not necessarily cosine similarity because vector norms also affect the score. Explicit query/key normalization is a later architectural choice, not part of the basic definition.

### 4.3. Why divide by $\sqrt{d_k}$

Assume for intuition that coordinates $q_r$ and $k_r$ are independent, zero-mean, and unit-variance. Then

$$
q^\top k=\sum_{r=1}^{d_k}q_rk_r,
$$

and under these simplifying assumptions,

$$
\operatorname{Var}(q^\top k)\approx d_k.
$$

The standard deviation grows like $\sqrt{d_k}$. Without scaling, wider keys can produce large logits, saturate softmax, make the distribution nearly one-hot, and leave small gradients for losing positions.

After division,

$$
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx 1.
$$

This is not a theorem about a trained network: learned coordinates are neither independent nor guaranteed to have unit variance. It is nevertheless an exact explanation of the natural scale and a phenomenon the notebook reproduces with random vectors.

### 4.4. Softmax and the correct axis

For query $i$ in unmasked attention,

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{m=1}^{T_k}\exp(s_{im})}.
$$

Therefore

$$
a_{ij}\geq 0, \qquad \sum_{j=1}^{T_k} a_{ij}=1.
$$

In code this is `softmax(scores, dim=-1)` when the final axis is $T_k$. The wrong choice, such as `dim=-2`, may not raise an exception. It will normalize queries rather than keys. A row-sum assertion is therefore an essential unit test. Masked attention applies the same normalization to $\widetilde S$ from §5.

### 4.5. Attention by hand: a complete numerical example

Use one head, three positions, and $d_k=d_v=2$:

$$
Q=K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
V=
\begin{bmatrix}
1&0\\
0&2\\
3&1
\end{bmatrix}.
$$

Scaled scores are

$$
\frac{QK^\top}{\sqrt 2}
=
\begin{bmatrix}
0.707&0&0.707\\
0&0.707&0.707\\
0.707&0.707&1.414
\end{bmatrix}.
$$

With a causal mask, the first position sees only the first key, the second sees the first two, and the third sees all three. After masking and softmax,

$$
A\approx
\begin{bmatrix}
1&0&0\\
0.330&0.670&0\\
0.248&0.248&0.503
\end{bmatrix}.
$$

Let $Z$ denote the head output. In matrix form $Z=AV$, while each query row is

$$
z_i=\sum_{j=1}^{T_k}a_{ij}v_j.
$$

For this example,

$$
Z=AV\approx
\begin{bmatrix}
1.000&0.000\\
0.330&1.340\\
1.759&1.000
\end{bmatrix}.
$$

The third row is especially useful to read in words:

$$
z_3=0.248v_1+0.248v_2+0.503v_3=[1.759,1.000].
$$

![VIZ m18/05 — numerical trace of causal attention](assets/modern-llms/en/module-18/m18_05_attention_trace.svg)
*The full route through one head: compatibility matrix, future-token prohibition, key-wise softmax, and value mixing. The module notebook reproduces every number.*

This example also exposes a crucial distinction: an **attention weight is a routing coefficient, not a complete explanation of the model’s decision**. A high weight says how much of a transformed value entered this local mixture. The result still travels through residual paths, other heads, FFNs, and later layers.

## 5. Masks: who is allowed to see whom

The word *mask* is used for two related but distinct objects: a Boolean visibility matrix and a numerical bias added to logits. Rather than hide the operation inside “scores + mask,” begin with a Boolean matrix

$$
C_{ij}\in\{0,1\},
$$

where $C_{ij}=1$ means query $i$ is allowed to read key $j$. Define masked scores directly:

$$
\widetilde s_{ij}=
\begin{cases}
s_{ij}, & C_{ij}=1,\\
-\infty, & C_{ij}=0.
\end{cases}
$$

Then

$$
A=\operatorname{softmax}_{\mathrm{row}}(\widetilde S),
\qquad
Z=AV.
$$

Because $\exp(-\infty)=0$, forbidden connections receive zero weight. This is exactly the paper's prose description: illegal entries **at the input to softmax** are set to $-\infty$.

Implementations often express the same operation as

$$
\widetilde S=S+B,
\qquad
B_{ij}=
\begin{cases}
0, & C_{ij}=1,\\
-\infty, & C_{ij}=0.
\end{cases}
$$

Here $B$ is not a learned component and not a Boolean mask. It is an **additive attention bias** derived from $C$. Optimized kernels may accept $C$ directly and perform the conversion internally; implementations may also use a very large negative value compatible with the dtype instead of mathematical $-\infty$.

![VIZ m18/06 — padding, causal, and cross-attention masks](assets/modern-llms/en/module-18/m18_06_attention_masks.png)
*Three different constraints on information flow. They may be combined: a causal decoder must hide both future positions and padding.*

### 5.1. Padding mask

Sequences in a batch have different lengths, but tensors are rectangular. Shorter examples are padded with `PAD`. Those positions contain no content and must not be used as keys. Query rows belonging to padding are normally excluded from the loss as well.

### 5.2. Causal mask

A decoder-only language model predicts token $t+1$ at position $t$ and must not use future context. Keys $j\leq i$ are allowed:

$$
C_{ij}=1 \text{ when } j\leq i, \qquad C_{ij}=0 \text{ when } j> i.
$$

This is a lower-triangular pattern. It makes parallel training valid: all positions are computed in one forward pass, but each position receives only its admissible prefix.

### 5.3. Encoder self-attention

A classic encoder has no reason to hide right context. Every source representation may read the complete source sequence. Only padding needs to be masked.

### 5.4. Cross-attention mask

The number of target queries and source keys may differ. Decoder queries read encoder memory; padded source keys are hidden. A causal triangle over the source axis is normally unnecessary because the complete source is already available.

### 5.5. A common Boolean-mask trap

In one API, `True` may mean “allow”; in another it may mean “mask out.” Do not infer semantics from intuition. Read the documentation for the exact function and write a test asserting that forbidden positions receive zero probability.

### 5.6. A fully masked row

If a query is allowed to read **no** keys, the row does not define a probability distribution. A naive call such as

```python
import torch

torch.softmax(torch.tensor([-float("inf"), -float("inf")]), dim=0)
# tensor([nan, nan])
```

produces `NaN`: subtracting the row maximum creates the indeterminate expression $-\infty-(-\infty)$. A safe educational implementation must therefore expose an explicit policy:

- `error` — the default: stop and surface a masking bug;
- `zero` — return zero weights and a zero output for that row when this is intentional, for example for a padded query excluded from the loss.

Under the `zero` policy the row sum of the weights is zero, not one. This is not an ordinary probability distribution; it is the contract “this query produces no attention output.” Silently substituting a uniform distribution or allowing `NaN` to propagate is incorrect.

## 6. Multi-head attention

A single head constructs one routing matrix. Multi-head attention performs this computation in several learned subspaces:

$$
\operatorname{head}_i=
\operatorname{Attention}(QW_Q^{(i)},KW_K^{(i)},VW_V^{(i)}),
$$

$$
\operatorname{MHA}(Q,K,V)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$

A common configuration uses

$$
d_k=d_v=d_{model}/h.
$$

Concatenation then restores width $d_{model}$.

![VIZ m18/07 — multi-head attention](assets/modern-llms/en/module-18/m18_07_multihead.svg)
*The three input projections are often computed by one packed linear layer, then reshaped by head. Independent attention results are concatenated and passed through $W_O$.*

### 6.1. Why several heads can help

Each head has different $W_Q$, $W_K$, and $W_V$ projections. One head can become sensitive to local syntax, another to long-range matches, and another to separators or structure. That is a capacity, not a guarantee of clean, human-readable specialization. Heads may be redundant, distribute a function among themselves, or change roles across layers and examples.

### 6.2. Parameter count of basic MHA

Ignoring biases and assuming the total head width equals $d_{model}$:

- $W_Q$, $W_K$, $W_V$ each contain $d_{model}^2$ weights;
- $W_O$ adds another $d_{model}^2$.

The total is approximately

$$
4d_{model}^2.
$$

At $d_{model}=512$, this is $1\,048\,576$ weights. With fixed total width, changing the head count changes the partitioning but not this leading parameter count.

### 6.3. What later modules change

MHA stores separate $K,V$ states for every head, which is expensive during autoregressive inference. MQA and GQA reduce the number of KV heads; MLA compresses them differently. These variants alter projections and caching while retaining the same conceptual path—scores, softmax, value mixture. Module 4 develops the details.

## 7. The FFN: nonlinear processing at each position

After attention has mixed information **across positions**, a position-wise feed-forward network transforms every position **independently with the same weights**:

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2.
$$

In the original paper,

$$
W_1: d_{model}\rightarrow d_{ff}, \qquad
W_2: d_{ff}\rightarrow d_{model},
$$

and $\sigma$ is ReLU. The base configuration uses $d_{model}=512$ and $d_{ff}=2048$.

### 7.1. What position-wise means

For $X$ of shape $B\times T\times d_{model}$, each linear layer acts on the final axis. Every token uses the same MLP; no positions are mixed at this step. No explicit loop over $T$ is needed because `nn.Linear` already operates on the last dimension.

### 7.2. Why the FFN is large

Ignoring bias, the number of weights is

$$
2d_{model}d_{ff}.
$$

For $512\rightarrow2048\rightarrow512$, that is $2\,097\,152$ parameters—twice the four MHA projections in the same layer. In a simple encoder layer, the FFN owns roughly two thirds of the attention+FFN projection weights. Attention decides **where information comes from**; the FFN decides **how the resulting representation is transformed nonlinearly**.

Modern LLMs often use gated FFNs such as SwiGLU and a different expansion ratio. The recognizable function remains: expand channels, apply a nonlinearity or gate, and return to residual width.

## 8. Residual connections and LayerNorm

A deep network must add transformations while preserving a stable path for signals and gradients. The Transformer wraps every sublayer with a residual connection and normalization.

### 8.1. LayerNorm over the final axis

For one position vector $x\in\mathbb{R}^{d_{model}}$,

$$
\mu=\frac{1}{d_{model}}\sum_{r=1}^{d_{model}}x_r,
$$

$$
\sigma^2=\frac{1}{d_{model}}\sum_{r=1}^{d_{model}}(x_r-\mu)^2,
$$

$$
\operatorname{LN}(x)=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta.
$$

$\gamma$ and $\beta$ are learned. Mean and variance are computed separately for each token position over features, not over batch items.

For $x=[1,2,3,4]$, $\gamma=1$, $\beta=0$, and negligible $\varepsilon$,

$$
\mu=2.5, \qquad \sigma^2=1.25,
$$

$$
\operatorname{LN}(x)\approx[-1.342,-0.447,0.447,1.342].
$$

### 8.2. Post-norm in the original architecture

`Attention Is All You Need` uses

$$
y=\operatorname{LN}(x+\operatorname{Sublayer}(x)).
$$

The sublayer runs first, residual addition follows, and LayerNorm comes last. The paper’s diagram labels this `Add & Norm`.

### 8.3. Pre-norm in modern LLMs

Many later decoder-only architectures normalize before the sublayer:

$$
y=x+\operatorname{Sublayer}(\operatorname{Norm}(x)).
$$

The residual route $x\rightarrow y$ is then direct. LayerNorm is also often replaced by RMSNorm. Module 5 develops the trade-offs.

![VIZ m18/08 — post-norm and pre-norm](assets/modern-llms/en/module-18/m18_08_block_order.svg)
*Both variants contain attention, an FFN, residual connections, and normalization, but they are different computation graphs. A pre-norm equation and post-norm code are not interchangeable descriptions.*

### 8.4. Dropout in the original recipe

The original Transformer applies dropout to sublayer outputs before residual addition and to the sum of token and positional embeddings. It is reasonable to omit dropout while learning the graph, but an exact reproduction of the paper must include it.

## 9. The encoder layer and encoder stack

One encoder layer performs two operations:

1. bidirectional self-attention;
2. a position-wise FFN.

Each has its own residual connection and LayerNorm. In post-norm notation,

$$
h' = \operatorname{LN}\left(h+\operatorname{MHA}(h,h,h)\right),
$$

$$
h^{next}=\operatorname{LN}\left(h'+\operatorname{FFN}(h')\right).
$$

The stack repeats this layer $N$ times. The base paper uses $N=6$. The final output is encoder memory:

$$
H_{enc}\in\mathbb{R}^{B\times T_{src}\times d_{model}}.
$$

Every source vector is now contextual: it depends on the complete unmasked source sequence.

### 9.1. Why learn the encoder when modern LLMs are decoder-only

An encoder is natural when the entire input is known and bidirectional context is useful: classification, retrieval embeddings, or masked-token objectives. A decoder-only model can read a full prompt as a prefix, but its attention remains causal. Architectural families differ not in whether they “have attention,” but in information direction and training objective.

## 10. The decoder layer and cross-attention

The original decoder layer contains three sublayers:

1. masked self-attention over the target prefix;
2. cross-attention to encoder memory;
3. an FFN.

In compact post-norm notation,

$$
z_1=\operatorname{LN}\left(z+\operatorname{MaskedMHA}(z,z,z)\right),
$$

$$
z_2=\operatorname{LN}\left(z_1+\operatorname{MHA}(z_1,H_{enc},H_{enc})\right),
$$

$$
z^{next}=\operatorname{LN}\left(z_2+\operatorname{FFN}(z_2)\right).
$$

The second line takes queries from the decoder and keys/values from the encoder. The decoder can therefore select relevant source information for each target step.

### 10.1. Why the target is shifted

Suppose the correct target sequence is

```text
I love models <EOS>
```

The decoder input is

```text
<BOS> I love models
```

while the loss targets are

```text
I love models <EOS>
```

Input and target are shifted by one token. The `<BOS>` position predicts “I,” the “I” position predicts “love,” and so on.

### 10.2. Linear projection and softmax

The final hidden state at each position is projected to vocabulary logits:

$$
\ell_t=h_tW_{vocab}+b, \qquad
\ell_t\in\mathbb{R}^{|\mathcal V|}.
$$

Probabilities are

$$
p(y_t=v\mid y_{< t},x)=\operatorname{softmax}(\ell_t)_v.
$$

Some models tie the output matrix to the input embedding matrix. Weight tying saves parameters and connects the token-reading and token-writing spaces, but it is not required by the Transformer architecture.

## 11. Training: teacher forcing and cross-entropy

With teacher forcing, position $t$ receives the **gold** prefix and minimizes the negative log-probability of the correct next token:

$$
\mathcal L
=-\sum_{t=1}^{T}
\log p_\theta(y_t\mid y_{< t},x).
$$

For a decoder-only LLM, source $x$ may be absent and conditioning remains only on the prefix.

### 11.1. Why training is parallel

Although the probability factorization is autoregressive, every correct target token is already available in a training example. The complete shifted target is passed as one tensor, and the causal mask preserves validity. Positions are computed in parallel while each sees only its allowed prefix.

The distinction is fundamental:

- the **distributional dependency** is autoregressive;
- the **training forward computation** is parallel over positions.

### 11.2. A numerical NLL example

Suppose one position has vocabulary logits

$$
[1.2,\;0.3,\;-0.4,\;2.0].
$$

Softmax gives

$$
[0.2608,\;0.1060,\;0.0527,\;0.5805].
$$

If the correct token has index 3, the position loss is

$$
-\log 0.5805\approx 0.5439 \text{ nats}.
$$

Cross-entropy, KL divergence, and the coding interpretation of NLL are derived in the companion course: **Course *Information Theory for ML*, Module 3.** The engineering reading here is sufficient: the loss charges the model according to the probability it assigned to the correct token.

### 11.3. Loss masks are not attention masks

Not every position should contribute to the average. Padding is excluded. In SFT, prompt tokens are often excluded and only answer tokens contribute. A causal mask controls **which information a position may access**; a loss mask controls **which predictions are scored**. They are different tensors with different jobs.

## 12. Generation: the same forward pass in a different loop

At inference time, the correct future no longer exists. The algorithm is:

1. pass the current prefix to the model;
2. read logits at the final position;
3. select the next token;
4. append it to the prefix;
5. repeat until `<EOS>` or a length limit.

![VIZ m18/09 — training and autoregressive inference](assets/modern-llms/en/module-18/m18_09_training_inference.svg)
*The objective is the same—next-token probability. Training uses a gold prefix and scores all positions in parallel; generation constructs its own prefix one decision at a time.*

### 12.1. Greedy selection, sampling, and decoding policy

The output distribution may be decoded with `argmax`, temperature sampling, top-k/top-p filtering, or another rule. The Transformer block does not change; only the policy for selecting a token from its distribution changes.

### 12.2. Why errors can accumulate

Training prefixes are correct. During generation, an early poor token becomes part of the next input. The model may enter prefix regions that were rare in training. This is one reason token-level loss and long-form generation quality are not identical quantities.

### 12.3. Why a KV cache works

At step $t+1$, keys and values for previous tokens have not changed. A naive implementation recomputes them; an efficient decoder stores $K,V$ for every layer and computes only the new token’s projections. The new query is one row, while cached keys and values grow along the time axis.

This serving optimization follows directly from the basic equation. Module 9 develops memory arithmetic, PagedAttention, batching, and compression. **Course *RL for LLM*, Module 6.** It treats the token factorization as a policy over actions and connects log-probabilities to KL regularization.

## 13. A minimal scaled dot-product attention implementation

The following code is intentionally unoptimized. Its value is that every line corresponds to the equation.

```python
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed: Tensor | None = None,
    *,
    empty_row: Literal["error", "zero"] = "error",
) -> tuple[Tensor, Tensor]:
    """Educational attention with an explicit fully-masked-row policy.

    q: [..., T_q, d_k]
    k: [..., T_k, d_k]
    v: [..., T_k, d_v]
    allowed: broadcastable bool mask [..., T_q, T_k], True = allowed
    """
    if empty_row not in {"error", "zero"}:
        raise ValueError("empty_row must be 'error' or 'zero'")
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("q, k and v must have at least two dimensions")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("q, k and v must be finite floating-point tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k and v must be on one device")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share d_k")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share T_k")

    dtype = torch.float64 if torch.float64 in {q.dtype, k.dtype, v.dtype} else torch.float32
    q, k, v = (t.to(dtype=dtype) for t in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    else:
        if allowed.dtype is not torch.bool:
            raise TypeError("allowed must be a bool mask")
        allowed = torch.broadcast_to(allowed.to(scores.device), scores.shape)

    empty = ~allowed.any(dim=-1)
    if empty_row == "error" and empty.any():
        raise ValueError("attention contains a fully masked row")

    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    row_max = torch.where(
        (~empty)[..., None],
        masked_scores.max(dim=-1, keepdim=True).values,
        torch.zeros_like(masked_scores[..., :1]),
    )
    exponentials = torch.where(
        allowed,
        torch.exp(masked_scores - row_max),
        torch.zeros_like(masked_scores),
    )
    denominator = exponentials.sum(dim=-1, keepdim=True)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    weights = exponentials / denominator
    output = weights @ v
    return output, weights
```

A fully masked row is not a probability distribution: naive `softmax([-inf, …, -inf])` produces `NaN`. The safe default is therefore an error. Enable the `zero` policy only explicitly—for example, for a padded query whose attention output should be zero.

### 13.1. A causal mask

```python
def causal_allowed(
    t_q: int,
    t_k: int,
    *,
    query_offset: int = 0,
    device=None,
) -> Tensor:
    q_pos = query_offset + torch.arange(t_q, device=device)[:, None]
    k_pos = torch.arange(t_k, device=device)[None, :]
    return k_pos <= q_pos
```

For ordinary self-attention, $t_q=t_k=T$ and `query_offset=0`. During cached decoding, the new query may have local index zero but global position $t$; use `query_offset=t`, otherwise the mask incorrectly permits only the first key.

### 13.2. A minimal multi-head module

```python
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(z: Tensor) -> Tensor:
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        allowed = None
        if causal:
            allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        y, _ = scaled_dot_product_attention(q, k, v, allowed)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(y)
```

Production code should use `torch.nn.functional.scaled_dot_product_attention` or a specialized backend. Those implementations choose efficient kernels and avoid naively materializing all intermediates. Learning to map each shape to the educational implementation comes first.

## 14. A minimal Transformer block

The next class is a pre-norm decoder-only block—not an exact copy of the paper, but a graph closer to modern LLMs:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal=True)
        x = x + self.ffn(self.ln2(x))
        return x
```

Read the graph in words:

1. `ln1(x)` normalizes every position;
2. causal self-attention mixes only the admissible prefix;
3. its output is added to the original `x`;
4. a second norm+FFN transforms every position;
5. another residual connection preserves route and shape.

### 14.1. Causal invariance as a unit test

A strong mask test changes a future token and checks that earlier outputs do not change. For position $i$,

$$
h_i(x_1,\ldots,x_i,x_{i+1},\ldots)
=
h_i(x_1,\ldots,x_i,x'_{i+1},\ldots).
$$

Floating-point outputs should be compared with `torch.testing.assert_close`. If the test fails, future information leaked through attention or another operation.

## 15. A complete shape trace for a toy model

Let

- $B=2$;
- $T=5$;
- $|\mathcal V|=100$;
- $d_{model}=32$;
- $h=4$;
- $d_k=d_v=8$;
- $d_{ff}=128$.

One decoder-only forward pass has the following shapes:

| Stage | Shape |
|---|---|
| token ids | $2\times5$ |
| embedding + position | $2\times5\times32$ |
| packed QKV | $2\times5\times96$ |
| Q/K/V after splitting heads | $2\times4\times5\times8$ |
| scores | $2\times4\times5\times5$ |
| per-head attention output | $2\times4\times5\times8$ |
| concatenation + $W_O$ | $2\times5\times32$ |
| FFN hidden state | $2\times5\times128$ |
| block output | $2\times5\times32$ |
| vocabulary logits | $2\times5\times100$ |

This table is a practical debugger. Most first-implementation failures are not in the mathematics. They are in `transpose`, `view`, mask broadcasting, or the softmax axis.

### 15.1. Approximate parameter floor of an original layer

Ignoring biases, LayerNorm, and embeddings:

- encoder self-attention: $4d_{model}^2=1\,048\,576$;
- FFN: $2d_{model}d_{ff}=2\,097\,152$;
- encoder layer: $3\,145\,728$;
- decoder layer: two attention sublayers plus the FFN = $4\,194\,304$.

This is not the full parameter count of the paper. Biases, norms, embeddings, output projection, and the number of layers still matter. The floor exposes the geometry: the FFN is a major parameter owner, and cross-attention adds another complete set of projections to the decoder.

## 16. The 2017 original and a modern LLM: what survived

The word “Transformer” can misleadingly suggest one fixed architecture. In practice it names a family.

![VIZ m18/10 — encoder-only, decoder-only, and encoder–decoder](assets/modern-llms/en/module-18/m18_10_family_map.svg)
*All three families use the same construction kit but differ in masks and information flow. Modern LLMs center the decoder-only branch, while encoders and cross-attention remain important for understanding the origin of the architecture and many multimodal systems.*

| Component | Original Transformer | Typical modern decoder-only LLM |
|---|---|---|
| architecture | encoder–decoder | decoder-only |
| attention | MHA | MHA, GQA, MQA, MLA, or a hybrid |
| position | sinusoidal absolute | usually RoPE and extensions |
| normalization order | post-norm | usually pre-norm |
| normalization | LayerNorm | often RMSNorm |
| FFN | ReLU MLP | often SwiGLU/gated MLP |
| task | seq2seq translation | next-token pretraining |
| inference | autoregressive decoder | autoregressive decoder + KV cache |
| efficiency | ordinary attention computation | FlashAttention, fused kernels, paging |

### 16.1. The stable core

Despite those replacements, the recognizable path remains:

1. discrete tokens become vectors;
2. positions and masks define sequence structure;
3. attention routes information across positions;
4. the FFN transforms each position nonlinearly;
5. residual connections and normalization support a deep stack;
6. a vocabulary head defines the next-token distribution.

This is the core you should carry into Module 1. The rest of the course asks how to scale, stabilize, train, serve, and extend it.

### 16.2. Why GPT is more than “a Transformer with the encoder removed”

Formally, a decoder-only stack removes the encoder and cross-attention. The more useful positive description is a stack of causal self-attention blocks trained on a next-token objective. Prompt and continuation occupy one token stream. The graph is not merely truncated; its dominant training and deployment regime is different.

## 17. Common confusions

### 17.1. “Attention stores knowledge”

Attention dynamically mixes current hidden states. Much of the parameter capacity lives in projection matrices and especially in FFNs; knowledge is distributed across the network. Attention is not a database, even when a retrieval analogy helps explain one operation.

### 17.2. “A large attention weight proves causality”

A weight is a local mixture coefficient for one head and one layer. It omits residual paths, other heads, transformed values, FFNs, and later layers. It is a diagnostic signal, not a universal explanation.

### 17.3. “A causal mask makes computation sequential”

It makes the **dependency** causal. The training forward pass is still parallel over positions. The generation loop is sequential because future tokens do not yet exist.

### 17.4. “Self-attention is always square”

Scores have shape $T_q\times T_k$. Ordinary self-attention uses $T_q=T_k$, but cross-attention and cached decoding may be rectangular.

### 17.5. “Each head receives the entire $d_{model}$ width”

Projections start from $d_{model}$, but their result is normally reshaped into $h$ heads of width $d_{model}/h$. Concatenation restores the full width.

### 17.6. “LayerNorm normalizes the batch”

LayerNorm normalizes features within one token position. BatchNorm uses statistics across examples; it is a different operation.

### 17.7. “Masking after softmax is equivalent”

Simply zeroing weights after softmax makes row sums smaller than one. Renormalization is possible but unnecessary and easy to destabilize. The canonical operation modifies logits before softmax.

### 17.8. “Post-norm and pre-norm differ only in the drawing”

Norm placement changes signal and gradient routes. The equation and the code must describe the same variant.

### 17.9. “Input embeddings and the output head must share weights”

Weight tying is common, not mandatory. Inspect the model configuration and implementation.

### 17.10. “The basic Transformer explains everything about modern LLMs”

It explains the computational skeleton. Data, optimization, scaling laws, post-training, retrieval, serving, multimodality, and evaluation are separate systems covered by the core course.

## 21. Reading route and sources

The following resources serve different purposes rather than duplicating one another.

1. **Primary source:** [Vaswani et al., “Attention Is All You Need”](https://arxiv.org/abs/1706.03762). Read the abstract, §3, Figure 1, §5, and the hyperparameter table before attempting the full training setup.
2. **Visual intuition:** [Jay Alammar, “The Illustrated Transformer”](https://jalammar.github.io/illustrated-transformer/). A strong first pass through encoder–decoder flow and Q/K/V.
3. **Text next to code:** [Harvard NLP, “The Annotated Transformer”](https://nlp.seas.harvard.edu/annotated-transformer/) and its [repository](https://github.com/harvardnlp/annotated-transformer). Useful for matching the paper to an implementation.
4. **Geometric intuition:** [3Blue1Brown, “Attention in transformers”](https://www.3blue1brown.com/lessons/attention/) and [“Transformers, the tech behind LLMs”](https://www.3blue1brown.com/lessons/gpt/).
5. **Building a decoder-only model from scratch:** [Andrej Karpathy, “Let’s build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) and the [Zero to Hero](https://karpathy.ai/zero-to-hero.html) route. This is the GPT branch, not an exact encoder–decoder reproduction.
6. **A textbook with exercises:** [Dive into Deep Learning, Transformer chapter](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html).
7. **Modern PyTorch API:** the [Scaled Dot Product Attention tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) and [function documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
8. **A compact self-attention implementation:** [Sebastian Raschka, “Understanding and Coding Self-Attention…”](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html).

A useful route for a newcomer is: this lesson → Alammar → the notebook → Figure 1 and §3 of the paper → Karpathy or the Annotated Transformer. Experienced readers may need only the paper, VIZ 11, and the notebook tests.

## 22. Final cheat sheet and return to the core course

![VIZ m18/11 — the basic Transformer in 60 seconds](assets/modern-llms/en/module-18/m18_11_cheatsheet.svg)
*Four panels worth reconstructing without the surrounding prose: shapes, core equations, implementation invariants, and distinctions that must not be conflated.*

The complete route is now compact:

1. a tokenizer produces ids; embeddings and positions turn them into $B\times T\times d_{model}$;
2. $QK^\top/\sqrt{d_k}$ produces connection logits, a mask removes forbidden links, and softmax normalizes over keys;
3. $AV$ mixes values, heads are joined through $W_O$;
4. the FFN transforms each position, while residual connections and normalization support a deep stack;
5. the encoder reads the complete source; the decoder reads only its target prefix and, when present, encoder memory;
6. teacher forcing trains all positions in parallel, while generation creates them sequentially;
7. modern LLMs replace positional methods, norms, FFNs, head layouts, and kernels, but this computational skeleton remains recognizable.

There is no hidden entry barrier now. Return to [Module 1, “The Modern LLM Landscape”](../../module-01/en/module_01_lecture_EN.md): the basic block will function as a coordinate system for the rest of the course rather than an unstated prerequisite.

---

*Reference material: stable classical foundations. Continue with Modules 2–5 and 9 for modern implementations. Cross-entropy/KL theory: Information Theory for ML, Module 3. The token model viewed as a policy: RL for LLM, Module 6.*
