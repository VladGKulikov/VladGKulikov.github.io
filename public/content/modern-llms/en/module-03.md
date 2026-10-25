# Module 3. The Basic Transformer

*Modern LLMs · Module 3 lecture*

Module 2 brought text to the entrance of the neural network: a tokenizer assigned identifiers to pieces of the string, and an embedding lookup selected the corresponding rows of a learned table. At that point, the same token receives the same starting vector wherever it occurs. Predicting a continuation, however, requires more than recognizing the token. It requires knowing what surrounds it.

This chapter follows the computation that makes that dependence possible. A Transformer builds **contextual representations**: positions exchange information, transform the features they have gathered, and pass the result to later layers. At the far end is the vocabulary projection we met in Module 2. Our task is to understand the computation connecting those two ends of the model.

The prerequisites are matrix multiplication, an understanding of shapes, and the embeddings introduced in Module 2. Section 2 introduces a tensor simply as a multidimensional array; the remaining concepts appear as the computation calls for them. We will first study the operations and assemble the original encoder–decoder Transformer, then turn to the code for a small causal block—the core of a decoder-only language model.

**Choose a reading route**

**On a first reading,** follow §§1–12 in order. In §2, concentrate on the input tensor: the full shape reference is expandable and can wait until attention is familiar. In §4.3, focus first on what scaling accomplishes; in §4.5, take time to work through the small matrices. Compare training with generation in §§11–12, then complete the account with §§16 and 22.

**To reproduce the computation,** return to the shape reference, the argument in §4.3, and the code in §§13–15. The module’s separate [exercises](/courses/modern-llms/module-03/exercises/) and [notebooks](/courses/modern-llms/module-03/notebooks/) let you follow the same path yourself.

**For later reference,** shapes are collected in §§2 and 15, masks in §5, normalization order in §8, and common confusions in §17. The separate [tests](/courses/modern-llms/module-03/tests/) provide a check on your understanding.

![VIZ m3/01 — three routes through the Transformer foundations](assets/modern-llms/en/module-03/m3_01_learning_route.svg)
*Start by following the information, then reproduce the computation, and keep the shape and mask contracts nearby as a reference.*

## 1. Why the Transformer was needed

Suppose the model already has its input vectors. Each describes the content at one position, but that alone is not enough to translate a sentence, summarize a document, or predict what comes next. At every position, the model must bring together two kinds of information:

1. the local content of the current token;
2. information from other positions, where the useful positions depend on context.

In “the bank raised the rate,” the meaning of “rate” is constrained by “bank.” In code, a closing delimiter may depend on a construct opened many lines earlier. During translation, each target token must use both the generated target prefix and the complete source sequence.

One way to carry context is to update a state in sequence. A recurrent network computes state $t$ from state $t-1$, so each step must wait for the preceding one. A Transformer organizes the exchange differently. Through **attention**, each position builds a weighted mixture of vectors from the positions it can access. When the requesting position and its sources belong to the same sequence, the operation is **self-attention**.

A **feed-forward network (FFN)** then transforms the features at each position. Two distinct jobs are taking shape: gather information from the context, and process the representation that results. Layers still depend on preceding layers, but positions within a layer can be computed together. Later we will see why this is compatible with generating tokens one at a time.

Being able to read other positions does not mean that every position is available. In next-token prediction, the future is unknown; a **causal mask** enforces that restriction by forbidding access to later positions. A **positional signal** has a different purpose: it represents where a token occurs. Sections 3 and 5 show how location and permission to read work together.

The same operations can be connected in different ways. The [original Transformer paper (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) addresses translation, with two streams: the source text and the translation being produced. Its **encoder** turns the source into contextual vectors, collectively called encoder memory. Its **decoder** builds the translation by reading both the available target prefix and that memory. Attention that reads from another stream is called **cross-attention**.

A **decoder-only** language model has no separate encoder: prompt and continuation occupy a single sequence. For now, keep the two designs distinct rather than trying to memorize every component. We will learn the shared operations first, assemble the translation model, and then return to causal language modeling. The diagram below is a view of the whole whose parts will gradually become familiar.

![VIZ m3/02 — the complete encoder–decoder Transformer](assets/modern-llms/en/module-03/m3_02_transformer_overview.svg)
*The source branch produces contextual vectors. The decoder uses them through cross-attention while building the translation. Sections 8–10 expand the sublayers and residual paths inside each branch.*

### 1.1. Scope of this lesson

The immediate goal is to understand the basic computation. The following links are a map of later topics, not a list of prerequisites:

- [Module 4](module-04.md) develops modern positional methods;
- [Module 5](module-05.md) moves from MHA to GQA, MLA, and FlashAttention;
- [Module 6](module-06.md) studies the modern residual block;
- [Module 10](module-10.md) turns the autoregressive forward pass into efficient serving.

This chapter is self-contained. Those links deepen the subject but are not prerequisites for completing this lesson.

## 2. Notation and tensor shapes

Before transforming the vectors, establish how to read the array that holds them. A token representation contains $d_{model}$ coordinates; a sequence contains $T$ positions. A **batch** groups $B$ examples to be processed together. Their vectors occupy a tensor—a multidimensional array—of shape $B\times T\times d_{model}$. Each axis answers a separate question: which example, which position, and which coordinate of its representation?

For now, keep $X\in\mathbb R^{B\times T\times d_{model}}$ and the meaning of its three axes in view. Other shapes will arrive with the operations that require them: queries, keys, and values in §4, and heads in §6. The reference below collects those shapes for later use. On a first reading, it is enough to continue to the residual stream.

**Full shape index — second pass and reference**

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $T$ | sequence length |
| $T_q$, $T_k$ | number of queries and keys; usually equal in self-attention |
| $d_{model}$ | number of features in the main representation stream |
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
Q \in \mathbb{R}^{B \times h \times T_q \times d_k}, \qquad
K \in \mathbb{R}^{B \times h \times T_k \times d_k}, \qquad
V \in \mathbb{R}^{B \times h \times T_k \times d_v}.
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

As a representation passes through the layers, its contents change while the main tensor keeps shape $B\times T\times d_{model}$. This main path is the **residual stream**. A sublayer—attention or an FFN—does not have to recreate the entire representation. It computes an update of the same shape, and a **residual connection** adds that update to the input:

$$
x \leftarrow x + \operatorname{Sublayer}(x).
$$

This expression shows the addition alone; §8 will place normalization around it. Addition requires compatible shapes for the input and the update. An unexpected change in the last dimension of such a block therefore usually points to a shape error or a missing output projection, rather than to a special property of the Transformer.

## 3. Step 1: tokens, embeddings, and positions

For this teaching model, IDs form the dense range $0,\ldots,|\mathcal V|-1$, and the table has exactly $|\mathcal V|$ rows. Real artifacts may distinguish defined IDs, the ID range, and allocated rows, as Module 2 explains.

Return now to the beginning of the forward pass. As Module 2 established, a neural network does not receive a string directly: a tokenizer first maps text to integer identifiers

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

Consider unmasked self-attention with no positional information. Permuting the input vectors permutes the output vectors in the same way. The mechanism sees a set of vectors and relations among them, but it does not know which vector came first.

For example, the vector associated with “dog” is unchanged when “dog bites person” becomes “person bites dog” under these assumptions; its output row simply moves. This property is **permutation equivariance**, not complete output invariance. A causal mask breaks that symmetry, so the argument applies specifically to unmasked attention without positional information.

### 3.2. Sinusoidal encoding in the original paper

Even and odd dimensions use sine/cosine pairs at different frequencies:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

Here $pos$ starts at zero, and $i$ indexes coordinate pairs from $0$ to $d_{model}/2-1$ for an even width. Small $i$ gives faster oscillations; large $i$ gives slower ones. This supplies several scales of variation. Being able to compute a positional code beyond the training lengths does not, by itself, guarantee useful model behavior there.

![VIZ m3/03 — sinusoidal positional encoding](assets/modern-llms/en/module-03/m3_03_positional_encoding.png)
*Four dimensions of one encoding. The high-frequency pair changes quickly from position to position, while the low-frequency pair supplies a slowly varying background.*

This fixed formula makes a convenient starting point. Module 4 compares it with learned tables, relative biases, and RoPE rotations. For now, the key result is that a token vector carries information about both content and position.

### 3.3. A quick shape check

Let $B=2$, $T=5$, and $d_{model}=8$:

- `token_ids`: `[2, 5]`;
- `embedding(token_ids)`: `[2, 5, 8]`;
- `position_encoding[:5]`: `[5, 8]`;
- after broadcasting and addition: `[2, 5, 8]`.

The positional table does not need a batch axis. The same coordinate $pos$ is used for every item in the batch.

## 4. Step 2: scaled dot-product attention

Each position now has a vector, but information from other positions still needs a way to enter it. Simply averaging every available vector would give every position the same rule for gathering context. Attention instead separates two questions: **which sources should this position match**, and **what should those sources contribute**?

A **query** represents the requesting side of the match. Each available source supplies a **key**, which is compared with the query to obtain a compatibility score. The source also supplies a **value**: the vector that will enter the resulting mixture. Keys help determine the weights; values determine what is mixed. The language of requests and searches is a guide to the computation, not a claim that these vectors implement hand-written retrieval rules. Their projections are learned.

One such computation is an **attention head**. The matrices $Q$, $K$, and $V$ collect queries, keys, and values as rows. Queries and keys have width $d_k$ per head; values have width $d_v$. There are $T_q$ queries and $T_k$ keys and values. Keys and values come in equal numbers because each source supplies one of each.

The computation for one head is

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

**Softmax** converts scores into nonnegative weights that sum to one. Here it operates separately for each query row, across the available keys. The transpose $K^\top$ exchanges the final two axes so that the matrix product compares every query with every key. We will add masking after understanding this mixture.

It is useful to unpack the equation into three steps:

$$
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{softmax}_{\mathrm{row}}(S),
\qquad
Z=AV.
$$

$S$ holds scaled compatibility scores, $A$ holds mixture weights, and $Z$ contains the resulting vectors. The symbol $W_O$ in §6 will instead name the output projection matrix.

The matrix identity $Z=AV$ means, row by row,

$$
z_i=\sum_{j=1}^{T_k} a_{ij}v_j.
$$

Every query position receives a weighted sum of value vectors. Section 5 keeps the same computation but replaces $S$ with masked scores $\widetilde S$ before softmax.

![VIZ m3/04 — the roles of query, key, and value](assets/modern-llms/en/module-03/m3_04_qkv_roles.svg)
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

A linear projection is multiplication by a learned matrix. It lets the same representation serve different purposes: matching a request, making a source available for matching, and providing the information to transmit. A position can request information from others while also acting as a source for them. We need three transformations of its representation, not three different kinds of token.

The projection matrices are model parameters learned during training. The queries, keys, and values themselves are computed from the current input. The transformation rules therefore persist across requests, while the particular pattern of interactions changes with the text.

In cross-attention, the sources differ:

$$
Q=X_{decoder}W_Q, \qquad K=X_{encoder}W_K, \qquad V=X_{encoder}W_V.
$$

Cross-attention changes where the vectors come from, while keeping the same scoring and mixing operation.

### 4.2. Why a dot product

Before scaling, the dot product of query $q_i$ with key $k_j$ is

$$
u_{ij}=\sum_{r=1}^{d_k}q_{ir}k_{jr}, \qquad s_{ij}=u_{ij}/\sqrt{d_k}.
$$

The scores $s_{ij}$ before softmax are **attention logits**. They are not yet probabilities: normalization turns each row into weights over keys. We will later use the same term for the model's output scores over vocabulary items.

A larger score receives a larger weight, all else equal. The sign alone does not establish relevance: softmax compares scores within a row. For nonzero vectors, **cosine similarity** is $q^\top k/(\|q\|\|k\|)$, which normalizes away their lengths. The dot-product score used here retains both magnitude and directional information.

### 4.3. Why divide by $\sqrt{d_k}$

There is still a question of scale. A dot product adds contributions from every coordinate, so increasing the width can increase the spread of the sum. To see why division helps, take a simple case: coordinates $q_r$ and $k_r$ are independent, zero-mean, and unit-variance. Then

$$
q^\top k=\sum_{r=1}^{d_k}q_rk_r,
$$

and under these simplifying assumptions,

$$
\operatorname{Var}(q^\top k)=d_k.
$$

The standard deviation grows like $\sqrt{d_k}$. Without scaling, wider keys can produce large logits, saturate softmax, make the distribution nearly one-hot, and leave small gradients for losing positions.

After division,

$$
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.
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

Apply the **causal constraint**: row 1 may read key 1, row 2 may read keys 1–2, and row 3 may read all three. Replace future-key scores with $-\infty$ **before** softmax, making their exponentials zero, and normalize over the remaining keys. This gives

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
z_3\approx0.248255v_1+0.248255v_2+0.503490v_3\approx[1.758725,1.000000].
$$

![VIZ m3/05 — numerical trace of causal attention](assets/modern-llms/en/module-03/m3_05_attention_trace.svg)
*The full route through one head: compatibility matrix, future-token prohibition, key-wise softmax, and value mixing. The module notebook reproduces every number.*

This example also exposes a crucial distinction: an **attention weight is a routing coefficient, not a complete explanation of the model’s decision**. A high weight says how much of a transformed value entered this local mixture. The result still travels through residual paths, other heads, FFNs, and later layers.

## 5. Masks: who is allowed to see whom

The numerical example has already prohibited access to future positions. We can now separate that access rule from the operation that enforces it. First decide which connections are allowed; then ensure that forbidden sources contribute nothing to the mixture.

This gives the word *mask* two related meanings: a Boolean visibility matrix and a numerical bias applied to logits. Begin with the more direct description, a Boolean matrix

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
\widetilde S=S+M,
\qquad
M_{ij}=
\begin{cases}
0, & C_{ij}=1,\\
-\infty, & C_{ij}=0.
\end{cases}
$$

$M$ is a numerical bias constructed from $C$, not a learned parameter. Allowed links receive 0 and forbidden links receive $-\infty$. A finite negative number only approximates this prohibition and does not resolve a row with no allowed keys (§5.6).

![VIZ m3/06 — padding, causal, and cross-attention masks](assets/modern-llms/en/module-03/m3_06_attention_masks.png)
*Padding-key and causal constraints can be combined. The fourth panel shows rectangular cross-attention. These plots mask keys; padded query rows are not independently suppressed.*

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

Even PyTorch uses two conventions: Boolean `True` allows a link in [`functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), but forbids one in [`MultiheadAttention.attn_mask`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). Our argument is named `allowed`, so `True` always permits a connection. A test checks that forbidden links receive zero weight.

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

So far, each position has received one mixture of values. It may be useful to compare that position with its context in several ways at once. Multi-head attention provides that capacity: each head builds its own projections and computes a mixture, and the results are then brought together.

These are parallel computations within a sublayer, not successive layers. To distinguish source representations from projected vectors, write $X_q,X_k,X_v$ for the inputs and $Q_i,K_i,V_i$ for the projected vectors of head $i$:

$$
Q_i=X_qW_Q^{(i)},\quad K_i=X_kW_K^{(i)},\quad V_i=X_vW_V^{(i)},
\qquad Z_i=\operatorname{Attention}(Q_i,K_i,V_i),
$$

$$
\operatorname{MHA}(X_q,X_k,X_v)=
\operatorname{Concat}(Z_1,\ldots,Z_h)W_O.
$$

Self-attention uses $X_q=X_k=X_v=X$; cross-attention takes $X_q$ from the decoder and $X_k=X_v$ from the encoder. Every head sees the full input vector before projecting it to a narrower space.

Let $h$ be the number of heads. A common configuration uses

$$
d_k=d_v=d_{model}/h.
$$

Concatenation then restores width $d_{model}$.

![VIZ m3/07 — multi-head attention](assets/modern-llms/en/module-03/m3_07_multihead.svg)
*The three input projections are often computed by one packed linear layer, then reshaped by head. Independent attention results are concatenated and passed through $W_O$.*

### 6.1. Why several heads can help

Each head has different $W_Q$, $W_K$, and $W_V$ projections. One head can become sensitive to local syntax, another to long-range matches, and another to separators or structure. That is a capacity, not a guarantee of clean, human-readable specialization. Heads may be redundant, distribute a function among themselves, or change roles across layers and examples.

### 6.2. Parameter count of basic MHA

Ignoring biases and assuming the total head width equals $d_{model}$:

- $W_Q$, $W_K$, $W_V$ each contain $d_{model}^2$ weights;
- $W_O$ adds another $d_{model}^2$.

Under these assumptions, the exact total is

$$
4d_{model}^2.
$$

At $d_{model}=512$, this is $1\,048\,576$ weights. With fixed total width, changing the head count changes the partitioning but not this leading parameter count.

### 6.3. What later modules change

**Looking ahead to Module 5.** MQA and GQA share keys and values across multiple query heads; MLA uses a compressed representation. Those variants can wait. Standard MHA first gives us a complete calculation with separate projections for every head.

## 7. The FFN: nonlinear processing at each position

Attention has allowed a position to gather information from its context. Gathering is not the whole job of a layer, however: the resulting features must also be transformed and combined. That is the role of the **position-wise FFN**.

Unlike attention, the FFN does not read other positions directly. This does not return us to an isolated token: its input representation has already been changed by the exchange of context. The distinction concerns the current operation—whether it mixes positions or transforms the coordinates within each one.

The basic FFN uses two linear maps with a nonlinearity between them. Keeping the chapter’s row-vector convention,

$$
\operatorname{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2.
$$

Let $d_{ff}$ denote the hidden FFN width. In the original paper,

$$
W_1\in\mathbb R^{d_{model}\times d_{ff}}, \qquad
W_2\in\mathbb R^{d_{ff}\times d_{model}},
$$

$\sigma$ is ReLU, which clips negative coordinates to zero: $\operatorname{ReLU}(u)=\max(0,u)$. Without a nonlinearity, the two linear maps could collapse into one. The base configuration uses $d_{model}=512$ and $d_{ff}=2048$.

### 7.1. What position-wise means

For $X$ of shape $B\times T\times d_{model}$, each linear layer acts on the final axis. Every token uses the same multilayer perceptron (MLP); no positions are mixed at this step. No explicit loop over $T$ is needed because `nn.Linear` already operates on the last dimension.

### 7.2. Why the FFN is large

Ignoring bias, the number of weights is

$$
2d_{model}d_{ff}.
$$

For $512\rightarrow2048\rightarrow512$, that is $2\,097\,152$ parameters—twice the four MHA projections in the same layer. In a simple encoder layer, the FFN owns roughly two thirds of the attention+FFN projection weights. Attention decides **where information comes from**; the FFN decides **how the resulting representation is transformed nonlinearly**.

Module 6 introduces **gating**, where one computed signal controls how much of another passes through; SwiGLU is one example. For this chapter, keep the division of labor clear: attention mixes positions, while the FFN transforms features at each position.

## 8. Residual connections and LayerNorm

We have now studied the layer’s two main transformations: attention exchanges information across positions, and the FFN processes features. Repeating these transformations in a deep network also requires a rule for combining each sublayer’s result with the representation already present.

Two mechanisms have different jobs here. A **residual connection** adds the computed update to the input, retaining an additional route for signals and gradients—the loss derivatives used to update weights during training. **Normalization** transforms a vector’s coordinates with reference to their scale. The two often sit beside each other in a diagram, but they are not one operation: changing their order changes the computation. We will first examine normalization itself, then the two ways of placing it around a residual branch.

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

$\gamma$ and $\beta$ are learned scale and offset vectors; $\odot$ means elementwise multiplication. A small positive $\varepsilon$ prevents division by zero when variance vanishes. Statistics are computed over one position’s features, separately from other tokens and batch items.

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

The sublayer first computes an update. That update is added to the input, and LayerNorm acts on the resulting sum. Normalization therefore processes the combined result of both branches. This is the ordering labeled `Add & Norm` in the paper’s diagram.

### 8.3. Pre-norm in modern LLMs

Many later decoder-only architectures normalize before the sublayer:

$$
y=x+\operatorname{Sublayer}(\operatorname{Norm}(x)).
$$

Now the input $x$ reaches the addition directly. Normalization lies inside the update branch: the sublayer reads a normalized representation, but its result is added to the original input. Moving the norm therefore changes the signal and gradient paths, not just the arrangement of labels in a diagram. The code in §14 will use this ordering. Module 6 studies modern blocks in more detail and compares LayerNorm with RMSNorm, which normalizes using the mean square of the coordinates.

![VIZ m3/08 — post-norm and pre-norm](assets/modern-llms/en/module-03/m3_08_block_order.svg)
*Each wrapper applies to one sublayer F, either attention or an FFN. The norm sits after residual addition on the left and inside the update branch on the right.*

### 8.4. Dropout in the original recipe

**Dropout** randomly zeros some values during training and rescales the retained values; ordinary generation disables it. The original Transformer applies it to sublayer outputs before residual addition and to the sum of token and positional embeddings. We omit it from the equations above to keep the paths visible.

## 9. The encoder layer and encoder stack

The components can now be assembled. First return to the Transformer’s original task, translation. The source text is available in full, so its encoder can let every position read all content-bearing positions in that text.

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

The stack repeats the layer structure $N$ times, with separate learned parameters in each layer. The base paper uses $N=6$. The final output is encoder memory:

$$
H_{enc}\in\mathbb{R}^{B\times T_{src}\times d_{model}}.
$$

Every source vector is now contextual: it can incorporate any unmasked source position. Here, “memory” means the tensor computed for this particular input, not a separate long-term store of knowledge. The decoder will consult it while constructing the translation.

### 9.1. When an encoder is useful

An encoder is useful when all input tokens are available, as in classification or document embeddings for search. Bidirectional attention lets a position read both left and right context. In a causal decoder-only model, even prompt positions can read only their own prefixes; the final prompt position can incorporate the entire prompt.

## 10. The decoder layer and cross-attention

The decoder faces a different situation. The source text is already represented by encoder memory, while the translation is available only as a prefix. It must use the complete source without looking ahead in the translation. These two requirements explain why the original decoder layer has two attention sublayers rather than one. Its three sublayers are:

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

Treat each word as one toy token here; a real tokenizer may split it further. `<BOS>` marks the start and `<EOS>` the end. The input is shifted by one token: the `<BOS>` position predicts “I,” the “I” position predicts “love,” and so on. The diagonal is allowed because the input token at a position is already part of the known prefix.

### 10.2. Linear projection and softmax

In Module 2, the output projection followed a contextual computation that was still a black box. That gap is now filled: attention and FFN layers have produced representations from which a continuation can be predicted. Each final position vector $h_t$, or **hidden state**, is multiplied by $W_{vocab}\in\mathbb R^{d_{model}\times|\mathcal V|}$. This produces one unnormalized score, called a **logit**, for each vocabulary item:

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

The forward pass already supplies probabilities. Training requires comparing them with the correct continuation. Because that continuation is available in a training example, we do not have to generate it using the model’s own predictions first. Instead, we feed in the correct sequence shifted by one position. This is **teacher forcing**.

The shift and causal mask work together: each scored position receives the correct prefix, but not the token it must predict. The penalty for that prediction is the **negative log-likelihood (NLL)** of the correct token. Summing over the sequence gives

$$
\mathcal L
=-\sum_{t=1}^{T}
\log p_\theta(y_t\mid y_{< t},x).
$$

$\theta$ denotes the learned parameters, $y_t$ the target token, $y_{<t}$ its prefix, and $x$ the source text in translation. Decoder-only modeling omits the separate source $x$. We use natural logarithms, so NLL is measured in nats. For a target distribution concentrated on one correct token, this loss is also the **cross-entropy** between the target and the model distribution.

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

If the correct token has zero-based index 3, the position loss is

$$
-\log p_3\approx 0.5439 \text{ nats}.
$$

Cross-entropy, KL divergence, and the coding interpretation of NLL are derived in the companion course: **Course *Information Theory for ML*, Module 3.** The engineering reading here is sufficient: the loss charges the model according to the probability it assigned to the correct token.

### 11.3. Loss masks are not attention masks

Let $m_t=1$ for a prediction that should be scored and $m_t=0$ otherwise. The mean loss is

$$
\overline{\mathcal L}=\frac{\sum_t m_t[-\log p_\theta(y_t\mid y_{<t},x)]}{\sum_t m_t},\qquad \sum_t m_t>0.
$$

Padding targets are excluded. Training on prompt–answer pairs may also score only the answer. An **attention mask** controls access to information; a **loss mask** selects predictions for scoring. A token excluded from the loss may still be available as context.

## 12. Generation: the same forward pass in a different loop

During generation, the model’s operations stay the same, but the supply of known correct prefixes disappears. Until the next token has been selected, it cannot become part of the next input. A sequential loop is therefore built around the forward pass:

1. pass the current prefix to the model;
2. read logits at the final position;
3. select the next token;
4. append it to the prefix;
5. repeat until `<EOS>` or a length limit.

![VIZ m3/09 — training and autoregressive inference](assets/modern-llms/en/module-03/m3_09_training_inference.svg)
*The same computation produces a next-token distribution. Training supplies known prefixes; generation extends a prefix with the model’s own choices.*

### 12.1. Greedy selection, sampling, and decoding policy

The forward pass ends with one logit $z_i$ per vocabulary ID. Turning those scores into a token is a separate decision. **Greedy decoding** takes the largest score. **Sampling** draws from a probability distribution, so repeated runs can continue the same prefix differently.

A positive **temperature** $\tau$ changes that distribution:

$$
p_i^{(\tau)}=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

At $\tau=1$ this is ordinary softmax. A temperature below one concentrates mass on higher logits; a temperature above one spreads it more evenly. Positive temperature preserves the ranking, so it does not change the greedy choice. The formula does not admit division by zero. An interface that uses `temperature=0` to request greedy decoding is selecting a separate rule.

For logits $[3,2,1,0]$, temperature one gives approximately $[0.643914,0.236883,0.087144,0.032059]$. At temperature two the probabilities become $[0.455054,0.276004,0.167405,0.101536]$. Greedy decoding still selects ID 0, while sampling is now more likely to pick one of the other IDs. Greater diversity alone says nothing about whether those alternatives solve the task better.

We can also restrict the candidate set $S$. **Top-k** keeps the $k$ most probable IDs. **Top-p**, also called nucleus sampling, sorts candidates by probability and keeps the shortest prefix whose cumulative mass reaches $p$. The candidate that crosses the threshold is included. Either filter requires renormalization before drawing a token:

$$
\widetilde p_i=\frac{p_i^{(\tau)}\,\mathbf{1}[i\in S]}{\sum_{j\in S}p_j^{(\tau)}}.
$$

Apply the filters separately at temperature one. Top-k with $k=2$ keeps IDs 0 and 1, yielding $[0.731059,0.268941,0,0]$. With top-p at $p=0.9$, those two IDs contribute only about 0.880797. Including ID 2 raises the retained mass to 0.967941, and renormalization gives $[0.665241,0.244728,0.090031,0]$. Top-k fixes a count; top-p lets the count adapt to the current distribution. See [Holtzman et al., Section 3](https://arxiv.org/abs/1904.09751) for the definition and original motivation of nucleus sampling.

![VIZ m3/12 — changing the token-selection distribution](assets/modern-llms/en/module-03/m3_12_sampling_policy.png)
*Every series starts from the same logits. The two filters are applied separately at temperature one; a zero-height bar marks an excluded ID.*

The notebook implements exponentiation, sorting, cumulative mass, and renormalization explicitly. It resolves equal logits by ID. When filters are combined, its order is temperature, top-k, then top-p; the final threshold is measured within the retained top-k distribution. Record that order as part of the generation protocol. Module 11 will preserve a target sampling distribution, which includes these transformations rather than only the model's raw softmax.

### 12.1.1. Stopping the loop

After selection, the new token joins the prefix used by the next forward pass. An **EOS token—end of sequence—** terminates the sequence. Dialogue formats may instead define particular end-of-turn markers; Module 13 explains that contract. A separate cap limits the number of new tokens. Reaching the cap is a different stopping reason from the model emitting an end marker.

A filter can exclude EOS at one step, while the next step has a different logit vector. Neither greedy decoding nor sampling by itself guarantees that an end marker will arrive in time. The notebook checks EOS termination and length termination on prepared score traces. These traces isolate loop control; they do not evaluate a trained model's answer quality.

### 12.2. Why errors can accumulate

Training prefixes are correct. During generation, an early poor token becomes part of the next input. The model may enter prefix regions that were rare in training. This is one reason token-level loss and long-form generation quality are not identical quantities.

### 12.3. Why a KV cache works

In a causal stack, an old token’s representation depends only on its existing prefix. Appending a token therefore leaves old keys and values unchanged as long as weights, positions, and available context stay fixed and dropout is disabled. A **KV cache** stores those keys and values at each layer. A new token contributes one query row while the stored key/value sequence grows. Editing the prefix or renumbering positions requires reconsidering cache validity.

Module 10 develops KV-cache memory and request serving. For an optional connection to reinforcement learning, *RL for LLM*, Module 6, treats token selection as an action under a policy. That perspective is not needed for the computation here.

## 13. A minimal scaled dot-product attention implementation

We can now turn the written computation into an implementation. The same steps should remain recognizable in the code: compare queries with keys, apply the mask before softmax, and mix values. Input checks make explicit the assumptions used in the equations.

Run the blocks in §§13–14 in order. The inputs are finite floating-point tensors with nonempty axes on one device, with a Boolean mask broadcastable to the score shape. This version computes in float32 or float64 and returns `(output, weights)`. The notebook adds diagnostic fields and padding support.

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
    if any(size <= 0 for t in (q, k, v) for size in t.shape):
        raise ValueError("q, k and v must have non-empty axes")
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
    batch_shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q, k, v = (t.expand(*batch_shape, *t.shape[-2:]) for t in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    if not torch.isfinite(scores).all():
        raise FloatingPointError("attention scores overflowed")

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
    if not torch.isfinite(output).all():
        raise FloatingPointError("attention output overflowed")
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
    if any(type(n) is not int for n in (t_q, t_k, query_offset)):
        raise TypeError("lengths and offset must be integers")
    if min(t_q, t_k) <= 0 or query_offset < 0:
        raise ValueError("positive lengths and non-negative offset required")
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
        if any(type(n) is not int or n <= 0 for n in (d_model, n_heads)):
            raise ValueError("d_model and n_heads must be positive integers")
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
        return self.out(y.to(dtype=x.dtype))
```

Once the explicit computation is clear, compare it with `torch.nn.functional.scaled_dot_product_attention`. The function selects an available implementation based on the device, dtype, and mask. Calling it does not guarantee FlashAttention or the absence of full intermediate matrices. The notebook compares outputs and gradients with `dropout_p=0.0` set explicitly.

## 14. A minimal Transformer block

Return now from translation to a single stream of text. A minimal **decoder-only** block needs no separate encoder memory: the available context is already in the same sequence. Causal self-attention and an FFN remain, while cross-attention is absent.

For this implementation, place normalization before each sublayer—the pre-norm arrangement from §8.3. The FFN uses **GELU**, the smooth nonlinearity $x\Phi(x)$, where $\Phi$ is the standard normal cumulative distribution function. This is a teaching decoder-only variant, not a reproduction of the 2017 block: the original uses post-norm and ReLU.

The organizing principle is familiar. Each sublayer computes an update, a residual connection adds it to the main stream, and the representation keeps its width. The following code makes those paths explicit.

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

Keep weights fixed and use `eval()` so dropout cannot create unrelated differences. Compare floating-point outputs with `torch.testing.assert_close`. Include an unmasked control to show that the chosen example can actually expose future leakage.

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

### 15.1. Counting the main projection weights

Return now from the toy dimensions to the paper’s base configuration: $d_{model}=512$, $d_{ff}=2048$. For one layer, excluding biases, LayerNorm parameters, and embeddings:

- encoder self-attention: $4d_{model}^2=1\,048\,576$;
- FFN: $2d_{model}d_{ff}=2\,097\,152$;
- encoder layer: $3\,145\,728$;
- decoder layer: two attention sublayers plus the FFN = $4\,194\,304$.

These counts are exact under the stated exclusions. A complete layer also has biases and normalization parameters; a complete model adds embeddings, the vocabulary projection, and multiple layers. The comparison explains the large FFN share: its two wide matrices contain twice as many weights as the four MHA projections.

## 16. The 2017 original and a modern LLM: what survived

We have followed the shared operations into two concrete designs: the original translation model and a small decoder-only block. Their differences can now be read as changes to a computation we understand, rather than as another list of architectural names. This is the useful sense in which “Transformer” names a family, not one fixed design: recognizable components can be connected and modified in different ways.

![VIZ m3/10 — encoder-only, decoder-only, and encoder–decoder](assets/modern-llms/en/module-03/m3_10_family_map.svg)
*All three families use the same construction kit but differ in masks and information flow. Modern LLMs center the decoder-only branch, while encoders and cross-attention remain important for understanding the origin of the architecture and many multimodal systems.*

**Reference map.** The replacement names below point to later chapters: positions in M4, attention variants in M5, norms and FFNs in M6, pretraining in M9, and execution in M10. They describe common choices rather than requirements for every LLM.

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

This connects Module 2’s embeddings to the positional mechanisms in Module 4. The rest of the course asks how to scale, stabilize, train, serve, and extend it.

### 16.2. Why GPT is more than “a Transformer with the encoder removed”

Formally, a decoder-only stack removes the encoder and cross-attention. The more useful positive description is a stack of causal self-attention blocks trained on a next-token objective. Prompt and continuation occupy one token stream. The graph is not merely truncated; its dominant training and deployment regime is different.

## 17. Common confusions

### 17.1. “Attention stores knowledge”

The weights $A$ are computed afresh from the current input. Learned projection, FFN, and embedding matrices persist across requests. Keep those model parameters distinct from temporary mixing weights. The size of the FFN alone does not establish what fraction of the model’s knowledge it stores.

### 17.2. “A large attention weight proves causality”

A weight is a local mixture coefficient for one head and one layer. It omits residual paths, other heads, transformed values, FFNs, and later layers. It is a diagnostic signal, not a universal explanation.

### 17.3. “A causal mask makes computation sequential”

It makes the **dependency** causal. The training forward pass is still parallel over positions. The generation loop is sequential because future tokens do not yet exist.

### 17.4. “Self-attention is always square”

Scores have shape $T_q\times T_k$. Ordinary self-attention uses $T_q=T_k$, but cross-attention and cached decoding may be rectangular.

### 17.5. “Heads split the input features before projection”

Each head projects from **all** $d_{model}$ input features. It is the projected result that is partitioned into heads, conventionally with width $d_{model}/h$. Different heads can therefore use the same input coordinates.

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

## 22. Final cheat sheet and the next step

![VIZ m3/11 — the basic Transformer in 60 seconds](assets/modern-llms/en/module-03/m3_11_cheatsheet.svg)
*Four panels to reconstruct from memory: tensor shapes, one attention head, the role of each sublayer, and correctness checks.*

At the start of the chapter, our vectors were rows selected from an embedding table. We can now trace how a position receives information from the available context, how layers transform that information, and how the final representation becomes a prediction. The complete route is:

1. a tokenizer produces ids; embeddings and positions turn them into $B\times T\times d_{model}$;
2. $QK^\top/\sqrt{d_k}$ produces connection logits, a mask removes forbidden links, and softmax normalizes over keys;
3. $AV$ mixes values, heads are joined through $W_O$;
4. the FFN transforms each position, while residual connections and normalization support a deep stack;
5. the encoder reads the complete source; the decoder reads only its target prefix and, when present, encoder memory;
6. teacher forcing trains all positions in parallel, while generation creates them sequentially;
7. modern LLMs replace positional methods, norms, FFNs, head layouts, and kernels, but this computational skeleton remains recognizable.

**Next:** [Module 4, positional representations](module-04.md), examines how the model represents token order and what happens when context grows beyond the lengths used in training.

---

*Stable mathematical foundations. Modules 4–6 and 10 develop modern variants of these mechanisms. Cross-entropy/KL theory: Information Theory for ML, Module 3. The token model viewed as a policy: RL for LLM, Module 6.*
