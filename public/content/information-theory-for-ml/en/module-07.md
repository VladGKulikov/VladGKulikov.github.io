# Module 7. Channel and capacity

> **How to read this module.** Sections 7.1–7.7 form the classical core: the channel model, capacity, three basic examples, and both halves of the coding theorem. In §7.6, it is worth following at least the central random-coding count, because that is where the threshold $R<I(X;Y)$ becomes visible. The normal approximation at the end of §7.7 is intended for a second pass. Sections 7.8–7.10 bring the channel language back to ML: noisy labels, ECOC, learned communication, and LLMs.

## 7.1. Reliable communication through noise

In Module 6, we tried to remove everything predictable from a message and represent it as compactly as possible. The new problem points in almost the opposite direction: the message must cross a medium that flips, erases, or corrupts symbols. Redundancy, which we had just worked to eliminate, suddenly becomes useful.

Imagine a binary channel that independently flips every transmitted bit with probability $0.1$. If we send one thousand information bits without protection, the receiver will see about one hundred errors on average. The most obvious defense is repetition followed by majority voting. It helps, but it spends rate quickly: triple repetition carries only $1/3$ information bit per channel use and still does not eliminate error completely.

Shannon's result says something much stronger. This channel has capacity

$$
C=1-h_2(0.1)\approx0.531
$$

bit per use. At every rate below $C$, one can choose longer and longer block codes whose probability of error tends to zero. To first asymptotic order, one thousand information bits require at least

$$
\frac{1000}{C}\approx1883
$$

channel uses, not three thousand. This number does not yet promise a good short code of length $1883$—finite blocklength will need its own analysis. But it already breaks the naive choice between “rate” and “reliability.”

> **Noise need not be paid for by repeating each symbol. A good code separates entire messages in the space of long sequences so that typical noise does not confuse them.**

This immediately separates two operations.

| Operation | What it does | Main criterion |
|---|---|---|
| **Source coding** | removes statistical redundancy | average description length |
| **Channel coding** | adds structured redundancy | error probability at a given rate |

The key word is **structured**. Appending random extra bits does not help. The additional symbols must keep codewords distinguishable under the distortions characteristic of the channel.

That is the content of the channel coding theorem: a given channel has a threshold $C$. Below it, reliable communication is asymptotically achievable; above it, no code can make the error disappear. The second half matters just as much as the first: capacity is not a property of one coding algorithm, but a limit of the probabilistic channel model under its stated constraints.

For ML, this is not merely a decorative analogy. Annotator errors, noisy latent layers, quantized signals, wireless transmission of learned features, and coded class representations genuinely define channels once the input, output, and transition law are specified. Channel language is useful for LLMs as well, but there we will need to separate a precise model from a metaphor.

![](assets/information-theory-for-ml/en/module-07/M7_source_channel_EN.png)

## 7.2. A discrete memoryless channel and a block code

To discuss a reliability limit, we first need a complete task. It has three parts: the channel, the codebook, and the decoder.

### A discrete memoryless channel

A **discrete memoryless channel** (DMC) consists of

- a finite input alphabet $\mathcal X$;
- a finite output alphabet $\mathcal Y$;
- a transition law
  $$
  W(y\mid x)=\Pr(Y=y\mid X=x).
  $$

For $n$ channel uses, the conditional distribution factorizes:

$$
W^n(y^n\mid x^n)
=
\prod_{i=1}^{n}W(y_i\mid x_i).
$$

“Memoryless” describes the noise **conditional on the input sequence**. It does not require the input symbols themselves to be independent. A useful codeword is usually highly structured across coordinates; that dependence is precisely where its protection comes from.

### Two basic channels

A **binary symmetric channel** (BSC) with crossover probability $\varepsilon$ has transition matrix

$$
W_{\mathrm{BSC}}
=
\begin{pmatrix}
1-\varepsilon & \varepsilon\\
\varepsilon & 1-\varepsilon
\end{pmatrix}.
$$

Each transmitted bit is independently flipped with probability $\varepsilon$. The receiver sees $0$ or $1$, but does not know where an error occurred.

A **binary erasure channel** (BEC) with erasure probability $\varepsilon$ has output alphabet $\{0,1,?\}$:

$$
W_{\mathrm{BEC}}
=
\begin{pmatrix}
1-\varepsilon & 0 & \varepsilon\\
0 & 1-\varepsilon & \varepsilon
\end{pmatrix}.
$$

The receiver does not know the value of an erased bit, but does know its location. We will soon see how valuable that side information is.

![](assets/information-theory-for-ml/en/module-07/M7_bsc_bec_EN.png)

### A block code

Let the sender choose one of $M$ equiprobable messages:

$$
J\sim\operatorname{Unif}\{1,\ldots,M\}.
$$

A length-$n$ code consists of an encoder

$$
f_n:\{1,\ldots,M\}\to\mathcal X^n
$$

and a decoder

$$
g_n:\mathcal Y^n\to\{1,\ldots,M\}.
$$

The encoder sends the codeword

$$
X^n=f_n(J),
$$

the channel produces $Y^n$, and the decoder returns

$$
\widehat J=g_n(Y^n).
$$

The average block error probability is

$$
P_e^{(n)}=\Pr(\widehat J\ne J).
$$

Sometimes the worst message must also be controlled:

$$
P_{e,\max}^{(n)}
=
\max_j\Pr(\widehat J\ne j\mid J=j).
$$

Maximum error is at least average error and imposes a stronger requirement. For an ordinary DMC, the two criteria lead to the same capacity; moving to maximum error requires one extra expurgation step that removes the worst codewords.

The code rate

$$
R_n=\frac{1}{n}\log_2M
$$

is measured in **information bits per channel use**. If $M=2^{nR}$, the code distinguishes $2^{nR}$ messages and transmits $nR$ bits of choice over $n$ uses.

A rate $R$ is achievable if there is a sequence of codes such that

$$
\liminf_{n\to\infty}R_n\ge R,
\qquad
P_e^{(n)}\to0.
$$

Capacity is defined operationally as

$$
C=\sup\{R:R\text{ is achievable}\}.
$$

So far, this tells us what quantity we want, but not how to calculate it. The next formula connects this engineering definition to the mutual information of Module 5.

A unit warning is also essential: bits per channel use are not bits per second. Physical throughput additionally depends on how many independent, or effectively independent, channel uses are available per second.

## 7.3. Capacity of a discrete channel

For a finite DMC with no additional input constraint, Shannon's theorem gives

$$
\boxed{
C=\max_{P_X}I(X;Y)
}.
$$

The input distribution $P_X$ and the transition law $W(y\mid x)$ induce the joint distribution

$$
P_{XY}(x,y)=P_X(x)W(y\mid x).
$$

For a fixed $P_X$, mutual information answers

> How much information about the random input remains, on average, in one output under this way of using the channel?

Capacity asks a second question:

> Which input distribution uses the channel best?

The distinction is easy to miss because both quantities have units of bits per use. But $I(X;Y)$ is an operating point under a chosen $P_X$, whereas $C$ is the optimum over all admissible operating points.

The theorem has two halves.

- **Achievability:** for every $R<C$, there are codes with $P_e^{(n)}\to0$.
- **Converse:** if $R>C$, the error cannot tend to zero. For a DMC, the strong converse states that $P_e^{(n)}\to1$.

Mutual information therefore acquires an operational meaning. It is not only a measure of dependence between $X$ and $Y$: after optimizing the input, it becomes the highest reliable communication rate.

### Reading the optimization over $P_X$

Suppose an annotator confusion matrix is treated as a channel from a clean label $Y$ to an observed label $\widetilde Y$. In an actual dataset, the class distribution is fixed, so

$$
I(Y;\widetilde Y)
$$

measures the information preserved at those class frequencies. By contrast,

$$
\max_{P_Y}I(Y;\widetilde Y)
$$

describes the best possible use of the same transition matrix. One number concerns the dataset; the other concerns the channel as a mathematical object.

### A few consequences

For a finite DMC,

$$
0\le C\le\min\{\log_2|\mathcal X|,\log_2|\mathcal Y|\}.
$$

Capacity is zero if and only if all rows of the transition matrix are identical:

$$
W(\cdot\mid x)=W(\cdot\mid x')
\quad\text{for all }x,x'.
$$

Then the output has the same distribution under every input and cannot distinguish messages.

If input symbols have different costs, capacity becomes

$$
C(\Gamma)
=
\max_{P_X:\,\mathbb E[c(X)]\le\Gamma}I(X;Y).
$$

This line is especially important for continuous channels. Without a power or amplitude constraint, one could keep separating signals farther apart and obtain unbounded rates. The resource constraint is part of the problem definition, not an afterthought attached to the formula.

## 7.4. BSC, BEC, and the Z-channel

Three binary channels show why the single word “noise” is not enough. What matters is not only how often symbols are damaged, but also what structure of error the receiver observes.

### Binary symmetric channel

Let $X\sim\operatorname{Bernoulli}(1/2)$. For a BSC, the output is also uniform and the conditional entropy is $h_2(\varepsilon)$. Hence

$$
I(X;Y)=H(Y)-H(Y\mid X)=1-h_2(\varepsilon).
$$

Channel symmetry makes the uniform input optimal:

$$
\boxed{
C_{\mathrm{BSC}}=1-h_2(\varepsilon)
}.
$$

At $\varepsilon=0.1$,

$$
C_{\mathrm{BSC}}\approx0.5310
\quad\text{bit/use}.
$$

It is usually enough to consider $0\le\varepsilon\le1/2$. If $\varepsilon>1/2$, the receiver can flip every output and obtain an equivalent BSC with crossover probability $1-\varepsilon$. At the extreme $\varepsilon=1$, the channel again transmits one bit without loss: it deterministically maps $0$ to $1$ and $1$ to $0$.

### Binary erasure channel

Let $X\sim\operatorname{Bernoulli}(p)$. When a symbol is not erased, the input is known exactly; when it is erased, all the original uncertainty remains. Therefore

$$
H(X\mid Y)=\varepsilon H(X)
$$

and

$$
I(X;Y)=(1-\varepsilon)H(X).
$$

The maximum is attained at $p=1/2$:

$$
\boxed{
C_{\mathrm{BEC}}=1-\varepsilon
}.
$$

At $\varepsilon=0.1$, this is $0.9$ bit/use—substantially larger than the BSC with the same numerical parameter. The point is not that erasure is intrinsically “weaker.” A BEC tells the decoder where damage occurred; a BSC first forces the decoder to infer whether an error occurred at all.

### An asymmetric Z-channel

Now consider

$$
W=
\begin{pmatrix}
1&0\\
\alpha&1-\alpha
\end{pmatrix}.
$$

Symbol $0$ passes without error, while $1$ changes to $0$ with probability $\alpha$. If

$$
p=\Pr(X=1),
$$

then

$$
I(p)=h_2\bigl((1-\alpha)p\bigr)-p\,h_2(\alpha).
$$

For $\alpha=0.25$, the maximum is attained approximately at

$$
p^*\approx0.4278,
\qquad
C\approx0.5582\text{ bit/use}.
$$

A uniform input gives slightly less:

$$
I(1/2)\approx0.5488.
$$

The numerical difference is modest but the principle is important: the vulnerable symbol should be used less often. This is why the capacity formula optimizes over $P_X$ instead of silently assuming a uniform input.

For a general finite channel matrix, the optimum can be found with the Blahut–Arimoto algorithm; an implementation appears in the computational exercise.

![](assets/information-theory-for-ml/en/module-07/M7_capacity_curves_EN.png)

## 7.5. The Gaussian channel and a power constraint

A discrete alphabet already limits how many signals can be distinguished. A real-valued channel has no such built-in ceiling, so its resource constraint must be stated explicitly.

Consider the discrete-time AWGN channel

$$
Y=X+Z,
\qquad
Z\sim\mathcal N(0,N),
$$

where the noise is independent of the input and the average signal power satisfies

$$
\mathbb E[X^2]\le P.
$$

Capacity per real channel use is

$$
\boxed{
C_{\mathrm{AWGN}}
=
\frac12\log_2\left(1+\frac PN\right)
}
$$

bit/use. The optimal input is Gaussian:

$$
X\sim\mathcal N(0,P).
$$

The mechanism behind the formula is direct. Since $Y=X+Z$,

$$
I(X;Y)=h(Y)-h(Z).
$$

The noise entropy is fixed. Under variance $P+N$, output entropy is maximized by a Gaussian distribution, and a Gaussian input makes the output Gaussian, achieving the bound.

For a continuous-time channel of bandwidth $B$ Hz, average signal power $S$, and total noise power $N$ in that band, the Shannon–Hartley formula is

$$
\boxed{
C_{\mathrm{bps}}
=
B\log_2\left(1+\frac SN\right)
}
$$

bit/s. The first formula is in bits per real channel use; the second is in bits per second. Their shared logarithm does not make the units interchangeable.

### What the curve says

Let the signal-to-noise ratio (SNR) be $\rho=P/N$. Then

- at low SNR, $\rho\ll1$,
  $$
  C\approx\frac{\rho}{2\ln2};
  $$
- at high SNR, $\rho\gg1$,
  $$
  C\approx\frac12\log_2\rho.
  $$

At low SNR, extra power raises rate almost linearly. At high SNR, returns become logarithmic. The exact gain from doubling SNR is

$$
C(2\rho)-C(\rho)
=
\frac12\log_2\frac{1+2\rho}{1+\rho}.
$$

It is below $0.5$ bit/use and approaches $0.5$ only at high SNR. Thus “doubling power buys half a bit” is a useful high-SNR approximation, not an identity over the entire curve.

The formula is a limit for an idealized model. A real system adds fading, interference, finite constellations, hardware constraints, latency, protocol overhead, and finite blocklength. A claim of operating “near the Shannon limit” is incomplete until it says which model and which error criterion define that limit.

![](assets/information-theory-for-ml/en/module-07/M7_shannon_hartley_EN.png)

## 7.6. Why random coding achieves $R<I(X;Y)$

We have now reached the most surprising part of the theorem. To prove that a good code exists, Shannon did not begin with an ingenious explicit construction. He drew the entire codebook at random and bounded its average error.

This is one of those cases where a random object is easier to analyze than the best concrete object. If the average error over an ensemble of random codebooks tends to zero, at least one deterministic codebook performs no worse than the average.

The whole count can first be seen in one line. A codebook contains about $2^{nR}$ codewords. For a fixed wrong codeword, the chance of looking plausible next to the received output is of order $2^{-nI(X;Y)}$. The total false-match risk is therefore of order

$$
2^{nR}\cdot2^{-nI(X;Y)}
=
2^{-n(I(X;Y)-R)}.
$$

If $R<I(X;Y)$, the number of competitors grows more slowly than the probability of a false match decays. That balance is the threshold.

Now let us state the argument precisely. Fix an input distribution $P_X$ and a rate

$$
R<I(X;Y).
$$

Let $P_Y$ be the output distribution induced by $P_X$ and the channel.

### Step 1. A random codebook

For each message $j\in\{1,\ldots,M\}$, with

$$
M\approx2^{nR},
$$

generate a codeword independently:

$$
X^n(j)\sim P_X^n.
$$

### Step 2. Information density

For one input-output pair, define the **information density**

$$
\imath(x;y)
=
\log_2\frac{W(y\mid x)}{P_Y(y)}.
$$

Its mean under the joint distribution is mutual information:

$$
\mathbb E_{P_XW}[\imath(X;Y)]=I(X;Y).
$$

For a memoryless channel,

$$
\imath(x^n;y^n)
=
\sum_{i=1}^{n}\imath(x_i;y_i).
$$

The law of large numbers gives, for the true pair,

$$
\frac1n\imath(X^n(J);Y^n)
\xrightarrow{P}
I(X;Y).
$$

### Step 3. A threshold decoder

The decoder looks for the unique message $j$ such that

$$
\imath(X^n(j);Y^n)
\ge
n\bigl(I(X;Y)-\delta\bigr).
$$

An error has two possible causes.

1. The true pair misses the threshold. Its probability tends to zero by the law of large numbers.
2. Some wrong codeword crosses the threshold.

For $j'\ne J$, the codeword $X^n(j')$ is independent of $Y^n$. Under the product distribution $P_X^nP_Y^n$,

$$
\mathbb E\left[2^{\imath(X^n;Y^n)}\right]=1.
$$

Markov's inequality gives

$$
\Pr\left[
\imath(X^n;Y^n)
\ge n(I-\delta)
\right]
\le
2^{-n(I-\delta)}.
$$

There are about $2^{nR}$ wrong codewords, so a union bound yields

$$
\Pr(\text{false match})
\le
2^{nR}2^{-n(I-\delta)}
=
2^{-n(I-R-\delta)}.
$$

Choose $\delta>0$ so that

$$
R<I(X;Y)-\delta.
$$

Both error terms then vanish. Since this holds for any fixed $P_X$, optimizing $P_X$ proves achievability of every rate below

$$
C=\max_{P_X}I(X;Y).
$$

![](assets/information-theory-for-ml/en/module-07/M7_joint_typical_EN.png)

### What the argument has established

First, it proves the existence of a deterministic code: a small ensemble-average error implies that some realization is good. More strongly, when the ensemble-average error vanishes, the fraction of codebooks with substantially larger error also vanishes under a suitable threshold. Good books are not isolated needles in the asymptotic ensemble.

Second, the argument explains **why** mutual information appears. It controls the exponent with which an unrelated codeword can masquerade as compatible with the output.

But an existence proof is not yet a convenient codec. A random book contains on the order of $2^{nR}$ words, is expensive to store, and requires exponential brute-force decoding. Practical families—LDPC, turbo, and polar codes—supply structure and efficient decoders. The theorem first showed that there was something worth finding; coding theory then learned how to find it constructively.

## 7.7. Converse and finite blocklength

Achievability asks what can be done below $C$. The converse sets a limit on every possible code: above $C$, reliability cannot be restored by a cleverer encoder or decoder.

### A short converse through Fano's inequality

Let $J$ be uniform over $M$ messages, with rate

$$
R=\frac1n\log_2M.
$$

Fano's inequality gives

$$
H(J\mid Y^n)
\le
1+P_e^{(n)}\log_2M.
$$

Also,

$$
\begin{aligned}
\log_2M
&=I(J;Y^n)+H(J\mid Y^n)\\
&\le I(X^n;Y^n)+1+P_e^{(n)}\log_2M\\
&\le nC+1+P_e^{(n)}\log_2M.
\end{aligned}
$$

The last line uses memorylessness:

$$
I(X^n;Y^n)
\le
\sum_{i=1}^{n}I(X_i;Y_i)
\le nC.
$$

Therefore

$$
\boxed{
P_e^{(n)}
\ge
1-\frac{C}{R}-\frac{1}{nR}
}.
$$

When $R>C$, the right-hand side remains positive for large $n$, so error cannot tend to zero. This is the **weak converse**.

For a DMC, the **strong converse** goes further: at a fixed $R>C$,

$$
P_e^{(n)}\to1.
$$

Above capacity, the channel does not merely enter a “slightly worse” regime. As blocklength grows, it almost entirely loses the ability to distinguish messages reliably at that rate.

### Why $R<C$ is not yet an engineering answer

Capacity is a first-order asymptotic result. A real system specifies the triple

$$
(n,R,\epsilon),
$$

where $n$ is blocklength, $R$ is rate, and $\epsilon$ is the target error probability. At short blocklength, a rate below $C$ may still have large error. A substantial engineering problem lies between “asymptotically possible” and “works under a 10 ms latency budget.”

### Mathematical deepening: the normal approximation

For many regular channels, the maximal rate at blocklength $n$ and target error $\epsilon$ is approximately

$$
R^*(n,\epsilon)
\approx
C-
\sqrt{\frac{V}{n}}\,Q^{-1}(\epsilon)
+O\left(\frac{\log n}{n}\right),
$$

where $V$ is the **channel dispersion** and $Q^{-1}$ is the inverse Gaussian tail function.

Capacity sets the center, while $V$ captures fluctuations of information density and therefore the price of a short block. The higher the required reliability and the smaller $n$, the farther the operating rate must back off from $C$.

![](assets/information-theory-for-ml/en/module-07/M7_finite_blocklength_EN.png)

## 7.8. Noisy labels as a channel

We can now move the same structure from a communication link into a dataset. Let $Y$ be a clean label and $\widetilde Y$ the label recorded after an annotator or an automated corruption pipeline. Under class-conditional noise, the transition is described by

$$
T_{ij}
=
\Pr(\widetilde Y=j\mid Y=i).
$$

This is a literal probabilistic channel

$$
Y\longrightarrow\widetilde Y.
$$

If the features $X$ influence the observed label only through the clean label, then

$$
X\longrightarrow Y\longrightarrow\widetilde Y
$$

is a Markov chain, and data processing gives

$$
I(X;\widetilde Y)
\le
I(X;Y).
$$

A noisy label cannot retain more information about the input than a clean label. The amount lost, however, depends on the full transition matrix and the class distribution—not only on a single error rate.

### Symmetric $K$-class noise

Suppose a label remains correct with probability $1-\eta$ and, on an error, is replaced uniformly by one of the other $K-1$ classes:

$$
\Pr(\widetilde Y=j\mid Y=i)
=
\begin{cases}
1-\eta,&j=i,\\
\eta/(K-1),&j\ne i.
\end{cases}
$$

For uniform $Y$, the output is uniform as well, so

$$
\boxed{
I(Y;\widetilde Y)
=
\log_2K-h_2(\eta)-\eta\log_2(K-1)
}.
$$

This channel is weakly symmetric, so the uniform input is capacity-achieving and the same expression is its capacity.

For $K=10$ and $\eta=0.2$,

$$
I(Y;\widetilde Y)\approx1.9660\text{ bits}
$$

out of the original $\log_2 10\approx3.3219$ bits.

The most interesting point is not $\eta=1$ but

$$
\eta=\frac{K-1}{K}.
$$

Here every row becomes uniform and all information disappears. At $\eta=1$, annotator accuracy is zero, but the recorded label still conveys one reliable fact: the true class is different from it. Thus

$$
I(Y;\widetilde Y)
=
\log_2\frac{K}{K-1}>0.
$$

For $K=10$, about $0.152$ bit remains. This is a clean example of why **accuracy and mutual information answer different questions**: a systematically wrong signal can still be fully or partially invertible.

### Where the noise matrix stops being enough

Real annotation errors often depend on the instance $X$, on the annotator, on ambiguity, and on the filtering pipeline. A single class-conditional matrix $T$ cannot represent all of those mechanisms.

When $T$ is known, it can be used for forward or backward loss correction. But inverting an ill-conditioned matrix amplifies statistical noise, and an estimated $\widehat T$ contributes its own error. Channel information loss, identifiability of the transition matrix, and numerical stability of correction are three separate problems.

![](assets/information-theory-for-ml/en/module-07/M7_label_noise_EN.png)

## 7.9. Channel coding inside ML: ECOC and DeepJSCC

In this section, “code” is no longer only an analogy. In both constructions, an encoder, a channel, and a decoder are part of the algorithm itself.

### Error-correcting output codes

In **error-correcting output codes** (ECOC), each of $K$ classes receives a codeword

$$
c_k\in\{-1,+1\}^m.
$$

Each column of the code matrix defines a binary task, so the learner trains $m$ binary classifiers. Their outputs form a predicted word $\widehat c(x)$, and prediction selects the nearest class codeword.

If the minimum Hamming distance between codewords is $d_{\min}$, an ideal nearest-codeword decoder corrects up to

$$
\left\lfloor\frac{d_{\min}-1}{2}\right\rfloor
$$

bit errors. The geometry is exactly the geometry of channel coding: error balls around different classes must not overlap.

In classification, however, the “error channel” is rarely an independent BSC. Binary classifiers share data, their errors are correlated, columns have different difficulty, and the error probability depends on $x$. A large $d_{\min}$ supplies a useful geometric margin, but not a complete statistical guarantee.

![](assets/information-theory-for-ml/en/module-07/M7_ecoc_EN.png)

### Learned joint source-channel coding

A classical digital system often separates three stages:

1. source compression;
2. error protection;
3. mapping coded symbols into a physical signal.

In **joint source-channel coding** (JSCC), the full mapping is optimized jointly for a distortion measure and a channel model. In DeepJSCC, a neural encoder, a differentiable channel layer, and a neural decoder form

$$
x
\xrightarrow{f_\theta}z
\xrightarrow{W}\widetilde z
\xrightarrow{g_\phi}\widehat x.
$$

The parameters minimize reconstruction or task loss. In the original image experiments, DeepJSCC outperformed the chosen separation-based baselines in several low-SNR and low-bandwidth regimes, and its quality degraded smoothly under SNR mismatch rather than showing a sharp digital cliff.

This does not overturn the source-channel separation theorem. Under standard stationary models, asymptotically long blocks, and matched criteria, separate coding can be optimal. If $\kappa$ channel uses are available per source symbol, the distortion condition is $R(D)<\kappa C$; writing $R(D)<C$ assumes $\kappa=1$ and compatible units. DeepJSCC is attractive precisely where the practical problem departs from that ideal asymptotic setting: blocks are short, latency is constrained, the channel model may be mismatched, and quality is measured by continuous distortion rather than only exact recovery of bits.

## 7.10. An LLM as a channel: a precise model and the limits of the analogy

The word “channel” also arises naturally around language models. The useful move is not to reject the analogy, but to start with a version in which it is mathematically exact.

### A channel from prompt to response

Fix the model weights, the context-construction rules, and the generation algorithm. Let $X$ be a random prompt and $Y$ the generated response. Then

$$
q_\theta(y\mid x)
$$

is a conditional output distribution and therefore defines a stochastic channel from $X$ to $Y$. Under a fixed prompt distribution, one can compute

$$
I(X;Y).
$$

With deterministic decoding, the channel simply becomes a deterministic map. With temperature sampling, top-$p$, or another randomized procedure, those rules are part of the transition law.

To discuss **capacity**, we must add an operational coding task. For example, a sender chooses a message $J$, encodes it as an admissible prompt $x(J)$ of at most $L$ tokens, the model generates a response, and a decoder tries to recover $J$. Only after specifying the prompt set, cost, unit of channel use, and error criterion does “maximum rate” become a precise quantity.

Such an experiment can be defined. But it measures the capacity of a particular prompt–model–response protocol, not a universal number attached to the Transformer architecture outside a task.

![](assets/information-theory-for-ml/en/module-07/M7_llm_channel_EN.png)

### Context length as a resource

A context window of $L$ tokens is a real resource: it constrains input volume and computational cost. But the token count does not directly determine the number of reliably transmitted bits.

To turn window length into a channel quantity, one would need to specify

- the prompt distribution and admissible prompt set;
- the cost of different tokens or sequences;
- the response transition distribution;
- an encoder and decoder for messages;
- an error criterion;
- the role of knowledge already stored in model parameters.

Without these pieces, context length remains a resource size rather than a Shannon capacity.

*Lost in the Middle* showed that retrieval quality can depend strongly on the location of relevant evidence: in the studied tasks, models often used the beginning and end of a long context more effectively than its middle. This is an important empirical property of specific models and protocols. It does not follow automatically from DPI or from an abstract “window capacity”; its mechanism depends on architecture, positional encoding, training, and task design.

RAG changes the protocol before the model sees the input: a retriever selects a small set of evidence rather than placing an entire corpus into the prompt. The practical gain can be described as saving context budget and reducing competition from irrelevant passages. There is no need to call it an automatic increase in capacity—the more precise statement is already strong enough.

### What attention weights measure

A row of attention weights

$$
\alpha_{i,\cdot}
$$

sums to one and therefore resembles a probability distribution. But it is a set of mixing coefficients over values, not a transition law from an input token to an output message. The layer output also depends on value vectors, multiple heads, the residual stream, and subsequent nonlinearities.

The entropy

$$
H(\alpha_{i,\cdot})
$$

has a clear local meaning: it describes how concentrated the routing weights are across keys. Low entropy means that mass is focused on a few positions; high entropy means it is spread more broadly. By itself, it is not the number of transmitted bits, causal influence, or the importance of a token to the final answer.

### Chain of thought

Chain of thought (CoT) increases the number of generated tokens, the amount of externally represented intermediate state, and test-time compute. That can improve performance on difficult tasks. A channel metaphor—“the model receives more sequential uses of an external scratch medium”—may be a useful research intuition. It is not yet a theorem that CoT raises the model's Shannon capacity. Module 13 will return to this boundary in detail.

## 7.11. Formalizing a channel in an ML system

Channel language is most useful not when it produces a striking metaphor, but when it forces us to specify the experiment. Before computing MI or capacity, it helps to fill in a short specification.

| Element | Question |
|---|---|
| Input | Which random variable is transmitted? |
| Output | What does the receiver or next model observe? |
| Transition law | Where does randomness enter, and how is $W(y\mid x)$ defined? |
| Constraint | What is limited: power, length, tokens, bits, or latency? |
| Code and decoder | How are messages mapped to inputs and recovered? |
| Criterion | Block error, MSE, top-1 accuracy, or semantic utility? |
| Regime | Fixed $n$ or asymptotics? Actual $P_X$ or maximization over $P_X$? |

Compare two settings.

| System | What is precisely defined | Which question can be asked |
|---|---|---|
| Noisy labels | $Y$, $\widetilde Y$, and transition matrix $T$ | actual $I(Y;\widetilde Y)$; after optimizing class frequencies, the matrix capacity |
| LLM with fixed sampling | prompt $X$, response $Y$, law $q_\theta(y\mid x)$ | $I(X;Y)$ under a prompt distribution; capacity only after a coding task and cost are specified |

Two distinctions are worth carrying forward:

$$
\boxed{
\text{capacity}
\ne
\text{MI at a fixed input distribution}
\ne
\text{tensor or window size}
}
$$

and

$$
\boxed{
\text{channel theorem}
\ne
\text{empirical result}
\ne
\text{architectural analogy}
}.
$$

This is not a ban on analogies. It is how they become productive: first identify which part of the construction is a genuine channel, then mark clearly where the heuristic begins.

## 7.13. Conclusion

In Module 6, a probabilistic model became a bill in bits. In Module 7, mutual information acquired a second operational meaning: after optimizing the input, it gives the limiting reliable communication rate through noise.

$$
\boxed{
C=\max_{P_X}I(X;Y)
}.
$$

A complete construction surrounds the short formula.

- The channel specifies a transition law and a resource constraint.
- The codebook maps messages into long, well-separated sequences.
- Below $C$, random coding proves the existence of codes with vanishing error.
- Above $C$, the converse forbids reliable communication for every code.
- At finite $n$, capacity alone is not enough: target error and channel dispersion matter.

The main conceptual shift is that noise stops being a local defect that must be repaired one symbol at a time. A code acts on the geometry of the whole block. Individual channel uses still make errors, yet the complete message can remain reliably distinguishable.

For ML, this idea has two levels. In noisy labels, ECOC, and DeepJSCC, the channel is part of a precise model or algorithm. In discussions of context, attention, and CoT, channel language is usually a disciplined analogy. In both cases the same question is useful: **can we explicitly write the input, output, transition law, resource constraint, and decoding task?**

This completes the classical Shannon core. Source coding removes predictable redundancy; channel coding adds redundancy designed to survive noise; mutual information connects both sides to exact limits.

Module 8 moves from describing sources and channels to a normative question: which distribution should we choose when only constraints are known? The answer will be the maximum-entropy principle.

## Primary sources and further reading

1. C. E. Shannon, [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), 1948.
2. T. Cover, J. Thomas, *Elements of Information Theory*, 2nd ed., 2006, Chapters 7–9.
3. Y. Polyanskiy, Y. Wu, *Information Theory: From Coding to Learning*, 2025, chapters on channel coding and finite blocklength.
4. Y. Polyanskiy, H. V. Poor, S. Verdú, [*Channel Coding Rate in the Finite Blocklength Regime*](https://people.lids.mit.edu/yp/homepage/data/finite_block.pdf), 2010.
5. R. Blahut, [*Computation of Channel Capacity and Rate-Distortion Functions*](https://doi.org/10.1109/TIT.1972.1054855), 1972; S. Arimoto, [*An Algorithm for Computing the Capacity of Arbitrary Discrete Memoryless Channels*](https://doi.org/10.1109/TIT.1972.1054753), 1972.
6. G. Patrini et al., [*Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf), 2017.
7. T. Dietterich, G. Bakiri, [*Solving Multiclass Learning Problems via Error-Correcting Output Codes*](https://arxiv.org/abs/cs/9501101), 1995.
8. E. Bourtsoulatze, D. B. Kurka, D. Gündüz, [*Deep Joint Source-Channel Coding for Wireless Image Transmission*](https://arxiv.org/abs/1809.01733), 2019.
9. N. F. Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://aclanthology.org/2024.tacl-1.9/), 2024.
10. S. Jain, B. C. Wallace, [*Attention is not Explanation*](https://arxiv.org/abs/1902.10186), 2019; S. Wiegreffe, Y. Pinter, [*Attention is not not Explanation*](https://arxiv.org/abs/1908.04626), 2019.
