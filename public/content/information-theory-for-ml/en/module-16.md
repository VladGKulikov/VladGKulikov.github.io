# Module 16. Course Reference Sheet

## 16.1. Course Reference Sheet: Key Ideas and Formulas

This is not another theory module. It is a working reference for three situations: recalling a definition, choosing the right information-theoretic object, and catching an overclaim before it turns into a modeling or experimental error.

You do not need to read it linearly. Return to the relevant block when the real question is: *what exactly am I measuring, relative to which distribution, and in which units?* Proofs, examples, code, and primary sources remain in Modules 1–15.

### Four diagnostic questions to ask first

Before using any information-theoretic formula, ask:

1. **Whose distribution appears inside the logarithm?** The data-generating law $P$, the model $Q_\theta$, a posterior, a policy, or a critic?
2. **Which distribution defines the average?** Data, model samples, a mixture, policy trajectories, or the randomness of the learning algorithm?
3. **What is the unit?** Bits, nats, bits per token, bits per byte, bits per pixel, bits per channel use, or total description length?
4. **What is the status of the quantity?** A population object, a finite-sample estimator, a training objective, a variational bound, or a physical bitstream?

Much of the confusion in ML comes not from difficult mathematics but from silently changing the answer to one of these questions.

### Notation and units

Here $P$ usually denotes the source or data distribution, and $Q$ denotes a model. Base-two logarithms produce bits; natural logarithms produce nats:

$$
\log_2 x=\frac{\ln x}{\ln 2}.
$$

If $p(x)>0$ but $q(x)=0$, the model has declared a possible event impossible, so the corresponding cross-entropy and KL divergence are infinite.

---

### 1. Surprisal, entropy, and the model's bill — M1–M3

#### Surprisal of one outcome

$$
s_P(x)=-\log_b p(x).
$$

The logarithm turns products of independent probabilities into additive information costs:

$$
s(pq)=s(p)+s(q).
$$

Rare events receive a large bill; likely events receive a small one. This bill says nothing by itself about semantic importance: an unlikely typo may be expensive in bits and still be meaningless.

#### Entropy

$$
H_b(X)
=
\mathbb E_{X\sim P}[-\log_b p(X)]
=
-\sum_x p(x)\log_b p(x).
$$

Entropy is the average bill for source uncertainty. For a finite alphabet of size $K$,

$$
0\le H_b(X)\le\log_b K,
$$

with equality at the upper end for the uniform distribution.

The chain rule is

$$
H(X,Y)=H(X)+H(Y\mid X),
$$

and for a sequence,

$$
H(X_{1:T})
=
\sum_{t=1}^{T}H(X_t\mid X_{<t}).
$$

This is the accounting identity behind autoregressive modeling: one global bill becomes a sequence of conditional bills.

#### Cross-entropy and KL

$$
H_b(P,Q)
=
\mathbb E_{X\sim P}[-\log_bq(X)],
$$

$$
D_{\mathrm{KL},b}(P\|Q)
=
\mathbb E_{X\sim P}
\left[
\log_b\frac{p(X)}{q(X)}
\right].
$$

The central decomposition is

$$
\boxed{
H_b(P,Q)=H_b(P)+D_{\mathrm{KL},b}(P\|Q)
}.
$$

$H(P)$ is irreducible source uncertainty. KL is the average surcharge for using the wrong probabilistic model.

For conditional prediction,

$$
\mathcal L(\theta)
=
\mathbb E_P[-\log q_\theta(Y\mid X)]
=
H_P(Y\mid X)
+
\mathbb E_X
D_{\mathrm{KL}}
\bigl(
P(\cdot\mid X)\|Q_\theta(\cdot\mid X)
\bigr).
$$

Maximum likelihood can reduce the mismatch term, but it cannot remove uncertainty inherent in the data distribution.

#### Autoregressive NLL and perplexity

$$
-\log q_\theta(x_{1:T})
=
\sum_{t=1}^{T}
-\log q_\theta(x_t\mid x_{<t}).
$$

An average log loss $\bar L_b$ becomes perplexity through

$$
\operatorname{PPL}_b=b^{\bar L_b}.
$$

Perplexity is a monotone re-expression of average NLL, not a separate measure of truth, reasoning, or factual correctness.

#### Softmax cross-entropy gradient

If $q=\operatorname{softmax}(z)$ and $r$ is the target distribution, then

$$
\boxed{\nabla_z\ell=q-r}.
$$

The model increases logits where target mass exceeds model mass and decreases them where it has allocated too much probability.

#### Do not conflate

- **Predictive entropy** $H(Q_\theta(\cdot\mid x))$ asks how spread out the model is before the answer is observed.
- **NLL** $-\log q_\theta(y\mid x)$ asks how much probability the model assigned to the realized answer.
- **Source entropy** $H_P(Y\mid X)$ belongs to the true data distribution.

A confidently wrong prediction can have low predictive entropy and enormous NLL.

---

### 2. Averaging through a nonlinearity: Jensen and variational bounds — M4

For convex $\phi$,

$$
\phi(\mathbb E[X])\le\mathbb E[\phi(X)].
$$

For concave functions the direction reverses. This is a reliable compass: identify curvature first, then decide whether averaging happens before or after the nonlinear map.

#### Probability ensembles

Because $-\log$ is convex,

$$
-\log\left(\sum_j\lambda_jq_j(y)\right)
\le
\sum_j\lambda_j[-\log q_j(y)].
$$

An arithmetic mixture of probabilities is no worse than the average member in per-example log loss. This does not automatically improve accuracy, calibration, or robustness under distribution shift.

#### ELBO

For a latent-variable model,

$$
\log p_\theta(x)
=
\operatorname{ELBO}(x)
+
D_{\mathrm{KL}}
\bigl(
q_\phi(z\mid x)\|p_\theta(z\mid x)
\bigr),
$$

where

$$
\operatorname{ELBO}(x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
$$

The ELBO is a lower bound on log evidence, and its gap is known exactly: posterior KL.

#### Coarse-graining

If a deterministic or stochastic map discards detail, KL distinguishability cannot increase:

$$
D_{\mathrm{KL}}(P_Y\|Q_Y)
\le
D_{\mathrm{KL}}(P_X\|Q_X).
$$

This is an early instance of a recurring principle: processing can reorganize information but cannot manufacture distinguishability from nothing.

---

### 3. Mutual information: the value of an observation — M5

$$
\boxed{
I(X;Y)
=
D_{\mathrm{KL}}(P_{XY}\|P_XP_Y)
=
H(Y)-H(Y\mid X)
}.
$$

Mutual information has three equivalent readings:

- reduction in population-optimal log loss after observing $X$;
- divergence from the world in which $X$ and $Y$ are independent;
- average strength of statistical dependence.

Conditional mutual information is

$$
I(X;Y\mid Z)
=
H(Y\mid Z)-H(Y\mid X,Z).
$$

It asks what $X$ adds about $Y$ after $Z$ is already known. This is why CMI captures redundancy and synergy that one-feature-at-a-time selection can miss.

#### Data processing inequality

If

$$
Y\longrightarrow X\longrightarrow T,
$$

then

$$
I(Y;T)\le I(Y;X).
$$

An encoder may make existing target information easier for a restricted classifier to extract without creating new bits about the target.

#### InfoNCE

With one positive candidate and $N-1$ independent negatives,

$$
I(X;Y)
\ge
\log N-\mathcal L_{\mathrm{NCE}}.
$$

The $\log N$ ceiling belongs to this particular lower bound, not to population MI and not to representation quality in general.

#### Estimating MI

Keep these three objects separate:

$$
\text{population MI}
\ne
\text{finite-sample estimate}
\ne
\text{training proxy}.
$$

Plug-in, nearest-neighbor, and neural estimators have different bias, variance, and failure modes. A numerical MI value without an estimation protocol is rarely meaningful.

---

### 4. From probabilities to bits and from messages to channels — M6–M7

#### Kraft inequality

For binary prefix-code lengths,

$$
\sum_i2^{-\ell_i}\le1.
$$

This is the budget of a binary tree: a short codeword occupies a large share of the available space.

#### Source coding theorem

For an optimal one-symbol prefix code,

$$
H_2(X)\le L^*<H_2(X)+1.
$$

For a code tuned to model $Q$, the ideal expected length is

$$
\mathbb E_P[-\log_2q(X)]
=
H_2(P)+D_{\mathrm{KL},2}(P\|Q).
$$

An arithmetic or range coder turns sequential conditional probabilities into an actual reversible stream, but finite precision, termination, headers, and metadata add overhead.

Therefore distinguish

$$
L_{\mathrm{ideal}},
\qquad
L_{\mathrm{stream}},
\qquad
L_{\mathrm{total}}.
$$

The last quantity may include the model, tokenizer, and decoding protocol.

#### AEP and entropy rate

For an i.i.d. source,

$$
-\frac1n\log_2P(X^n)
\longrightarrow
H_2(X).
$$

For a stationary source with memory, the central object is entropy rate:

$$
h
=
\lim_{n\to\infty}\frac1nH(X_{1:n}).
$$

#### Channel capacity

For a discrete memoryless channel,

$$
\boxed{C=\max_{P_X}I(X;Y)}.
$$

$I(X;Y)$ at a fixed input distribution is one operating point. Capacity additionally optimizes over admissible input laws.

Canonical examples are

$$
C_{\mathrm{BSC}}=1-h_2(\varepsilon),
$$

$$
C_{\mathrm{BEC}}=1-\varepsilon,
$$

$$
C_{\mathrm{AWGN}}
=
\frac12\log_2\left(1+\frac{P}{\sigma^2}\right).
$$

Reliable communication is asymptotically possible at $R<C$ and impossible with vanishing error at $R>C$.

For joint source and channel coding with $\kappa$ channel uses per source symbol, separation requires

$$
R(D)<\kappa C.
$$

A context window, attention matrix, or hidden state is not automatically a channel capacity. Messages, admissible inputs, noise, cost, a unit of channel use, and an error criterion must first be defined.

---

### 5. MaxEnt, exponential families, and exponential tilting — M8

#### Relative maximum entropy

Given a reference measure $m$ and linear expectation constraints, solve the information projection

$$
q^*
=
\arg\min_{q\in\mathcal C}
D_{\mathrm{KL}}(q\|m).
$$

The solution has the form

$$
\boxed{
q_\lambda(x)
=
\frac{m(x)e^{\lambda^\top f(x)}}{Z(\lambda)}
}.
$$

Exponential tilting is a recurring mechanism: redistribute reference mass toward high-score states, then pay the normalization cost.

#### Exponential family

$$
p_\eta(x)
=
h(x)
\exp\left(
\eta^\top T(x)-A(\eta)
\right).
$$

For a regular family,

$$
\nabla A(\eta)=\mathbb E_\eta[T(X)],
$$

$$
\nabla^2A(\eta)
=
\operatorname{Cov}_\eta(T(X)).
$$

The gradient of the log-partition function gives mean sufficient statistics; its Hessian gives their covariance and Fisher information in natural coordinates.

#### Softmax as an exact optimum

$$
\boxed{
\operatorname{softmax}(z/\tau)
=
\arg\max_{q\in\Delta}
\left\{
q^\top z+\tau H(q)
\right\}
}.
$$

With entropy in nats, the exact optimality gap is

$$
\max_{u\in\Delta}
\{u^\top z+\tau H(u)\}
-
\{q^\top z+\tau H(q)\}
=
\tau D_{\mathrm{KL}}(q\|q^*).
$$

#### Energy-based model

$$
q_\theta(x)
=
\frac{e^{-E_\theta(x)}}{Z_\theta}.
$$

Its NLL gradient is

$$
\nabla_\theta[-\log q_\theta(x)]
=
\nabla_\theta E_\theta(x)
-
\mathbb E_{q_\theta}
[\nabla_\theta E_\theta(X)].
$$

The positive phase lowers data energy; the negative phase raises energy for typical model samples.

#### KL-regularized policy

The objective

$$
\max_\pi
\left\{
\mathbb E_\pi[r]
-
\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
\right\}
$$

has the Gibbs optimum

$$
\boxed{
\pi^*(y\mid x)
=
\frac{
\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}
}{Z(x)}
}.
$$

KL prices movement away from the reference policy; it does not certify the reward itself.

Along the exponential path $\pi_\alpha\propto\pi_0e^{\alpha r_{\mathrm{proxy}}}$,

$$
\frac{d}{d\alpha}
\mathbb E_{\pi_\alpha}[g]
=
\operatorname{Cov}_{\pi_\alpha}
(g,r_{\mathrm{proxy}}).
$$

Proxy reward increases because its derivative is its variance. True quality increases only while its covariance with the proxy remains positive.

---

### 6. Useful forgetting: rate–distortion, IB, and VAE — M9

#### Rate–distortion

$$
\boxed{
R(D)
=
\min_{P_{\widehat X\mid X}:
\mathbb E[d(X,\widehat X)]\le D}
I(X;\widehat X)
}.
$$

$R(D)$ asks how much information must be retained to keep expected distortion below $D$. The answer depends on the distortion function; information becomes “irrelevant” only relative to a task and an error criterion.

#### Information Bottleneck

$$
\mathcal L_{\mathrm{IB}}
=
I(X;T)-\beta I(T;Y).
$$

Under $Y-X-T$,

$$
I(X;Y)-I(T;Y)
=
\mathbb E_{X,T}
D_{\mathrm{KL}}
\left(
P(Y\mid X)\|P(Y\mid T)
\right).
$$

A representation is compressed relative to the task when it merges inputs without materially changing what can be predicted about $Y$.

#### VIB and VAE

In VIB, KL to a chosen prior upper-bounds the rate:

$$
\mathbb E_X
D_{\mathrm{KL}}
\bigl(q_\phi(T\mid X)\|r(T)\bigr)
=
I(X;T)
+
D_{\mathrm{KL}}
\bigl(q_\phi(T)\|r(T)\bigr).
$$

For a VAE,

$$
-\operatorname{ELBO}(x)
=
\underbrace{
\mathbb E_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
}_{\text{distortion}}
+
\underbrace{
D_{\mathrm{KL}}
\bigl(q_\phi(z\mid x)\|p(z)\bigr)
}_{\text{variational rate}}.
$$

A standard VAE is a rate–distortion system for modeling $X$, but it is not identical to supervised IB with an external relevance variable $Y$.

#### I–MMSE

For the Gaussian channel

$$
Y_\gamma=\sqrt\gamma X+N,
\qquad
N\sim\mathcal N(0,1),
$$

mutual information in nats satisfies

$$
\boxed{
\frac{d}{d\gamma}I(X;Y_\gamma)
=
\frac12\operatorname{mmse}(\gamma)
}.
$$

The rate at which information accumulates with SNR equals half the optimal estimation error. This is a powerful bridge to denoising and diffusion, not a complete choice of architecture or noise schedule.

---

### 7. Individual descriptions and MDL — M10

Kolmogorov complexity of one string is

$$
K_U(x)
=
\min_{p:\,U(p)=x}|p|.
$$

It depends on the universal machine only up to an additive constant, but is uncomputable in general.

Most length-$n$ strings are incompressible:

$$
\Pr\{K(X)<n-c\}<2^{-c}
$$

for uniform $X\in\{0,1\}^n$.

For a computable distribution $P$, Shannon and algorithmic descriptions meet on average:

$$
H_2(P)
\le
\mathbb E_P[K(X)]
\le
H_2(P)+K(P)+O(1).
$$

Minimum description length uses

$$
L(h,D)=L(h)+L(D\mid h).
$$

A model pays both for its own description and for the data residual it fails to explain. Description length is still not semantic value, computational cost, or practical compressor output.

---

### 8. Comparing implicit distributions — M11

The general $f$-divergence is

$$
D_f(P\|Q)
=
\mathbb E_Q
\left[
f\left(\frac{p(X)}{q(X)}\right)
\right].
$$

With different convex generators it includes KL, total variation, $\chi^2$, squared Hellinger, and Jensen–Shannon.

#### Classifier-based density-ratio estimation

If samples from $P$ have prior probability $\pi$, then the optimal discriminator obeys

$$
\operatorname{logit}D^*(x)
=
\log\frac{p(x)}{q(x)}
+
\log\frac{\pi}{1-\pi}.
$$

Strong source classification does not by itself guarantee a numerically accurate density ratio; calibrated probabilities are also required.

#### Variational representation

Under suitable conditions,

$$
D_f(P\|Q)
=
\sup_T
\left\{
\mathbb E_P[T]
-
\mathbb E_Q[f^*(T)]
\right\}.
$$

In practice, distinguish the exact supremum, the best critic in a restricted class, and the critic obtained from finite-sample training.

For the original minimax GAN with an optimal discriminator,

$$
\max_DV(D,Q)
=
-\log4+2D_{\mathrm{JS}}(P_{\mathrm{data}},Q).
$$

This is a population identity, not a description of each finite optimization step.

When supports are separated, KL and JS may provide poor optimization geometry. Integral probability metrics offer different structures: Wasserstein measures transport cost, while MMD compares RKHS mean embeddings.

---

### 9. Adaptivity and generalization — M12

#### PAC-Bayes-kl

For bounded loss and a valid prior $P$ independent of the certification sample, with probability at least $1-\delta$,

$$
\boxed{
\operatorname{kl}
\bigl(\widehat L_S(Q)\|L(Q)\bigr)
\le
\frac{
D_{\mathrm{KL}}(Q\|P)
+
\ln\dfrac{2\sqrt n}{\delta}
}{n}
}.
$$

The guarantee is for the Gibbs predictor $W\sim Q$, not automatically for a network with averaged weights.

A generalized Gibbs distribution minimizes empirical risk plus a KL price:

$$
Q_\lambda(dw)
\propto
P(dw)e^{-\lambda n\widehat L_S(w)}.
$$

#### MI generalization bound

For $\sigma^2$-sub-Gaussian losses,

$$
\left|
\mathbb E[L(W)-\widehat L_S(W)]
\right|
\le
\sqrt{
\frac{2\sigma^2I(W;S)}{n}
}.
$$

This controls an expected gap over the sample and algorithmic randomness; it is not a high-probability certificate for one realized run.

PAC-Bayes and MI meet in the exact average decomposition

$$
\boxed{
\mathbb E_S
D_{\mathrm{KL}}(Q_S\|P)
=
I(W;S)
+
D_{\mathrm{KL}}(P_W\|P)
}.
$$

MI measures dependence of the learned result on the sample. The second term measures mismatch between the chosen prior and the algorithm's marginal output distribution.

---

### 10. Reasoning and test-time computation — M13

Suppose an internal scratchpad is generated as

$$
C=G_\theta(X,U),
\qquad
U\perp Y^*\mid X.
$$

Then

$$
\boxed{I(Y^*;C\mid X)=0}.
$$

An internally generated chain adds no new fact about the correct answer beyond the full input for an ideal observer. It can still help as sequential computation, working memory, or search.

If the system receives an external observation $O$,

$$
I(Y^*;X,O)
=
I(Y^*;X)+I(Y^*;O\mid X),
$$

and the final term may be positive.

For $N$ independent candidates, each correct with probability $p$, an ideal verifier obtains

$$
P_{\mathrm{oracle}}(N)=1-(1-p)^N.
$$

Real performance depends on both generator coverage and selection quality. Wider search can find more correct candidates while also creating more opportunities to exploit verifier errors.

For arbitrary non-nested states,

$$
H(Y\mid S_{j-1})-H(Y\mid S_j)
=
I(Y;S_j\mid S_{j-1})
-
I(Y;S_{j-1}\mid S_j).
$$

State improvement is a balance between newly available relevant information and relevant information discarded when the previous state is replaced.

---

### 11. Computer vision: four different budgets — M14

#### Learned image compression

A typical learned codec optimizes

$$
\mathcal L=R+\lambda D,
$$

where $R$ is expected code length for quantized latents and side information, and $D$ is an explicitly chosen distortion.

For a hyperprior,

$$
R
=
\mathbb E[-\log_2p_\phi(\widehat Z)]
+
\mathbb E[-\log_2p_\psi(\widehat Y\mid\widehat Z)].
$$

Side information is useful only when the savings in the main latent code exceed its own transmission cost.

#### Denoising and score

For

$$
X_t=\alpha_tX_0+\sigma_t\varepsilon,
$$

the MSE-optimal denoiser is the posterior mean:

$$
f^*(x_t)=\mathbb E[X_0\mid X_t=x_t].
$$

The score of the noisy marginal is

$$
\nabla_{x_t}\log p_t(x_t)
=
\frac{
\alpha_t\mathbb E[X_0\mid x_t]-x_t
}{\sigma_t^2}.
$$

This is an exact bridge from denoising to score estimation, not a complete derivation of a diffusion architecture.

#### Keep four budgets separate

- a data codec is measured in actual bits or bits per pixel;
- a representation is judged by information useful for a chosen task;
- a noisy observation limits estimation quality;
- a deployed model pays in storage, bit width, memory bandwidth, and latency.

---

### 12. Information geometry — M15

#### Fisher as the local form of KL

For a smooth family $p_\theta$,

$$
D_{\mathrm{KL}}
\bigl(
p_\theta\|p_{\theta+d\theta}
\bigr)
=
\frac12d\theta^\top G(\theta)d\theta
+o(\|d\theta\|^2),
$$

where

$$
G(\theta)
=
\mathbb E_{p_\theta}
\left[
\nabla_\theta\log p_\theta(X)
\nabla_\theta\log p_\theta(X)^\top
\right].
$$

Fisher measures local statistical distinguishability; it does not turn globally asymmetric KL into a metric distance.

#### Natural gradient

The locally steepest decrease under a small KL budget points along

$$
\widetilde\nabla_\theta L
=
G(\theta)^\dagger\nabla_\theta L.
$$

This vector field is coordinate invariant to first order. A finite step, damping, and an approximate Fisher reintroduce implementation dependence.

#### Two geometries

Fisher–Rao asks how distinguishable neighboring distributions are. Wasserstein asks how much it costs to transport probability mass through observation space:

$$
W_2^2(P,Q)
=
\inf_{\gamma\in\Pi(P,Q)}
\mathbb E_{(X,Y)\sim\gamma}\|X-Y\|^2.
$$

No single geometry is universally correct. Choose the object first: predictive distribution, policy, image density, or physical mass flow.

For an exponential family,

$$
G(\eta)=\nabla^2A(\eta),
$$

and KL is the Bregman divergence of the log-partition function:

$$
D_{\mathrm{KL}}(p_\eta\|p_\xi)
=
A(\xi)-A(\eta)
-
\nabla A(\eta)^\top(\xi-\eta).
$$

---

### 13. A quick lookup table

| Question | Primary object | What it does not promise |
|---|---|---|
| How spread out is the model itself? | predictive entropy | truth or calibration |
| What bill did the model receive on data? | NLL / cross-entropy | source entropy without additional assumptions |
| What is the cost of model mismatch? | KL | symmetry or metric geometry |
| What does one variable reveal about another? | MI / CMI | causal direction |
| How many bits does a source require? | entropy, entropy rate, actual code length | model and protocol cost unless counted |
| How fast can messages pass through noise? | channel capacity | an answer without channel and input constraints |
| Which distribution should satisfy partial information? | MaxEnt / I-projection | correctness of the constraints themselves |
| How much information must be retained at allowed error? | rate–distortion / IB | a universal distortion function |
| How can implicit distributions be compared from samples? | variational $f$-divergence, MMD, Wasserstein | accuracy with a weak critic or small sample |
| How should adaptive model choice be paid for? | PAC-Bayes KL or $I(W;S)$ | the same type of guarantee |
| What do extra reasoning tokens buy? | sequential computation, search, external observations | automatic creation of new information |
| What counts as a small model update? | Fisher/KL, Bregman, or Wasserstein geometry | one geometry for every object |

---

### 14. Common overclaims

| Overcompressed claim | Precise version |
|---|---|
| Low entropy means the answer is correct | Low entropy means the model distribution is concentrated |
| NLL is the entropy of language | NLL estimates data-model cross-entropy under a particular protocol |
| KL is a distance | KL is asymmetric; its local quadratic form defines Fisher geometry |
| An encoder creates target information | Without external data it can only preserve, lose, or reorganize it |
| InfoNCE measures all of MI | It gives one lower bound under one sampling scheme |
| A VAE is the same as supervised IB | They share rate–distortion structure but use different relevance variables |
| KL prevents Goodhart | KL limits distribution shift but does not repair a misspecified reward |
| A longer chain of thought adds new facts | Internal CoT adds computation; new facts require external observations |
| A strong classifier accurately estimates $p/q$ | Density-ratio estimation also requires calibrated probabilities |
| Better compression equals intelligence | Compression exactly measures predictive log loss and may correlate with some capabilities |
| PPO clipping guarantees small KL | Clipping controls a local surrogate on sampled actions |
| One information quantity explains all of ML | Variables, protocol, constraints, and guarantee type must be specified first |

---

### 15. Seven mechanisms that connect the course

1. **Logarithms turn probability products into additive costs.** This gives surprisal, NLL, code length, and autoregressive factorization.
2. **KL isolates mismatch cost.** It appears in cross-entropy, variational inference, MaxEnt, RLHF, PAC-Bayes, and information geometry—but between different objects each time.
3. **Jensen controls the order of averaging and nonlinearity.** Ensemble guarantees, ELBO, log-sum, and many variational bounds grow from it.
4. **MI measures the predictive value of an observation.** CMI measures added value, DPI limits processing, and InfoNCE supplies one trainable lower bound.
5. **Coding gives probabilities an operational meaning.** A probabilistic model becomes a compressor only together with an entropy coder and a shared decoding protocol.
6. **Exponential tilting is a universal constrained update.** It unifies MaxEnt, exponential families, softmax, and KL-regularized policies.
7. **Information does not replace the task, computation, or semantics.** A formula becomes useful only after variables, quality criterion, available observations, and resource constraints are fixed.

With this map in place, the formulas stop looking like a collection of tricks. They become different answers to one recurring question:

> **How should we measure, transmit, preserve, compare, or redistribute uncertainty under explicit constraints?**

This reference sheet introduces no new results. Primary sources and full conditions remain in the reference lists of Modules 1–15 next to the corresponding topics.
