# Module 15. Information Geometry: Fisher, Natural Gradient, and Distribution Flows

> **How to read this module.** This is the final bonus module of the course. The main route is §§15.1–15.8: why ordinary gradient descent depends on coordinates, how Fisher information defines the local geometry of distributions, where the natural gradient comes from, which curvature approximations are actually used in deep networks, how this geometry appears in TRPO, PPO, and RLHF, and why moving probability mass requires a different geometry—Wasserstein geometry. §15.9 is a mathematical deepening on α-geometry, exact geodesics, and Chentsov’s theorem. It can be saved for a second pass.

## 15.1. What Does a “Small Step” Mean for a Probabilistic Model?

An optimizer moves parameters. What we usually care about, however, is not the parameter vector itself but the distribution, function, or policy that it represents.

Start with the smallest possible probabilistic model: a Bernoulli distribution. We may parameterize the event probability directly as

$$
p\in(0,1)
$$

or through a logit

$$
z=\log\frac{p}{1-p},
\qquad
p=\sigma(z).
$$

These are two coordinate systems for the same statistical object. If we take an ordinary gradient step in $p$ and then solve the same problem in $z$, the resulting probability trajectories generally differ. The algorithm changed merely because we renamed the coordinate.

There is a more physical issue as well. The same numerical probability increment does not represent the same distributional change everywhere. The move

$$
0.50\longrightarrow0.51
$$

has

$$
D_{\mathrm{KL}}
\bigl(
\operatorname{Bern}(0.50)
\|\operatorname{Bern}(0.51)
\bigr)
\approx 2.00\cdot10^{-4}
$$

nats. The move

$$
0.01\longrightarrow0.02
$$

has the same Euclidean step length $0.01$, but its KL cost is

$$
D_{\mathrm{KL}}
\bigl(
\operatorname{Bern}(0.01)
\|\operatorname{Bern}(0.02)
\bigr)
\approx3.12\cdot10^{-3}
$$

nats—more than fifteen times larger. The first move slightly perturbs an almost maximally uncertain coin. The second doubles the probability of a rare event.

In a deep network the same problem is hidden inside millions of coordinates:

- permuting neurons may leave the represented function unchanged;
- scales of adjacent layers may compensate for one another;
- softmax is unchanged when the same constant is added to all logits;
- the same Euclidean step in weight space may barely affect the output distribution or may radically reshape it.

The central question of information geometry is therefore:

> **How should we measure model change in terms of the probabilistic object itself rather than an arbitrary coordinate system?**

There is no single “correct geometry for all of ML.” Three different constructions will appear in this module.

1. **Fisher–Rao geometry and local KL** measure the distinguishability of nearby probabilistic models.
2. **Bregman divergences and mirror descent** define useful update geometries on the simplex and in exponential families.
3. **Wasserstein geometry** measures the cost of transporting mass through the observation space.

![](assets/information-theory-for-ml/en/module-15/M15_big_picture_EN.png)

A useful compact statement is:

> **A geometry is a contract about which changes count as small.**

For a policy, smallness may mean a small average KL between action distributions. For a model that transports noise into data, it may mean a small displacement in pixel or feature space. For ordinary deterministic regression, Fisher may not be the relevant object at all.

We begin with the geometry of neighboring distributions.

## 15.2. Fisher Information and the Local Form of KL

Let a model define a smooth family of distributions

$$
\mathcal M
=
\left\{
 p_\theta(x):
 \theta\in\Theta\subseteq\mathbb R^d
\right\}.
$$

The vector

$$
s_\theta(x)
=
\nabla_\theta\log p_\theta(x)
$$

is the parameter score function. Under standard regularity conditions its mean is zero:

$$
\mathbb E_{p_\theta}[s_\theta(X)]
=
\int \nabla_\theta p_\theta(x)\,dx
=
\nabla_\theta1
=0.
$$

The Fisher information matrix is

$$
\boxed{
G(\theta)
=
\mathbb E_{X\sim p_\theta}
\left[
 s_\theta(X)s_\theta(X)^\top
\right]
}.
$$

For any direction $v$, its quadratic form has a transparent interpretation:

$$
v^\top G(\theta)v
=
\operatorname{Var}_{p_\theta}
\left[
 v^\top s_\theta(X)
\right].
$$

If this number is large, moving along $v$ noticeably changes the log probabilities of typical observations. Neighboring models are easier to distinguish in that direction. If the number is zero, a small move along $v$ does not change the represented distribution to first order.

### Why Fisher Appears from KL

Consider a neighboring model $p_{\theta+d\theta}$. Expand its log density around $\theta$:

$$
\log p_{\theta+d\theta}(x)
=
\log p_\theta(x)
+s_\theta(x)^\top d\theta
+
\frac12
 d\theta^\top
 \nabla_\theta^2\log p_\theta(x)
 d\theta
+
o(\|d\theta\|^2).
$$

Insert this expansion into

$$
D_{\mathrm{KL}}
\bigl(
 p_\theta
 \|p_{\theta+d\theta}
\bigr)
=
\mathbb E_{p_\theta}
\left[
 \log p_\theta(X)
 -\log p_{\theta+d\theta}(X)
\right].
$$

The linear term vanishes because the score has zero mean. When differentiation can be exchanged with integration,

$$
-\mathbb E_{p_\theta}
\left[
 \nabla_\theta^2\log p_\theta(X)
\right]
=
G(\theta).
$$

Hence

$$
\boxed{
D_{\mathrm{KL}}
\bigl(
 p_\theta
 \|p_{\theta+d\theta}
\bigr)
=
\frac12
 d\theta^\top G(\theta)d\theta
+
o(\|d\theta\|^2)
}.
$$

Read this formula in words:

> **Fisher information is the second-order cost of a small distributional change measured by KL.**

Forward and reverse KL share the same quadratic term. Their asymmetry appears at higher orders. Fisher therefore defines a local Riemannian geometry, but it does not turn KL itself into a global metric: KL remains asymmetric and does not satisfy the triangle inequality.

![](assets/information-theory-for-ml/en/module-15/M15_local_kl_natural_EN.png)

### Why the Length Is Coordinate Independent

Suppose the same model is written in new coordinates $\phi$, with $\theta=\theta(\phi)$ and

$$
J=\frac{\partial\theta}{\partial\phi}.
$$

Then

$$
G_\phi
=
J^\top G_\theta J,
$$

while

$$
d\theta=J\,d\phi.
$$

Therefore

$$
d\theta^\top G_\theta d\theta
=
d\phi^\top G_\phi d\phi.
$$

The entries of the matrix change, but the local statistical length does not. This is what it means for Fisher to transform as a metric tensor rather than as an arbitrary preconditioner.

### An Exact Example: Bernoulli

For $X\sim\operatorname{Bernoulli}(p)$ in the $p$ coordinate,

$$
G_p(p)=\frac1{p(1-p)}.
$$

In logit coordinates,

$$
G_z(z)=p(1-p).
$$

The two expressions look opposite: one diverges near the edge of the simplex, while the other tends to zero. Yet the line element is the same:

$$
\frac{dp^2}{p(1-p)}
=
p(1-p)\,dz^2.
$$

The coordinate

$$
u(p)=2\arcsin\sqrt p
$$

straightens this one-dimensional geometry:

$$
ds^2=du^2.
$$

Thus the exact Fisher–Rao distance between two Bernoulli distributions is

$$
\boxed{
d_{\mathrm{FR}}(p_1,p_2)
=
2\left|
\arcsin\sqrt{p_2}
-
\arcsin\sqrt{p_1}
\right|
}.
$$

For $p_1=0.1$ and $p_2=0.9$,

$$
d_{\mathrm{FR}}
\approx1.85459.
$$

![](assets/information-theory-for-ml/en/module-15/M15_bernoulli_geometry_EN.png)

### The Same Matrix Limits Estimation Accuracy

For $n$ independent observations and an unbiased estimator under the usual regularity assumptions, the Cramér–Rao bound gives

$$
\operatorname{Cov}(\widehat\theta)
\succeq
\frac1nG(\theta)^{-1}.
$$

This is another side of the same idea. If neighboring distributions are easy to distinguish, the parameter can be estimated more precisely. If different parameter values barely alter the observation law, no unbiased estimator can recover high precision from the same data.

### Where the Local Picture Ends

Three boundaries matter.

- The quadratic approximation is reliable only for sufficiently small steps.
- Positive definiteness requires local identifiability. Symmetries and redundant parameters often make Fisher singular in deep networks.
- Fisher depends on which observation distribution and conditional model were declared to be the statistical object. “The Fisher matrix of the network” is incomplete until the expectations are specified.

The last point will become central in §15.4. First we derive the update induced by measuring step size with Fisher.

## 15.3. Natural Gradient: Steepest Descent Under Small Model Change

Suppose we minimize $L(\theta)$ and write

$$
g=\nabla_\theta L.
$$

The ordinary gradient is the steepest descent direction under an Euclidean constraint. If smallness is measured by local KL, the optimization problem becomes

$$
\min_\delta g^\top\delta
\qquad
\text{subject to}
\qquad
\frac12\delta^\top G\delta\le\varepsilon.
$$

The Lagrangian gives the direction

$$
\boxed{
\widetilde\nabla_\theta L
=G(\theta)^\dagger g
},
$$

where $G^\dagger$ is an inverse or pseudoinverse. If $G$ is nonsingular, the step that uses the full budget is

$$
\boxed{
\delta^*
=-
\sqrt{
\frac{2\varepsilon}{g^\top G^{-1}g}
}
G^{-1}g
}.
$$

Consider

$$
G=
\begin{pmatrix}
100&0\\
0&1
\end{pmatrix},
\qquad
g=
\begin{pmatrix}
1\\1
\end{pmatrix}.
$$

The Euclidean gradient moves equally in both coordinates. The natural-gradient direction is

$$
G^{-1}g
=
\begin{pmatrix}
0.01\\1
\end{pmatrix}.
$$

The first coordinate is one hundred times more sensitive in distribution space, so the same statistical budget permits a much smaller displacement there.

![](assets/information-theory-for-ml/en/module-15/M15_natural_gradient_EN.png)

### Coordinate Invariance in the Bernoulli Example

Let the objective depend on the probability: $L=L(p)$. In the $p$ coordinate, the natural-gradient flow is

$$
\frac{dp}{dt}
=-p(1-p)\frac{dL}{dp}.
$$

In logit coordinates,

$$
\frac{dL}{dz}
=
\frac{dL}{dp}p(1-p),
\qquad
G_z=p(1-p),
$$

so

$$
\frac{dz}{dt}
=-G_z^{-1}\frac{dL}{dz}
=-\frac{dL}{dp}.
$$

Transforming back,

$$
\frac{dp}{dt}
=p(1-p)\frac{dz}{dt}
=-p(1-p)\frac{dL}{dp}.
$$

Both coordinate systems describe the same continuous flow of distributions. Ordinary gradient flow does not have this property.

This is the strong positive result of natural gradient: **the vector field transforms correctly under a smooth one-to-one reparameterization of the statistical model.**

### Why a Finite Step Still Depends on the Implementation

A practical algorithm does not take an infinitesimal displacement. It takes an Euler step:

$$
\theta_{t+1}
=
\theta_t-
\rho G(\theta_t)^{-1}g_t.
$$

After a nonlinear coordinate change, the finite endpoints agree only to first order in $\rho$. Exact finite-step invariance would require geodesic integration or a compatible retraction.

Practical implementations therefore rely on

- small or adaptively chosen steps;
- checking the realized KL after the update;
- line search;
- damping;
- and an adequate approximation to the curvature matrix.

The damped step

$$
(G+\lambda I)^{-1}g
$$

is often numerically essential. Yet the added $\lambda I$ introduces an Euclidean component and therefore changes the original geometry. It is a useful engineering compromise, not a neutral detail.

### Natural Gradient Is Not Newton’s Method

Newton’s method uses the Hessian of the objective:

$$
\delta_{\mathrm N}
=-H_L^{-1}g.
$$

Natural gradient uses the Fisher matrix of the statistical model:

$$
\delta_{\mathrm{NG}}
=-G^{-1}g.
$$

They can coincide in special likelihood settings—for example at the level of an expected negative log likelihood under appropriate model specification. In neural networks, Fisher is also closely related to the generalized Gauss–Newton matrix for several exponential-family output distributions. But the empirical Hessian, model Fisher, and GGN differ in general. Martens gives a detailed account of these relationships and their optimization consequences. ([Martens, 2020](https://jmlr.org/papers/v21/17-678.html))

A useful summary is:

> **The inverse Fisher is not a magical accelerator. It changes the units of the step so that equal budgets correspond approximately to equal changes in the model distribution.**

## 15.4. What “Curvature” Is Actually Used in a Deep Network?

The full Fisher matrix of a model with billions of parameters cannot be stored or inverted directly. Before discussing approximations, however, we must decide which matrix is being approximated.

### Model Fisher

For a conditional model $p_\theta(y\mid x)$ and an input distribution $\rho(x)$,

$$
\boxed{
G_{\mathrm{model}}(\theta)
=
\mathbb E_{x\sim\rho}
\mathbb E_{y\sim p_\theta(\cdot\mid x)}
\left[
 g_\theta(x,y)g_\theta(x,y)^\top
\right]
},
$$

where

$$
g_\theta(x,y)
=
\nabla_\theta\log p_\theta(y\mid x).
$$

The crucial detail is that the inner expectation samples labels **from the current model**.

### Empirical Fisher

On a dataset one often computes

$$
\widehat G_{\mathrm{emp}}
=
\frac1n\sum_{i=1}^n
 g_\theta(x_i,y_i)g_\theta(x_i,y_i)^\top.
$$

Here the labels come from the data, not from $p_\theta(\cdot\mid x_i)$. This matrix is the second moment of observed per-example gradients, but in general it is not an unbiased estimator of the model Fisher and need not approximate the Hessian. Kunstner, Balles, and Hennig analyze this distinction and its pathologies in detail. ([Kunstner et al., 2019](https://arxiv.org/abs/1905.12558))

A small example shows the size of the mismatch. For a Bernoulli model in logit coordinates,

$$
G_{\mathrm{model}}
=p(1-p).
$$

If the data have true positive rate $q$, the population second moment of the observed gradient is

$$
G_{\mathrm{data}}
=q(1-p)^2+(1-q)p^2.
$$

At

$$
q=0.8,
\qquad
p=0.2,
$$

we obtain

$$
G_{\mathrm{model}}=0.16,
\qquad
G_{\mathrm{data}}=0.52.
$$

Both quantities are built from squared gradients, but they answer different questions.

### Generalized Gauss–Newton

Let the network output be $f_\theta(x)$ and the loss be $\ell(f,y)$. The generalized Gauss–Newton matrix is

$$
G_{\mathrm{GGN}}
=
\mathbb E
\left[
J_f^\top
H_{\ell,f}
J_f
\right],
$$

where $J_f$ is the Jacobian of the output with respect to parameters and $H_{\ell,f}$ is the Hessian of the loss with respect to the output. If $\ell$ is convex in $f$, GGN is positive semidefinite. For some probabilistic output models it coincides with the appropriate Fisher matrix, but this requires a specific output parameterization and loss.

### K-FAC, Adam, and Shampoo

**K-FAC** approximates large Fisher or GGN blocks by Kronecker products:

$$
G_{\mathrm{layer}}
\approx
A\otimes B.
$$

The factor $A$ describes input-activation statistics, while $B$ describes backpropagated-gradient statistics. This makes an approximate inverse much cheaper to apply, but only part of the correlation structure is retained and damping remains necessary. K-FAC was introduced explicitly as an efficient approximation to natural gradient descent. ([Martens & Grosse, 2015](https://arxiv.org/abs/1503.05671))

**Adam** tracks coordinatewise first and second moments of stochastic gradients and scales updates by roughly

$$
\frac1{\sqrt{v_t}+\epsilon}.
$$

This is an adaptive diagonal preconditioner. Squared gradients may resemble a diagonal empirical-Fisher statistic, but Adam does not invert the model Fisher and does not inherit full natural-gradient invariance.

**Shampoo** constructs structured second-moment matrices along tensor axes. It is related to full-matrix AdaGrad and preserves more structure than coordinatewise methods, but it is not automatically a Fisher approximation.

![](assets/information-theory-for-ml/en/module-15/M15_curvature_zoo_EN.png)

| Method | Object used | Structure retained | What it does not guarantee automatically |
|---|---|---|---|
| Newton | objective Hessian | local curvature of the objective | statistical reparameterization invariance |
| Natural gradient | model Fisher | local KL geometry | an exact global trust region |
| GGN | output-loss curvature | positive semidefiniteness for convex output losses | equality with Fisher in every task |
| Empirical Fisher | second moment of observed gradients | direct availability from data | correct Fisher geometry away from special regimes |
| Adam | coordinatewise gradient moments | cheap adaptive scaling | natural-gradient invariance |
| K-FAC | Kronecker blocks of Fisher/GGN | within-layer cross-coordinate structure | exact inversion of the full matrix |
| Shampoo | tensor-structured second moments | structure across parameter axes | a Fisher interpretation without extra assumptions |

That is the practical lesson: the word “curvature” does not identify one universal matrix. Each method decides which dependencies to retain, which to discard, and how much computation to spend on them.

## 15.5. Policy Geometry: NPG, TRPO, PPO, and KL in RLHF

A policy is itself a conditional distribution:

$$
\pi_\theta(a\mid s).
$$

The idea of measuring a small distributional change is therefore especially natural. For a chosen state distribution $d(s)$, the policy Fisher matrix is

$$
G_\pi(\theta)
=
\mathbb E_{s\sim d,
a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)^\top
\right].
$$

The natural policy gradient is

$$
\widetilde\nabla J
=G_\pi^\dagger\nabla J.
$$

It rescales the update so that sensitive directions in the action distribution move more cautiously.

### TRPO: A Trust Region in Policy Space

TRPO starts from a local surrogate objective and constrains the average policy change:

$$
\boxed{
\mathbb E_{s\sim d^{\pi_{\mathrm{old}}}}
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{old}}(\cdot\mid s)
\|\pi_\theta(\cdot\mid s)
\right)
\le\delta
}.
$$

Linearizing the objective and quadratically approximating the KL yields the same constrained problem as in §15.3, hence a natural-gradient direction. Practical TRPO uses conjugate gradients and line search rather than an exact geodesic. The original paper explicitly derives the practical method through a sequence of approximations to a theoretically motivated trust-region update. ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477))

Two levels should be kept separate:

- theoretical improvement bounds use stronger control of distribution shift;
- the practical algorithm controls an empirical average KL and checks the step numerically.

### PPO: A Useful Proxy, Not a Hard Trust Region

PPO uses the probability ratio

$$
r_t
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

and the clipped surrogate

$$
L^{\mathrm{clip}}
=
\mathbb E
\left[
\min
\left(
 r_tA_t,
 \operatorname{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t
\right)
\right].
$$

This supports multiple epochs of minibatch updates and often stabilizes optimization. Yet clipping controls ratios for actions observed in the sample, not the entire policy.

A three-action counterexample makes this explicit. Let

$$
\pi_{\mathrm{old}}
=(0.50,0.49,0.01),
$$

and suppose the batch contains only the first action. Consider

$$
\pi_q
=(0.50,q,0.50-q).
$$

The sampled-action ratio is always one, so the corresponding sampled surrogate is unchanged. But

$$
D_{\mathrm{KL}}
(\pi_{\mathrm{old}}\|\pi_q)
\supset
0.49\log\frac{0.49}{q}
\longrightarrow\infty
$$

as $q\to0$. Thus clipping is not a global KL constraint. Implementations commonly monitor approximate KL, stop early, or add an explicit penalty. PPO was proposed as a simpler first-order method with some practical benefits of TRPO, not as an exact solution to the same constrained problem. ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

![](assets/information-theory-for-ml/en/module-15/M15_policy_geometry_EN.png)

### Two Different KL Terms in RLHF

RLHF-style training can contain two distinct constraints.

1. **Stepwise KL** compares the current policy with the snapshot that generated the data for the present update. It limits the size of one optimization step.
2. **Fixed-reference KL** compares the current model with a fixed SFT or reference model. It limits accumulated drift over the whole run.

A compact way to remember the distinction is:

> **TRPO asks, “How far did we move this step?” Reference KL asks, “How far have we drifted from the starting model?”**

For a fixed context, the idealized problem

$$
\max_\pi
\left
\{
\mathbb E_\pi[r]
-
\beta
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
\right
\}
$$

has the exact Gibbs optimum

$$
\boxed{
\pi^*(y)
\propto
\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}
}.
$$

Module 8 gave the full derivation. Here the geometric reading matters: KL prices departure from the reference distribution, and the exponential tilt is the exact proximal solution of the idealized objective.

DPO uses this analytic relation to express preferences through policy-to-reference log ratios. It does not perform a natural-gradient or TRPO step on every minibatch. Shared KL algebra does not make the algorithms identical.

One more boundary: policy KL in RLHF compares response distributions, whereas PAC-Bayes KL in Module 12 compares distributions over hypotheses. The same formula does not transfer a generalization certificate from one setting to the other.

## 15.6. Wasserstein Geometry and Distribution Flows

Fisher geometry answers how distinguishable neighboring models are. Sometimes, however, the crucial structure lies in the observation space itself.

Consider two point masses:

$$
P=\delta_0,
\qquad
Q_\varepsilon=\delta_\varepsilon.
$$

For every $\varepsilon\ne0$, the supports are disjoint. Both directed KL divergences are infinite, total variation is one, and Jensen–Shannon is $\log2$, whether the second point is $10^{-6}$ or $10^6$ units away.

Wasserstein sees the relevant structure:

$$
W_2(P,Q_\varepsilon)=|\varepsilon|.
$$

It uses the geometry of the data space.

### Definition Through a Transport Plan

With quadratic cost,

$$
\boxed{
W_2^2(P,Q)
=
\inf_{\gamma\in\Pi(P,Q)}
\mathbb E_{(X,Y)\sim\gamma}
\|X-Y\|^2
}.
$$

A coupling $\gamma$ specifies which mass from $P$ is paired with which mass from $Q$. The ground metric is part of the problem. Rescaling one coordinate changes the transport cost.

![](assets/information-theory-for-ml/en/module-15/M15_two_geometries_EN.png)

### Dynamic Form

The Benamou–Brenier formula expresses the same distance as minimal kinetic action:

$$
W_2^2(P_0,P_1)
=
\inf_{p_t,v_t}
\int_0^1
\int
\|v_t(x)\|^2p_t(x)\,dx\,dt
$$

subject to the continuity equation

$$
\partial_tp_t
+
\nabla\cdot(p_tv_t)
=0.
$$

The point in the space is now a full distribution $p_t$, and the vector field $v_t$ transports its mass. Not every path between $P_0$ and $P_1$ is a geodesic; a geodesic minimizes the action.

### Continuous Normalizing Flows

Let a particle follow the ODE

$$
\frac{dx_t}{dt}
=v_t(x_t).
$$

Then its density evolves along the trajectory according to

$$
\boxed{
\frac{d}{dt}\log p_t(x_t)
=-\nabla\cdot v_t(x_t)
}.
$$

This is the continuous counterpart of the change-of-variables formula and underlies continuous normalizing flows. Neural ODEs showed how to train such models and compute likelihoods through the integrated divergence. ([Chen et al., 2018](https://arxiv.org/abs/1806.07366))

### Flow Matching: The Path Is Chosen First

Flow Matching does not begin by evaluating an unavailable log density. One chooses a family of intermediate distributions or conditional probability paths and trains a neural vector field by regression to the velocity that realizes the chosen evolution.

There are two central design freedoms:

- the coupling between initial and final points;
- the path used between each coupled pair.

The same endpoint marginals permit many couplings, and therefore many velocity fields and trajectories. Flow Matching is compatible with diffusion paths as well as paths based on optimal-transport displacement interpolation. This flexibility is part of the original method. ([Lipman et al., 2022](https://arxiv.org/abs/2210.02747))

![](assets/information-theory-for-ml/en/module-15/M15_flow_paths_EN.png)

In one-dimensional Gaussian transport the role of the coupling is especially clear. Let

$$
X_0=\mu_0+\sigma_0Z,
\qquad
X_1=\mu_1+\sigma_1Z
$$

share the same $Z\sim\mathcal N(0,1)$. The interpolation

$$
X_t=(1-t)X_0+tX_1
$$

is the Wasserstein geodesic between the Gaussian marginals. If $X_0$ and $X_1$ are sampled independently, the endpoints remain correct but the kinetic action is larger. A coupling is not bookkeeping; it determines the path geometry.

### Rectified Flow

Rectified Flow trains an ODE to follow straight lines between pairs drawn from a chosen coupling and may repeatedly “rectify” the resulting transport. The original work shows that rectification does not increase convex transport costs of the induced coupling and can produce trajectories that are easier to simulate with coarse discretization. ([Liu, Gong & Liu, 2022](https://arxiv.org/abs/2209.03003))

This does not imply that every trained rectified flow is the exact optimal-transport map or a global Wasserstein geodesic. Path straightness, coupling quality, regression error, and numerical integration error are distinct issues.

### Probability-Flow ODE and Diffusion

For the SDE

$$
dX_t
=f(X_t,t)dt+g(t)dW_t
$$

with scalar state-independent diffusion coefficient, the corresponding probability-flow ODE has drift

$$
\boxed{
\frac{dX_t}{dt}
=f(X_t,t)
-
\frac12g(t)^2
\nabla_x\log p_t(X_t)
}.
$$

With the exact score, the SDE and ODE have the same one-time marginals $p_t$. Their trajectories and couplings differ: one is stochastic and the other deterministic. This equivalent neural ODE is derived in the score-based SDE framework. ([Song et al., 2021](https://arxiv.org/abs/2011.13456))

Finally, distinguish two scores:

$$
\nabla_x\log p_t(x)
$$

is a gradient in observation space used in diffusion, whereas

$$
G(\theta)^{-1}\nabla_\theta L
$$

is a parameter-space natural gradient transformed by Fisher geometry. Both involve distributions and gradients, but they live in different spaces.

## 15.7. Exponential Families, Dual Coordinates, and Mirror Descent

Module 8 derived the exponential-family form

$$
p_\theta(x)
=h(x)
\exp\left(
\theta^\top T(x)-A(\theta)
\right).
$$

We can now read it geometrically.

The expectation parameters are

$$
\eta
=\nabla A(\theta)
=\mathbb E_\theta[T(X)],
$$

and the Fisher matrix is

$$
\boxed{
G(\theta)
=\nabla^2A(\theta)
=\operatorname{Cov}_\theta(T(X))
}.
$$

KL between two family members is a Bregman divergence of the log-partition function:

$$
\boxed{
D_{\mathrm{KL}}
(p_{\theta_1}\|p_{\theta_2})
=
B_A(\theta_2,\theta_1)
}.
$$

The order matters: the tangent to $A$ is taken at $\theta_1$ and evaluated at $\theta_2$. In dual coordinates,

$$
D_{\mathrm{KL}}
(p_{\theta_1}\|p_{\theta_2})
=
B_{A^*}(\eta_1,\eta_2).
$$

![](assets/information-theory-for-ml/en/module-15/M15_dual_coordinates_EN.png)

Natural coordinates $\theta$ and expectation coordinates $\eta$ form two affine coordinate systems related by Legendre duality. This is the meaning of a dually flat structure. “Flat” refers to the pair of e/m affine connections; it does not imply zero Riemannian curvature for the Fisher–Rao Levi–Civita connection. The Gaussian family in §15.9 gives an explicit counterexample.

### Mirror Descent

A mirror-descent step has the form

$$
w_{t+1}
=
\arg\min_w
\left\{
\rho g_t^\top w
+D_\psi(w,w_t)
\right\},
$$

where $D_\psi$ is the Bregman divergence generated by a convex function $\psi$.

On the probability simplex, the natural choice

$$
\psi(p)=\sum_i p_i\log p_i
$$

gives

$$
D_\psi(p,q)
=D_{\mathrm{KL}}(p\|q).
$$

The update is

$$
\boxed{
p_{t+1,i}
=
\frac{
 p_{t,i}e^{-\rho g_{t,i}}
}{
 \sum_jp_{t,j}e^{-\rho g_{t,j}}
}
}.
$$

This is the same exponential tilt that appeared in Module 8. There it arose as a MaxEnt or KL-regularized optimum; here it appears as a proximal update on the simplex.

Locally, a Bregman divergence has the quadratic form

$$
D_\psi(w+dw,w)
=
\frac12dw^\top\nabla^2\psi(w)dw
+o(\|dw\|^2).
$$

Mirror descent and natural gradient are therefore closely related through the corresponding local metric; in exponential families the relation is especially clean in dual coordinates. Globally they are not the same algorithm for an arbitrary $\psi$ and a finite step. Mirror descent uses the full asymmetric Bregman divergence, while natural gradient uses its local quadratic metric. Raskutti and Mukherjee analyze this equivalence on the dual Riemannian manifold of exponential families. ([Raskutti & Mukherjee, 2013](https://arxiv.org/abs/1310.7780))

## 15.8. Practical Map: Choose the Object Before the Geometry

After all these formulas, one final overreach is tempting: call Fisher “the true geometry of machine learning,” or replace it everywhere with Wasserstein. The useful map is more modest.

| Task | Point in the space | Meaning of a small change | Useful construction |
|---|---|---|---|
| likelihood model | $p_\theta$ | neighboring distributions are hard to distinguish | Fisher / local KL |
| policy update | $\pi_\theta(a\mid s)$ | action distributions change little on relevant states | state-averaged policy KL / Fisher |
| simplex weights | distribution $p$ | small Bregman cost from the current point | mirror descent / KL |
| sample transport | distribution in data space | mass moves a short distance | Wasserstein / OT |
| deterministic regression | parameters or outputs | small task-specific functional error | Hessian, GGN, or a task metric |
| large network | structured parameter blocks | a useful curvature is approximated within budget | K-FAC, Shampoo, adaptive preconditioners |

![](assets/information-theory-for-ml/en/module-15/M15_decision_map_EN.png)

Before using geometric language, ask five questions.

1. **What is the object being updated?** Parameters, response distributions, representations, or sample paths?
2. **What should count as small?** KL, transport cost, logit change, or task error?
3. **Is the approximation local?** If so, is the realized step actually small enough?
4. **Which matrix or divergence does the code compute?** Model Fisher, empirical Fisher, GGN, or only a diagonal moment?
5. **What is checked after the update?** Realized KL, task quality, integration cost, or policy drift?

The central synthesis is:

> **Geometry does not choose the learning objective. It determines which motions count as nearby while optimizing that objective.**

Natural gradient cannot repair a misspecified reward. Wasserstein does not guarantee a convenient path for a finite neural network. K-FAC does not make a nonconvex problem convex. But an appropriate geometry can remove artificial coordinate dependence, make a trust region meaningful, and identify which curvature structure is worth preserving.

## 15.9. Mathematical Deepening: α-Geometry and Exact Examples

> **Optional on a first pass.** The main ML route is complete. Here we examine how one Fisher metric can coexist with several asymmetric connections, and we write two geodesies in closed form.

### α-Divergences in Amari’s Convention

For $\alpha\ne\pm1$,

$$
\boxed{
D^{(\alpha)}(P\|Q)
=
\frac{4}{1-\alpha^2}
\left[
1-
\int
p^{(1-\alpha)/2}
q^{(1+\alpha)/2}
\right]
}.
$$

The limits are

$$
\alpha\to-1:
\qquad
D^{(\alpha)}(P\|Q)
\longrightarrow
D_{\mathrm{KL}}(P\|Q),
$$

$$
\alpha\to+1:
\qquad
D^{(\alpha)}(P\|Q)
\longrightarrow
D_{\mathrm{KL}}(Q\|P).
$$

At $\alpha=0$ one obtains a constant multiple of squared Hellinger distance.

This $\alpha$ is Amari’s geometry parameter, not the order of Rényi divergence. Different sign and scaling conventions exist, so the limits must be read together with the definition.

All smooth α-divergences in this family induce the same Fisher metric at second order. Their difference appears at third order and determines different affine connections:

- $\alpha=+1$: e-connection;
- $\alpha=-1$: m-connection;
- $\alpha=0$: the Fisher-metric Levi–Civita connection.

![](assets/information-theory-for-ml/en/module-15/M15_alpha_geometry_EN.png)

This is the key conceptual point: **a metric describes local length, but a metric alone does not recover the asymmetry of a divergence or its notion of a straight line.**

### Categorical Distributions as Part of a Sphere

For an interior categorical distribution $p=(p_1,\ldots,p_K)$, the Fisher line element is

$$
ds^2
=
\sum_{i=1}^K\frac{dp_i^2}{p_i},
\qquad
\sum_i dp_i=0.
$$

Set

$$
\psi_i=2\sqrt{p_i}.
$$

Then

$$
\sum_i\psi_i^2=4
$$

and

$$
ds^2=\sum_i d\psi_i^2.
$$

Thus the interior of the simplex maps to the positive orthant of a sphere of radius $2$. The exact distance is

$$
\boxed{
d_{\mathrm{FR}}(P,Q)
=
2\arccos
\left(
\sum_i\sqrt{p_iq_i}
\right)
}.
$$

For $K=2$ this reduces to the Bernoulli formula from §15.2. The coefficient

$$
\sum_i\sqrt{p_iq_i}
$$

already appeared in Module 11 as the Bhattacharyya coefficient and in Hellinger distance. Here it becomes the cosine of a spherical angle.

### Bernoulli Geodesic

In the coordinate

$$
u=2\arcsin\sqrt p,
$$

the geodesic is an ordinary line:

$$
u(t)=(1-t)u(p_0)+tu(p_1).
$$

Transforming back,

$$
\boxed{
p(t)
=
\sin^2
\left(
\frac{u(t)}2
\right)
}.
$$

It is not the linear probability interpolation $(1-t)p_0+tp_1$.

### The Normal Family and the Hyperbolic Plane

For

$$
X\sim\mathcal N(\mu,\sigma^2),
$$

Fisher in coordinates $(\mu,\sigma)$ is

$$
G(\mu,\sigma)
=
\begin{pmatrix}
1/\sigma^2&0\\
0&2/\sigma^2
\end{pmatrix}.
$$

Let

$$
x=\frac\mu{\sqrt2},
\qquad
y=\sigma.
$$

Then

$$
ds^2
=2\frac{dx^2+dy^2}{y^2}.
$$

This is a scaled Poincaré upper-half-plane metric with constant Gaussian curvature

$$
K=-\frac12.
$$

![](assets/information-theory-for-ml/en/module-15/M15_gaussian_geometry_EN.png)

Thus the normal family is an exponential family and is dually flat, but it is not Riemann-flat with respect to the Fisher–Rao Levi–Civita connection. The two uses of “flat” refer to different connections.

### Chentsov’s Theorem

In classical finite-dimensional statistics, Chentsov’s theorem says, roughly, that up to an overall scale the Fisher–Rao metric is the unique Riemannian metric on probability simplices compatible with sufficient statistics and suitable stochastic mappings.

This is a powerful justification for Fisher as a statistical geometry. It does not imply that Fisher is the correct cost for moving images, physical coordinates, or hidden states. The ground metric of the data space does not appear in the theorem. Wasserstein answers a different question, so the two constructions do not compete.

## 15.11. Conclusion

By the end of the course, familiar information quantities have acquired another role. Entropy and KL began as bills for uncertainty and model mismatch. Their second-order behavior now defines the local shape of distribution space and changes what a gradient step means.

Three questions keep the map clear:

- **Fisher–Rao:** how distinguishable are neighboring probabilistic models?
- **Bregman geometry and mirror descent:** how should a constrained distribution be updated relative to its current point?
- **Wasserstein:** what does it cost to physically move probability mass through data space?

These geometries cannot be substituted for one another mechanically. Together, however, they organize a broad range of modern ML methods:

$$
\boxed{
\begin{array}{c}
\text{local distinguishability}
\longrightarrow
\text{natural gradient and trust regions}
\\[1mm]
\text{asymmetric proximal cost}
\longrightarrow
\text{mirror and exponential updates}
\\[1mm]
\text{mass-transport cost}
\longrightarrow
\text{CNFs, Flow Matching, and diffusion paths}
\end{array}
}.
$$

The main lesson is not that every popular algorithm is “really” one geometric theorem. It is this:

> **First define the object, then define what closeness means, and only then choose a geometry and a computable approximation to it.**

Information geometry closes the course not with one more universal slogan, but with a discipline of problem formulation. A formula becomes stronger when we know exactly what it measures—and where its promise ends.

## Main Sources

1. S.-I. Amari, [*Information Geometry and Its Applications*](https://link.springer.com/book/10.1007/978-4-431-55978-8), Springer, 2016.
2. S.-I. Amari, [*Natural Gradient Works Efficiently in Learning*](https://doi.org/10.1162/089976698300017746), 1998.
3. J. Martens, [*New Insights and Perspectives on the Natural Gradient Method*](https://jmlr.org/papers/v21/17-678.html), 2020.
4. J. Martens and R. Grosse, [*Optimizing Neural Networks with Kronecker-factored Approximate Curvature*](https://arxiv.org/abs/1503.05671), 2015.
5. F. Kunstner, L. Balles, and P. Hennig, [*Limitations of the Empirical Fisher Approximation*](https://arxiv.org/abs/1905.12558), 2019.
6. S. Kakade, [*A Natural Policy Gradient*](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html), NeurIPS 2001.
7. J. Schulman et al., [*Trust Region Policy Optimization*](https://arxiv.org/abs/1502.05477), 2015.
8. J. Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), 2017.
9. R. Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290), 2023.
10. R. T. Q. Chen et al., [*Neural Ordinary Differential Equations*](https://arxiv.org/abs/1806.07366), 2018.
11. Y. Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747), 2022.
12. X. Liu, C. Gong, and Q. Liu, [*Flow Straight and Fast*](https://arxiv.org/abs/2209.03003), 2022.
13. Y. Song et al., [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), 2021.
14. G. Raskutti and S. Mukherjee, [*The Information Geometry of Mirror Descent*](https://arxiv.org/abs/1310.7780), 2013.
