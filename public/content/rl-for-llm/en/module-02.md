# Module 2. State, Value, and Bellman Equations

> **Material version:** 2026.7  
> **Content snapshot:** 2026-08-04  
> **Language:** EN  
> **Core practice:** Tier A — browser, Python, NumPy, and a central processing unit (CPU)  
> **Structure:** 4 lessons, 20 steps  
> **Estimated time:** 8–12 hours including the core lab

Module 1 framed RL as a loop of action selection and feedback. We now face a harder question: **how can we evaluate a state or an action before the final outcome is known?** The answer begins with a well-chosen state representation, continues with value functions, and leads to the Bellman equations.

A prior classical RL course is still not required. New concepts are introduced in plain language, formalized, and only then mapped to LLMs. Readers who already know RL can move quickly through §§2.1.1–2.1.4 and focus on the limits of the token-level model, termination versus truncation, the exact conditions behind convergence statements, and the bridges to modern LLM post-training.

**Learning goals.** By the end of the module, you will be able to:

1. check whether a state representation contains enough information for a Markov model;
2. distinguish a state from an observation and task termination from technical trajectory truncation;
3. compute $V^\pi$, $Q^\pi$, and the advantage $A^\pi$, and interpret them for LLM generation;
4. derive the Bellman expectation equation from the return recursion;
5. solve a small tabular decision process as a linear system or by iteration;
6. distinguish Monte Carlo evaluation, one-step temporal-difference learning, and $n$-step targets;
7. explain two tabular control methods through behavior and target policies;
8. recognize maximization bias and explain how separating selection from evaluation reduces it;
9. reproduce numerical experiments on Gridworld, FrozenLake, the Corridor, and CliffWalking.

The public computational lab is the executable practice. It implements exact policy evaluation, value iteration, Monte Carlo prediction, TD(0), multi-step TD, Q-learning, SARSA, and Double Q-learning. NumPy, Matplotlib, and a CPU are sufficient; no GPU is required.

---

> **Practice route.** Graded tasks, worked solutions revealed after an attempt, and executable practice follow the lecture. The explanation remains self-contained.

## Lesson 2.1. From history to state, then to value

> **Lesson 2.1. From history to state, then to value**
>
> 1. Why we need a state
> 2. A finite MDP and the reward convention
> 3. Task termination and technical truncation are different
> 4. Three value functions: V, Q, and advantage A
> 5. Reading value functions for an LLM
>
> then 7 assessment steps

### Step 2.1.1 — Why we need a state

The complete interaction history at time $t$ may be long:

$$
H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t).
$$

The word *state* is used for two related but distinct objects.

1. An **environment state** is a variable relative to which future dynamics are Markov. It need not be a compression of anything the agent has observed.
2. An **agent state or representation** is constructed from available history: a transcript, a finite window, recurrent memory, or an external-memory record. It is only a candidate sufficient state.

Let $S_t=f(H_t)$ be the chosen representation. It is one-step sufficient when

$$
P(S_{t+1},R_{t+1}\mid H_t,A_t)
=
P(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

This equation does not claim that the world has forgotten its past. It says that, once $S_t$ and $A_t$ are fixed, earlier history should not further change the conditional distribution of the next state and reward.

> **Three questions, not one.** What does the environment actually store? What can the agent observe? What internal record does the agent use to choose an action? Simple textbook tasks make the three coincide; tool-using systems often do not.

An observation $O_t$ may be incomplete. A sufficient state supports a **Markov decision process** (MDP). When hidden environment variables affect future transitions, a **partially observable Markov decision process** (POMDP) is the natural model, and the agent may rely on observation history, memory, or a belief over hidden state.

![When history is a state](assets/rl-for-llm/en/module-02/M2_state_sufficiency_EN.png)

For pure autoregressive generation, one may write

$$
S_t=(x,y_{<t}),
\qquad
A_t=y_t,
\qquad
S_{t+1}=(x,y_{\le t}).
$$

If the transition only appends the chosen token, model parameters and decoding rules are fixed, and stopping depends on the visible prefix, the complete prefix is sufficient. Random token sampling is not a violation: it belongs to the policy $\pi(a\mid s)$, while the transition to the next prefix is deterministic once the token action has been fixed.

A browser, file session, user, or external tool can carry hidden state. Then two identical transcripts may lead to different responses because permissions, time, files, or session state differ. The transcript can be every message available to the model and still fail to define an MDP.

> **Practical sufficiency test.** Could two identical records $S_t$, followed by the same action, induce different next-observation distributions because relevant information is absent from $S_t$? If yes, the representation is incomplete. Differences caused only by randomness already described by one common distribution do not break the Markov property.

**Architecture context:** the agents and tools module in *Modern LLMs* (`modern_llms.module_16_agents`) develops the systems view. Here we only need the distinction among environment state, observation, and the agent's internal representation.

---

### Step 2.1.2 — A finite MDP and the reward convention

We define a finite MDP by

$$
\mathcal{M}
=
(\mathcal{S},\mathcal{A},P,P_R,\rho_0,\gamma).
$$

Here:

- $\mathcal{S}$ is the state space;
- $\mathcal{A}$ is the action space;
- $P(s'\mid s,a)$ is the next-state distribution;
- $P_R(r\mid s,a,s')$ is the reward distribution on a transition;
- $\rho_0(s)$ is the initial-state distribution;
- $\gamma\in[0,1]$ is the discount factor.

Tabular calculations often store the conditional mean reward rather than the full reward distribution:

$$
\bar r(s,a,s')
=
\mathbb{E}[R_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s'].
$$

Some tasks genuinely permit a shorter notation such as $\bar r(s,a)$ or $\bar r(s)$. These are not automatically interchangeable formulas: each asserts an additional conditional-independence structure. This module uses $\bar r(s,a,s')$ as its default form.

The policy $\pi(a\mid s)$ is **not part of the environment dynamics**. The same MDP can be paired with different policies. Under a fixed policy, a one-step joint distribution factors as

$$
P(A_t=a,S_{t+1}=s',R_{t+1}=r\mid S_t=s)
=
\pi(a\mid s)P(s'\mid s,a)P_R(r\mid s,a,s').
$$

This factorization will soon become the Bellman equation: average over policy actions, then over environment transitions and rewards.

An **episodic** task reaches a terminal state. A **continuing** task may have no natural terminal state; for an infinite horizon, one commonly uses $\gamma<1$ or another objective that remains finite.

Throughout this module, reward belongs to a **transition**. If entering the goal pays $+1$, that reward remains in $R_{t+1}$ even though $V(s_{terminal})=0$. Zero terminal value removes future continuation; it does not erase the final reward.

---

### Step 2.1.3 — Task termination and technical truncation are different

A trajectory file may record the same surface fact—“there is no next sample”—for two mathematically different reasons.

1. **Termination** means the task itself reached a terminal state: success, failure, or an answer-ending token. The specified MDP has no continuation after that transition.
2. **Truncation** means collection stopped because of an external budget: steps, wall-clock time, buffer capacity, or batch length. The underlying task could have continued.

Define a continuation mask

$$
m_t=\mathbf 1\{\text{transition }t\text{ is not terminal}\}.
$$

A one-step target becomes

$$
y_t=R_{t+1}+\gamma m_tV(S_{t+1}),
$$

or, equivalently,

$$
y_t=
\begin{cases}
R_{t+1}, & \text{for true termination},\\[4pt]
R_{t+1}+\gamma V(S_{t+1}), & \text{for external truncation of a continuing process}.
\end{cases}
$$

![True termination and external truncation](assets/rl-for-llm/en/module-02/M2_termination_truncation_EN.png)

When a time limit defines the **task**, remaining time belongs in state and exhausting it may be a true terminal outcome. When the limit exists only for data collection, zeroing continuation is usually wrong.

For LLM systems:

- an end-of-sequence (EOS) token usually represents answer termination;
- `max_tokens` is often truncation when a longer completion would still be meaningful;
- “answer within 256 tokens” makes the same budget part of the objective;
- stopping a rollout because a batch is full is infrastructure, not task completion.

Conflating the cases creates systematic bias. With positive continuation value, treating truncation as terminal pushes the target down; with negative continuation value, it pushes the target up. Gymnasium therefore exposes `terminated` and `truncated` separately. Two flags do not remove the modeling decision: the environment author must still decide whether a limit belongs to the task.

---

### Step 2.1.4 — Three value functions: V, Q, and advantage A

Fix a policy $\pi$ and define the return

$$
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}
$$

for a continuing task, or the corresponding finite sum for an episode.

The **state-value function** is the expected outcome when starting from state $s$ and following the policy:

$$
V^\pi(s)
=
\mathbb{E}_\pi[G_t\mid S_t=s].
$$

The **action-value function** fixes the first action and follows the policy afterward:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

Therefore the state value is the policy-weighted average of the action values:

$$
V^\pi(s)
=
\sum_a \pi(a\mid s)Q^\pi(s,a).
$$

The **advantage function** compares an action with the current policy's average behavior:

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
$$

![Value and advantage for the next token](assets/rl-for-llm/en/module-02/M2_value_advantage_EN.png)

The definition immediately implies

$$
\sum_a \pi(a\mid s)A^\pi(s,a)=0.
$$

A positive $A^\pi(s,a)$ means “better than the current policy's average action in this state,” not “objectively good under every policy.”

A useful invariance is also immediate. Add the same constant $c(s)$ to all action values in a state. The recomputed state value shifts by the same amount and the advantages stay unchanged:

$$
Q'(s,a)=Q(s,a)+c(s),
\quad
V'(s)=V(s)+c(s),
\quad
A'(s,a)=A(s,a).
$$

This is a statement about shifting **action-value estimates within a fixed state**. It does not imply that an arbitrary change to the reward function leaves advantages unchanged throughout an MDP.

Under the transition-reward convention, $V^\pi(s_{terminal})=0$. A final episode or response score is paid on the transition into that terminal and is already part of $G_t$.

---

### Step 2.1.5 — Reading value functions for an LLM

In a token-level model, state is the prompt plus generated prefix and action is the next token. Then

- $Q^\pi(s,a)$ is expected final reward after fixing token $a$ now and continuing under $\pi$;
- $V^\pi(s)$ is the policy average over possible next-token choices;
- $A^\pi(s,a)$ measures the token's deviation from that current-policy average.

A tiny prefix tree makes the definitions concrete. Suppose the prefix “because” leads to “check” with probability $0.7$ and reward $1$, or to “skip” with probability $0.3$ and reward $0.4$. Then

$$
V^\pi(\text{because})=0.7\cdot1+0.3\cdot0.4=0.82.
$$

If the alternative prefix “guess” has value $-0.10$, and the empty prefix chooses “because” with probability $0.55$, then

$$
V^\pi(\varnothing)=0.55\cdot0.82+0.45\cdot(-0.10)=0.406.
$$

The notebook computes exactly this tree. It is not a miniature claim about real LLM scale; it isolates what token-level $Q$ means: fix one next token, then average every continuation that follows under the policy.

Real value estimation is difficult. Prefix spaces are enormous, final reward often arrives only after a complete response, and nearly identical reasoning traces may receive different evaluations. Neural RL therefore uses learned critics, process evaluators, continuation predictors, or algorithms without a separately trained critic.

A common overreach concerns **group relative policy optimization** (GRPO). In the original DeepSeekMath construction, several complete responses are sampled for one prompt and their rewards are normalized within the group. This supplies a relative signal without a learned value model. The finite-group statistic is nevertheless a response-level sample baseline, not an exact $V^\pi(s_t)$ at every token prefix. Later GRPO variants alter normalization, aggregation, and token assignment, so the method name alone does not specify every detail.

> **The bridge worth keeping.** A value function asks what an unseen continuation is worth in expectation. A PPO critic, a GRPO group baseline, and a partial-reasoning evaluator approximate that broad question in different ways; they are not the same mathematical object.

Module 4 (`rl_llm.module_04_policy_gradients`) will connect advantage to policy gradients and GAE. Module 9 (`rl_llm.module_09_rlvr_grpo`) develops GRPO variants. The reasoning module in *Modern LLMs* (`modern_llms.module_13_reasoning`) supplies the architectural context for scoring intermediate steps.

---

## Lesson 2.2. Bellman equations: evaluation and improvement

> **Lesson 2.2. Bellman equations: evaluation and improvement**
>
> 1. One-step decomposition of the return
> 2. Exact policy evaluation as a linear system
> 3. Optimality and the exact contraction condition
> 4. Evaluation, improvement, and generalized policy iteration
> 5. Reproducible example: FrozenLake
>
> then 6 assessment steps

### Step 2.2.1 — One-step decomposition of the return

Module 1 gave us the return recursion

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

Take the conditional expectation at $S_t=s$ under a fixed policy $\pi$:

$$
V^\pi(s)
=
\sum_a \pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')+\gamma V^\pi(s')
\right].
$$

Terminal states may remain in the sum when their continuation values are defined as zero. Reward paid on entry remains in $\bar r(s,a,s')$.

In words:

> state value is expected immediate reward plus discounted continuation value, averaged over the policy's action and the environment's transition.

For action value, the first action is already fixed:

$$
Q^\pi(s,a)
=
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')
+
\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
\right].
$$

Define the fixed-policy Bellman operator

$$
(T_\pi V)(s)
=
\sum_a \pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')+\gamma V(s')
\right].
$$

The true value is its fixed point:

$$
V^\pi=T_\pi V^\pi.
$$

An operator is simply a rule that maps an entire value vector to a new one-step-backed-up vector.

> **What Bellman consistency does not specify.** It does not choose a data source, a function class, or an optimizer, and it does not guarantee that neural training will reach the fixed point. Different exact and approximate algorithms are built from the same consistency relation.

---

### Step 2.2.2 — Exact policy evaluation as a linear system

For a fixed policy, combine the action choice into

$$
P_\pi(s,s')
=
\sum_a\pi(a\mid s)P(s'\mid s,a),
$$

$$
r_\pi(s)
=
\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)\bar r(s,a,s').
$$

The Bellman equation becomes

$$
V^\pi=r_\pi+\gamma P_\pi V^\pi,
$$

so

$$
(I-\gamma P_\pi)V^\pi=r_\pi.
$$

For a finite discounted MDP with $0\le\gamma<1$, $I-\gamma P_\pi$ is invertible and the solution is unique. At $\gamma=1$, the generic discount-based guarantee disappears. An episodic problem can still be solved over non-terminal states when the fixed policy reaches a terminal state almost surely and the resulting matrix is nonsingular.

A classic $4\times4$ grid has terminal states in the upper-left and lower-right corners, reward $-1$ per move, and an equiprobable random policy. Its exact value function for $\gamma=1$ is

$$
V^\pi=
\begin{bmatrix}
0 & -14 & -20 & -22\\
-14 & -18 & -20 & -20\\
-20 & -20 & -18 & -14\\
-22 & -20 & -14 & 0
\end{bmatrix}.
$$

The entries are the negative expected numbers of steps to a terminal state under the random policy.

For an approximate value function $\widehat V$, define the **Bellman residual**

$$
e_\pi=T_\pi\widehat V-\widehat V.
$$

A zero residual solves the equation exactly. For $\gamma<1$,

$$
\|\widehat V-V^\pi\|_\infty
\le
\frac{\|e_\pi\|_\infty}{1-\gamma}.
$$

Thus a small one-step residual can still correspond to a noticeable long-horizon error when $\gamma$ is close to one.

---

### Step 2.2.3 — Optimality and the exact contraction condition

Policy evaluation asks, “how good is this fixed policy?” Control asks, “which action should be selected?”

The optimal value function satisfies

$$
V^*(s)
=
\max_a
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')+\gamma V^*(s')
\right].
$$

Define the optimality operator

$$
(T_*V)(s)
=
\max_a
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')+\gamma V(s')
\right].
$$

![Two Bellman operators](assets/rl-for-llm/en/module-02/M2_bellman_operators_EN.png)

For $0\le\gamma<1$, both operators are $\gamma$-contractions in the max norm. The key step for $T_*$ is

$$
\begin{aligned}
|(T_*V)(s)-(T_*W)(s)|
&\le
\max_a
\left|
\gamma\sum_{s'}P(s'\mid s,a)(V(s')-W(s'))
\right|\\
&\le
\gamma\|V-W\|_\infty.
\end{aligned}
$$

Taking the maximum over states gives

$$
\|T_*V-T_*W\|_\infty
\le
\gamma\|V-W\|_\infty.
$$

This gives a unique fixed point and geometric convergence of $V_{k+1}=T_*V_k$.

The boundary matters. At $\gamma=1$, the contraction factor is one, so this proof no longer establishes convergence. Some finite episodic problems at $\gamma=1$ are still well posed and admit convergent dynamic-programming schemes, but they require additional assumptions. In a stochastic-shortest-path formulation, for example, there must be a policy that reaches a terminal state from every relevant state almost surely in finite expected time, and non-terminating policies cannot be allowed to remain competitive as though they had an ordinary finite cost. Merely adding a terminal state is not enough. “The Bellman operator is always a strict contraction” is false without a regime qualifier.

---

### Step 2.2.4 — Evaluation, improvement, and generalized policy iteration

The Bellman equations suggest two alternating operations.

1. **Policy evaluation:** approximate $V^\pi$ for the current $\pi$.
2. **Policy improvement:** select actions that maximize the one-step estimate

$$
Q(s,a)=
\sum_{s'}P(s'\mid s,a)
\left[
\bar r(s,a,s')+\gamma V(s')
\right].
$$

Evaluating to convergence and then improving greedily gives **policy iteration**. Maximizing on every sweep gives **value iteration**:

$$
V_{k+1}(s)=\max_a Q_k(s,a).
$$

The broader pattern is **generalized policy iteration** (GPI): evaluation and improvement may be partial, asynchronous, and performed on different timescales, while each operation pushes the other toward consistency.

| Method | What is fixed inside the update | Result at convergence |
|---|---|---|
| Iterative evaluation | policy $\pi$ | $V^\pi$ |
| Policy iteration | full evaluation, then greedy improvement | an optimal policy |
| Value iteration | maximization on every one-step backup | $V^*$ and a greedy policy |

GPI is not a single algorithm with one formula. It is a pattern. In actor–critic methods, the critic evaluates while the actor improves. Tabular control exposes the same two operations without neural networks or optimizers.

---

### Step 2.2.5 — Reproducible example: FrozenLake

`module_2_reference.py` contains a self-contained NumPy version of slippery $4\times4$ FrozenLake. For each intended direction, the intended move and two lateral deviations have probability $1/3$. Holes and the goal are terminal, and reward $1$ is paid only on entry to the goal.

Let $T\ge1$ be the number of transitions until the episode ends, either in the goal or a hole. Because reward $1$ is emitted only on entry to the goal, a successful trajectory contributes $\gamma^{T-1}$ to $G_0$, while an unsuccessful one contributes zero. Under optimal behavior,

$$
V^*(s)
=
\mathbb E\!\left[
\gamma^{T-1}\mathbf 1\{\text{goal reached}\}
\mid S_0=s
\right].
$$

With $\gamma=0.9$, value is therefore not a bare success probability: earlier success receives more weight.

```python
from module_2_reference import build_frozen_lake_4x4, value_iteration

P, R, terminal = build_frozen_lake_4x4()
V, sweeps = value_iteration(P, R, gamma=0.9, terminals=terminal)
print(sweeps)
print(V.reshape(4, 4))
```

Starting from zero, synchronous value iteration with tolerance $10^{-10}$ takes 145 complete sweeps and returns

```text
[[0.068891 0.061415 0.074410 0.055807]
 [0.091855 0.000000 0.112208 0.000000]
 [0.145436 0.247497 0.299618 0.000000]
 [0.000000 0.379936 0.639020 0.000000]]
```

![Value propagation through FrozenLake](assets/rl-for-llm/en/module-02/M2_frozen_lake_propagation_EN.png)

After one sweep, value is nonzero only where the goal can be entered in one transition. Later sweeps carry the signal backward through stochastic dynamics. Nothing physical is “flowing” through the grid; the picture shows repeated application of a one-step operator to an improving continuation estimate.

The number 145 belongs to this exact protocol: dynamics, action indexing, $\gamma$, zero initialization, synchronous updates, and stopping tolerance. Change any of them and the sweep count may change.

The tabular example matters for an LLM course because it exposes three durable ideas:

- a one-step target links the present estimate to continuation value;
- evaluation and improvement form a feedback loop;
- continuation error propagates over the horizon.

Neural approximation removes the simple tabular guarantees. Data distribution, optimization, architecture, and jointly changing parameters now matter. The Bellman equation remains a target relation, not an automatic convergence certificate.

---

## Lesson 2.3. Model-free prediction: from full trajectories to bootstrapping

> **Lesson 2.3. Model-free prediction: from full trajectories to bootstrapping**
>
> 1. Prediction from sampled trajectories
> 2. Monte Carlo evaluation
> 3. One observed step and a continuation estimate
> 4. A multi-step target connects MC and TD
> 5. The Corridor: compare estimators, not slogans
>
> then 6 assessment steps

### Step 2.3.1 — Prediction from sampled trajectories

So far, $P$ and $\bar r$ were known. In an actual environment, an agent usually sees only sampled transitions

$$
S_t,A_t,R_{t+1},S_{t+1}.
$$

The **prediction problem** estimates $V^\pi$ or $Q^\pi$ for a fixed policy from interaction data without an explicit dynamics table.

Such a method chooses a target $Y_t$ and moves the current estimate toward it. For tabular $V$,

$$
V(S_t)
\leftarrow
V(S_t)+\alpha\bigl(Y_t-V(S_t)\bigr),
$$

where $\alpha$ is the step size. The main difference among methods is the choice of $Y_t$.

- A full observed return barely depends on the current estimate, but requires waiting for the episode tail.
- A one-step target is available immediately, but uses the imperfect estimate $V(S_{t+1})$.
- Intermediate $n$-step targets consume several observed rewards and then bootstrap.

**Bootstrapping** means updating one estimate from another current estimate. It is unrelated to the statistical bootstrap based on resampling.

---

### Step 2.3.2 — Monte Carlo evaluation

A **Monte Carlo** (MC) method uses the observed return itself:

$$
Y_t^{\mathrm{MC}}=G_t.
$$

For first visits to state $s$, one may average all subsequent returns:

$$
V_n(s)=\frac{1}{n}\sum_{i=1}^{n}G^{(i)}(s).
$$

When episodes truly come from a fixed policy, termination is handled correctly, and returns have finite expectation, this is an ordinary sample mean whose expectation is $V^\pi(s)$. In this restricted sense, the MC target does not contain bias from the current continuation estimate.

That does not make every practical MC estimator automatically unbiased. Technical truncation, a changing policy, constant step sizes, initialization, and dependent data reuse can alter its statistical properties.

Strengths:

- no transition model is required;
- there is no bootstrapping;
- current value-estimation error is absent from the target.

Limitations:

- enough of the trajectory, often the whole episode, must be observed;
- a single terminal reward may produce high variance;
- long episodes are expensive;
- rarely visited states accumulate data slowly.

A full LLM response with one final score is an MC-like sequence-level signal: the outcome is observed after generation, but the contribution of each token remains hidden.

---

### Step 2.3.3 — One observed step and a continuation estimate

**Temporal-difference learning** (TD) updates before an episode ends. The one-step TD(0) target is

$$
Y_t^{\mathrm{TD}(0)}
=
R_{t+1}+\gamma V(S_{t+1})
$$

for a non-terminal transition. The **temporal-difference error** is

$$
\delta_t
=
R_{t+1}+\gamma V(S_{t+1})-V(S_t).
$$

The update is

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t.
$$

At a true terminal state, the continuation is zero:

$$
\delta_t=R_{t+1}-V(S_t).
$$

At a technical truncation where the process could continue, $V(S_{t+1})$ is normally retained.

TD(0) uses fewer random future rewards but relies on the current estimate. Its finite-sample target therefore inherits value-estimation error. This often reduces variance at the cost of bias, but there is no universal ordering in which TD always beats MC or vice versa. The outcome depends on the environment, representation, step size, and data budget.

A useful intuition: MC waits for the bill for the whole route; TD revises the current estimate after each newly observed segment.

---

### Step 2.3.4 — A multi-step target connects MC and TD

If the task does not terminate within the next $n$ transitions, the $n$-step target is

$$
G_{t:t+n}
=
\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+
\gamma^nV(S_{t+n}).
$$

At $n=1$ this is TD(0). If a true terminal occurs before the end of the window, the observed sum stops at that transition and no continuation term is added. When the window covers the remaining episode, the target becomes Monte Carlo.

![One-step, multi-step, and full-return targets](assets/rl-for-llm/en/module-02/M2_mc_td_targets_EN.png)

Increasing $n$ usually means

- more observed reward in the target;
- less dependence on the current $V$ estimate;
- a longer wait before updating;
- more exposure to randomness in a longer trajectory segment.

Those tendencies are not a monotone quality guarantee. Representation error, step size, temporal correlation, and episode length can change the best horizon.

Generalized advantage estimation (GAE) will later combine temporal-difference residuals across horizons with exponential weights. The important precursor is already here: an $n$-step target chooses where observed trajectory data give way to a current continuation estimate.

The reference `n_step_target()` receives observed rewards, a bootstrap value, $\gamma$, and an explicit terminal flag for the end of the window. That interface prevents external truncation from being silently treated as task termination.

---

### Step 2.3.5 — The Corridor: compare estimators, not slogans

Consider a symmetric five-cell Corridor with terminals at 0 and 4, a middle start, and equal-probability left/right actions. Reward $1$ is paid only on entry to the right terminal. At $\gamma=1$, the values are the probabilities of reaching the right side before the left:

$$
V^\pi=[0,\ 0.25,\ 0.50,\ 0.75,\ 0].
$$

The figure uses one controlled protocol:

- episode budgets from 20 to 2,500;
- 40 independent random seeds;
- first-visit Monte Carlo;
- TD(0) and three-step TD with constant $\alpha=0.1$;
- root mean squared error (RMSE) over the three non-terminal states;
- a median line and interquartile band.

![MC, TD(0), and a three-step target](assets/rl-for-llm/en/module-02/M2_prediction_rmse_EN.png)

The experiment is designed to resist a universal slogan. At small budgets, noise and initialization dominate. With more data, the MC sample mean continues toward the exact value. Constant-step-size TD retains residual variation. Three-step TD is not guaranteed to sit between the other curves at every budget; it follows its own interaction between observed rewards and current estimates.

Reference outputs include

```text
exact: [0.00, 0.25, 0.50, 0.75, 0.00]
MC, 1000 episodes, seed=7:
       [0.00, 0.255539, 0.496000, 0.733728, 0.00]
TD(0), 1000 episodes, seed=42, alpha=0.1:
       [0.00, 0.233105, 0.582660, 0.829850, 0.00]
3-step TD, 1000 episodes, seed=42, alpha=0.1:
       [0.00, 0.248712, 0.523243, 0.769480, 0.00]
```

The different seeds in these lines are reproducibility fixtures, not a contest. Annealed step sizes, another horizon, function approximation, reused data, or non-stationarity would change the result.

**LLM bridge.** A final score on a complete response resembles an MC observation. A critic, process evaluator, or continuation predictor introduces bootstrapping. Group normalization compares complete responses but does not by itself become an exact token-level value function.

---

## Lesson 2.4. Control: behavior and target policies

> **Lesson 2.4. Control: behavior and target policies**
>
> 1. From fixed-policy prediction to behavior improvement
> 2. Two update targets differ by one line, not by one detail
> 3. CliffWalking: one protocol, different routes
> 4. Why a maximum overestimates noisy values
> 5. What transfers to LLM RL, and what remains a tabular teaching model
>
> then 6 assessment steps

### Step 2.4.1 — From fixed-policy prediction to behavior improvement

In control, the policy changes during learning. It helps to distinguish:

- the **behavior policy**, which actually selects actions and collects data;
- the **target policy**, whose value is being estimated or improved.

When they match, learning is **on-policy**. When they differ, learning is **off-policy**. These labels are not defined by the presence of a replay buffer or by random exploration itself; they describe **which policy is inferred from which policy's data**.

A simple exploration rule is $\varepsilon$-greedy:

- with probability $1-\varepsilon$, choose an action maximizing $Q(s,a)$;
- with probability $\varepsilon$, choose a random action.

This prevents the learner from immediately locking onto its first apparent route. Systematic exploration is the subject of Module 3; here $\varepsilon$ lets us expose the difference between two update targets.

---

### Step 2.4.2 — Two update targets differ by one line, not by one detail

Both methods update tabular $Q(S_t,A_t)$:

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha\bigl(Y_t-Q(S_t,A_t)\bigr).
$$

The **state–action–reward–state–action** method (SARSA) uses the next action actually selected by the behavior policy:

$$
Y_t^{\mathrm{SARSA}}
=
R_{t+1}+\gamma Q(S_{t+1},A_{t+1}).
$$

With $\varepsilon$-greedy behavior, SARSA evaluates the consequences of future exploration. It is on-policy.

Q-learning uses a greedy target regardless of the action that will actually be executed next:

$$
Y_t^{Q}
=
R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a').
$$

Data may be gathered by an $\varepsilon$-greedy behavior policy, while the target refers to the greedy policy. Q-learning is therefore off-policy.

At a terminal transition, both targets reduce to $R_{t+1}$. At a technical truncation, the continuation is normally retained.

In a finite discounted tabular MDP with bounded rewards, classical Q-learning convergence requires every state–action pair to be visited infinitely often and, for each pair, step sizes satisfying $\sum_t\alpha_t(s,a)=\infty$ and $\sum_t\alpha_t^2(s,a)<\infty$. This is an asymptotic tabular stochastic-approximation result. Constant step size, finite data, neural $Q$ functions, changing targets, and jointly trained components do not inherit it automatically.

---

### Step 2.4.3 — CliffWalking: one protocol, different routes

In CliffWalking, the agent moves from the lower-left to the lower-right corner. An ordinary step yields $-1$. Entering the cliff yields $-100$ and returns the agent to the start. Behavior remains $\varepsilon$-greedy during training.

The static figure uses

- random seed 0;
- 500 episodes;
- $\alpha=0.5$;
- $\gamma=1$;
- $\varepsilon=0.1$;
- greedy evaluation after training.

![Q-learning and SARSA routes](assets/rl-for-llm/en/module-02/M2_cliff_routes_EN.png)

Reproduced results for this run are

| Method | Greedy-route return | Mean training return |
|---|---:|---:|
| Q-learning | $-13$ | $-54.72$ |
| SARSA | $-17$ | $-33.656$ |

Q-learning evaluates a greedy continuation and discovers the short route along the cliff. Actual exploratory behavior occasionally deviates and pays a large penalty. SARSA evaluates the continuation of that exploratory behavior and, in this run, learns a longer route with more margin.

This is **not a universal ranking**. Change $\varepsilon$, the penalty, episode budget, initialization, or evaluation rule and the numbers may move. The point is the causal relation between an update target and the risk represented in the learned $Q$ function.

---

### Step 2.4.4 — Why a maximum overestimates noisy values

Suppose several actions have equal true value but noisy estimates:

$$
\widehat Q(a)=Q(a)+\varepsilon_a,
\qquad
\mathbb{E}[\varepsilon_a]=0.
$$

Each estimate may be unbiased on its own, yet the maximum selects the action that happened to receive a positive error:

$$
\mathbb{E}\left[\max_a \widehat Q(a)\right]
\ge
\max_a Q(a).
$$

This is **maximization bias**. Q-learning uses one table both to select the action and to evaluate it in the target, allowing selection noise to enter as an inflated continuation value.

**Double Q-learning** separates the two roles. When updating the first table, select with $Q^A$ and evaluate with $Q^B$:

$$
a^*=\arg\max_a Q^A(s',a),
$$

$$
Y_t^{\mathrm{Double}}
=
R_{t+1}+\gamma Q^B(s',a^*).
$$

The tables can then swap roles. If their estimation errors are sufficiently independent, the positive error that drove selection in $Q^A$ need not recur in $Q^B$. The method reduces typical overestimation, but it does not eliminate all error and can underestimate in some regimes.

There is a useful LLM analogy. When many responses are ranked by a noisy reward model and only the maximum is retained, the winner may benefit from evaluator noise as well as genuine quality. Separating generation, selection, and independent verification can reduce this effect. This is a shared statistical mechanism, not a claim that LLM reranking is literally Q-learning.

---

### Step 2.4.5 — What transfers to LLM RL, and what remains a tabular teaching model

The module's durable ideas are:

1. **State is not merely an input record.** It must be sufficient for future dynamics. A full prefix works for pure autoregression, while an external environment may make the task partially observable.
2. **$V$, $Q$, and $A$ answer different questions.** $V$ averages over the policy, $Q$ fixes the first action, and $A$ compares an action with the current-policy average.
3. **A Bellman equation is one-step consistency, not a complete learning algorithm.** It links present value to reward and continuation value.
4. **MC and TD choose where bootstrapping begins.** MC observes the remaining return; TD substitutes a current estimate; multi-step targets connect the endpoints.
5. **Behavior and target policies may differ.** That mismatch—not the mere presence of a replay buffer—makes Q-learning off-policy.
6. **Maximizing noisy estimates creates selection bias.** Double Q-learning separates selection from evaluation and addresses one specific source of overestimation.
7. **The reason a trajectory stops belongs to the mathematics.** True termination masks continuation; external truncation normally does not.

Tabular guarantees rely on finite spaces, a correct Markov model, sufficient exploration, and particular step-size schedules. LLM RL uses enormous spaces, neural approximators, potentially misspecified rewards, and a data distribution that moves with the policy. The equations provide a language and local training targets; they do not certify convergence of the complete system.

Abbreviation reference:

| Abbreviation | Full name | Role in this module |
|---|---|---|
| MDP | Markov decision process | fully observed Markov decision model |
| POMDP | partially observable Markov decision process | model with hidden environment state |
| MC | Monte Carlo | evaluation from observed returns |
| TD | temporal-difference learning | bootstrapped updating |
| GPI | generalized policy iteration | alternating evaluation and improvement |
| GAE | generalized advantage estimation | weighted combination of multi-horizon TD residuals |
| RMSE | root mean squared error | experiment error metric |
| EOS | end of sequence | sequence-termination token |

## Sources and suggested reading path

Links were checked on August 4, 2026. Each item supports a specific part of the module and has an explicit boundary.

1. **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.** — <https://incompleteideas.net/book/the-book-2nd.html>. The foundational source for finite MDPs, value functions, Bellman equations, MC, TD, SARSA, Q-learning, and GPI. Its tabular theory is not a proof of convergence for neural LLM training.
2. **Farama Foundation, “Handling Time Limits”.** — <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>. Supports the practical `terminated`/`truncated` distinction and bootstrap rule. It is interface and environment-design guidance, not a universal theorem about every finite-horizon objective.
3. **Watkins & Dayan, “Q-learning” (1992).** — <https://doi.org/10.1007/BF00992698>. The primary source for classical tabular Q-learning and its convergence conditions. Those conditions should not be silently transferred to nonlinear approximation.
4. **van Hasselt, “Double Q-learning” (2010).** — <https://proceedings.neurips.cc/paper/3964-double-q-learning.pdf>. The primary source for maximization bias and decoupled selection/evaluation. Double Q-learning reduces a characteristic overestimation mechanism; it is not universally unbiased.
5. **Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation”.** — <https://arxiv.org/abs/1506.02438>. Connects multi-step temporal-difference residuals to GAE. Its policy-gradient role is deferred to Module 4.
6. **Shao et al., *DeepSeekMath*.** — <https://arxiv.org/abs/2402.03300>. An author report describing the original GRPO group construction without a learned critic. It is not an independent replication of every empirical claim and does not cover all later variants.
7. **Official Gymnasium FrozenLake and CliffWalking pages.** — <https://gymnasium.farama.org/environments/toy_text/frozen_lake/> and <https://gymnasium.farama.org/environments/toy_text/cliff_walking/>. These document the canonical teaching environments. The numerical values in this course come from its own tested NumPy implementations under a stated protocol.

The expanded claim registry, verification status, and scope boundaries are in [`Module_2_Sources.md`](Module_2_Sources.md).

**Next: Module 3, bandits and exploration.** We temporarily remove multi-step dynamics and study the cost of exploration, upper-confidence methods, Thompson sampling, and the connection between choosing among several LLM responses and contextual bandits.
