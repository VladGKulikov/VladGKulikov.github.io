# Module 11. RL for LLM agents: tools, environments, and long trajectories

> **Material version:** 2026.23  
> **Explanatory text revised:** 2026-10-27  
> **Factual snapshot:** 2026-08-06  
> **Language:** EN  
> **Core practice:** tier A — browser and CPU; tier B — free Colab T4 or a paid API for the optional model run  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 10–12 hours without the optional model run

In earlier modules, one answer could often be treated as a complete trajectory: the model received a prompt, generated text, and received one reward. For an agent, an answer boundary is no longer a task boundary. The model calls a tool, observes the result, revises its plan, acts again, and reaches or misses the goal only after several turns.

**Reinforcement learning for LLM agents**, usually called **agentic reinforcement learning** or **Agentic RL**, studies this regime. A policy repeatedly interacts with an external environment, and the outcome depends not only on text quality but also on the sequence of actions, observations, side effects, and stopping rules.

The practical goal of the module is:

> **Design a trainable agent whose state, actions, loss mask, reward, and rollout log describe the same real system.**

After completing the module, you will be able to:

1. distinguish one-shot generation from an interactive episode;
2. formulate an agent task as an MDP, a POMDP, or a semi-Markov model;
3. separate hidden environment state, observation, history, and external memory;
4. separate policy tokens from user messages and tool outputs;
5. build a correct loss mask for a multi-turn trajectory;
6. retain behavior log-probabilities and the policy version for every turn;
7. distinguish trajectory-, turn-, and token-level loss reduction;
8. derive the telescoping identity for TD residuals;
9. explain what an outcome reward provides and what it does not identify about individual actions;
10. design potential-based shaping without changing the optimal policy when the theorem's assumptions hold;
11. recognize loops, reward farming, and verifier failures;
12. compute the probability that a binary-reward group is informative;
13. choose a difficulty band and group size instead of trying to repair degenerate data with the learning rate;
14. distinguish trajectory-level, sequence-level, branching, replay-based, and offline agent-training methods;
15. specify a reproducible and safe environment contract;
16. run a synthetic tool-using agent and train it on CPU.

The lecture can be read on its own: the definitions, examples, and reasoning needed to follow the argument are developed in the text.

---

> **Practice route.** [Exercises](/courses/rl-for-llm/module-11/exercises/), [review questions](/courses/rl-for-llm/module-11/tests/), and [executable practice](/courses/rl-for-llm/module-11/notebooks/) are provided separately. Reading the lecture does not require opening these materials first.

## Lesson 11.1. From an answer to environment interaction

### Step 11.1.1 — What changes when the model starts acting

Let an ordinary LLM receive a prompt $x$ and generate an answer $y$. In a one-shot abstraction, the entire answer can be treated as one action:

$$
y\sim\pi_\theta(\cdot\mid x).
$$

For an agent, an action produces a new observation that depends on the state of the world:

$$
a_t\sim\pi_\theta(\cdot\mid h_t),
$$

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),
\qquad
o_{t+1}\sim O(\cdot\mid s_{t+1}),
$$

$$
h_{t+1}=\operatorname{update}(h_t,a_t,o_{t+1}).
$$

Here $s_t$ is environment state, $o_t$ is an observation, $h_t$ is the history or other information state available to the policy, and $a_t$ is the agent's action. An action may be a structured function call, a shell command, a GUI click, or the final answer to the user.

![The LLM-agent and environment loop](assets/rl-for-llm/en/module-11/M11_agent_loop_EN.png)

The central distinction is simple: **the policy chooses an action, but it does not choose the tool's result**. The `pytest` command belongs to the agent; the test report belongs to the environment. A search query belongs to the agent; the returned documents belong to the environment. This distinction determines the decision process, the loss mask, and which log-probabilities exist at all.

ReAct historically popularized an interface that interleaves reasoning and acting. In this course, ReAct is a useful interaction pattern rather than a definition of Agentic RL. An agent may use a different planner, latent state, external memory, or a deterministic controller.

---

### Step 11.1.2 — State, observation, history, and memory are different objects

A deployed agent usually does not observe the full state $s_t$. A browser exposes the current page, a terminal exposes the latest command output, and an API exposes only selected fields. This is naturally modeled as a **partially observable Markov decision process** (POMDP).

Four levels should remain distinct.

1. **Environment state $s_t$.** File-system contents, database rows, open tabs, permissions, and hidden simulator variables.
2. **Observation $o_t$.** Text, an image, an error code, a DOM fragment, or test output returned after an action.
3. **History $h_t$.** The sequence of messages and actions presented to the policy on the current turn.
4. **External memory $m_t$.** A maintained summary, fact table, vector store, or log that the agent can read and update.

A history is not automatically a Bayesian **belief state**. It is merely the information on which the policy conditions. A belief state must represent a distribution over hidden states and be updated consistently with an observation model. An ordinary text context provides no such guarantee. For a fixed environment model and initial distribution, the complete action–observation–reward history can itself be a growing Markov information state. A belief summarizes that history for prediction without necessarily identifying the actual world state. A message window or summary that has lost relevant history has no such guarantee.

![Hidden state, observations, and agent history](assets/rl-for-llm/en/module-11/M11_state_history_EN.png)

Two engineering consequences follow.

- Truncating or summarizing context changes the policy's information state and may remove a fact needed later.
- Adding external memory changes the agent's state space. Writing to memory becomes part of the policy or a separate controller, and retrieval errors become part of system dynamics.

> **Extension:** ReAct, Plan-and-Execute, CodeAct, function calling, MCP, and production agent architecture are developed in *Modern LLMs*, Module 18 “Agents and Tool Use” ([open module](/courses/modern-llms/module-18/)). Here they appear only as interfaces inside an RL formulation.

---

### Step 11.1.3 — Token, tool call, and macro-action

At the transformer level, an action is still the next token. The environment, however, usually reacts to a completed structured call rather than to each token:

```json
{"name": "read_file", "arguments": {"path": "src/main.py"}}
```

It is useful to distinguish three scales.

- **Micro-action:** one token $y_k$.
- **Agent turn:** one assistant message, possibly containing reasoning, a tool call, or a final answer.
- **Macro-action:** a completed environment operation with variable duration and its own outcome.

If a macro-action lasts $\tau_t$ base time units, a **semi-Markov decision process** (semi-MDP) is natural. First define the reward accumulated during that macro-action as

$$
R_t^{(\tau_t)}
=
\sum_{j=1}^{\tau_t}\gamma^{j-1}r_{t,j}.
$$

The discounted continuation then uses $\gamma^{\tau_t}$:

$$
Q(s_t,a_t)
=
\mathbb{E}\left[
R_t^{(\tau_t)}
+
\gamma^{\tau_t}V(s_{t+1})
\mid s_t,a_t
\right].
$$

For an LLM agent, this is more than notation. A calculator call, a minute-long test run, and waiting for a user response have different costs. If the objective includes latency or API cost, action duration must appear in the reward, a constraint, or an explicit cost report.

Unless stated otherwise, the remainder of this module uses one completed turn as one time step. A single factor $\gamma$ therefore separates adjacent turns. This is a modeling convention, not a claim that every tool takes the same time. If discounting is tied to elapsed time instead, retain $\gamma^{\tau_t}$ and the matching within-action return. The two clocks should not be mixed silently in one objective.

There is another subtlety. Several strings may denote the same semantic action. Two JSON strings with different whitespace can parse to the same function call. The probability of that semantic action is the sum of the probabilities of all strings that the parser maps to it. Practical losses usually operate on the sampled string, so serialization and parsing rules are part of the behavior policy's definition.

The log should consequently preserve the route from sampled text to executed operation. If a controller repairs malformed JSON, the original text remains the model's sampled action; the repair is another transformation. Scoring the repaired string and assigning that likelihood to the original choice would describe an action the model did not actually take.

![Three action scales and variable duration](assets/rl-for-llm/en/module-11/M11_action_granularity_EN.png)

---

### Step 11.1.4 — Success, termination, truncation, and infrastructure failure

A multi-turn environment should distinguish at least four stopping causes.

| Cause | Example | Meaning for training |
|---|---|---|
| Success | tests pass or the target state is reached | terminal outcome with positive reward |
| Task failure | the agent violates a rule or enters an irreversible wrong state | terminal outcome with the specified reward |
| Technical truncation | an external time or length cap ends the run but is not a task rule | continuation is unknown; bootstrapping or exclusion may be required |
| Environment failure | the container fails, an API is unavailable, or the verifier crashes | not necessarily a policy error; record a separate status |

This course’s Module 2 “State, Value, and Bellman Equations” ([open module](/courses/rl-for-llm/module-02/)) separated `terminated` from `truncated`. For agents, the distinction is even more important: an expensive episode may be cut off by infrastructure even though the policy could still finish.

There is no universal rule that all truncated trajectories must be excluded or assigned zero reward.

- If the limit is **part of the task**, such as “solve in at most 15 calls,” exceeding it is a genuine terminal failure.
- If the limit is imposed only by the rollout collector, zero reward confounds task difficulty with bad behavior. A critic may require bootstrapping; a purely trajectory-level method needs an explicit exclusion or special handling policy.
- If the environment failed, the record should carry an infrastructure status rather than masquerade as task failure.

This bookkeeping belongs in the environment contract, not in an ad hoc analysis script.

---

### Step 11.1.5 — A minimal formal model of an agent episode

One agent episode can be recorded as

$$
\tau=
(s_0,h_0,a_0,o_1,r_1,
 s_1,h_1,a_1,o_2,r_2,\ldots,s_T).
$$

The policy need not observe $s_t$; it acts from information state $h_t$. The objective remains familiar:

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_{t+1}
\right].
$$

The trajectory distribution now contains external dynamics and observations:

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)
P(s_{t+1}\mid s_t,a_t)
O(o_{t+1}\mid s_{t+1}).
$$

This compact factorization treats the reward rule and history update as specified mechanisms and includes the initial observation in the initial setup. A stochastic reward would contribute its conditional law as well. The relevant assumption is that these environment mechanisms are fixed with respect to the trainable policy parameters.

For a recorded trajectory, explicit differentiation of its log-probability therefore leaves the policy action terms. The environment has not disappeared from the learning problem: its responses determine future context, rewards, and reachable continuations. By changing its choices, the policy changes which states it visits without differentiating through the implementation of a file system or search service.

A reproducible description requires more than the phrase “we used Agentic RL.” It must specify:

- what counts as an action;
- which object the policy observes;
- which messages are produced by the environment;
- how the episode ends;
- where reward is generated;
- which actions are irreversible;
- which parts of the trajectory enter the loss.

The next lesson develops the last item.

---

## Lesson 11.2. Trajectories, masks, and loss construction

### Step 11.2.1 — Who generated each token

An agent transcript may look like one sequence:

```text
system → user → assistant(tool call) → tool → assistant → tool → assistant(final)
```

For the policy, this sequence is heterogeneous.

- `assistant` segments were sampled from the behavior policy.
- `tool`, `user`, and `system` segments were inserted by the environment or template.
- Structured tool-call JSON is a policy action only if the model actually generated it.
- Service tokens appended after generation should not automatically be treated as actions.

The decisive criterion is token provenance rather than a role label:

> **A token belongs in the policy objective considered here when the behavior policy actually chose it under the recorded context and the probability of that choice is stored or can be reproduced exactly.**

If a deterministic controller constructs JSON from a latent model command, only the latent command is the learned action. If the controller repairs or rewrites an action, that transformation belongs to the environment or to a composite policy and must be logged separately.

Private or unavailable reasoning tokens cannot be inserted into an RL update after the fact unless their token identifiers and behavior log-probabilities were recorded.

---

### Step 11.2.2 — Trainable-token masks and signal dilution

Let $m_k\in\{0,1\}$ indicate whether token $z_k$ is a policy action. With one trajectory-level advantage $\widehat A$, a REINFORCE-like loss can be written as

$$
L(\theta)
=
-
\frac{1}{\sum_k m_k}
\sum_k
m_k\widehat A\log\pi_\theta(z_k\mid z_{<k}).
$$

![Masking a multi-turn trajectory](assets/rl-for-llm/en/module-11/M11_trajectory_mask_EN.png)

The diagram uses its own transcript size: 39 policy tokens out of 269 total tokens. The calculation below illustrates the same effect with round numbers; it is a separate numerical example, not a recalculation of the diagram.

The first question answered by the mask is **which choices belong to the policy**. After `inspect`, the door-color clue comes from the environment. There is no policy-loss term rewarding the model for having “generated” that color. The subsequent key choice, however, depends on the clue and is a trainable action.

Masking the clue as a target does not prevent gradients from passing through its processing. A loss on the next action can update the parameters that turn tool text into a useful context representation. **The loss mask excludes observations from the action targets; the attention mask controls which information the model can read.** In an ordinary causal forward pass, later policy tokens still attend to the observation. Detaching part of the computation or reusing a detached cache would be an additional implementation decision, not a consequence of the loss mask itself.

The mask's second role in the displayed expression is to specify the averaging denominator. We have chosen a mean over the policy tokens of one episode. That choice is separate from the requirement to exclude environment tokens from the numerator.

Suppose the policy generated 40 tokens and the environment added 290. Dividing by the full length 330 rather than by the 40 policy tokens reduces the contribution of each policy token by

$$
\frac{330}{40}=8.25.
$$

For this particular episode the ratio is a constant. Across episodes, tool-output length changes. Dividing by the entire transcript therefore downweights episodes with verbose observations even when the policy has little control over that verbosity.

A correct mask does not make the remaining normalization neutral either. The exact trajectory log-probability contains a **sum** of action log-probabilities. Dividing each episode by its own policy-token count changes the relative weight of short and long trajectories. The expression is therefore described as REINFORCE-like: it illustrates masking and one reduction, rather than declaring that reduction to be the unique form of the original policy gradient. Step 11.2.4 compares the alternatives.

Excluding observations from the policy loss does not forbid world-model training. Such training is possible, but it is a separate objective with a separate interpretation. It should not enter the policy gradient through an accidentally unmasked transcript.

---

### Step 11.2.3 — Log-probabilities must be captured at every turn

A one-shot generation call can return tokens and log-probabilities in one operation. An agent rollout contains several generation calls, with environment observations changing the context between them.

Suppose the key-selection call received the full door-color message, but the later door-opening call received only a controller-generated summary. The final log might contain both. Running the model over that complete log would expose the second action to information that was not in its original input. It would compute a different conditional probability, even if the action text were unchanged.

Reevaluation must reconstruct the **actual input to each call**: selected messages, summaries, retrieved memory, service templates, and positions. A recurrent policy also requires an account of its memory and how that memory was obtained. Evaluating a new policy normally means replaying its processing of the stored inputs; substituting an old hidden vector is a separate approximation.

For every policy token, the record should retain:

- token identifier;
- position and turn identifier;
- log-probability under the behavior policy;
- behavior-policy version;
- decoding parameters;
- the exact context or a sufficient recipe for reconstructing it.

A turn's log-probability is still a sum of token log-probabilities, but the conditioning context must be unambiguous. Below, $z_{<k}$ is shorthand for the context actually available when the token was chosen, not necessarily the complete prefix of the archived transcript. For token $k$ in turn $t$, a more explicit notation is

$$
c_{t,k}=\operatorname{serialize}(h_t,z_{t,<k}),
\qquad
\log\pi_\theta(z_{t,k}\mid c_{t,k}).
$$

If history is append-only and serialization is unchanged, one causal pass over the combined sequence may reproduce all these inputs. When context is shortened or rearranged, the training examples must preserve the corresponding per-turn views. Under that convention, ordinary factorization gives:

$$
\log\pi_{\mathrm{beh}}(a_t\mid h_t)
=
\sum_{k\in\mathcal I_t}
\log\pi_{\mathrm{beh}}(z_k\mid z_{<k}).
$$

If training uses a new policy $\pi_\theta$, the turn-level sequence ratio is

$$
r_t^{\mathrm{seq}}
=
\exp\left(
\sum_{k\in\mathcal I_t}
\left[
\log\pi_\theta(z_k\mid z_{<k})
-
\log\pi_{\mathrm{beh}}(z_k\mid z_{<k})
\right]
\right).
$$

Two common errors are easy to miss.

- Recomputing behavior log-probabilities with a different model, tokenizer, or chat template.
- Using base-softmax probabilities instead of the actual behavior distribution after temperature, top-p, masks, and other transformations.

This course’s Module 6 “The LLM as a Policy: Tokens, Log-Probabilities, and KL” ([open module](/courses/rl-for-llm/module-06/)) established this contract for one answer. An agent system must satisfy it at every turn.

---

### Step 11.2.4 — Trajectory-, turn-, and token-level reduction

Even with identical token-level loss terms, the final gradient depends on the order of averaging.

**Token-level reduction:**

$$
L_{\mathrm{token}}
=
\frac{
\sum_i\sum_{k\in\mathcal I_i}\ell_{i,k}
}{
\sum_i|\mathcal I_i|
}.
$$

**Trajectory-level reduction:**

$$
L_{\mathrm{traj}}
=
\frac1B
\sum_{i=1}^{B}
\frac1{|\mathcal I_i|}
\sum_{k\in\mathcal I_i}\ell_{i,k}.
$$

**Turn-level reduction:**

$$
L_{\mathrm{turn}}
=
\frac1{\sum_i T_i}
\sum_i\sum_{t=1}^{T_i}
\frac1{|\mathcal I_{i,t}|}
\sum_{k\in\mathcal I_{i,t}}\ell_{i,t,k}.
$$

The three objectives answer different statistical questions.

Here, weight means an arithmetic coefficient on a loss term, not the norm of the resulting parameter-gradient vector. Different actions can produce gradients of different magnitudes and directions. Equal total coefficients do not force two episodes to have equally large effects on the parameters.

- Token reduction gives more weight to longer answers.
- Trajectory reduction gives each episode equal total weight.
- Turn reduction gives each turn equal weight, although longer episodes still contain more turns.

No reduction is unconditionally correct. The statistical unit must be stated, and the chosen denominator must be checked for unwanted incentives toward length, brevity, or excessive tool calls.

---

### Step 11.2.5 — The multi-turn rollout contract

A minimal rollout record should answer five questions:

1. what the policy did;
2. what the environment returned;
3. which policy version chose each action;
4. why the episode stopped;
5. whether the initial state and verification can be reproduced.

A practical contract contains:

```text
task_id
trajectory_id
environment_id
environment_version
initial_state_id / reset_seed
policy_version per turn
token_ids
policy_mask
behavior_logprobs
turn_ids
action_text and parsed_action
observations
reward components
terminated / truncated / infra_error
stop_reason
wall-clock and tool latency
verifier_version
```

A text transcript alone is insufficient. Identical-looking messages may have different tokenizations; identical calls may execute against different database states; identical rewards may be produced by different verifier versions.

A useful pre-training audit reconstructs from the log:

- the number of trainable tokens;
- each turn's log-probability;
- the stopping cause;
- the final reward;
- the environment snapshot checksum.

If these quantities cannot be recovered, the rollout may be suitable for a demo but not for reproducible RL.

---

## Lesson 11.3. Credit assignment over long horizons

### Step 11.3.1 — What one outcome reward provides

Start with a finite, undiscounted episode, $\gamma=1$, receiving one outcome reward $R(\tau)$. The simplest policy-gradient estimator is

$$
\widehat g
=
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid h_t).
$$

With on-policy sampling, fixed environment mechanisms, and the usual differentiation conditions, this is an unbiased estimator of the expected-outcome gradient. For a discounted objective, the same expression applies when $R(\tau)$ denotes the complete discounted return; an undiscounted terminal score cannot silently replace it. Trajectories from an earlier policy require a separate justification for their sampling distribution.

The same return may legitimately multiply every action. Unlike the illustrative reduction in Step 11.2.2, this estimator does not divide each episode by its own length. Its principal difficulties lie elsewhere.

- **High variance.** A random early choice, a lucky tool response, and a late mistake are compressed into one number.
- **Coarse credit.** Every action in a successful trajectory receives the same sign, including redundant or harmful actions.
- **Long horizon.** The number of plausible causes grows while environment rollouts remain expensive.

A suitable baseline can reduce variance. Action independence preserves the expected gradient when the baseline is used correctly; it does not promise that an arbitrary inaccurate predictor will reduce noise. Even a useful baseline does not identify the decisive turn by itself. Group-relative advantages compare trajectories for the same prompt, but remain trajectory-level when one coefficient is broadcast to every turn.

Outcome reward is therefore a valid starting point rather than a complete theory of credit assignment.

---

### Step 11.3.2 — Turn returns and TD residuals

When the environment or a critic provides information at turn boundaries, the familiar objects from this course’s Module 2 “State, Value, and Bellman Equations” ([open module](/courses/rl-for-llm/module-02/)) and Module 4 “Policy Gradients: From REINFORCE to PPO” ([open module](/courses/rl-for-llm/module-04/)) become available.

The return from turn $t$ is

$$
G_t
=
\sum_{j=t}^{T-1}\gamma^{j-t}r_{j+1}.
$$

A one-step **temporal-difference residual** (TD residual) is

$$
\delta_t
=
r_{t+1}
+
\gamma V(h_{t+1})
-
V(h_t).
$$

Without premature zeroing of the terminal value, the residuals satisfy

$$
\sum_{t=0}^{T-1}\gamma^t\delta_t
=
G_0
-
V(h_0)
+
\gamma^T V(h_T).
$$

![The telescoping TD identity](assets/rl-for-llm/en/module-11/M11_credit_telescope_EN.png)

The proof is short: substitute $\delta_t$ and observe that each intermediate term $+\gamma^{t+1}V(h_{t+1})$ cancels the next $-\gamma^{t+1}V(h_{t+1})$.

This identity is an excellent implementation check, but it does not turn $\delta_t$ into a causal diagnosis. A small residual can result from critic error, a noisy observation, tool failure, hidden state, or an action whose value appears much later.

A TD signal measures disagreement among reward and successive value estimates. It does not automatically identify a guilty action.

---

### Step 11.3.3 — Process feedback and hindsight critics

Three common routes produce denser feedback.

1. **Verifiable intermediate events.** A file exists, a test module was found, or the correct key was picked up. These events should be checked against world state.
2. **A process reward model** (PRM). It scores an intermediate prefix, but its meaning depends on the label definition and training distribution.
3. **A hindsight critic.** After the episode ends, it analyzes the trajectory and assigns turn- or transition-level scores.

Density is not truth. A hindsight model may produce a persuasive causal story and still be wrong. A PRM may learn style cues. A verifiable milestone may be farmed in a loop.

The next question is where the new number enters learning. Adding it to reward, using it as a supervised target for a critic, and multiplying a policy log-gradient by it are different operations. The word “score” does not establish that they optimize the same objective.

A completed outcome can legitimately supply a training target for a predictor that conditions on the current state. But a retrospective score computed after observing the chosen action and its consequences is not automatically an action-independent baseline. Detaching that number removes a computational derivative; it does not remove its statistical dependence on the action. Using it in an update needs an estimator argument, not merely a convincing account of the episode.

For each intermediate signal, ask:

- which object is being estimated;
- who produced the label;
- whether it can be checked independently;
- whether the value depends on the current continuation policy;
- whether the agent can collect the same reward repeatedly;
- whether the signal changes the original optimal policy.

The final question leads to potential-based shaping.

---

### Step 11.3.4 — Potential-based reward shaping

A naive reward such as “$+0.1$ for every useful-looking step” often creates a loop: the agent repeats the paying action instead of completing the task.

A safer starting point is **potential-based reward shaping**. For a potential function $\Phi(s)$, add

$$
F(s_t,s_{t+1})
=
\gamma\Phi(s_{t+1})-\Phi(s_t),
$$

$$
r'_{t+1}
=
r_{t+1}+F(s_t,s_{t+1}).
$$

![Naive progress rewards and potential-based shaping](assets/rl-for-llm/en/module-11/M11_reward_shaping_EN.png)

The diagram ends its illustrated fragment at a state with nonzero potential. Treating that vertex as a true terminal requires a boundary convention—the issue developed next.

The discounted shaping terms telescope:

$$
\sum_{t=0}^{T-1}\gamma^tF_t
=
-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

Why does this cancellation preserve a policy preference? Compare complete paths from the same initial state. If **every true terminal state has zero potential**, the final term vanishes regardless of the path taken or the number of turns. Under this sufficient episodic convention,

$$
\begin{aligned}
G'_0 &= G_0-\Phi(s_0),\\
V'^{\pi}(s) &= V^{\pi}(s)-\Phi(s),\\
Q'^{\pi}(s,a) &= Q^{\pi}(s,a)-\Phi(s).
\end{aligned}
$$

Every action from a given state has received the same shift in expected return. Shaping alone has not made one action preferable to another. The same conclusion holds for policy comparisons under a fixed initial-state distribution. This is the reason for invariance; it is not a special ability of the potential to recognize good actions.

Consider the key-and-door task. A designer may assign higher potential to selected intermediate stages of progress, but cannot leave an arbitrary bonus at the terminal boundary. Otherwise, the agent would optimize the original task reward plus the remaining potential. A negative shaping increment on the final transition is not necessarily a mistake: with zero terminal potential, it can be the correction that balances earlier progress bonuses.

An arbitrary $\Phi(s_T)$ makes the residual depend on the reached outcome. Even the same nonzero constant at every terminal can alter timing preferences when $\gamma<1$ and episode lengths differ, because the residual contains $\gamma^T$. Boundary handling is consequently part of the construction. In a continuing discounted problem, the corresponding tail vanishes as the horizon grows when the potential is bounded.

A collector's truncation is a different boundary. If recording stops while the task continues, setting the remaining potential to zero silently treats that interruption as task termination. A bootstrapped continuation must be transformed consistently with the shaped reward. If the limit really defines the task, include it in the state—for example as remaining budget—and terminate that finite task explicitly.

The classical policy-invariance claim therefore comes with the following conditions.

- The shaping term has exactly the potential form with the same $\gamma$.
- The state used by $\Phi$ belongs to the Markov formulation.
- Boundary terms are handled consistently: the sufficient episodic convention above sets all terminal-outcome potentials to zero; a continuing formulation needs its tail-vanishing conditions.
- The reward and stopping rules are not changed elsewhere.
- Under partial observability, specify the state or history on which the potential depends and what information the policy receives. Telescoping also works for a sequence of information states, but transferring invariance to the task still requires consistent boundaries and an unchanged class of admissible policies. If the shaping reward is exposed to the acting policy and reveals new hidden information, its information problem has changed as well.

Potential-based shaping preserves optimality under the theorem's assumptions. It does not guarantee faster learning, accurate value approximation, or stable neural-network optimization.

The resemblance to the TD identity in Step 11.3.2 is useful for checking the algebra, but the two mechanisms do different jobs. TD compares rewards with continuation predictions. Shaping changes the rewards and controls what that change does to the complete return. The potential need not be an accurate value function for the identity to hold. With duration-aware macro-actions, the matching increment is $\gamma^{\tau_t}\Phi(s_{t+1})-\Phi(s_t)$, not a single factor $\gamma$ regardless of elapsed time.

[Grześ, *Reward Shaping in Episodic Reinforcement Learning*](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf) examines this episodic boundary issue in detail. The distinction needed here is narrower: producing denser feedback and preserving preferences among complete paths are separate requirements.

---

### Step 11.3.5 — Verify world state, not persuasive text

An agent's final reward should, whenever possible, depend on the outcome in the external environment.

- For software tasks: the modified repository and test results.
- For browser tasks: database, cart, or publication state.
- For shell tasks: file contents, exit codes, and system invariants.
- For conversational tool tasks: the completed operation and policy compliance, not merely a polite final response.

The sentence “done” does not prove completion. Conversely, a clumsy final message may accompany a correct world state.

A practical verifier should return diagnostics rather than only one scalar:

```text
success
world_state_checks
format_checks
safety_checks
invalid_actions
resource_usage
infra_status
```

Useful partial credit is based on monotone or first-visit changes of state. The first successful login may be rewarded once; repeated `login` calls should not be paid repeatedly. Even then, the designer must verify that the partial reward does not reorder the task's intended preferences.

The lesson's main rule is:

> **Reward should describe the desired change in the world; agent text is only one possible means of causing that change.**

---

## Lesson 11.4. Exploration and agent-policy optimization

### Step 11.4.1 — When a rollout group contains no relative signal

Suppose $G$ independent trajectories are generated for one prompt and reward is binary. If one trajectory succeeds with probability $p$, the group contains relative signal only when it includes both a success and a failure.

For an agent, “one prompt” must include a clear account of the task instance. Identical instructions can start from different database contents, object locations, or door colors. Below, $p$ refers either to one declared instance or to one fixed distribution of instances from which each episode independently begins. If a hidden initial state is sampled once for the whole group, independence must instead be considered conditional on that state before averaging across states. Each execution also needs an isolated environment: one attempt's side effect must not become another attempt's unrecorded clue.

The probability of a degenerate group is

$$
P_{\mathrm{deg}}
=
p^G+(1-p)^G.
$$

The probability of an informative group is

$$
P_{\mathrm{info}}
=
1-p^G-(1-p)^G.
$$

![Probability that a rollout group is informative](assets/rl-for-llm/en/module-11/M11_group_informativeness_EN.png)

For $p=0.01$ and $G=8$,

$$
P_{\mathrm{info}}
\approx0.0773.
$$

The expected number of groups needed to see one internal comparison is therefore approximately

$$
\frac1{P_{\mathrm{info}}}\approx12.94.
$$

A larger learning rate cannot create a missing relative reward signal: a zero coefficient remains zero. A separate KL penalty or auxiliary objective may still have a nonzero gradient. More direct ways to obtain a reward comparison include:

- increasing group size;
- selecting tasks of intermediate difficulty;
- improving the initial SFT policy;
- changing decoding to increase useful diversity;
- adding verifiable partial feedback;
- replaying or using offline data with explicit distribution-shift controls;
- dynamically filtering zero-variance tasks.

---

### Step 11.4.2 — Difficulty bands and dynamic task filtering

Tasks with $p\approx0$ almost always produce all failures; tasks with $p\approx1$ produce all successes. At a fixed group size, observing both outcomes is more likely between the extremes. That is a statement about nondegenerate internal comparisons, not an automatic measure of learning value or transfer.

This creates a **difficulty band**. It is not a fixed dataset: once the policy improves, yesterday's difficult prompt may become easy.

A simple dynamic protocol is:

1. estimate policy success for each task family;
2. retain minimum sample counts and uncertainty intervals;
3. sample more often where both success and failure remain plausible;
4. keep some easy tasks to monitor forgetting;
5. keep some hard tasks to detect frontier expansion;
6. never drop a task permanently after one unlucky group.

Degeneracy must be computed **conditional on the prompt**. A global mean success rate of 0.5 may hide a mixture of tasks with $p=0.01$ and $p=0.99$; almost every prompt-specific group remains degenerate.

Dynamic sampling changes the training distribution. A report should therefore include:

- the original task distribution;
- the sampled rollout distribution;
- the sampling rule or probability;
- metrics on a fixed evaluation set.

Otherwise, improvement within the training band can be mistaken for general progress.

---

### Step 11.4.3 — Token-level and sequence-level probability ratios

For one turn, token ratios are

$$
r_{t,k}
=
\frac{
\pi_\theta(z_{t,k}\mid h_t,z_{t,<k})
}{
\pi_{\mathrm{beh}}(z_{t,k}\mid h_t,z_{t,<k})
},
$$

while one sequence ratio is

$$
r_t^{\mathrm{seq}}
=
\prod_k r_{t,k}.
$$

For a whole trajectory, the product extends across turns. On long horizons it becomes numerically and statistically fragile: small differences multiply, and rare large weights dominate.

Why are tool-output probabilities absent from the product? Compare the probability of the same recorded trajectory under two policies while holding the initial distribution, transitions, and observation mechanism fixed. The environment factors in Step 11.1.5 are identical and cancel. Subject to the required support condition,

$$
\frac{p_\theta(\tau)}{p_{\mathrm{beh}}(\tau)}
=
\prod_{t=0}^{T-1}
\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\mathrm{beh}}(a_t\mid h_t)}
=
\prod_{t=0}^{T-1}r_t^{\mathrm{seq}}.
$$

The cancellation relies on an unchanged environment; it does not make tool outcomes irrelevant. A new database version, modified tool, or changed verifier can alter the task itself. Policy ratios alone do not correct that replacement. If behavior versions differ across turns, each denominator must describe the choice actually made on that turn.

Practical methods therefore make different compromises:

- clip local ratios;
- aggregate at token, turn, or sequence level;
- limit rollout age;
- replay only selected trajectories;
- add a behavior anchor;
- replace direct importance weighting with offline preferences.

These choices do not all optimize one identical estimator. A comparison must state:

- where the ratio is computed;
- which denominator is used;
- how length is normalized;
- how many updates one rollout receives;
- how old the behavior policy may be;
- which rule rejects extreme weights.

Module 12 “RL Infrastructure for LLMs: Memory, Rollouts, and Asynchrony” ([open module](/courses/rl-for-llm/module-12/)) develops policy lag and asynchronous pipelines. Here the essential point is that long agent trajectories magnify the off-policy problem.

---

### Step 11.4.4 — Branching after tool observations and ARPO

Regenerating a full trajectory spends compute on repeating identical early turns. An alternative is to branch where a tool observation makes the next decision uncertain.

Let a shared prefix reach $h_t$. Generate several continuations:

$$
a_t^{(1)},\ldots,a_t^{(K)}
\sim
\pi_\theta(\cdot\mid h_t).
$$

Return to the key-and-door setting. The agent has called `inspect` and received the clue. Several continuations can now be compared without repeating the initial inspection. Before branching, however, preserve **both sides of the situation**: the world after inspection and the information from which the policy will act. For a recurrent agent this includes the memory carrying the color; for an LLM it includes the actual context and controller state.

Give each branch a separate restored environment: the same hidden color, carried key, door state, and remaining task budget. The policy then chooses a continuation, the environment executes it, and the verifier checks the outcome. Picking up a key in one branch must not alter another branch's world. A stochastic environment also needs an explicit randomness protocol: a shared starting state does not require all future random outcomes to be identical.

This produces comparisons **after the same observation**. Copying the clue text is not enough. In a real service, identical messages can conceal different files, sessions, or completed operations. If the relevant world state cannot be restored, branching may be suitable for a simulator but not for a live resource with irreversible effects.

The outcomes must then be turned into learning data. Distinct continuations provide evidence about choices after the branch point. The shared prefix was obtained once; several descendant outcomes are not independent repetitions showing that the earlier action should have been chosen. They can inform an estimate of continuation value, but the prefix's statistical weight and advantage assignment need an explicit rule. Duplicating it across training rows otherwise changes its contribution simply because more branches were explored.

Cost has the same distinction. Earlier calls were executed once, but snapshots, new continuations, and verification still consume resources. Storing the same text in several rows does not mean it was generated repeatedly, just as saving prefix generation does not make branch execution free.

This design may offer a more local comparison, but raises new questions:

- who chooses the branch point;
- whether environment state can be restored at that point;
- whether branches share the same hidden state;
- how common-prefix cost is counted;
- how credit is assigned before branching;
- whether branch selection changes the training distribution.

**Agentic Reinforced Policy Optimization** (ARPO) uses uncertainty-aware adaptive branching after tool interaction together with a step-oriented advantage attribution mechanism. It is a specific 2025 algorithm, not a generic name for all agent RL.

ARPO distinguishes shared segments from branch-specific segments when assigning advantages. Their common origin matters; repeated prefix tokens are not new independent choices. This is one concrete response to the questions above, not a guarantee that every branching experiment supplies causally exact credit. The details belong to the [original algorithm](https://arxiv.org/html/2507.19849v1).

Module 10 used a similar search structure to choose an answer with fixed weights. Here, branch outcomes become data for changing the policy. A snapshot and a tree can look the same in both systems, while serving different purposes: selecting a result now and learning how to behave later.

A different method is called **Agentic Replay Policy Optimization**, also abbreviated ARPO. It reuses successful experience when training GUI agents. The acronym collision is genuine; a table or first mention must expand the name.

---

### Step 11.4.5 — A map of contemporary Agentic RL methods

Contemporary methods are easier to distinguish by three questions than by branding: **where states come from**, **at what granularity the learning signal is constructed**, and **how closely the data match the current policy**.

![A map of Agentic RL methods](assets/rl-for-llm/en/module-11/M11_method_map_EN.png)

| Method or line | Data | Signal unit | Addition | Main boundary |
|---|---|---|---|---|
| Search-R1 / ReTool | fresh multi-turn rollouts | task outcome | real search or interpreter calls and environment-token masking | narrow verifiable environment and expensive rollouts |
| RAGEN / StarPO | fresh environment trajectories | trajectory and state | stabilization of the long interactive loop | sensitivity to environment and filtering design |
| Agentic Reinforced Policy Optimization | fresh branching trajectories | post-tool turn and outcome | uncertainty-aware branching and step-oriented attribution | environment state must be restored correctly at the branch point |
| AT$^2$PO | a tree of multi-turn continuations | turn | tree expansion and turn-wise credit assignment | search changes the collected data distribution |
| TRACE / Turn-level Reward Assignment via Credit Estimation | fresh long trajectories | turn or tool boundary | denser attribution to individual actions | signal quality depends on the chosen state estimator |
| Agentic Replay Policy Optimization / DSPO | replayed or dynamically selected trajectories | trajectory or sequence | replay or dynamic filtering | off-policy bias and changed task distribution |
| Agent Lightning | traces from an arbitrary agent | hierarchical transitions | decoupled execution and a credit-assignment layer | trace and decomposition correctness |
| Agentic-DPO | expert states and one-step negative actions | state-conditioned action preference | offline optimization without new full rollouts during gradient steps | expert coverage and negative-action quality |

This map reflects sources available on **6 August 2026** and is not a leaderboard. Even the word “turn” may refer to an assistant message, a tool call, an environment transition, or a tree node. A comparison is meaningful only after aligning the environment, model, compute budget, verifier, and reduction unit.

The practical choice begins with one question: **can the system obtain new world states safely and cheaply enough?** If yes, online RL can learn from current-policy mistakes and visit new states. If no, replay and offline preferences are cheaper but depend more strongly on coverage. A separate decision concerns where exact outcome signals suffice, where a critic is acceptable, and where branching truly restores the same environment state.

---

## Lesson 11.5. Environments, safety, and reproducible practice

### Step 11.5.1 — A training environment is stricter than a demo environment

A demonstration agent may open a website or run a script once. A training environment must survive thousands of repetitions and answer, on every episode, “which task was instantiated and what state did it start from?”

Minimum properties include:

1. **Reset.** Return to a known initial state.
2. **Versioning.** Record the container image, data, rules, tools, and verifier.
3. **Isolation.** One rollout cannot alter another's initial state.
4. **Deterministic seeds where possible.** Environment randomness is logged rather than disappearing.
5. **World-state verification.** Outcomes are checked programmatically or through a transparent protocol.
6. **Infrastructure observability.** Latency, errors, retries, and timeouts are retained.
7. **Resource limits.** CPU, memory, network, disk, and call counts have explicit caps.

A benchmark is not automatically a training environment. Training additionally requires reset, step interfaces, safe execution, and adequate throughput. Conversely, a training environment is not automatically a fair benchmark: the policy may have seen its states, rules, or verifier.

Current TRL documentation supports per-rollout state through `environment_factory` and integrations such as OpenEnv and Harbor. This is a dated property of one library rather than an algorithmic requirement; the API must be rechecked before use.

---

### Step 11.5.2 — Sandboxes, permissions, and irreversible actions

Training amplifies an action through repeated execution. Production privileges are therefore unsafe merely because one demo looked harmless.

A baseline permission model is:

- no network or external secrets by default;
- an ephemeral or snapshotted file system;
- an allowlist of tools and arguments;
- confirmation or prohibition for dangerous operations;
- a timeout and output cap for every call;
- limits on child processes and resource consumption;
- an immutable side-effect log;
- a verifier isolated from the agent.

Actions that cannot be undone by `reset` are especially dangerous: sending a message, purchasing an item, deleting a remote object, publishing content, or modifying an account. Training should replace them with a simulator, shadow mode, or a two-phase “prepare → verify → commit” protocol.

Observations may also be adversarial. A webpage or file can contain instructions that attempt to override the agent's task. The trust boundary must therefore be part of the state: environment content is data, not a system instruction.

Safety is not an appendix to RL here. It defines the permitted action space and therefore the optimization problem itself.

---

### Step 11.5.3 — Environments and benchmarks: what they actually test

Different environment families measure different capabilities.

- **WebArena** evaluates interaction with reproducible web applications and functional outcome checks.
- **OSWorld** covers multimodal control of real desktop applications and cross-application workflows.
- **$\tau$-bench** combines an agent, a simulated user, APIs, and domain rules; the final database state is compared with an annotated goal state.
- **SWE-bench and related environments** require repository changes, test execution, and patch verification.
- **Search and computation environments** expose sequential search, Python, or calculator calls with outcome verification.

A benchmark number combines at least:

- base-model quality;
- observation format;
- controller and memory;
- decoding parameters;
- action budget;
- environment stability;
- verifier behavior;
- retry strategy.

Thus, “the agent improved on a benchmark” does not by itself show that RL improved the policy. Ablations should compare against SFT, prompting, Best-of-N, retries, and infrastructure changes.

A useful evaluation hierarchy separates:

- training environments;
- held-out tasks in known environments;
- new initial states;
- new tools and schemas;
- new environment families.

The farther evaluation moves from training, the more transfer it tests and the harder error attribution becomes.

---

### Step 11.5.4 — Practice: a key-and-door tool agent

The hidden environment state contains the target-door color, whether the room was inspected, the carried key, the opened door, and the remaining task budget. The color clue appears only after `inspect`; the recurrent policy must retain it internally on later turns. Available macro-actions are:

```text
inspect
pickup_red
pickup_blue
open_red
open_blue
finish
```

The controller uses a **state-phase action-admissibility mask**—for example, it does not expose `open_*` before a key has been picked up—but it does not reveal the hidden color. A command can be syntactically valid yet unavailable in the current phase. The mask removes meaningless actions without solving the semantic choice for the policy.

The practical route demonstrates the complete loop:

1. `reset`, `step`, task termination, and collector truncation;
2. action/observation traces, policy versions, and behavior log-probabilities;
3. sparse outcome reward and potential-based shaping;
4. a recurrent PyTorch actor–critic policy;
5. on-policy rollout collection and return-based updates;
6. separate greedy evaluation on red and blue tasks;
7. verification that the potential term densifies feedback while telescoping into a boundary term.

![Training a recurrent agent in the key-and-door environment](assets/rl-for-llm/en/module-11/M11_training_results_EN.png)

The smoke profile runs on CPU. The plotted numbers belong to the course’s fixed synthetic environment and do not transfer to real LLMs. The full profile increases the number of independent runs. An optional Qwen3-0.6B branch with `GRPOTrainer(environment_factory=...)` is provided as an engineering template, disabled by default, and not used for quality claims.

The point is not to call a small recurrent agent a “realistic LLM.” The realistic part is the **contract**: partial observability, multi-turn actions, environment state, token provenance, stop reasons, policy versions, and reproducible outcome verification.

The verifier is allowed to know the true door color and inspect world state. That does not grant the acting policy the same information. Only the declared observations and signals should reach it. In particular, a mask that depended on the hidden color would reveal part of the solution, unlike a mask based on an openly known task phase. Reward design and branch restoration both need this information-boundary check.

---

### Step 11.5.5 — The final agent-experiment protocol

Before running multi-turn RL, freeze six layers.

**1. Task and environment**

- initial-state distribution;
- observations and hidden variables;
- available actions;
- reset, versions, and seeds;
- success and failure conditions.

**2. Behavior policy**

- model, tokenizer, and template;
- tools and serialization;
- decoding;
- weight version at every turn;
- controller and memory.

**3. Rollout log**

- tokens, mask, and log-probabilities;
- actions and observations;
- environment versions;
- stopping causes;
- time, cost, and errors.

**4. Reward and credit**

- final world-state check;
- partial credit;
- potential-based shaping;
- critic or PRM;
- treatment of truncation and infrastructure failure.

**5. Update**

- grouping unit;
- token-, turn-, or trajectory-level reduction;
- probability ratios and clipping;
- data reuse;
- maximum behavior-policy age.

**6. Evaluation**

- success and repeated-trial reliability;
- turns and tool calls;
- cost and latency;
- infrastructure failures;
- transfer to held-out states and environments;
- safety audit and unintended side effects.

The module's central idea is now complete:

> **Agentic RL does not begin with a new optimizer. It begins by defining where the policy ends, where the environment begins, and which data connect an action to a change in the world.**

Module 12 “RL Infrastructure for LLMs: Memory, Rollouts, and Asynchrony” ([open module](/courses/rl-for-llm/module-12/)) carries this contract into a distributed pipeline: generation and training move into separate processes, introducing stragglers, weight versions, queues, freshness controls, and systems cost.

---

## Sources and snapshot status

The primary sources below support the main external claims. Formula conventions and the scope of the synthetic practice are stated in the relevant steps; executable materials are available through the practice links.

- [Kaelbling, Littman, and Cassandra — *Planning and Acting in Partially Observable Stochastic Domains*](https://www.sciencedirect.com/science/article/pii/S000437029800023X) provides the classical distinction among hidden state, observation, and belief state; it does not make an LLM transcript an exact belief state automatically.
- [Ng, Harada, and Russell — *Policy Invariance Under Reward Transformations*](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) states the potential-based shaping result and its conditions; arbitrary progress bonuses do not inherit the guarantee.

- [Grześ — *Reward Shaping in Episodic Reinforcement Learning*](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf) examines terminal potentials and the boundary term in finite episodes.
- [Spinning Up — *Introduction to Policy Optimization*](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) develops trajectory log-gradients and the assumptions behind baseline subtraction.
- [ReAct](https://arxiv.org/abs/2210.03629), [Search-R1](https://arxiv.org/abs/2503.09516), and [ReTool](https://arxiv.org/abs/2504.11536) connect the reasoning–action–observation interface to multi-turn training with search or execution. Their empirical results remain specific to the reported environments and budgets.
- [Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849), [AT$^2$PO](https://arxiv.org/abs/2601.04767), [TRACE](https://arxiv.org/abs/2607.13988), and [ToolVerse](https://arxiv.org/abs/2607.15660) implement different forms of branching and turn-level attribution. Their shared vocabulary does not imply identical definitions of a turn or an identical estimator.
- [Agent Lightning](https://arxiv.org/abs/2508.03680) and [Agentic-DPO](https://arxiv.org/abs/2607.10601) respectively decouple agent execution from training and convert expert states into offline action preferences; neither removes trace-correctness or coverage requirements.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer), [OpenEnv](https://huggingface.co/docs/trl/openenv), and [Harbor](https://huggingface.co/docs/trl/harbor) document a dated engineering contract for stateful environment training. Pin the actually installed version before execution.
- [WebArena](https://arxiv.org/abs/2307.13854), [OSWorld](https://arxiv.org/abs/2404.07972), [$\tau$-bench](https://arxiv.org/abs/2406.12045), and [SWE-bench](https://arxiv.org/abs/2310.06770) represent different environment classes. A final score entangles the model, controller, budget, infrastructure, and verifier.

The methods and software survey retains its factual snapshot of **6 August 2026**. The explanatory revision dated **27 October 2026** clarifies existing mechanisms without updating that survey as a whole. Practice results remain course-owned synthetic experiments, not external benchmarks.
