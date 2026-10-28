# Module 5. Learning from fixed data: imitation, deep Q-methods, and offline RL

> **Material version:** 2026.10  
> **Text revision:** 2026-10-26  
> **Factual snapshot:** 2026-08-05  
> **Language:** EN  
> **Core practice:** NumPy for local calculations plus a separate English PyTorch notebook with a self-contained KeyDoorGrid environment, BC, DAgger, DQN, Double DQN, a BCQ-like constraint, CQL, IQL, FQE, Decision Transformer, and a token-level ILQL-like pipeline  
> **Structure:** 5 lessons, 25 steps  
> **Estimated time:** 9–13 hours including the main notebook; the full experiment profile is optional

In [Module 4](/courses/rl-for-llm/module-04/), an updated policy could collect another batch of trajectories and find out whether its behavior had improved. This module asks what remains possible **when there will be no next batch from the environment**. Demonstrations, transition logs, rewards, and preference pairs are still available. The learner can revisit them repeatedly, but it cannot ask what would have happened after a choice that the records never contain.

Avoiding fresh interaction can reduce both collection costs and the risks of exploration. It also removes a way of correcting mistaken predictions. An unseen action may remain uncertain not because the network is too small, but because several incompatible explanations fit exactly the same observations.

We will approach that restriction in stages. Behavioral cloning first shows what can be learned by imitating demonstrations. DAgger then adds labels at states visited by the learner, while ordinary DQN keeps replenishing a transition buffer. **These two methods still have access to new interaction.** They let us see what additional experience can repair before Lesson 5.3 takes that option away. We will then study methods designed for a fixed dataset, followed by Decision Transformer and language-based ILQL: two different ways of turning recorded trajectories into behavior.

No separate background course in classical RL is needed. We will build on the value functions, bootstrapping, and behavior–target policy distinction already developed, introducing each new construction when those familiar tools encounter a specific limitation.

**Learning objectives.** By the end of the module, you will be able to:

1. write behavioral cloning as maximum likelihood and state when token-level SFT is a special case;
2. distinguish the expert state distribution from the distribution induced by the learned policy;
3. derive the exact loss in a simple tightrope problem and explain the worst-case quadratic dependence on horizon;
4. describe Dataset Aggregation (DAgger) without confusing expert action labels with preferences or scalar rewards;
5. construct the Deep Q-Network (DQN) target and explain the distinct roles of replay memory and the target network;
6. compute a Double DQN target and identify the source of maximization bias;
7. distinguish offline RL from ordinary off-policy learning and from behavioral cloning;
8. formulate the data-coverage problem and prove a simple impossibility result for an unobserved action;
9. explain Batch-Constrained Q-learning (BCQ), Twin Delayed Deep Deterministic Policy Gradient plus behavioral cloning (TD3+BC), Conservative Q-Learning (CQL), and Implicit Q-Learning (IQL) without claiming universal superiority;
10. describe Decision Transformer as a conditional sequence model and correctly update a desired return-to-go;
11. separate SFT, classical offline RL, Direct Preference Optimization (DPO), Implicit Language Q-Learning (ILQL), and online policy optimization for LLMs.

Exercises, quizzes, and executable practice are published on separate pages for this module. Worked exercise solutions expand on request. The English and Russian executable routes share a computational core and use independently written explanations. The [source list](#sources-and-further-reading) identifies the primary papers and the scope of the claims supported by each.

The practical thread uses a single compact environment, **KeyDoorGrid**. The agent starts from several cells, must collect a key, pass a locked door, and reach the goal while avoiding traps. A trap is a true terminal; an external step limit merely truncates data collection. Reusing one environment connects demonstrations, DAgger, DQN, a fixed offline log, policy evaluation, and conditional trajectory modeling without pretending that their objectives are identical.

![KeyDoorGrid with a key, locked door, traps, and a goal](assets/rl-for-llm/en/module-05/M5_keydoor_world_EN.png)

---

> **Practice route.** [Exercises](/courses/rl-for-llm/module-05/exercises/), [quizzes](/courses/rl-for-llm/module-05/tests/), and [executable practice](/courses/rl-for-llm/module-05/notebooks/) have separate pages. This lecture provides a self-contained explanation.

## Lesson 5.1. Imitation learning and SFT

We start with records that already specify an action. Before estimating its value, we need to understand what a model learns by reproducing someone else’s choices—and what changes when its own choices determine the next input.

### Step 5.1.1 — Demonstrations instead of reward

Begin with the simplest fixed data: an expert showed what to do but did not provide a reward function. A demonstration set contains pairs

$$
D_E=\{(s_i,a_i)\}_{i=1}^{N},
$$

where $s_i$ is a state visited by the expert and $a_i$ is the action selected there.

**BC** trains a parameterized policy to predict the expert action:

$$
\theta^*=\arg\max_\theta\sum_{(s,a)\in D_E}\log \pi_\theta(a\mid s).
$$

This is ordinary maximum likelihood. With discrete actions it is equivalent to minimizing cross-entropy. Reward, value functions, and Bellman equations do not appear in the objective.

Now translate the notation to an LLM. Let a demonstration contain a prompt $x$, a response $y=(y_1,\ldots,y_T)$, and a mask $m_t$ that is one only for tokens included in the loss. Then

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=-\sum_{(x,y)\in D_E}\sum_{t=1}^{T}m_t\log \pi_\theta(y_t\mid x,y_{<t}).
$$

Under the token-level abstraction $s_t=(x,y_{<t})$ and $a_t=y_t$, this is BC on expert prefixes. The qualification matters: the equality depends on the unit of action, the construction of the prefix, and the mask. If system tokens, user messages, or tool outputs are also trained on, the objective is still maximum likelihood, but it clones a broader sequence-generation process.

![Behavioral cloning and token-level SFT](assets/rl-for-llm/en/module-05/M5_bc_sft_bridge_EN.png)

Demonstration preparation, sequence packing, masking, and the complete SFT pipeline are developed in *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)). Tokenization, special tokens, and message boundaries are covered in *Modern LLMs*, Module 2 “Tokenization and Embeddings” ([open module](/courses/modern-llms/module-02/)). Both references are optional here: the algorithmic fact needed in this module is narrower—training observes demonstrator actions in states generated by the demonstrator.

For a deeper treatment of negative log-likelihood, cross-entropy, and KL divergence, see *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” ([open module](/courses/information-theory-for-ml/module-03/)).

---

### Step 5.1.2 — Dataset prefixes during training, model prefixes during generation

During training, a token model receives the preceding prefix from the demonstration. This is **teacher forcing**: the prefix is the one recorded in the training example, not necessarily the only valid way to express the answer. During generation, the next input contains the model’s own tokens.

Let $d_t^{\pi_E}(s)$ be the state distribution at time $t$ under the expert and $d_t^{\pi_\theta}(s)$ the distribution under the learned policy. A standard BC error is measured on expert states:

$$
\varepsilon_t
=\mathbb E_{s\sim d_t^{\pi_E}}
\left[\Pr_{a\sim\pi_\theta(\cdot\mid s)}\{a\ne \pi_E(s)\}\right].
$$

Here the expert supplies a particular action $\pi_E(s)$, so the error counts disagreement with that choice. For language, disagreement is not automatically a failure of the user’s task: another wording or solution may be equally valid. The formula measures reproduction of the chosen demonstrator, which must be distinguished from the quality of the resulting behavior.

Deployment quality, however, depends on states visited by the learned policy: $s\sim d_t^{\pi_\theta}$.

After the first mistake, the two distributions may diverge. The policy enters a state that rarely occurred in demonstrations, makes a larger error there, and moves farther away. In sequential prediction this is called **compounding error**; in autoregressive modeling the train–generation mismatch is often called **exposure bias**.

A careful boundary is important. Distribution shift does not prove that every long generation must fail. The model may generalize to novel prefixes, and an error may be recoverable. The precise statement is:

> Small error on the expert-state distribution does not by itself guarantee small error on the state distribution induced by the learned policy.

This separates predictive accuracy from closed-loop policy quality.

![Expert and learner distributions and the DAgger loop](assets/rl-for-llm/en/module-05/M5_distribution_shift_EN.png)

---

### Step 5.1.3 — Tightrope: an exact compounding-error calculation

Consider a deliberately simple task. The expert survives $T$ steps and earns $+1$ at every successful step. On each step before failure, the imitator independently makes an error with probability $\varepsilon$. After the first error all subsequent rewards are zero.

The probability of receiving the reward at step $t$ is $(1-\varepsilon)^t$, so the expected return is

$$
J_{\mathrm{BC}}(\varepsilon,T)=\sum_{t=1}^{T}(1-\varepsilon)^t.
$$

For $\varepsilon>0$,

$$
J_{\mathrm{BC}}=
\frac{(1-\varepsilon)\left[1-(1-\varepsilon)^T\right]}{\varepsilon}.
$$

The expert earns $T$, hence

$$
\operatorname{Regret}(\varepsilon,T)=T-J_{\mathrm{BC}}(\varepsilon,T).
$$

When $\varepsilon T\ll1$, the expansion $(1-\varepsilon)^t=1-t\varepsilon+O(t^2\varepsilon^2)$ gives

$$
\operatorname{Regret}(\varepsilon,T)
=\frac{\varepsilon T(T+1)}{2}+O(\varepsilon^2T^3).
$$

The leading dependence is $O(\varepsilon T^2)$. An early error can cost the entire remaining suffix, not just one reward.

This is not a universal law for every environment. It isolates a worst-case mechanism in an irreversible task. Recoverable errors can cost less; errors that lead into increasingly dangerous states can cost more. The practical question is therefore not only “How accurate is the policy on demonstrations?” but also “What does an error cost, and can the policy recover from states that it creates itself?”

---

### Step 5.1.4 — DAgger: label the states visited by the learner

**Dataset Aggregation** (DAgger) changes the training-state distribution.

1. Train an initial policy on expert demonstrations.
2. Run the current policy, or a mixture of the current policy and the expert.
3. Collect the states actually visited by the learner.
4. Ask the expert for the correct action in each collected state.
5. Add the new $(s,\pi_E(s))$ pairs to the aggregated dataset and retrain.
6. Repeat.

Instead of one supervised problem under $d^{\pi_E}$, DAgger creates a sequence of problems whose states are tied to learner behavior. The original paper reduces this process to no-regret online learning. With a bounded cost-to-go difference and small average regret of the underlying learner, the additional cost scales as $O(T\varepsilon)$ rather than the standard $O(T^2\varepsilon)$ worst-case bound for naive BC.

That sentence is meaningful only together with its assumptions. The cost-to-go bound, online-learning regret, number of iterations, expert–learner mixture, and rule for selecting the final policy all enter the theorem. One extra DAgger round does not automatically repair arbitrary representation error or every environment.

In the course run, an initial BC policy trained on only four KeyDoorGrid demonstrations solved 14 of 24 evaluation episodes. After one DAgger round labeled learner-visited states, it solved all 24 and retained that rate in later rounds. This single-seed synthetic run illustrates the mechanism; it is not an external performance claim. The exact protocol is stored in `data/M5_experiment_protocol.json`.

The tightrope example makes one assumption especially concrete. The linear-in-horizon statement contains a bound on the cost of taking a wrong action and then following the expert. If one error destroys the entire remaining route, that bound can itself grow with $T$. DAgger does not automatically turn the resulting quadratic loss into a linear one. Additional labels tell the learner what to do at more of its own states; they do not make an irreversible mistake reversible. A more favorable horizon dependence requires the cost of recovery to be genuinely controlled.

This also explains DAgger’s place in the module. It is not restricted to the original fixed dataset: it needs fresh state visits and fresh expert labels. While that access remains available, we can repair the distribution of training inputs as well as the predictor. Offline learning will have to proceed without that option.

DAgger is also not the same as reinforcement learning from human feedback (RLHF). DAgger asks an expert for the **action** that should be taken in a particular state. RLHF more often obtains a comparison, rating, or preference over **whole responses**. Both collect feedback on behavior produced by the current policy, but the signal and objective differ.

---

### Step 5.1.5 — What BC explains in LLM post-training, and what it does not

Token-level SFT inherits both strengths and limitations of BC.

**What it does well:**

- transfers the style and format of demonstrations;
- introduces useful behavior without defining a reward;
- uses a stable and scalable likelihood objective;
- can preserve the base model's knowledge when data and optimization are chosen carefully.

**Where the limitation appears:**

- training prefixes come from demonstrations, while generation prefixes come from the model;
- recovery behavior may be rare in the dataset;
- equally plausible actions are not ranked by external utility unless that ranking is represented in the demonstrations;
- BC fits the conditional action distribution in the data rather than maximizing a separately specified reward.

RL after SFT is therefore useful for more than distribution shift. It can exploit verifiable rewards, preferences, interaction, and exploration. Conversely, every post-training pipeline need not begin with a new SFT stage: a base model may already be sufficiently prepared for the target signal.

A useful boundary is:

> BC asks, “What would the demonstrator do in this state?” RL asks, “Which action improves the expected outcome under the chosen reward?”

The answers may coincide, but they need not.

---

## Lesson 5.2. DQN: Q-learning with a neural network

Observed outcomes now join states and actions. A value function lets us compare consequences, and a neural network replaces the table. The buffer can still receive fresh transitions; that permission will be removed in the following lesson.

### Step 5.2.1 — From a Q-table to a deep Q-network

[Module 2](/courses/rl-for-llm/module-02/) updated a tabular estimate through

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)\right].
$$

A table is impossible for images, long observations, or enormous state spaces. A **Deep Q-Network** (DQN) replaces it with $Q_\theta(s,a)$. For a discrete action space, the network commonly maps one state to a vector of values, one per action.

This changes more than the storage format. A table update affects one entry; a neural update changes shared parameters and may alter predictions for many states and actions. That coupling makes generalization possible, but it also means that fitting one transition can move estimates elsewhere.

Follow one recorded transition. Action $a$ has already been taken, reward $r$ observed, and successor $s'$ stored. The main network predicts $Q_\theta(s,a)$ for the action that actually appears in the record. A separate target network supplies the continuation estimate. For a transition $(s,a,r,s',d)$, construct

$$
y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a'),
$$

where $d=1$ only for a true terminal transition and $\theta^-$ denotes a separate target network. Then minimize, for example,

$$
\mathcal L_Q(\theta)=\mathbb E_D\left[\ell\bigl(Q_\theta(s,a)-y\bigr)\right].
$$

For this optimization step, $y$ is a fixed numerical target. The gradient changes the main network’s prediction $Q_\theta(s,a)$; it does not run through the target network. The reward component is an observation, while the continuation component remains a prediction. A later target-network update will change future regression targets, not the target halfway through this step.

Updated values then influence which actions will be collected next. Data collection usually uses an $\varepsilon$-greedy policy:

$$
a_t=\begin{cases}
\text{a random action}, & \text{with probability }\varepsilon,\\
\arg\max_a Q_\theta(s_t,a), & \text{otherwise}.
\end{cases}
$$

This remains Q-learning because the target maximizes over the next action rather than using the action actually selected by the behavior policy.

---

### Step 5.2.2 — Experience replay

Adjacent transitions are strongly correlated: nearby frames look alike and the policy may remain in one part of the environment for many steps. DQN stores transitions in an **experience replay buffer** and trains on randomly sampled mini-batches.

The buffer has three practical roles:

1. a transition can be reused in multiple updates;
2. shuffling reduces short-range correlation within a mini-batch;
3. data from several recent policy versions are mixed, making the update off-policy.

Replay does not turn DQN into offline RL. Standard DQN continues interacting with the environment and adds new transitions, so it can eventually repair missing coverage. Offline RL receives a fixed dataset; an absent action cannot be tried later.

Nor does random sampling make transitions strictly independent and identically distributed. They still arise from one dynamical system and a changing collection process. Replay merely makes the optimization stream less locally correlated.

---

### Step 5.2.3 — The target network: a slower right-hand side

If one network supplies both $Q_\theta(s,a)$ and the target $r+\gamma\max_{a'}Q_\theta(s',a')$, every update moves both sides of the equation. The learner aims at a target that moves immediately with the prediction.

DQN maintains a second network $Q_{\theta^-}$. Its parameters are either copied periodically,

$$
\theta^-\leftarrow\theta,
$$

or updated by a slow moving average. Over several optimization steps the target changes more slowly than $Q_\theta(s,a)$.

![Data flow and two networks in DQN](assets/rl-for-llm/en/module-05/M5_dqn_dataflow_EN.png)

The DQN target network and an RLHF reference model may both be frozen copies, but their mathematical roles differ.

- The DQN target network supplies a bootstrapped value on the right-hand side of a Bellman target and is updated periodically.
- The RLHF reference policy defines a distance or regularization term, such as a Kullback–Leibler (KL) penalty, and is not a future-return estimate.

The shared engineering pattern—a frozen copy—does not make the mathematical roles identical.

A slower target is not necessarily a more accurate target. If it overvalues an unobserved action, the target network can pass that error to the learner quite consistently. Online interaction still offers a way to test the action and collect corrective evidence. A fixed log removes that route. The target network organizes optimization; it does not by itself establish that the data support the values being propagated.

---

### Step 5.2.4 — Maximization bias and Double DQN

Suppose two actions have equal true values but noisy estimates:

$$
\widehat Q(a)=Q^*(a)+\eta_a,\qquad \mathbb E[\eta_a]=0.
$$

Even if each estimate is unbiased, their maximum is usually high:

$$
\mathbb E\left[\max_a \widehat Q(a)\right]\ge\max_a Q^*(a).
$$

The maximum tends to select whichever action received favorable positive noise. Bootstrapping can then propagate the overestimate backward.

**Double DQN** separates action selection from action evaluation:

$$
a^*=\arg\max_{a'}Q_\theta(s',a'),
$$

$$
y_{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s',a^*).
$$

The online network selects the action and the target network evaluates it. Their errors are not fully independent, but this separation reduces the systematic overestimation observed in DQN.

![Selection and evaluation are separated in Double DQN](assets/rl-for-llm/en/module-05/M5_double_dqn_EN.png)

Double DQN does not eliminate every form of bias. Underestimation, correlated errors, and approximation error remain possible. It addresses one specific mechanism: using the same noisy estimator to choose a maximum and to evaluate that maximum.

---

### Step 5.2.5 — The deadly triad and the boundary of the LLM analogy

Approximate Q-methods combine:

- neural function approximation;
- bootstrapping from their own estimates;
- off-policy data collected by another policy.

This combination is known as the **deadly triad** because it can produce divergence or severe instability. The slogan should not be inverted into “any two elements are always safe.” Convergence still depends on the algorithm, representation, data distribution, step sizes, and other assumptions.

A value-based approach to an LLM is mathematically possible: a token is a discrete action and a prefix is a state. Yet the action space is large, states almost never repeat exactly, rewards often arrive only after a whole response, and a strong pretrained policy already exists. Directly optimizing the autoregressive policy, or using it to constrain candidate actions, is therefore often more convenient.

This is an engineering preference, not a theorem of impossibility. Lesson 5.5 introduces ILQL, a value-based offline method for text generation.

The notebook implements DQN and Double DQN with replay, a target network, $\varepsilon$-greedy collection, separate termination and truncation flags, and a small replay warm-start from noisy expert-guided trajectories. In the pinned `seed=5` run, after 130 episodes, DQN solved 20 of 24 evaluation episodes and Double DQN solved all 24. This verifies the end-to-end mechanics and exposes the target difference; it is not evidence of universal Double-DQN superiority. Another seed, replay composition, or exploration schedule can reverse the ranking.

---

## Lesson 5.3. Offline RL: what can be learned from a fixed log

DAgger could acquire new labels, and an online Q-learner could test its predictions through further interaction. The log is now closed. We need to distinguish conclusions supported by its records from improvements that depend on untested extrapolation.

### Step 5.3.1 — A fixed transition dataset

The restriction now applies throughout optimization: neither a value update nor a policy change can add fresh transitions. In **offline reinforcement learning** (offline RL), we receive

$$
D=\{(s_i,a_i,r_i,s'_i,d_i)\}_{i=1}^{N},
$$

collected by one or more behavior policies $\beta$. No new interaction with the environment is allowed during training. The goal is still to build a policy with high expected return.

| Setting | Data | Objective | New data during training? |
|---|---|---|---:|
| BC | demonstrator states and actions | demonstrator action likelihood | not required |
| ordinary off-policy RL | transitions and rewards in a replay buffer | return | yes |
| offline RL | fixed transitions and rewards | return | no |

Offline RL tries to improve beyond average logged behavior by using action consequences and multi-step structure. Such improvement is possible only under sufficient coverage and appropriate assumptions about generalization. A reward label does not turn an absent transition into an observed one.

---

### Step 5.3.2 — Impossibility without coverage: two worlds, one dataset

Consider one state and two actions, $a_0$ and $a_1$. The log contains one thousand instances of $a_0$ with reward zero. Action $a_1$ never appears.

At least two environments are consistent with the data:

- **world A:** $r(a_1)=+1$;
- **world B:** $r(a_1)=-1$.

The observed dataset is identical in both worlds. Any algorithm that sees only this dataset must return the same output for both, so it cannot both select the optimal action in A and avoid the harmful action in B with a universal guarantee.

This is not a failure of a particular neural network or of computation. It is **non-identifiability**: the answer is absent from the observations.

In a sequential task, an out-of-coverage action is evaluated through generalization. A maximum or actor may prefer an overestimate, and bootstrapping can propagate that error to earlier states.

![Fixed-data support and extrapolation error](assets/rl-for-llm/en/module-05/M5_offline_support_EN.png)

Theoretical guarantees therefore require coverage assumptions. In high-dimensional or continuous spaces, coverage is not literal state equality; it is expressed through density ratios, concentrability coefficients, representation assumptions, or related conditions.

---

### Step 5.3.3 — Extrapolation error and a self-reinforcing loop

Suppose $Q_\theta$ is trained on logged actions, but the target uses $\max_{a'}Q_\theta(s',a')$.

The maximum also considers actions that are rare or absent in the dataset. If one is accidentally overestimated, it enters the target:

$$
y=r+\gamma\max_{a'}Q_\theta(s',a').
$$

The new policy then prefers this unverified action. The loop is

> weak coverage → value error → selection of the error by maximization → bootstrapping → a more confident policy outside the data.

Early work often called this **extrapolation error**. A broader modern description includes distribution shift between the logged data and the learned policy, uncertainty in value estimates, and the policy-improvement operator itself.

Stronger generic regularization is not a universal solution. It may shrink the function but cannot reveal which unobserved action is good. The method must constrain improvement, pessimistically alter values, or train only on dataset actions.

---

### Step 5.3.4 — Four protection mechanisms for out-of-data actions

The same extrapolation failure can be addressed at different points in the learning process: restrict which actions may be considered, make departures from recorded behavior costly, change the values themselves, or construct value updates without querying new actor-generated actions. The four approaches below are alternatives at those interfaces, not stages in a ranking.

#### 1. Restrict the action set

**Batch-Constrained deep Q-learning** (BCQ) learns a model of actions similar to those in the dataset and searches for improvement within that supported set. The actor is not allowed to freely exploit a Q-value for an action on which the value model was not trained.

The ordering matters: first propose admissible candidates, then use value estimates to select among them. In a continuous action space, a supported candidate need not be an exact copy of a stored vector. It comes from a learned model, whose generalization can also be imperfect. The restriction reduces freedom to exploit the value function; it does not turn the behavior model into an infallible list of safe actions.

#### 2. Add a behavioral anchor

**TD3+BC** combines high estimated value with proximity to the logged action. In a simplified form,

$$
\max_\phi\mathbb E_{(s,a)\sim D}
\left[\lambda Q_\theta(s,\pi_\phi(s))-\|\pi_\phi(s)-a\|_2^2\right].
$$

The first term improves according to $Q$; the second keeps the policy close to the data. An apparently valuable action far from the recorded choice must offer enough estimated benefit to offset the penalty. Unlike a hard candidate restriction, this need not prohibit a departure altogether: it gives that departure a price within the objective.

Here an action is a numerical vector and $\pi_\phi(s)$ is the output of a deterministic actor in a continuous-control problem. Distance between those vectors has a specified meaning. Distance between two token IDs does not: neighboring vocabulary indices need not describe similar decisions. A behavioral anchor is a transferable idea; this particular squared-distance term is not a ready-made language-model loss.

#### 3. Lower the values of unsupported actions

**Conservative Q-Learning** (CQL) intervenes in value learning. Rather than only restraining an actor’s eventual choice, it changes the estimates on which that choice will rely. For discrete actions, one form is

$$
\mathcal L_{\mathrm{CQL}}
=\mathcal L_{\mathrm{Bellman}}
+\alpha\mathbb E_{s\sim D}
\left[\log\sum_a e^{Q(s,a)}-\mathbb E_{a\sim D(\cdot\mid s)}Q(s,a)\right].
$$

The Bellman term still asks the value of an observed action to agree with reward and continuation. The additional expression compares the collection of action values at a state with the values assigned to actions represented in the log. Log-sum-exp is particularly sensitive to large entries, so an unsupported action with an inflated Q-value can contribute strongly to the penalty. Subtracting the dataset-action average counterbalances that pressure where the data actually provide actions.

To make the direction explicit, write the conditional dataset-action distribution as $p_D(a\mid s)$ and call the bracketed penalty $C_s(Q)$. Treating the Q-values at this state temporarily as separate coordinates gives

$$
\frac{\partial C_s}{\partial Q(s,a)}
=
\frac{e^{Q(s,a)}}{\sum_b e^{Q(s,b)}}-p_D(a\mid s).
$$

For an action absent from that conditional data distribution, the second term vanishes: this component of the gradient pushes its value downward under minimization. A frequently recorded action receives a compensating term that may weaken or reverse the direction. What matters is not merely whether an action appears, but how its dataset frequency compares with its mass under the softmax of current Q-values. That softmax arises from differentiating the penalty; it need not be the policy actually deployed.

A common shift provides another useful reading of the formula. Add the same constant to every action value at a state. Log-sum-exp increases by that constant, and so does the dataset average. Their difference stays unchanged. The penalty therefore does not simply lower every number. It changes relative values, while the reward and Bellman term still help determine their absolute level.

With a neural network, these coordinate derivatives propagate through shared parameters and combine with other gradient contributions. They do not guarantee that every unsupported action’s predicted value decreases after every optimizer step. The calculation explains the pressure supplied by this loss component, not a pointwise guarantee about the complete training trajectory.

Under its stated assumptions, the relevant construction in the original analysis gives a lower bound on expected policy value. It should not be read as certifying every individual neural Q-value as a lower bound for arbitrary data and hyperparameters. Excessive conservatism can also suppress useful improvement: the coefficient expresses a substantive trade-off, not merely a convenient numerical scale.

#### 4. Avoid querying out-of-data actions while learning value

**Implicit Q-Learning** (IQL) changes how the continuation target is constructed. An ordinary maximizing target asks the Q-function about possible next actions, including choices that the data do not support. IQL seeks improvement among values supported by the dataset-action distribution without introducing arbitrary new actor actions into value learning.

It does so using a function called $V(s)$, but the familiar letter needs a fresh interpretation. Earlier modules used $V^\pi$ for the expected continuation of a specified policy. Here the fitted quantity is an **upper expectile** of the values of dataset actions. We should not automatically identify it with that earlier fixed-policy expectation.

For $u=Q(s,a)-V(s)$, it uses

$$
L_2^\tau(u)=\left|\tau-\mathbf 1\{u<0\}\right|u^2,\qquad \tfrac12<\tau<1.
$$

Hold the Q-function fixed for a moment and consider one state. When an action’s Q-value exceeds $V$, the prediction underestimates it and the squared discrepancy receives weight $\tau$. When the Q-value is below $V$, the overestimate receives the smaller weight $1-\tau$. For $\tau>1/2$, errors on the upper side exert more pressure, shifting the fitted value above the ordinary mean. At $\tau=1/2$, both sides would receive equal weight and ordinary mean regression would be recovered.

The pointwise quantity that the learned value function aims to approximate is

$$
V_\tau(s)
=\arg\min_v\,
\mathbb E_{a\sim D(\cdot\mid s)}
\left[L_2^\tau\bigl(Q(s,a)-v\bigr)\right].
$$

In a continuous state space, we do not form a separate list of repeated observations for every exact state. The network approximates this conditional relationship across the observed pairs. The target’s meaning remains important: the randomness here comes from drawing an **action from the data distribution**, not from choosing the luckiest realization of an environment transition.

An expectile is not a quantile. Squared loss balances the magnitudes of deviations, not just the probability mass above and below the fitted point. With $[z]_+=\max(z,0)$, differentiating the objective with respect to $v$ gives the balance condition

$$
\tau\,\mathbb E_{a\sim D(\cdot\mid s)}
\left[(Q(s,a)-v)_+\right]
=
(1-\tau)\,\mathbb E_{a\sim D(\cdot\mid s)}
\left[(v-Q(s,a))_+\right].
$$

The two sides balance distances with different weights. Thus $\tau$ is not an instruction to retain the best $\tau$ fraction of actions. Every recorded pair still participates; the price of its discrepancy changes. Nor is an upper expectile the exact maximum at an arbitrary fixed $\tau<1$.

Now return to a recorded transition $(s,a,r,s',d)$. After fitting $V$, train $Q$ toward $r+\gamma V(s')$. As before, a true terminal has no continuation, so the target with its terminal mask is $r+\gamma(1-d)V(s')$. The left-hand prediction concerns an action actually present in the log. The right-hand side combines observed reward with a continuation estimate that favors higher-valued supported choices at the successor state.

The division of labor is now clear. Expectile regression shifts $V$ toward the better values among dataset actions. The Q-update propagates that continuation backward through an observed transition. This is more than copying each trajectory’s average performance: improvements can propagate through a sequence of states. At the same time, the Q-regression does not retain only lucky rewards or successor states; ordinary regression averages the observed transition randomness. Approximation error and incomplete coverage remain possible.

The learned values do not yet supply an explicit policy. IQL extracts one by advantage-weighted BC: larger $Q(s,a)-V(s)$ produces a larger coefficient,

$$
w(s,a)=\exp\!\left(\beta[Q(s,a)-V(s)]\right),
$$

usually with weight clipping. In this expression, $\beta>0$ is a scalar controlling the strength of reweighting, not the behavior-policy symbol used in Step 5.3.1. The coefficients enter a familiar likelihood objective:

$$
\max_\phi\,
\mathbb E_{(s,a)\sim D}
\left[w(s,a)\log\pi_\phi(a\mid s)\right].
$$

During the policy update, those weights are treated as already computed numbers. The demonstration target remains the recorded action; what changes is each example’s influence. A negative value difference produces a small **positive** weight, not a negative REINFORCE coefficient that directly reverses the direction of a cloning update. Actions still compete through policy normalization, but weighted imitation and a signed policy-gradient update should not be conflated.

Clipping prevents a very large weight from dominating the update; it does not establish that the underlying value estimate is correct. Likewise, avoiding out-of-data action queries during value learning is not a hard guarantee that the fitted policy can never emit an unseen action. The policy also generalizes. IQL changes where targets come from and how behavior is extracted; it does not remove uncertainty from a fixed dataset.

![Families of offline RL methods](assets/rl-for-llm/en/module-05/M5_offline_method_map_EN.png)

The four approaches can now be located precisely. BCQ restricts candidates. TD3+BC anchors actor outputs to recorded actions. CQL applies conservative pressure to value estimates. IQL builds improvement through a statistic of dataset-action values and transfers it to a policy by weighted imitation. Their shared caution does not make them one equation: the interfaces, induced biases, and hyperparameters differ.

---

### Step 5.3.5 — Dataset quality, evaluation, and model selection

Offline-RL data quality is not determined by transition count alone. Two logs of equal size can support very different policy-improvement claims. What matters includes:

- which states and actions are covered;
- whether the log comes from one policy or a mixture;
- whether successful and failed trajectories are both represented;
- whether rewards, termination flags, and transition order are reliable;
- whether deployment dynamics match collection dynamics;
- whether evaluator training can be separated from hyperparameter selection.

**Datasets for Deep Data-Driven Reinforcement Learning** (D4RL) was introduced as a common collection of tasks and fixed logs for offline RL, including demonstrations, policy mixtures, and multi-task data. It was historically influential, but a D4RL result is not a certificate for another domain: dynamics, coverage geometry, reward scale, and failure cost may all differ.

Model selection is particularly difficult. A low held-out Bellman error need not imply a good policy: an estimator may fit logged actions accurately while failing on actions selected by the improved policy. One tool is **fitted Q evaluation** (FQE). For a fixed evaluation policy $\pi$, it iteratively fits targets

$$
y_i^{\mathrm{FQE}}
=
r_i+
\gamma(1-d_i)
\sum_{a'}\pi(a'\mid s_i')\widehat Q_{\bar\phi}(s_i',a').
$$

There is no action maximum here: FQE evaluates a chosen policy rather than improving it. The parameters denoted by $\bar\phi$ are held fixed while constructing the current targets, and the fitted Q-function is regressed toward those numbers. Every backup asks about continuation under the same evaluation policy $\pi$.

A policy-level score also requires an initial-state distribution $\rho_0$. Average the fitted values over the policy’s actions at a starting state, then over starting states themselves:

$$
\widehat J_{\mathrm{FQE}}(\pi)
=
\mathbb E_{s_0\sim\rho_0}
\left[\sum_a\pi(a\mid s_0)\widehat Q(s_0,a)\right].
$$

With a finite collection of starts, the outer expectation becomes an average using the declared weights. Averaging over every state in the log need not estimate the same quantity: a frequently visited intermediate state is not thereby a frequent starting point for a new task. FQE first fits continuation values and then uses them to score a specified evaluation setting. It still bootstraps from a fixed log. Weak coverage, finite data, approximation error, and policy–data shift can therefore corrupt the estimate. FQE should be paired with coverage diagnostics, multiple data splits, hyperparameter sensitivity checks, and—when safe—an independent simulator or a limited online test.

![Methods compared on one fixed KeyDoorGrid log](assets/rl-for-llm/en/module-05/M5_offline_results_EN.png)

In the pinned course experiment, one log of 80 trajectories was used for BC, naive offline DQN, a BCQ-like support restriction, CQL, and IQL. With `seed=41`, 16 training epochs, and 24 evaluation episodes, BC, the BCQ-like method, CQL, and IQL solved every episode, whereas naive offline DQN solved none and averaged $-2.175$ return. This is one concrete extrapolation failure in this environment, not a universal method ranking. In the same run, FQE assigned initial-state values of roughly $-0.005$ and $-0.056$ to BC and IQL even though direct simulator evaluation gave both policies about $0.703$. On this small log, the evaluator was poorly calibrated and separated policies with the same observed performance. The full protocol is stored in `data/M5_experiment_protocol.json`.

Training on the log, selecting hyperparameters, and directly testing behavior are three distinct procedures. Simulator evaluation does not itself make training online: those transitions need not enter the training dataset. But choosing a regularization coefficient or checkpoint from simulator outcomes uses additional environmental feedback for selection. That access should be reported separately, and the final independent check kept distinct from tuning.

[Module 3](/courses/rl-for-llm/module-03/) evaluated policies from one-step logs with known logging propensities. The sequential setting is harder: value errors propagate through transitions, and a new policy may visit a different part of state space. Pessimism does not remove model selection. Too little allows exploitation of error; too much collapses improvement toward mediocre behavior cloning.

## Lesson 5.4. Decision Transformer: a policy as a conditional trajectory model

A fixed trajectory can carry useful reward information without requiring an explicit Q-function. Decision Transformer puts that information into the condition for predicting an action. The key is to follow how the condition changes meaning between training and deployment.

### Step 5.4.1 — Put desired return into the sequence

**Decision Transformer** (DT) does not build an explicit Q-function or apply a policy gradient. It represents a trajectory as

$$
(G_1,s_1,a_1,G_2,s_2,a_2,\ldots,G_T,s_T,a_T),
$$

where

$$
G_t=\sum_{k=t}^{T}r_k
$$

is the return-to-go when $\gamma=1$. In this local DT notation, $r_t$ is the reward for action $a_t$, rather than the $r_{t+1}$ convention used after $a_t$ in earlier lessons.

At dataset preparation time, the future is already a recorded past: the trajectory has ended. Its actual rewards can therefore be summed retrospectively. Here $G_t$ is neither a user request nor a model prediction. It describes the continuation in one stored example. Similar states and actions can occur within trajectories of different quality, and the numerical condition lets the model distinguish those records.

The causal transformer predicts an action conditioned on that recorded return-to-go and the history available at the decision point:

$$
\pi_\theta\left(a_t\mid G_{\le t},s_{\le t},a_{<t}\right).
$$

Discrete actions can use negative log-likelihood; continuous actions can use a regression loss. Optimization resembles supervised sequence modeling on fixed trajectories. When predicting $a_t$, the action label itself is not among the available actions $a_{<t}$; the causal computation must preserve that boundary.

Does the return condition reveal the future? During training it deliberately describes the outcome of a recorded continuation, but the learned object is a **conditional** action distribution. At deployment, nobody supplies the unknown future outcome. A requested value takes its place. Predicting actions well in return-labeled trajectories is therefore not a proof that every requested return can be achieved.

In that sense DT is **conditional imitation learning**: it copies actions while receiving an additional trajectory-quality condition. It can distinguish behavior in high-return trajectories from behavior in low-return ones.

---

### Step 5.4.2 — Updating the return condition at deployment

Now replace the completed record with a live interaction. Before the episode, choose $G_1^{\mathrm{target}}$. The model receives that condition and the current state, then selects an action. The **environment**, not the transformer, supplies the successor state and reward $r_t$. The condition is then updated for the next decision. Under the undiscounted convention,

$$
G_{t+1}^{\mathrm{target}}=G_t^{\mathrm{target}}-r_t.
$$

For the discounted recurrence

$$
G_t=r_t+\gamma G_{t+1},
$$

with $\gamma>0$,

$$
G_{t+1}^{\mathrm{target}}=\frac{G_t^{\mathrm{target}}-r_t}{\gamma}.
$$

The model sees how much reward remains to be achieved, the current state, and previous actions. Subtracting what was received is bookkeeping for the remaining request, not a test of feasibility. An unexpected reward or an unfavorable successor state may leave a condition that is easy to compute but impossible to satisfy. The recorded history must contain what the environment actually returned, not the continuation one hoped to generate.

![Training and deployment of Decision Transformer](assets/rl-for-llm/en/module-05/M5_decision_transformer_EN.png)

Desired return is a condition, not a promise. A target far outside the training range asks the model to extrapolate. Setting `target_return=1000` does not create behavior that the model has never observed or learned to compose.

The notebook tests that distinction directly. A Decision Transformer is trained on 120 KeyDoorGrid trajectories and deployed with initial conditions $-0.5$, $0.2$, $0.7$, and $1.2$. Across six evaluation episodes per condition, every setting reached the goal, while mean return stayed between about $0.66$ and $0.70$ and did not increase monotonically with the requested value. The model learned robust successful behavior, but the numerical condition did not become a calibrated return controller. This is one synthetic run, not a comparison with external DT implementations.

---

### Step 5.4.3 — How DT differs from Q-learning

Q-learning uses a recursive relationship involving the best next action:

$$
Q(s,a)\approx r+\gamma\max_{a'}Q(s',a').
$$

DT uses neither this maximum nor an explicit value function. It learns a conditional action distribution over trajectories.

This yields different strengths and failure modes.

- DT reuses standard sequence-modeling machinery and can condition on long histories.
- Q-methods explicitly exploit local dynamic recursion and can combine estimated consequences of actions.
- DT depends on coverage of relevant state–history–return combinations.
- Q-methods depend on stable bootstrapping and reliable values for candidate actions.

Neither dominates in every setting. The original DT paper reported competitive results on selected offline benchmarks; that is an empirical result for particular tasks, architectures, and datasets.

---

### Step 5.4.4 — Stitching behavior fragments is not guaranteed

Suppose one trajectory shows how to go from $A$ to $B$ and another from $B$ to $C$. We might hope the model will generate $A\to B\to C$ even though the complete trajectory is absent.

For the fragments to compose, $B$ must identify the same relevant state, not merely a similar observation. In KeyDoorGrid, the same cell with or without the key can permit different continuations. If the representation merges those situations, matching the symbol $B$ does not make the second segment executable.

Dynamic programming makes this composition explicit through state values. A sequence model may learn it through representation generalization and attention, but its supervised objective does not supply a universal theorem of stitching for arbitrary data.

The opposite absolute claim is also wrong: DT is not fundamentally incapable of novel trajectory composition. Its ability depends on data structure, architecture, context length, scale, and optimization.

A precise summary is:

> DT converts policy improvement into conditional trajectory modeling; it does not remove the coverage problem, but expresses it in the space of sequences and return conditions.

---

### Step 5.4.5 — Why the LLM resemblance is useful but limited

DT and an LLM both use a causal transformer and autoregressive prediction. Engineering ideas such as masking, positional representations, sequence packing, context selection, and scaling therefore transfer naturally.

The analogy does not establish a direct lineage from Decision Transformer to every LLM agent, and a system prompt is not automatically a return-to-go. DT's numerical condition has a defined trajectory meaning and is updated from observed rewards. A text instruction may express a goal, style, or constraint without being a calibrated estimate of remaining return.

The useful connection is that a complex policy can be represented as a conditional sequence model. The boundary is that good conditional generation does not replace environment dynamics, data coverage, or a reward aligned with the real objective.

---

## Lesson 5.5. A map of fixed-data LLM post-training

We return to language models with a more precise question: does a text record provide a demonstration, a preference, or a reward-labeled transition? Its learning role depends on that information, not on the fact that all three can be stored as text.

### Step 5.5.1 — First identify what was logged

Algorithm choice begins with the unit of data.

| Data | Typical objective | Fresh rollouts during optimization? | What is learned |
|---|---|---:|---|
| demonstrations $(x,y)$ | SFT / BC | no | demonstrator actions on demonstrator prefixes |
| transitions $(s,a,r,s')$ | classical offline RL | no | action consequences and a reward-optimizing policy |
| pairs $(x,y^+,y^-)$ | DPO and related methods | no | relative preference between responses |
| fresh responses with reward | PPO, RLOO, GRPO | yes | an update from current-policy rollouts |
| candidate set without weight updates | Best-of-N, search | no training | response selection at inference time |

“Offline” is sometimes used broadly for any optimization on a fixed dataset. In this course, **classical offline RL** means a sequential transition-and-reward problem in which action consequences are estimated and the policy is improved without further interaction.

---

### Step 5.5.2 — Why DPO should not simply be renamed offline RL

**Direct Preference Optimization** (DPO) trains on fixed preference pairs and does not need to generate fresh rollouts inside each optimization step. In that broad sense it is offline policy optimization.

Its mathematical setting nevertheless differs from classical offline RL:

- data contain comparisons between whole responses rather than necessarily reward-labeled transitions;
- no Q-function is trained and no Bellman bootstrap is performed;
- the objective is derived from a KL-regularized RLHF formulation and a pairwise preference model;
- training becomes a classification-style loss involving likelihood ratios under the current and reference policies.

This course therefore calls DPO **direct preference optimization on fixed pairs**, not a synonym for classical offline RL. The full derivation appears in [Module 8](/courses/rl-for-llm/module-08/). For parallel reading, *Modern LLMs*, Module 14 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” ([open module](/courses/modern-llms/module-14/)) places DPO in a modern LLM pipeline, while *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” ([open module](/courses/information-theory-for-ml/module-08/)) develops the KL-regularized variational optimum.

---

### Step 5.5.3 — ILQL: value-based offline RL for language

**Implicit Language Q-Learning** (ILQL) adapts IQL ideas to text generation. A prefix is a state, a next token is an action, and fixed text trajectories carry rewards or outcome scores.

The method learns $Q$ and $V$ while treating out-of-data actions conservatively, then uses advantages to tilt generation toward higher-valued tokens. Unlike SFT, example quality affects more than token frequency because reward distinguishes preferred continuations. Unlike DPO, token-level value functions and sequential transitions remain central objects.

Follow a stored response through that construction. Each prefix and its next recorded token define a transition. The outcome score is assigned at the end or distributed across the trajectory according to the stated reward convention. Q-learning relates a recorded token to reward and continuation, while V is fitted through expectile regression. With terminal reward, repeated value updates can propagate its influence to earlier prefixes. This does not require sampling counterfactual responses at every update, but it does require the recorded transitions and rewards to represent the intended task.

Fitting values still leaves a policy-extraction question. The original ILQL combines a behavioral language model with learned values at decoding time: add a scaled $Q(s,a)-V(s)$ to the token’s behavioral log-probability, then normalize the modified scores. In compact form, this is the exponential reweighting already encountered in IQL, now applied to generation: $\pi(a\mid s)\propto\pi_{\mathrm{beh}}(a\mid s)\exp(\beta[Q(s,a)-V(s)])$.

The two components have different jobs. The behavioral model supplies plausible tokens for the prefix; the value adjustment changes their relative attractiveness under the specified reward. Because $V(s)$ is common to the candidate actions, it cancels in normalization at a fixed state. Writing the adjustment as $Q-V$ connects it to the familiar value difference without creating an extra independent label for each token. Nor does a merely nonzero softmax probability prove that an action is well supported by data. Conservative value learning and restrictions on selection therefore remain substantive choices.

This decoding route does not require fitting a separate actor with a weighted cloning loss. It differs from distilling the values into a new policy and then using that policy directly. The reduced course pipeline below contains both interfaces—advantage-weighted policy fitting and value-guided decoding—which is another reason not to present its results as a precise replication of the original ILQL algorithm.

ILQL matters here as a counterexample to the claim that Q-methods cannot apply to LLMs. They can. The practical question is whether their computational cost, value-estimation stability, and reward requirements beat more direct objectives in the target setting.

The language section of the notebook builds a small but genuine offline pipeline: a fixed response log for several prompt classes, a behavior policy, token-level $Q$ and $V$ heads, expectile value regression, advantage-weighted policy learning, and decoding constrained by behavior-model support. The implementation is deliberately described as **ILQL-like**. It reproduces the key interface, not the full architecture, scale, or every loss term of the original paper.

![From a fixed text log to token-level value guidance](assets/rl-for-llm/en/module-05/M5_language_ilql_EN.png)

In the pinned synthetic run, exact-answer success in the logged data was about $30.7\%$. Plain BC produced $33.3\%$ exact greedy answers, the advantage-weighted policy produced $100\%$, and support-constrained value-guided decoding produced $83.3\%$. These numbers validate mechanics under a known reward; they are not a language-quality benchmark and do not establish that an ILQL-like objective generally beats SFT.

---

### Step 5.5.4 — A KL anchor and pessimism solve different problems

RLHF often penalizes deviation from a reference policy:

$$
\mathbb E[r(x,y)]-\beta D_{\mathrm{KL}}\left(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

An offline-RL method may instead distrust values for poorly supported actions. Both express an engineering intuition: abrupt movement into an unknown region is risky.

Their mechanisms differ.

- KL controls a policy distribution relative to a chosen reference.
- CQL modifies value estimates.
- BCQ restricts the candidate action set.
- IQL learns values from dataset actions and extracts a policy by weighted cloning.

Small KL does not prove that the reward model is correct or that important states are covered. A conservative Q-function does not validate the reward and can become excessively pessimistic. The analogy is useful as a risk map, not as an identity of objectives.

The full variational derivation of a KL-anchored policy appears in *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” ([open module](/courses/information-theory-for-ml/module-08/)). Here the important point is the interface: KL acts on a policy distribution, whereas pessimistic methods alter candidate actions, value estimates, or example weights.

---

### Step 5.5.5 — Module summary

We can now organize the methods by data source and improvement mechanism.

1. **BC / SFT:** copy demonstrator actions on demonstrator states.
2. **DAgger:** add expert actions for states visited by the current policy.
3. **DQN:** collect new transitions, reuse them, and bootstrap a Q-function.
4. **Offline RL:** improve from fixed transitions and rewards while explicitly respecting coverage.
5. **Decision Transformer:** predict actions as part of a trajectory conditioned on desired return-to-go.
6. **DPO:** optimize fixed preference pairs without an explicit Q-function.
7. **ILQL:** learn token-level values from fixed rewarded text trajectories.

The central conclusion is:

> Fixing the data is not a minor implementation detail. It determines which counterfactual actions can be evaluated, which errors cannot be corrected by new experience, and which forms of regularization are meaningful.

[Module 6](/courses/rl-for-llm/module-06/) returns directly to the LLM policy: token log-probabilities, KL to a reference model, sampling, and sequence-level objectives. The question of where data came from will become a question about concrete probabilities: does a saved log-probability describe the policy that actually selected the token?

---

## Practice route

Practice contains [20 core tasks](/courses/rl-for-llm/module-05/exercises/) and [12 additional quiz questions](/courses/rl-for-llm/module-05/tests/). Worked exercise solutions expand on request. The core route covers:

- masked BC/SFT loss and an exact compounding-error calculation;
- DQN and Double-DQN targets with the correct terminal mask;
- coverage diagnostics and unsupported actions;
- CQL, expectile regression, and IQL weights;
- FQE and fixed-log model-selection risk;
- return-to-go and the Decision Transformer condition;
- token-level value guidance and the boundary of the LLM analogy.

---

## Sources and further reading

These primary papers support the module’s historical and algorithm-specific claims. Their theoretical results retain their stated assumptions. The KeyDoorGrid numbers reported in the lecture are separate course experiments, not replications of external benchmarks.

1. [Ross, Gordon, and Bagnell — DAgger](https://proceedings.mlr.press/v15/ross11a.html). Read with Step 5.1.4 for the online-learning reduction and its assumptions; the paper does not promise that one aggregation round repairs every task.
2. [Mnih et al. — DQN](https://www.nature.com/articles/nature14236). The source for combining a deep Q-network, replay, and a target network. Atari results do not transfer automatically to KeyDoorGrid or language models.
3. [van Hasselt, Guez, and Silver — Double DQN](https://arxiv.org/abs/1509.06461). Supports separating next-action selection from evaluation and reports reduced overestimation; it does not guarantee a win on every seed.
4. [Fujimoto, Meger, and Precup — BCQ](https://arxiv.org/abs/1812.02900). Introduces support-constrained action selection and analyzes extrapolation error in fixed batches.
5. [Kumar et al. — CQL](https://arxiv.org/abs/2006.04779). Provides a conservative Q-objective and theoretical properties under stated assumptions; a chosen coefficient does not make every neural implementation an automatic lower bound.
6. [Kostrikov, Nair, and Levine — IQL](https://arxiv.org/abs/2110.06169). The source for expectile value fitting, dataset-action updates, and advantage-weighted policy extraction.
7. [Fu et al. — D4RL](https://arxiv.org/abs/2004.07219). A historically important suite of offline-RL datasets and tasks; its normalized scores are not a universal quality scale.
8. [Chen et al. — Decision Transformer](https://arxiv.org/abs/2106.01345). Introduces return-conditioned trajectory modeling. The reported benchmark results do not prove universal trajectory stitching.
9. [Rafailov et al. — DPO](https://arxiv.org/abs/2305.18290). Supports the classification-style objective on fixed preference pairs and its KL-regularized derivation; DPO is not a Bellman method.
10. [Snell et al. — ILQL](https://arxiv.org/abs/2206.11871). The source for token-level values and value-guided language generation. The course implementation is a reduced ILQL-like model, not a reproduction of every paper result.

11. [Fujimoto and Gu — TD3+BC](https://arxiv.org/abs/2106.06860). The source for the actor’s behavioral-cloning term in continuous control. Squared action distance cannot be transferred unchanged to token IDs.

**Optional cross-course depth:** *Modern LLMs*, Module 2 “Tokenization and Embeddings” ([open module](/courses/modern-llms/module-02/)) covers special tokens and message boundaries; *Modern LLMs*, Module 13 “Supervised Fine-Tuning and Data Work” ([open module](/courses/modern-llms/module-13/)) covers demonstration preparation and masking; *Modern LLMs*, Module 14 “Learning from Feedback: RLHF, DPO, RLOO, and GRPO” ([open module](/courses/modern-llms/module-14/)) places fixed preferences in the LLM post-training stack; *Information Theory for ML*, Module 3 “Cross-Entropy and KL Divergence” ([open module](/courses/information-theory-for-ml/module-03/)) develops log-likelihood and cross-entropy; *Information Theory for ML*, Module 8 “Maximum entropy, exponential models, and KL regularization” ([open module](/courses/information-theory-for-ml/module-08/)) develops the variational logic of a KL anchor.
