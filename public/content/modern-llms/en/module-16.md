# Module 16. Agents and Tool Use

*“Modern LLMs” course · Module 16 lecture · edition 2026.8*

> **What this module is about.** A language model can propose an answer. An agent system must maintain a stateful job: choose an action, execute it under policy, inspect the result, recover from failure, and know when the task is complete. That makes an agent less a new neural architecture than a composition of a model, an execution loop, tools, memory, permissions, budgets, and verification. We will follow one running task—diagnose a failing software build, inspect the repository, make a repair, run tests, and report the evidence. The task exposes four recurring engineering debts: converting probabilistic text into a safe operation; carrying a growing trajectory without paying for the whole history again; preventing local errors from compounding; and parallelizing work without losing coordination. The module’s frozen examples make those debts quantitative: 345,000 reprocessed prompt tokens versus 33,000 with prefix reuse; a 778.5 ms illustrative step; a twenty-step reliability of $0.98^{20}=0.668$; and two legitimate but different speedups—8.89× and 9.44×—for the same multi-agent schedule, depending on the denominator.
>
> **Prerequisites.** Module 9 supplies prefill and KV-cache economics, Module 10 supplies generation acceleration, Module 13 supplies inference-time effort, and Module 14 supplies multimodal computer interaction. Every concept required for the argument is restated here.

---

## 1. Motivation: a response becomes a trajectory

Consider the instruction: “Find why the tests started failing after the dependency update, fix the code, and verify the repair.” A chat model can describe a plausible patch. An agent must interrogate the actual repository, run commands, react to diagnostics, alter its hypothesis, edit files, and confirm that the environment has reached the requested state.

The unit of work is therefore no longer one completion. It is a trajectory

$$
\tau=(o_0,a_0,o_1,a_1,\ldots,o_T),
$$

where $o_t$ is an observation and $a_t$ is an action. The model emits an action proposal; an external runtime decides whether and how that proposal affects the world.

Four questions appear immediately.

- **Action contract.** How does free-form model output become a typed, authorized operation?
- **Trajectory state.** Which parts of the history remain in the live context, which become durable artifacts, and which may be compressed?
- **Reliability.** How does the system detect failure, retry safely, and stop before it exhausts the user’s budget?
- **Orchestration.** When should work remain sequential, and when can independent specialists shorten the critical path?

Each answer changes the others. More history may improve local decisions while increasing prefill and KV cost. Retries improve reliability only when they change the conditions of failure. Subagents reduce elapsed time only when the work is genuinely separable and the cost of planning, merging, and duplicated context remains controlled.

A useful operational definition follows: **an agent is a bounded feedback system in which a model proposes actions, an environment returns observations, and the harness enforces the rules of execution and termination**.

## 2. Historical arc: the control loop became a software platform

![VIZ m16/01 — from prompted loops to agent runtimes](assets/modern-llms/en/module-16/m16_01_timeline.svg)

**Reasoning and Acting (ReAct)** in 2022 brought two previously separate behaviors into one alternating trace: reasoning updated the plan, and actions gathered new evidence from an environment. The lasting contribution was the feedback loop, not the literal `Thought/Action/Observation` labels.

The next step made an action an API object rather than a substring that a fragile parser had to recover. Function-calling interfaces and grammar-constrained generation reduced malformed outputs. This solved one narrow problem—syntax—but revealed a broader one: a well-formed request can still be semantically wrong, unauthorized, or destructive.

In 2024, **CodeAct** and **SWE-agent** moved attention from prompting to the interface itself. CodeAct used executable code as a compositional action space. SWE-agent argued that a model is a new kind of computer user and introduced an **Agent–Computer Interface (ACI)** designed around that user’s strengths and limitations. The same checkpoint could behave differently when commands, observations, and editing operations were redesigned.

Anthropic introduced the **Model Context Protocol (MCP)** in the same period as an open connection layer between LLM applications and external capabilities. The specification dated 28 July 2026 continues that line with a formal protocol core, transports, authorization rules, and extensions.

By 2025–2026, agent SDKs and products offered durable sessions, hosted or local tools, computer interaction, tracing, checkpoints, background execution, and specialist agents. The agent loop had become an execution platform. As a result, improving “the agent” now means deciding whether the bottleneck belongs to the model, the action interface, context management, permissions, the verifier, or the runtime.

## 3. Classical foundations: planning, feedback control, and operating systems

The terminology is new; much of the engineering is not.

### Plans are hypotheses about state transitions

Classical planning represents actions through preconditions and effects. The Stanford Research Institute Problem Solver formalism (STRIPS) made this explicit; **Hierarchical Task Networks (HTN)** decomposed high-level goals into smaller tasks. A language model can perform similar decomposition without a complete symbolic world model, which is useful in messy real environments.

The cost of that flexibility is uncertainty. A natural-language plan may omit a dependency, refer to a state that does not exist, or become obsolete after an unexpected observation. A production plan should therefore be treated as an inspectable, revisable hypothesis—not as a script that must be completed after reality has changed.

### An agent is a closed-loop controller

A fixed workflow follows a predefined graph. An agent chooses the next action after observing the result of the previous one:

$$
\text{state}_t \xrightarrow{\pi_\theta} a_t
\xrightarrow{\text{environment}} o_{t+1}
\xrightarrow{\text{update}} \text{state}_{t+1}.
$$

The LLM implements only part of the policy. Validation, authorization, execution, state storage, and stopping rules sit outside it. This is why model capability and system reliability must be measured separately.

### Capability security and distributed systems still apply

A tool that can read files, send mail, or execute code must be constrained by least privilege, process isolation, timeouts, quotas, audit logs, and idempotency. These are ordinary operating-system and distributed-system requirements, not special “AI safety” decorations.

The same is true of delegation. Twenty specialists working on one repository create dependencies, conflicts, stragglers, redundant work, and a merge barrier. Amdahl’s law remains relevant even when the workers are language models.

These foundations organize the rest of the lecture: define the loop, make actions safe, account for state and latency, then address reliability and parallel execution.

## 4. Anatomy of an agent loop

It helps to separate the model from the **agent harness**. The model proposes what to do next. The harness decides whether the proposal is admissible, executes it, records the outcome, and constructs the next state.

A minimal loop contains eight distinct responsibilities.

1. **Goal and initial state.** The task, accessible resources, permissions, and budgets.
2. **Step context.** Selected history, durable memory, tool descriptions, and the current plan.
3. **Model proposal.** A final answer, tool call, code block, clarification request, or revised plan.
4. **Protocol validation.** Grammar, schema, required fields, and types.
5. **Authorization.** Whether this principal may use this operation on this resource.
6. **Execution.** A sandbox, timeout, resource limits, and an operation identity.
7. **Observation.** Results, errors, environment changes, and a typed trace event.
8. **Termination check.** Completion, repair, escalation, or budget stop.

![VIZ m16/02 — a bounded feedback loop](assets/modern-llms/en/module-16/m16_02_loop.svg)

This decomposition turns an opaque failure into a diagnosable one. Selecting the wrong tool is a policy error. Producing invalid JSON is a protocol error. A denied write is an authorization outcome. A crashed process is an environment failure. A confident but incorrect completion after successful calls points toward planning or verification.

For that reason, an agent trace should contain typed events, schema versions, budgets, permissions, and stop states—not merely a concatenated transcript.

## 5. Function calling and guided generation: syntax is only the first boundary

A function-calling interface normally gives the model a name, description, and argument schema. The model emits a structured object; the client validates it, executes the handler, and returns the observation.

Three layers of correctness must remain separate.

**Syntax.** The output must belong to a JSON or grammar language. Guided generation can compile a regular expression, JSON schema, or grammar into an automaton and mask tokens that cannot lead to a valid continuation. The Outlines work formalized this finite-state view of generation.

**Semantic schema.** Valid JSON may still contain the wrong type, an unknown field, or a missing argument. A validator enforces the application contract.

**Execution policy.** A perfectly valid `delete_file` request may be forbidden for this user or this directory. Authorization, confirmation, and side-effect policy belong outside the model.

The frozen teaching calculation compares 150,000 simple token-admissibility checks with the approximate $2P=1.606\cdot10^{10}$ FLOPs associated with one token of an 8.03B model. The ratio is

$$
9.34\cdot10^{-6}.
$$

This is an operation-count illustration, not a measured overhead for a grammar engine. Real cost depends on tokenization, automaton construction, allowed-token sets, and runtime integration. The robust conclusion is narrower: **guided generation can remove many syntax failures, but it cannot authorize an operation or establish that the operation is appropriate**.

## 6. ReAct, Plan-and-Execute, and CodeAct: choosing an action language

These patterns are best understood as complementary control choices.

**ReAct** selects an action after each observation. It adapts quickly but pays for more model–environment round trips, more growing history, and more opportunities for local failure.

**Plan-and-Execute** creates an explicit plan before executing expensive actions. The plan can be reviewed, checked for missing prerequisites, and tracked as a set of obligations. It must still be revised when an observation invalidates an assumption.

**CodeAct** places loops, variables, branching, and tool composition inside executable code. The CodeAct paper showed that this richer action space can improve agent performance on tasks that require composing APIs.

Our numerical example is intentionally not a benchmark. Five atomic JSON actions use 90 tokens each and require five model–tool round trips. One code block uses 160 tokens and one round trip:

$$
450/160=2.8125,
\qquad
5/1=5.
$$

![VIZ m16/03 — three action representations](assets/modern-llms/en/module-16/m16_03_action.svg)

CodeAct’s compactness expands the security surface. General code requires a real sandbox, not `eval` in the orchestrator process. A practical system often combines the patterns: typed calls for external side effects, sandboxed code for local computation, and a plan that can be revised after verification.

## 7. Worked example A: context growth across the loop

Assume the immutable beginning of the session—the system instruction, tool schemas, and user task—contains 3,000 tokens. Each of 20 steps appends 1,500 tokens of reasoning, action, and observation.

If every step reprocesses the entire history, cumulative prefill work is

$$
\sum_{i=0}^{19}(3000+1500i)=345{,}000\ \text{tokens}.
$$

If the stable prefix is reused and only new suffixes are processed,

$$
3000+20\cdot1500=33{,}000\ \text{tokens}.
$$

The ratio is 10.4545.

![VIZ m16/04 — history cost and step latency](assets/modern-llms/en/module-16/m16_04_context_step.png)

The reason is structural. Full reprocessing sums an arithmetic progression; prefix reuse pays for each appended token once. The saving, however, depends on literal prefix stability. Reordering tool schemas, editing an early message, or changing the system prompt may invalidate reuse. Branching needs block sharing and copy-on-write rather than one linear cache.

The final context contains 33,000 tokens. At the module’s illustrative KV payload of 128 KiB per token,

$$
33000\cdot128\ \text{KiB}=4.0283\ \text{GiB}.
$$

That is one context under one model geometry. Long-lived agents therefore need context management as well as caching: external artifacts, retrieval, summaries, checkpoints, and provenance-aware memory. “Keep the complete transcript forever” is not a scalable memory policy.

## 8. Worked example B: what one agent step costs

For one illustrative step, suppose the model receives 1,500 new input tokens, emits 90 action tokens, and waits 300 ms for the external tool.

For an 8.03B model at 495 TFLOP/s effective throughput, the dense prefill arithmetic floor is

$$
\frac{2\cdot8.03\cdot10^9\cdot1500}
{495\cdot10^{12}}
=48.67\ \text{ms}.
$$

At 209.4 generated tokens per second,

$$
90/209.4=429.80\ \text{ms}.
$$

The additive scenario total is

$$
48.67+429.80+300=778.47\ \text{ms}.
$$

Decode contributes 55.2%, the tool 38.5%, and the arithmetic prefill estimate 6.3%. Those percentages are not an agent-wide law. A local calculator returns almost immediately; a remote enterprise system may take seconds. Batch size, speculative decoding, tool parallelism, and model size all change the profile.

The value of the breakdown is diagnostic. Faster decoding does little when a slow remote service dominates the critical path. Prefix reuse attacks prefill. Parallel independent calls reduce elapsed time but not total work. A twenty-step strictly sequential trajectory at this same lower bound would already take 15.57 seconds before orchestration, serialization, validation, and tail latency.

## 9. MCP: standardizing connection without outsourcing trust

**MCP (Model Context Protocol)** standardizes how an LLM application connects to external context and capabilities. A typical topology distinguishes:

- the **host**, which owns the user session and policy;
- an **MCP client**, which maintains a connection to one server;
- an **MCP server**, which advertises tools, resources, and other supported primitives.

![VIZ m16/05 — MCP and trust boundaries](assets/modern-llms/en/module-16/m16_05_mcp.svg)

The protocol reduces bespoke integration work. A capability server can be reused by multiple compatible hosts, and clients can discover descriptions dynamically.

What it does **not** do is decide that a tool should be called or that its description is trustworthy. A remote server may expose a destructive operation, return untrusted text, or ask for more authority than the user intended. The host still has to enforce origin trust, scopes, read/write separation, human confirmation, audit, and the handling of server-provided instructions.

Function calling and MCP therefore occupy different layers. Function calling describes an action selected by the model. MCP describes how a host reaches and exchanges messages with an external capability.

The current specification snapshot used for this lecture is dated 28 July 2026. A reproducible system should record the protocol and SDK versions because transports, authorization details, and extensions evolve.

## 10. Production anatomy: the harness is part of the algorithm

A teaching agent can fit into ten lines around a model call. That sketch exposes the decision loop, but it says almost nothing about the system that must run for hours, survive tool failures, preserve authority boundaries, and leave an auditable trail.

Two terms are worth separating. An **agent harness** is the full runtime around the model: tool registry, authorization, context management, execution, memory, verification, budgets, recovery, and observability. A **sandbox** is one component of that runtime—a constrained place for commands or code, with limits on filesystem access, networking, CPU, memory, and time. A sandbox reduces the blast radius of execution; it does not supply the rest of the agent contract.

### Claude Code as an unusual source-level case study

On 31 March 2026, version 2.1.88 of the `@anthropic-ai/claude-code` npm package shipped with a JavaScript source map. The artifact exposed roughly 512,000 lines of client-side TypeScript across about 1,900 files. Anthropic described the incident as a release-packaging error and stated that customer data, credentials, and model weights were not exposed.

The boundary matters. The material revealed the **client harness**—agent loop, context assembly, tool registration, permissions, compaction, persistence, and subagent machinery. It did not reveal the server-side implementation of Claude models, nor did it turn Claude Code into an open-source project. The useful evidence is therefore architectural: source-level studies can describe one concrete production client without reproducing proprietary code.

The clearest way to read that architecture is to follow a single turn through the system.

### Step 1. The decision loop is small; preparing and closing a turn is not

At the center sits a familiar ReAct-style loop:

> assemble context → call the model → parse actions → authorize → execute tools → append observations → repeat.

A systematic analysis of v2.1.88 identifies one `queryLoop` shared across interfaces and, under the authors’ code classification, attributes about 1.6% of the codebase to AI decision logic and 98.4% to deterministic infrastructure. Those percentages are not a law of agent design. They are a useful scale indicator: the loop is easy to imitate, while permissions, context management, tool routing, and recovery are the hard-to-copy parts.

Before a model call, the client resolves settings, assembles several context sources, determines the active tool surface, applies safety policy, and checks resource constraints. After the response, it has to recognize streamed calls, decide which operations may overlap, pass each one through permission gates, execute it, and evaluate stop conditions. The model proposes the next move; the harness maintains the invariants of the whole run.

### Step 2. Context degrades through a ladder, not one summary button

A long coding session cannot be managed by a single instruction to “summarize the conversation.” The source-level study describes five ordered compaction stages, moving from cheap local reduction to more destructive transformation:

1. **Budget Reduction** trims optional sources and overhead.
2. **Snip** replaces oversized tool results with smaller representations.
3. **Microcompact** compresses selected older material.
4. **Context Collapse** constructs a reduced read-time projection without necessarily rewriting the durable transcript.
5. **Auto-Compact** asks a model to create a summary continuation as the last resort.

The ordering is the design. If removing duplicated metadata is sufficient, a full model-written summary is unnecessary. If the durable log can remain intact while the next call receives a smaller projection, audit and recovery become easier.

Current product documentation exposes the outer part of the same hierarchy. Root `CLAUDE.md` instructions are reintroduced after `/compact`; auto memory uses a concise `MEMORY.md` index plus topic files; details are read on demand. The source-level analysis further describes an LLM scan over memory-file headers that selects a small set of relevant files. No vector database is required for this mechanism: in the analyzed version, the core memory path was file-based, inspectable, and editable.

### Step 3. The tool registry defines the effective action space

The source-level study counts 54 tool definitions in v2.1.88. That is a version-specific inventory including gated capabilities; the actual set varies with platform, mode, policy, and connected MCP servers.

The architectural point is stronger than the number. A mature tool is not merely a callable function. It has a name, argument schema, model-facing description, executor, side-effect class, permission rules, and availability conditions. The registry is filtered before the model sees it, and external capabilities are merged with local ones. The result resembles a **tool microkernel**: complex behavior is composed from bounded primitives rather than delegated to one unrestricted executor.

The same analysis identifies several extension surfaces—hooks, skills, plugins, and MCP—with different context costs and insertion points. It counts 27 hook events in the analyzed build. Hooks are therefore more than convenience scripts: they are deterministic points at which code can block an action, transform input, validate output, or stop a run without trusting the model to remember the rule.

### Step 4. Authorization and containment answer different questions

A permission system asks **may this principal perform the action?** A sandbox asks **what can physically happen if the action runs?** Production systems need both.

The v2.1.88 analysis describes a deny-first authorization pipeline, several permission modes, and checks before tool execution. It also finds auxiliary model calls and classifiers for some automated permission decisions. This supports a general multi-tier design: a cheaper model role can classify a local decision while a stronger model drives the task. It does not establish a permanent “Haiku routes, Opus reasons” rule, fixed latencies, or a universal 3–5× saving. Model roles are version- and mode-dependent.

Official documentation confirms the execution boundary. Claude Code uses Seatbelt on macOS and bubblewrap on Linux and WSL2; network access is mediated by an external proxy and domain policy. If the sandbox cannot start, behavior depends on `failIfUnavailable`. Consequently, “supports sandboxing” is not a complete security statement. A harness passport must record whether isolation is mandatory, which paths and domains are reachable, and what happens when containment is unavailable.

### Step 5. Memory, recovery, and delegation live outside the model’s hidden state

State is split by lifetime. `CLAUDE.md` carries durable project instructions. `MEMORY.md` is a compact auto-memory index, while topic files hold details. Session history is stored separately; the source-level analysis describes append-only JSONL transcripts and read-time message-chain patching rather than destructive rewriting of the entire record.

File edits receive checkpoints, but rollback has scope. Changes made through the tracked file tools can be restored; arbitrary shell side effects, external services, and work performed in another process require Git, idempotency, or domain-specific compensation.

Subagents receive separate context windows, prompts, tools, and permissions. The parent receives a compact result rather than all of the side transcript. This is not free parallelism: it introduces another authority boundary, another context budget, and a merge step. It does show that multi-context orchestration belongs to the harness rather than to one mysterious hidden state inside the LLM.

![VIZ m16/06 — the production harness](assets/modern-llms/en/module-16/m16_06_harness.svg)

### An evidence-aware architecture passport

| Observation for Claude Code v2.1.88 | Design implication | Evidence status |
|---|---|---|
| One `queryLoop`, 54 tool definitions, 27 hook events, several permission modes | Most engineering complexity surrounds a small model loop | Counts from an independent source-level analysis of one version |
| Five compaction stages and file-based memory without a required vector database | Context is a managed hierarchy, not an ever-growing string | Source-level analysis; public memory behavior is also documented |
| Auxiliary model calls and selectable subagent models | Different roles can be assigned to models of different cost and strength | Architectural capability, not a fixed router/reasoner pair |
| Deny-first policy, hooks, Seatbelt/bubblewrap, and network controls | Authorization and containment form separate defense layers | Source analysis plus current product documentation |
| Durable transcripts, checkpoints, and subagent sidechains | Persistence and recovery are designed outside the context window | Source analysis plus documented user-facing mechanisms |
| Universal latency numbers, 3–5× savings, or a 30–50 point “harness delta” | These cannot be derived from architecture alone | Requires a controlled comparative experiment |

Experimental branches and feature flags found in the exposed tree should not be treated as released product guarantees. Likewise, the 98.4% figure is the classification of one study, not a physical constant. The robust conclusion is qualitative: **the model chooses candidate actions, while much of capability, safety, and durability is created by deterministic systems around it**.

That conclusion leads directly to the next section. Without verification, idempotency, recovery, and bounded retries in the harness, a modest local error rate compounds into poor reliability over a long trajectory.

## 11. Worked example C: reliability over a long trajectory

Under the simplifying assumption of independent required steps with success probability $p$,

$$
P(\text{trajectory success})=p^n.
$$

At $p=0.98$ and $n=20$,

$$
0.98^{20}=0.6676.
$$

At $p=0.95$,

$$
0.95^{20}=0.3585.
$$

![VIZ m16/07 — reliability and Amdahl’s law](assets/modern-llms/en/module-16/m16_07_reliability_amdahl.png)

Allow one independent retry after failure. Step success becomes

$$
p'=1-(1-p)^2.
$$

For $p=0.98$, $p'=0.9996$ and the twenty-step trajectory reaches 0.9920. For $p=0.95$, it reaches 0.9512.

Independence is the decisive caveat. Repeating the same malformed call under the same conditions may be almost perfectly correlated with the first failure. A useful retry introduces information: a diagnostic, a revised argument, a different tool, or human assistance.

Side effects create another failure mode. The operation may have succeeded while the response was lost. A blind retry then duplicates the action. Idempotency or a read-before-write check is required.

The inverse calculation is sobering. To achieve 90% success over 50 required independent steps without retries,

$$
p\ge0.9^{1/50}=0.997895.
$$

This is why verification, bounded repair, checkpoints, and escalation are structural parts of long-horizon agents rather than optional polishing.

## 12. Subagents and Amdahl’s law

A **subagent** is a separately scoped worker with its own task, context, and limits whose result returns to an orchestrator. Delegation is useful when work can proceed independently—different repository modules, sources, or candidate solutions.

Use the module’s scenario:

- 400 minutes of parallelizable work;
- 5 minutes of planning;
- 20 minutes of merging;
- 20 workers.

Ideal divided work takes 20 minutes, so the critical path is

$$
T(20)=5+20+20=45\ \text{minutes}.
$$

Relative to the 400-minute useful-work number, speedup is

$$
400/45=8.89.
$$

Relative to a strict one-worker end-to-end baseline that includes the same planning and merge,

$$
425/45=9.44.
$$

Both ratios are correct; they answer different questions. As the worker count tends to infinity, the 25 serial minutes remain, yielding ceilings of 16× and 17× for the two denominators.

![VIZ m16/08 — delegation and the critical path](assets/modern-llms/en/module-16/m16_08_dynamic_workflows.svg)

Parallel contexts also consume memory. At 30,000 tokens and 128 KiB of KV per token, twenty workers require

$$
20\cdot30000\cdot128\ \text{KiB}=73.242\ \text{GiB}.
$$

That excludes weights, allocator overhead, and temporary buffers. Real workloads also suffer from uneven task size, duplicate exploration, merge conflicts, and stragglers.

Current agent SDKs and products expose subagents and workflow orchestration, but no API feature repeals Amdahl’s law. Before delegation, construct a dependency graph and decide which outputs can be verified and merged independently.

## 13. Level-1 code: a bounded protocol loop

The following loop is intentionally more interested in boundaries than intelligence. An action must contain exactly a tool name and arguments; the tool comes from an allow-list; argument names and types are checked before execution; and both step count and tool-call count are bounded.

```python
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

@dataclass(frozen=True)
class Tool:
    """Allowlisted handler and its exact scalar argument contract."""
    name: str
    arguments: Mapping[str, type]
    handler: Callable[..., Any]

def strict_json_loads(payload: str) -> dict[str, Any]:
    """Parse JSON; reject duplicate keys and non-standard numeric constants."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    obj = json.loads(
        payload,
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )
    if not isinstance(obj, dict):
        raise ValueError("Action must be a JSON object")
    return obj

def invoke(toolbox: Mapping[str, Tool], action: Mapping[str, Any]) -> Any:
    """Validate the tool name and arguments before executing the handler."""
    if set(action) != {"tool", "args"}:
        raise ValueError("Action requires exactly tool and args")
    tool = toolbox.get(action["tool"])
    if tool is None:
        raise ValueError("unknown tool")
    args = action["args"]
    if not isinstance(args, dict) or set(args) != set(tool.arguments):
        raise ValueError("argument names do not match the schema")
    for name, expected_type in tool.arguments.items():
        value = args[name]
        if not isinstance(value, expected_type) or (
            expected_type is int and isinstance(value, bool)
        ):
            raise TypeError(f"{name} must be {expected_type.__name__}")
    return tool.handler(**args)

def react_loop(
    llm: Callable[[Sequence[dict[str, Any]]], str],
    toolbox: Mapping[str, Tool],
    user_query: str,
    *,
    max_steps: int = 10,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    """Run a bounded Action/Observation loop and retain a structured trace."""
    events = [{"kind": "user", "text": user_query}]
    tool_calls = 0
    for step in range(1, max_steps + 1):
        response = llm(tuple(events)).strip()
        events.append({"kind": "model", "step": step, "text": response})
        if response.startswith("Answer:"):
            answer = response[len("Answer:"):].strip()
            return {"status": "completed", "answer": answer, "events": events}
        if not response.startswith("Action:"):
            events.append({"kind": "protocol_error", "step": step})
            continue
        if tool_calls >= max_tool_calls:
            return {"status": "tool_budget_exhausted", "answer": None, "events": events}
        tool_calls += 1
        try:
            action = strict_json_loads(response[len("Action:"):].strip())
            result = invoke(toolbox, action)
            events.append({"kind": "tool_result", "step": step, "value": result})
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            events.append({
                "kind": "tool_error",
                "step": step,
                "error_type": type(error).__name__,
                "message": str(error),
            })
    return {"status": "max_steps_exhausted", "answer": None, "events": events}
```

This is not a production runtime. It lacks resource-level authorization, idempotency, checkpoints, observation-size limits, and a real code sandbox. It does establish one critical property: a model error becomes a typed trace event instead of arbitrary execution.

A tool failure is returned as an observation so that the model may repair the action. The budgets prevent an endlessly repeated failure from becoming an unbounded session.

## 15. System landscape: compare action surfaces, not logos

Agent systems can be classified by what they are allowed to manipulate.

### API and SDK agents

OpenAI Agents SDK and Anthropic Agent SDK provide runtime primitives for tools, handoffs, tracing, and multi-step execution. They reduce orchestration boilerplate but do not choose the right permissions, tool descriptions, or verifier for an application.

### Terminal and coding agents

Claude Code, Codex-style environments, SWE-agent, OpenHands, and related systems read files, run commands, edit code, and execute tests. Here the ACI matters enormously: command design, observation formatting, edit primitives, network access, and failure feedback all alter the effective policy.

### Browser and desktop agents

WebArena offers reproducible websites; OSWorld offers real desktop applications across operating systems; computer-use APIs expose screenshots plus mouse and keyboard actions. The richer surface adds visual grounding errors, UI nondeterminism, and prompt-injection risks from untrusted page content.

### Durable and multi-worker workflows

Background jobs, checkpoints, queues, and subagents extend work beyond one context or one process. The harness now behaves like a workflow engine: it must restore state, manage versions, and avoid repeating side effects after recovery.

![VIZ m16/09 — four agent system surfaces](assets/modern-llms/en/module-16/m16_09_landscape.svg)

This is not a ranking. The same checkpoint may score differently under two harnesses because the action space, prompts, memory, retries, and observations changed.

## 16. Benchmarks: each measures a different agent–environment pair

Agent benchmarks differ primarily in the environment and verifier they provide.

**BFCL (Berkeley Function Calling Leaderboard)** focuses on function choice, arguments, parallel calls, and multi-turn tool use. It is well suited to protocol behavior but does not by itself measure a long-running mutable environment.

**General AI Assistants (GAIA)** asks general-assistant questions that may require browsing, files, multimodality, and tools, with concise verifiable answers. It measures coordination across capabilities rather than durable side effects.

**WebArena** supplies reproducible websites and evaluates end states. **OSWorld** extends execution-based evaluation to desktop applications and multiple operating systems.

**Software Engineering Benchmark (SWE-bench)** uses real GitHub issues and test-based patch verification. **Terminal-Bench** covers a broader range of command-line tasks. Results in these settings depend strongly on the commands and observations exposed by the harness, container versions, time limits, and network policy.

Two scores are not comparable unless they share, or explicitly report differences in:

- model revision and sampling;
- system prompt and tool descriptions;
- step, token, retry, and wall-clock budgets;
- network and external-data access;
- environment and repository revision;
- verifier implementation;
- number of attempts and aggregation rule.

Research on agent harness design makes the same point: the benchmark evaluates a system. This is not a flaw. The scientific requirement is to describe the system well enough to reproduce it.

## 17. Reading an agent-harness passport

A useful passport records the conditions under which a trajectory was produced.

### Model

Exact model/API revision, sampling parameters, reasoning-effort setting, context limit, and history-compression policy.

### Prompt and tools

System prompt, tool schemas and descriptions, action/observation format, and MCP specification/SDK versions where relevant.

### Permissions and environment

Network, filesystem, secrets, and external-service access; container/OS and initial state; read/write boundaries, confirmation rules, and idempotency behavior.

### Loop policy

Limits on steps, tokens, time, cost, tool calls, and concurrency; retry and replanning conditions; memory/checkpoint behavior; definitions of `completed`, `blocked`, `needs_user`, `budget_exhausted`, and `failed`.

### Evaluation

Task and verifier versions, number of runs, success together with cost and latency, protocol/tool/permission/budget failure rates, and redacted traces.

This passport matters outside papers. When an upgrade changes performance, the record lets the team determine whether the model changed or the prompt, schema, environment, verifier, or budget moved underneath it.

## 20. Key takeaways and sources

![VIZ m16/10 — the agent system in one page](assets/modern-llms/en/module-16/m16_10_cheatsheet.svg)

**An agent is a system, not merely a model.** The model proposes; the harness validates, authorizes, executes, stores state, verifies, and stops.

**Action correctness has layers.** Grammar protects syntax, schema protects the data contract, and policy protects authority and side effects.

**ReAct, Plan-and-Execute, and CodeAct offer different tradeoffs.** Adaptivity, inspectable plans, and compact composition can be combined.

**Loop history has a compute and memory bill.** The teaching scenario falls from 345K processed prefill tokens to 33K with prefix reuse, ending at 4.03 GiB of KV.

**Small local failure compounds.** $0.98^{20}=0.668$. Retries help only with correlated-error and side-effect controls.

**Delegation obeys Amdahl’s law.** Twenty workers do not create 20× speedup in the scenario, and their contexts alone require 73.2 GiB of KV.

**A harness is broader than a sandbox.** The sandbox constrains execution; the harness also manages tools, authority, memory, verification, recovery, and observability.

**MCP standardizes connectivity, not trust.** Function calling, protocol transport, authorization, and sandboxing are separate layers.

**A benchmark score belongs to a configuration.** Model, prompt, tools, budgets, environment, verifier, and attempts form the measurement unit.

**Primary sources:**

- Yao et al., ReAct — [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Wang et al., CodeAct — [arxiv.org/abs/2402.01030](https://arxiv.org/abs/2402.01030)
- Willard & Louf, guided generation / Outlines — [arxiv.org/abs/2307.09702](https://arxiv.org/abs/2307.09702)
- Yang et al., SWE-agent and ACI — [arxiv.org/abs/2405.15793](https://arxiv.org/abs/2405.15793)
- SWE-bench — [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
- WebArena — [arxiv.org/abs/2307.13854](https://arxiv.org/abs/2307.13854)
- GAIA — [arxiv.org/abs/2311.12983](https://arxiv.org/abs/2311.12983)
- OSWorld — [arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)
- Terminal-Bench — [arxiv.org/abs/2601.11868](https://arxiv.org/abs/2601.11868)
- MCP specification 2026-07-28 — [modelcontextprotocol.io/specification/2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)
- OpenAI Agents SDK — [developers.openai.com/api/docs/guides/agents](https://developers.openai.com/api/docs/guides/agents)
- Anthropic Agent SDK — [docs.anthropic.com/en/docs/claude-code/sdk](https://docs.anthropic.com/en/docs/claude-code/sdk)
- Claude Code as a harness case study — [tool reference](https://code.claude.com/docs/en/tools-reference), [memory](https://code.claude.com/docs/en/memory), [sandboxing](https://code.claude.com/docs/en/sandboxing), [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks), [subagents](https://code.claude.com/docs/en/sub-agents), [checkpointing](https://docs.anthropic.com/en/docs/claude-code/checkpointing)
- Claude Code source-map exposure on 31 March 2026 — [InfoQ: incident description and Anthropic statement](https://www.infoq.com/news/2026/04/claude-code-source-leak/)
- Liu et al., *Dive into Claude Code: The Design Space of Today’s and Future AI Agent Systems* — [arxiv.org/abs/2604.14228](https://arxiv.org/abs/2604.14228), [analysis repository](https://github.com/VILA-Lab/Dive-into-Claude-Code)

> **Further study.** Training agent policies, reinforcement learning in environments, and large-scale rollout infrastructure are developed in *RL for LLM*. This module focuses on the runtime of an already trained model.

**Next:** Module 17 turns to evaluation. Agent benchmarks make the central lesson especially visible: a number without an environment, verifier, budget, and harness is not a complete measurement.

---

*Landscape verified: 5 August 2026. Protocol, SDK, and environment information was checked against primary sources; all numerical examples in Sections 7–12 are reproducible teaching scenarios under stated assumptions, not measurements of a named product.*
