# Module 1. The Modern LLM Landscape

From a language model to a working system

We usually meet a language model at the moment it begins to answer. An explanation appears on the screen, or a piece of code, a translation, a solution. The interface makes the exchange feel simple. Almost everything that made it possible remains out of sight: training on large datasets, computation inside the network, the choice of each continuation, the preparation of the input and, sometimes, an entire program coordinating searches and tools.

This course will gradually bring those hidden parts into view. We will trace how text becomes numbers, how those numbers become a probability distribution, how training changes a model, and why the same weights can produce different results under different conditions. We will read equations, write code, and build a small model of our own. Before examining the individual mechanisms, however, we need a sense of the whole: which questions concern the neural network itself, which concern its training, and which concern the system that puts it to work.

Let us begin with a task that requires no prior knowledge of the Transformer. As we examine it, it will lead us to almost every major theme of the course.

**How to read this chapter.** The main text forms a continuous argument, introducing each concept as it becomes useful. Resource estimates, a worked example using an open model, and the MiniLLM configurations follow in the technical appendices. They extend the discussion; you do not need to read them to understand the next section of the main chapter.

A general understanding of a neural network as a trainable function is enough to begin. The calculations in the appendices assume algebra and conditional probability. No knowledge of Transformer internals is required.

## 1. One answer, several problems to solve

Suppose we are building an assistant for an organization’s internal documents. An employee asks, “How many days should we retain an experiment log after the experiment ends?” The assistant must find the relevant rule, answer the question, and identify the document that supports its answer.

To begin, let us keep just two documents in the collection. These are invented examples, not real data-retention requirements.

**Document A · Experiment logs**

An experiment log must be retained for 30 days after the experiment ends.

**Document B · Temporary exports**

A temporary export must be deleted 7 days after it is created.

The answer seems obvious: “30 days, according to Document A.” But what, exactly, have we just judged to be correct?

The assistant first had to distinguish an experiment log from a temporary export. Both documents concern stored data, and both contain a number of days. Matching a few familiar words would not be enough: the object in the question has to be matched to the object governed by the rule. The event that starts the clock matters too. The end of an experiment and the creation of a file are not the same event. Finally, the citation must support the claim. “30 days, according to Document B” contains the right number but the wrong evidence.

There is an earlier requirement as well: the assistant must have access to the documents. If the program failed to retrieve Document A, or never passed it to the model, an instruction to “answer from the source” cannot supply the missing source. The model might decline to answer, ask for more information, or produce a plausible continuation. None of these is equivalent to extracting a rule from a document it cannot see.

Even this small example contains several distinct ways to fail. The relevant material may not be found. It may be found but misread. The rule may be read correctly but applied under the wrong conditions. The answer may be right while its citation is wrong. From the outside, all of these can look like the same thing—a poor answer from the assistant. Yet each calls for a different repair.

We have reached our first distinction, one that will recur throughout the course: **a language model is not the same object as the program that uses it**. The neural network performs a particular computation. The program decides what to supply as input, how to arrange a sequence of calls, and what counts as success. Improving the system requires understanding both, without crediting one for work done by the other.

For now, let us set retrieval and source verification aside. Assume that the question and both passages have already reached the neural network. What happens next?

## 2. What a language model predicts

The input text is first converted into a sequence of **tokens**. A token might be a word, part of a word, a punctuation mark, or another unit defined by the tokenizer. Each token is assigned an integer identifier. What reaches the model is no longer a string in the ordinary human sense, but a sequence of numerical identifiers. Module 2 will explain how these units are chosen and why the same text can have different token counts for different models.

Our main subject will be the autoregressive language model. At each step, it estimates the probabilities of the next token given the preceding content available to it. “Autoregressive” refers to this order of construction: the sequence already in place supplies the condition for its own continuation.

Write the next token as $x_{t}$ and all preceding tokens as $x_{<t}$. The central computation then has a remarkably compact form:

$$
p_\theta(x_t\mid x_{<t})
$$

The subscript $θ$ denotes the model’s parameters: the numerical values obtained through training. Read the expression as “the probability of token $x_{t}$, given the preceding text and the model’s parameters.” That preceding sequence may include an instruction, documents, the user’s question, and the part of the answer generated so far.

Imagine that the assistant has begun, “The experiment log should be retained for…” The model assigns scores to possible continuations. Computing a distribution, however, does not itself append anything to the answer. A **decoding algorithm** chooses the next token. It might select the most probable option, or sample according to a specified rule. The chosen token is appended to the sequence, and the computation repeats. Generation stops when the designated end signal appears or a limit imposed by the program is reached.

1.  **Available sequence**

    Instructions, question, data, answer so far

2.  **Model forward pass**

    Probability estimates for the next token

3.  **Choose a continuation**

    The decoding rule selects a token

4.  **Take another step**

    Append the token and repeat the cycle

Using a trained model is called **inference**. In ordinary inference, the parameters $θ$ remain fixed. The input sequence changes, and with it the internal representations and the distribution of the next token. The same model can therefore give different answers to different questions. With a sampling-based decoding rule, it can also produce different answers to the same input.

This picture reveals what several apparently different tasks have in common. Translation, document question answering, code completion, and problem solving can all be framed as producing suitable text under a given condition. What is shared is the form of the computation—not a promise of equally good performance on every task. Success depends on what the model has learned and whether the input helps it bring those capabilities to bear.

Nor does the brevity of the equation imply that the underlying computation is simple. Predicting continuations of complex text may call for layered representations, long-range dependencies, and substantial transformations of information. “Predicting the next token” describes the output and the training task; by itself, it establishes nothing about the simplicity of the internal algorithm. The opposite inference would be equally mistaken: an ability to continue an argument does not guarantee that the argument is correct.

Above all, a token probability is not a probability that a claim is true. A high value means that the model assigns high probability to that continuation in that context. It is not, without further justification, a statistical estimate that a retention rule is valid, a cited paper exists, or a solution is correct. Those conclusions require their own evidence and checks.

We will study the autoregressive route in detail. It is not the only route available: Module 12 introduces another way to construct text, through successive recovery of masked content. For now, we have identified our first well-defined object—a function that maps an available context to a distribution over continuations. The next question is how that function acquires useful behavior. [\[M20\]](/courses/modern-llms/module-20/)

## 3. How models learn to work with language

Before training, a neural network’s parameters do not yet define a useful language model. Its architecture specifies which operations will be performed and how they connect. It does not supply the rules of English or Russian, the syntax of Python, or the contents of a technical manual. Training must turn regularities in the data into useful structure in the parameters.

**Pretraining** uses a large corpus of text. In the autoregressive setting, the text itself provides many training examples. The beginning of a sequence serves as the input; the next token, already known from the corpus, serves as the target. A single passage can supply a prediction task after its first token, another after its first two, another after its first three, and so on. No person has to label each continuation separately. The supervision comes from the structure of the data.

The model produces probabilities, a loss function measures how well they match the observed continuation, and an optimization algorithm adjusts the parameters. Assigning too little probability to the actual next token incurs a larger penalty. Repeated across many examples, this process aims to improve prediction on text resembling the training data and, where possible, to carry the learned regularities over to new cases. Appendix A gives the mathematical objective; Modules 3 and 9 will show how the computation works.

Knowing the target during training is not the same as allowing the model to see it while making its prediction. The training program knows the correct tokens, but an autoregressive model must not receive the future answer as part of its permitted context. In a Transformer, a special mask enforces this restriction. Many positions in a training sequence can nevertheless be processed in parallel: unlike during generation, the correct input for each position is already available. We will later examine how restricted access to the future and parallel computation fit together.

What can a model gain from this process? Predicting text well takes more than memorizing the frequencies of neighboring words. A continuation depends on grammar, topic, genre, notation, facts, the structure of an argument, and statements made many sentences earlier. Training creates pressure to form representations that capture these dependencies. We do not, however, prescribe a complete inventory of those representations, nor do we receive a guarantee that every useful regularity will be discovered.

It is therefore helpful to think of trained parameters as a distributed representation of learned regularities, not as a neatly indexed document collection. A model can memorize particular sequences—sometimes very precisely—while also generalizing to combinations it has never encountered. The balance depends on the data, the training procedure, and the task. Being able to reproduce a fact does not imply that the model stores a verifiable record of it, complete with a source, a date, and conditions of applicability.

Consider our document assistant again. Pretraining may give it the language skills needed to distinguish “after creation” from “after completion.” Our organization’s particular rule, however, may never have appeared in its training data. The rule might even have changed yesterday. Language competence and access to the current policy solve different problems. We expect the former from the trained model; we must arrange the latter through the way we use it.

Model size affects the space of possible solutions, but does not determine what will actually be learned. The amount, composition, and quality of the data matter, as do the compute budget and optimization procedure. The Chinchilla study made a particularly clear case for choosing parameter count and training-token count together under a fixed compute budget, rather than increasing parameters alone. Its empirical relationships belong to a particular experimental setting; treating them as an eternal ratio for every training run would miss the point. [\[3\]](https://arxiv.org/abs/2203.15556)

Architecture, then, defines a way to compute; training determines the particular parameters of that computational system. But predicting text well is still not the same as being a useful assistant.

## 4. From text prediction to helpful behavior

A text corpus contains many kinds of continuation: explanations, dialogue, fiction, mistakes, arguments, instructions, and examples of behavior we would rather not reproduce. A model trained to continue such a corpus has no obligation to adopt the role of a careful assistant. Given a question, it might continue a list of questions, imitate the surrounding discussion, or produce something that looks like an answer without meeting the user’s intention.

To make the desired behavior more consistent, we can use **supervised fine-tuning**, or SFT, on instructions and suitable responses. Instead of arbitrary passages, the training examples demonstrate how to respond to particular requests. The model learns patterns of interaction: following an instruction, explaining an idea, transforming data, or respecting a requested output structure. This still changes the parameters through training. It is not merely a setting applied to a chat window.

For a document assistant, SFT examples might show how to distinguish a rule from an exception, cite a source, and acknowledge insufficient evidence. Those examples need to demonstrate a real connection between the answer and the supplied material. If every example can be imitated by writing a confident sentence and attaching a plausible document title, the model may learn that surface pattern rather than the behavior we intended.

Demonstrations are not the only source of supervision. Comparing two answers and choosing the better one is often easier than writing an ideal answer from scratch for every request. Preference-based training uses this kind of feedback. In a classic RLHF pipeline, a reward model is first trained to score responses, and reinforcement learning is then used to update the language model. DPO takes a different route: it optimizes the model directly on preference pairs, without a separate policy-optimization stage driven by an explicitly trained reward model. Module 14 will explore why these distinctions matter. [\[4\]](https://arxiv.org/abs/2203.02155), [\[5\]](https://arxiv.org/abs/2305.18290)

Some tasks offer feedback through a verifiable outcome rather than a human preference. A program passes the specified tests or it does not; a numerical answer satisfies the required condition or it does not. Such signals can be valuable, but they also have limits. Passing an incomplete test suite does not establish that a program is correct on every input. Matching a reference answer does not necessarily validate the explanation that accompanies it. Training quality depends not only on the optimization method, but also on how faithfully the check captures the real objective.

**Post-training**—the collective name for stages following the main pretraining run—can therefore change much more than a model’s tone. It may affect instruction following, problem-solving strategies, tool use, responses to uncertainty, and other aspects of behavior. It does not confer infallibility. Even an excellent training signal remains a limited description of what we want.

For our example, the practical distinction is straightforward. Adding “always cite your source” to a prompt and training on verified answers with supporting citations are different interventions. The first changes the current input. The second changes the parameters and may influence many future requests. Checking whether a document actually supports a particular answer is a third task. A helpful style is not a substitute for evidence.

We can now read model release names more carefully. Two models with the same architecture and parameter count may have different training data and training histories. A base model and its fine-tuned counterpart are not simply the same object behind different interface settings. Understanding the difference requires asking both how the model is built and how its weights were obtained.

## 5. What changes during a conversation

Suppose the assistant first answers from Document A. The user then adds, “A special directive applies to the current experiment: retain the log for 60 days.” The next answer may change. A natural question follows: has the model just learned something?

In an ordinary chat interaction, the new information has entered the context while the network’s parameters have remained unchanged. The model is applying an existing ability to take task conditions into account, now to a different input. It helps to distinguish two senses of “learning”: adapting behavior to examples in the current context, and updating parameters through an optimization procedure. These processes can look similar from the outside while working in very different ways.

**In-context learning** is the ability to use examples or a task description supplied in the prompt without updating the weights. We might, for instance, show several examples of document-type labels and ask the model to classify new documents in the same format. The GPT-3 paper made this setting central to the study of large language models: in the reported experiments, new tasks were specified in text, without gradient updates to the model’s parameters. [\[2\]](https://arxiv.org/abs/2005.14165)

But a new sentence in the context is not automatically a valid rule. In our task, the user’s message might be a legitimate clarification, a mistake, or an unsupported assertion. The task specification and the system’s policies determine what counts as an authoritative directive and which source takes precedence. Being able to use information is different from being entitled to treat it as authoritative.

It is also worth distinguishing three places where information can reside. The weights encode the outcome of training. The context contains what has been supplied for the current computation. An external store may hold documents, interaction history, notes, and results of earlier actions. For information in that store to influence an ordinary model call, the program must retrieve it and present it in a form the model can process.

**Weights**

Updated during training. They determine how the model transforms its input.

**Context**

Changes between requests. It supplies the available conditions for the current answer.

**External memory**

Managed by the program. It stores information that can be retrieved and supplied again.

An assistant that appears to “remember the whole conversation” may therefore rely on several different arrangements. The program might pass the entire available history, provide a summary instead, or select a few records from a store. Those choices determine what the model can take into account and what remains out of reach. The distinction will become central when we study RAG and agents.

Updating weights during deployment is possible too, but it requires a deliberately arranged procedure; we should not assume it happens with every message. Along the course’s main path, we will keep parameter-updating training separate from fixed-weight inference. This distinction will spare us considerable ambiguity when we discuss reasoning, memory, and adaptation.

We now know where the parameters come from and what reaches the input. It is time to look inside the computation itself—without yet unpacking every matrix.

## 6. Inside the computation

After tokenization, each identifier is mapped to a numerical vector: an **embedding**. Its coordinates need not have tidy human labels such as “formality,” “date,” or “importance.” An embedding is a learned representation whose meaning depends on how the rest of the network uses it to solve the task.

The initial vector cannot do the whole job. The word “log,” for example, might refer to an experimental record, a file of system events, or a piece of timber. Token representations must therefore be transformed in light of their surroundings. Repeating this contextual transformation at several levels accounts for a substantial part of the network’s work.

In the **Transformer** architecture, attention plays a central role. It allows the representation at one position to draw information from other permitted positions in the sequence. A question about retention may require connecting an object’s name, a number of days, and the event that starts the clock. Attention supplies a computational mechanism for exchanging that information; useful patterns of interaction emerge through training. It is an operation on internal vectors, not a separate document-search program. [\[1\]](https://arxiv.org/abs/1706.03762)

Alongside attention sits a feed-forward network, usually abbreviated **FFN**. In its basic form, it transforms each position’s representation separately, using the same learned matrices across positions. Attention mixes information between positions; the FFN applies a nonlinear transformation to the resulting features. Alternating these operations makes increasingly complex representations possible.

The network also needs a way to represent positions if it is to take order into account. Normalization and residual connections help make deep computation trainable; a residual connection adds the previous representation to the result of a new transformation. For now, understanding the purpose of these components is enough. We will assemble a basic Transformer in Module 3, then examine positions, attention, and block internals separately in Modules 4–6.

At the end of the computation, a representation is converted into scores for the vocabulary’s tokens. Normalizing those scores produces the next-token distribution with which we began. A compact probabilistic expression thus stands for a long but well-defined sequence of operations. Each can be written as an equation, implemented, and checked on a small example.

Why does the Transformer occupy so much of this course? Introduced in 2017, the architecture combined information exchange across positions with convenient parallel training on sequences. That does not make the field a study of one fixed design. Variants change the form of attention, the feed-forward blocks, positional representations, and the organization of computation. Similar names can conceal quite different paths through the data. [\[1\]](https://arxiv.org/abs/1706.03762)

There are deeper alternatives as well. **State space models**, or SSMs, can carry information about the past through an evolving state rather than a full collection of separately accessible representations of earlier positions. That changes the balance between memory, computation, and access to past detail. The Mamba family shows how state updates can depend on the current input; hybrid models combine such mechanisms with attention. We will study them in Module 8. [\[7\]](https://arxiv.org/abs/2312.00752)

A separate design question concerns the order in which an answer is constructed. An autoregressive model extends it sequentially. A diffusion language model may instead start from a partially or entirely masked sequence and refine it over several steps. LLaDA, for example, uses a Transformer to predict masked tokens. “Transformer” and “diffusion model” are therefore not necessarily opposing categories: one describes an internal architecture, the other an important aspect of training and generation. Module 12 will make these levels explicit. [\[8\]](https://arxiv.org/abs/2502.09992)

The point to retain for now is simple. A language model is not one particular machine with a mandatory inventory of parts. It is a class of trainable systems whose architectures make different trade-offs in processing sequences. To compare them meaningfully, we first need to understand which resource a design choice changes.

## 7. Why parameter count is only part of the story

When we first encounter language models, parameter count looks like an appealing summary. It is compact, easy to compare, and seems to express capability in a single number. Yet the distinction between architecture and training already tells us where that intuition breaks down: the same number of parameters can accompany different data, different training procedures, and different computations.

Size still matters. Parameters have to be stored and used. A model with 8 billion parameters, each stored in two bytes, needs 16 billion bytes for the weight values alone: 16 GB, or approximately 14.90 GiB. That is not the full memory requirement of a running model. Context-related state, intermediate results, working buffers, and the runtime itself also take space. Even this first estimate, however, establishes the scale of the problem.

Here and throughout the chapter, B in a model size means billion, or 10⁹ parameters; T means trillion, or 10¹². For storage, we distinguish decimal GB from binary GiB: one GiB is 2³⁰ bytes. The difference may be easy to overlook in conversation, but matters when we are deciding whether a model will fit on a device.

Now consider the feed-forward blocks. In a **dense model**, tokens pass through the same set of these blocks. In a **mixture of experts**, or MoE, some blocks contain a collection of alternative subnetworks. A small learned router chooses only a subset for each token. These subnetworks are called experts. The term is architectural: it does not imply that one expert necessarily handles mathematics, another Russian, and a third programming. [\[6\]](https://arxiv.org/abs/2101.03961)

MoE makes it possible to enlarge the total parameter pool without requiring every token to use all of it. We therefore need two quantities: the **total parameter count**, $N_{total}$, and the **active parameter count**, $N_{active}$, attributed to processing one token under a stated counting convention.

Consider a deliberately simplified model. Its shared part contains 1 billion parameters, while its selectable expert bank contains 16 equal portions of 0.5 billion each. Suppose a token uses two of those portions. The total count is 1 + 16 × 0.5 = 9 billion parameters; the active count is 1 + 2 × 0.5 = 2 billion. In a real multilayer model, routing happens within individual layers, and the shared part has its own structure. The toy calculation isolates the distinction.

These two numbers answer different questions. Total size helps estimate weight storage. Active size gives a first indication of part of the computation per token. We cannot store only the 2 billion selected on one step and declare the rest unnecessary: the next token may follow a different route. Weights can be distributed across devices or loaded on demand, but then moving the data becomes part of the cost.

Even active size is not a direct measure of speed. Runtime depends on memory traffic, communication between devices, batch size, and how effectively the implementation uses the hardware. “Active” also needs a definition. The full input embedding table exists in the model, for example, but a particular token usually selects just one row. The output projection, by contrast, may be used to score the entire vocabulary. Module 7 will examine this distinction in real code. [\[M7\]](/courses/modern-llms/module-07/)

Another way to reduce memory demand is **quantization**: storing weights with fewer bits. Values are represented less precisely, and additional information may be needed to recover their scales. This often reduces storage, but its effect on quality and speed depends on the method and implementation. A four-bit weight format does not justify dividing the memory of an entire BF16-based run by four. Tensors need not all use the same format, and the context state and working buffers have not disappeared. Module 11 is devoted to these questions.

Parameter count, then, should be read alongside architecture and storage format. Quality must be tested separately. A large weight bank, a small active path, a compact checkpoint, and a fast response are different properties. Their relationships can be investigated; none is a universal substitute for the others.

## 8. The cost of reading a long document

Let us return to the assistant. With only two short sentences, length barely entered the discussion. A real collection contains long policies, appendices, tables, and several versions of the same document. It is tempting to supply everything at once and let the model find what matters.

To assess that option, we need the idea of a **context window**. It limits the sequence length that the model and runtime handle in a given mode. In an ordinary autoregressive setting, the sequence includes not just documents but instructions, history, the question, and the generated continuation. An interface may impose separate input and output limits as well, so the documented numbers have to be read together.

Suppose our teaching environment permits a total sequence of 8,192 tokens. Instructions, history, and the question use 1,000; the documents use 6,000; and we want to reserve 1,500 for the answer. The total is 8,500—308 over the limit. Documents shorter than the window do not necessarily leave enough room for the complete request and the desired continuation. We do not yet know how to count tokens in a string accurately; Module 2 will close that gap.

Admissible length is only one aspect of context. Another is how well the model uses the information it receives. A model may accept a sequence yet overlook an important exception, confuse similar passages, or fail to connect distant pieces. In the experiments reported in *Lost in the Middle*, performance on some models and tasks depended substantially on the position of the relevant information. This was an experimental finding, not an immutable law governing all future models. Its wider lesson is to test context use rather than infer it from an advertised token limit. [\[9\]](https://arxiv.org/abs/2307.03172)

A third aspect is processing cost. For an autoregressive Transformer, it is useful to distinguish two phases. First comes **prefill**, which processes the input sequence already available. Its result lets us select the first continuation token. Then comes **decode**, which processes appended tokens in sequence and chooses the continuation that follows. Much of the input can be processed together. Future answer tokens are not yet known, so ordinary generation proceeds one step at a time.

These phases create different kinds of waiting. A user cares both about the delay before the first answer token and about how quickly the rest arrive. The first measure is **time to first token**, or TTFT. It includes not just model computation but potentially queuing, request preparation, and other system delays. The interval between subsequent tokens is **inter-token latency**, or ITL. A short answer emitted quickly after a long document-reading delay is a different performance profile from an answer generated slowly after a short input.

To avoid recomputing all the information it needs from earlier positions, a Transformer usually retains attention-related vectors in a **KV cache**. For now, its role matters more than its details: it is intermediate numerical state that allows computation to continue. It is neither the model’s weights nor an external document database. With full attention, this state grows with the length of the retained sequence. Modules 5 and 10 will explain its geometry and ways to reduce its cost; Appendix B already provides a numerical example. [\[M5\]](/courses/modern-llms/module-05/), [\[M10\]](/courses/modern-llms/module-10/)

Our assistant now faces three independent questions. Does the material fit? Can the model use it reliably for our task? Are the memory, monetary, and time costs acceptable? “Supports a long context” answers only part of the first question.

This gives retrieval a concrete motivation. Supplying every document is not always necessary; finding a few relevant passages may be preferable. Shortening the input has a risk of its own, however: retrieval may leave out a necessary condition or exception. Long context and retrieval are not successive rungs on a ladder from an obsolete solution to a superior one. They are different ways of arranging access to information, often best combined and always worth testing on the task.

## 9. Spending more compute on an answer

Some questions ask for a single explicit statement. Others require several rules to be reconciled, an exception to be considered, or a calculation to be made. Our assistant might be asked, “Can the log be deleted if the general retention period has expired but the experiment’s review is still open?” One retrieved number will not settle that question. We need more information and a way to put it together.

We tend to think of a model’s capabilities as something acquired during training. But the work performed after a particular request arrives also affects the result. The model may develop an intermediate solution. The program may obtain several candidates, check them, perform a calculation with a tool, or retrieve further evidence. These are different ways to spend more **test-time compute**—computation devoted to solving the current task.

This budget should not be confused with training. Training changes parameters, potentially benefiting many future requests. Ordinary fixed-weight problem solving changes the sequence, intermediate state, and results of actions. A model may have been trained to use a productive strategy; applying that strategy to a new question is a different process.

Extra work is useful when it addresses a particular uncertainty. In our example, the system might retrieve the exception policy, compare the conditions, and check whether the retention period applies to the right object. Rephrasing the same sentence indefinitely would accomplish little: repetition cannot conjure up a missing source. More computation and more available information are different resources, even though a system often uses them together.

Nor is improvement guaranteed when the necessary information is available. Several attempts can repeat the same mistake. A checking model can prefer a persuasive but incorrect answer. Extra steps can even turn a correct solution into a wrong one. Research on allocating test-time compute shows that the value of a strategy depends, among other things, on task difficulty and how candidates are assessed. The budget and the evaluation procedure must therefore be chosen together; more computation is not automatically better. [\[10\]](https://arxiv.org/abs/2408.03314)

This gives us a more precise way to discuss “reasoning models.” From an engineering perspective, the important questions are observable: how the result was obtained, which intermediate actions were permitted, how many attempts were made, how much time and computation were spent, and how the final answer was checked. The length of an intermediate explanation is not itself a measure of reasoning depth or correctness. An explanation can be a useful output, but its substance must be evaluated too.

**Configurable reasoning effort** lets the calling program request a level of effort. **Adaptive thinking** means that the model or a controller decides whether a particular request needs additional reasoning and how much to use. The two can work together: the effort setting guides behavior while the actual work varies with the request. An interface’s effort setting sometimes requests a particular operating mode. Unless the interface explicitly specifies otherwise, it should not be interpreted as an exact number of operations or seconds. Comparing runs requires recording the actual conditions: output limits, number of attempts, available tools, and stopping criterion.

Module 15 will develop these ideas, including the difference between producing at least one correct candidate and being able to select that candidate. For now, we have another level of explanation: observed quality depends not only on the weights but also on how the system goes about solving the particular task.

## 10. When the document is an image

So far, we have assumed that the documents are already available as text. Now imagine a scanned policy. The retention period is in a table, a qualification appears beneath it, and a note in the margin marks an earlier version as superseded. Accepting the file is not enough. To answer correctly, the system must preserve the relationships among its contents.

A **modality** is a kind of data: text, images, audio, or video. A multimodal system works with more than one. The label alone, however, tells us little about the actual design. A system may accept images but produce only text. It may recognize speech without synthesizing it. It may generate images while lacking the document-analysis capabilities we need. Inputs and outputs should be described separately.

In one common image-processing design, a dedicated encoder converts the image into a sequence of vectors, and a connecting module brings those representations into a form usable by the language component. Other architectures combine modalities differently. Visual representations need not be ordinary words or identifiers from the text vocabulary. A unified interface does not require every kind of data to have the same internal format. Module 16 will examine concrete designs. [\[M16\]](/courses/modern-llms/module-16/)

Our assistant also has another option: a separate program can first recognize the text and then pass the result to a text-only model. Such a pipeline can be convenient, but introduces its own failure modes. Table extraction may lose the relationship between a row and a column. Recognition may confuse a digit or omit a note. The language model then receives information that has already been distorted. Answering confidently from that input cannot repair a loss that occurred upstream.

Direct image input does not remove the need for checks. A critical detail may be too small to resolve; splitting the page into pieces may change which relationships are available; higher resolution usually requires more resources. “Can the model read images?” is therefore only the beginning of the question. Which images, in which operating mode, at what accuracy, and at what cost?

Audio and video introduce time as well. Answering a question about a meeting recording may require not only recognizing words but assigning them to speakers and moments in the recording. Video may require connecting events across frames. Sampling, representation, and compression choices become part of the task conditions, not invisible implementation details.

Additional modalities do not erase the distinctions we have already established. There are still learned parameters, a current input, a processing procedure, and a surrounding system. Some information simply arrives in a form other than text. To understand the route from document to answer, we must trace where and how that information becomes a representation the model can use.

## 11. Building a system around the model

We can now bring back the parts of the task we temporarily set aside. An employee has a question, the organization has a collection of documents, and we have a trained model. A program must connect the three.

In a simple arrangement, the program retrieves relevant passages, adds them to the question and instructions, calls the model, and presents the answer with its sources. This combination of retrieval and generation is commonly called **retrieval-augmented generation**, or RAG. Retrieved information becomes part of the condition under which the answer is produced; using it need not require retraining the generator each time. The original RAG paper studied a specific trainable architecture. Engineering practice also uses the term more broadly for generation systems that retrieve external material. [\[11\]](https://arxiv.org/abs/2005.11401), [\[M17\]](/courses/modern-llms/module-17/)

Our assistant can fail at any point along this route. The current policy may be missing from the collection. Retrieval may return an obsolete version. Splitting a document into passages may separate a rule from its exception. Too many similar fragments may crowd the context. The model may attach the wrong source to its answer. To diagnose the failure, we need enough records to reconstruct what material actually reached the model and where necessary information was lost.

Here the distinction between improving a model and improving a system becomes particularly clear. A stronger model may be better at reconciling the conditions in retrieved passages. But if the current document is absent from the available collection, a desire to answer correctly will not reliably reconstruct its contents. Conversely, better retrieval can improve results substantially without changing the weights when access to the right information was the original bottleneck.

A single predetermined search is not always enough. The assistant might find a general rule, notice a reference to a separate directive, and request that directive next. Or it might obtain the experiment’s end date from another system and calculate the deadline. We then have a sequence of actions: choose the next step, call a tool, receive its result, update the context, and continue.

In this course, an **agent** is a system able to choose and execute actions in such a loop to accomplish a task. The program that coordinates model and tool calls is the **agent harness**. A model’s suggestion to “find the document” does not itself perform a search. The action occurs when the surrounding program recognizes the request, validates it, and invokes the appropriate tool. ReAct is one research example of organizing the interaction between reasoning and action; Module 18 will examine the agent loop in detail. [\[12\]](https://arxiv.org/abs/2210.03629), [\[M18\]](/courses/modern-llms/module-18/)

Permissions belong to the execution system. An assistant intended only to read documents has no reason to receive permission to modify them. When an action affects other people’s data or has irreversible consequences, access controls and required approvals must enforce the limits. A sentence in a prompt is not a security boundary. Instructions can guide behavior; they cannot replace technical restrictions on authority.

The same distinction applies to retrieved content. An external document may contain text that resembles an instruction to the assistant. But the document was supplied as material to analyze, not as a source of new permissions. Keeping data separate from control instructions becomes especially important once the system can act. Module 18 will examine the resulting risks and ways to limit potential harm.

Finally, the system determines when to stop and how to check the outcome. “The model has finished generating” and “the task is complete” are different events. In our example, we still need to establish that the answer concerns the right object, respects the applicable conditions, and is supported by its source. With two invented sentences, we can check this by hand. An unrestricted document collection needs an evaluation procedure of its own.

A single message in the interface may therefore be the product of one model call or a long sequence of searches, calculations, and checks. The quality we observe belongs to that particular arrangement of work. Naming the model alone does not describe it adequately.

## 12. From understanding to choosing

We began with an employee’s question. We can now separate the problem into parts: the language model’s capabilities, the availability of documents, retrieval quality, context length, generation settings, and answer verification. This is the point at which choosing a model for the task becomes meaningful.

Start with an acceptance criterion. In our teaching example, an answer is accepted if it identifies the correct object and retention period, preserves the event that starts the clock, and cites a supporting document. When the available material contains no answer, an accurate statement that the information is missing may also count as success—provided that this is what the task calls for. We are evaluating whether the task was handled correctly, not whether every response contains a confident assertion.

A useful comparison needs more than easy questions. It should include similar-looking objects, exceptions, conflicting document versions, and missing information. The composition of this set determines what the evaluation means. Testing only the extraction of a single number tells us nothing about reconciling several rules. Questions too close to the examples used to tune the system may produce an overly optimistic estimate.

To measure differences between models, keep the surrounding conditions fixed wherever possible: the same document collection, retrieval procedure, questions, comparable limits, and acceptance checks. Changing both the model and its harness is a valid experiment too, but its result compares two complete systems. Module 19 will develop the design of these evaluations. [\[M19\]](/courses/modern-llms/module-19/)

Consider the following invented results for two candidates. Every number is for illustration; none is a measurement of a real model or a current price. Each candidate receives the same 200 questions in the same harness, with one attempt per question.

| Measure                                | Candidate A | Candidate B |
|----------------------------------------|-------------|-------------|
| Questions evaluated                    | 200         | 200         |
| Accepted outcomes                      | 100         | 180         |
| Acceptance rate                        | 50%         | 90%         |
| Total evaluation cost, arbitrary units | 40          | 60          |
| Cost per question                      | 0.20        | 0.30        |
| Cost per accepted outcome              | 0.40        | ≈ 0.33      |

Candidate B costs more per question but less per accepted outcome. No unusual pricing rule is involved. We have simply changed the unit of comparison: what we need is a successfully completed task, not merely a model call.

In general, the measure is:

$$
C_{\mathrm{accepted}}=\frac{C_{\mathrm{total}}}{n_{\mathrm{accepted}}}
$$

The numerator includes every expense within our chosen accounting boundary: model calls, retrieval, tools, verification, and retries where the procedure allows them. The denominator counts accepted outcomes, not all generated answers. If no outcome is accepted, there is no finite observed cost per accepted outcome. Report that there were no successes; do not report a cost of zero.

This measure does not remove the need for a quality requirement. Suppose our teaching pilot sets a target acceptance rate of at least 85%, and both candidates meet the latency limit. On the evidence of this run, we choose B for further testing: A’s success rate is too low, while B has both a higher success rate and a lower cost per accepted outcome. This is not yet a deployment decision. We must still examine performance on other questions, uncertainty in the estimate, and the severity of the remaining errors.

Retries deserve particular caution. We cannot automatically assume that a failed answer will be repaired by an independent second attempt. The same missing document or misunderstood condition may cause the error again. The cost of a reliable system is better estimated from complete observed runs than from a promise to “try once more.”

We can now collect our understanding into two concise records: a **model profile** and a **system profile**. The first identifies the trained artifact: its version, architecture, input and output formats, documented limits, weight availability, and training information. The second records how we use it: available documents and tools, prompt construction, permissions, and the procedures for measuring quality, latency, and cost. Appendix E provides templates for both.

Not every field carries the same kind of evidence. A matrix dimension can be read from a configuration and checked against code. A performance claim can be found in the developer’s report. Performance on our documents must be measured under specified conditions. If a closed model’s architecture has not been disclosed, the correct entry is “not published,” not a guess based on the product name. Observing performance does not reveal an otherwise unknown architectural detail.

Access also needs a concrete description. Being able to call an API, download weights, inspect code, and reproduce training are different properties. Available weights do not imply access to all the data and procedures used to obtain them. Usage conditions must be checked for the particular release and its components; the broad label “open model” is no substitute for that reading.

Saved model weights are commonly called a **checkpoint**. The configuration records architectural dimensions and settings. The code defines the operations. A **model card** adds information about intended use, training, limitations, and evaluation. None of these sources replaces the others. A repository’s current branch can change, so a reproducible analysis should identify an exact revision, not just a model family.

The profiles are now the result of an argument rather than a collection of unexplained fields. We know why parameters, context, training, and execution settings matter. We also know why a table of specifications cannot replace an experiment on the task we actually need to solve.

## 13. What we will build

Understanding existing models is valuable, but descriptions alone can take us only so far. Parts of the mechanism remain abstract until we can inspect and change them ourselves. The course therefore connects its explanations to an implementation of our own: **MiniLLM**, a small trainable language model in which we can follow the entire path from tokens to parameter updates and generation.

We will approach it in several ways. A small calculation will first make an equation tangible. A notebook will then let us vary the input and check array shapes, units, and boundary cases. Reading open code will show how an existing model implements the same mechanism. Finally, the continuing project will bring the pieces together into a working whole. These activities complement one another: a short example isolates an idea; a complete project reveals the agreements between components that make it usable.

MiniLLM is not a promise to reproduce a large commercial assistant on a home computer. Its purpose is to make the computation visible and open to experimentation. We will be able to count parameters, train a manageable configuration, save weights, change an architectural detail, and investigate the consequences. The quality we obtain will depend on data and training budget, not merely on assembling familiar blocks.

The practical work offers dense and MoE variants in S, M, and L sizes. The small configurations support short experiments; larger ones allow more substantial investigations and resource estimates. The L profiles are optional and require a separate budget. The particularly large L-MoE profile is especially useful for resource planning. Exact dimensions are collected in Appendix D, where they are easy to find without interrupting our first encounter with the project.

The core final project has a clear scope: an autoregressive text model. Multimodality, document retrieval, and agents are separate topics and possible extensions, not a promise to fit everything into the required MiniLLM implementation. Distributed training, specialized compute kernels, and a fault-tolerant production platform also require preparation of their own. We will learn to understand the associated constraints without presenting a teaching implementation as a production-ready system.

The course sequence can now be read as a route from data representation to a complete working model. Module 2 covers tokens and embeddings; Module 3 assembles a basic Transformer. Modules 4–8 explore positional information, attention, block internals, experts, and alternative ways to retain information about the past. With those pieces in place, Module 9 can give a substantive account of data, optimization, and the training budget.

Modules 10–12 return to the use of a trained model: context-state memory, faster generation, quantization, and an alternative order for constructing text. Modules 13–15 study useful behavior through demonstrations, feedback, and computation at problem-solving time. Modules 16–18 widen the scope of interaction to images, audio and video, external documents, tools, and actions. Module 19 brings these concerns together in an evaluation procedure.

The supplementary Module 20 takes the reverse journey, joining the components into a complete mathematical construction. Here, we are learning to recognize the main objects and ask the right questions. There, we will be able to follow every essential transformation: from a token identifier through the network to probabilities, then to a loss, a gradient, a weight update, and the next generation step. [\[M20\]](/courses/modern-llms/module-20/)

A new model name or an unfamiliar abbreviation in a technical report will then cease to be an isolated fact to memorize. We will ask what part of a familiar computation has changed, which problem the change addresses, and what trade-off it introduces. That way of reading remains useful even after the products and their positions in comparison tables have changed.

## 14. What to take forward

We began with a short answer drawn from a document and gradually uncovered several distinct objects. A language model computes a distribution over continuations. Training shapes the parameters of that computation. Context supplies the information and conditions of the current task. A generation algorithm turns distributions into a sequence of tokens. The surrounding program provides material, manages actions, and determines how the result will be checked.

These distinctions give us a way to investigate failures rather than merely record them. When an answer is wrong, we can ask whether the necessary information was available, whether it was represented faithfully, whether the model related it correctly, and whether verification caught the error. When an answer is too expensive or too slow, we can examine weight storage, input processing, context state, and the budget for further actions separately. When a new model looks attractive on a specification sheet, we can ask which of those properties actually matter to our task.

Choosing a model therefore begins with understanding and ends with testing. Architecture and training history help us form a hypothesis about a candidate’s suitability. Measurement under specified conditions shows how well that hypothesis holds up. A profile records both the object being compared and the conditions of the experiment, so that a successful result does not dissolve into the vague claim that “the model works well.”

For the document assistant, we can already state the task, distinguish a correct number from a supported answer, and explain how retrieval, context, and verification affect the outcome. Yet one of our earliest practical questions remains open: how much text will actually fit into the chosen model’s input?

Counting words or characters will not settle it. We need to understand exactly how a string becomes a sequence of identifiers and then the vectors on which the network operates. That is where the next module begins.

Next[Module 2. Tokenization and Embeddings →](/courses/modern-llms/module-02/)

From human-readable text to the model’s input.

------------------------------------------------------------------------

The appendices can be read independently as the need arises. They collect the calculations, configurations, reference material, and exercises that develop the chapter’s ideas in greater technical detail.

## Appendix A. Probability, training, and generation

A closer look at §§2–5. This appendix provides a first mathematical reading; Modules 3, 9, and 20 connect the notation to the full computation.

### A.1. From token probabilities to a sequence probability

For a sequence $x_{1:T}$, the autoregressive factorization is:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

For the first token, the condition may contain a special start marker or a supplied prefix. When we model only the answer to a request, the available request also conditions every factor. The principle is unchanged: the probability of a sequence is expressed through the probabilities of its successive continuations.

This equation describes the model’s distribution. A generation algorithm can use it in different ways. Sampling directly from the original conditional distributions produces a sample from the corresponding autoregressive distribution. Greedily selecting the most probable token, changing the temperature, or restricting the candidate set defines a different procedure for producing text. We therefore cannot equate a sequence’s probability under the model with its frequency under an arbitrary decoding procedure.

### A.2. The training objective

For $T$ predictions included in the objective, the standard mean negative log-likelihood is:

$$
\mathcal{L}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

We use natural logarithms here. Positions that should not contribute to training are excluded by a mask, and the normalization uses the number of included tokens. In SFT, for example, the masking convention determines which portions of a conversation contribute to the loss.

If the observed token receives probability 0.5, its contribution is −ln 0.5 ≈ 0.693. Raising the probability to 0.9 reduces that contribution to −ln 0.9 ≈ 0.105. Minimizing this loss encourages the model to assign more probability to observed continuations. It does not install a separate, universal measure of the factual truth of every possible statement. The content and structure of the training data remain essential.

The simplest gradient-descent update is:

$$
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}(\theta)
$$

Here $η$ is a positive learning rate. The equation illustrates parameter updating, not the full range of optimizers used in the course; Module 9 develops more substantial procedures. Ordinary generation does not perform this update. It computes a distribution, selects a token, and changes the input to the next step. [\[M3\]](/courses/modern-llms/module-03/), [\[M9\]](/courses/modern-llms/module-09/), [\[M20\]](/courses/modern-llms/module-20/)

## Appendix B. First estimates of memory and compute

A closer look at §§7–9. Always state what an estimate counts: weights, matrix operations, or context state. Adding these estimates together still does not replace measuring a running system.

### B.1. Memory for weights

Let $N_{total}$ be the total parameter count and $b$ the effective number of storage bits per parameter. Then:

$$
\operatorname{Mem}_{\mathrm{weights}}\approx N_{\mathrm{total}}\frac{b}{8}\ \text{bytes}
$$

Uniform BF16 storage gives $b$ = 16, or two bytes per weight. With quantization, the effective storage cost may include group scales and other required information. When different parts of a model use different formats, sum their storage requirements separately rather than assigning one nominal bit width to the entire checkpoint.

For 8 × 10⁹ BF16 parameters, the result is 16 × 10⁹ bytes = 16 GB ≈ 14.90 GiB. An idealized four bits for every parameter would give 4 GB ≈ 3.73 GiB. That second number excludes extra storage and mixed formats; it is not a promise about the size of a real file or the minimum memory needed to run the model.

Training needs additional memory for gradients, optimizer state, and saved intermediate results. The total bytes per parameter in a training run depend on numerical precision, the optimizer, how state is distributed, and other choices. The weight-file estimate must not be presented as a training-memory requirement.

### B.2. Reading the 2N approximation

A matrix–vector multiplication requires roughly one multiplication and one addition per weight used. Under our convention, these count as two floating-point operations. A common estimate for the weight-dependent matrix work is therefore:

$$
\operatorname{FLOP/token}\approx2N_{\mathrm{active}}
$$

The quantity $N_{active}$ must match the stated computational convention. The formula does not include all attention interactions with the context, routing, activation functions, token selection, data movement, or other overhead. Counting a bias as an “active parameter” also does not turn it into a matrix multiply. This is an order-of-magnitude estimate, not an exact instruction count.

FLOP measures an amount of work; FLOP/s measures a rate. Even exact knowledge of the first would not determine the second. On real hardware, reading weights and cached state can be a major bottleneck; MoE may also require communication between devices. The ratio of total to active parameters is consequently not a guaranteed speedup factor.

For a dense model trained on $D$ tokens, another familiar approximation is:

$$
C_{\mathrm{train}}\approx6ND
$$

Here $N$ counts the weights included in the chosen model of matrix computation. The factor of 6 approximates the forward and backward passes. Long contexts, different architectures, and additional work require corrections. For MoE, we cannot simply insert the total expert parameter count as though every expert processed every token. Module 9 discusses the derivation and limits of this approximation. [\[M9\]](/courses/modern-llms/module-09/)

### B.3. Memory added by the context

Ordinary KV caching retains two sets of vectors: keys and values. Let $L$ be the number of identical cached layers, $H_{kv}$ the number of key/value heads, $d_{h}$ the dimension of each head, and $b_{KV}$ the number of **bytes** per element. Each retained token then requires:

$$
M_{\mathrm{KV/token}}=2LH_{\mathrm{kv}}d_hb_{\mathrm{KV}}
$$

The factor of 2 accounts for both keys and values. Notice the units: $b$ in the weight estimate was measured in bits, while $b_{KV}$ here is measured in bytes. For one request retaining $T$ positions with full attention in every layer, multiply the result by $T$.

Use the dimensions revisited in Modules 5 and 10: 36 layers, 8 KV heads, 64 components per head, and BF16 storage at two bytes per component. The calculation gives 2 × 36 × 8 × 64 × 2 = 73,728 bytes = 72 KiB per token. Retaining 131,072 tokens then requires exactly 9 GiB for one request.

Now suppose only 18 layers retain every position, while the other 18 retain a window of the most recent 128 positions. An implementation that actually limits the local cache to this window gives:

$$
\frac{2\cdot8\cdot64\cdot2\cdot(18\cdot131072+18\cdot128)}{2^{30}}=4.50439453125\ \text{GiB}
$$

The estimate is approximately 4.5044 GiB rather than 9 GiB. We changed neither the head dimension nor the number of layers, but the number of retained positions in some layers. A different storage implementation may use a different amount of memory. The calculation excludes bookkeeping structures, temporary buffers, and allocation overhead.

Several independent requests generally require state to be counted for each one. Sharing an identical prefix is possible with suitable runtime support, but should not be assumed for every system. Mixed architectures, compressed representations, and recurrent blocks also need their own accounting. [\[M5\]](/courses/modern-llms/module-05/), [\[M10\]](/courses/modern-llms/module-10/)

## Appendix C. An open artifact we can inspect

Reference material for a closer inspection. We use gpt-oss-120b and Llama 4 Scout as concrete examples connected to later modules, not as recommendations or claims about current leaders.

### C.1. What the gpt-oss-120b configuration tells us

An open configuration lets us replace broad descriptions with exact dimensions. The gpt-oss-120b variant examined here has the following relevant fields:

| Component or setting | Value | Explained in |
|----|----|----|
| Vocabulary size | 201,088 | M2 |
| Number of layers; representation width | 36; 2,880 | M3, M6 |
| Query heads / KV heads; head dimension | 64 / 8; 64 | M5 |
| Experts per layer / selected per token | 128 / 4 | M7 |
| Full-attention and local-attention layers | 18 and 18, alternating | M5, M10 |
| Local window size | 128 | M5 |
| Maximum positional length in the configuration | 131,072 | M4 |

The values come from `config.json`. To identify a particular state of the file, we use revision `8c0580383cb1e6a9157669336ade6797a024cd9a`. At this stage, the mechanism names are signposts, not concepts you are expected to master on a first reading. [\[14\]](https://huggingface.co/openai/gpt-oss-120b/blob/8c0580383cb1e6a9157669336ade6797a024cd9a/config.json)

The configuration is still not a complete description of the computation. The precise activation function, the presence of biases, and trainable attention-normalization components, for example, must be checked against the implementation. A positional limit also does not reveal the full training history across sequence lengths. Dimensions tell us the model’s geometry, not its entire biography.

### C.2. Total and active parameter counts

Module 7 uses the following component-by-component count for gpt-oss-120b:

| Component                                     | Parameters          |
|-----------------------------------------------|---------------------|
| Two vocabulary matrices                       | 1,158,266,880       |
| Attention across all layers, including biases | 955,802,880         |
| Complete expert bank, including biases        | 114,701,598,720     |
| Routers                                       | 13,275,648          |
| Normalization parameters                      | 210,240             |
| Trainable attention-sink parameters           | 2,304               |
| **Total**                                     | **116,829,156,672** |

Under the course’s computational convention, the active path includes the full output vocabulary projection, shared block parameters, and four of the 128 experts in each layer. It does not include the entire input embedding table: selecting one row is not treated as multiplying by the whole table. The result is **5,132,849,472** active parameters. These values are consistent with the rounded description of 117B total and 5.1B active. [\[M7\]](/courses/modern-llms/module-07/), [\[13\]](https://openai.com/index/introducing-gpt-oss/)

We can check the active total directly from the table. Subtract the complete expert bank and one input vocabulary matrix from the total, then add back 4/128 of the expert bank. This verifies the arithmetic of the chosen convention. It does not replace reading the architecture or imply that every parameter contributes equally to runtime.

### C.3. Weight formats and actual storage

The gpt-oss release uses low-bit MXFP4 expert weights; other components may use different formats. For a teaching estimate of grouped storage, take four bits per value plus an eight-bit scale for each group of 32 values. The effective cost is 4 + 8/32 = 4.25 bits per weight. [\[13\]](https://openai.com/index/introducing-gpt-oss/), [\[14\]](https://huggingface.co/openai/gpt-oss-120b/blob/8c0580383cb1e6a9157669336ade6797a024cd9a/config.json)

If all of the rounded 117 billion parameters were stored this way, the result would be 117 × 10⁹ × 4.25 / 8 = 62.15625 GB ≈ 57.89 GiB. “If all” is essential: this is an idealized calculation, not the size of a particular checkpoint or the full memory needed to run it. A real release must be counted tensor by tensor and format by format. A running model additionally needs context state and working buffers.

### C.4. Similar sizes, different systems

| Property | gpt-oss-120b | Llama 4 Scout |
|----|----|----|
| Published total / active parameters | 117B / 5.1B | 109B / 17B |
| Documented input and output | Text → text | Text and images → text |
| Advertised context length | 131,072 tokens | 10M tokens in the specified operating mode |
| What our task most immediately requires us to check | Generation settings, document representation, memory | Image-reading quality, long-context behavior, resource cost |

The table records properties of particular published artifacts, not results from our own quality comparison. Context lengths are developer claims; we have not measured usefulness at those lengths here. Nor do these limits guarantee that a complete system will fit on a particular device. [\[13\]](https://openai.com/index/introducing-gpt-oss/), [\[14\]](https://huggingface.co/openai/gpt-oss-120b/blob/8c0580383cb1e6a9157669336ade6797a024cd9a/config.json), [\[15\]](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

Even this brief comparison shows why total parameter count is an inadequate basis for choosing. Text documents and scanned pages impose different requirements; active computation and storage differ too. A decision requires measurements under a shared task specification, not a conclusion drawn from one large number.

## Appendix D. MiniLLM sizes and the course map

Reference material for §13, including the dimensions of all six MiniLLM teaching profiles.

### D.1. Dense and MoE profiles

| Profile | Layers × width | Vocabulary | Context | Dense, M | MoE total, M | MoE active, M |
|---------|----------------|------------|---------|----------|--------------|---------------|
| S       | 6 × 256        | 8,192      | 512     | 6.33     | 20.36        | 6.59          |
| M       | 8 × 512        | 16,384     | 1,024   | 30.94    | 174.88       | 30.37         |
| L       | 20 × 2,048     | 32,768     | 2,048   | 1,020.35 | 10,020.54    | 1,055.21      |

Each dense/MoE pair shares the same external dimensions: depth, width, vocabulary, and configured context length. Parameter counts are shown in millions, rounded to two decimal places. The practical work’s convention counts shared matrices, including vocabulary matrices, in full; among the selectable experts, it counts only those used along one token’s path. For a dense model, this convention gives the total parameter count.

This convention **differs** from the gpt-oss computational convention in Appendix C, where the full input embedding table is excluded from the active path. Different definitions are acceptable when stated explicitly. Comparing their results as though they measured exactly the same quantity is not.

Profile S is suitable for short end-to-end checks; M supports more substantial teaching experiments. L takes the dense model to roughly one billion parameters and the MoE variant to roughly ten billion total parameters. These sizes do not commit you to training the largest configurations. Required exercises also use smaller variants; the large profile defines an optional research scale and its own budget.

### D.2. Where each question leads

| Question raised in Module 1 | Continue in |
|----|----|
| How does a string become a sequence and then vectors? | [M2. Tokenization and Embeddings](/courses/modern-llms/module-02/) |
| How do vectors become probabilities? | [M3. The Basic Transformer](/courses/modern-llms/module-03/) |
| How is order represented, and how do the internal blocks work? | [M4. Positions](/courses/modern-llms/module-04/), [M5. Attention](/courses/modern-llms/module-05/), [M6. Block Anatomy](/courses/modern-llms/module-06/) |
| How can a token’s computational path change? | [M7. MoE](/courses/modern-llms/module-07/), [M8. SSMs and Hybrids](/courses/modern-llms/module-08/) |
| How are weights learned, and how should training compute be allocated? | [M9. Pretraining](/courses/modern-llms/module-09/) |
| What determines memory use and the cost of inference? | [M10. KV Caches](/courses/modern-llms/module-10/), [M11. Speculative Decoding and Quantization](/courses/modern-llms/module-11/) |
| Must text be generated from left to right? | [M12. Diffusion LLMs](/courses/modern-llms/module-12/) |
| How can we make model behavior more useful? | [M13. SFT](/courses/modern-llms/module-13/), [M14. Feedback](/courses/modern-llms/module-14/), [M15. Test-Time Compute](/courses/modern-llms/module-15/) |
| How can models work with other data and an external environment? | [M16. Multimodality](/courses/modern-llms/module-16/), [M17. RAG](/courses/modern-llms/module-17/), [M18. Agents](/courses/modern-llms/module-18/) |
| What evidence makes a system suitable for its task? | [M19. Evaluating Models and Systems](/courses/modern-llms/module-19/) |
| How do the mathematical pieces fit together? | [M20. The Complete Construction](/courses/modern-llms/module-20/) |

## Appendix E. Two profiles and a way to read new releases

A practical continuation of §12. A profile is a concise record of questions we understand, not a substitute for explanation or measurement.

### E.1. The model profile

| Field | What to record |
|----|----|
| Identity | Exact artifact name, version or revision, and date checked |
| Architecture and parameters | Design, total/active counts, and counting convention; “not published” for undisclosed details |
| Training | Documented stages and data; where the published information ends |
| Inputs and outputs | Modalities, tokenizer, message format, and required components |
| Context and generation | Documented input, total-sequence, and output limits; supported operating modes |
| Artifacts and access | Weights, configuration, code, model card, and usage conditions |
| Evidence | Sources for specifications and evaluation references, including the conditions of those evaluations |

### E.2. The system profile

| Field | What to record |
|----|----|
| Model and runtime | Exact version, weight format, execution environment, and generation settings |
| Available information | Document collection and version, retrieval, image processing, and external memory |
| Input construction | Instructions, context assembly, and history truncation or summarization |
| Actions and authority | Tools, permissions, network access, isolation, and required approvals |
| Problem-solving budget | Time, output, call-count, and retry limits |
| Verification | Task set, acceptance criteria, evaluation procedure, and error severity |
| Observed results | Quality, uncertainty, latency, cost, and causes of failure |

These tables have no universally required length. A small experiment may need only a brief record; a reproducible comparison will require more detail. What matters is not the number of completed cells, but whether another reader can tell what was investigated and what the result means.

### E.3. Reading an announcement without overextending its claims

A new release may involve several distinct events: publication of a description, access to a service, availability of weights, and release of training code or data. One does not establish that the others have occurred. Similarly, an API model identifier, a checkpoint version, and the name of a user-facing mode may refer to different levels of a product.

Read any reported benchmark result alongside its protocol. Which model variant was tested? What tools could it use? How many attempts were allowed? How was computation allocated, and who or what checked the answer? Missing conditions limit the conclusions we can draw. It is better to preserve an unknown field explicitly than to fill it with a persuasive reconstruction unsupported by evidence.

The field’s history is most useful as a history of constraints being addressed: sequence processing, training scale, data quality, useful behavior, generation-time memory, access to external information, and compute allocation per request. These directions coexist. They do not form a single ladder on which a new name automatically makes an older architecture or engineering technique obsolete.

Dated release catalogs and price lists can be maintained separately. Without a specified task, a ranking of “today’s best models” does not replace understanding, and rapidly changing names should not determine the structure of an introduction. The artifacts in Appendix C remain inspectable teaching examples after newer releases appear.

## Sources and connections to the course

References in the chapter point to the papers and modules supporting specific distinctions. The two retention policies, the context-budget example, and the A/B comparison are invented teaching examples. MiniLLM and the exact gpt-oss count use the numerical conventions adopted in this course. The chapter makes no claims about current market prices or model rankings.

### Research and open artifacts

**\[1\]** Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017. The original Transformer architecture: attention and parallel sequence processing.

**\[2\]** Brown et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020. Specifying tasks through in-context examples without weight updates.

**\[3\]** Hoffmann et al. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022. Choosing model size and data volume together under a limited compute budget.

**\[4\]** Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022. Demonstrations, preferences, and RLHF.

**\[5\]** Rafailov et al. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023. Direct training on preferences.

**\[6\]** Fedus, Zoph, Shazeer. [Switch Transformers](https://arxiv.org/abs/2101.03961), 2021. Sparse expert activation and its constraints.

**\[7\]** Gu, Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), 2023. Input-dependent state updates.

**\[8\]** Nie et al. [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992), 2025. LLaDA, masking, and token reconstruction.

**\[9\]** Liu et al. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), 2023. Testing the use of information within long contexts.

**\[10\]** Snell et al. [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314), 2024. Allocating computation during problem solving.

**\[11\]** Lewis et al. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020. Combining a parametric model with external retrieval.

**\[12\]** Yao et al. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022. Loops connecting model reasoning, actions, and observations.

**\[13\]** OpenAI. [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/), August 5, 2025. The published models and their specifications; used as evidence about a particular release, not as a current ranking.

**\[14\]** OpenAI. [gpt-oss-120b: config.json](https://huggingface.co/openai/gpt-oss-120b/blob/8c0580383cb1e6a9157669336ade6797a024cd9a/config.json), pinned revision `8c0580383cb1e6a9157669336ade6797a024cd9a`. Architectural dimensions and release formats.

**\[15\]** Meta. [Llama 4 Scout: official model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct). Reported parameter counts, modalities, and context length.

### Course material

**\[M3\]** [M3. The Basic Transformer](/courses/modern-llms/module-03/): forward computation, masks, training, and generation.

**\[M5\]** [M5. Modern Attention Mechanisms](/courses/modern-llms/module-05/): GQA, KV geometry, and mixed attention schemes.

**\[M7\]** [M7. Mixture of Experts](/courses/modern-llms/module-07/): component-level total and active counts and their conventions.

**\[M9\]** [M9. Pretraining](/courses/modern-llms/module-09/): data, optimization, and compute budgets.

**\[M10\]** [M10. KV Caches and Efficient Inference](/courses/modern-llms/module-10/): state, prefill, decode, and memory.

**\[M16\]** [M16. Multimodal LLMs](/courses/modern-llms/module-16/): representations and ways of connecting modalities.

**\[M17\]** [M17. Embeddings, Retrieval, and RAG](/courses/modern-llms/module-17/): document access and use of sources.

**\[M18\]** [M18. Agents and Tool Use](/courses/modern-llms/module-18/): execution, state, and authority.

**\[M19\]** [M19. Evaluating Modern LLMs](/courses/modern-llms/module-19/): protocols, task-specific evaluation, and engineering decisions.

**\[M20\]** [M20. The Complete LLM](/courses/modern-llms/module-20/): distinguishing forward computation, training, and generation, then assembling MiniLLM.
