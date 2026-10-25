# Module 2. Tokenization and Embeddings

From Text to a Neural Network’s Input

Text seems a natural input to a language model—until we ask which numbers actually enter the network. This chapter follows a written string all the way to trainable vectors, showing how the representation shapes context use, learning, and computational cost.

Read the main chapter in order. Appendices A–E develop the alternatives, derivations, code, and real artifacts without interrupting that path. Basic familiarity with vectors and matrices is enough; notation and operations are introduced as they become useful.

## 1. How much room does text take?

Module 1 ended with a question that arises almost whenever an assistant works with documents: how much text will actually fit into the model’s context? At first, this sounds like a matter of looking up a limit and measuring the document. But the limit is usually expressed in tokens, while we see pages, words, and characters. There is no fixed exchange rate between the two.

Imagine that our assistant receives an instruction, a user’s question, and two passages from a document. To us, these are meaningful parts of a task. At the entrance to a text neural network, they begin as something much less familiar: a sequence of integers. The program that turns text into this sequence is called a **tokenizer**. Its output consists of **identifiers**, or **IDs**, for units in a particular vocabulary.

A token need not be a word. It might represent a common word, part of an unfamiliar word, a space attached to the following fragment, punctuation, or even part of a multibyte character. The units depend on the algorithm and its learned vocabulary. Two tokenizers can therefore represent the same string with sequences of different lengths while preserving exactly the same text.

We will use a small byte-level tokenizer whose operations we can inspect. The module retains Russian examples because the accompanying experiment trains its vocabulary on Russian text and then tests it on aligned Russian and English data. For one frozen tokenizer from the practical materials, the greeting `Привет` follows this path:

| Representation | Result |
|----|----|
| Text | `Привет` |
| Unicode code points | 6 numerical designations for letters |
| UTF-8 bytes | `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` — 12 bytes |
| IDs from the teaching BBPE after 4,096 merges | `[943, 370, 301]` — 3 tokens |
| Input vectors | Rows $E_{943}$, $E_{370}$, and $E_{301}$ of a trainable table |

1.  **A string and its bytes**

    Exact spelling represented in valid UTF-8.

2.  **Boundaries and merges**

    Rules define chunks; training defines the vocabulary.

3.  **A sequence of IDs**

    Stable addresses within a particular artifact.

4.  **Trainable vectors**

    Select rows of E: the network’s computation begins.

One path, two distinct tasks: construct a sequence, then learn how to use its elements.

Those three IDs belong to **one specific vocabulary**. They are not properties of the Russian word, and another tokenizer need not produce them. What matters is the sequence of operations. First, a program selects discrete units and their identifiers. Then the neural network assigns a vector—a collection of numerical coordinates—to each identifier. This vector is the token’s **embedding**.

We have uncovered two distinct questions. How should text be divided into a useful sequence? And how should the resulting units be represented inside a neural network? The tokenizer answers the first. The embedding table answers the second. In the usual training pipeline, the vocabulary is built and fixed before the main model is trained; the vectors are then learned alongside the network’s other parameters.

Sequence length has immediate practical consequences. Other things being equal, more tokens occupy more context positions. At the same price per input token, they also increase the input component of the bill proportionally. Computation time is less simple: it depends on the architecture, processing stage, and hardware. The same proportional rule cannot simply be applied to latency.

To understand where sequence length comes from, we should start before the vocabulary—with what a computer stores when we type a letter.

## 2. A letter, a code point, and a byte

On a screen, text appears to be a sequence of visible marks. A program deals with several layers beneath that appearance. **Unicode** assigns code points: numerical values for letters, signs, combining marks, and other elements. **UTF-8** specifies how the corresponding values are represented as bytes. A byte contains eight bits and can take a value from 0 to 255.

The ASCII letter `A` needs one byte. Each letter in `Привет` needs two. The rocket `🚀` needs four. Even a very short string can therefore have different code-point and byte counts.

| String | Code points | UTF-8 bytes | Hexadecimal representation |
|----|----|----|----|
| `A` | 1 | 1 | `41` |
| `Привет` | 6 | 12 | `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` |
| `é` | 1 | 2 | `C3 A9` |
| `e\u0301` — a letter followed by a combining accent | 2 | 3 | `65 CC 81` |
| `🚀` | 1 | 4 | `F0 9F 9A 80` |
| `ток` | 3 | 6 | `D1 82 D0 BE D0 BA` |

The `e\u0301` entry uses Python escape notation. The backslash sequence stands for a combining accent; it is not six literal characters in the underlying string. Once interpreted, the letter and accent are usually perceived as one visible unit. The relevant concept is a **grapheme cluster**. Thus even the everyday word “character” can refer to something other than a single code point. [\[1\]](https://www.unicode.org/reports/tr29/)

Why make these distinctions in a course about language models? Because a tokenizer receives a particular digital representation, not a picture of a word. Two strings that look alike can contain different bytes and receive different segmentations. Our teaching tokenizer preserves that difference rather than silently replacing one spelling representation with the other.

A transformation that first puts strings into a chosen form is called **normalization**. It can be useful: equivalent spellings may no longer need separate representations. But information can be lost. Once repeated spaces have been collapsed or one representation has been replaced by another, a decoder cannot infer which version appeared in the original document.

The claim that a tokenizer can recover text therefore needs a contract. For our BBPE, which neither normalizes the input nor discards bytes, the contract is

$$
\operatorname{decode}(\operatorname{encode}(s))=s.
$$

Here $s$ is a string that can be encoded as valid UTF-8. In particular, Python strings containing unpaired Unicode surrogates fall outside this contract: strict UTF-8 encoding rejects them. If a tokenizer applies normalization $N$, the recoverable result may instead be $N(s)$. If it also substitutes an unknown token for unsupported material, even some of the normalized text may be lost.

These are not merely defensive qualifications. An extra space may make little difference in prose, while indentation can change a program. Exact quotations and identifiers introduce similar requirements. Before relying on reversibility, we need to know which representation the system preserves.

Bytes give us a dependable starting point: a valid UTF-8 input can be reconstructed from them. It remains to ask whether each byte should also become a separate computational position for the language model.

## 3. Choosing the units of language

Words seem an obvious choice. Give each word an identifier and the model receives a fairly short sequence of familiar linguistic units. The difficulty arrives with a new name, an unusual inflected form, a typo, or a technical compound that the vocabulary has never seen.

There is no final list of every word that a system might encounter. A traditional solution was to replace unfamiliar words with a shared `<UNK>` token, meaning *unknown*. Distinct new words could then become indistinguishable inputs. Languages with rich inflection create a further problem: the vocabulary grows rapidly as many related forms require separate entries.

At the other extreme, we could use characters. An unfamiliar word would retain its composition instead of disappearing into `<UNK>`. Yet familiar long words would repeatedly consume many positions, and the network would have to learn how to assemble them into useful patterns. A restricted character alphabet would still leave a coverage problem outside that alphabet.

Bytes make coverage especially straightforward. There are just 256 possible byte values, and any valid UTF-8 text can be expressed without an unknown token. The cost is already visible in our greeting: `Привет` takes twelve positions, although a reader sees six letters and one word.

A useful compromise is to retain small base units while allowing frequent combinations to become larger tokens. A familiar fragment can then occupy one position; unfamiliar material can still be assembled from smaller pieces. We no longer have to choose between recognizing an entire word and losing its identity altogether.

**Byte Pair Encoding**, or **BPE**, constructs a vocabulary by repeatedly merging adjacent pairs of units. The idea originated in data compression and was later applied to subword representations in neural machine translation. In our version, the initial units are bytes, so we will call it **byte-level BPE**, or **BBPE**. [\[2\]](https://aclanthology.org/P16-1162/)

The algorithm’s criterion is statistical rather than linguistic. It does not explicitly search for stems, suffixes, or morpheme boundaries. It searches for frequent adjacent pairs under the constraints it has been given. Some resulting tokens will resemble meaningful word parts. Others will represent whitespace patterns. Still others will contain bytes that cannot be displayed on their own as a valid character. All are legitimate outcomes of the same mechanism.

BPE is not the only way to do this. Unigram segmentation uses a probabilistic criterion; different tokenizers choose different normalization and pre-tokenization rules; some models work from bytes without a fixed subword vocabulary. Appendix A places these alternatives in context. First, we will understand one algorithm well enough that the alternatives become differences in mechanism, rather than just names.

## 4. Boundaries chosen before learning

Before counting pairs, we need to decide where a merge is allowed. Given a document as one continuous byte stream, an algorithm could merge any adjacent bytes. A tokenizer designer may prefer to rule out some combinations in advance—for example, arbitrary joins across neighboring words.

**Pre-tokenization** divides the input into chunks. BPE operates within each chunk, and a chunk boundary becomes a boundary that a merge cannot cross. This is a separate decision from learning which pairs are frequent.

Our teaching implementation uses a regular expression. For three small inputs, its behavior is easy to inspect:

| Input          | Chunks, shown as Python strings |
|----------------|---------------------------------|
| `Hello world!` | `['Hello', ' world', '!']`      |
| `1234567`      | `['123', '456', '7']`           |
| `Привет`       | `['Привет']`                    |

The space before `world` belongs to the second chunk. No amount of evidence for a frequent pair can make BPE merge the end of `Hello` with the beginning of that chunk. Likewise, the number has already been divided into groups of at most three digits. A later merge can combine units within a group but cannot erase the chosen boundary.

A regular expression is a compact way to describe families of strings. We do not need to memorize its full syntax to understand its role here. The important point is that the input is constrained **before** pair frequencies are examined. Appendix C contains the executable pattern and explains its branches.

This is an example of an **inductive bias**: a restriction on the solutions the learning procedure can produce. Short digit groups make numerical segmentation more regular, but do not by themselves teach arithmetic. A boundary policy changes the representation of the task; the model must still learn to solve the task.

Pre-tokenization also explains why two equally large vocabularies may behave differently. They may have been learned from different data, but they may also start with different permissible boundaries. Looking only at vocabulary size misses a decision that shapes every subsequent merge.

With the boundaries fixed, each chunk can be converted to UTF-8 bytes. We now have everything needed to watch a vocabulary grow.

## 5. How BPE builds a vocabulary

Consider `ababab`. With one base unit per ASCII byte, the input has six positions. The pair `a, b` appears three times. If we introduce a new token `ab` and replace those non-overlapping occurrences, the sequence becomes `ab, ab, ab`: three positions instead of six.

That is one BPE training step. The next step counts adjacent pairs in the **updated** representation. Earlier merges therefore change the candidates and frequencies seen by later steps. The vocabulary is not a list of frequent substrings selected independently; it is the product of a sequence of decisions.

For a corpus, the pair selected at a step can be written as

$$
(a,b)^*=\operatorname*{arg\,max}_{(a,b)}\operatorname{count}(a,b).
$$

The count includes only neighboring units within allowed chunks. A pair may consist of two original bytes, a byte and a previously created token, or two larger tokens. Merging always means concatenating their underlying byte sequences.

Now take the small Russian corpus

```text
низко низко летали ласточки, ласточки летали низко
```

It contains 93 UTF-8 bytes. Applying our pre-tokenization and using the first encountered pair to break frequency ties produces the following first six merges:

| Step | Pair | Count | New byte fragment | Positions remaining |
|----|----|----|----|----|
| 1 | `D0` + `B8` | 7 | `D0 B8` — the letter `и` | 86 |
| 2 | space + `D0` | 6 | `20 D0` — a space and the start of a letter | 80 |
| 3 | `D0` + `BA` | 5 | `D0 BA` — the letter `к` | 75 |
| 4 | `D0` + `BE` | 5 | `D0 BE` — the letter `о` | 70 |
| 5 | token `20 D0` + `BB` | 4 | `20 D0 BB` — the fragment `' л'` | 66 |
| 6 | token `20 D0 BB` + `D0` | 4 | `20 D0 BB D0` — the start of a longer fragment | 62 |

The second step already rules out a tempting story: BBPE does not necessarily climb a ladder from bytes to letters to syllables to words. `20 D0` contains a space and only the first byte of a Cyrillic letter. It is not a complete UTF-8 string on its own, but it is perfectly usable inside a longer byte stream. The frequency criterion has no obligation to respect the visible letter boundary.

After six steps, 93 initial positions have become 62. New tokens are added to the vocabulary; earlier ones remain available. A frequent long fragment can be used as a unit, while unfamiliar text can still fall back to smaller units, all the way down to individual bytes.

Counting and replacing also require a careful distinction. In `aaa`, the pair `a, a` occurs twice, at positions 1–2 and 2–3. Both occurrences cannot be replaced in one pass, because the middle element would be used twice. Our implementation counts both adjacencies but merges from left to right without overlap. One step therefore yields `aa, a`, not overlapping tokens.

Training must eventually stop. We normally set a merge budget or target vocabulary size, although a tiny corpus can run out of eligible pairs earlier. As long as each executed step introduces one new ordinary ID, our vocabulary contains $256+N_{\text{merges}}$ ordinary units. Special dialogue tokens have not yet entered the picture.

What we have learned is not a grammar of Russian. It is an ordered record of decisions about which neighboring units should be combined. The next question is how that record is used on text that was absent from the training corpus.

## 6. How a learned vocabulary reads new text

It would be easy to count frequent pairs again in the new request and start merging them. But that would train a different tokenizer for each request. IDs would cease to have stable meanings, and a neural network could no longer rely on a fixed relationship between an identifier and its input representation.

During **encoding**, the merge list is fixed. Each rule has a **rank**: its position in the order established during training. The encoder pre-tokenizes the new string, converts its chunks to bytes, and repeatedly selects the currently available permitted pair with the lowest rank. It stops when no learned merge can be applied.

Rank answers “which learned rule has priority?”, not “which pair is most frequent in this request?”. Our format assigns new IDs consecutively from 256, so their order also determines merge priority. Other formats can store these roles separately.

The distinction is worth making concrete. Training examines a corpus to decide what to add to the vocabulary. Encoding examines an input to decide which **existing** rules apply. New observations during encoding do not revise those rules. That separation lets the tokenizer remain aligned with a fixed embedding table.

Decoding is simpler. Each ordinary ID denotes a byte sequence. The **token decoder** concatenates those sequences in order and then interprets the combined bytes as UTF-8. It needs neither corpus frequencies nor neural vectors nor a search for similar words.

The order—join the bytes first, interpret the text second—matters. A lone `D0` is an incomplete UTF-8 sequence. Follow it with `9F` and the two bytes represent `П`. A streaming decoder should hold an unfinished byte prefix until enough bytes arrive. Replacing each unfinished token with `�` may be convenient for display, but it destroys exact information.

For output produced by our encoder, recovery follows from construction: the original bytes were grouped, not changed. An arbitrary list of IDs is another matter. It may contain an undefined identifier or concatenate into invalid UTF-8. The implementation in Appendix C rejects these cases explicitly.

There is a subtler asymmetry too. `decode(encode(s)) == s` does not imply `encode(decode(ids)) == ids` for every ID sequence. The same bytes may be represented by one composite token or several smaller tokens. Both decode to the same text; re-encoding selects the segmentation prescribed by the frozen ranks. **Recovering the original text does not make every possible token representation unique.**

**Training**

Corpus → pair counts → a new rule. The vocabulary and corpus representation change.

**Encoding**

New string → learned ranks → IDs. The vocabulary stays fixed.

We now have integers that identify the input fragments precisely. We do not yet have a useful numerical geometry for a neural network. That is the next step.

## 7. An identifier is not an embedding

The number 943 has ordinary arithmetic properties: it is greater than 370, and the distance between them can be calculated. A tokenizer makes no promise that these properties tell us anything about the corresponding text. An ID is an address in a vocabulary. Nearby addresses need not refer to similar words, and a large identifier does not imply greater importance or more elaborate meaning.

A typical text model therefore does not treat its token ID as a scalar coordinate of meaning. Instead, it maintains an **embedding table**. If the table has $V$ rows and each vector has $d$ coordinates, it is a trainable matrix

$$
E\in\mathbb R^{V\times d}.
$$

An ID selects a row. The resulting vector can be added to other vectors, multiplied by matrices, and passed through nonlinear functions. Its coordinates are model parameters, free to change during training. This operation is called **embedding lookup**. It is direct indexing, not nearest-neighbor search.

Consider a table with three rows and two coordinates:

$$
E=\begin{pmatrix}1&0\\2&0\\0&1\end{pmatrix},\qquad
[0,2,0]\longmapsto
X=\begin{pmatrix}1&0\\0&1\\1&0\end{pmatrix}.
$$

The three input IDs produce three vectors. The first and third are identical because both select row 0. The identifier is not multiplied into the vector; in particular, ID 0 does not imply a zero embedding. It simply addresses the first row.

A sequence of length $T$ produces $X\in\mathbb R^{T\times d}$. If $B$ sequences of a common padded or otherwise standardized length are processed together, the result has shape $B\times T\times d$. This collection of examples is a **batch**. In Module 3, the width $d$ will be written $d_{model}$: it plays the same role as the number of features in the model’s input stream.

At lookup time, the same ID does indeed return the same initial vector. But its uses need not mean the same thing. The surrounding text might concern a bank account in one case and a river bank in another. Contextual differences must arise from computation over the available sequence, not from a vocabulary address spontaneously changing its meaning.

We should therefore keep two objects distinct: the **input embedding** $E_{i_t}$ and the **contextual state** at position $t$. The former is retrieved from a table. The latter is computed by the network from the input representations. Explaining the transition between them is the purpose of the next module.

An input token embedding is also different from a **text embedding used for retrieval**. Representing a sentence or document involves another computation, usually contextual processing and aggregation, and may use an objective designed specifically for comparing texts. The geometry of isolated input rows is not a substitute for such a retrieval model. Module 17 returns to this distinction.

The table also does not, by itself, say where a token occurs. Positional signals and restrictions on which other positions may be read belong to the subsequent computation. Keeping them separate now lets us understand lookup without prematurely unpacking the entire architecture.

## 8. How the network returns to tokens

In Module 1, we described a language model as a function that computes a distribution over the next token given a context. We can now see the input to that function: a sequence of vectors. We have not yet examined the internal processing, but we can already understand the interface at the other end.

Suppose the network has produced a final-position hidden state, written as a row $h\in\mathbb R^{1\times d}$. An **output projection**, with matrix $U\in\mathbb R^{d\times V}$, scores possible continuations:

$$
z=hU.
$$

The result contains $V$ values, one for each output vocabulary position. These values are called **logits**. They can be positive or negative and need not sum to one. **Softmax** turns them into probabilities:

$$
p_i=\frac{\exp(z_i-m)}{\sum_{j=0}^{V-1}\exp(z_j-m)},\qquad
m=\max_j z_j.
$$

Exponentiation makes the terms positive; division by their sum normalizes them. Subtracting the same maximum from every logit leaves the relative exponentials unchanged while helping prevent numerical overflow. We are considering a simple output without a bias or additional logit transformations. Our toy vocabulary is fully defined, so we can also leave invalid-ID policies aside for now.

The input and output matrices can be learned independently. Alternatively, we can use **weight tying**, setting $U=E^\top$. One set of parameters then serves two different operations. [\[3\]](https://arxiv.org/abs/1608.05859)

For our table and $h=[1,0]$, tied weights give

$$
z=[1,2,0],\qquad
p\approx[0.2447,\;0.6652,\;0.0900].
$$

Selecting the most probable ID would give 1, even though $h$ equals row $E_0$. The projection is therefore not identifying the row that was previously looked up. Lookup takes a known index and retrieves a vector. The output projection uses the hidden state and learned directions to score possible continuations. These are not inverse operations.

For an untied example, take

$$
U=\begin{pmatrix}0&1&-1\\2&0&1\end{pmatrix}.
$$

The same $h=[1,0]$ now produces $z=[0,1,-1]$. In this particular example, every logit has shifted by the same constant relative to the tied case, so softmax gives the same probabilities. That is a property of these chosen numbers, not a general equivalence between tied and untied weights. Other matrices or hidden states need not behave this way.

A probability distribution is not yet the emitted token. A generation rule must select an ID—for example, by choosing the maximum or sampling from a distribution. Once the ID has been selected, token decoding maps it to bytes and then text. That last operation is performed by the tokenizer, not by the matrix $U$.

It helps to keep the two return paths separate. The network predicts **which identifier comes next**. The token decoder reconstructs **which bytes that identifier denotes**. Calling both operations “decoding” is common, but they do different work. The mathematical output projection does not undo the input embedding lookup, and token decoding does not need an embedding table at all.

**Retrieve a representation**

Known ID → row of E → input vector.

**Predict a continuation**

Contextual state h → scores hU → probabilities → selected ID.

We have now connected both ends of the model. Text becomes IDs, IDs become vectors, contextual computation produces a hidden state, and that state yields scores for the next ID. What makes the learned vectors useful in the first place?

## 9. Where useful vectors come from

We chose our table entries by hand so that the arithmetic would be visible. A real model presents a more interesting question: why should the numbers in its table help predict language?

At the beginning of ordinary training from scratch, the table is initialized without knowing the relationships between particular words. Its usefulness emerges from the training objective. The model reads a context, assigns probabilities to possible next tokens, and incurs a larger loss when it assigns too little probability to the observed continuation. The resulting learning signal changes the parameters—including the input vectors—so that similar predictions can improve.

Embedding coordinates therefore do not arrive with compulsory labels such as “animate” or “past tense.” Information can be distributed across coordinates, and the representation works jointly with the rest of the network. Its geometry can be worth studying without expecting every arbitrary axis to have a standalone human interpretation.

One essential property of learning an embedding table can be seen without a Transformer. If an ID appears twice in an input, both uses refer to **the same row of parameters**. Their gradient contributions must be added. They are not independent copies that can be updated separately.

Appendix B demonstrates this with `[0,2,0]`. We give the table an artificial objective: move all selected vectors toward zero. With a learning rate of 0.1, row 0 changes from `[1,0]` to `[0.8,0]`, row 2 changes from `[0,1]` to `[0,0.9]`, and row 1 stays unchanged. The toy loss decreases from 1.5 to 1.045. Zero targets do not teach language; they let us isolate the accumulation of updates.

In an actual model, the signal travels through all the intervening layers. The output matrix also behaves differently from a lookup table. With a full softmax, every output direction participates in the normalization sum. Parameters associated with a token can therefore receive a gradient even when that token is not the correct continuation for this example. In our orientation of $U$, the output directions are its columns. When input and output weights are tied, both computational paths contribute to the same parameters.

This gives a more precise account of rare tokens. A rare input ID gets few direct observations in its input role. A rare target ID gets few positive prediction examples. Neither statement means its parameters are never updated, but learning a reliable representation may still be difficult. A large vocabulary is not enough: the model needs data from which to learn how to use its entries.

The tokenizer and the language model may also be trained at different stages on different mixtures of text. A fragment frequent enough to be selected during vocabulary construction need not remain frequent during LLM pretraining. Merge rank records a tokenizer decision; it is not a substitute for the model’s actual token-use statistics.

The trade-off is now clearer. Larger tokens can shorten the sequence, but additional vocabulary entries cost parameters and need training evidence. We can put numbers on the first part of that cost.

## 10. The price of a larger vocabulary

The input table $E$ contains $Vd$ parameters. An independent output matrix adds another $Vd$. Tying the weights leaves one unique parameter set. In BF16, each parameter occupies two bytes, so two independent tables require

$$
N_{\text{parameters}}=2Vd,\qquad
M_{\text{weights}}=4Vd\ \text{bytes}.
$$

Here $V$ is the number of **rows allocated by the model**, not necessarily the number of defined tokens. An ID range can contain gaps, and a model may allocate extra rows. Appendix E explains how to distinguish these quantities. For direct lookup, the table must at least be large enough to accommodate every valid index.

Consider two architectural examples and one hypothetical expansion:

| Scenario | Parameters in one table | Parameters in two independent tables | BF16 weight storage for both |
|----|----|----|----|
| gpt-oss-120b: $V=201088$, $d=2880$ | 579,133,440 | 1,158,266,880 | 2.1574 GiB |
| Llama-3.1-8B: $V=128256$, $d=4096$ | 525,336,576 | 1,050,673,152 | 1.9570 GiB |
| Increase $V$ from exactly 100,000 to 262,000 at $d=4096$ | +663,552,000 | +1,327,104,000 | +2.4719 GiB |

We use $1\ \mathrm{GiB}=2^{30}$ bytes. The first two rows illustrate particular architectures; they are not model recommendations. The last row is only the difference for a stipulated vocabulary expansion, not the configuration of Gemma or another family. The gpt-oss dimensions refer to the same open-model example used in Module 1. [\[4\]](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) [\[5\]](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)

The two vocabulary matrices in the Llama-3.1-8B example contain about 1.05 billion parameters. Compared with the rounded “8B” label, that is roughly 13%. Input and output representations are not necessarily a negligible service layer around the network. Their contribution can be substantial, particularly in smaller models.

Weight tying reduces parameter storage but does not remove the output calculation. A dense projection that scores all $V$ candidates still performs work of order $Vd$ per hidden state. Nor is weight storage the whole training-memory budget: gradients, optimizer states, activations, and temporary buffers require additional memory.

On the other side of the trade-off is sequence length. Larger fragments can reduce $T$, saving work and state storage later in the network. Yet vocabulary size alone does not tell us the size of that benefit. The entries and the text matter. A vocabulary well suited to one language or domain need not represent another equally efficiently.

There is no single “larger is better” control. We are balancing input length, table size, output computation, and the data required to train the entries. To discuss that balance concretely, we need to measure a tokenizer on text it did not use to build its vocabulary.

## 11. What one experiment can teach us

The practical materials train BBPE on Russian text from **Parallel Universal Dependencies**, or **PUD**, fixed to the UD 2.15 release. Its aligned Russian and English segments let us ask several distinct questions, instead of one vague question about which language “tokenizes better.” [\[6\]](https://github.com/UniversalDependencies/UD_Russian-PUD/tree/7284f9241af8755cbecdc6f6ce3eaf338f0503c4)

Training uses 833 Russian segments from 331 source documents. Another 167 aligned segments from 66 different documents form the evaluation set. Documents do not overlap between the two parts. We can first ask how well the vocabulary represents new Russian documents, then ask what it does with the English versions of those same evaluation segments.

Begin with **bytes per token**:

$$
\operatorname{bytes/token}=\frac{N_{\text{bytes}}}{N_{\text{tokens}}}.
$$

For a fixed text, a larger value means a shorter token sequence. With no merges, every byte is a token and the ratio is one.

| Merges | RU training: bytes/token | RU evaluation: bytes/token | EN evaluation: bytes/token |
|----|----|----|----|
| 0 | 1.000 | 1.000 | 1.000 |
| 64 | 2.236 | 2.244 | 1.009 |
| 256 | 3.231 | 3.213 | 1.011 |
| 1024 | 4.531 | 4.360 | 1.047 |
| 4096 | 6.535 | 5.542 | 1.317 |

The early merges already assemble many common Russian byte combinations. The same vocabulary shortens the English sequence much less. That is unsurprising for this Russian-trained experiment, but the claim must remain at the level of the evidence: **this vocabulary behaves this way on these data**. Isolating the effect of training language calls for a symmetric English-trained experiment, not just one direction of comparison.

The difference between 6.535 on training text and 5.542 on held-out Russian text reveals another effect: the learned units fit the material from which they were selected more closely. This is not, by itself, a universal measure of overfitting. The two finite partitions differ in documents, vocabulary, and frequencies. Understanding the experimental design comes before naming the cause of a gap.

Bytes per token also has a natural limitation. Russian text often uses more UTF-8 bytes per letter than English. A byte ratio therefore does not directly measure the cost of conveying matched content. For that question, it is more useful to encode aligned segments separately and compare their token counts.

With 4,096 merges, the 167 evaluation segments encoded individually without an added line feed produce **6,079 Russian tokens and 13,789 English tokens**: averages of 36.40 and 82.57 per segment. This is a more direct comparison of aligned content for one vocabulary. It is still not a property of entire languages: topic selection, translation direction, and the chosen segments remain part of the result.

Notice that the input protocol has changed too. The preceding table encodes files assembled with line feeds between segments and a final LF. The segment-level comparison encodes each example separately. Subtracting a count of line feeds from the file-level token total will not reproduce it: a separator can change pre-tokenization and merges near a fragment’s end.

Another common measure is **fertility**, the number of tokens per word. It sounds closer to ordinary reading, but first we must define a word. In this module’s examples, words are runs of letters or digits matched by `\p{L}+|\p{N}+`. This is a reproducible counting convention, not a universal linguistic definition that applies equally to prose, programs, and dates.

For a corpus-level measure, we use the ratio of totals:

$$
\operatorname{fertility}=\frac{\sum_i t_i}{\sum_i w_i},
$$

where $t_i$ and $w_i$ are the token and chosen word counts for example $i$. Averaging the individual ratios $t_i/w_i$ answers a different question: it gives a short example the same weight as a long one. Appendix D illustrates the difference and records the complete experimental protocol.

This brings us back to the document assistant. Before promising that a document will fit—or estimating the cost of processing it—we should measure it **with the intended tokenizer and the intended request format**. The second condition matters because a conversation contains more than the visible document text.

## 12. When text becomes a conversation

So far, we have encoded ordinary strings. An assistant interface distinguishes instructions, user messages, model replies, and tool results. A reader sees these distinctions in separate boxes and labels. The model’s input needs a representation of them too.

A **chat template** serializes the message structure into the form the model has been trained to process. The sequence may contain special markers for message boundaries, separators between headers and bodies, end-of-turn signals, or transfers of control to a tool.

Not every word in this framing must be a special token. A role or channel name, for example, can remain ordinary text between special delimiters. We therefore need the actual template and tokenizer, rather than assuming that one family’s conventions apply to another.

Consider the visible string `<|start|>`. It might be a control marker intentionally inserted by the application. It might also occur in a user’s document discussing the protocol itself. The first use calls for a special ID; the second normally calls for literal text. The spelling alone does not tell the application which interpretation is appropriate.

The `tiktoken` library makes that choice explicit. Its ordinary `encode()` call rejects registered special-token strings by default unless they are allowed. The developer can permit selected markers through `allowed_special`, or choose to encode their spellings as ordinary text. `encode_ordinary()` likewise ignores special-token semantics. Appendix E gives the exact calls. [\[7\]](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py)

This is one point where tokenization meets interface security. The program should construct trusted protocol framing; an accidental match inside an untrusted document should not silently gain the same status. But proper handling of special tokens does not prevent every prompt-injection attack. An attempt to redirect a model can be written in entirely ordinary language. Here we are separating one specific layer: textual data from protocol control units.

**Harmony**, used by the open gpt-oss family, provides a concrete artifact to inspect. It combines ordinary tokens with protocol and reserved IDs. Counting them reveals a small but instructive detail: one ID can have two names. The number of special-token names is therefore not always the number of distinct identifiers that a matrix must accommodate. [\[8\]](https://developers.openai.com/cookbook/articles/openai-harmony)

For a practical request-length estimate, count the final serialized input: instructions, messages, tool descriptions, and any other included parts. The token count of a document in isolation is still useful, but it is not yet the length of the model’s complete input.

## 13. What changes for images and sound?

After the text examples, “token” can start to sound unambiguous: an ID for a fragment in a vocabulary. Multimodal models make that definition too narrow. They still need units of computation, but those units do not all have to be discrete text IDs.

A **Vision Transformer** commonly divides an image into rectangular **patches**. Each patch is transformed into a vector, and a vision network processes the resulting sequence. There need not be an intermediate lookup of an ID from the text vocabulary. The patch vector can be a continuous transformation of pixel values. [\[9\]](https://arxiv.org/abs/2010.11929)

An image of 896 × 896 pixels with 14 × 14 patches has 64 patches along each axis, or $64\times64=4096$ in total. That is the number of original patches, not necessarily the number of positions the image occupies in a language model. A **connector** between the vision network and the LLM may change the feature width and sometimes reduce the number of vectors.

If the resulting visual vectors enter the language decoder’s self-attention sequence, they contribute to its context budget and intermediate-state storage. Other architectures exchange information differently. Original patches, vectors passed to the LLM, and billable API units should therefore not be treated as interchangeable counts without checking the system.

Audio can use a representation closer to the discrete text case. Some neural codecs apply **residual vector quantization**, or **RVQ**. A vector is approximated by an entry from a finite table, a **codebook**; the next codebook represents the residual error, and later codebooks refine it further. The selected indices can be transmitted, and a decoder reconstructs a lossy audio signal from them. [\[10\]](https://arxiv.org/abs/2210.13438)

If each frame carries $K$ indices and the frame rate is $f$, the stream contains $fK$ indices per second. A hypothetical rate of 50 frames per second with eight codebooks gives 400 IDs per second before further packing. We again have discrete units and a decoder, but their meaning and organization differ from text BBPE.

The common engineering question is unchanged: **what representation does the model actually receive, and how many computational positions does it occupy?** Module 16 develops this question for vision, audio, and video. For now, the important lesson is not to transfer properties between input types merely because both use the word “token.”

## 14. From a vector table to context

We began by asking how much of a document would fit. The answer led to two sides of the model’s input interface. One is discrete: a string has a particular byte representation; pre-tokenization sets boundaries; BPE learns and applies ordered merges. The ID sequence preserves the text, but its length depends on the vocabulary and the data.

The other side is trainable. An ID selects a row of $E$, turning a sequence of addresses into a matrix of vectors. The addresses themselves supply no useful semantic geometry. The vector values, however, can change with the rest of the network as the model learns its prediction task.

The output does not simply run the input operations backward. A hidden state becomes logits, softmax defines probabilities, and a generation rule chooses a new ID. Only then does the tokenizer recover its byte content. Tying weights lets one table serve both ends of the network without making prediction an inverse lookup.

These distinctions give us a way to reason about engineering trade-offs. Larger tokens can shorten a sequence, but need vocabulary entries and adequately trained parameters. A representation that works well on one corpus may work less well on another. An accurate input count requires not just the document but the normalization rules, special-token policy, message template, and artifact version.

One central computation remains unexplained. In a table, one ID always selects the same vector. A useful answer, however, depends on nearby tokens, earlier information, and the parts of the context relevant to the task. How can identical initial vectors acquire different states in different surroundings?

That is the subject of **Module 3: the basic Transformer**. We will build the computation between $X$ and $h$: attention, per-position transformations, residual connections, and the restriction on reading future tokens. Module 4 then develops positional representations in their own right. The table $E$ does not encode a position by itself; at the same time, a causal mask affects access to ordering information, so the absence of an explicit positional vector does not mean that a network has no ordering cues whatsoever.

To move on, you should be able to trace two paths: from a string to the input matrix, and from a completed hidden state to a selected token. We now have both. The next module explains the computation that joins them.

Next[Module 3. The Basic Transformer →](/courses/modern-llms/module-03/)

How an initial token representation becomes a representation shaped by its context.

------------------------------------------------------------------------

From the chapter’s continuous explanation to detailed calculations, algorithmic code, and real artifacts.

## Appendix A. Alternative segmentations and the information-theoretic connection

### A.1. One interface, several ways to build a vocabulary

BPE is a good first algorithm because its decisions can be followed as local replacements. But the interface “string → IDs” does not dictate BPE. Tokenizers differ both in how they learn a vocabulary and in how they find a segmentation for new text. The history is useful not as a procession of generations, but as a record of different problems being addressed.

Philip Gage’s original BPE appeared in 1994 as a data-compression algorithm. Sennrich, Haddow, and Birch adapted the idea to rare words in neural machine translation, first in a 2015 preprint and then at ACL 2016. Common fragments received their own representations, while rare words could be assembled from known parts. Starting from bytes is an additional design choice, not a necessary property of every algorithm called BPE. [\[2\]](https://aclanthology.org/P16-1162/)

**SentencePiece** is not the name of a single segmentation algorithm. It is a toolkit that supports, among other choices, BPE and **Unigram** models trained directly on strings, without requiring a pre-existing word segmentation. Spaces usually receive an explicit representation, `▁`: they are incorporated into the problem rather than silently discarded. [\[11\]](https://arxiv.org/abs/1808.06226)

Unigram assigns probabilities to vocabulary pieces and scores a segmentation through the product of its piece probabilities. With a fixed model, it can select the most probable valid segmentation. Stochastic segmentation can also be used as a form of regularization when training the language model. BPE instead applies a learned ordering of merges. Thus “BPE is deterministic, Unigram is random” is the wrong distinction: determinism depends on the chosen mode, while the algorithms differ in vocabulary learning and segmentation criteria.

We also need to inspect what happens **before** segmentation. SentencePiece uses `nmt_nfkc` normalization by default and can remove extra whitespace. That can be appropriate for one task and destructive for another. Code, exact quotations, and visually similar spellings require attention to the actual configuration. The reversibility guarantee of our byte-preserving BBPE does not automatically transfer to every tokenizer. [\[12\]](https://google.github.io/sentencepiece/doc/normalization/)

The byte-level design associated with GPT-2 covers valid UTF-8 without an unknown-word token. The open tiktoken encodings `r50k_base`, `cl100k_base`, and `o200k_base` illustrate changes in vocabulary size and pre-tokenization rules. They are not points on a universal scale of tokenizer quality: each artifact must be considered alongside its data, model, and intended task. [\[7\]](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py)

There are also models without a fixed learned subword vocabulary. MEGABYTE, MambaByte, and Byte Latent Transformer explore different ways to organize computation over bytes. BLT, for instance, groups bytes into dynamic patches, so *tokenizer-free* should not be read as “without any segmentation.” A particular fixed vocabulary of text fragments disappears, but the representation still has to fit the computational architecture. Such systems should be compared by quality and overall cost, not merely by the length of an intermediate sequence. [\[13\]](https://arxiv.org/abs/2305.07185)

### A.2. A shorter sequence is not necessarily a smaller file

The main chapter measured how many source bytes a token covers on average. That describes the length of the model’s input. Estimating the size of a **stored encoding** also requires a decision about how the IDs themselves are represented.

Suppose all IDs lie between 0 and $R-1$, with $R\ge2$. A fixed-width representation needs $\lceil\log_2R\rceil$ bits per ID. For $T$ tokens representing $N_{\mathrm{bytes}}$ source bytes, the payload cost is

$$
\mathrm{bits\ per\ input\ byte}
=\frac{T\lceil\log_2R\rceil}{N_{\mathrm{bytes}}}.
$$

After 4,096 merges, our teaching vocabulary has $R=256+4096=4352$, requiring 13 bits per ID under this scheme. That is more than the eight bits of a source byte. A space saving appears only when the reduction in the number of units compensates for their increased width.

Even this calculation covers only the payload under the chosen packing rule. A self-contained archive also needs the vocabulary, merges, and metadata. An in-memory ID array may use `int32` or `int64`, not densely packed 13-bit values. Conversely, entropy coding can assign different effective lengths to frequent and rare events. Input length, a packed stream, an array in memory, and a complete archive are different objects with different costs.

If the ID range contains gaps, $R$ is not the number of defined tokens. We could remap the IDs into a dense range before packing, but that mapping would itself become part of the format. Appendix E examines a real artifact where the distinction matters.

### A.3. What information theory actually says

Suppose a source produces event $x$ with probability $p(x)$, while a probabilistic code is designed for $q(x)$. Provided $q(x)>0$ wherever $p(x)>0$, the ideal expected code length satisfies

$$
\mathbb E_p[-\log_2q(x)]
=H_2(p)+D_{\mathrm{KL},2}(p\Vert q),
$$

$$
D_{\mathrm{KL},2}(p\Vert q)
=\sum_xp(x)\log_2\frac{p(x)}{q(x)}.
$$

The source entropy $H_2(p)$ measures its own uncertainty; KL divergence supplies the extra cost of using a mismatched distribution. The identity follows by adding and subtracting $-\log_2p(x)$ inside the expected length. If $q$ assigns zero probability to an event that can occur, the ideal cost becomes infinite.

This is a useful framework for reasoning about a model’s fit to data. BPE, however, is not this probabilistic code. A difference in `bytes/token` between corpora is not a KL divergence and does not identify either $p$ or $q$. In the Russian-trained, English-evaluated experiment, language changes as well as the documents’ participation in training. Information theory helps frame the question; it does not license renaming one measured quantity as another. Coding and relative entropy are developed in detail in *Information Theory for ML*.

### A.4. Why embeddings are connected with co-occurrence statistics

Word vectors have an important history of their own. In **skip-gram with negative sampling**, or **SGNS**, one table represents words and another represents contexts. An idealized analysis connects an optimal pair score with **pointwise mutual information**, or PMI:

$$
\operatorname{PMI}(w,c)=\log\frac{p(w,c)}{p(w)p(c)},
\qquad
 e_w^\top u_c\approx\operatorname{PMI}(w,c)-\log k.
$$

Here $k$ is the number of negative samples. The fraction gives the intuition directly: how much more often do the word and context occur together than would be expected under independence? Their association makes their joint occurrence informative.

The relationship has assumptions. For independently optimized pair scores, the $\log k$ shift arises when negative contexts follow the marginal context distribution. A different negative-sampling distribution changes the expression for the optimum. Shared finite-dimensional vectors also cannot assign every pair score independently. The result is therefore an analysis of an objective, not an exact identity promised by every trained table. [\[14\]](https://papers.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)

GloVe uses a different, explicit objective for co-occurrence statistics. A Transformer’s embedding table retains the “ID → vector” interface, but its values are trained as part of the full prediction task. It need not factorize the same PMI matrix. The shared interface reveals a continuity of ideas; the different losses explain why the resulting representations should not simply be equated.

## Appendix B. Embeddings: shapes, gradients, and a small experiment

### B.1. Lookup as a linear operation

For ID $i$, define a **one-hot vector** $e^{(i)}$: a row of length $V$ with one at position $i$ and zeros elsewhere. Then

$$
e^{(i)}E=E_i.
$$

Stacking these rows for an input sequence gives $S\in\mathbb R^{T\times V}$ and $X=SE$. This mathematical form connects lookup to ordinary linear algebra. It does not mean an implementation should allocate an enormous, almost entirely zero matrix $S$. `E[ids]` reads the selected rows directly.

For our example,

$$
E=\begin{pmatrix}1&0\\2&0\\0&1\end{pmatrix},
\qquad
S=\begin{pmatrix}1&0&0\\0&0&1\\1&0&0\end{pmatrix},
\qquad
X=\begin{pmatrix}1&0\\0&1\\1&0\end{pmatrix}.
$$

We have $V=3$, $d=2$, and $T=3$. Two equal rows of $X$ tell us only that an ID was used twice. Later computation can transform them differently because they occupy different positions with different available contexts.

### B.2. A training step with nothing hidden

To isolate table learning, give every selected position the target $Y_t=[0,0]$ and minimize

$$
L(E)=\frac12\sum_{t=1}^{T}\|E_{i_t}-Y_t\|^2.
$$

The zero targets make the arithmetic simple. They do not define useful language-model training: if every vector became zero, the input would no longer distinguish tokens. We are isolating one operation—how a shared parameter receives signals from several uses.

The derivative of $\frac12(x-y)^2$ with respect to $x$ is $x-y$. Apply this coordinate by coordinate to a vector. Because row $E_j$ can occur in several terms of the loss, its gradient adds those contributions:

$$
\frac{\partial L}{\partial E_j}
=\sum_{t:\,i_t=j}(E_j-Y_t).
$$

With `[0,2,0]`, row 0 is used twice and receives $[1,0]+[1,0]=[2,0]$. Row 2 receives $[0,1]$. Row 1 does not participate in this loss. Thus

$$
\nabla_EL=\begin{pmatrix}2&0\\0&0\\0&1\end{pmatrix},
\qquad
E_{\mathrm{new}}=E-\eta\nabla_EL.
$$

For learning rate $\eta=0.1$,

$$
E_{\mathrm{new}}=\begin{pmatrix}0.8&0\\2&0\\0&0.9\end{pmatrix}.
$$

The initial loss is $\frac12(1+1+1)=1.5$. After the update it is $\frac12(0.8^2+0.9^2+0.8^2)=1.045$. Rather than merely calling the table trainable, we have specified an objective, calculated its derivative, and checked the consequence of changing the parameters. A much larger step could increase the loss: the gradient supplies local information, not permission to move arbitrarily far.

### B.3. Repeated indices need deliberate accumulation

In NumPy, the accumulation can be written directly:

```python
X = E[ids]
grad = np.zeros_like(E)
np.add.at(grad, ids, X - targets)
E_new = E - 0.1 * grad
```

Every occurrence of an ID adds a contribution to the same row of `grad`. The superficially similar `grad[ids] += X - targets` uses a temporary advanced-indexing result and does not provide the same accumulation for repeated indices. For ID 0, this is essential: two uses must contribute twice. `np.add.at` has the semantics required here. [\[15\]](https://numpy.org/doc/2.4/reference/generated/numpy.ufunc.at.html)

A separate check uses **finite differences**. Perturb one parameter $E_{ja}$ slightly in either direction and compare the losses:

$$
\frac{\partial L}{\partial E_{ja}}
\approx
\frac{L(E+\varepsilon D^{(ja)})-L(E-\varepsilon D^{(ja)})}{2\varepsilon}.
$$

The matrix $D^{(ja)}$ has a one only at the coordinate being tested. A small $\varepsilon$ approximates the derivative, while an excessively small one magnifies round-off problems. For a small numerical check, a possible step size is $10^{-6}$. Checking parameters this way is too costly as a method for training a large network, but valuable for validating a small computation.

### B.4. What the language-model objective adds

An LLM usually predicts a next-token ID rather than a prescribed target vector. For a target $y$, the single-example cross-entropy loss is $\ell=-\log p_y$. Under an ordinary full softmax,

$$
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=y].
$$

An output parameter can therefore receive a signal even when its token is not the correct answer: the model learns to reduce excessive probability assigned to alternatives. Rare tokens have few **positive observations**, not necessarily no updates of their output parameters.

With untied weights, distinguish input row $E_j$ from output column $U_{:,j}$. With $U=E^\top$, they share parameters, and the complete gradient combines input and output paths. The quadratic experiment above contains no $hU$ output. Its zero gradient for an unused row cannot be transferred to an entire language model. This is the boundary between a precise teaching example and an unjustified generalization.

## Appendix C. Teaching BBPE: encoding, decoding, and checks

### C.1. What the implementation covers

The short example below isolates BPE training: split the text into chunks, count adjacent pairs, and apply the chosen merges in sequence. The function `train_bbpe` returns the rules that an encoder will later use. This fragment requires only the `regex` library.

Unlike Python’s standard `re` module, `regex` understands the `\p{L}` and `\p{N}` Unicode properties used in the pattern.

The regular expression is deliberately kept visible as a separate object. Its alternatives handle English contractions, letter runs with an optional leading character, one-to-three-digit groups, punctuation, and whitespace sequences. It is a **cl100k-like teaching pattern**, not a claim of compatibility with a production tokenizer. The real `o200k_base` has more elaborate alternatives involving case and Unicode categories.

### C.2. From the training rule to Python

`train_bbpe` counts pairs and selects the most frequent. Ties go to the first encountered pair—a specific tie-break rule that affects later training. `encode` no longer consults corpus frequencies. It selects the available pair with the lowest learned rank and repeats until no permitted merge remains.

Pair counts can include overlapping adjacencies. In `aaa`, `aa` occurs at two neighboring positions, but a merge pass can replace only one non-overlapping occurrence. The code exposes both operations: `zip` counts neighboring pairs, while a left-to-right replacement pass skips two elements after a match.

The initial vocabulary contains all 256 bytes. Each executed merge adds a new ID, so $M$ completed merges produce $256+M$ IDs in this implementation, which has no special tokens. If there are no adjacent pairs left, training stops before reaching the requested budget. An empty training string leaves only the byte alphabet.

The naive trainer scans the corpus at every step, giving an upper time bound of $O(MN)$ for $M$ merges and $N$ initial bytes. Production trainers maintain pair indexes and update affected regions. The existing course notebook also demonstrates an optimization that stores each distinct chunk once together with its frequency. Such optimizations must preserve the intended tie-break semantics as well as the counts.

```python
import regex as re
from collections import Counter

PAT = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

def get_stats(ids):
    counts = Counter()
    for chunk in ids:
        for pair in zip(chunk[:-1], chunk[1:]):
            counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    out = []
    for chunk in ids:
        new_chunk, i = [], 0
        while i < len(chunk):
            if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                new_chunk.append(new_id)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        out.append(new_chunk)
    return out

def train_bbpe(text, num_merges=1000):
    if type(num_merges) is not int or num_merges < 0:
        raise ValueError("num_merges must be a non-negative integer")
    chunks = re.findall(PAT, text)
    ids = [list(chunk.encode("utf-8")) for chunk in chunks]
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, 256 + i)
        merges[pair] = 256 + i
    return merges
```

### C.3. What round-trip checks establish

The six-merge example in the main chapter gives frequencies `7, 6, 5, 5, 4, 4` and sequence lengths `86, 80, 75, 70, 66, 62`. These numbers connect the hand-worked trace to the operations performed by `get_stats` and `merge`.

For an encoder and decoder, round-trip preservation concerns the entire original text—including an empty string, Cyrillic, Latin text, emoji, combining marks, and whitespace. An unpaired surrogate falls outside our valid-UTF-8 contract. An undefined ID or an unfinished UTF-8 sequence under strict decoding must likewise raise an error rather than silently corrupting the data.

During incremental decoding, byte `D0` does not yet specify a complete character: the decoder can emit `П` only after `9F` arrives. It must retain the unfinished byte sequence between fragments. Decoding each token independently and replacing errors with `�` does not solve the same problem.

## Appendix D. Reproducing and interpreting a tokenizer comparison

### D.1. Fix the object before measuring it

The corpus experiment uses **Parallel Universal Dependencies, UD 2.15**. Text is extracted verbatim from the `# text` fields, without Unicode normalization; order and segment alignment are retained. Files join segments with LF and include a final LF. The split is determined by the source document ID: convert the first eight hexadecimal digits of the SHA-256 hash of its ASCII representation into an integer; a remainder of zero modulo five assigns the document to evaluation, and every other remainder assigns it to training. Both languages use the same document partition. [\[6\]](https://github.com/UniversalDependencies/UD_Russian-PUD/tree/7284f9241af8755cbecdc6f6ce3eaf338f0503c4)

| File                                    | Segments | Documents | Bytes   |
|-----------------------------------------|----------|-----------|---------|
| RU train                                | 833      | 331       | 180,112 |
| RU eval                                 | 167      | 66        | 33,654  |
| EN train — for the symmetric experiment | 833      | 331       | 93,253  |
| EN eval                                 | 167      | 66        | 18,168  |

Most original PUD texts were written in English. The corpus topics and the translated origin of some material limit generalization to entire languages.

This is a teaching split for tokenization. Upstream PUD labels the material as a parsing test set; the experiment does not establish a new standard parsing benchmark. “Segment” is also more accurate than “sentence”: an aligned unit can contain more than one grammatical sentence.

The Russian source is pinned to revision `7284f9241af8755cbecdc6f6ce3eaf338f0503c4`, and the English source to `6e328e8195c04a297c1e375d21fe663be1ce6178`. The corpus is distributed under CC BY-SA 3.0, with attribution to PUD contributors, DFKI translation teams, Google, and the Universal Dependencies community. The course materials retain the full manifest, licenses, and original notices.

The frozen teaching tokenizer is version `bbpe-course-1`: 256 base bytes, 4,096 merges, and 4,352 IDs. The SHA-256 of `bbpe_ru_4096.json` is `333abcfd91dcb03d6270e44ec171a36f856eeed4fdf0d5f1f099819822fa34cd`. This is the artifact that produces `[943,370,301]` for `Привет`. A different trainer or tie-break rule need not reproduce those identifiers.

### D.2. Rounded ratios need raw counts behind them

The main chapter reports `bytes/token`. Keeping the denominators makes the results easier to check. The following table gives token counts when each **whole file**, including line feeds, is encoded:

| Merges | RU train | RU eval | EN eval |
|--------|----------|---------|---------|
| 0      | 180,112  | 33,654  | 18,168  |
| 64     | 80,545   | 14,994  | 18,005  |
| 256    | 55,751   | 10,474  | 17,963  |
| 1024   | 39,754   | 7,719   | 17,346  |
| 4096   | 27,560   | 6,073   | 13,793  |

For example, $33\,654/6\,073\approx5.542$ bytes per token. This is total bytes divided by total tokens, not an arithmetic mean of sentence-level ratios. Those aggregation rules weight short and long examples differently and generally do not agree.

The two control strings below are not part of the aligned evaluation and cannot replace it. They make the effect on a particular input easy to see:

| Merges | `Внимание — это всё, что вам нужно.` | `Attention is all you need.` |
|--------|--------------------------------------|------------------------------|
| 0      | 61                                   | 26                           |
| 64     | 35                                   | 26                           |
| 256    | 19                                   | 26                           |
| 1024   | 15                                   | 26                           |
| 4096   | 13                                   | 22                           |

Encoding the 167 aligned examples **separately, without an added LF**, gives 6,079 RU tokens and 13,789 EN tokens. These totals differ from the whole-file columns. A separator can affect pre-tokenization and merges near a fragment boundary, so subtracting a number of newline characters from one token count does not convert it into the other measurement protocol.

Later merges in this experiment yield a smaller saving per added rule. That claim must be assessed against the **number** of merges, not the visual distances between points on an arbitrarily chosen logarithmic axis. A persuasive-looking curve does not remove the need to define the slope being compared.

### D.3. Four strings are four observations, not four rankings

For **fertility**, define a word as a match of `\p{L}+|\p{N}+`: a continuous run of letters or digits. This is reproducible but is not a linguistic analysis of programs, formulas, or dates. With the 4,096-merge vocabulary, the examples give:

| Example           | Tokens | Letter/digit runs | Tokens per word | Bytes per token |
|-------------------|--------|-------------------|-----------------|-----------------|
| Russian prose     | 16     | 7                 | 2.29            | 6.94            |
| English prose     | 44     | 8                 | 5.50            | 1.18            |
| Python            | 63     | 13                | 4.85            | 1.08            |
| Numbers and dates | 29     | 6                 | 4.83            | 1.03            |

The exact strings follow. In the Python example, `\n` denotes an actual line break rather than a literal backslash and letter.

```text
Языковые модели предсказывают следующий токен по контексту.
Language models predict the next token from context.
def encode(text):\n    return [merges.get(p, p) for p in pairs(text)]
1234 + 5678 = 6912; 2026-07-04
```

The English string has the highest fertility among these four observations. That is consistent with a vocabulary trained on Russian prose, but a single example per category cannot separate language from length, lexical choice, punctuation, or content. A domain comparison needs sets of examples and a view of their variability.

Whitespace is part of the experiment too. `one two` contains seven bytes and six tokens; `one two` contains eight bytes and seven tokens. Their bytes-per-token ratios are 1.1667 and 1.1429. “The same words” does not mean “the same input.” Similarly, **single-token retention**—the share of selected words represented by one token—requires a decision about whether a word is encoded in isolation, with a leading space, or within an actual sentence.

For a hypothetical aggregation example, suppose a short passage has one word represented by four tokens, while a longer passage has ten words represented by ten tokens. The mean of the two fertility values is $(4+1)/2=2.5$. The ratio of totals is $(4+10)/(1+10)=14/11\approx1.273$. The first gives equal weight to passages; the second measures tokens per counted word across the entire set. Neither should be silently substituted for the other.

**Chars per token** can provide another view, with code points used as the character-counting unit. A report should state that choice: counting grapheme clusters would give a different quantity. This complements the byte measure without becoming a measure of semantic content.

### D.4. Turning an observation into a controlled comparison

To ask about new documents, compare training and evaluation data within one language. To ask about matched content, use aligned text. To investigate the training-language effect, run the symmetric experiment: train another tokenizer on English with a comparable budget and repeat the checks. The EN training file is retained for precisely this purpose. Each design controls a different part of the explanation.

In multilingual use, token-count differences can affect cost and the amount of content that fits. But UTF-8 bytes are not units of meaning, and fertility depends on how words are counted. Research on cross-language tokenization disparities examines real consequences of these differences; conclusions about a particular system still require that system’s artifacts and data. [\[16\]](https://arxiv.org/abs/2305.15425)

At a fixed rate, three times as many billable input tokens means three times the **input charge**. It does not establish the ratio of complete request costs: outputs, caching, tools, and multimodal inputs may be accounted for separately. Latency also depends on the implementation and processing stage, not just one length measurement.

For a closed service, use its official counting method and preserve the complete request: model, messages, tools, and attachments. For a local model, also record the tokenizer version, normalization, special-token policy, and chat template. The useful result is not a label saying “best vocabulary,” but an answer to a specific question under stated conditions.

The data, frozen tokenizer, and reference metrics are available in the [Module 2 materials](https://github.com/VladGKulikov/VladGKulikov.github.io/tree/main/public/downloads/modern-llms/modules/module-02/data). The [module notebooks](/courses/modern-llms/module-02/notebooks/) contain the full vocabulary-training experiment. The values in this chapter belong to that fixed setup, not to an arbitrary fresh download of a corpus or a different library version.

## Appendix E. Reading a real tokenizer artifact

### E.1. Three numbers hidden inside “vocabulary size”

The question “how large is the vocabulary?” can refer to three quantities. Let $I$ be the set of **defined unique IDs**, combining ordinary and special identifiers. Then $|I|$ counts how many distinct IDs the artifact actually defines.

The value $R=\max(I)+1$ describes the index range starting at zero. If there are gaps, $R>|I|$. Finally, $V$ is the number of rows allocated by the model’s input table and, in the usual setup, the number of output scores. This is generally what a model configuration’s `vocab_size` specifies. Direct indexing without a separate remapping requires $V\ge R$.

| Artifact | Ordinary IDs | Additional unique IDs | $\vert I\vert $ | $R$ | Model $V$ |
|----|----|----|----|----|----|
| `o200k_base` | 199,998 | 2 | 200,000 | 200,019 | Not determined by the tokenizer |
| `o200k_harmony` / gpt-oss-120b | 199,998 | 1,090 | 201,088 | 201,088 | 201,088 |
| `Qwen/Qwen3-8B` | 151,643 | 26 | 151,669 | 151,669 | 151,936 |

In `o200k_base`, the special IDs are 199999 and 200018. Together with ordinary IDs 0 through 199997, they leave 19 unused values inside the range. `tiktoken.n_vocab` returns $R$, not the cardinality of $I$. Conversely, Qwen3-8B allocates 267 more rows than its defined IDs require. Those rows cannot automatically be called new decodable tokens or assigned a particular purpose. The implementation must determine whether the corresponding output positions are permitted. [\[7\]](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py) [\[17\]](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/vocab_snapshot_2026-10-11.json)

### E.2. Harmony: more names than identifiers

In the pinned tiktoken source, `o200k_harmony` inherits the ordinary tokens and regular expression of `o200k_base`, then extends the special-token mapping. The `.tiktoken` file for ordinary units stores `base64(token_bytes) rank` pairs. In this implementation, a rank serves both as the ordinary token’s ID and as its merge priority.

There are 199,998 ordinary IDs. There are 1,091 special-token **names**, but only 1,090 unique special **IDs**. This is not rounding: `<|endofprompt|>` and `<|reserved_200018|>` are aliases for ID 200018. Hence

$$
|I|=199\,998+1\,090=201\,088.
$$

Ten special names do not have the `reserved` form; the other 1,081 take the form `<|reserved_N|>`. A reserved label does not establish that its row has already been trained for a useful function. We are inspecting the artifact’s structure, not inferring undisclosed training details from names.

The exact counts refer to tiktoken revision `4e71bbe0c078468e00fefbf94b39849389f346e5`. That precision is more useful than “roughly 200,000 tokens” when checking an array size, a permitted ID, or its relationship to the model configuration. [\[7\]](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py)

### E.3. The literal spelling of a control marker

The API policy determines what happens when text resembles a registered special token. In tiktoken, the ordinary `encode()` call rejects such a string by default unless explicitly allowed. `encode_ordinary()` preserves it as ordinary text. Passing the marker in `allowed_special` authorizes its special interpretation.

```python
import tiktoken

enc = tiktoken.get_encoding("o200k_harmony")
text = "<|start|>"

ordinary = enc.encode_ordinary(text)
assert enc.decode(ordinary) == text

special = enc.encode(text, allowed_special={"<|start|>"})
assert special == [200006]  # For the pinned revision.

same_as_ordinary = enc.encode(text, disallowed_special=())
assert same_as_ordinary == ordinary
```

The last call disables the forbidden-string check while retaining the **default empty** `allowed_special`, so it uses ordinary pieces. If a special token has separately been allowed, merely disabling the prohibition does not undo that permission. The behavior depends on the combination of arguments, a detail worth keeping explicit in an API explanation. [\[7\]](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py)

This snippet uses a separate library and is not part of Appendix C’s self-contained test program. It illustrates a production interface; exact IDs should be checked against the pinned artifact. Neither `o200k_harmony` nor the assumption that every visible role string is a special ID should be carried over automatically to another model.

### E.4. Which files answer which questions?

In Hugging Face artifacts, inspect `model.vocab`, `model.merges`, `pre_tokenizer`, and `added_tokens` in `tokenizer.json` separately. `normalizer` describes input transformations, `decoder` describes reconstruction, and other processing stages may insert tokens. `tokenizer_config.json` often contains the `chat_template` used to serialize roles and messages. These are different layers: the template uses the tokenizer but is not the BPE algorithm.

Then read the **model’s own configuration**, rather than trying to infer it entirely from the tokenizer file. The following values come from the course’s pinned snapshot of **11 October 2026**. The table compares allocated model dimensions only—not model quality, recency, or the number of words in a natural language. [\[17\]](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/vocab_snapshot_2026-10-11.json)

| Published configuration | Field read | Value |
|----|----|----|
| `openai/gpt-oss-120b` | `vocab_size` | 201,088 |
| `deepseek-ai/DeepSeek-V4-Flash` | `vocab_size` | 129,280 |
| `Qwen/Qwen3-8B` | `vocab_size` | 151,936 |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `params.json: vocab_size` | 131,072 |
| `moonshotai/Kimi-K3` | `text_config.vocab_size` | 163,840 |
| `zai-org/GLM-5` | `vocab_size` | 154,880 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `vocab_size` | 131,072 |

The snapshot retains revisions, URLs, SHA-256 hashes, and copies of the sources read. The configurations for gpt-oss-120b, Qwen3-8B, DeepSeek-V4-Flash, and Kimi-K3 have input/output weight tying disabled. A configuration **class default** is a different kind of evidence: in the pinned source, `Gemma3TextConfig` defaults to tying and a vocabulary dimension of 262208, while `Llama4TextConfig` defaults to 202048. A class default is not the configuration of every checkpoint in its family. [\[18\]](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/llama4/configuration_llama4.py)

This is why the historical Llama-3.1-8B memory example is kept separate. It demonstrates the arithmetic for stated $V$ and $d$, rather than standing in for a newly retrieved artifact. In the original snapshot, some Llama, Gemma, and Command A files required authorization and were excluded from the table of confirmed reads.

### E.5. What changes behind a closed API?

For services such as Claude or Gemini, a local vocabulary may not be available. The practical answer then comes from the official token-counting method applied to the specific request. Claude’s documentation notes that a preliminary estimate can differ from actual usage; Gemini likewise makes counting depend on the selected model and input contents. A relationship between Gemini and the open Gemma family does not prove identical tokenization. [\[19\]](https://platform.claude.com/docs/en/build-with-claude/token-counting)

A reproducible report records more than the returned number: the model, call date, interface version, messages, tool definitions, and multimodal parts matter too. This is the same principle as in the local BBPE experiment. A measurement becomes meaningful when the object and conditions that produced it are retained.

## Sources and course connections

The corpus experiment is fixed to PUD 2.15, and comparative configurations belong to the 11 October 2026 snapshot. The values describe those artifacts, not the latest releases. The chapter’s small calculations and checks are reproducible with its embedded program.

**\[1\] Unicode Consortium — Unicode Text Segmentation (UAX \#29)**\
[Unicode UAX \#29](https://www.unicode.org/reports/tr29/)\
Code points, grapheme clusters, and text boundaries. The chapter’s byte examples are calculated directly using UTF-8.

**\[2\] Gage (1994); Sennrich, Haddow & Birch (2016) — Byte Pair Encoding and subword units**\
[Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)\
The origins of BPE and subword representations in translation. The six-step Russian example uses the explicitly defined teaching algorithm.

**\[3\] Press & Wolf — Using the Output Embedding to Improve Language Models**\
[arXiv:1608.05859](https://arxiv.org/abs/1608.05859)\
Input/output weight tying. The numerical matrices in this chapter are independent teaching examples.

**\[4\] OpenAI — gpt-oss-120b configuration**\
[config.json, revision b5c939d](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) · [Retained source](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/sources_2026-10-11/configs__gpt-oss-120b.txt)\
Vocabulary dimension 201088, hidden width 2880, untied input/output weights. The revision is pinned in the course snapshot.

**\[5\] Meta — Llama 3.1 architecture definitions**\
[Official llama-models: sku_list.py](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)\
Historical Llama-3.1-8B example: width 4096 and vocabulary dimension 128256. It is kept separate from the artifacts read in the 11 October 2026 snapshot.

**\[6\] Universal Dependencies — Parallel Universal Dependencies, UD 2.15**\
[RU, pinned revision](https://github.com/UniversalDependencies/UD_Russian-PUD/tree/7284f9241af8755cbecdc6f6ce3eaf338f0503c4) · [EN, pinned revision](https://github.com/UniversalDependencies/UD_English-PUD/tree/6e328e8195c04a297c1e375d21fe663be1ce6178) · [corpus_manifest.json](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/corpus_manifest.json) · [reference_metrics.json](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/reference_metrics.json)\
Source texts and retained results of the teaching experiment. PUD contributors, DFKI translation teams, Google, and the Universal Dependencies community; CC BY-SA 3.0.

**\[7\] OpenAI — tiktoken, revision 4e71bbe0c078468e00fefbf94b39849389f346e5**\
[Encoding constructors](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py) · [Encoding API and n_vocab](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/core.py)\
Ranks, regular expressions, special names and IDs, encode policies, and n_vocab semantics. Exact counts refer to this revision.

**\[8\] OpenAI — Harmony response format**\
[Official format guide](https://developers.openai.com/cookbook/articles/openai-harmony) · [openai/harmony](https://github.com/openai/harmony)\
Conversation and tool-call structure; not every visible header component is a special token.

**\[9\] Dosovitskiy et al. — An Image is Worth 16x16 Words**\
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)\
Images represented as patch sequences. The 896 × 896 image with 14 × 14 patches is a hypothetical calculation, not a claimed configuration from the paper.

**\[10\] Défossez et al. — High Fidelity Neural Audio Compression (EnCodec)**\
[arXiv:2210.13438](https://arxiv.org/abs/2210.13438) · [SoundStream](https://arxiv.org/abs/2107.03312)\
Neural audio codecs and residual vector quantization. The 50 × 8 calculation illustrates explicitly assumed parameters.

**\[11\] Kudo & Richardson — SentencePiece**\
[arXiv:1808.06226](https://arxiv.org/abs/1808.06226) · [Official implementation](https://github.com/google/sentencepiece)\
Learning tokenizers from raw strings, BPE and Unigram, and whitespace representation.

**\[12\] SentencePiece — normalization and model options**\
[Normalization documentation](https://google.github.io/sentencepiece/doc/normalization/) · [Options](https://google.github.io/sentencepiece/doc/options/)\
nmt_nfkc normalization, whitespace handling, and configuration-dependent reversibility.

**\[13\] Byte-level language models — MEGABYTE, MambaByte, BLT**\
[MEGABYTE](https://arxiv.org/abs/2305.07185) · [MambaByte](https://arxiv.org/abs/2401.13660) · [Byte Latent Transformer](https://arxiv.org/abs/2412.09871)\
Different ways to remove a fixed learned subword vocabulary; doing so does not eliminate bytes or intermediate patch structures.

**\[14\] Levy & Goldberg — Neural Word Embedding as Implicit Matrix Factorization**\
[NeurIPS 2014](https://papers.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html) · [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)\
The SGNS/shifted-PMI connection under stated assumptions; GloVe uses a different objective.

**\[15\] NumPy — ufunc.at and advanced indexing**\
[numpy.ufunc.at](https://numpy.org/doc/2.4/reference/generated/numpy.ufunc.at.html) · [Indexing semantics](https://numpy.org/doc/stable/user/basics.indexing.html)\
Accumulation at repeated indices.

**\[16\] Petrov et al. — Language Model Tokenizers Introduce Unfairness Between Languages**\
[arXiv:2305.15425](https://arxiv.org/abs/2305.15425)\
Consequences of cross-language tokenization disparities. This chapter’s PUD results refer only to its own fixed experiment.

**\[17\] Modern LLMs — vocabulary artifact snapshot, 11 October 2026**\
[vocab_snapshot_2026-10-11.json](https://github.com/VladGKulikov/VladGKulikov.github.io/blob/main/public/downloads/modern-llms/modules/module-02/data/vocab_snapshot_2026-10-11.json) · [Retained primary sources](https://github.com/VladGKulikov/VladGKulikov.github.io/tree/main/public/downloads/modern-llms/modules/module-02/data/sources_2026-10-11) · [Qwen3-8B pinned config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)\
Revisions, fields, hashes, retained files, and explicit records of unavailable artifacts. Not a survey of the latest releases.

**\[18\] Hugging Face Transformers — configuration classes, pinned source**\
[Llama4TextConfig](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/llama4/configuration_llama4.py) · [Gemma3TextConfig](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/gemma3/configuration_gemma3.py)\
Class defaults, not mandatory values for every checkpoint.

**\[19\] Anthropic and Google — official token-counting interfaces**\
[Claude token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) · [Gemini token counting](https://ai.google.dev/gemini-api/docs/tokens) · [Gemini countTokens reference](https://ai.google.dev/api/tokens)\
Counting a specific structured request; consult the service documentation for supported input types and estimation behavior.

Course connections: [M1 — model and system](/courses/modern-llms/module-01/) · [M3 — the basic Transformer](/courses/modern-llms/module-03/) · [M4 — position](/courses/modern-llms/module-04/) · [M13 — SFT and chat formatting](/courses/modern-llms/module-14/) · [M16 — multimodality](/courses/modern-llms/module-16/).
