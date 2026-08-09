# Module 2. Tokenization and Embeddings

*“Modern LLMs” course · Module 2 lecture · edition 2026.8*

> **What this module does.** Module 1 treated `vocab_size` as one field in a model passport. Here we open that field. We follow text from Unicode through pre-tokenization, UTF-8 bytes, BPE merges, token IDs, and embedding lookup. Along the way we will see why tokenization changes context length, multilingual cost, numerical inputs, protocol boundaries, and parameter count. The module ends by reading the public `o200k_harmony` implementation and comparing documented vocabularies across current model families.
>
> **Prerequisites.** Module 1 and basic Python: lists, `Counter`, regular expressions, and dictionaries.

---

## 1. Tokenization is an architectural decision

Most text LLMs do not learn their tokenizer jointly with the main network. A discrete tokenizer is trained or designed first, frozen, and then used to turn every training and inference string into integer IDs. The Transformer never receives the original Python string; it receives embedding rows selected by those IDs.

This decision has several consequences.

**Sequence length and cost.** Commercial APIs usually meter tokens. If one translation yields three times as many input tokens under the same model and price schedule, it consumes three times as much billed input and three times as much context capacity. The multiplier is not a property of the language alone: it depends on the exact text, tokenizer version, chat template, and provider contract.

**Problem representation.** The strings `123456`, an indented Python block, and a function call can be segmented in many ways. Each segmentation presents a different sequence-learning problem. Regular digit grouping may provide a useful inductive bias, but no token boundary by itself teaches arithmetic or tool use.

**Cross-lingual access.** Fertility—the number of tokens per word—can differ sharply across languages. High fertility reduces the amount of content that fits in a fixed context and raises per-token cost. It reflects UTF-8, vocabulary allocation, and training data rather than one universal property of BPE.

**Protocol boundaries.** Special IDs encode message boundaries, channels, tool calls, and other control structures. A tokenizer API must decide whether a literal string such as `<|start|>` is ordinary user text, a forbidden special-token spelling, or an authorized control token. That policy is part of the system’s security boundary, even though prompt injection is a broader application-level problem.

The vocabulary also consumes parameters. Llama 3.1 8B uses untied input and output tables of shape `128256 × 4096`, totaling roughly 1.05 billion parameters. Before the first attention block runs, the model has already paid substantially for its discrete interface.

## 2. From word vocabularies to byte-level subwords

Early neural NLP commonly used a fixed word vocabulary and an `<UNK>` token. The design was simple but brittle: unseen names and misspellings became **OOV—Out of Vocabulary**, and morphologically rich languages produced enormous word-form inventories.

Character models removed OOV at the cost of much longer sequences. The useful compromise arrived through **BPE—Byte Pair Encoding**. Philip Gage introduced BPE as a data-compression algorithm in 1994; Sennrich, Haddow, and Birch adapted repeated pair merging to neural machine translation in 2015. Frequent strings become single symbols, while rare strings remain decomposable.

**SentencePiece** made subword training independent of language-specific whitespace tokenizers. It supports both BPE and a probabilistic Unigram model and represents spaces explicitly, usually with `▁`. SentencePiece appears in T5, Llama 1/2, Gemma, and many other open families. Architectural ancestry does not prove that a closed service uses the same tokenizer artifact, so Gemini should not simply be inferred from Gemma.

GPT-2 took another step: **BBPE—Byte-level Byte Pair Encoding** begins with the 256 possible bytes rather than Unicode characters. Any UTF-8 string is representable without `<UNK>`. The tradeoff is visible in the starting sequence: Cyrillic characters take two bytes, and many emoji take four. The merge list must first reconstruct common characters before it can learn morphemes and words.

OpenAI’s public `tiktoken` family grew from `r50k_base` to `cl100k_base` and then `o200k_base`. A larger vocabulary can shorten target-domain sequences, but it also enlarges embeddings and creates more rows that must receive enough training signal.

In August 2025, the open gpt-oss release introduced the Harmony response format and the `o200k_harmony` encoding. Harmony reuses `o200k_base` content ranks and adds a large special-ID range for messages, channels, calls, and future protocol extensions. The official mapping in `tiktoken` assigns `gpt-oss-*` to `o200k_harmony`; `gpt-5-*` remains mapped to `o200k_base`.

A separate research branch avoids a fixed subword vocabulary. MEGABYTE, MambaByte, Byte Latent Transformer, and related systems model bytes or form dynamic patches. As of 5 August 2026, mainstream text serving still assumes token sequences throughout KV caching, prompt caching, structured generation, and speculative decoding, so replacing the tokenizer is a system migration rather than a local preprocessing change.

## 3. Compression is the right analogy—with limits

BPE shortens a sequence by replacing recurring substrings with one ID. That is a form of dictionary compression. A convenient engineering measure is

$$
\text{bytes per token}=\frac{N_{\text{input bytes}}}{N_{\text{tokens}}}.
$$

Higher values mean that each model position covers more source bytes. For LLM work this matters directly because attention, KV state, and token billing scale with sequence length.

It is not a complete bit-level compression ratio. If token IDs were stored with a fixed-width code, the payload would also depend on the vocabulary size:

$$
\text{bits per input byte}
\approx
\frac{N_{\text{tokens}}\log_2 V}{N_{\text{bytes}}}.
$$

A complete archive would additionally store the vocabulary, merge ranks, and metadata. In an LLM, however, the primary concern is usually the number of model positions rather than the smallest serialized file.

Information theory gives a precise statement for probabilistic source coding: using a mismatched distribution incurs an excess codelength related to **KL—Kullback–Leibler divergence**. A deterministic BPE segmentation is not arithmetic coding under an explicit source distribution. The train/held-out gap we measure later is therefore an engineering analogue of distribution mismatch, not a direct KL estimate.

> **Further study.** Entropy, Kraft’s inequality, source coding, and KL divergence are developed in *Information Theory for ML*, Modules 2 and 6.

Embeddings provide another classical connection. Skip-gram word2vec can be interpreted as implicit factorization of a **PMI—Pointwise Mutual Information** matrix. A Transformer embedding table preserves the ID-to-vector interface, while context-dependent meaning is produced by the network above it.

## 4. The tokenizer as a deterministic program

A tokenizer exposes two maps:

$$
\operatorname{encode}:\text{str}\rightarrow[0,V)^*,
\qquad
\operatorname{decode}:[0,V)^*\rightarrow\text{str}.
$$

For ordinary text we expect round-trip preservation:

$$
\operatorname{decode}(\operatorname{encode}(s))=s.
$$

### Pre-tokenization

Before pair merging, a regular expression partitions the string into chunks. Merges are not allowed to cross chunk boundaries. The teaching implementation uses a pattern close to `cl100k_base`:

```text
'(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?+\p{L}+ | \p{N}{1,3} | ?[^\s\p{L}\p{N}]++[\r\n]* | \s*[\r\n] | \s+(?!\S) | \s+
```

It separates English clitics, words with optional leading whitespace, short digit groups, punctuation runs, line breaks, and other whitespace. The actual `o200k_base` expression is more elaborate: it has separate Unicode branches for case patterns and different whitespace details. The teaching pattern captures the mechanism; it is not a bit-exact copy of every GPT tokenizer.

Digit groups of one to three characters make a string such as `1234567` segment as `123|456|7` before BPE. This regularity can help numerical learning, but it is neither necessary nor sufficient for arithmetic competence.

### BPE training

After pre-tokenization, each chunk is represented as base symbols. At each merge step, the most frequent adjacent pair is selected:

$$
(a,b)^*=\arg\max_{(a,b)}\operatorname{count}(a,b).
$$

All non-overlapping occurrences are replaced with a new symbol. With a byte alphabet,

$$
V=256+N_{\text{merges}}+N_{\text{unique special IDs}}.
$$

Tie-breaking matters. Two trainers can see the same counts, choose different equal-frequency pairs, and produce incompatible vocabularies.

### Encoding

A new string follows the same pre-tokenizer and byte conversion. Available merges are applied by rank. In `tiktoken`, lower rank means higher priority; the encoder repeatedly merges the lowest-ranked pair present in the chunk.

### Byte-level guarantees and edge cases

A byte alphabet eliminates OOV because any UTF-8 input can remain as individual bytes. It does not guarantee that every arbitrary token-ID prefix decodes to complete UTF-8. A streaming decoder may have to retain an incomplete byte suffix until more tokens arrive, and every implementation must define its error behavior.

The useful metrics—bytes/token, characters/token, fertility, or single-token retention—must always be reported on a named corpus.

## 5. Worked example: from UTF-8 to a compression curve

The complete path is shown first.

![VIZ m2/01 — text to model vectors](assets/modern-llms/en/module-02/m2_01_pipeline.svg)

### 5.1. The byte starting point

| String | Characters | UTF-8 bytes | Hex bytes |
|---|---:|---:|---|
| `Привет` | 6 | **12** | `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` |
| `é` | 1 | 2 | `C3 A9` |
| `🚀` | 1 | 4 | `F0 9F 9A 80` |
| `ток` | 3 | 6 | `D1 82 D0 BE D0 BA` |

![VIZ m2/02 — UTF-8 anatomy](assets/modern-llms/en/module-02/m2_02_utf8_anatomy.svg)

Before merges, `Hello` has five byte tokens and `Привет` has twelve. Vocabulary training can reduce the gap, but the byte representation determines the starting point.

### 5.2. Six visible merge steps

The toy corpus is `низко низко летали ласточки, ласточки летали низко`, totaling 93 bytes after chunking. The first six merges are:

| Step | Pair | Count | Resulting fragment |
|---:|---|---:|---|
| 1 | `D0` + `B8` | 7 | Cyrillic `и` |
| 2 | space + `D0` | 6 | space plus first Cyrillic byte |
| 3 | `D0` + `BA` | 5 | `к` |
| 4 | `D0` + `BE` | 5 | `о` |
| 5 | `(space+D0)` + `BB` | 4 | `' л'` |
| 6 | `' л'` + `D0` | 4 | prefix of a longer fragment |

After six merges, 93 bytes become **62 tokens**. The list first reconstructs characters and then begins to form word-initial pieces. Intermediate symbols remain in the vocabulary even when longer merges later make them rare.

![VIZ m2/03 — six BPE merges](assets/modern-llms/en/module-02/m2_03_bpe_trace.svg)

### 5.3. Training distribution versus held-out distribution

We train on the frozen Russian Module-2 corpus of 99,925 bytes and evaluate on a separate 68,070-byte English corpus.

| Merges | RU train bytes/token | EN held-out bytes/token | RU phrase | EN phrase |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 61 | 26 |
| 64 | 1.761 | 1.075 | 30 | 24 |
| 256 | 2.365 | 1.319 | 21 | 19 |
| 1024 | 3.371 | 1.701 | 15 | 15 |
| 4096 | **5.151** | **2.169** | **11** | **12** |

![VIZ m2/04 — compression curve](assets/modern-llms/en/module-02/m2_04_compression_curve.png)

The curve rises quickly while characters and frequent fragments are learned, then flattens as new merges apply to rarer strings. The 5.151 versus 2.169 gap is specialization to the training distribution, not an intrinsic ranking of Russian and English.

### 5.4. Four domains, one tokenizer

| Domain | Tokens | Words | Fertility | Bytes/token |
|---|---:|---:|---:|---:|
| Russian prose | 20 | 7 | **2.86** | 5.55 |
| English prose | 24 | 8 | 3.00 | 2.17 |
| Python | 31 | 13 | 2.38 | 2.19 |
| Numbers and dates | 22 | 6 | **3.67** | 1.36 |

The small Russian corpus did not contain enough recurring number strings to learn efficient numeric pieces. Production tokenizers often address this with pre-tokenization rules, but no fixed grouping is universally optimal.

## 6. Level-1 code: a transparent BBPE trainer

The core implementation deliberately uses a cl100k-like teaching pattern rather than claiming bit-level equivalence with `o200k_base`.

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
    result = []
    for chunk in ids:
        merged, i = [], 0
        while i < len(chunk):
            if i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(chunk[i])
                i += 1
        result.append(merged)
    return result

def train_bbpe(text, num_merges=1000):
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

The naive trainer rescans the corpus at every step, so its cost is $O(N_{\text{merges}}N_{\text{corpus}})$. Industrial trainers update pair statistics incrementally. The code also exposes its tie-break: Python’s insertion order decides between equal counts, making the choice part of the tokenizer specification.

## 8. Reading `o200k_harmony` correctly

### 8.1. Pre-tokenization is part of the format

The teaching regex illustrates the design space, but `o200k_base` uses a distinct, more detailed Unicode pattern. It has multiple letter branches for case structure and its own whitespace behavior. Harmony inherits the `o200k_base` pattern and mergeable ranks unchanged.

This distinction is operationally important: a tokenizer is not only a vocabulary size and merge table. The regex determines which merges are even possible.

### 8.2. Names, IDs, and one alias

A `.tiktoken` file stores `base64(token_bytes) rank`. In the public implementation:

- `o200k_base` provides **199,998 mergeable IDs** numbered `0…199997`;
- Harmony exposes **1,091 special-token names**;
- those names map to **1,090 unique special IDs**;
- ten names have current protocol meanings;
- 1,081 names are spelled as reserved entries;
- `<|endofprompt|>` and `<|reserved_200018|>` are aliases for ID 200018.

Therefore,

$$
199{,}998+1{,}090=201{,}088
$$

unique IDs. The difference between special names and special IDs is explained entirely by the alias; no rounding or unexplained gap is involved.

![VIZ m2/05 — the Harmony ID layout](assets/modern-llms/en/module-02/m2_05_harmony_layout.svg)

### 8.3. Special-token policy is explicit

`tiktoken.Encoding.encode()` rejects registered special-token spellings by default. A caller must choose one of three behaviors:

- authorize selected spellings with `allowed_special`;
- disable the rejection with `disallowed_special=()` and treat them as ordinary text;
- use `encode_ordinary()`.

Thus a literal `<|start|>` neither silently becomes a control ID nor necessarily passes as ordinary text. The API call defines the boundary.

### 8.4. Hugging Face artifacts

A BPE `tokenizer.json` typically contains `model.vocab`, `model.merges`, pre-tokenizer configuration, and added tokens. `tokenizer_config.json` may also contain a Jinja chat template. The template inserts protocol structure around messages; it builds on the tokenizer but is a separate layer discussed in Module 11.

## 9. A documented vocabulary landscape

Vocabulary sizes should be read from an artifact, configuration, or official API—not inferred from a family name.

| Family or artifact | Published size | Representation | Evidence |
|---|---:|---|---|
| GPT-5 API family | 200,019 | `o200k_base` BBPE | official `tiktoken` mapping |
| gpt-oss | **201,088** | `o200k_harmony` BBPE | official code and config |
| Llama 3.1 | 128,256 | tiktoken-based BPE | published tokenizer/config |
| Llama 4 | **202,048** | tiktoken-based BPE | `Llama4TextConfig` |
| DeepSeek V4 | 129,280 | published BPE tokenizer | official config |
| Qwen3 | 151,936 | BPE | published config/tokenizer |
| Gemma 3 | 262,144 | SentencePiece family | published artifact/config |
| Mistral Small 3.1 | 131,072 | Tekken/BPE | official `params.json` |
| Kimi K3 | 163,840 | custom BPE | official config |
| GLM-5 | 154,880 | published tokenizer/config | official config |
| Command A | 256,000 | published tokenizer/config | open-weight artifact |
| Nemotron 3 | 131,072 | NemotronH config | official config |
| Claude API | local artifact not published | server-side tokenizer | token-count endpoint |
| Gemini API | local artifact not published | server-side tokenizer | `models.countTokens` |

![VIZ m2/06 — published vocabulary sizes](assets/modern-llms/en/module-02/m2_06_vocab_landscape.png)

The corrections are instructive. Llama 4 does not simply retain Llama 3.1’s 128,256-entry vocabulary; its text config reports 202,048. Mistral Small 3.1 reports 131,072 rather than the older 32K scale. A closed Gemini tokenizer should not be assigned Gemma’s vocabulary by analogy.

Claude and Gemini both provide official counting endpoints. These endpoints can account for the real request structure, including tools or multimodal content, but they do not expose a local merge table. Counts are also model-version-specific, so an application should recount using the exact model it will call.

Tokenizer-free research should be described carefully. Systems such as BLT eliminate a fixed subword vocabulary but still form dynamic computational patches. The meaningful comparison includes sequence length, compute, robustness, and compatibility with the serving stack.

## 10. Fertility, language, and domain

**Fertility** is tokens per word; **bytes/token** is source bytes per token. Both require a named evaluation corpus. Fertility additionally requires a definition of “word,” which is not neutral across writing systems.

![VIZ m2/07 — domain fertility](assets/modern-llms/en/module-02/m2_07_fertility_domains.png)

### Cross-lingual evaluation

Published studies find large token-count differences for parallel content. A defensible report records the corpus, tokenizer revision, chat formatting, word definition, and price schedule. If the same provider charges the same input rate per token, three times as many input tokens mean three times the input-token charge—not necessarily three times the entire interaction cost.

### Numbers

BPE and Unigram can both be deterministic once trained. They differ in how the vocabulary is learned and how segmentation is selected. Digit-group regexes reduce segmentation variability; they do not guarantee arithmetic reasoning.

### Code

A code-oriented tokenizer can learn indentation patterns, operators, and frequent identifier fragments. Its benefit depends on the pretraining data as well as the vocabulary. Adding a token that rarely appears during model training creates an undertrained embedding row rather than an automatic capability.

### Undertrained tokens

The tokenizer’s corpus and the model’s corpus are not always identical. Rare vocabulary entries may receive very few updates, producing poorly calibrated input or output embeddings. Detecting them requires occurrence statistics and model-weight analysis, not only the merge rank.

## 11. Embedding cost and weight tying

The input table is

$$E\in\mathbb R^{V\times d},$$

and the output projection is

$$U\in\mathbb R^{d\times V}.$$

With untied weights, the model pays for both matrices.

| Scenario | One table | Two tables | BF16 payload for two |
|---|---:|---:|---:|
| gpt-oss, `201088 × 2880` | 579M params | 1.158B | 2.16 GiB |
| Llama 3.1 8B, `128256 × 4096` | 525M | 1.051B | 1.96 GiB |
| Increase 100K → 262K at `d=4096` | +663M | +1.327B | +2.47 GiB |

Weight tying sets $U=E^\top$, saving $Vd$ parameters. It is not simply “for small models” or “for old models”; it is an architectural choice documented in configuration.

| Configuration | Tied? |
|---|---:|
| Llama 1/2/3 | no |
| Gemma / Gemma 3 | yes |
| Llama 4 | no |
| DeepSeek V4 | no |
| Kimi K3 | no |
| Command A | yes (`use_embedding_sharing`) |

![VIZ m2/08 — tied and untied embeddings](assets/modern-llms/en/module-02/m2_08_tied_untied.svg)

A closed API does not reveal this choice through pricing or output behavior. Configuration or weights are required.

## 12. Where the token analogy extends to other modalities

A Vision Transformer divides an image into patches and projects every patch to a vector. This resembles pre-tokenization but usually does not assign a discrete vocabulary ID to each input patch. An 896×896 image with 14×14 patches starts with $64\times64=4096$ patch positions before merging or compression.

Neural audio codecs such as EnCodec and SoundStream use **RVQ—Residual Vector Quantization**. Several codebooks convert audio into discrete codec IDs and a decoder reconstructs the waveform. This is closer to text tokenization because the representation is a sequence drawn from finite codebooks.

![VIZ m2/09 — representations across modalities](assets/modern-llms/en/module-02/m2_09_multimodal_tokens.svg)

Module 14 develops the full multimodal architecture, including connectors, dynamic image resolution, video, and streaming speech.

## 13. Connections to the rest of the course

Token IDs do not carry order by themselves; Module 3 introduces positional mechanisms. Message templates and protocol tokens return in Module 11. Visual-token memory is treated in Modules 9 and 14. Output-distribution entropy appears in Module 10.

> **Further study.** Source coding, KL divergence, and information-theoretic mismatch are developed in *Information Theory for ML*, Module 6.

## 16. Key takeaways

![VIZ m2/10 — tokenization in one page](assets/modern-llms/en/module-02/m2_10_cheatsheet.svg)

- Most text LLMs freeze a separately constructed tokenizer before model training.
- BBPE can represent every UTF-8 string, but not with equal efficiency across distributions.
- Pre-tokenization constrains which merges may exist.
- Bytes/token is a sequence-length metric, not a complete bit-level compression ratio.
- Train/held-out differences reveal vocabulary specialization; the strict KL result belongs to probabilistic coding.
- Harmony has 199,998 mergeable IDs and 1,090 unique special IDs represented by 1,091 names.
- Special-token interpretation is an explicit API policy.
- Vocabulary size must be read from the actual model artifact or service contract.
- Fertility should be paired with bytes/token and measured on fixed data.
- Weight tying is family-specific: Llama and Gemma provide contrasting documented choices.

## 17. and sources

**Primary and technical sources:**

- Gage, *A New Algorithm for Data Compression* — [PDF](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)
- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units* — [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
- Kudo, Richardson, *SentencePiece* — [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
- OpenAI `tiktoken` — [official repository](https://github.com/openai/tiktoken)
- OpenAI Harmony — [official repository](https://github.com/openai/harmony)
- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* — [arxiv.org/abs/2305.15425](https://arxiv.org/abs/2305.15425)
- Press, Wolf, *Using the Output Embedding to Improve Language Models* — [arxiv.org/abs/1608.05859](https://arxiv.org/abs/1608.05859)
- Levy, Goldberg, *Neural Word Embedding as Implicit Matrix Factorization* — [arxiv.org/abs/1402.3722](https://arxiv.org/abs/1402.3722)
- Yu et al., *MEGABYTE* — [arxiv.org/abs/2305.07185](https://arxiv.org/abs/2305.07185)
- Pagnoni et al., *Byte Latent Transformer* — [arxiv.org/abs/2412.09871](https://arxiv.org/abs/2412.09871)
- Claude token-count endpoint — [official documentation](https://platform.claude.com/docs/en/api/messages/count_tokens)
- Gemini `models.countTokens` — [official documentation](https://ai.google.dev/api/tokens)

**Next:** Module 3 introduces position. Tokenization gives the model IDs, but self-attention still needs a mechanism that distinguishes one ordering from another.

---

*Landscape verified: 5 August 2026. Open vocabulary sizes were verified from published configurations or tokenizer artifacts; only official token-counting APIs are listed for closed Claude and Gemini tokenizers. Course calculations are locally reproducible, and closed tokenizers are not reconstructed from model behavior.*
