"""Byte-level BPE from scratch—the capstone tokenizer (M2).

Why byte-level? The base alphabet is not the hundreds of thousands of Unicode
characters but 256 bytes. Any string can be encoded without OOV tokens, and
learned merges address sequence length. The cost is known and quantified in the
course: before the first merge, every Cyrillic letter occupies two tokens. A
Russian vocabulary is therefore only as good as the amount of Russian text in
its training corpus.

Implemented here:
  regex pre-tokenization .................. M2 §4
  frequency-based merge training .......... M2 §4, §6
  special tokens and allowed_special ...... M2 §8
  fertility as a vocabulary metric ........ M2 §10

An explicit limitation: training is pure Python with incremental pair-count
updates. This is sufficient for tens of megabytes; a real pretraining corpus
needs Hugging Face `tokenizers`. The module 2 notebook compares both.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

try:  # \p{L} and \p{N} are available only in the third-party regex package
    import regex as _re
    SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    _HAS_REGEX = True
except ImportError:  # pragma: no cover - environment-dependent
    import re as _re
    # Standard-re fallback: Python's \w is Unicode-aware, so Cyrillic and Latin
    # split consistently. Digit and punctuation handling differs from the GPT-4
    # pattern, which is acceptable for an educational vocabulary.
    SPLIT_PATTERN = r"'(?i:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"
    _HAS_REGEX = False


class BBPETokenizer:
    """A trained byte-level BPE tokenizer.

    Its contract matches `ByteTokenizer` from `data.py`:
    `encode(str) -> list[int]`, `decode(list[int]) -> str`. Replacing the
    fallback tokenizer with a learned vocabulary therefore leaves the data
    pipeline unchanged.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 specials: dict[str, int] | None = None,
                 pattern: str = SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self.merges = [tuple(m) for m in (merges or [])]
        self.specials = dict(specials or {})
        self._compiled = _re.compile(self.pattern)
        self._rebuild()

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        # new token ID = 256 + merge index
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    # --------------------------------------------------------------------- sizes
    @property
    def base_size(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return self.base_size + len(self.specials)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] | None = None,
              pattern: str = SPLIT_PATTERN, verbose: bool = False) -> "BBPETokenizer":
        """Train merges up to the requested vocabulary size.

        `vocab_size` includes the 256 base bytes and all special tokens, just
        like the `vocab_size` field in a real model's config.json.
        """
        specials = list(specials or [])
        target_merges = vocab_size - 256 - len(specials)
        if target_merges < 0:
            raise ValueError(f"vocab_size={vocab_size} меньше 256 + {len(specials)} "
                             "специальных токенов")

        compiled = _re.compile(pattern)
        chunk_freqs = Counter(compiled.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_freqs]
        freqs = list(chunk_freqs.values())

        pair_counts: Counter = Counter()
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)

        def add_word(i: int, sign: int = 1) -> None:
            word, f = words[i], freqs[i]
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += sign * f
                if sign > 0:
                    pair_words[(a, b)].add(i)

        for i in range(len(words)):
            add_word(i)

        merges: list[tuple[int, int]] = []
        for step in range(target_merges):
            # Candidates with zero or negative counts are stale after merging.
            best, best_count = None, 0
            for pair, count in pair_counts.items():
                if count > best_count:
                    best, best_count = pair, count
            if best is None:
                if verbose:
                    print(f"  no pairs left at merge {step}")
                break

            new_id = 256 + len(merges)
            merges.append(best)
            affected = list(pair_words.get(best, ()))
            for i in affected:
                word = words[i]
                if len(word) < 2:
                    continue
                add_word(i, sign=-1)          # remove the word's old contribution
                merged, j = [], 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(word[j])
                        j += 1
                words[i] = merged
                add_word(i, sign=+1)          # add its new contribution
            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            if verbose and (step + 1) % 256 == 0:
                print(f"  merge {step + 1}/{target_merges}: {best} -> {new_id}")

        special_ids = {name: 256 + len(merges) + i for i, name in enumerate(specials)}
        return cls(merges, special_ids, pattern)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            # Apply the lowest-rank merge. Encoding must reproduce training
            # order or vocabularies become incompatible.
            best = min(((self.ranks.get((a, b), float("inf")), i)
                        for i, (a, b) in enumerate(zip(ids, ids[1:]))),
                       default=(float("inf"), 0))
            rank, position = best
            if rank == float("inf"):
                break
            pair = (ids[position], ids[position + 1])
            new_id = 256 + self.ranks[pair]
            ids[position:position + 2] = [new_id]
        return ids

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encode text; special tokens are NOT recognized by default.

        This is the rule from module 2 §8: only runtime code may insert a
        control ID, never user text. Otherwise a literal `<|endoftext|>` in a
        user's message could silently terminate the dialogue.
        """
        allowed = set(allowed_special or ())
        invalid = allowed - set(self.specials)
        if invalid:
            raise ValueError(f"allowed_special содержит неизвестные токены: {sorted(invalid)}")
        unknown = [s for s in self.specials if s in text and s not in allowed]
        if unknown:
            raise ValueError(f"специальные токены в тексте без allowed_special: {unknown}")

        if not allowed:
            return [i for chunk in self._compiled.findall(text)
                    for i in self._encode_chunk(chunk.encode("utf-8"))]

        # Split on allowed special tokens and encode ordinary pieces separately.
        pieces: list[str] = [text]
        # Longest-first makes overlapping control strings deterministic.
        for token in sorted(allowed, key=lambda item: (-len(item), item)):
            nxt: list[str] = []
            for piece in pieces:
                if piece in self.specials:
                    nxt.append(piece)
                    continue
                parts = piece.split(token)
                for i, part in enumerate(parts):
                    if i:
                        nxt.append(token)
                    if part:
                        nxt.append(part)
            pieces = nxt

        out: list[int] = []
        for piece in pieces:
            if piece in self.specials:
                out.append(self.specials[piece])
            else:
                out.extend(i for chunk in self._compiled.findall(piece)
                           for i in self._encode_chunk(chunk.encode("utf-8")))
        return out

    def decode(self, ids: list[int]) -> str:
        inverse = {v: k for k, v in self.specials.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in inverse:
                parts.append(inverse[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise KeyError(f"идентификатор вне словаря: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- metrics
    def fertility(self, text: str) -> dict[str, float]:
        """Vocabulary fertility on a text sample (M2 §10).

        `tokens_per_char` is the primary cross-language metric. It is
        systematically higher for Russian when the vocabulary was trained on
        an English corpus, directly increasing context and KV-cache cost.
        """
        ids = self.encode(text)
        chars = max(len(text), 1)
        words = max(len(text.split()), 1)
        return {
            "tokens": len(ids),
            "chars": len(text),
            "tokens_per_char": round(len(ids) / chars, 4),
            "tokens_per_word": round(len(ids) / words, 4),
            "bytes_per_token": round(len(text.encode("utf-8")) / max(len(ids), 1), 4),
        }

    # ------------------------------------------------------------- serialization
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "pattern": self.pattern,
            "uses_regex_module": _HAS_REGEX,
            "merges": [list(m) for m in self.merges],
            "specials": self.specials,
            "vocab_size": self.vocab_size,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BBPETokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("uses_regex_module") != _HAS_REGEX:
            # Different patterns produce different pre-tokenization and thus
            # different IDs for the same vocabulary. This must not be silent.
            raise RuntimeError(
                "словарь обучен с другим движком регулярных выражений: "
                f"файл uses_regex_module={blob.get('uses_regex_module')}, "
                f"текущее окружение {_HAS_REGEX}. Установите/удалите пакет "
                "`regex`, либо переобучите словарь.")
        return cls(blob["merges"], blob["specials"], blob["pattern"])
