"""Byte-level BPE с нуля — токенизатор капстоуна (M2).

Почему byte-level. Базовый алфавит — не символы Unicode (их сотни тысяч), а 256
байт. Любая строка кодируется без OOV, а длину лечат обученные merge'ы. Цена
известна и в курсе посчитана: до первого merge каждая кириллическая буква — два
токена, поэтому качество словаря для русского целиком определяется тем, было ли
в обучающем корпусе достаточно русского текста.

Что здесь реализовано:
  pre-tokenization регуляркой ............ M2 §4
  обучение merge'ов по частоте пар ....... M2 §4, §6
  специальные токены и allowed_special ... M2 §8
  fertility как метрика словаря .......... M2 §10

Ограничение, названное вслух: обучение — чистый Python с инкрементальным
пересчётом пар. Для корпуса в десятки мегабайт этого достаточно, для настоящего
претрейн-корпуса нужен `tokenizers` от HF. Ноутбук модуля 2 сравнивает оба.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

try:  # \p{L} и \p{N} доступны только в стороннем regex
    import regex as _re
    SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    _HAS_REGEX = True
except ImportError:  # pragma: no cover - зависит от окружения
    import re as _re
    # Запасной вариант на стандартном re: \w в Python юникодный, поэтому
    # кириллица и латиница разбиваются одинаково. Разница с GPT-4-паттерном —
    # в обработке цифр и знаков препинания; для учебного словаря приемлемо.
    SPLIT_PATTERN = r"'(?i:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"
    _HAS_REGEX = False


class BBPETokenizer:
    """Обученный byte-level BPE.

    Контракт совпадает с `ByteTokenizer` из `data.py`: `encode(str) -> list[int]`,
    `decode(list[int]) -> str`. Поэтому конвейер данных не меняется при замене
    заглушки на обученный словарь.
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
        # id нового токена = 256 + порядковый номер merge'а
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    # ------------------------------------------------------------------ размеры
    @property
    def base_size(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return self.base_size + len(self.specials)

    # ---------------------------------------------------------------- обучение
    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] | None = None,
              pattern: str = SPLIT_PATTERN, verbose: bool = False) -> "BBPETokenizer":
        """Учит merge'ы до заданного размера словаря.

        `vocab_size` считается вместе с 256 базовыми байтами и специальными
        токенами — так же, как поле `vocab_size` в config.json живой модели.
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
            # Кандидаты с нулевым или отрицательным счётом — мусор после слияний.
            best, best_count = None, 0
            for pair, count in pair_counts.items():
                if count > best_count:
                    best, best_count = pair, count
            if best is None:
                if verbose:
                    print(f"  пары кончились на merge {step}")
                break

            new_id = 256 + len(merges)
            merges.append(best)
            affected = list(pair_words.get(best, ()))
            for i in affected:
                word = words[i]
                if len(word) < 2:
                    continue
                add_word(i, sign=-1)          # снять старый вклад слова
                merged, j = [], 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(word[j])
                        j += 1
                words[i] = merged
                add_word(i, sign=+1)          # вернуть новый
            pair_counts.pop(best, None)
            pair_words.pop(best, None)
            if verbose and (step + 1) % 256 == 0:
                print(f"  merge {step + 1}/{target_merges}: {best} -> {new_id}")

        special_ids = {name: 256 + len(merges) + i for i, name in enumerate(specials)}
        return cls(merges, special_ids, pattern)

    # -------------------------------------------------------------- кодирование
    def _encode_chunk(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            # Применяем merge с наименьшим рангом — порядок обучения обязан
            # воспроизводиться при кодировании, иначе словари несовместимы.
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
        """Кодирует текст. Специальные токены по умолчанию НЕ распознаются.

        Это правило модуля 2 §8: управляющий идентификатор вправе вставить только
        код рантайма, а не пользовательский текст. Иначе строка `<|endoftext|>`
        в письме пользователя молча оборвала бы диалог.
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

        # Режем текст по разрешённым специальным токенам и кодируем куски отдельно.
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

    # ------------------------------------------------------------------ метрики
    def fertility(self, text: str) -> dict[str, float]:
        """Fertility словаря на тексте (M2 §10).

        `tokens_per_char` — главный показатель для сравнения языков: у русского
        он систематически выше при словаре, обученном на англоязычном корпусе,
        и это прямая надбавка к цене контекста и KV-кэша.
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

    # ------------------------------------------------------------ сериализация
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
            # Разные паттерны дают разную пре-токенизацию, а значит другие
            # идентификаторы при том же словаре. Молчать об этом нельзя.
            raise RuntimeError(
                "словарь обучен с другим движком регулярных выражений: "
                f"файл uses_regex_module={blob.get('uses_regex_module')}, "
                f"текущее окружение {_HAS_REGEX}. Установите/удалите пакет "
                "`regex`, либо переобучите словарь.")
        return cls(blob["merges"], blob["specials"], blob["pattern"])
