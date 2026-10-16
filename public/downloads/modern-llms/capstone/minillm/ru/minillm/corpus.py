"""Подготовка корпуса: фильтры и дедупликация (M9 §5).

Порядок стадий в конвейере не произволен — от дешёвых к дорогим. Эвристики
стоят проход по строке, дедуп стоит хеширование, модельный классификатор стоит
инференса. Поэтому модельный классификатор видит только то, что пережило
эвристики и дедуп, и стоимость конвейера определяется первыми стадиями.

Здесь реализованы первые четыре ступени; модельный классификатор качества
(FineWeb-Edu-стиль) в капстоун не входит — он требует отдельной обученной
модели и в учебном масштабе даёт больше шума, чем сигнала.

`pipeline()` возвращает воронку с числом документов на каждом шаге — ту самую,
которую рисует иллюстрация модуля 9.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field


CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """NFC, схлопывание пробелов, снятие управляющих символов.

    NFC важен для кириллицы: «й» бывает записана и одним кодпойнтом, и как «и»
    с комбинирующей краткой. Без нормализации это два разных слова для BPE.
    """
    text = unicodedata.normalize("NFC", text)
    text = CONTROL.sub("", text)
    text = WHITESPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


@dataclass
class FilterConfig:
    min_chars: int = 40
    max_chars: int = 100_000
    max_symbol_ratio: float = 0.3      # доля не-букв и не-пробелов
    max_line_repeat: float = 0.3       # доля повторяющихся строк внутри документа
    min_mean_word_len: float = 2.0
    max_mean_word_len: float = 12.0


def quality_reasons(doc: str, cfg: FilterConfig) -> list[str]:
    """Список причин отбраковки. Пустой список — документ проходит.

    Возвращаем именно причины, а не булев флаг: при отладке корпуса нужно знать,
    какая эвристика съела половину данных.
    """
    reasons = []
    if len(doc) < cfg.min_chars:
        reasons.append("too_short")
    if len(doc) > cfg.max_chars:
        reasons.append("too_long")
    letters = sum(1 for c in doc if c.isalpha() or c.isspace())
    if doc and 1 - letters / len(doc) > cfg.max_symbol_ratio:
        reasons.append("symbol_ratio")
    lines = [l for l in doc.split("\n") if l.strip()]
    if len(lines) > 3:
        unique = len(set(lines))
        if 1 - unique / len(lines) > cfg.max_line_repeat:
            reasons.append("line_repeat")
    words = doc.split()
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not (cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len):
            reasons.append("word_length")
    return reasons


def dedup_exact(docs: list[str]) -> list[str]:
    """Точный дедуп по SHA-256. Дёшево и снимает основную массу копий."""
    seen: set[str] = set()
    out = []
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append(doc)
    return out


def _shingles(doc: str, size: int = 5) -> set[str]:
    words = doc.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def minhash_signature(doc: str, permutations: int = 64, shingle: int = 5) -> tuple[int, ...]:
    """Сигнатура MinHash: минимальный хеш шингла по каждой из `permutations` схем.

    Оценка сходства Жаккара — доля совпавших позиций сигнатуры. Точный подсчёт
    пересечений множеств стоил бы O(n²) сравнений документов; MinHash сводит это
    к сравнению коротких векторов, а LSH-бэндинг — к поиску по словарю.
    """
    grams = _shingles(doc, shingle)
    if not grams:
        return tuple([0] * permutations)
    signature = []
    for p in range(permutations):
        salt = str(p).encode()
        signature.append(min(int.from_bytes(
            hashlib.blake2b(salt + g.encode("utf-8"), digest_size=8).digest(), "big")
            for g in grams))
    return tuple(signature)


def jaccard_estimate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def minhash_dedup(docs: list[str], threshold: float = 0.8, permutations: int = 64,
                  bands: int = 16, shingle: int = 5) -> list[str]:
    """Near-dup дедуп через MinHash + LSH-бэндинг (M9 §5).

    Сигнатура режется на `bands` полос; два документа становятся кандидатами,
    если совпала хотя бы одна полоса целиком. Кандидаты затем проверяются
    оценкой Жаккара по полной сигнатуре — бэндинг лишь сокращает перебор.
    """
    if permutations % bands:
        raise ValueError("permutations должно делиться на bands")
    rows = permutations // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    signatures = [minhash_signature(d, permutations, shingle) for d in docs]

    keep, dropped = [], set()
    for i, sig in enumerate(signatures):
        candidates: set[int] = set()
        band_keys = [(b, sig[b * rows:(b + 1) * rows]) for b in range(bands)]
        for key in band_keys:
            candidates.update(buckets.get(key, ()))
        if any(jaccard_estimate(sig, signatures[j]) >= threshold
               for j in candidates if j not in dropped):
            dropped.add(i)
            continue
        for key in band_keys:
            buckets.setdefault(key, []).append(i)
        keep.append(i)
    return [docs[i] for i in keep]


@dataclass
class PipelineReport:
    stages: list[tuple[str, int]] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, count: int) -> None:
        self.stages.append((name, count))

    @property
    def survival(self) -> float:
        return self.stages[-1][1] / max(self.stages[0][1], 1)

    def __str__(self) -> str:
        lines = ["воронка конвейера:"]
        first = max(self.stages[0][1], 1)
        for name, count in self.stages:
            lines.append(f"  {name:<20} {count:>8,}  ({count / first:.1%})")
        if self.reasons:
            lines.append("причины отбраковки: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.reasons.items(), key=lambda x: -x[1])))
        return "\n".join(lines)


def pipeline(docs: list[str], cfg: FilterConfig | None = None,
             near_dup_threshold: float = 0.8) -> tuple[list[str], PipelineReport]:
    """Полный проход: нормализация → эвристики → точный дедуп → near-dup."""
    cfg = cfg or FilterConfig()
    report = PipelineReport()
    report.add("исходные", len(docs))

    docs = [normalize(d) for d in docs]
    report.add("нормализация", len(docs))

    survivors = []
    for doc in docs:
        reasons = quality_reasons(doc, cfg)
        if reasons:
            for r in reasons:
                report.reasons[r] = report.reasons.get(r, 0) + 1
        else:
            survivors.append(doc)
    report.add("эвристики", len(survivors))

    survivors = dedup_exact(survivors)
    report.add("точный дедуп", len(survivors))

    survivors = minhash_dedup(survivors, threshold=near_dup_threshold)
    report.add("near-dup дедуп", len(survivors))
    return survivors, report
