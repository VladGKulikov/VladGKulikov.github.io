"""Chat-шаблон, маскирование лосса и упаковка примеров (M13).

Три вещи, на которых спотыкаются чаще всего:

1. **Шаблон — это протокол, а не оформление.** Модель обучена на конкретной
   разметке ролей; чужой шаблон на инференсе не вызывает ошибки, он просто
   тихо роняет качество. Поэтому шаблон живёт рядом с весами, а не в коде
   приложения (M13 §4).

2. **Маска лосса решает, чему модель учится.** Учим только ответы ассистента,
   не учим системный промпт и реплики пользователя. И — ключевая деталь —
   **закрывающий `<|eot_id|>` входит в обучаемую часть**. Забыть его значит
   получить модель, которая генерирует бесконечно (M13 §5). Это воспроизводится
   флагом `train_eot=False` и разбирается в ноутбуке 2.

3. **Упаковка окупается сразу.** Диалоги короче окна, и без упаковки батч на
   90% состоит из padding. First-fit decreasing сокращает число бинов в разы
   (M13 §12).

Контракт форм: `encode_with_mask` возвращает `ids` длины T и `mask` той же
длины, где 1 — «этот токен модель обязана предсказать». `labels` получаются из
пары подстановкой −100 на месте нулей маски: именно это значение
`MiniLLM.forward` передаёт в `ignore_index`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import torch


class SupportsEncode(Protocol):
    specials: dict[str, int]

    def encode(self, text: str, allowed_special: set[str] | None = ...) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str


# Разметка в стиле Llama-3: заголовок роли, тело, закрывающий токен хода.
BOS = "<|begin_of_text|>"
HEADER_START = "<|start_header_id|>"
HEADER_END = "<|end_header_id|>"
EOT = "<|eot_id|>"
CHAT_SPECIALS = [BOS, HEADER_START, HEADER_END, EOT]
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass
class ChatTemplate:
    """Рендер диалога в текст и в пару (ids, маска обучаемых позиций)."""

    trainable_roles: tuple[str, ...] = ("assistant",)
    specials: tuple[str, ...] = field(default=tuple(CHAT_SPECIALS))

    # ------------------------------------------------------------------ рендер
    def header(self, role: str) -> str:
        if role not in VALID_ROLES:
            raise ValueError(f"неизвестная роль {role!r}; допустимы {sorted(VALID_ROLES)}")
        return f"{HEADER_START}{role}{HEADER_END}\n\n"

    def render(self, messages: Iterable[Message], add_generation_prompt: bool = False) -> str:
        parts = [BOS]
        for m in messages:
            parts.append(self.header(m.role))
            parts.append(m.content)
            parts.append(EOT)
        if add_generation_prompt:
            parts.append(self.header("assistant"))
        return "".join(parts)

    # ------------------------------------------------------- кодирование с маской
    def encode_with_mask(self, messages: Iterable[Message], tokenizer: SupportsEncode,
                         train_eot: bool = True,
                         add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
        """Возвращает (ids, mask). mask=1 там, где считается лосс.

        `train_eot=False` воспроизводит классическую ошибку: ответ учится, а
        сигнал остановки — нет. Модель после такого SFT генерирует, пока не
        упрётся в лимит токенов.
        """
        allowed = set(self.specials)
        ids: list[int] = []
        mask: list[int] = []

        def push_control(text: str, trainable: bool) -> None:
            """Только шаблон рантайма вправе создавать управляющие ID."""
            chunk = tokenizer.encode(text, allowed_special=allowed)
            ids.extend(chunk)
            mask.extend([1 if trainable else 0] * len(chunk))

        def push_content(text: str, trainable: bool) -> None:
            # Пользовательский или датасетный текст не получает разрешение на
            # специальные токены. BBPE выдаст явную ошибку вместо тихой инъекции
            # `<|eot_id|>` или поддельного заголовка роли.
            chunk = tokenizer.encode(text, allowed_special=set())
            ids.extend(chunk)
            mask.extend([1 if trainable else 0] * len(chunk))

        push_control(BOS, False)
        for m in messages:
            trainable = m.role in self.trainable_roles
            # Заголовок роли не учим никогда: его пишет рантайм, а не модель.
            push_control(self.header(m.role), False)
            push_content(m.content, trainable)
            push_control(EOT, trainable and train_eot)
        if add_generation_prompt:
            push_control(self.header("assistant"), False)
        return ids, mask

    def to_labels(self, ids: list[int], mask: list[int]) -> list[int]:
        if len(ids) != len(mask):
            raise ValueError(f"ids/mask length mismatch: {len(ids)} != {len(mask)}")
        return [i if m else -100 for i, m in zip(ids, mask)]

    def trainable_fraction(self, mask: list[int]) -> float:
        return sum(mask) / max(len(mask), 1)


# =============================================================================
# Упаковка (M13 §12)
# =============================================================================
def pack_first_fit_decreasing(lengths: list[int], capacity: int) -> list[list[int]]:
    """Раскладывает примеры по бинам ёмкости `capacity`. Возвращает индексы.

    FFD не оптимален (оптимальная упаковка NP-трудна), но даёт результат в
    пределах ~22% от оптимума и считается за один проход после сортировки.
    Примеры длиннее ёмкости отбрасываются вызывающим кодом — молча резать их
    нельзя, это испортило бы разметку ролей.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    remaining: list[int] = []
    for i in order:
        if lengths[i] > capacity:
            raise ValueError(f"пример {i} длиной {lengths[i]} не влезает в окно {capacity}")
        for b, free in enumerate(remaining):
            if free >= lengths[i]:
                bins[b].append(i)
                remaining[b] -= lengths[i]
                break
        else:
            bins.append([i])
            remaining.append(capacity - lengths[i])
    return bins


@dataclass
class SFTDataset:
    """Packed SFT rows with strict cross-dialogue isolation.

    ``segment_ids`` assigns a separate segment to every dialogue in a packed row
    and ``-1`` to padding. MiniLLM uses it to build a block-diagonal causal mask
    and reset RoPE positions inside each segment. EOT remains part of the learned
    protocol, but correctness no longer relies on the model voluntarily ignoring
    an earlier unrelated dialogue.
    """

    ids: torch.Tensor          # [N, seq_len]
    labels: torch.Tensor       # [N, seq_len]
    segment_ids: torch.Tensor  # [N, seq_len], -1 on padding
    seq_len: int

    @classmethod
    def from_dialogues(cls, dialogues: list[list[Message]], tokenizer: SupportsEncode,
                       template: ChatTemplate, seq_len: int, pad_id: int,
                       train_eot: bool = True) -> tuple["SFTDataset", dict]:
        if not dialogues:
            raise ValueError("SFT requires at least one dialogue")
        encoded = [template.encode_with_mask(d, tokenizer, train_eot) for d in dialogues]
        kept = [(i, m) for i, m in encoded if len(i) <= seq_len]
        dropped = len(encoded) - len(kept)
        if not kept:
            raise ValueError("all SFT dialogues are longer than seq_len")
        lengths = [len(i) for i, _ in kept]
        bins = pack_first_fit_decreasing(lengths, seq_len)

        rows, label_rows, segment_rows = [], [], []
        for b in bins:
            row_ids, row_labels, row_segments = [], [], []
            for local_segment, index in enumerate(b):
                sample_ids, sample_mask = kept[index]
                row_ids.extend(sample_ids)
                row_labels.extend(template.to_labels(sample_ids, sample_mask))
                row_segments.extend([local_segment] * len(sample_ids))
            pad = seq_len - len(row_ids)
            rows.append(row_ids + [pad_id] * pad)
            label_rows.append(row_labels + [-100] * pad)
            segment_rows.append(row_segments + [-1] * pad)

        stats = {
            "dialogues": len(dialogues),
            "dropped_too_long": dropped,
            "bins": len(bins),
            "packing_ratio": round(len(kept) / max(len(bins), 1), 2),
            "padding_fraction": round(
                1 - sum(lengths) / max(len(bins) * seq_len, 1), 4),
            "trainable_fraction": round(
                sum(sum(1 for x in template.to_labels(i, m) if x != -100)
                    for i, m in kept) / max(sum(lengths), 1), 4),
            "cross_dialogue_isolation": "block_diagonal_attention",
        }
        return cls(torch.tensor(rows), torch.tensor(label_rows),
                   torch.tensor(segment_rows), seq_len), stats

    def __len__(self) -> int:
        return self.ids.shape[0]

    def batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[indices], self.labels[indices], self.segment_ids[indices]

    def loader(self, batch_size: int, seed: int = 0, shuffle: bool = True,
               start_batch: int = 0):
        """Same deterministic cursor contract as TokenStream."""
        n = len(self)
        per_epoch = n // batch_size
        if per_epoch == 0:
            raise ValueError(f"batch_size={batch_size} больше числа бинов {n}")
        global_batch = start_batch
        while True:
            epoch, within = divmod(global_batch, per_epoch)
            if shuffle:
                gen = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
                order = torch.randperm(n, generator=gen)
            else:
                order = torch.arange(n)
            for i in range(within, per_epoch):
                yield self.batch(order[i * batch_size:(i + 1) * batch_size])
                global_batch += 1

