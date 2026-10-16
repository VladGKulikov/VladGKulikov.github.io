"""Задача с управляемой дальней зависимостью для ноутбука 4.

Случайная последовательность повторяется дважды; обучаемые цели находятся
во второй половине. Длина половины задаёт расстояние до соответствующего
элемента первой. Точность измеряет решение задачи копирования, но сама
по себе не устанавливает наличие конкретной induction head.

sample возвращает ids и labels формы [B, 2n]. Первая половина и первый
токен второй половины исключены из loss меткой -100. Сдвиг целей выполняет
модель, а repeat_accuracy сопоставляет предсказание позиции t-1 с целью t.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import MiniLLM


@dataclass
class RepeatTask:
    """Задача «повтори последовательность»: `[x_1..x_n, x_1..x_n]`.

    Раскладка:

        позиции 0..n-1     — случайный префикс, лоссом не учитывается
        позиция  n         — начало повтора, исключено из обучаемых целей
        позиции n+1..2n-1  — обучаемые; цель сопоставляется после каузального сдвига

    `half_len` при обучении выбирается случайно из `train_range`, что заставляет
    модель работать с разными расстояниями, а не запоминать одно.
    """

    vocab_size: int
    train_range: tuple[int, int] = (16, 64)
    seed: int = 0

    def sample(self, batch_size: int, generator: torch.Generator | None = None,
               half_len: int | None = None):
        gen = generator or torch.Generator().manual_seed(self.seed)
        if half_len is None:
            low, high = self.train_range
            half_len = int(torch.randint(low, high + 1, (1,), generator=gen))
        prefix = torch.randint(0, self.vocab_size, (batch_size, half_len),
                               generator=gen)
        ids = torch.cat([prefix, prefix], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, half_len + 1:] = ids[:, half_len + 1:]
        return ids, labels

    # ---------------------------------------------- интерфейс потока для train()
    def __len__(self) -> int:
        return 1 << 20          # поток бесконечный; число нужно только loader'у

    def batch(self, indices: torch.Tensor):
        gen = torch.Generator().manual_seed(self.seed + int(indices[0]))
        return self.sample(len(indices), generator=gen)

    def loader(self, batch_size: int, seed: int = 0, shuffle: bool = True,
               start_batch: int = 0):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + start_batch)
        while True:
            yield self.sample(batch_size, generator=gen)


@torch.no_grad()
def repeat_accuracy(model: MiniLLM, task: RepeatTask, half_len: int,
                    batches: int = 8, batch_size: int = 16,
                    device: str = "cpu", seed: int = 1234) -> float:
    """Доля верно восстановленных токенов второй половины.

    Accuracy оценивает восстановление повторённого текста, но не определяет,
    каким внутренним механизмом модель решила задачу. Кросс-энтропия дополнительно
    учитывает вероятности целевых токенов.
    """
    was_training = model.training
    model.eval()
    hits = total = 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(batches):
        ids, labels = task.sample(batch_size, generator=gen, half_len=half_len)
        ids = ids.to(device)
        logits, _ = model(ids)
        # Предсказание позиции t делается из позиции t-1.
        predicted = logits[:, :-1].argmax(-1).cpu()
        target = labels[:, 1:]
        mask = target != -100
        hits += int((predicted[mask] == target[mask]).sum())
        total += int(mask.sum())
    if was_training:
        model.train()
    return hits / max(total, 1)


def chance_level(task: RepeatTask) -> float:
    """Уровень случайного угадывания — базовая линия для accuracy."""
    return 1.0 / task.vocab_size
