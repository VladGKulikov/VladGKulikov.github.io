"""A controllable long-range dependency task for Notebook 4.

A random sequence is repeated twice, with supervised targets in its second
half. Half-length determines the distance to the corresponding prefix token.
Accuracy measures copying-task performance; it does not by itself establish
the presence of a particular induction head.

sample returns ids and labels of shape [B, 2n]. The first half and the first
token of the second half are excluded with -100. The model shifts training
targets; repeat_accuracy compares the prediction at t-1 with the target at t.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import MiniLLM


@dataclass
class RepeatTask:
    """The “repeat the sequence” task: `[x_1..x_n, x_1..x_n]`.

    Layout:

        positions 0..n-1     — random prefix, excluded from the loss
        position  n          — start of repetition, also not trained
        positions n+1..2n-1  — trainable; each token is scored after a causal shift

    During training, `half_len` is sampled from `train_range`, forcing the
    model to handle different distances rather than memorize one.
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

    # ------------------------------------------------ stream interface for train()
    def __len__(self) -> int:
        return 1 << 20          # infinite stream; only the loader needs a length

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
    """Fraction of correctly reconstructed tokens in the second half.

    Accuracy measures recovery of repeated text without identifying the
    internal mechanism that solved the task. Cross-entropy also reflects
    the probabilities assigned to target tokens.
    """
    was_training = model.training
    model.eval()
    hits = total = 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(batches):
        ids, labels = task.sample(batch_size, generator=gen, half_len=half_len)
        ids = ids.to(device)
        logits, _ = model(ids)
        # Position t is predicted from position t-1.
        predicted = logits[:, :-1].argmax(-1).cpu()
        target = labels[:, 1:]
        mask = target != -100
        hits += int((predicted[mask] == target[mask]).sum())
        total += int(mask.sum())
    if was_training:
        model.train()
    return hits / max(total, 1)


def chance_level(task: RepeatTask) -> float:
    """Chance-level accuracy baseline."""
    return 1.0 / task.vocab_size
