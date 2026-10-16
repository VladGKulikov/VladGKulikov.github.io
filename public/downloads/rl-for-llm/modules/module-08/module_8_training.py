"""Realistic small-scale autoregressive DPO experiment for Module 8.

The implementation is intentionally compact enough for CPU execution while
preserving the contracts that matter in real LLM preference optimization:

* one prompt and two complete responses per pair;
* a frozen reference policy and a trainable policy initialized from it;
* causal token alignment and completion-only masks;
* sequence log-probabilities, not independent response scores;
* DPO loss, pair accuracy, absolute chosen/rejected log-probabilities, and
  reference-drift diagnostics;
* deterministic generation-based evaluation on held-out prompt templates.

It is not a benchmark and does not claim to predict the behavior of a large
language model.  The optional TRL/Qwen branch lives in the bilingual notebooks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PAD = "<pad>"
BOS = "<bos>"
SEP = "<sep>"
EOS = "<eos>"


@dataclass(frozen=True)
class PreferenceExample:
    prompt: tuple[str, ...]
    chosen: tuple[str, ...]
    rejected: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[SEP]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        missing = [token for token in tokens if token not in self.token_to_id]
        if missing:
            raise KeyError(f"unknown tokens: {missing}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(index)] for index in ids]


def toy_preference_examples() -> list[PreferenceExample]:
    """Return a small, diverse preference set with shared response structure."""
    rows = [
        (("task", "sum", "2", "3"), ("answer", "5"), ("answer", "4"), "arithmetic"),
        (("task", "sum", "3", "4"), ("answer", "7"), ("answer", "6"), "arithmetic"),
        (("task", "sky", "color"), ("answer", "blue"), ("answer", "green"), "factual"),
        (("task", "water", "state", "ice"), ("answer", "solid"), ("answer", "liquid"), "factual"),
        (("task", "tone", "request"), ("polite", "reply"), ("rude", "reply"), "style"),
        (("task", "concise", "request"), ("short", "reply"), ("long", "long", "reply"), "style"),
        (("task", "unsafe", "delete", "files"), ("refuse", "harmful"), ("comply", "harmful"), "safety"),
        (("task", "unsafe", "steal", "password"), ("refuse", "harmful"), ("comply", "harmful"), "safety"),
    ]
    return [
        PreferenceExample(tuple(prompt), tuple(chosen), tuple(rejected), label)
        for prompt, chosen, rejected, label in rows
    ]


def build_vocabulary(examples: Sequence[PreferenceExample]) -> Vocabulary:
    tokens = {PAD, BOS, SEP, EOS}
    for row in examples:
        tokens.update(row.prompt)
        tokens.update(row.chosen)
        tokens.update(row.rejected)
    ordered = [PAD, BOS, SEP, EOS] + sorted(tokens - {PAD, BOS, SEP, EOS})
    return Vocabulary({token: i for i, token in enumerate(ordered)}, tuple(ordered))


def encode_prompt_completion(
    vocab: Vocabulary,
    prompt: Sequence[str],
    completion: Sequence[str],
) -> tuple[list[int], int]:
    """Encode BOS + prompt + SEP + completion + EOS and return prompt length.

    ``prompt_length`` counts tokens up to and including SEP.  Completion loss
    starts at the first token after SEP and includes EOS.
    """
    prefix = [BOS, *prompt, SEP]
    sequence = [*prefix, *completion, EOS]
    return vocab.encode(sequence), len(prefix)


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    prompt_lengths: Sequence[int],
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if len(sequences) == 0 or len(sequences) != len(prompt_lengths):
        raise ValueError("aligned non-empty sequences and prompt_lengths required")
    max_len = max(len(row) for row in sequences)
    if max_len < 2:
        raise ValueError("each sequence must contain at least two tokens")
    ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    response_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.bool)
    for i, (row, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
        if not 1 <= prompt_len < len(row):
            raise ValueError("prompt length must leave at least one completion token")
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[i, : len(row)] = True
        # Shifted label position j corresponds to original token j+1.
        # Completion begins at original index prompt_len.
        response_mask[i, prompt_len - 1 : len(row) - 1] = True
    return ids, attention, response_mask


class TinyCausalLM(nn.Module):
    """A small causal GRU language model with an LLM-like token interface."""

    def __init__(self, vocab_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        hidden, _ = self.gru(self.embedding(input_ids))
        return self.lm_head(hidden)


def selected_token_logprobs(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Return causal log-probabilities for tokens 1..L-1, shape [B, L-1]."""
    logits = model(input_ids)
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("model logits must align with input_ids")
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def completion_sequence_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    response_mask: Tensor,
) -> Tensor:
    selected = selected_token_logprobs(model, input_ids)
    if response_mask.shape != selected.shape:
        raise ValueError("response_mask must match shifted token log-probabilities")
    return (selected * response_mask.to(selected.dtype)).sum(dim=-1)


def pair_batch(
    examples: Sequence[PreferenceExample],
    vocab: Vocabulary,
) -> dict[str, Tensor]:
    chosen_seq: list[list[int]] = []
    rejected_seq: list[list[int]] = []
    chosen_prompt_lengths: list[int] = []
    rejected_prompt_lengths: list[int] = []
    for row in examples:
        seq, prompt_len = encode_prompt_completion(vocab, row.prompt, row.chosen)
        chosen_seq.append(seq)
        chosen_prompt_lengths.append(prompt_len)
        seq, prompt_len = encode_prompt_completion(vocab, row.prompt, row.rejected)
        rejected_seq.append(seq)
        rejected_prompt_lengths.append(prompt_len)
    chosen_ids, chosen_attention, chosen_mask = pad_sequences(
        chosen_seq, chosen_prompt_lengths, vocab.pad_id
    )
    rejected_ids, rejected_attention, rejected_mask = pad_sequences(
        rejected_seq, rejected_prompt_lengths, vocab.pad_id
    )
    return {
        "chosen_ids": chosen_ids,
        "chosen_attention": chosen_attention,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_attention": rejected_attention,
        "rejected_mask": rejected_mask,
    }


def dpo_loss_from_logprobs(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if not (
        policy_chosen.shape
        == policy_rejected.shape
        == reference_chosen.shape
        == reference_rejected.shape
    ):
        raise ValueError("all pair log-probability tensors must share a shape")
    margin = (policy_chosen - reference_chosen) - (
        policy_rejected - reference_rejected
    )
    logits = float(beta) * margin
    losses = F.softplus(-logits)
    return losses.mean(), margin, logits


def sft_loss_on_sequences(
    model: nn.Module,
    ids: Tensor,
    response_mask: Tensor,
) -> Tensor:
    selected = selected_token_logprobs(model, ids)
    mask = response_mask.to(selected.dtype)
    denom = mask.sum().clamp_min(1.0)
    return -(selected * mask).sum() / denom


def train_reference_model(
    examples: Sequence[PreferenceExample],
    vocab: Vocabulary,
    *,
    hidden_size: int = 48,
    steps: int = 180,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> TinyCausalLM:
    """Train a reference LM on both candidates so DPO has room to choose."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = TinyCausalLM(len(vocab.id_to_token), hidden_size=hidden_size)
    batch = pair_batch(examples, vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        chosen_loss = sft_loss_on_sequences(model, batch["chosen_ids"], batch["chosen_mask"])
        rejected_loss = sft_loss_on_sequences(model, batch["rejected_ids"], batch["rejected_mask"])
        # Equal mixture models both candidate families; the preference stage decides.
        loss = 0.5 * (chosen_loss + rejected_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def pair_metrics(
    policy: nn.Module,
    reference: nn.Module,
    batch: dict[str, Tensor],
    beta: float,
) -> dict[str, float]:
    pc = completion_sequence_logprobs(policy, batch["chosen_ids"], batch["chosen_mask"])
    pr = completion_sequence_logprobs(policy, batch["rejected_ids"], batch["rejected_mask"])
    rc = completion_sequence_logprobs(reference, batch["chosen_ids"], batch["chosen_mask"])
    rr = completion_sequence_logprobs(reference, batch["rejected_ids"], batch["rejected_mask"])
    losses, margin, logits = dpo_loss_from_logprobs(pc, pr, rc, rr, beta)
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    return {
        "loss": float(losses),
        "pair_accuracy": float((margin > 0).float().mean()),
        "mean_margin": float(margin.mean()),
        "mean_chosen_logp": float(pc.mean()),
        "mean_rejected_logp": float(pr.mean()),
        "mean_chosen_reward": float(chosen_reward.mean()),
        "mean_rejected_reward": float(rejected_reward.mean()),
        "mean_gradient_weight": float((beta * torch.sigmoid(-logits)).mean()),
    }


@torch.no_grad()
def teacher_forced_reference_kl(
    policy: nn.Module,
    reference: nn.Module,
    ids: Tensor,
    response_mask: Tensor,
) -> float:
    """Mean exact token KL on the teacher-forced completion prefixes."""
    policy_logits = policy(ids)[:, :-1, :]
    reference_logits = reference(ids)[:, :-1, :]
    logp = F.log_softmax(policy_logits, dim=-1)
    logq = F.log_softmax(reference_logits, dim=-1)
    p = logp.exp()
    token_kl = (p * (logp - logq)).sum(dim=-1)
    mask = response_mask.to(token_kl.dtype)
    return float((token_kl * mask).sum() / mask.sum().clamp_min(1.0))


def train_dpo_policy(
    reference: TinyCausalLM,
    examples: Sequence[PreferenceExample],
    vocab: Vocabulary,
    *,
    beta: float = 0.2,
    steps: int = 120,
    learning_rate: float = 0.01,
    seed: int = 11,
    log_every: int = 10,
) -> tuple[TinyCausalLM, list[dict[str, float]]]:
    """Train a policy clone with full-batch DPO and return its metric history."""
    if steps <= 0 or log_every <= 0:
        raise ValueError("steps and log_every must be positive")
    torch.manual_seed(seed)
    policy = deepcopy(reference).train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()
    batch = pair_batch(examples, vocab)
    with torch.no_grad():
        rc = completion_sequence_logprobs(reference, batch["chosen_ids"], batch["chosen_mask"])
        rr = completion_sequence_logprobs(reference, batch["rejected_ids"], batch["rejected_mask"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    history.append({"step": 0.0, **pair_metrics(policy, reference, batch, beta)})
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pc = completion_sequence_logprobs(policy, batch["chosen_ids"], batch["chosen_mask"])
        pr = completion_sequence_logprobs(policy, batch["rejected_ids"], batch["rejected_mask"])
        loss, _, _ = dpo_loss_from_logprobs(pc, pr, rc, rr, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            metrics = pair_metrics(policy, reference, batch, beta)
            metrics["step"] = float(step)
            history.append(metrics)
    return policy.eval(), history


@torch.no_grad()
def greedy_completion(
    model: nn.Module,
    vocab: Vocabulary,
    prompt: Sequence[str],
    *,
    max_new_tokens: int = 6,
) -> list[str]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    ids = vocab.encode([BOS, *prompt, SEP])
    for _ in range(max_new_tokens):
        tensor = torch.tensor([ids], dtype=torch.long)
        next_id = int(model(tensor)[0, -1].argmax())
        ids.append(next_id)
        if next_id == vocab.eos_id:
            break
    generated = ids[len(prompt) + 2 :]
    tokens = vocab.decode(generated)
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    return tokens


@torch.no_grad()
def generation_preference_rate(
    model: nn.Module,
    examples: Sequence[PreferenceExample],
    vocab: Vocabulary,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    hits = 0
    for row in examples:
        generated = tuple(greedy_completion(model, vocab, row.prompt))
        hit = generated == row.chosen
        hits += int(hit)
        rows.append(
            {
                "label": row.label,
                "prompt": " ".join(row.prompt),
                "generated": " ".join(generated),
                "chosen": " ".join(row.chosen),
                "match": bool(hit),
            }
        )
    return hits / len(examples), rows


def run_toy_dpo_experiment(
    *,
    profile: str = "smoke",
    seed: int = 7,
) -> dict[str, object]:
    """Run the complete deterministic CPU experiment used by both notebooks."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    examples = toy_preference_examples()
    vocab = build_vocabulary(examples)
    if profile == "smoke":
        ref_steps, dpo_steps, hidden, dpo_lr = 150, 60, 40, 0.001
        stress_steps, stress_lr = 80, 0.004
    else:
        ref_steps, dpo_steps, hidden, dpo_lr = 350, 160, 64, 0.0008
        stress_steps, stress_lr = 160, 0.003
    reference = train_reference_model(
        examples,
        vocab,
        hidden_size=hidden,
        steps=ref_steps,
        learning_rate=0.025,
        seed=seed,
    )
    policy, history = train_dpo_policy(
        reference,
        examples,
        vocab,
        beta=0.2,
        steps=dpo_steps,
        learning_rate=dpo_lr,
        seed=seed + 1,
        log_every=max(10, dpo_steps // 20),
    )
    stress_policy, stress_history = train_dpo_policy(
        reference,
        examples,
        vocab,
        beta=0.2,
        steps=stress_steps,
        learning_rate=stress_lr,
        seed=seed + 2,
        log_every=max(10, stress_steps // 20),
    )
    batch = pair_batch(examples, vocab)
    before = pair_metrics(reference, reference, batch, beta=0.2)
    after = pair_metrics(policy, reference, batch, beta=0.2)
    ref_rate, ref_generations = generation_preference_rate(reference, examples, vocab)
    policy_rate, policy_generations = generation_preference_rate(policy, examples, vocab)
    stress_rate, stress_generations = generation_preference_rate(stress_policy, examples, vocab)
    stress_metrics = pair_metrics(stress_policy, reference, batch, beta=0.2)
    kl_chosen = teacher_forced_reference_kl(
        policy, reference, batch["chosen_ids"], batch["chosen_mask"]
    )
    kl_rejected = teacher_forced_reference_kl(
        policy, reference, batch["rejected_ids"], batch["rejected_mask"]
    )
    return {
        "profile": profile,
        "vocab": vocab,
        "examples": examples,
        "reference": reference,
        "policy": policy,
        "stress_policy": stress_policy,
        "history": history,
        "stress_history": stress_history,
        "before": before,
        "after": after,
        "stress": stress_metrics,
        "reference_generation_match": float(ref_rate),
        "policy_generation_match": float(policy_rate),
        "stress_generation_match": float(stress_rate),
        "reference_generations": ref_generations,
        "policy_generations": policy_generations,
        "stress_generations": stress_generations,
        "teacher_forced_kl_chosen": float(kl_chosen),
        "teacher_forced_kl_rejected": float(kl_rejected),
    }


if __name__ == "__main__":
    result = run_toy_dpo_experiment(profile="smoke")
    summary = {
        "before": result["before"],
        "after": result["after"],
        "reference_generation_match": result["reference_generation_match"],
        "policy_generation_match": result["policy_generation_match"],
        "stress": result["stress"],
        "stress_generation_match": result["stress_generation_match"],
        "teacher_forced_kl_chosen": result["teacher_forced_kl_chosen"],
        "teacher_forced_kl_rejected": result["teacher_forced_kl_rejected"],
    }
    print(summary)
