#!/usr/bin/env python3
"""A compact autoregressive RLOO/PPO implementation for Module 4.

The model and verifier are intentionally small so every tensor can be printed
inside a CPU notebook. The training contract is nevertheless the real one used
by causal-language-model post-training: variable-length responses, generated
EOS tokens, prompt/padding masks aligned to shifted labels, frozen old token
log-probabilities, group leave-one-out advantages, token probability ratios,
PPO clipping, entropy, and a nonnegative reverse-KL diagnostic.

This is a mechanism demonstration, not a language-quality benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from module_4_reference import response_token_mask, rloo_advantages

PAD = 0
BOS = 1
EOS = 2
TOKEN_A = 3
TOKEN_B = 4
TOKEN_C = 5
PROMPT_0 = 6
PROMPT_1 = 7
PROMPT_2 = 8
VOCAB_SIZE = 9
TOKEN_NAMES = {
    PAD: "<PAD>", BOS: "<BOS>", EOS: "<EOS>",
    TOKEN_A: "A", TOKEN_B: "B", TOKEN_C: "C",
    PROMPT_0: "<P0>", PROMPT_1: "<P1>", PROMPT_2: "<P2>",
}
TARGETS = {
    0: (TOKEN_A, TOKEN_B),
    1: (TOKEN_B, TOKEN_C),
    2: (TOKEN_C, TOKEN_A),
}


class TinyAutoregressivePolicy(nn.Module):
    def __init__(self, embedding_dim: int = 24, hidden_size: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, VOCAB_SIZE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.08)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.head.weight, gain=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.embedding(input_ids))
        return self.head(hidden)


@dataclass
class SequenceBatch:
    prompt_classes: np.ndarray
    input_ids: torch.Tensor
    prompt_lengths: np.ndarray
    response_mask: torch.Tensor
    old_token_log_probs: torch.Tensor
    rewards: np.ndarray
    advantages: np.ndarray


@dataclass(frozen=True)
class SequenceUpdateDiagnostics:
    policy_loss: float
    entropy: float
    approx_kl: float
    ratio_outside_fraction: float
    active_clip_fraction: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def token_log_probs_and_entropy(
    model: TinyAutoregressivePolicy,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-probability and entropy aligned with labels ``ids[:, 1:]``."""
    logits = model(input_ids[:, :-1])
    distribution = Categorical(logits=logits)
    labels = input_ids[:, 1:]
    return distribution.log_prob(labels), distribution.entropy()


@torch.no_grad()
def sample_responses(
    model: TinyAutoregressivePolicy,
    prompt_classes: np.ndarray,
    *,
    max_response_tokens: int = 4,
    deterministic: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Sample variable-length responses without forcing an unsampled EOS."""
    device = device or torch.device("cpu")
    prompts = np.asarray(prompt_classes, dtype=int)
    if prompts.ndim != 1 or np.any((prompts < 0) | (prompts > 2)):
        raise ValueError("prompt_classes must be a vector containing 0, 1, or 2")
    if max_response_tokens < 1:
        raise ValueError("max_response_tokens must be positive")
    sequences: list[list[int]] = [[BOS, PROMPT_0 + int(p)] for p in prompts]
    finished = np.zeros(len(sequences), dtype=bool)
    model.eval()
    for _ in range(max_response_tokens):
        width = max(len(seq) for seq in sequences)
        padded = np.full((len(sequences), width), PAD, dtype=np.int64)
        lengths = []
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
            lengths.append(len(seq))
        ids = torch.as_tensor(padded, dtype=torch.long, device=device)
        logits = model(ids)
        last_logits = torch.stack([logits[i, lengths[i] - 1] for i in range(len(sequences))])
        # Only response symbols and EOS are legal actions.
        legal = torch.full_like(last_logits, -1e9)
        legal[:, [EOS, TOKEN_A, TOKEN_B, TOKEN_C]] = last_logits[:, [EOS, TOKEN_A, TOKEN_B, TOKEN_C]]
        distribution = Categorical(logits=legal)
        next_tokens = torch.argmax(legal, dim=-1) if deterministic else distribution.sample()
        for i, token in enumerate(next_tokens.cpu().tolist()):
            if finished[i]:
                continue
            sequences[i].append(int(token))
            if token == EOS:
                finished[i] = True
        if bool(np.all(finished)):
            break
    width = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), width), PAD, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return torch.as_tensor(padded, dtype=torch.long, device=device), np.full(len(sequences), 2, dtype=int)


def verifier_reward(prompt_class: int, token_ids: np.ndarray | list[int]) -> float:
    """Dense deterministic reward for a two-symbol target followed by EOS."""
    tokens = [int(x) for x in token_ids]
    response = tokens[2:]
    if EOS in response:
        eos_at = response.index(EOS)
        content = response[:eos_at]
        has_eos = True
    else:
        content = [x for x in response if x != PAD]
        has_eos = False
    target = TARGETS[int(prompt_class)]
    reward = 0.0
    for i, expected in enumerate(target):
        if i < len(content) and content[i] == expected:
            reward += 0.5
    if has_eos and len(content) == len(target):
        reward += 0.2
    if len(content) > len(target):
        reward -= 0.05 * (len(content) - len(target))
    if not has_eos:
        reward -= 0.2
    return float(reward)


def decode_sequence(token_ids: np.ndarray | list[int]) -> str:
    return " ".join(TOKEN_NAMES.get(int(token), f"<{int(token)}>") for token in token_ids if int(token) != PAD)


@torch.no_grad()
def collect_sequence_batch(
    model: TinyAutoregressivePolicy,
    *,
    group_size: int,
    max_response_tokens: int = 4,
    device: torch.device | None = None,
) -> SequenceBatch:
    if group_size < 2:
        raise ValueError("RLOO requires group_size >= 2")
    device = device or torch.device("cpu")
    prompt_classes = np.repeat(np.arange(3, dtype=int), group_size)
    input_ids, prompt_lengths = sample_responses(
        model,
        prompt_classes,
        max_response_tokens=max_response_tokens,
        deterministic=False,
        device=device,
    )
    mask_np = response_token_mask(
        input_ids.cpu().numpy(),
        prompt_lengths,
        pad_token_id=PAD,
        eos_token_id=EOS,
        include_eos=True,
    )
    mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
    token_log_probs, _ = token_log_probs_and_entropy(model, input_ids)
    rewards = np.asarray([
        verifier_reward(prompt, row)
        for prompt, row in zip(prompt_classes, input_ids.cpu().numpy())
    ], dtype=float)
    advantages = np.empty_like(rewards)
    for prompt in range(3):
        idx = np.flatnonzero(prompt_classes == prompt)
        advantages[idx] = rloo_advantages(rewards[idx])
    return SequenceBatch(
        prompt_classes=prompt_classes,
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        response_mask=mask,
        old_token_log_probs=token_log_probs.detach(),
        rewards=rewards,
        advantages=advantages,
    )


def sequence_ppo_update(
    model: TinyAutoregressivePolicy,
    optimizer: torch.optim.Optimizer,
    batch: SequenceBatch,
    *,
    epochs: int = 2,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> SequenceUpdateDiagnostics:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = next(model.parameters()).device
    mask = batch.response_mask.to(device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
    if float(torch.sum(mask)) <= 0.0:
        raise ValueError("batch contains no response tokens")
    token_advantages = advantages[:, None].expand_as(mask)
    values: list[tuple[float, ...]] = []
    model.train()
    for _ in range(epochs):
        new_log_probs, entropy = token_log_probs_and_entropy(model, batch.input_ids.to(device))
        old = batch.old_token_log_probs.to(device)
        log_ratio = new_log_probs - old
        ratio = torch.exp(log_ratio)
        raw = ratio * token_advantages
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages
        pessimistic = torch.minimum(raw, clipped)
        denominator = torch.sum(mask)
        policy_loss = -torch.sum(pessimistic * mask) / denominator
        entropy_mean = torch.sum(entropy * mask) / denominator
        loss = policy_loss - entropy_coef * entropy_mean
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            approx_kl = torch.sum(((ratio - 1.0) - log_ratio) * mask) / denominator
            outside = torch.sum(
                (((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()) * mask
            ) / denominator
            active = torch.sum(((pessimistic < raw - 1e-12).float()) * mask) / denominator
            values.append((float(policy_loss), float(entropy_mean), float(approx_kl), float(outside), float(active)))
    mean = np.mean(np.asarray(values, dtype=float), axis=0)
    return SequenceUpdateDiagnostics(*map(float, mean))


@torch.no_grad()
def evaluate_sequence_policy(
    model: TinyAutoregressivePolicy,
    *,
    samples_per_prompt: int = 32,
    deterministic: bool = False,
    max_response_tokens: int = 4,
) -> dict[str, Any]:
    prompts = np.repeat(np.arange(3, dtype=int), samples_per_prompt)
    ids, _ = sample_responses(
        model,
        prompts,
        max_response_tokens=max_response_tokens,
        deterministic=deterministic,
    )
    rewards = np.asarray([
        verifier_reward(prompt, row)
        for prompt, row in zip(prompts, ids.cpu().numpy())
    ])
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(rewards >= 1.19)),
        "rewards": rewards,
        "input_ids": ids.cpu().numpy(),
        "prompt_classes": prompts,
    }


def sequence_profile_config(profile: str) -> dict[str, int | float]:
    """Return the documented CPU profiles for the autoregressive example."""
    if profile == "smoke":
        return {
            "iterations": 40,
            "group_size": 8,
            "learning_rate": 5e-3,
            "ppo_epochs": 3,
        }
    if profile == "full":
        return {
            "iterations": 120,
            "group_size": 16,
            "learning_rate": 5e-3,
            "ppo_epochs": 3,
        }
    raise ValueError("profile must be 'smoke' or 'full'")


def train_tiny_sequence_policy(
    *,
    seed: int = 0,
    iterations: int = 40,
    group_size: int = 8,
    learning_rate: float = 5e-3,
    ppo_epochs: int = 3,
) -> dict[str, Any]:
    """Train the tiny response policy with group RLOO advantages and PPO.

    The defaults are the documented smoke profile. They are deliberately large
    enough to produce a visible learning curve on CPU while remaining quick
    enough for an instructional notebook. The resulting score is a check of the
    implementation contract, not a benchmark of language-model quality.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    seed_everything(seed)
    model = TinyAutoregressivePolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    history = {
        "iteration": [],
        "batch_mean_reward": [],
        "evaluation_mean_reward": [],
        "success_rate": [],
        "approx_kl": [],
        "active_clip_fraction": [],
    }
    first_batch: SequenceBatch | None = None
    for iteration in range(1, iterations + 1):
        batch = collect_sequence_batch(model, group_size=group_size)
        if first_batch is None:
            first_batch = batch
        diagnostic = sequence_ppo_update(
            model,
            optimizer,
            batch,
            epochs=ppo_epochs,
        )
        evaluation = evaluate_sequence_policy(model, samples_per_prompt=24)
        history["iteration"].append(iteration)
        history["batch_mean_reward"].append(float(np.mean(batch.rewards)))
        history["evaluation_mean_reward"].append(evaluation["mean_reward"])
        history["success_rate"].append(evaluation["success_rate"])
        history["approx_kl"].append(diagnostic.approx_kl)
        history["active_clip_fraction"].append(diagnostic.active_clip_fraction)
    final = evaluate_sequence_policy(model, samples_per_prompt=64)
    return {
        "model": model,
        "history": history,
        "first_batch": first_batch,
        "final_evaluation": final,
    }


__all__ = [
    "BOS", "EOS", "PAD", "PROMPT_0", "PROMPT_1", "PROMPT_2",
    "SequenceBatch", "SequenceUpdateDiagnostics", "TARGETS", "TOKEN_NAMES",
    "TinyAutoregressivePolicy", "collect_sequence_batch", "decode_sequence",
    "evaluate_sequence_policy", "sample_responses", "seed_everything",
    "sequence_ppo_update", "sequence_profile_config", "token_log_probs_and_entropy",
    "train_tiny_sequence_policy", "verifier_reward",
]
