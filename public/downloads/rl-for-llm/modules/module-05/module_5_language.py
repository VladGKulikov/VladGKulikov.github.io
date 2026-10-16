#!/usr/bin/env python3
"""A compact neural offline-RL experiment for autoregressive text.

This is an educational ILQL-like implementation, not a line-by-line
reproduction of the original ILQL codebase.  It preserves the mechanisms that
matter for Module 5:

* a fixed log of prompt-conditioned token trajectories;
* terminal sequence rewards converted into token transitions;
* a behavior policy trained by ordinary token-level maximum likelihood;
* Q(s, token) and V(s) learned from logged actions only;
* expectile value regression and advantage-weighted policy fitting;
* value-guided decoding anchored to the behavior policy's support.

The synthetic task is deliberately small enough for a deterministic CPU smoke
run, while the data contract and update equations match realistic
prefix/action offline RL for language.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PAD = 0
BOS = 1
EOS = 2
VOCAB = (
    "<pad>", "<bos>", "<eos>",
    "red", "blue", "green", "circle", "square", "triangle", "then",
)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}

PROMPTS = (
    "red shape",
    "blue shape",
    "green shape",
    "red then blue",
    "blue then green",
    "green then red",
)
TARGETS = (
    (TOKEN_TO_ID["red"], TOKEN_TO_ID["circle"], EOS),
    (TOKEN_TO_ID["blue"], TOKEN_TO_ID["square"], EOS),
    (TOKEN_TO_ID["green"], TOKEN_TO_ID["triangle"], EOS),
    (TOKEN_TO_ID["red"], TOKEN_TO_ID["then"], TOKEN_TO_ID["blue"], EOS),
    (TOKEN_TO_ID["blue"], TOKEN_TO_ID["then"], TOKEN_TO_ID["green"], EOS),
    (TOKEN_TO_ID["green"], TOKEN_TO_ID["then"], TOKEN_TO_ID["red"], EOS),
)
COMMON_WRONG = (
    (TOKEN_TO_ID["red"], TOKEN_TO_ID["square"], EOS),
    (TOKEN_TO_ID["blue"], TOKEN_TO_ID["circle"], EOS),
    (TOKEN_TO_ID["green"], TOKEN_TO_ID["square"], EOS),
    (TOKEN_TO_ID["red"], TOKEN_TO_ID["then"], TOKEN_TO_ID["green"], EOS),
    (TOKEN_TO_ID["blue"], TOKEN_TO_ID["then"], TOKEN_TO_ID["red"], EOS),
    (TOKEN_TO_ID["green"], TOKEN_TO_ID["then"], TOKEN_TO_ID["blue"], EOS),
)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class LanguageEpisode:
    prompt_id: int
    tokens: tuple[int, ...]
    reward: float


@dataclass(frozen=True)
class TokenTransition:
    prompt_id: int
    prefix: tuple[int, ...]
    action: int
    reward: float
    next_prefix: tuple[int, ...]
    terminated: bool


def sequence_reward(prompt_id: int, tokens: Sequence[int]) -> float:
    """Deterministic verifier reward for a complete candidate sequence."""
    sequence = tuple(int(x) for x in tokens)
    target = TARGETS[prompt_id]
    if sequence == target:
        return 1.0
    # Give a weak partial signal without making the task token-supervised.
    prefix_match = 0
    for candidate, gold in zip(sequence, target):
        if candidate != gold:
            break
        prefix_match += 1
    if prefix_match >= max(1, len(target) - 2):
        return 0.15
    return -0.40


def _random_candidate(prompt_id: int, rng: np.random.Generator) -> tuple[int, ...]:
    target_length = len(TARGETS[prompt_id])
    lexical = np.arange(3, len(VOCAB), dtype=np.int64)
    body = [int(rng.choice(lexical)) for _ in range(target_length - 1)]
    return tuple(body + [EOS])


def collect_fixed_language_log(
    *,
    seed: int = 71,
    episodes_per_prompt: int = 90,
    correct_probability: float = 0.28,
    common_wrong_probability: float = 0.52,
) -> list[LanguageEpisode]:
    """Create a fixed mixed-quality log with support for the optimal sequence.

    The common wrong completion is deliberately more frequent than the correct
    one.  Plain BC therefore tends to imitate the mode, while reward-aware
    weighting can recover the less frequent but better sequence.
    """
    if episodes_per_prompt < 3:
        raise ValueError("episodes_per_prompt must be at least 3")
    if not (0.0 < correct_probability < 1.0):
        raise ValueError("correct_probability must lie in (0, 1)")
    if not (0.0 <= common_wrong_probability < 1.0):
        raise ValueError("common_wrong_probability must lie in [0, 1)")
    if correct_probability + common_wrong_probability >= 1.0:
        raise ValueError("leave positive probability for exploratory candidates")
    rng = seed_everything(seed)
    episodes: list[LanguageEpisode] = []
    for prompt_id in range(len(PROMPTS)):
        # Insert each supported behavior at least once, then sample the rest.
        seeds = [TARGETS[prompt_id], COMMON_WRONG[prompt_id], _random_candidate(prompt_id, rng)]
        for tokens in seeds:
            episodes.append(LanguageEpisode(prompt_id, tuple(tokens), sequence_reward(prompt_id, tokens)))
        for _ in range(episodes_per_prompt - len(seeds)):
            draw = rng.random()
            if draw < correct_probability:
                tokens = TARGETS[prompt_id]
            elif draw < correct_probability + common_wrong_probability:
                tokens = COMMON_WRONG[prompt_id]
            else:
                tokens = _random_candidate(prompt_id, rng)
            episodes.append(LanguageEpisode(prompt_id, tuple(tokens), sequence_reward(prompt_id, tokens)))
    rng.shuffle(episodes)
    return episodes


def episodes_to_transitions(episodes: Iterable[LanguageEpisode]) -> list[TokenTransition]:
    transitions: list[TokenTransition] = []
    for episode in episodes:
        prefix: tuple[int, ...] = (BOS,)
        for index, action in enumerate(episode.tokens):
            terminated = index == len(episode.tokens) - 1 or action == EOS
            reward = episode.reward if terminated else 0.0
            next_prefix = prefix + (int(action),)
            transitions.append(
                TokenTransition(
                    prompt_id=episode.prompt_id,
                    prefix=prefix,
                    action=int(action),
                    reward=float(reward),
                    next_prefix=next_prefix,
                    terminated=terminated,
                )
            )
            prefix = next_prefix
            if terminated:
                break
    if not transitions:
        raise ValueError("episodes must be non-empty")
    return transitions


class TransitionDataset(Dataset[TokenTransition]):
    def __init__(self, transitions: Sequence[TokenTransition]) -> None:
        self.transitions = list(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> TokenTransition:
        return self.transitions[index]


def pad_prefixes(prefixes: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.as_tensor([len(prefix) for prefix in prefixes], dtype=torch.long)
    max_length = int(lengths.max().item())
    tokens = torch.full((len(prefixes), max_length), PAD, dtype=torch.long)
    for row, prefix in enumerate(prefixes):
        tokens[row, : len(prefix)] = torch.as_tensor(prefix, dtype=torch.long)
    return tokens, lengths


def collate_transitions(batch: Sequence[TokenTransition]) -> dict[str, torch.Tensor]:
    prefixes, lengths = pad_prefixes([item.prefix for item in batch])
    next_prefixes, next_lengths = pad_prefixes([item.next_prefix for item in batch])
    return {
        "prompt": torch.as_tensor([item.prompt_id for item in batch], dtype=torch.long),
        "prefix": prefixes,
        "length": lengths,
        "action": torch.as_tensor([item.action for item in batch], dtype=torch.long),
        "reward": torch.as_tensor([item.reward for item in batch], dtype=torch.float32),
        "next_prefix": next_prefixes,
        "next_length": next_lengths,
        "terminated": torch.as_tensor([item.terminated for item in batch], dtype=torch.float32),
    }


class PrefixEncoder(nn.Module):
    def __init__(self, hidden: int = 72) -> None:
        super().__init__()
        self.prompt_embedding = nn.Embedding(len(PROMPTS), hidden)
        self.token_embedding = nn.Embedding(len(VOCAB), hidden, padding_idx=PAD)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.prompt_projection = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prefixes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(prefixes)
        h0 = torch.tanh(self.prompt_projection(self.prompt_embedding(prompt_ids))).unsqueeze(0)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed, h0)
        return self.norm(hidden.squeeze(0))


class LanguagePolicy(nn.Module):
    def __init__(self, hidden: int = 72) -> None:
        super().__init__()
        self.encoder = PrefixEncoder(hidden)
        self.head = nn.Linear(hidden, len(VOCAB))

    def forward(self, prompt: torch.Tensor, prefix: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(prompt, prefix, length))


class LanguageQValue(nn.Module):
    def __init__(self, hidden: int = 88) -> None:
        super().__init__()
        self.encoder = PrefixEncoder(hidden)
        self.q_head = nn.Linear(hidden, len(VOCAB))
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, prompt: torch.Tensor, prefix: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(prompt, prefix, length)
        return self.q_head(encoded), self.v_head(encoded).squeeze(-1)


_ALLOWED_TOKENS = torch.zeros(len(VOCAB), dtype=torch.bool)
_ALLOWED_TOKENS[EOS:] = True


def _mask_special_logits(logits: torch.Tensor) -> torch.Tensor:
    allowed = _ALLOWED_TOKENS.to(logits.device)
    return logits.masked_fill(~allowed, -1e9)


def train_language_bc(
    transitions: Sequence[TokenTransition],
    *,
    seed: int = 73,
    epochs: int = 28,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
) -> tuple[LanguagePolicy, list[float]]:
    seed_everything(seed)
    model = LanguagePolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TransitionDataset(transitions),
        batch_size=min(batch_size, len(transitions)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_transitions,
    )
    history: list[float] = []
    for _ in range(epochs):
        losses: list[float] = []
        for batch in loader:
            logits = _mask_special_logits(model(batch["prompt"], batch["prefix"], batch["length"]))
            loss = F.cross_entropy(logits, batch["action"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


def train_language_iql(
    transitions: Sequence[TokenTransition],
    *,
    seed: int = 79,
    epochs: int = 45,
    batch_size: int = 128,
    learning_rate: float = 1.5e-3,
    gamma: float = 1.0,
    expectile: float = 0.8,
    inverse_temperature: float = 4.0,
    max_weight: float = 40.0,
    target_rate: float = 0.03,
) -> dict[str, object]:
    if not 0.5 <= expectile < 1.0:
        raise ValueError("expectile must lie in [0.5, 1)")
    seed_everything(seed)
    qv = LanguageQValue()
    target = copy.deepcopy(qv).eval()
    policy = LanguagePolicy()
    q_optimizer = torch.optim.AdamW(qv.parameters(), lr=learning_rate, weight_decay=1e-4)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TransitionDataset(transitions),
        batch_size=min(batch_size, len(transitions)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_transitions,
    )
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        q_losses: list[float] = []
        v_losses: list[float] = []
        p_losses: list[float] = []
        weights_seen: list[float] = []
        for batch in loader:
            q_values, values = qv(batch["prompt"], batch["prefix"], batch["length"])
            q_data = q_values.gather(1, batch["action"][:, None]).squeeze(1)
            with torch.no_grad():
                target_q, _ = target(batch["prompt"], batch["prefix"], batch["length"])
                target_q_data = target_q.gather(1, batch["action"][:, None]).squeeze(1)
                _, next_values = target(batch["prompt"], batch["next_prefix"], batch["next_length"])
                td_target = batch["reward"] + gamma * (1.0 - batch["terminated"]) * next_values

            residual = target_q_data - values
            expectile_weight = torch.where(residual >= 0.0, expectile, 1.0 - expectile)
            value_loss = (expectile_weight * residual.square()).mean()
            q_loss = F.smooth_l1_loss(q_data, td_target)
            critic_loss = q_loss + value_loss
            q_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(qv.parameters(), 1.0)
            q_optimizer.step()

            with torch.no_grad():
                q_for_weight, v_for_weight = target(
                    batch["prompt"], batch["prefix"], batch["length"]
                )
                advantage = q_for_weight.gather(1, batch["action"][:, None]).squeeze(1) - v_for_weight
                weights = torch.exp(inverse_temperature * advantage).clamp(max=max_weight)
            logits = _mask_special_logits(policy(batch["prompt"], batch["prefix"], batch["length"]))
            log_prob = F.log_softmax(logits, dim=-1).gather(1, batch["action"][:, None]).squeeze(1)
            policy_loss = -(weights * log_prob).mean()
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimizer.step()

            with torch.no_grad():
                for target_parameter, parameter in zip(target.parameters(), qv.parameters()):
                    target_parameter.mul_(1.0 - target_rate).add_(target_rate * parameter)
            q_losses.append(float(q_loss.detach()))
            v_losses.append(float(value_loss.detach()))
            p_losses.append(float(policy_loss.detach()))
            weights_seen.append(float(weights.mean()))
        history.append(
            {
                "q_loss": float(np.mean(q_losses)),
                "v_loss": float(np.mean(v_losses)),
                "policy_loss": float(np.mean(p_losses)),
                "mean_weight": float(np.mean(weights_seen)),
            }
        )
    return {"qv": qv, "target": target, "policy": policy, "history": history}


def _single_prefix(prefix: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return pad_prefixes([prefix])


@torch.no_grad()
def decode_policy(
    policy: LanguagePolicy,
    prompt_id: int,
    *,
    max_new_tokens: int = 6,
) -> tuple[int, ...]:
    prefix: list[int] = [BOS]
    result: list[int] = []
    for _ in range(max_new_tokens):
        x, length = _single_prefix(prefix)
        logits = _mask_special_logits(policy(torch.tensor([prompt_id]), x, length))[0]
        action = int(logits.argmax().item())
        result.append(action)
        prefix.append(action)
        if action == EOS:
            break
    if not result or result[-1] != EOS:
        result.append(EOS)
    return tuple(result)


@torch.no_grad()
def decode_value_guided(
    behavior: LanguagePolicy,
    qv: LanguageQValue,
    prompt_id: int,
    *,
    beta: float = 3.0,
    support_ratio: float = 0.08,
    max_new_tokens: int = 6,
) -> tuple[int, ...]:
    """Decode with log pi_beta + beta * (Q - V) under a support mask."""
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not 0.0 <= support_ratio <= 1.0:
        raise ValueError("support_ratio must lie in [0, 1]")
    prefix: list[int] = [BOS]
    result: list[int] = []
    prompt = torch.tensor([prompt_id])
    for _ in range(max_new_tokens):
        x, length = _single_prefix(prefix)
        behavior_logits = _mask_special_logits(behavior(prompt, x, length))[0]
        log_behavior = F.log_softmax(behavior_logits, dim=-1)
        probabilities = log_behavior.exp()
        q_values, values = qv(prompt, x, length)
        scores = log_behavior + beta * (q_values[0] - values[0])
        support = probabilities >= support_ratio * probabilities.max()
        support[:EOS] = False
        if not bool(support.any()):
            support[probabilities.argmax()] = True
        scores = scores.masked_fill(~support, -1e9)
        action = int(scores.argmax().item())
        result.append(action)
        prefix.append(action)
        if action == EOS:
            break
    if not result or result[-1] != EOS:
        result.append(EOS)
    return tuple(result)


def evaluate_decoder(decoder) -> dict[str, object]:
    sequences: list[tuple[int, ...]] = []
    rewards: list[float] = []
    exact: list[float] = []
    for prompt_id in range(len(PROMPTS)):
        tokens = tuple(decoder(prompt_id))
        sequences.append(tokens)
        rewards.append(sequence_reward(prompt_id, tokens))
        exact.append(float(tokens == TARGETS[prompt_id]))
    return {
        "mean_reward": float(np.mean(rewards)),
        "exact_match": float(np.mean(exact)),
        "sequences": sequences,
        "rewards": rewards,
    }


def token_text(tokens: Sequence[int]) -> str:
    return " ".join(VOCAB[int(token)] for token in tokens)


def language_offline_experiment(
    *,
    seed: int = 71,
    episodes_per_prompt: int = 90,
    bc_epochs: int = 28,
    iql_epochs: int = 45,
) -> dict[str, object]:
    episodes = collect_fixed_language_log(
        seed=seed, episodes_per_prompt=episodes_per_prompt
    )
    transitions = episodes_to_transitions(episodes)
    behavior, bc_history = train_language_bc(
        transitions, seed=seed + 1, epochs=bc_epochs
    )
    iql = train_language_iql(
        transitions, seed=seed + 2, epochs=iql_epochs
    )
    behavior_eval = evaluate_decoder(lambda prompt: decode_policy(behavior, prompt))
    policy_eval = evaluate_decoder(lambda prompt: decode_policy(iql["policy"], prompt))
    guided_eval = evaluate_decoder(
        lambda prompt: decode_value_guided(behavior, iql["target"], prompt)
    )
    dataset_rewards = np.asarray([episode.reward for episode in episodes], dtype=np.float32)
    return {
        "episodes": episodes,
        "transitions": transitions,
        "behavior": behavior,
        "iql": iql,
        "bc_history": bc_history,
        "dataset": {
            "mean_reward": float(dataset_rewards.mean()),
            "success_rate": float(np.mean(dataset_rewards == 1.0)),
        },
        "evaluation": {
            "Behavior cloning": behavior_eval,
            "Advantage-weighted policy": policy_eval,
            "Value-guided behavior": guided_eval,
        },
    }


__all__ = [
    "PAD", "BOS", "EOS", "VOCAB", "TOKEN_TO_ID", "PROMPTS", "TARGETS",
    "COMMON_WRONG", "LanguageEpisode", "TokenTransition", "sequence_reward",
    "collect_fixed_language_log", "episodes_to_transitions", "PrefixEncoder",
    "LanguagePolicy", "LanguageQValue", "train_language_bc",
    "train_language_iql", "decode_policy", "decode_value_guided",
    "evaluate_decoder", "token_text", "language_offline_experiment",
]
