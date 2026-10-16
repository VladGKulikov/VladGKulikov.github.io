"""Small but structurally realistic RLVR/GRPO training loop for Module 9.

The implementation is deliberately compact enough for deterministic CPU tests,
while preserving the contracts that matter in language-model reinforcement
learning:

* prompts are token sequences and responses are generated autoregressively;
* a versioned verifier returns decomposed rewards and explicit failure states;
* several responses are sampled for each prompt from a frozen old policy;
* response-level group advantages are broadcast over active response tokens;
* current/old/reference policies have separate mathematical roles;
* token-level probability ratios, clipping, response masks, and reduction units
  are explicit;
* held-out generation, entropy, clipping, KL drift, length, and informative-group
  statistics are reported together.

The grammar is intentionally small and constrained.  This keeps the experiment
fast and lets the verifier be exhaustive; it does not claim to reproduce the
scale or open-ended output space of a production LLM.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PAD = "<pad>"
BOS = "<bos>"
SEP = "<sep>"
EOS = "<eos>"
DIRECT = "<direct>"
WORK = "<work>"
ANS_OPEN = "<answer>"
ANS_CLOSE = "</answer>"
PLUS = "+"
EQUALS = "="

NUMBER_TOKENS = tuple(str(i) for i in range(19))
SPECIAL_TOKENS = (PAD, BOS, SEP, EOS, DIRECT, WORK, ANS_OPEN, ANS_CLOSE, PLUS, EQUALS)


@dataclass(frozen=True)
class ArithmeticPrompt:
    prompt_id: str
    a: int
    b: int
    split: Literal["train", "heldout"]

    @property
    def answer(self) -> int:
        return self.a + self.b

    @property
    def prompt_tokens(self) -> tuple[str, ...]:
        return ("add", str(self.a), str(self.b))


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        missing = [token for token in tokens if token not in self.token_to_id]
        if missing:
            raise KeyError(f"unknown tokens: {missing}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(index)] for index in ids]

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


@dataclass(frozen=True)
class VerifierResult:
    reward: float
    accuracy: float
    consistency: float
    format_ok: float
    status: str
    parsed_answer: int | None


@dataclass(frozen=True)
class GeneratedResponse:
    prompt_id: str
    group_id: str
    token_ids: tuple[int, ...]
    response_tokens: tuple[str, ...]
    old_logprobs: tuple[float, ...]
    action_mask: tuple[int, ...]
    reward: float
    accuracy: float
    consistency: float
    format_ok: float
    verifier_status: str
    style: str
    terminated: bool
    truncated: bool
    generator_version: str


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 6
    prompt_batch: int = 6
    updates: int = 30
    update_epochs: int = 2
    learning_rate: float = 0.015
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    scale_rewards: bool = True
    loss_type: Literal["grpo", "dapo", "dr_grpo"] = "dapo"
    fixed_completion_length: int = 10
    entropy_coef: float = 0.004
    kl_coef: float = 0.0
    max_new_tokens: int = 10
    filter_degenerate: bool = False


class TinyCausalLM(nn.Module):
    """A tiny GRU causal language model with an LLM-like token interface."""

    def __init__(self, vocab_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        hidden, _ = self.gru(self.embedding(input_ids))
        return self.lm_head(hidden)


def arithmetic_prompts() -> list[ArithmeticPrompt]:
    """Return train and held-out prompts with answers representable by one token."""
    train_pairs = [(0, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4),
                   (4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (6, 7)]
    heldout_pairs = [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 8)]
    prompts: list[ArithmeticPrompt] = []
    for i, (a, b) in enumerate(train_pairs):
        prompts.append(ArithmeticPrompt(f"train-{i:02d}", a, b, "train"))
    for i, (a, b) in enumerate(heldout_pairs):
        prompts.append(ArithmeticPrompt(f"heldout-{i:02d}", a, b, "heldout"))
    return prompts


def build_vocabulary(prompts: Sequence[ArithmeticPrompt]) -> Vocabulary:
    tokens = set(SPECIAL_TOKENS) | set(NUMBER_TOKENS) | {"add"}
    for prompt in prompts:
        tokens.update(prompt.prompt_tokens)
    ordered = list(SPECIAL_TOKENS) + ["add"] + list(NUMBER_TOKENS)
    extras = sorted(tokens - set(ordered))
    ordered.extend(extras)
    return Vocabulary({token: i for i, token in enumerate(ordered)}, tuple(ordered))


def prompt_prefix(vocab: Vocabulary, prompt: ArithmeticPrompt) -> list[int]:
    return vocab.encode([BOS, *prompt.prompt_tokens, SEP])


def allowed_next_tokens(
    vocab: Vocabulary,
    prompt: ArithmeticPrompt,
    response_tokens: Sequence[str],
) -> tuple[str, ...]:
    """Return the constrained response grammar after the current prefix.

    The grammar is deliberately explicit so that the actual behavior policy is
    exactly the model distribution *after* a token mask, mirroring constrained
    decoding in production systems.
    """
    n = len(response_tokens)
    if n == 0:
        return (DIRECT, WORK)
    style = response_tokens[0]
    if style == DIRECT:
        grammar: tuple[tuple[str, ...], ...] = (
            (DIRECT,),
            (ANS_OPEN,),
            NUMBER_TOKENS,
            (ANS_CLOSE,),
            (EOS,),
        )
    elif style == WORK:
        grammar = (
            (WORK,),
            (str(prompt.a),),
            (PLUS,),
            (str(prompt.b),),
            (EQUALS,),
            NUMBER_TOKENS,
            (ANS_OPEN,),
            NUMBER_TOKENS,
            (ANS_CLOSE,),
            (EOS,),
        )
    else:
        return (EOS,)
    if n >= len(grammar):
        return (EOS,)
    return grammar[n]


def masked_log_probs(logits: Tensor, allowed_ids: Sequence[int]) -> Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not allowed_ids:
        raise ValueError("allowed_ids must be non-empty")
    mask = torch.full_like(logits, float("-inf"))
    mask[list(allowed_ids)] = logits[list(allowed_ids)]
    return F.log_softmax(mask, dim=-1)


def verify_response(prompt: ArithmeticPrompt, tokens: Sequence[str]) -> VerifierResult:
    """Verify the constrained arithmetic response and return decomposed reward."""
    if not tokens:
        return VerifierResult(0.0, 0.0, 0.0, 0.0, "empty", None)
    try:
        open_index = max(i for i, token in enumerate(tokens) if token == ANS_OPEN)
        close_index = next(i for i in range(open_index + 1, len(tokens)) if tokens[i] == ANS_CLOSE)
    except (ValueError, StopIteration):
        return VerifierResult(0.0, 0.0, 0.0, 0.0, "format_error", None)
    payload = tokens[open_index + 1:close_index]
    if len(payload) != 1 or payload[0] not in NUMBER_TOKENS:
        return VerifierResult(0.0, 0.0, 0.0, 0.0, "parse_error", None)
    parsed = int(payload[0])
    accuracy = float(parsed == prompt.answer)
    style = tokens[0]
    if style == WORK and len(tokens) >= 8 and tokens[5] in NUMBER_TOKENS:
        intermediate = int(tokens[5])
        consistency = float(intermediate == parsed == prompt.answer)
    else:
        consistency = accuracy
    format_ok = 1.0
    reward = accuracy + 0.1 * consistency
    return VerifierResult(reward, accuracy, consistency, format_ok, "ok", parsed)


@torch.no_grad()
def generate_response(
    model: TinyCausalLM,
    vocab: Vocabulary,
    prompt: ArithmeticPrompt,
    *,
    generator: torch.Generator,
    group_id: str,
    generator_version: str,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
) -> GeneratedResponse:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    ids = prompt_prefix(vocab, prompt)
    response_tokens: list[str] = []
    token_logprobs: list[float] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits = model(input_ids)[0, -1] / float(temperature)
        allowed_tokens = allowed_next_tokens(vocab, prompt, response_tokens)
        allowed_ids = vocab.encode(allowed_tokens)
        log_probs = masked_log_probs(logits, allowed_ids)
        next_id = int(torch.multinomial(log_probs.exp(), 1, generator=generator))
        token = vocab.id_to_token[next_id]
        ids.append(next_id)
        response_tokens.append(token)
        token_logprobs.append(float(log_probs[next_id]))
        if token == EOS:
            break
    terminated = bool(response_tokens and response_tokens[-1] == EOS)
    truncated = not terminated
    result = verify_response(prompt, response_tokens)
    style = response_tokens[0] if response_tokens else "<none>"
    # token_ids contains the response only; every entry is an actual action.
    return GeneratedResponse(
        prompt_id=prompt.prompt_id,
        group_id=group_id,
        token_ids=tuple(ids[len(prompt_prefix(vocab, prompt)):]),
        response_tokens=tuple(response_tokens),
        old_logprobs=tuple(token_logprobs),
        action_mask=tuple(1 for _ in token_logprobs),
        reward=float(result.reward),
        accuracy=float(result.accuracy),
        consistency=float(result.consistency),
        format_ok=float(result.format_ok),
        verifier_status=result.status,
        style=style,
        terminated=terminated,
        truncated=truncated,
        generator_version=generator_version,
    )


def response_logprobs_and_entropy(
    model: TinyCausalLM,
    vocab: Vocabulary,
    prompt: ArithmeticPrompt,
    response_tokens: Sequence[str],
) -> tuple[Tensor, Tensor]:
    """Teacher-force one generated response under the same constrained policy."""
    ids = prompt_prefix(vocab, prompt)
    logps: list[Tensor] = []
    entropies: list[Tensor] = []
    prefix_response: list[str] = []
    for token in response_tokens:
        input_ids = torch.tensor([ids], dtype=torch.long)
        logits = model(input_ids)[0, -1]
        allowed = allowed_next_tokens(vocab, prompt, prefix_response)
        allowed_ids = vocab.encode(allowed)
        log_probs = masked_log_probs(logits, allowed_ids)
        token_id = vocab.token_to_id[token]
        if token_id not in allowed_ids:
            raise ValueError(f"token {token!r} is outside the constrained policy support")
        logps.append(log_probs[token_id])
        probs = log_probs[allowed_ids].exp()
        entropies.append(-(probs * log_probs[allowed_ids]).sum())
        ids.append(token_id)
        prefix_response.append(token)
    if not logps:
        raise ValueError("response_tokens must be non-empty")
    return torch.stack(logps), torch.stack(entropies)


def exact_teacher_forced_kl(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    vocab: Vocabulary,
    prompt: ArithmeticPrompt,
    response_tokens: Sequence[str],
) -> Tensor:
    """Mean exact categorical KL over prefixes visited by one response."""
    ids = prompt_prefix(vocab, prompt)
    prefix_response: list[str] = []
    terms: list[Tensor] = []
    for token in response_tokens:
        input_ids = torch.tensor([ids], dtype=torch.long)
        p_logits = policy(input_ids)[0, -1]
        q_logits = reference(input_ids)[0, -1]
        allowed = allowed_next_tokens(vocab, prompt, prefix_response)
        allowed_ids = vocab.encode(allowed)
        p_log = masked_log_probs(p_logits, allowed_ids)[allowed_ids]
        q_log = masked_log_probs(q_logits, allowed_ids)[allowed_ids]
        p = p_log.exp()
        terms.append((p * (p_log - q_log)).sum())
        token_id = vocab.token_to_id[token]
        ids.append(token_id)
        prefix_response.append(token)
    if not terms:
        raise ValueError("response_tokens must be non-empty")
    return torch.stack(terms).mean()


def _group_advantages(rewards: Tensor, scale: bool, eps: float = 1e-8) -> Tensor:
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not scale:
        return centered
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return torch.where(std > eps, centered / (std + eps), torch.zeros_like(centered))


def pretrain_base_model(
    prompts: Sequence[ArithmeticPrompt],
    vocab: Vocabulary,
    *,
    hidden_size: int = 48,
    steps: int = 140,
    learning_rate: float = 0.025,
    seed: int = 9,
) -> TinyCausalLM:
    """Warm-start a model on a noisy mixture of correct and near-correct traces."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = TinyCausalLM(len(vocab.id_to_token), hidden_size=hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    training = [p for p in prompts if p.split == "train"]
    rng = random.Random(seed)
    model.train()
    for step in range(steps):
        prompt = training[step % len(training)]
        correct = prompt.answer
        # A controlled noisy curriculum: roughly two thirds correct, otherwise a
        # nearby answer. Both direct and worked styles appear in the base model.
        is_correct = (step % 3) != 0
        guess = correct if is_correct else max(0, min(18, correct + rng.choice([-2, -1, 1, 2])))
        if step % 2 == 0:
            response = [DIRECT, ANS_OPEN, str(guess), ANS_CLOSE, EOS]
        else:
            response = [WORK, str(prompt.a), PLUS, str(prompt.b), EQUALS,
                        str(guess), ANS_OPEN, str(guess), ANS_CLOSE, EOS]
        logps, _ = response_logprobs_and_entropy(model, vocab, prompt, response)
        loss = -logps.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval()


def collect_rollouts(
    old_policy: TinyCausalLM,
    vocab: Vocabulary,
    prompts: Sequence[ArithmeticPrompt],
    config: GRPOConfig,
    *,
    generator: torch.Generator,
    generator_version: str,
) -> tuple[list[list[GeneratedResponse]], float]:
    """Collect one grouped on-policy batch and report discarded prompt fraction."""
    if config.group_size < 2 or config.prompt_batch <= 0:
        raise ValueError("group_size >= 2 and prompt_batch > 0 required")
    training = [p for p in prompts if p.split == "train"]
    if not training:
        raise ValueError("at least one training prompt is required")
    indices = torch.randperm(len(training), generator=generator).tolist()
    groups: list[list[GeneratedResponse]] = []
    discarded = 0
    attempts = 0
    cursor = 0
    max_attempts = max(4 * config.prompt_batch, len(training) * 3)
    while len(groups) < config.prompt_batch and attempts < max_attempts:
        prompt = training[indices[cursor % len(indices)]]
        cursor += 1
        attempts += 1
        group_id = f"{generator_version}:{prompt.prompt_id}:{attempts:03d}"
        group = [
            generate_response(
                old_policy,
                vocab,
                prompt,
                generator=generator,
                group_id=group_id,
                generator_version=generator_version,
                max_new_tokens=config.max_new_tokens,
            )
            for _ in range(config.group_size)
        ]
        rewards = {round(row.reward, 8) for row in group}
        if config.filter_degenerate and len(rewards) == 1:
            discarded += 1
            continue
        groups.append(group)
    if len(groups) < config.prompt_batch:
        raise RuntimeError("could not collect enough rollout groups")
    return groups, discarded / max(1, attempts)


def grpo_loss(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    vocab: Vocabulary,
    prompts_by_id: dict[str, ArithmeticPrompt],
    groups: Sequence[Sequence[GeneratedResponse]],
    config: GRPOConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Compute a clipped group-relative objective over a stored rollout batch."""
    reward_tensor = torch.tensor(
        [[row.reward for row in group] for group in groups], dtype=torch.float32
    )
    advantages = _group_advantages(reward_tensor, config.scale_rewards)
    token_terms: list[Tensor] = []
    token_entropies: list[Tensor] = []
    kl_terms: list[Tensor] = []
    per_response_terms: list[Tensor] = []
    clipped_low = 0
    clipped_high = 0
    active_tokens = 0
    ratios_all: list[Tensor] = []

    for g, group in enumerate(groups):
        for i, row in enumerate(group):
            prompt = prompts_by_id[row.prompt_id]
            current_logp, entropy = response_logprobs_and_entropy(
                policy, vocab, prompt, row.response_tokens
            )
            old_logp = torch.tensor(row.old_logprobs, dtype=current_logp.dtype)
            if old_logp.shape != current_logp.shape:
                raise ValueError("stored old log-probabilities must align with response tokens")
            ratio = torch.exp(current_logp - old_logp)
            advantage = advantages[g, i]
            clipped_ratio = ratio.clamp(
                1.0 - config.epsilon_low, 1.0 + config.epsilon_high
            )
            surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
            clipped_low += int(((advantage < 0) & (ratio < 1.0 - config.epsilon_low)).sum())
            clipped_high += int(((advantage > 0) & (ratio > 1.0 + config.epsilon_high)).sum())
            active_tokens += ratio.numel()
            ratios_all.append(ratio.detach())
            token_entropies.append(entropy)
            if config.kl_coef > 0:
                kl = exact_teacher_forced_kl(
                    policy, reference, vocab, prompt, row.response_tokens
                )
                kl_terms.append(kl)
            else:
                kl = surrogate.new_zeros(())

            if config.loss_type == "grpo":
                response_term = surrogate.mean() - config.kl_coef * kl
                per_response_terms.append(response_term)
            elif config.loss_type == "dr_grpo":
                scaled = surrogate.sum() / float(config.fixed_completion_length)
                per_response_terms.append(scaled - config.kl_coef * kl)
            elif config.loss_type == "dapo":
                # The exact KL is averaged over the visited prefixes. Subtracting
                # the same scalar from every active token preserves one global
                # active-token denominator without accidentally doubling it.
                token_terms.append(surrogate - config.kl_coef * kl)
            else:
                raise ValueError(f"unsupported loss_type: {config.loss_type}")

    if config.loss_type == "dapo":
        objective = torch.cat(token_terms).mean()
    else:
        objective = torch.stack(per_response_terms).mean()
    entropy_mean = torch.cat(token_entropies).mean()
    loss = -objective - config.entropy_coef * entropy_mean
    ratios_cat = torch.cat(ratios_all)
    metrics = {
        "objective": float(objective.detach()),
        "entropy": float(entropy_mean.detach()),
        "clip_low_fraction": clipped_low / max(1, active_tokens),
        "clip_high_fraction": clipped_high / max(1, active_tokens),
        "ratio_mean": float(ratios_cat.mean()),
        "ratio_min": float(ratios_cat.min()),
        "ratio_max": float(ratios_cat.max()),
        "reward_mean": float(reward_tensor.mean()),
        "reward_std": float(reward_tensor.std(unbiased=False)),
        "informative_fraction": float(
            (reward_tensor.std(dim=1, unbiased=False) > 1e-8).float().mean()
        ),
    }
    if kl_terms:
        metrics["teacher_forced_kl"] = float(torch.stack(kl_terms).mean().detach())
    return loss, metrics


@torch.no_grad()
def evaluate_policy(
    policy: TinyCausalLM,
    reference: TinyCausalLM,
    vocab: Vocabulary,
    prompts: Sequence[ArithmeticPrompt],
    *,
    split: Literal["train", "heldout"] = "heldout",
    samples_per_prompt: int = 32,
    seed: int = 123,
    max_new_tokens: int = 10,
) -> dict[str, float]:
    selected = [p for p in prompts if p.split == split]
    if not selected or samples_per_prompt <= 0:
        raise ValueError("non-empty split and positive samples_per_prompt required")
    generator = torch.Generator().manual_seed(seed)
    accuracies: list[float] = []
    rewards: list[float] = []
    lengths: list[int] = []
    entropies: list[float] = []
    kls: list[float] = []
    styles: list[str] = []
    truncated = 0
    prompt_successes: list[int] = []
    for prompt in selected:
        successes = 0
        for sample in range(samples_per_prompt):
            row = generate_response(
                policy, vocab, prompt,
                generator=generator,
                group_id=f"eval:{prompt.prompt_id}:{sample}",
                generator_version="eval",
                max_new_tokens=max_new_tokens,
            )
            _, entropy = response_logprobs_and_entropy(
                policy, vocab, prompt, row.response_tokens
            )
            kl = exact_teacher_forced_kl(
                policy, reference, vocab, prompt, row.response_tokens
            )
            accuracies.append(row.accuracy)
            rewards.append(row.reward)
            lengths.append(len(row.response_tokens))
            entropies.append(float(entropy.mean()))
            kls.append(float(kl))
            styles.append(row.style)
            truncated += int(row.truncated)
            successes += int(row.accuracy)
        prompt_successes.append(successes)
    n = len(accuracies)
    pass4 = np.mean([
        1.0 - math.comb(samples_per_prompt - c, 4) / math.comb(samples_per_prompt, 4)
        if samples_per_prompt >= 4 and samples_per_prompt - c >= 4 else float(c > 0)
        for c in prompt_successes
    ])
    return {
        "accuracy": float(np.mean(accuracies)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "entropy": float(np.mean(entropies)),
        "teacher_forced_kl": float(np.mean(kls)),
        "direct_fraction": float(np.mean([style == DIRECT for style in styles])),
        "truncated_fraction": truncated / n,
        "pass_at_4": float(pass4),
    }


@torch.no_grad()
def prompt_level_accuracy(
    policy: TinyCausalLM,
    vocab: Vocabulary,
    prompts: Sequence[ArithmeticPrompt],
    *,
    split: Literal["train", "heldout"] = "heldout",
    samples_per_prompt: int = 32,
    seed: int = 321,
) -> dict[str, float]:
    """Return sampled accuracy for each prompt in one split.

    The table is useful for diagnosing coverage collapse: average accuracy can
    improve while success becomes concentrated on a smaller subset of prompts.
    """
    selected = [p for p in prompts if p.split == split]
    if not selected or samples_per_prompt <= 0:
        raise ValueError("non-empty split and positive samples_per_prompt required")
    generator = torch.Generator().manual_seed(seed)
    result: dict[str, float] = {}
    for prompt in selected:
        successes = 0
        for sample in range(samples_per_prompt):
            row = generate_response(
                policy,
                vocab,
                prompt,
                generator=generator,
                group_id=f"prompt-eval:{prompt.prompt_id}:{sample}",
                generator_version="prompt-eval",
            )
            successes += int(row.accuracy)
        result[prompt.prompt_id] = successes / samples_per_prompt
    return result


def run_toy_grpo_experiment(
    *,
    profile: Literal["smoke", "full"] = "smoke",
    seed: int = 9,
    loss_type: Literal["grpo", "dapo", "dr_grpo"] = "dapo",
) -> dict[str, object]:
    """Run a deterministic end-to-end autoregressive RLVR/GRPO experiment."""
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    prompts = arithmetic_prompts()
    vocab = build_vocabulary(prompts)
    pretrain_steps = 48 if profile == "smoke" else 260
    config = GRPOConfig(
        group_size=4 if profile == "smoke" else 8,
        prompt_batch=4 if profile == "smoke" else 10,
        updates=24 if profile == "smoke" else 70,
        update_epochs=2 if profile == "smoke" else 3,
        learning_rate=0.012 if profile == "smoke" else 0.008,
        loss_type=loss_type,
        entropy_coef=0.004,
        filter_degenerate=False,
    )
    base = pretrain_base_model(prompts, vocab, steps=pretrain_steps, seed=seed)
    reference = deepcopy(base).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy = deepcopy(base).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    prompts_by_id = {p.prompt_id: p for p in prompts}
    generator = torch.Generator().manual_seed(seed + 1000)

    before_train = evaluate_policy(
        policy, reference, vocab, prompts, split="train",
        samples_per_prompt=8 if profile == "smoke" else 48, seed=seed + 1,
    )
    before_heldout = evaluate_policy(
        policy, reference, vocab, prompts, split="heldout",
        samples_per_prompt=10 if profile == "smoke" else 64, seed=seed + 2,
    )

    history: list[dict[str, float]] = []
    for update in range(1, config.updates + 1):
        old_policy = deepcopy(policy).eval()
        for parameter in old_policy.parameters():
            parameter.requires_grad_(False)
        groups, discarded_fraction = collect_rollouts(
            old_policy, vocab, prompts, config,
            generator=generator, generator_version=f"update-{update:03d}",
        )
        last_metrics: dict[str, float] = {}
        for _ in range(config.update_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, last_metrics = grpo_loss(
                policy, reference, vocab, prompts_by_id, groups, config
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if update == 1 or update % 4 == 0 or update == config.updates:
            evaluation = evaluate_policy(
                policy, reference, vocab, prompts, split="train",
                samples_per_prompt=6 if profile == "smoke" else 24,
                seed=seed + 100 + update,
            )
            history.append({
                "update": float(update),
                **last_metrics,
                "discarded_fraction": float(discarded_fraction),
                "eval_accuracy": evaluation["accuracy"],
                "eval_entropy": evaluation["entropy"],
                "eval_length": evaluation["mean_length"],
                "eval_direct_fraction": evaluation["direct_fraction"],
            })
    policy.eval()
    after_train = evaluate_policy(
        policy, reference, vocab, prompts, split="train",
        samples_per_prompt=10 if profile == "smoke" else 72, seed=seed + 3,
    )
    after_heldout = evaluate_policy(
        policy, reference, vocab, prompts, split="heldout",
        samples_per_prompt=12 if profile == "smoke" else 96, seed=seed + 4,
    )
    before_train_by_prompt = prompt_level_accuracy(
        reference, vocab, prompts, split="train",
        samples_per_prompt=12 if profile == "smoke" else 48, seed=seed + 5,
    )
    after_train_by_prompt = prompt_level_accuracy(
        policy, vocab, prompts, split="train",
        samples_per_prompt=12 if profile == "smoke" else 48, seed=seed + 5,
    )
    return {
        "profile": profile,
        "seed": seed,
        "config": config,
        "vocab_size": len(vocab.id_to_token),
        "before_train": before_train,
        "before_heldout": before_heldout,
        "after_train": after_train,
        "after_heldout": after_heldout,
        "before_train_by_prompt": before_train_by_prompt,
        "after_train_by_prompt": after_train_by_prompt,
        "history": history,
        "policy": policy,
        "reference": reference,
        "vocab": vocab,
        "prompts": prompts,
    }
