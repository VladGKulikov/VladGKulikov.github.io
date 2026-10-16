"""Preference optimization: DPO (M14).

The idea in one line: RLHF trains a reward model and then optimizes a policy
against it with RL. DPO observes that, under a KL-constrained objective, the
optimal policy expresses the reward analytically. The reward can therefore be
eliminated, optimizing the policy **directly** on “better / worse” pairs.

Loss:

    L = −log σ( β · [ (log π(y_w|x) − log π_ref(y_w|x))
                    − (log π(y_l|x) − log π_ref(y_l|x)) ] )

The bracketed quantity is the difference between **implicit rewards**. This
gives two diagnostics that should always be logged (M14 §8):

  * `margin` — mean implicit-reward difference; it should increase;
  * `accuracy` — fraction of pairs where the preferred answer receives the
    higher reward. A value near 0.5 means the pairs are not separated yet. A
    rapid approach to 1.0 is a diagnostic signal: the pairs may truly be easy,
    but a template artifact, leakage, or overfitting is also possible. The
    number alone does not establish the cause.

Role of `β`: it controls the KL anchor to the reference policy. A small β lets
the model move far from the SFT checkpoint; a large β keeps it nearby. The
reference model is frozen and is not needed for inference, only for training.

The full derivation from the KL-constrained objective belongs in the RL-LLM
course; here we study the mechanism and what can be verified in code.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import MiniLLM
from .precision import autocast_context, make_grad_scaler, resolve_compute_dtype


@dataclass(frozen=True)
class PreferencePair:
    """One preference example: a shared prompt and better/worse answers."""

    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]

    def __post_init__(self) -> None:
        # Causal scoring predicts the first answer token from the last prompt
        # token. An empty prompt would silently drop that first answer token
        # after the standard one-position shift. Require an explicit BOS or
        # another non-empty prompt prefix instead.
        if not self.prompt_ids:
            raise ValueError("DPO-пара должна содержать непустой prompt (например, BOS)")
        if not self.chosen_ids or not self.rejected_ids:
            raise ValueError("chosen и rejected ответы должны быть непустыми")


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 1e-4
    steps: int = 100
    batch_size: int = 4
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    compute_dtype: str = "auto"   # auto | fp32 | bf16 | fp16
    log_every: int = 10
    label_smoothing: float = 0.0     # cDPO: annotator-label smoothing


def sequence_logprob(model: MiniLLM, ids: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Sum log p(token | prefix) over positions selected by the mask.

    ids [B, T], mask [B, T] -> [B]. The mask refers to the same positions as
    labels: token t is predicted from position t−1, so we compare
    logits[:, :-1] with ids[:, 1:].
    """
    logits, _ = model(ids)
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    target = ids[:, 1:].unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, target).squeeze(-1)     # [B, T-1]
    return (token_logprobs * mask[:, 1:]).sum(dim=-1)


def build_batch(pairs: list[PreferencePair], pad_id: int, device: str = "cpu"):
    """Build four tensors: ids and answer masks for chosen and rejected.

    The mask covers only the answer. The prompt is shared by both branches, so
    including it in the log-probability sum is pointless: it cancels in the
    difference.
    """
    if not pairs:
        raise ValueError("build_batch требует хотя бы одну preference-пару")

    def encode(side: str):
        rows, masks = [], []
        for p in pairs:
            answer = p.chosen_ids if side == "chosen" else p.rejected_ids
            rows.append(p.prompt_ids + answer)
            masks.append([0] * len(p.prompt_ids) + [1] * len(answer))
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width))
        for i, (r, m) in enumerate(zip(rows, masks)):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
        return ids.to(device), mask.to(device)

    chosen_ids, chosen_mask = encode("chosen")
    rejected_ids, rejected_mask = encode("rejected")
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float, label_smoothing: float = 0.0):
    """Return (loss, metrics); every input is a [B] vector of summed log-probs."""
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward

    if label_smoothing > 0:
        # cDPO: annotators sometimes err, so penalizing confidence can help.
        loss = -(1 - label_smoothing) * F.logsigmoid(margin) \
               - label_smoothing * F.logsigmoid(-margin)
    else:
        loss = -F.logsigmoid(margin)

    metrics = {
        "loss": loss.mean().item(),
        "margin": margin.mean().item(),
        "accuracy": (margin > 0).float().mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
    return loss.mean(), metrics


def freeze_reference(model: MiniLLM) -> MiniLLM:
    """A frozen policy copy that serves as the KL-constraint anchor."""
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    return reference


@dataclass
class DPOState:
    step: int = 0
    history: list[dict] = field(default_factory=list)


def train_dpo(policy: MiniLLM, pairs: list[PreferencePair], cfg: DPOConfig,
              pad_id: int, reference: MiniLLM | None = None) -> DPOState:
    """DPO loop; freeze a policy copy at startup if no reference is supplied."""
    if not pairs:
        raise ValueError("для DPO нужна хотя бы одна preference-пара")
    if not 0.0 <= cfg.label_smoothing <= 0.5:
        raise ValueError("label_smoothing должен лежать в интервале [0, 0.5]")
    torch.manual_seed(cfg.seed)
    if reference is None:
        reference = freeze_reference(policy)
    else:
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    policy.to(cfg.device).train()
    reference.to(cfg.device).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    compute_dtype = resolve_compute_dtype(cfg.device, cfg.compute_dtype)
    scaler = make_grad_scaler(cfg.device, compute_dtype)
    state = DPOState()
    generator = torch.Generator().manual_seed(cfg.seed)

    for step in range(cfg.steps):
        index = torch.randint(0, len(pairs), (cfg.batch_size,), generator=generator)
        batch = [pairs[int(i)] for i in index]
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = build_batch(
            batch, pad_id, cfg.device)

        with torch.no_grad(), autocast_context(cfg.device, compute_dtype):
            ref_chosen = sequence_logprob(reference, chosen_ids, chosen_mask)
            ref_rejected = sequence_logprob(reference, rejected_ids, rejected_mask)
        with autocast_context(cfg.device, compute_dtype):
            pol_chosen = sequence_logprob(policy, chosen_ids, chosen_mask)
            pol_rejected = sequence_logprob(policy, rejected_ids, rejected_mask)
            loss, metrics = dpo_loss(
                pol_chosen, pol_rejected, ref_chosen, ref_rejected,
                cfg.beta, cfg.label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        state.step = step + 1
        if state.step % cfg.log_every == 0 or state.step == cfg.steps:
            metrics["step"] = state.step
            metrics["compute_dtype"] = compute_dtype
            state.history.append(metrics)
            print(f"  step {state.step:>4} | loss {metrics['loss']:.4f} "
                  f"| margin {metrics['margin']:+.4f} "
                  f"| acc {metrics['accuracy']:.2f}")
    return state
