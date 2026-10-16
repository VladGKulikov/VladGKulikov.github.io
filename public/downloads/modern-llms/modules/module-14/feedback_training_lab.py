"""Complete CPU feedback loops on a restricted one-token answer space.

Prompt: is a > b? Answer: NO or YES, followed by a deterministic stop.
This is a contextual bandit, not a language model or a reasoning benchmark.
"""
import copy
import io
import math
import importlib.util
import subprocess
import sys
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch'])
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(140)
pairs = [(a, b) for a in range(8) for b in range(8)]
train_pairs = [p for p in pairs if abs(p[0]-p[1]) != 4]
test_pairs = [p for p in pairs if abs(p[0]-p[1]) == 4]
assert len(train_pairs) == 56 and len(test_pairs) == 8
assert not set(train_pairs) & set(test_pairs)

def features(prompts):
    # Deliberately useful, declared features; no claim of learned arithmetic.
    return torch.tensor([[1., (a-b)/7.] for a, b in prompts])

def verifier(prompts, answer_ids):
    correct = torch.tensor([int(a > b) for a, b in prompts])
    return answer_ids.eq(correct).float()

class AnswerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2, 2))
    def forward(self, prompts):
        return F.log_softmax(features(prompts) @ self.weights, dim=-1)

def chosen_logps(model, prompts, ids):
    # One stochastic completion token; the fixed stop has log probability zero.
    return model(prompts).gather(1, ids[:, None]).squeeze(1)

@torch.no_grad()
def metrics(model, prompts, reference):
    lp = model(prompts)
    correct = torch.tensor([int(a > b) for a, b in prompts])
    return dict(correct_probability=float(lp.exp().gather(1, correct[:, None]).mean()),
                greedy_accuracy=float(verifier(prompts, lp.argmax(-1)).mean()),
                kl=float((lp.exp() * (lp-reference(prompts))).sum(-1).mean()))

def reload_check(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = AnswerPolicy()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(model(test_pairs), restored(test_pairs))

def run_dpo(beta=.5, *, steps=140, seed=140, init_std=0., verbose=True):
    """Train on fixed pairs; seed controls only optional random initialization.

    The reference copies the initial policy. Match seed and init_std across beta
    values to hold both starting weights and cached reference scores constant.
    With init_std=0 the full-batch experiment is deterministic for every seed.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if not math.isfinite(init_std) or init_std < 0:
        raise ValueError('init_std must be finite and nonnegative')
    model = AnswerPolicy()
    if init_std > 0:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            model.weights.normal_(0., init_std, generator=generator)
    reference = copy.deepcopy(model).requires_grad_(False)
    frozen = reference.weights.detach().clone()
    chosen = torch.tensor([int(a > b) for a, b in train_pairs])
    rejected = 1-chosen
    with torch.no_grad():
        cached_gap = chosen_logps(reference, train_pairs, chosen)-chosen_logps(reference, train_pairs, rejected)
        initial_gap = chosen_logps(model, train_pairs, chosen)-chosen_logps(model, train_pairs, rejected)
        initial_loss = float(F.softplus(-beta*(initial_gap-cached_gap)).mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=.12)
    before = metrics(model, test_pairs, reference)
    for _ in range(steps):
        gap = chosen_logps(model, train_pairs, chosen)-chosen_logps(model, train_pairs, rejected)
        loss = F.softplus(-beta*(gap-cached_gap)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        gap = chosen_logps(model, train_pairs, chosen)-chosen_logps(model, train_pairs, rejected)
        final_loss = float(F.softplus(-beta*(gap-cached_gap)).mean())
        pair_accuracy = float((gap-cached_gap > 0).float().mean())
    after = metrics(model, test_pairs, reference)
    # Check the computation, not a quality threshold tied to one beta and budget.
    assert math.isfinite(final_loss) and torch.isfinite(model.weights).all()
    assert torch.equal(frozen, reference.weights) and reference.weights.grad is None
    reload_check(model)
    result = dict(beta=beta, steps=steps, seed=seed, init_std=init_std,
                  initial_weights=frozen.tolist(), before=before, after=after,
                  initial_loss=initial_loss, loss=final_loss,
                  pair_accuracy=pair_accuracy, pairs=len(train_pairs))
    if verbose:
        print('DPO:', result)
    return result

def run_grpo():
    torch.manual_seed(141)
    model = AnswerPolicy()
    reference = copy.deepcopy(model).requires_grad_(False)
    frozen = reference.weights.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.08)
    before = metrics(model, test_pairs, reference)
    group_size, steps, updates_per_rollout = 8, 100, 2
    zero_groups = 0
    clips = []
    expanded_prompts = [p for p in train_pairs for _ in range(group_size)]
    for _ in range(steps):
        # This snapshot describes the sampler. The reference stays fixed throughout.
        with torch.no_grad():
            sampler_lp = model(train_pairs)
            answers = torch.multinomial(sampler_lp.exp(), group_size, replacement=True)
            old_lp = sampler_lp.gather(1, answers)
            rewards = verifier(expanded_prompts, answers.flatten()).reshape(-1, group_size)
            centered = rewards-rewards.mean(1, keepdim=True)
            std = rewards.std(1, correction=0, keepdim=True)
            valid = std > 0
            advantages = torch.where(valid, centered/std.clamp_min(1e-8), torch.zeros_like(centered))
            zero_groups += int((~valid).sum())
        for _ in range(updates_per_rollout):
            lp = model(train_pairs)
            new_lp = lp.gather(1, answers)
            ratio = (new_lp-old_lp).exp()
            clipped = ratio.clamp(.8, 1.2)
            surrogate = torch.minimum(ratio*advantages, clipped*advantages).mean()
            # Exact categorical KL is available for this two-action teaching model.
            kl = (lp.exp()*(lp-reference(train_pairs))).sum(-1).mean()
            loss = -surrogate + .02*kl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            clips.append(float(((ratio < .8) | (ratio > 1.2)).float().mean()))
    after = metrics(model, test_pairs, reference)
    assert after['correct_probability'] > before['correct_probability'] + .3
    assert not advantages.requires_grad and not old_lp.requires_grad
    assert torch.equal(frozen, reference.weights) and reference.weights.grad is None
    reload_check(model)
    result = dict(before=before, after=after, generated_answers=steps*len(train_pairs)*group_size,
                  optimizer_steps=steps*updates_per_rollout, zero_groups=zero_groups,
                  mean_clip_fraction=sum(clips)/len(clips), zero_group_policy='zero advantages; keep KL')
    print('GRPO:', result)
    return result

if __name__ == '__main__':
    dpo_training_result = run_dpo()
    grpo_training_result = run_grpo()
