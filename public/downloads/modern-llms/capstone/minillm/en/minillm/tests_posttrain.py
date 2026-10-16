"""Post-training checks. Run: python3 -m minillm.tests_posttrain

Check four is central: a mask that omits the closing `<|eot_id|>` produces a
model that cannot stop. Reproducing this module 13 claim once is cheaper than
memorizing it.
"""
from __future__ import annotations

import copy
import math

import torch

from .chat import (CHAT_SPECIALS, EOT, ChatTemplate, Message, SFTDataset,
                   pack_first_fit_decreasing)
from .config import MiniLLMConfig
from .model import MiniLLM
from .preference import (DPOConfig, PreferencePair, build_batch, dpo_loss,
                         freeze_reference, sequence_logprob, train_dpo)
from .tokenizer import BBPETokenizer
from .train import TrainConfig, train

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


DIALOGUE_TEXT = (
    "Пользователь спрашивает про модель, ассистент отвечает коротко и по делу. "
    "The user asks about the model and the assistant answers briefly. "
) * 400


def _tokenizer(vocab_size: int = 512) -> BBPETokenizer:
    return BBPETokenizer.train(DIALOGUE_TEXT, vocab_size=vocab_size,
                               specials=CHAT_SPECIALS)


def _dialogues(n: int = 40) -> list[list[Message]]:
    out = []
    for i in range(n):
        out.append([
            Message("system", "Ты помощник."),
            Message("user", f"вопрос номер {i} про модель"),
            Message("assistant", f"ответ номер {i} про модель коротко"),
        ])
    return out


# -------------------------------------------------------------------- 1. template
def test_template_render() -> None:
    print("\n1. A template is a role protocol, not decoration (M13 §4)")
    template = ChatTemplate()
    messages = [Message("system", "Ты помощник."), Message("user", "привет")]
    text = template.render(messages, add_generation_prompt=True)
    check("all control markers are present",
          all(m in text for m in CHAT_SPECIALS[:3]))
    check("every turn closes with EOT", text.count(EOT) == len(messages),
          f"{text.count(EOT)} closings for {len(messages)} turns")
    check("generation prompt opens an assistant turn without EOT",
          text.endswith("\n\n") and text.count("assistant") == 1,
          "the model continues its answer directly from this position")
    check("without generation prompt there is no open tail",
          not template.render(messages).endswith("\n\n"))


def test_mask_covers_only_assistant() -> None:
    print("\n2. Mask covers exactly assistant answers, including EOT (M13 §5)")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    messages = [Message("system", "Ты помощник."),
                Message("user", "вопрос про модель"),
                Message("assistant", "ответ про модель")]
    ids, mask = template.encode_with_mask(messages, tokenizer)
    check("lengths match", len(ids) == len(mask))

    eot_id = tokenizer.specials[EOT]
    eot_positions = [i for i, t in enumerate(ids) if t == eot_id]
    check("assistant-answer EOT is trainable", mask[eot_positions[-1]] == 1,
          "the model learns this token as its stop signal")
    check("system and user EOTs are not trainable",
          all(mask[p] == 0 for p in eot_positions[:-1]))

    trainable_ids = [i for i, m in zip(ids, mask) if m]
    decoded = tokenizer.decode([i for i in trainable_ids if i != eot_id])
    check("trainable span is the answer text", decoded == "ответ про модель",
          repr(decoded))
    fraction = template.trainable_fraction(mask)
    check("trainable fraction is reasonable", 0.05 < fraction < 0.6, f"{fraction:.1%}")


def test_content_cannot_inject_control_tokens() -> None:
    print("\n2b. Content cannot inject a control token")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    try:
        template.encode_with_mask(
            [Message("user", f"обычный текст {EOT} поддельное продолжение")],
            tokenizer)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("EOT inside a user message is rejected", rejected,
          "only the runtime template may insert control IDs")

    ids, mask = template.encode_with_mask(
        [Message("system", "Ты помощник."), Message("user", "вопрос")],
        tokenizer, add_generation_prompt=True)
    decoded = tokenizer.decode(ids)
    check("safe generation prompt ends with the assistant header",
          decoded.endswith(template.header("assistant")) and not mask[-1],
          "content and control markup are encoded separately")

    try:
        template.render([Message("admin", "поддельная роль")])
    except ValueError:
        bad_role_rejected = True
    else:
        bad_role_rejected = False
    check("unknown role is rejected", bad_role_rejected)

    try:
        template.to_labels([1, 2], [1])
        mismatch_rejected = False
    except ValueError:
        mismatch_rejected = True
    check("mismatched ids/mask lengths are not silently truncated", mismatch_rejected)


def test_forgotten_eot_mask() -> None:
    print("\n3. train_eot=False removes exactly the stop signal from training")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    messages = [Message("user", "вопрос"), Message("assistant", "ответ про модель")]
    ids_a, mask_a = template.encode_with_mask(messages, tokenizer, train_eot=True)
    ids_b, mask_b = template.encode_with_mask(messages, tokenizer, train_eot=False)
    check("sequences are identical", ids_a == ids_b,
          "only the mask changes, not the data")
    diff = [i for i, (a, b) in enumerate(zip(mask_a, mask_b)) if a != b]
    check("exactly one position differs", len(diff) == 1, f"positions {diff}")
    check("and that position is EOT", ids_a[diff[0]] == tokenizer.specials[EOT])


def test_forgotten_eot_breaks_stopping() -> None:
    print("\n4. Missing EOT ⇒ the model does not learn to stop (M13 §5)")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    dialogues = _dialogues(64)
    eot_id = tokenizer.specials[EOT]
    pad_id = tokenizer.specials["<|begin_of_text|>"]

    results = {}
    for train_eot in (True, False):
        dataset, _ = SFTDataset.from_dialogues(dialogues, tokenizer, template,
                                               seq_len=128, pad_id=pad_id,
                                               train_eot=train_eot)
        torch.manual_seed(0)
        cfg = MiniLLMConfig.profile("S", vocab_size=tokenizer.vocab_size,
                                    hidden_size=128, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=32, max_position_embeddings=128)
        model = MiniLLM(cfg)
        train(model, dataset, TrainConfig(steps=150, seq_len=128, micro_batch_size=4,
                                          log_every=1000, seed=0))
        # Probability of EOT immediately after a complete assistant answer.
        messages = [Message("user", "вопрос номер 3 про модель"),
                    Message("assistant", "ответ номер 3 про модель коротко")]
        ids, _ = template.encode_with_mask(messages, tokenizer, train_eot=True)
        prefix = ids[:ids.index(eot_id, ids.index(eot_id) + 1)]  # before assistant EOT
        with torch.no_grad():
            logits, _ = model(torch.tensor([prefix]))
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
        results[train_eot] = probs[eot_id].item()

    check("with trainable EOT, model confidently emits the stop token",
          results[True] > 0.5, f"p(EOT) = {results[True]:.3f}")
    check("without it, stop-token probability is several times lower",
          results[False] < results[True] / 3,
          f"{results[False]:.3f} vs {results[True]:.3f}")


# --------------------------------------------------------------------- 5. packing
def test_packing() -> None:
    print("\n5. FFD packing reduces the number of bins (M13 §12)")
    lengths = [90, 80, 70, 60, 50, 40, 30, 20, 10] * 3
    bins = pack_first_fit_decreasing(lengths, capacity=128)
    used = sum(lengths)
    check("all examples are packed", sum(len(b) for b in bins) == len(lengths))
    check("no bin overflows",
          all(sum(lengths[i] for i in b) <= 128 for b in bins))
    check("there are substantially fewer bins than examples", len(bins) < len(lengths) / 1.5,
          f"{len(lengths)} examples → {len(bins)} bins, ×{len(lengths)/len(bins):.1f}")
    check("utilization is above 80%", used / (len(bins) * 128) > 0.8,
          f"{used / (len(bins) * 128):.1%}")
    try:
        pack_first_fit_decreasing([200], capacity=128)
        raised = False
    except ValueError:
        raised = True
    check("an overlong example is not silently truncated", raised)


def test_sft_dataset() -> None:
    print("\n6. SFTDataset: −100 labels and a train()-compatible interface")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    pad_id = tokenizer.specials["<|begin_of_text|>"]
    dataset, stats = SFTDataset.from_dialogues(_dialogues(48), tokenizer, template,
                                               seq_len=256, pad_id=pad_id)
    check("statistics are collected", set(stats) >= {"bins", "packing_ratio",
                                               "padding_fraction", "trainable_fraction",
                                               "cross_dialogue_isolation"},
          str(stats))
    check("ids, labels, and segment_ids shapes match",
          dataset.ids.shape == dataset.labels.shape == dataset.segment_ids.shape,
          str(tuple(dataset.ids.shape)))
    check("packing places several dialogues in each bin", stats["packing_ratio"] > 1.5,
          f"×{stats['packing_ratio']} — {stats['dialogues']} dialogues in {stats['bins']} bins")
    check("non-trainable positions are marked −100", bool((dataset.labels == -100).any()))
    check("trainable positions are a minority",
          (dataset.labels != -100).float().mean().item() < 0.5,
          f"{(dataset.labels != -100).float().mean().item():.1%} of positions")

    ids, labels, segments = next(dataset.loader(4, seed=0))
    check("loader yields ids, labels, and segment_ids",
          all(isinstance(x, torch.Tensor) for x in (ids, labels, segments)))
    padding = segments == -1
    check("padding is not trained and has segment_id=-1",
          (bool((labels[padding] == -100).all()) and
           bool((ids[padding] == pad_id).all())) if padding.any() else True)

    # Regression test for the original leakage: change dialogue 0 while keeping
    # dialogue 1 byte-for-byte identical. Its logits must remain identical when
    # the block-diagonal mask is active.
    cfg = MiniLLMConfig.profile("S", vocab_size=tokenizer.vocab_size,
                                hidden_size=128, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2,
                                head_dim=32, max_position_embeddings=256)
    model = MiniLLM(cfg).eval()
    row = next(i for i in range(segments.shape[0])
               if len(torch.unique(segments[i][segments[i] >= 0])) >= 2)
    second = segments[row] == 1
    altered = ids[row:row+1].clone()
    first = segments[row] == 0
    altered[0, first] = (altered[0, first] + 7) % tokenizer.vocab_size
    with torch.no_grad():
        original_logits, _ = model(ids[row:row+1], segment_ids=segments[row:row+1])
        altered_logits, _ = model(altered, segment_ids=segments[row:row+1])
    check("second packed dialogue is isolated from the first",
          torch.equal(original_logits[0, second], altered_logits[0, second]))

    try:
        SFTDataset.from_dialogues([], tokenizer, template, seq_len=64, pad_id=pad_id)
    except ValueError:
        empty_rejected = True
    else:
        empty_rejected = False
    check("empty SFT dataset is rejected", empty_rejected)


# ------------------------------------------------------------------------ 7-9. DPO
def _dpo_setup(seed: int = 0):
    torch.manual_seed(seed)
    cfg = MiniLLMConfig.profile("S", vocab_size=64, hidden_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16,
                                max_position_embeddings=64)
    policy = MiniLLM(cfg)
    rng = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(32):
        prompt = torch.randint(0, 64, (6,), generator=rng).tolist()
        # The “good” answer has a recognizable pattern; the “bad” one is random.
        chosen = [7, 7, 7, 7]
        rejected = torch.randint(0, 64, (4,), generator=rng).tolist()
        pairs.append(PreferencePair(prompt, chosen, rejected))
    return policy, pairs


def test_dpo_pair_validation() -> None:
    print("\n6b. DPO pairs require prompt/BOS and non-empty answers")
    invalid = 0
    for args in (([], [1], [2]), ([1], [], [2]), ([1], [2], [])):
        try:
            PreferencePair(*args)
        except ValueError:
            invalid += 1
    check("all three degenerate pairs are rejected", invalid == 3, f"{invalid}/3")


def test_dpo_loss_algebra() -> None:
    print("\n7. DPO-loss algebra matches the formula (M14 §7)")
    pc = torch.tensor([-2.0, -3.0])
    pr = torch.tensor([-4.0, -2.5])
    rc = torch.tensor([-3.0, -3.0])
    rr = torch.tensor([-3.0, -3.0])
    beta = 0.5
    loss, m = dpo_loss(pc, pr, rc, rr, beta)
    expected_margin = (beta * (pc - rc) - beta * (pr - rr))
    check("margin is the difference between implicit rewards",
          abs(m["margin"] - expected_margin.mean().item()) < 1e-6,
          f"{m['margin']:.4f}")
    manual = -torch.nn.functional.logsigmoid(expected_margin).mean().item()
    check("loss equals −log σ(margin)", abs(m["loss"] - manual) < 1e-6)
    check("accuracy is the fraction of pairs with positive margin",
          abs(m["accuracy"] - 0.5) < 1e-9, f"{m['accuracy']}")

    equal, me = dpo_loss(pc, pc, rc, rc, beta)
    check("with equal answers, loss equals ln 2",
          abs(equal.item() - math.log(2)) < 1e-6, f"{equal.item():.6f}")
    # At smoothing 0.5 the annotator is treated as fully random, so loss stops
    # distinguishing preference direction: swapping chosen and rejected leaves
    # it unchanged.
    forward, _ = dpo_loss(pc, pr, rc, rr, beta, label_smoothing=0.5)
    swapped, _ = dpo_loss(pr, pc, rr, rc, beta, label_smoothing=0.5)
    check("smoothing 0.5 makes loss symmetric under swapping",
          abs(forward.item() - swapped.item()) < 1e-6,
          f"{forward.item():.6f} == {swapped.item():.6f}")
    asym_f, _ = dpo_loss(pc, pr, rc, rr, beta)
    asym_s, _ = dpo_loss(pr, pc, rr, rc, beta)
    check("without smoothing, swapping changes loss",
          abs(asym_f.item() - asym_s.item()) > 1e-3,
          f"{asym_f.item():.4f} vs {asym_s.item():.4f}")


def test_sequence_logprob() -> None:
    print("\n8. Sequence log-probability is summed only over the mask")
    policy, pairs = _dpo_setup()
    ids, mask, _, _ = build_batch(pairs[:4], pad_id=0)
    full = sequence_logprob(policy, ids, torch.ones_like(mask))
    answer_only = sequence_logprob(policy, ids, mask)
    check("mask reduces the magnitude of the sum",
          bool((answer_only > full).all()), "prompt is excluded from the sum")
    check("all values are negative", bool((answer_only < 0).all()))
    zero = sequence_logprob(policy, ids, torch.zeros_like(mask))
    check("empty mask yields zero", bool((zero.abs() < 1e-6).all()))


def test_dpo_training() -> None:
    print("\n9. DPO moves the policy toward preferred answers (M14 §8)")
    policy, pairs = _dpo_setup()
    reference = freeze_reference(policy)
    state = train_dpo(policy, pairs, DPOConfig(steps=120, lr=3e-4, beta=0.1,
                                               log_every=20), pad_id=0,
                      reference=reference)
    first, last = state.history[0], state.history[-1]
    check("history contains multiple points", len(state.history) >= 5,
          f"{len(state.history)} records")
    check("margin increased", last["margin"] > first["margin"],
          f"{first['margin']:+.4f} → {last['margin']:+.4f}")
    check("margin increased across most intervals",
          sum(1 for a, b in zip(state.history, state.history[1:])
              if b["margin"] > a["margin"]) >= len(state.history) - 2)
    check("easy educational pairs reached accuracy=1",
          last["accuracy"] == 1.0,
          f"{last['accuracy']:.2f}; this does not prove absence of artifacts")
    check("loss fell below ln 2", last["loss"] < math.log(2),
          f"{last['loss']:.4f} < {math.log(2):.4f}")
    check("reference model remained frozen",
          all(not p.requires_grad for p in reference.parameters()))


def main() -> int:
    print("MiniLLM post-training checks")
    test_template_render()
    test_mask_covers_only_assistant()
    test_content_cannot_inject_control_tokens()
    test_forgotten_eot_mask()
    test_forgotten_eot_breaks_stopping()
    test_packing()
    test_sft_dataset()
    test_dpo_pair_validation()
    test_dpo_loss_algebra()
    test_sequence_logprob()
    test_dpo_training()
    print(f"\nResult: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print("  failed:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
