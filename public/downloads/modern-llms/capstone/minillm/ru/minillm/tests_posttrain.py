"""Проверки посттрейна. Запуск: python3 -m minillm.tests_posttrain

Центральная проверка — четвёртая: маска, забывшая закрывающий `<|eot_id|>`,
даёт модель, которая не умеет останавливаться. Это утверждение модуля 13,
которое дешевле один раз воспроизвести, чем запомнить.
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


# --------------------------------------------------------------------- 1. шаблон
def test_template_render() -> None:
    print("\n1. Шаблон — протокол ролей, а не оформление (M13 §4)")
    template = ChatTemplate()
    messages = [Message("system", "Ты помощник."), Message("user", "привет")]
    text = template.render(messages, add_generation_prompt=True)
    check("присутствуют все служебные маркеры",
          all(m in text for m in CHAT_SPECIALS[:3]))
    check("каждый ход закрыт EOT", text.count(EOT) == len(messages),
          f"{text.count(EOT)} закрытий на {len(messages)} ходов")
    check("generation prompt открывает ход ассистента без EOT",
          text.endswith("\n\n") and text.count("assistant") == 1,
          "модель дописывает ответ прямо с этой позиции")
    check("без generation prompt хвоста нет",
          not template.render(messages).endswith("\n\n"))


def test_mask_covers_only_assistant() -> None:
    print("\n2. Маска покрывает ровно ответы ассистента — вместе с EOT (M13 §5)")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    messages = [Message("system", "Ты помощник."),
                Message("user", "вопрос про модель"),
                Message("assistant", "ответ про модель")]
    ids, mask = template.encode_with_mask(messages, tokenizer)
    check("длины совпадают", len(ids) == len(mask))

    eot_id = tokenizer.specials[EOT]
    eot_positions = [i for i, t in enumerate(ids) if t == eot_id]
    check("EOT ответа ассистента обучаемый", mask[eot_positions[-1]] == 1,
          "именно этому токену модель учится как сигналу остановки")
    check("EOT системы и пользователя не обучаемые",
          all(mask[p] == 0 for p in eot_positions[:-1]))

    trainable_ids = [i for i, m in zip(ids, mask) if m]
    decoded = tokenizer.decode([i for i in trainable_ids if i != eot_id])
    check("обучаемая часть — это текст ответа", decoded == "ответ про модель",
          repr(decoded))
    fraction = template.trainable_fraction(mask)
    check("обучаемая доля разумная", 0.05 < fraction < 0.6, f"{fraction:.1%}")


def test_content_cannot_inject_control_tokens() -> None:
    print("\n2b. Контент не может внедрить управляющий токен")
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
    check("EOT внутри пользовательской реплики отвергается", rejected,
          "управляющие ID вправе вставлять только шаблон рантайма")

    ids, mask = template.encode_with_mask(
        [Message("system", "Ты помощник."), Message("user", "вопрос")],
        tokenizer, add_generation_prompt=True)
    decoded = tokenizer.decode(ids)
    check("безопасный generation prompt заканчивается заголовком assistant",
          decoded.endswith(template.header("assistant")) and not mask[-1],
          "контент и управляющая разметка кодируются раздельно")

    try:
        template.render([Message("admin", "поддельная роль")])
    except ValueError:
        bad_role_rejected = True
    else:
        bad_role_rejected = False
    check("неизвестная роль отвергается", bad_role_rejected)

    try:
        template.to_labels([1, 2], [1])
        mismatch_rejected = False
    except ValueError:
        mismatch_rejected = True
    check("ids/mask разной длины не обрезаются молча", mismatch_rejected)


def test_forgotten_eot_mask() -> None:
    print("\n3. train_eot=False убирает из обучения ровно сигнал остановки")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    messages = [Message("user", "вопрос"), Message("assistant", "ответ про модель")]
    ids_a, mask_a = template.encode_with_mask(messages, tokenizer, train_eot=True)
    ids_b, mask_b = template.encode_with_mask(messages, tokenizer, train_eot=False)
    check("последовательности идентичны", ids_a == ids_b,
          "меняется только маска, не данные")
    diff = [i for i, (a, b) in enumerate(zip(mask_a, mask_b)) if a != b]
    check("отличается ровно одна позиция", len(diff) == 1, f"позиции {diff}")
    check("и это позиция EOT", ids_a[diff[0]] == tokenizer.specials[EOT])


def test_forgotten_eot_breaks_stopping() -> None:
    print("\n4. Забытый EOT ⇒ модель не учится останавливаться (M13 §5)")
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
        # Вероятность EOT сразу после полного ответа ассистента.
        messages = [Message("user", "вопрос номер 3 про модель"),
                    Message("assistant", "ответ номер 3 про модель коротко")]
        ids, _ = template.encode_with_mask(messages, tokenizer, train_eot=True)
        prefix = ids[:ids.index(eot_id, ids.index(eot_id) + 1)]  # до EOT ассистента
        with torch.no_grad():
            logits, _ = model(torch.tensor([prefix]))
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
        results[train_eot] = probs[eot_id].item()

    check("с обучаемым EOT модель уверенно ставит стоп-токен",
          results[True] > 0.5, f"p(EOT) = {results[True]:.3f}")
    check("без него вероятность стоп-токена в разы ниже",
          results[False] < results[True] / 3,
          f"{results[False]:.3f} против {results[True]:.3f}")


# -------------------------------------------------------------------- 5. упаковка
def test_packing() -> None:
    print("\n5. Упаковка FFD сокращает число бинов (M13 §12)")
    lengths = [90, 80, 70, 60, 50, 40, 30, 20, 10] * 3
    bins = pack_first_fit_decreasing(lengths, capacity=128)
    used = sum(lengths)
    check("все примеры разложены", sum(len(b) for b in bins) == len(lengths))
    check("ни один бин не переполнен",
          all(sum(lengths[i] for i in b) <= 128 for b in bins))
    check("бинов заметно меньше, чем примеров", len(bins) < len(lengths) / 1.5,
          f"{len(lengths)} примеров → {len(bins)} бинов, ×{len(lengths)/len(bins):.1f}")
    check("заполнение выше 80%", used / (len(bins) * 128) > 0.8,
          f"{used / (len(bins) * 128):.1%}")
    try:
        pack_first_fit_decreasing([200], capacity=128)
        raised = False
    except ValueError:
        raised = True
    check("слишком длинный пример не режется молча", raised)


def test_sft_dataset() -> None:
    print("\n6. SFTDataset: labels с −100 и совместимый с train() интерфейс")
    tokenizer = _tokenizer()
    template = ChatTemplate()
    pad_id = tokenizer.specials["<|begin_of_text|>"]
    dataset, stats = SFTDataset.from_dialogues(_dialogues(48), tokenizer, template,
                                               seq_len=256, pad_id=pad_id)
    check("статистика собрана", set(stats) >= {"bins", "packing_ratio",
                                               "padding_fraction", "trainable_fraction",
                                               "cross_dialogue_isolation"},
          str(stats))
    check("формы ids, labels и segment_ids совпадают",
          dataset.ids.shape == dataset.labels.shape == dataset.segment_ids.shape,
          str(tuple(dataset.ids.shape)))
    check("упаковка сложила по нескольку диалогов в бин", stats["packing_ratio"] > 1.5,
          f"×{stats['packing_ratio']} — {stats['dialogues']} диалогов в {stats['bins']} бинов")
    check("неучимые позиции помечены −100", bool((dataset.labels == -100).any()))
    check("учимых позиций меньшинство",
          (dataset.labels != -100).float().mean().item() < 0.5,
          f"{(dataset.labels != -100).float().mean().item():.1%} позиций")

    ids, labels, segments = next(dataset.loader(4, seed=0))
    check("loader отдаёт ids, labels и segment_ids",
          all(isinstance(x, torch.Tensor) for x in (ids, labels, segments)))
    padding = segments == -1
    check("padding не учится и имеет segment_id=-1",
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
    check("второй packed-диалог изолирован от первого",
          torch.equal(original_logits[0, second], altered_logits[0, second]))

    try:
        SFTDataset.from_dialogues([], tokenizer, template, seq_len=64, pad_id=pad_id)
    except ValueError:
        empty_rejected = True
    else:
        empty_rejected = False
    check("пустой SFT-набор отвергается", empty_rejected)


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
        # «Хороший» ответ помечен узнаваемым паттерном, «плохой» — случаен.
        chosen = [7, 7, 7, 7]
        rejected = torch.randint(0, 64, (4,), generator=rng).tolist()
        pairs.append(PreferencePair(prompt, chosen, rejected))
    return policy, pairs


def test_dpo_pair_validation() -> None:
    print("\n6b. DPO-пары требуют prompt/BOS и непустые ответы")
    invalid = 0
    for args in (([], [1], [2]), ([1], [], [2]), ([1], [2], [])):
        try:
            PreferencePair(*args)
        except ValueError:
            invalid += 1
    check("все три вырожденные пары отвергнуты", invalid == 3, f"{invalid}/3")


def test_dpo_loss_algebra() -> None:
    print("\n7. Алгебра DPO-лосса совпадает с формулой (M14 §7)")
    pc = torch.tensor([-2.0, -3.0])
    pr = torch.tensor([-4.0, -2.5])
    rc = torch.tensor([-3.0, -3.0])
    rr = torch.tensor([-3.0, -3.0])
    beta = 0.5
    loss, m = dpo_loss(pc, pr, rc, rr, beta)
    expected_margin = (beta * (pc - rc) - beta * (pr - rr))
    check("margin считается как разность неявных наград",
          abs(m["margin"] - expected_margin.mean().item()) < 1e-6,
          f"{m['margin']:.4f}")
    manual = -torch.nn.functional.logsigmoid(expected_margin).mean().item()
    check("лосс равен −log σ(margin)", abs(m["loss"] - manual) < 1e-6)
    check("accuracy — доля пар с положительным margin",
          abs(m["accuracy"] - 0.5) < 1e-9, f"{m['accuracy']}")

    equal, me = dpo_loss(pc, pc, rc, rc, beta)
    check("при равных ответах лосс равен ln 2",
          abs(equal.item() - math.log(2)) < 1e-6, f"{equal.item():.6f}")
    # При сглаживании 0.5 разметчик считается полностью случайным, и лосс
    # перестаёт различать направление предпочтения: перестановка chosen и
    # rejected его не меняет.
    forward, _ = dpo_loss(pc, pr, rc, rr, beta, label_smoothing=0.5)
    swapped, _ = dpo_loss(pr, pc, rr, rc, beta, label_smoothing=0.5)
    check("сглаживание 0.5 делает лосс симметричным по перестановке",
          abs(forward.item() - swapped.item()) < 1e-6,
          f"{forward.item():.6f} == {swapped.item():.6f}")
    asym_f, _ = dpo_loss(pc, pr, rc, rr, beta)
    asym_s, _ = dpo_loss(pr, pc, rr, rc, beta)
    check("без сглаживания перестановка меняет лосс",
          abs(asym_f.item() - asym_s.item()) > 1e-3,
          f"{asym_f.item():.4f} против {asym_s.item():.4f}")


def test_sequence_logprob() -> None:
    print("\n8. Логвероятность последовательности считается только по маске")
    policy, pairs = _dpo_setup()
    ids, mask, _, _ = build_batch(pairs[:4], pad_id=0)
    full = sequence_logprob(policy, ids, torch.ones_like(mask))
    answer_only = sequence_logprob(policy, ids, mask)
    check("маска уменьшает сумму по модулю",
          bool((answer_only > full).all()), "промпт исключён из суммы")
    check("все значения отрицательны", bool((answer_only < 0).all()))
    zero = sequence_logprob(policy, ids, torch.zeros_like(mask))
    check("пустая маска даёт ноль", bool((zero.abs() < 1e-6).all()))


def test_dpo_training() -> None:
    print("\n9. DPO двигает политику к предпочтённым ответам (M14 §8)")
    policy, pairs = _dpo_setup()
    reference = freeze_reference(policy)
    state = train_dpo(policy, pairs, DPOConfig(steps=120, lr=3e-4, beta=0.1,
                                               log_every=20), pad_id=0,
                      reference=reference)
    first, last = state.history[0], state.history[-1]
    check("история содержит несколько точек", len(state.history) >= 5,
          f"{len(state.history)} записей")
    check("margin вырос", last["margin"] > first["margin"],
          f"{first['margin']:+.4f} → {last['margin']:+.4f}")
    check("margin рос монотонно по большинству отсечек",
          sum(1 for a, b in zip(state.history, state.history[1:])
              if b["margin"] > a["margin"]) >= len(state.history) - 2)
    check("учебные легко разделимые пары достигли accuracy=1",
          last["accuracy"] == 1.0,
          f"{last['accuracy']:.2f}; это не доказывает отсутствие артефакта")
    check("лосс упал ниже ln 2", last["loss"] < math.log(2),
          f"{last['loss']:.4f} < {math.log(2):.4f}")
    check("референсная модель осталась замороженной",
          all(not p.requires_grad for p in reference.parameters()))


def main() -> int:
    print("Проверки посттрейна MiniLLM")
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
    print(f"\nИтог: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    for name in FAILED:
        print("  провал:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
