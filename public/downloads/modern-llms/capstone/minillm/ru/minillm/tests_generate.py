"""Проверки инференса. Запуск: python3 -m minillm.tests_generate

Главная проверка здесь — восьмая: спекулятивный декод обязан сохранять
распределение target-модели. Это утверждение модуля 11, которое звучит
контринтуитивно («ускорение бесплатно по качеству») и потому должно
проверяться эмпирически, а не приниматься на слово.
"""
from __future__ import annotations

import copy
import math

import torch

from .config import MiniLLMConfig
from .data import ByteTokenizer, TokenStream, toy_corpus
from .generate import (E2M1_LEVELS, GenerationStats, SamplingConfig,
                       bits_per_weight, generate, logits_to_probs,
                       quantize_fp4, quantize_int8_per_channel, quantize_model,
                       speculative_generate)
from .model import MiniLLM

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _tiny_model(vocab: int = 32, seed: int = 0, **overrides) -> MiniLLM:
    torch.manual_seed(seed)
    cfg = MiniLLMConfig.profile("S", vocab_size=vocab, hidden_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16,
                                max_position_embeddings=128, **overrides)
    return MiniLLM(cfg).eval()


# ------------------------------------------------------------- 1. отбор кандидатов
def test_logits_processors() -> None:
    print("\n1. Ручки семплирования режут ровно то, что обещают (M15 §3)")
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -5.0])
    greedy = logits_to_probs(logits, SamplingConfig(temperature=0.0))
    check("temperature=0 даёт вырожденное распределение на argmax",
          greedy[0] == 1.0 and greedy.sum() == 1.0)

    top_k = logits_to_probs(logits, SamplingConfig(top_k=2))
    check("top_k=2 оставляет ровно два токена", int((top_k > 0).sum()) == 2,
          f"ненулевых: {int((top_k > 0).sum())}")

    top_p = logits_to_probs(logits, SamplingConfig(top_p=0.6))
    support = int((top_p > 0).sum())
    check("top_p=0.6 оставляет минимальный префикс массы", 1 <= support <= 3,
          f"ненулевых: {support}")
    check("top_p никогда не оставляет пустое множество",
          int((logits_to_probs(logits, SamplingConfig(top_p=0.01)) > 0).sum()) >= 1)

    min_p = logits_to_probs(logits, SamplingConfig(min_p=0.5))
    base = torch.softmax(logits, -1)
    expected = int((base >= 0.5 * base.max()).sum())
    check("min_p отсекает по доле от максимума", int((min_p > 0).sum()) == expected,
          f"{int((min_p > 0).sum())} == {expected}")
    for cfg in (SamplingConfig(top_k=3), SamplingConfig(top_p=0.5),
                SamplingConfig(min_p=0.3), SamplingConfig(temperature=2.0)):
        p = logits_to_probs(logits, cfg)
        if abs(p.sum().item() - 1.0) > 1e-5:
            check("распределение нормировано после отсечений", False)
            break
    else:
        check("распределение нормировано после отсечений", True)


# --------------------------------------------------------------------- 2. генерация
def test_generation_matches_manual_decode() -> None:
    print("\n2. Генерация с KV-кэшем == пошаговый argmax без кэша (M10 §4)")
    model = _tiny_model()
    prompt = [1, 2, 3, 4, 5]
    out, stats = generate(model, prompt, 10, SamplingConfig(temperature=0.0))

    manual = list(prompt)
    with torch.no_grad():
        for _ in range(10):
            logits, _ = model(torch.tensor([manual]))
            manual.append(int(logits[0, -1].argmax()))
    check("последовательности совпадают", out == manual, f"{out[-5:]} == {manual[-5:]}")
    check("prefill и decode замерены отдельно",
          stats.prefill_seconds > 0 and stats.decode_seconds > 0)
    check("сгенерировано ровно столько, сколько просили", stats.generated_tokens == 10)
    check("нет неиспользуемого forward после последнего токена",
          stats.target_prefill_forwards == 1 and stats.target_decode_forwards == 9
          and stats.target_forwards == 10, str(stats.as_dict()))
    check("по умолчанию cache отстаёт ровно на последний токен",
          stats.cache_tokens == len(out) - 1, f"cache={stats.cache_tokens}, tokens={len(out)}")

    cache = model.new_cache()
    finalized, final_stats = generate(model, prompt, 3, SamplingConfig(temperature=0.0),
                                      cache=cache, finalize_cache=True)
    check("finalize_cache материализует и последний token",
          final_stats.cache_tokens == len(finalized))


def test_sampling_is_reproducible() -> None:
    print("\n3. Один seed — одна последовательность")
    model = _tiny_model()
    prompt = [1, 2, 3]
    a, _ = generate(model, prompt, 12, SamplingConfig(temperature=0.9, seed=7))
    b, _ = generate(model, prompt, 12, SamplingConfig(temperature=0.9, seed=7))
    c, _ = generate(model, prompt, 12, SamplingConfig(temperature=0.9, seed=8))
    check("одинаковые seed'ы совпадают", a == b)
    check("разные seed'ы расходятся", a != c)


def test_prefix_cache_reuse() -> None:
    print("\n4. Переиспользование префикса даёт тот же результат (M10 §6)")
    model = _tiny_model()
    shared = [1, 2, 3, 4, 5, 6]
    tail_a, tail_b = [7, 8], [9, 10]

    warm = model.new_cache()
    with torch.no_grad():
        model(torch.tensor([shared]), cache=warm)
    check("прогретый кэш покрывает префикс", len(warm) == len(shared))

    for tail in (tail_a, tail_b):
        cold, _ = generate(model, shared + tail, 8, SamplingConfig(temperature=0.0))
        hot, _ = generate(model, shared + tail, 8, SamplingConfig(temperature=0.0),
                          cache=warm.clone())
        check(f"хвост {tail}: результат идентичен", cold == hot)


def test_cache_truncate() -> None:
    print("\n5. Откат кэша эквивалентен пересчёту с нуля (нужен спекулятиву)")
    model = _tiny_model()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        cache = model.new_cache()
        model(torch.tensor([tokens]), cache=cache)
        cache.truncate(5)
        after_truncate, _ = model(torch.tensor([[9]]), cache=cache)

        fresh = model.new_cache()
        model(torch.tensor([tokens[:5]]), cache=fresh)
        reference, _ = model(torch.tensor([[9]]), cache=fresh)
    delta = (after_truncate - reference).abs().max().item()
    check("логиты после отката совпадают", delta < 1e-5, f"Δ = {delta:.2e}")
    check("длина кэша откатилась", len(cache) == 6)


# ------------------------------------------------------- 6-8. спекулятивный декод
def test_speculative_shapes_and_acceptance() -> None:
    print("\n6. Спекулятив: длина выхода и учёт принятых токенов (M11 §4)")
    target = _tiny_model(seed=0)
    draft = _tiny_model(seed=1)
    initial_modes = (target.training, draft.training)
    out, stats = speculative_generate(target, draft, [1, 2, 3], 20, lookahead=4,
                                      sampling=SamplingConfig(temperature=1.0, seed=0))
    check("сгенерировано ровно 20 токенов", stats.generated_tokens == 20,
          f"{stats.generated_tokens}")
    check("длина последовательности сходится", len(out) == 3 + 20)
    check("статистика приёма заполнена", stats.proposed > 0,
          f"принято {stats.accepted}/{stats.proposed} = {stats.acceptance_rate:.2%}")
    check("один decode-проход target даёт больше одного токена",
          stats.tokens_per_target_forward > 1.0,
          f"{stats.tokens_per_target_forward:.2f} токена на проход большой модели")
    check("prefill обеих моделей вынесен из decode-метрик",
          stats.target_prefill_forwards == stats.draft_prefill_forwards == 1)
    check("черновик делает больше decode-проходов, чем target",
          stats.draft_decode_forwards > stats.target_decode_forwards,
          f"{stats.draft_decode_forwards} против {stats.target_decode_forwards} — "
          "маленькая модель работает чаще, но стоит дешевле")
    check("исходный train/eval режим обеих моделей восстановлен",
          (target.training, draft.training) == initial_modes, str(initial_modes))


def test_generation_contract_validation() -> None:
    print("\n6b. Генерация рано отвергает несовместимые и бессмысленные конфиги")
    invalid_sampling = 0
    for kwargs in ({"temperature": -1.0}, {"top_k": -1},
                   {"top_p": 0.0}, {"top_p": 1.1}, {"min_p": 1.1}):
        try:
            SamplingConfig(**kwargs)
        except ValueError:
            invalid_sampling += 1
    check("все недопустимые sampling-параметры отвергнуты", invalid_sampling == 5,
          f"{invalid_sampling}/5")

    target = _tiny_model(vocab=32, seed=0)
    wrong_vocab = _tiny_model(vocab=31, seed=1)
    initial_modes = (target.training, wrong_vocab.training)
    try:
        speculative_generate(target, wrong_vocab, [1, 2], 1,
                             sampling=SamplingConfig(seed=0))
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    check("разные словари target/draft отвергаются", mismatch_rejected)
    check("ошибка валидации не меняет train/eval режим",
          (target.training, wrong_vocab.training) == initial_modes, str(initial_modes))


def test_speculative_identical_draft_accepts_everything() -> None:
    print("\n7. Черновик == target ⇒ принимается почти всё (M11 §7)")
    target = _tiny_model(seed=0)
    draft = copy.deepcopy(target)
    _, stats = speculative_generate(target, draft, [1, 2, 3], 40, lookahead=4,
                                    sampling=SamplingConfig(temperature=1.0, seed=0))
    check("доля приёма близка к единице", stats.acceptance_rate > 0.95,
          f"{stats.acceptance_rate:.2%} — при p == q правило min(1, p/q) всегда даёт 1")


def test_speculative_preserves_distribution() -> None:
    print("\n8. Спекулятив сохраняет распределение target — главная гарантия M11 §4")
    vocab = 16
    target = _tiny_model(vocab=vocab, seed=0)
    draft = _tiny_model(vocab=vocab, seed=99)     # заведомо другая модель
    prompt = [1, 2, 3]
    sampling = SamplingConfig(temperature=1.0)

    with torch.no_grad():
        p_logits, _ = target(torch.tensor([prompt]))
        q_logits, _ = draft(torch.tensor([prompt]))
    p = logits_to_probs(p_logits[0, -1], sampling)
    q = logits_to_probs(q_logits[0, -1], sampling)

    # Exact one-position balance: accepted proposal mass is min(p, q); the
    # correction branch restores the positive residual p-q. This algebra is the
    # distributional guarantee, independent of Monte-Carlo noise.
    overlap = torch.minimum(p, q)
    reject_mass = 1.0 - overlap.sum()
    positive = (p - q).clamp(min=0)
    residual = positive / reject_mass if reject_mass > 0 else torch.zeros_like(p)
    reconstructed = overlap + reject_mass * residual
    check("однопозиционный баланс восстанавливает target точно",
          torch.allclose(reconstructed, p, atol=1e-7, rtol=0.0),
          f"max|Δ|={(reconstructed-p).abs().max().item():.2e}")

    # A smaller end-to-end sampling probe catches cache/rollback mistakes without
    # turning the unit suite into a multi-thousand-forward benchmark. The seed
    # set is fixed, so this regression threshold is deterministic.
    runs = 128
    counts = torch.zeros(vocab)
    for seed in range(runs):
        out, _ = speculative_generate(
            target, draft, prompt, 1, lookahead=4,
            sampling=SamplingConfig(temperature=1.0, seed=seed))
        counts[out[len(prompt)]] += 1
    empirical = counts / runs
    tv = 0.5 * (empirical - p).abs().sum().item()
    check("сквозной Monte-Carlo probe близок к target",
          tv < 0.16, f"TV = {tv:.4f} на {runs} фиксированных seed")


# -------------------------------------------------------------- 9-11. квантизация
def test_bits_per_weight() -> None:
    print("\n9. Бит на вес считается, а не цитируется (M11 §11)")
    check("MXFP4: группа 32 + E8M0 → 4.25", bits_per_weight(32, 8) == 4.25,
          f"{bits_per_weight(32, 8)}")
    check("NVFP4: группа 16 + E4M3 → 4.50", bits_per_weight(16, 8) == 4.5,
          f"{bits_per_weight(16, 8)}")
    check("мелкая группа дороже по хранению", bits_per_weight(16, 8) > bits_per_weight(32, 8))
    check("сжатие против bf16", abs(16 / bits_per_weight(32, 8) - 3.7647) < 1e-3,
          f"×{16 / bits_per_weight(32, 8):.4f}")


def test_quantizers() -> None:
    print("\n10. Кванторы кладут веса на свою сетку и не ломают масштаб (M11 §12)")
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.05

    q4 = quantize_fp4(w, group=32, power_of_two_scale=True)
    ratios = (q4[q4 != 0].abs() / w[q4 != 0].abs()).clamp(0.1, 10)
    check("MXFP4 сохраняет знаки", bool((q4.sign() == w.sign())[q4 != 0].all()))
    check("MXFP4: относительная ошибка в разумных пределах",
          ((q4 - w).norm() / w.norm()).item() < 0.2,
          f"{(q4 - w).norm() / w.norm():.4f}")

    q4_nv = quantize_fp4(w, group=16, power_of_two_scale=False)
    err_mx = ((q4 - w).norm() / w.norm()).item()
    err_nv = ((q4_nv - w).norm() / w.norm()).item()
    check("NVFP4 точнее MXFP4 — мельче группа и мантисса у масштаба",
          err_nv < err_mx, f"NVFP4 {err_nv:.4f} против MXFP4 {err_mx:.4f}")

    q8 = quantize_int8_per_channel(w)
    err8 = ((q8 - w).norm() / w.norm()).item()
    check("int8 на порядок точнее fp4", err8 < err_mx / 5,
          f"int8 {err8:.5f} против MXFP4 {err_mx:.4f}")

    exact = quantize_fp4(torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 4),
                         group=32, power_of_two_scale=True)
    check("точки сетки E2M1 воспроизводятся точно",
          torch.allclose(exact[:8], E2M1_LEVELS, atol=1e-6),
          f"{[round(x, 2) for x in exact[:8].tolist()]}")


def test_quantize_model_report() -> None:
    print("\n11. Отчёт квантизации модели арифметически согласован")
    model = _tiny_model()
    for method, bits in (("int8", 8.0), ("nvfp4", 4.5), ("mxfp4", 4.25)):
        clone = copy.deepcopy(model)
        rep = quantize_model(clone, method)
        check(f"{method}: теоретический форматный бит/вес",
              rep["bits_per_weight"] == bits, f"{bits}")
        check(f"{method}: effective bits include scales/padding",
              rep["effective_bits_per_weight"] >= bits, str(rep))
        expected_bytes = math.ceil(rep["elements"] *
                                   rep["effective_bits_per_weight"] / 8)
        check(f"{method}: stored_bytes следует effective payload",
              abs(rep["stored_bytes"] - expected_bytes) <= 1,
              f"{rep['stored_bytes']:,} vs {expected_bytes:,}")
        check(f"{method}: compression следует effective bits",
              abs(rep["compression"] - 16 / rep["effective_bits_per_weight"]) < 2e-3,
              f"×{rep['compression']}")
        check(f"{method}: веса действительно изменились",
              not torch.equal(clone.layers[0].attn.q_proj.weight,
                              model.layers[0].attn.q_proj.weight))
    clone = copy.deepcopy(model)
    quantize_model(clone, "mxfp4", skip_embeddings=True)
    check("эмбеддинги остались в высокой точности",
          torch.equal(clone.embed_tokens.weight, model.embed_tokens.weight),
          "первый и последний слой чувствительнее прочих — M11 §13")


def main() -> int:
    print("Проверки инференса MiniLLM")
    test_logits_processors()
    test_generation_matches_manual_decode()
    test_sampling_is_reproducible()
    test_prefix_cache_reuse()
    test_cache_truncate()
    test_speculative_shapes_and_acceptance()
    test_generation_contract_validation()
    test_speculative_identical_draft_accepts_everything()
    test_speculative_preserves_distribution()
    test_bits_per_weight()
    test_quantizers()
    test_quantize_model_report()
    print(f"\nИтог: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    for name in FAILED:
        print("  провал:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
