"""Проверки данных: токенизатор, корпус, шарды. Запуск: python3 -m minillm.tests_data"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .corpus import (FilterConfig, dedup_exact, jaccard_estimate, minhash_dedup,
                     minhash_signature, normalize, pipeline, quality_reasons)
from .data import ByteTokenizer, ShardStream, TokenStream, toy_corpus, write_shards
from .config import MiniLLMConfig
from .model import MiniLLM
from .probes import RepeatTask, chance_level, repeat_accuracy
from .tokenizer import BBPETokenizer

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _varied_corpus(docs: int = 6000, seed: int = 0) -> str:
    """Корпус с достаточным разнообразием, чтобы merge'ы не кончились досрочно."""
    rng = random.Random(seed)
    ru = "абвгдеёжзийклмнопрстуфхцчшщыьэюя"
    en = "abcdefghijklmnopqrstuvwxyz"
    def word(alphabet: str) -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 10)))
    ru_words = [word(ru) for _ in range(1500)]
    en_words = [word(en) for _ in range(1500)]
    lines = []
    for i in range(docs):
        pool = ru_words if i % 2 else en_words
        lines.append(" ".join(rng.choice(pool) for _ in range(rng.randint(5, 15))) + ".\n")
    return "".join(lines)


# ----------------------------------------------------------------- 1. токенизатор
def test_bbpe_roundtrip() -> None:
    print("\n1. BBPE: обучение, round-trip и размер словаря (M2 §4)")
    text = _varied_corpus()
    tok = BBPETokenizer.train(text, vocab_size=2048, specials=["<|bos|>", "<|eos|>"])
    check("словарь ровно запрошенного размера", tok.vocab_size == 2048,
          f"{tok.vocab_size} = 256 базовых + {len(tok.merges)} merge + 2 специальных")

    samples = ["привет мир", "hello world", "смешанный mixed текст 123",
               "эмодзи 🚀 и знаки !?;", "", "\n\n  "]
    bad = [s for s in samples if tok.decode(tok.encode(s)) != s]
    check("round-trip без потерь на всех образцах", not bad, f"сломалось: {bad}")

    arbitrary = "".join(chr(i) for i in range(0x400, 0x460))
    check("byte-level не знает OOV — кодируется любой Unicode",
          tok.decode(tok.encode(arbitrary)) == arbitrary)
    check("merge'ы действительно сокращают длину",
          len(tok.encode(text[:2000])) < len(text[:2000].encode("utf-8")),
          f"{len(tok.encode(text[:2000]))} токенов против "
          f"{len(text[:2000].encode('utf-8'))} байт")


def test_bbpe_specials() -> None:
    print("\n2. Специальные токены вставляет только рантайм (M2 §8)")
    tok = BBPETokenizer.train(_varied_corpus(2000), vocab_size=1024,
                              specials=["<|bos|>", "<|eos|>"])
    try:
        tok.encode("текст с <|eos|> внутри")
        raised = False
    except ValueError:
        raised = True
    check("специальный токен в пользовательском тексте отвергается", raised,
          "иначе строка в письме пользователя оборвала бы диалог")

    ids = tok.encode("привет<|eos|>пока", allowed_special={"<|eos|>"})
    check("с allowed_special токен кодируется одним идентификатором",
          tok.specials["<|eos|>"] in ids)
    check("декодирование восстанавливает специальный токен",
          tok.decode(ids) == "привет<|eos|>пока", tok.decode(ids))

    overlap = BBPETokenizer.train("ordinary text " * 500, vocab_size=512,
                                  specials=["<|x|>", "<|x|><|y|>"])
    both = {"<|x|>", "<|x|><|y|>"}
    ids_a = overlap.encode("A<|x|><|y|>B", allowed_special=both)
    ids_b = overlap.encode("A<|x|><|y|>B", allowed_special=set(reversed(sorted(both))))
    check("перекрывающиеся control strings выбираются longest-first",
          ids_a == ids_b and overlap.specials["<|x|><|y|>"] in ids_a)
    try:
        overlap.encode("x", allowed_special={"<|unknown|>"})
        bad_allowed = False
    except ValueError:
        bad_allowed = True
    check("allowed_special обязан быть подмножеством зарегистрированных токенов", bad_allowed)


def test_bbpe_fertility() -> None:
    print("\n3. Fertility зависит от того, что было в корпусе (M2 §10)")
    rng = random.Random(1)
    ru = "".join(" ".join("".join(rng.choice("абвгдежзиклмнопрстуфхцчшщыэюя")
                                  for _ in range(rng.randint(3, 9)))
                          for _ in range(10)) + ".\n" for _ in range(4000))
    ru_only = BBPETokenizer.train(ru, vocab_size=2048)
    probe = " ".join(ru.split()[:200])
    f_ru = ru_only.fertility(probe)
    plain = ByteTokenizer()
    baseline = len(plain.encode(probe)) / len(probe)
    check("обученный словарь резко лучше byte-level на своём языке",
          f_ru["tokens_per_char"] < baseline / 2,
          f"{f_ru['tokens_per_char']:.3f} против {baseline:.3f} у чистых байтов")
    check("bytes_per_token больше единицы", f_ru["bytes_per_token"] > 1.0,
          f"{f_ru['bytes_per_token']:.2f}")


def test_bbpe_save_load() -> None:
    print("\n4. Словарь переживает сохранение и загрузку")
    tok = BBPETokenizer.train(_varied_corpus(2000), vocab_size=1024,
                              specials=["<|eos|>"])
    text = "проверка сохранения словаря"
    with tempfile.TemporaryDirectory() as tmp:
        path = tok.save(Path(tmp) / "tokenizer.json")
        restored = BBPETokenizer.load(path)
    check("merge'ы совпадают", restored.merges == tok.merges)
    check("кодирование совпадает", restored.encode(text) == tok.encode(text))
    check("размер словаря совпадает", restored.vocab_size == tok.vocab_size)


# --------------------------------------------------------------------- 5-7. корпус
def test_normalize_and_filters() -> None:
    print("\n5. Нормализация и эвристики отбраковки (M9 §5)")
    combining = "и\u0306"                      # «й» через комбинирующую краткую
    check("NFC схлопывает комбинирующие знаки", len(normalize(combining)) == 1,
          f"{len(combining)} → {len(normalize(combining))} символов")
    check("управляющие символы удаляются", "\x07" not in normalize("текст\x07текст"))
    check("пробелы схлопываются", normalize("a    b") == "a b")

    cfg = FilterConfig()
    check("короткий документ отбраковывается", "too_short" in quality_reasons("ок", cfg))
    check("мусорный символьный документ отбраковывается",
          "symbol_ratio" in quality_reasons("!@#$%^&*()" * 20, cfg))
    spam = "\n".join(["купите слоника прямо сейчас недорого"] * 20)
    check("повтор строк ловится", "line_repeat" in quality_reasons(spam, cfg))
    good = ("Языковые модели предсказывают следующий токен по контексту, "
            "и качество предсказания зависит от данных обучения.")
    check("осмысленный документ проходит", quality_reasons(good, cfg) == [],
          f"причины: {quality_reasons(good, cfg)}")


def test_dedup() -> None:
    print("\n6. Дедупликация: точная и near-dup через MinHash (M9 §5)")
    base = ("Языковые модели предсказывают следующий токен по контексту "
            "предложения и обучаются на больших корпусах текста")
    docs = [base, base, base + " данных", "Совершенно другой документ про погоду "
            "в городе и температуру воздуха завтра утром"]
    exact = dedup_exact(docs)
    check("точный дедуп снимает буквальные копии", len(exact) == 3, f"{len(docs)} → {len(exact)}")

    sig_a = minhash_signature(base)
    sig_b = minhash_signature(base + " данных")
    sig_c = minhash_signature(docs[3])
    check("похожие документы дают близкие сигнатуры",
          jaccard_estimate(sig_a, sig_b) > 0.7,
          f"J≈{jaccard_estimate(sig_a, sig_b):.2f}")
    check("непохожие документы — далёкие сигнатуры",
          jaccard_estimate(sig_a, sig_c) < 0.2,
          f"J≈{jaccard_estimate(sig_a, sig_c):.2f}")

    near = minhash_dedup(exact, threshold=0.7)
    check("near-dup дедуп снимает почти-копию", len(near) == 2, f"{len(exact)} → {len(near)}")
    check("уникальный документ выжил", docs[3] in near)


def test_pipeline_funnel() -> None:
    print("\n7. Конвейер даёт воронку с числами на каждой стадии (M9 §5)")
    rng = random.Random(0)
    good = [f"Документ номер {i} про языковые модели и обучение на текстовых "
            f"корпусах достаточной длины для прохождения фильтров." for i in range(60)]
    docs = good + good[:20] + ["!!!" * 30] * 10 + ["ок"] * 10
    rng.shuffle(docs)
    survivors, report = pipeline(docs)
    names = [n for n, _ in report.stages]
    check("все стадии присутствуют",
          names == ["исходные", "нормализация", "эвристики", "точный дедуп",
                    "near-dup дедуп"], str(names))
    check("воронка монотонно невозрастающая",
          all(report.stages[i][1] >= report.stages[i + 1][1]
              for i in range(len(report.stages) - 1)))
    check("мусор отброшен, осмысленное осталось", len(survivors) == 60,
          f"{len(docs)} → {len(survivors)}, выживаемость {report.survival:.1%}")
    check("причины отбраковки записаны", set(report.reasons) >= {"too_short"},
          str(report.reasons))


# ---------------------------------------------------------------- 8-9. потоки
def test_shard_stream_matches_memory() -> None:
    print("\n8. Шардированный поток читается так же, как поток в памяти")
    tok = ByteTokenizer()
    memory = TokenStream.from_text(toy_corpus(80), tok, seq_len=64)
    first_two = memory.batch(torch.tensor([0, 1]))
    check("окно содержит seq_len+1 токен и seq_len переходов",
          first_two.shape[1] == memory.seq_len + 1, str(tuple(first_two.shape)))
    check("граница соседних окон обучается ровно один раз",
          first_two[0, -1].item() == first_two[1, 0].item())
    with tempfile.TemporaryDirectory() as tmp:
        index = write_shards(memory.tokens.tolist(), tmp, shard_tokens=2500)
        with ShardStream(tmp, seq_len=64, verify=True) as shards, \
                ShardStream(tmp, seq_len=64) as second_view:
            check("шардов больше одного", len(index["shards"]) > 1,
                  f"{len(index['shards'])} шардов по {index['total_tokens']:,} токенов")
            check("число окон совпадает", len(shards) == len(memory))
            idx = torch.tensor([0, 1, 17, 42])
            check("батчи совпадают побитово",
                  torch.equal(shards.batch(idx), memory.batch(idx)))
            # Окно, пересекающее границу шарда, — главный источник ошибок здесь.
            boundary = 2500 // 64
            check("окно через границу шарда читается верно",
                  torch.equal(shards.batch(torch.tensor([boundary])),
                              memory.batch(torch.tensor([boundary]))))
            check("контрольная сумма стабильна",
                  shards.checksum() == second_view.checksum())


def test_shard_dtype_range_is_checked_before_cast() -> None:
    print("\n8b. Шардер проверяет диапазон ID до сужения dtype")
    with tempfile.TemporaryDirectory() as tmp:
        index = write_shards(np.array([0, 65_535], dtype=np.int64), tmp,
                             shard_tokens=10, dtype="uint16")
        with ShardStream(tmp, seq_len=1) as stream:
            check("граница uint16 65 535 допустима",
                  stream.maps[0].tolist() == [0, 65_535], str(index["dtype"]))

    rejected = []
    for values in (np.array([70_000], dtype=np.int64),
                   np.array([-1, 2], dtype=np.int64)):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                write_shards(values, tmp, dtype="uint16")
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
    check("переполнение и отрицательные ID отвергаются до cast", all(rejected),
          f"результаты {rejected}")


def test_loader_cursor() -> None:
    print("\n9. Курсор данных: возобновление продолжает поток, а не начинает заново")
    tok = ByteTokenizer()
    stream = TokenStream.from_text(toy_corpus(80), tok, seq_len=64)

    straight = stream.loader(4, seed=5)
    reference = [next(straight) for _ in range(12)]
    resumed = stream.loader(4, seed=5, start_batch=8)
    tail = [next(resumed) for _ in range(4)]
    check("продолжение совпадает с непрерывным чтением",
          all(torch.equal(a, b) for a, b in zip(reference[8:], tail)))

    per_epoch = len(stream) // 4
    check("переход через границу эпохи детерминирован",
          torch.equal(next(stream.loader(4, seed=5, start_batch=per_epoch)),
                      [next(straight) for _ in range(per_epoch - 12 + 1)][-1]))
    unshuffled = stream.loader(4, seed=5, shuffle=False)
    check("без перемешивания порядок естественный",
          torch.equal(next(unshuffled), stream.batch(torch.arange(4))))


def test_repeat_probe() -> None:
    print("\n10. Проба с дальней связью: раскладка и базовая линия (M4 §6)")
    task = RepeatTask(vocab_size=64, train_range=(12, 48), seed=0)
    ids, labels = task.sample(8, half_len=20)
    check("вход — префикс, повторённый дважды",
          bool(torch.equal(ids[:, :20], ids[:, 20:])), f"форма {tuple(ids.shape)}")
    check("первая половина не учится", bool((labels[:, :20] == -100).all()))
    check("первый токен повтора не учится — его нечем предсказать",
          bool((labels[:, 20] == -100).all()))
    check("остальная вторая половина учится целиком",
          bool((labels[:, 21:] == ids[:, 21:]).all()),
          f"{int((labels != -100).sum())} обучаемых позиций из {labels.numel()}")

    # Плотность сигнала — причина, по которой первая версия пробы («иголка в
    # стоге» с одной обучаемой позицией) не обучалась вовсе.
    density = (labels != -100).float().mean().item()
    check("плотность сигнала близка к половине", 0.4 < density < 0.5,
          f"{density:.1%} позиций против ~2.5% у варианта с одной иголкой")

    cfg = MiniLLMConfig.profile("S", vocab_size=64, hidden_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16,
                                max_position_embeddings=256)
    torch.manual_seed(0)
    untrained = MiniLLM(cfg)
    accuracy = repeat_accuracy(untrained, task, half_len=24, batches=4)
    check("необученная модель держится уровня случая",
          accuracy < 5 * chance_level(task),
          f"{accuracy:.4f} против {chance_level(task):.4f} — есть куда расти")
    check("loader совместим с train()",
          isinstance(next(task.loader(4, seed=0)), tuple))


def main() -> int:
    print("Проверки данных MiniLLM")
    test_bbpe_roundtrip()
    test_bbpe_specials()
    test_bbpe_fertility()
    test_bbpe_save_load()
    test_normalize_and_filters()
    test_dedup()
    test_pipeline_funnel()
    test_shard_stream_matches_memory()
    test_shard_dtype_range_is_checked_before_cast()
    test_loader_cursor()
    test_repeat_probe()
    print(f"\nИтог: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    for name in FAILED:
        print("  провал:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
