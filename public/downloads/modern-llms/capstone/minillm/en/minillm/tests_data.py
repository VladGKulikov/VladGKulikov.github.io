"""Data checks: tokenizer, corpus, and shards. Run: python3 -m minillm.tests_data"""
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
    """A corpus varied enough that merge candidates do not run out early."""
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


# ------------------------------------------------------------------- 1. tokenizer
def test_bbpe_roundtrip() -> None:
    print("\n1. BBPE: training, round trip, and vocabulary size (M2 §4)")
    text = _varied_corpus()
    tok = BBPETokenizer.train(text, vocab_size=2048, specials=["<|bos|>", "<|eos|>"])
    check("vocabulary has exactly the requested size", tok.vocab_size == 2048,
          f"{tok.vocab_size} = 256 base + {len(tok.merges)} merges + 2 special")

    samples = ["привет мир", "hello world", "смешанный mixed текст 123",
               "эмодзи 🚀 и знаки !?;", "", "\n\n  "]
    bad = [s for s in samples if tok.decode(tok.encode(s)) != s]
    check("lossless round trip on every sample", not bad, f"failed: {bad}")

    arbitrary = "".join(chr(i) for i in range(0x400, 0x460))
    check("byte-level has no OOV—any Unicode can be encoded",
          tok.decode(tok.encode(arbitrary)) == arbitrary)
    check("merges genuinely shorten the sequence",
          len(tok.encode(text[:2000])) < len(text[:2000].encode("utf-8")),
          f"{len(tok.encode(text[:2000]))} tokens vs "
          f"{len(text[:2000].encode('utf-8'))} bytes")


def test_bbpe_specials() -> None:
    print("\n2. Only the runtime inserts special tokens (M2 §8)")
    tok = BBPETokenizer.train(_varied_corpus(2000), vocab_size=1024,
                              specials=["<|bos|>", "<|eos|>"])
    try:
        tok.encode("текст с <|eos|> внутри")
        raised = False
    except ValueError:
        raised = True
    check("special token in user text is rejected", raised,
          "otherwise a literal in a user message could terminate the dialogue")

    ids = tok.encode("привет<|eos|>пока", allowed_special={"<|eos|>"})
    check("with allowed_special, the token encodes as one ID",
          tok.specials["<|eos|>"] in ids)
    check("decoding restores the special token",
          tok.decode(ids) == "привет<|eos|>пока", tok.decode(ids))

    overlap = BBPETokenizer.train("ordinary text " * 500, vocab_size=512,
                                  specials=["<|x|>", "<|x|><|y|>"])
    both = {"<|x|>", "<|x|><|y|>"}
    ids_a = overlap.encode("A<|x|><|y|>B", allowed_special=both)
    ids_b = overlap.encode("A<|x|><|y|>B", allowed_special=set(reversed(sorted(both))))
    check("overlapping control strings use longest-first matching",
          ids_a == ids_b and overlap.specials["<|x|><|y|>"] in ids_a)
    try:
        overlap.encode("x", allowed_special={"<|unknown|>"})
        bad_allowed = False
    except ValueError:
        bad_allowed = True
    check("allowed_special must be a subset of registered tokens", bad_allowed)


def test_bbpe_fertility() -> None:
    print("\n3. Fertility depends on the training corpus (M2 §10)")
    rng = random.Random(1)
    ru = "".join(" ".join("".join(rng.choice("абвгдежзиклмнопрстуфхцчшщыэюя")
                                  for _ in range(rng.randint(3, 9)))
                          for _ in range(10)) + ".\n" for _ in range(4000))
    ru_only = BBPETokenizer.train(ru, vocab_size=2048)
    probe = " ".join(ru.split()[:200])
    f_ru = ru_only.fertility(probe)
    plain = ByteTokenizer()
    baseline = len(plain.encode(probe)) / len(probe)
    check("a learned vocabulary sharply beats byte-level on its own language",
          f_ru["tokens_per_char"] < baseline / 2,
          f"{f_ru['tokens_per_char']:.3f} vs {baseline:.3f} for raw bytes")
    check("bytes_per_token is above one", f_ru["bytes_per_token"] > 1.0,
          f"{f_ru['bytes_per_token']:.2f}")


def test_bbpe_save_load() -> None:
    print("\n4. Vocabulary survives save and load")
    tok = BBPETokenizer.train(_varied_corpus(2000), vocab_size=1024,
                              specials=["<|eos|>"])
    text = "проверка сохранения словаря"
    with tempfile.TemporaryDirectory() as tmp:
        path = tok.save(Path(tmp) / "tokenizer.json")
        restored = BBPETokenizer.load(path)
    check("merges match", restored.merges == tok.merges)
    check("encoding matches", restored.encode(text) == tok.encode(text))
    check("vocabulary size matches", restored.vocab_size == tok.vocab_size)


# --------------------------------------------------------------------- 5-7. corpus
def test_normalize_and_filters() -> None:
    print("\n5. Normalization and rejection heuristics (M9 §5)")
    combining = "и\u0306"                      # SHORT I as I + combining breve
    check("NFC composes combining marks", len(normalize(combining)) == 1,
          f"{len(combining)} → {len(normalize(combining))} characters")
    check("control characters are removed", "\x07" not in normalize("текст\x07текст"))
    check("spaces are collapsed", normalize("a    b") == "a b")

    cfg = FilterConfig()
    check("short document is rejected", "too_short" in quality_reasons("ок", cfg))
    check("symbol-heavy garbage is rejected",
          "symbol_ratio" in quality_reasons("!@#$%^&*()" * 20, cfg))
    spam = "\n".join(["купите слоника прямо сейчас недорого"] * 20)
    check("repeated lines are detected", "line_repeat" in quality_reasons(spam, cfg))
    good = ("Языковые модели предсказывают следующий токен по контексту, "
            "и качество предсказания зависит от данных обучения.")
    check("meaningful document passes", quality_reasons(good, cfg) == [],
          f"reasons: {quality_reasons(good, cfg)}")


def test_dedup() -> None:
    print("\n6. Deduplication: exact and MinHash near-dup (M9 §5)")
    base = ("Языковые модели предсказывают следующий токен по контексту "
            "предложения и обучаются на больших корпусах текста")
    docs = [base, base, base + " данных", "Совершенно другой документ про погоду "
            "в городе и температуру воздуха завтра утром"]
    exact = dedup_exact(docs)
    check("exact dedup removes literal copies", len(exact) == 3, f"{len(docs)} → {len(exact)}")

    sig_a = minhash_signature(base)
    sig_b = minhash_signature(base + " данных")
    sig_c = minhash_signature(docs[3])
    check("similar documents yield similar signatures",
          jaccard_estimate(sig_a, sig_b) > 0.7,
          f"J≈{jaccard_estimate(sig_a, sig_b):.2f}")
    check("unrelated documents yield distant signatures",
          jaccard_estimate(sig_a, sig_c) < 0.2,
          f"J≈{jaccard_estimate(sig_a, sig_c):.2f}")

    near = minhash_dedup(exact, threshold=0.7)
    check("near-dup removes the near copy", len(near) == 2, f"{len(exact)} → {len(near)}")
    check("unique document survives", docs[3] in near)


def test_pipeline_funnel() -> None:
    print("\n7. Pipeline reports a funnel count at every stage (M9 §5)")
    rng = random.Random(0)
    good = [f"Документ номер {i} про языковые модели и обучение на текстовых "
            f"корпусах достаточной длины для прохождения фильтров." for i in range(60)]
    docs = good + good[:20] + ["!!!" * 30] * 10 + ["ок"] * 10
    rng.shuffle(docs)
    survivors, report = pipeline(docs)
    names = [n for n, _ in report.stages]
    check("all stages are present",
          names == ["input", "normalization", "heuristics", "exact dedup",
                    "near-dup dedup"], str(names))
    check("funnel is monotonically non-increasing",
          all(report.stages[i][1] >= report.stages[i + 1][1]
              for i in range(len(report.stages) - 1)))
    check("garbage is removed and meaningful text remains", len(survivors) == 60,
          f"{len(docs)} → {len(survivors)}, survival {report.survival:.1%}")
    check("rejection reasons are recorded", set(report.reasons) >= {"too_short"},
          str(report.reasons))


# ---------------------------------------------------------------- 8-9. streams
def test_shard_stream_matches_memory() -> None:
    print("\n8. Sharded stream reads identically to an in-memory stream")
    tok = ByteTokenizer()
    memory = TokenStream.from_text(toy_corpus(80), tok, seq_len=64)
    first_two = memory.batch(torch.tensor([0, 1]))
    check("window contains seq_len+1 tokens and seq_len transitions",
          first_two.shape[1] == memory.seq_len + 1, str(tuple(first_two.shape)))
    check("boundary between adjacent windows is trained exactly once",
          first_two[0, -1].item() == first_two[1, 0].item())
    with tempfile.TemporaryDirectory() as tmp:
        index = write_shards(memory.tokens.tolist(), tmp, shard_tokens=2500)
        with ShardStream(tmp, seq_len=64, verify=True) as shards, \
                ShardStream(tmp, seq_len=64) as second_view:
            check("there is more than one shard", len(index["shards"]) > 1,
                  f"{len(index['shards'])} shards over {index['total_tokens']:,} tokens")
            check("window counts match", len(shards) == len(memory))
            idx = torch.tensor([0, 1, 17, 42])
            check("batches match bit-for-bit",
                  torch.equal(shards.batch(idx), memory.batch(idx)))
            # A window crossing a shard boundary is the main source of bugs here.
            boundary = 2500 // 64
            check("window crossing a shard boundary reads correctly",
                  torch.equal(shards.batch(torch.tensor([boundary])),
                              memory.batch(torch.tensor([boundary]))))
            check("checksum is stable",
                  shards.checksum() == second_view.checksum())


def test_shard_dtype_range_is_checked_before_cast() -> None:
    print("\n8b. Sharder validates ID range before narrowing dtype")
    with tempfile.TemporaryDirectory() as tmp:
        index = write_shards(np.array([0, 65_535], dtype=np.int64), tmp,
                             shard_tokens=10, dtype="uint16")
        with ShardStream(tmp, seq_len=1) as stream:
            check("uint16 boundary 65,535 is valid",
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
    check("overflow and negative IDs are rejected before cast", all(rejected),
          f"results {rejected}")


def test_loader_cursor() -> None:
    print("\n9. Data cursor: resume continues the stream instead of restarting")
    tok = ByteTokenizer()
    stream = TokenStream.from_text(toy_corpus(80), tok, seq_len=64)

    straight = stream.loader(4, seed=5)
    reference = [next(straight) for _ in range(12)]
    resumed = stream.loader(4, seed=5, start_batch=8)
    tail = [next(resumed) for _ in range(4)]
    check("resumed tail matches uninterrupted reading",
          all(torch.equal(a, b) for a, b in zip(reference[8:], tail)))

    per_epoch = len(stream) // 4
    check("crossing an epoch boundary is deterministic",
          torch.equal(next(stream.loader(4, seed=5, start_batch=per_epoch)),
                      [next(straight) for _ in range(per_epoch - 12 + 1)][-1]))
    unshuffled = stream.loader(4, seed=5, shuffle=False)
    check("without shuffling, order is natural",
          torch.equal(next(unshuffled), stream.batch(torch.arange(4))))


def test_repeat_probe() -> None:
    print("\n10. Long-range probe: layout and baseline (M4 §6)")
    task = RepeatTask(vocab_size=64, train_range=(12, 48), seed=0)
    ids, labels = task.sample(8, half_len=20)
    check("input is a prefix repeated twice",
          bool(torch.equal(ids[:, :20], ids[:, 20:])), f"shape {tuple(ids.shape)}")
    check("first half is not trained", bool((labels[:, :20] == -100).all()))
    check("first repeated token is not trained—nothing predicts it",
          bool((labels[:, 20] == -100).all()))
    check("the rest of the second half is fully trained",
          bool((labels[:, 21:] == ids[:, 21:]).all()),
          f"{int((labels != -100).sum())} trainable positions of {labels.numel()}")

    # Signal density explains why the first “needle in a haystack” probe, with
    # only one trainable position, did not learn at all.
    density = (labels != -100).float().mean().item()
    check("signal density is close to one half", 0.4 < density < 0.5,
          f"{density:.1%} of positions vs ~2.5% for the single-needle variant")

    cfg = MiniLLMConfig.profile("S", vocab_size=64, hidden_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16,
                                max_position_embeddings=256)
    torch.manual_seed(0)
    untrained = MiniLLM(cfg)
    accuracy = repeat_accuracy(untrained, task, half_len=24, batches=4)
    check("untrained model remains at chance level",
          accuracy < 5 * chance_level(task),
          f"{accuracy:.4f} vs {chance_level(task):.4f} — room to improve")
    check("loader is compatible with train()",
          isinstance(next(task.loader(4, seed=0)), tuple))


def main() -> int:
    print("MiniLLM data checks")
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
    print(f"\nResult: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print("  failed:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
