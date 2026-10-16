"""Проверки reference implementation против формул курса.

Это не «тесты ради тестов»: каждая проверка соответствует утверждению, которое
лекция делает словами. Если проверка падает — расходятся либо код, либо лекция.

Запуск:  python3 -m minillm.tests
"""
from __future__ import annotations

import math
import tempfile

import torch

from .config import MiniLLMConfig, gqa_variants, kv_bytes_per_token
from .model import MiniLLM, RMSNorm, RotaryEmbedding, apply_rope

torch.manual_seed(0)
PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- 1. бюджет параметров
def test_parameter_accounting() -> None:
    print("\n1. Бюджет параметров: формула M6 §10 против model.parameters()")
    for profile in ("S", "M"):
        cfg = MiniLLMConfig.profile(profile)
        model = MiniLLM(cfg)
        analytic = cfg.parameter_report()["total"]
        actual = model.num_parameters()
        check(f"профиль {profile}", analytic == actual,
              f"{analytic:,} против {actual:,}")
    # Формула обязана совпадать с кодом при ЛЮБОЙ комбинации флагов, а не только
    # на умолчаниях. Именно здесь раньше пряталось расхождение: для hidden_act
    # "relu" формула считала две матрицы FFN, а FeedForward строил три.
    mismatch = None
    for act in ("swiglu", "geglu", "relu"):
        for extra in ({}, {"attention_sinks": True}, {"qk_norm": True},
                      {"tie_word_embeddings": False}, {"attention_bias": True},
                      {"norm_placement": "peri"}, {"norm_placement": "olmo2"},
                      {"num_local_experts": 4, "num_experts_per_tok": 2,
                       "num_shared_experts": 1}):
            cfg = MiniLLMConfig.profile("S", hidden_act=act, **extra)
            model = MiniLLM(cfg)
            name = f"{act} + {list(extra) or 'база'}"
            if cfg.parameter_report()["total"] != model.num_parameters():
                mismatch = (name, cfg.parameter_report()["total"], model.num_parameters())
                break
        if mismatch:
            break
    check("формула совпадает с кодом на 24 комбинациях флагов", mismatch is None,
          "3 активации × 8 наборов флагов" if mismatch is None
          else f"{mismatch[0]}: {mismatch[1]:,} != {mismatch[2]:,}")

    # Профили L и L-MoE считаем на meta-устройстве: десять миллиардов параметров
    # незачем аллоцировать ради проверки формулы.
    for profile in ("L", "L-MoE"):
        cfg = MiniLLMConfig.profile(profile)
        with torch.device("meta"):
            model = MiniLLM(cfg)
        actual = sum(p.numel() for p in model.parameters())
        check(f"профиль {profile} (meta)", cfg.parameter_report()["total"] == actual,
              f"{cfg.parameter_report()['total']:,} против {actual:,}")


def test_moe_twins() -> None:
    print("\n1b. Разрежённые близнецы: близкое число активных параметров, разная ёмкость (M7 §4–5)")
    for base in ("S", "M", "L"):
        dense = MiniLLMConfig.profile(base)
        moe = MiniLLMConfig.twin(base)
        check(f"{base}: близнец возвращается симметрично",
              MiniLLMConfig.twin(f"{base}-MoE").to_dict() == dense.to_dict())

        ratio = moe.active_parameters() / dense.active_parameters()
        check(f"{base}: число активных параметров различается не более чем на 5%", 0.95 <= ratio <= 1.05,
              f"×{ratio:.4f}: {moe.active_parameters():,} против {dense.active_parameters():,}")

        sparsity = moe.parameter_report()["total"] / moe.active_parameters()
        check(f"{base}: общее число параметров больше активного более чем втрое", sparsity > 3.0,
              f"total ×{moe.parameter_report()['total'] / dense.parameter_report()['total']:.2f}, "
              f"разрежённость ×{sparsity:.2f}")

        # Пара отличается ТОЛЬКО FFN — значит, вся арифметика внимания общая.
        check(f"{base}: KV-кэш на токен совпадает до байта",
              kv_bytes_per_token(dense) == kv_bytes_per_token(moe),
              f"{kv_bytes_per_token(dense):,} Б/токен у обоих")
        check(f"{base}: первый слой оставлен плотным",
              not moe.moe_layer(0) and moe.moe_layer(1),
              "роутер ранних слоёв не успевает специализироваться")

    for profile in ("S-MoE", "M-MoE"):
        cfg = MiniLLMConfig.profile(profile)
        model = MiniLLM(cfg)
        check(f"{profile}: формула совпадает с моделью",
              cfg.parameter_report()["total"] == model.num_parameters(),
              f"{cfg.parameter_report()['total']:,}")


# ------------------------------------------------------------------ 2. прямой/обратный
def test_forward_backward() -> None:
    print("\n2. Forward и backward проходят, лосс конечен")
    cfg = MiniLLMConfig.profile("S")
    model = MiniLLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(ids, labels=ids)
    check("форма логитов", tuple(logits.shape) == (2, 32, cfg.vocab_size))
    check("лосс конечен", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("лосс близок к ln(V) при инициализации",
          abs(loss.item() - math.log(cfg.vocab_size)) < 1.0,
          f"ln(V)={math.log(cfg.vocab_size):.3f}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    check("градиенты не None и конечны",
          all(g is not None and torch.isfinite(g).all() for g in grads))


# ----------------------------------------------------------------------- 3. KV-кэш
def test_kv_cache_equivalence() -> None:
    print("\n3. Инкрементальный декод с кэшем == полный проход (M10 §4)")
    cfg = MiniLLMConfig.profile("S")
    model = MiniLLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full, _ = model(ids)
        cache = model.new_cache()
        stepwise = [model(ids[:, i:i + 1], cache=cache)[0] for i in range(ids.shape[1])]
    stepwise = torch.cat(stepwise, dim=1)
    delta = (full - stepwise).abs().max().item()
    check("максимальное расхождение < 1e-4", delta < 1e-4, f"Δ={delta:.2e}")
    check("длина кэша равна числу токенов", len(cache) == ids.shape[1])


def test_kv_cache_size_formula() -> None:
    print("\n4. Размер кэша == 2·L·Hkv·dh·sizeof(dtype) для MHA/GQA/MQA (M5 §4)")
    base = MiniLLMConfig.profile("S")
    for name, cfg in gqa_variants(base).items():
        model = MiniLLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        cache = model.new_cache()
        with torch.no_grad():
            model(ids, cache=cache)
        expected = kv_bytes_per_token(cfg, dtype_bytes=4) * 24   # float32 на CPU
        check(f"{name}: Hkv={cfg.num_key_value_heads}", cache.nbytes() == expected,
              f"{cache.nbytes()} байт == {expected}")
    mha = kv_bytes_per_token(gqa_variants(base)["MHA"])
    gqa = kv_bytes_per_token(base)
    check("GQA режет кэш ровно в heads/kv_heads раз",
          mha // gqa == base.num_attention_heads // base.num_key_value_heads,
          f"×{mha // gqa}")


# ---------------------------------------------------------------------- 5. RoPE
def test_rope_relativity() -> None:
    print("\n5. Относительность RoPE: скалярное произведение зависит только от m−n (M4 §4)")
    for pairing in ("rotate_half", "adjacent"):
        cfg = MiniLLMConfig.profile("S", rope_pairing=pairing)
        rope = RotaryEmbedding(cfg)
        q = torch.randn(1, 1, 1, cfg.head_dim)
        k = torch.randn(1, 1, 1, cfg.head_dim)
        products = []
        for shift in (0, 5, 17):
            m, n = 3 + shift, 1 + shift          # разность m−n = 2 всегда
            cos_m, sin_m = rope(torch.tensor([m]))
            cos_n, sin_n = rope(torch.tensor([n]))
            qm = apply_rope(q, cos_m, sin_m, pairing)
            kn = apply_rope(k, cos_n, sin_n, pairing)
            products.append((qm * kn).sum().item())
        spread = max(products) - min(products)
        check(f"{pairing}: разброс по абсолютным позициям", spread < 1e-4,
              f"spread={spread:.2e}")


def test_rope_conventions_differ() -> None:
    print("\n6. Две конвенции спаривания несовместимы (M4 §5)")
    cfg_a = MiniLLMConfig.profile("S", rope_pairing="rotate_half")
    cfg_b = MiniLLMConfig.profile("S", rope_pairing="adjacent")
    x = torch.randn(1, 1, 4, cfg_a.head_dim)
    pos = torch.arange(4)
    ca, sa = RotaryEmbedding(cfg_a)(pos)
    cb, sb = RotaryEmbedding(cfg_b)(pos)
    out_a = apply_rope(x, ca, sa, "rotate_half")
    out_b = apply_rope(x, cb, sb, "adjacent")
    check("выходы различаются", (out_a - out_b).abs().max().item() > 1e-3,
          "перепутать конвенции = молча испортить модель")


def test_yarn_spectrum() -> None:
    print("\n7. YaRN: высокие частоты экстраполируются, низкие интерполируются (M4 §7)")
    factor = 8.0
    plain = RotaryEmbedding(MiniLLMConfig.profile("M")).inv_freq
    yarn = RotaryEmbedding(MiniLLMConfig.profile(
        "M", rope_scaling={"rope_type": "yarn", "factor": factor,
                           "original_max_position_embeddings": 1024})).inv_freq
    ratio = (plain / yarn)
    check("самое высокочастотное измерение не тронуто", abs(ratio[0].item() - 1.0) < 1e-3,
          f"ratio={ratio[0].item():.4f}")
    check("самое низкочастотное сжато в factor раз", abs(ratio[-1].item() - factor) < 0.1,
          f"ratio={ratio[-1].item():.3f}")
    check("ratio монотонно неубывает", bool((ratio[1:] >= ratio[:-1] - 1e-6).all()))
    linear = RotaryEmbedding(MiniLLMConfig.profile(
        "M", rope_scaling={"rope_type": "linear", "factor": factor})).inv_freq
    check("linear (PI) сжимает все частоты одинаково",
          torch.allclose(plain / linear, torch.full_like(linear, factor), atol=1e-3))


# ------------------------------------------------------------- 8. размещение нормы
def test_norm_placements() -> None:
    print("\n8. Пять размещений нормы: Peri-LN и OLMo2 имеют разные графы (M6 §6)")
    expected_norms = {"pre": 2, "post": 2, "peri": 4, "olmo2": 2, "none": 0}
    for placement, count in expected_norms.items():
        cfg = MiniLLMConfig.profile("S", norm_placement=placement)
        model = MiniLLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        _, loss = model(ids, labels=ids)
        norms = sum(1 for n, _ in model.layers[0].named_modules() if n.endswith("norm"))
        ok = torch.isfinite(loss).item() and norms == count
        check(f"{placement}", ok, f"норм в блоке: {norms}, loss={loss.item():.3f}")
        check(f"{placement}: бюджет сходится",
              cfg.parameter_report()["total"] == model.num_parameters())


def test_depth_variance() -> None:
    print("\n9. Абляция §6: rms потока по глубине — в постановке лекции и в реальной модели")
    # (а) Игрушечная цепочка ровно из M6 §6: h ← h + W·g(h) с W ~ N(0, 1/d),
    #     единичный gain у слоя. Один и тот же h0 и один набор W во всех режимах.
    torch.manual_seed(1)
    d, layers = 256, 36
    h0 = torch.randn(1, d)
    weights = [torch.randn(d, d) / math.sqrt(d) for _ in range(layers)]
    norm = RMSNorm(d, 1e-6)
    trace = {}
    with torch.no_grad():
        for mode in ("none", "pre", "post"):
            h = h0.clone()
            for W in weights:
                if mode == "none":
                    h = h + torch.tanh(h) @ W
                elif mode == "pre":
                    h = h + torch.tanh(norm(h)) @ W
                else:
                    h = norm(h + torch.tanh(h) @ W)
            trace[mode] = h.pow(2).mean().sqrt().item()
    check("без нормы поток растёт сильнее всех", trace["none"] > trace["pre"] > 0,
          f"none={trace['none']:.1f}, pre={trace['pre']:.2f}, post={trace['post']:.3f}")
    check("post-norm держит ровно единичный rms", abs(trace["post"] - 1.0) < 1e-3,
          f"post={trace['post']:.6f}")

    # (б) Та же абляция на реальной модели при штатной инициализации 0.02.
    #     Эффекта НЕТ: вклады подслоёв слишком малы, чтобы шина успела разъехаться.
    #     Это не баг реализации, а граница масштаба — материал для раздела
    #     «чего этот масштаб показать не может» в ноутбуке абляций.
    rms = {}
    for placement in ("none", "pre"):
        cfg = MiniLLMConfig.profile("S", norm_placement=placement, num_hidden_layers=12)
        model = MiniLLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            x = model.embed_tokens(ids)
            cos, sin = model.rope(torch.arange(16))
            for layer in model.layers:
                x = layer(x, cos, sin)
            rms[placement] = x.float().pow(2).mean().sqrt().item()
    check("на нетренированной 6M-модели взрыва не видно (и это ожидаемо)",
          rms["none"] < 10 * rms["pre"],
          f"none={rms['none']:.3f}, pre={rms['pre']:.3f} — эффект требует "
          f"unit-gain слоёв или обученных весов")


# --------------------------------------------------------------------- 10. sinks
def test_attention_sinks() -> None:
    print("\n10. Attention sinks: сильно отрицательный sink ≈ обычный softmax (M5 §12)")
    cfg = MiniLLMConfig.profile("S", attention_sinks=True)
    model = MiniLLM(cfg).eval()
    ref = MiniLLM(MiniLLMConfig.profile("S", attention_sinks=False)).eval()
    ref.load_state_dict({k: v for k, v in model.state_dict().items()
                         if "sinks" not in k}, strict=False)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.sinks.fill_(-30.0)      # exp(-30) ≈ 0 — колонка sink «выключена»
        a, _ = model(ids)
        b, _ = ref(ids)
    check("совпадение с SDPA-путём", (a - b).abs().max().item() < 1e-3,
          f"Δ={(a - b).abs().max().item():.2e}")
    check("sink добавляет ровно H параметров на слой",
          cfg.parameter_report()["total"] == model.num_parameters())


# ------------------------------------------------------------ 11. скользящее окно
def test_sliding_window() -> None:
    print("\n11. Sliding window: токен не видит дальше окна (M5 §11)")
    cfg = MiniLLMConfig.profile(
        "S", sliding_window=4,
        layer_types=["sliding", "full", "sliding", "full", "sliding", "full"])
    model = MiniLLM(cfg).eval()
    positions = torch.arange(8)
    attn = model.layers[0].attn
    allowed = attn._causal_mask(positions, key_start=0, key_count=8)
    check("позиция 7 видит ровно 4 ключа", allowed[7].sum().item() == 4,
          f"видимых: {allowed[7].sum().item()}")
    full_allowed = model.layers[1].attn._causal_mask(positions, key_start=0, key_count=8)
    check("full-слой видит все 8", full_allowed[7].sum().item() == 8)

    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = model(ids)
        cache = model.new_cache()
        incremental = torch.cat([model(ids[:, i:i+1], cache=cache)[0]
                                 for i in range(ids.shape[1])], dim=1)
    check("bounded cache сохраняет точность decode",
          (full - incremental).abs().max().item() < 1e-4)
    stored = [entry.key.shape[-2] for entry in cache.layers]
    check("sliding-слои ограничены окном, full-слои растут",
          stored == [4, 12, 4, 12, 4, 12], str(stored))


# ---------------------------------------------------------- 11b. packed isolation
def test_packed_dialogue_isolation() -> None:
    print("\n11b. Packed SFT: independent dialogues cannot attend across segments")
    cfg = MiniLLMConfig.profile("S", vocab_size=128, max_position_embeddings=64)
    model = MiniLLM(cfg).eval()
    first_a = torch.tensor([[1, 2, 3, 4, 10, 11, 12, 13]])
    first_b = torch.tensor([[8, 7, 6, 5, 10, 11, 12, 13]])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        isolated_a, _ = model(first_a, segment_ids=segments)
        isolated_b, _ = model(first_b, segment_ids=segments)
        mixed_a, _ = model(first_a)
        mixed_b, _ = model(first_b)
    check("второй диалог инвариантен к содержимому первого",
          torch.equal(isolated_a[:, 4:], isolated_b[:, 4:]))
    check("контроль без segment mask действительно обнаруживает утечку",
          (mixed_a[:, 4:] - mixed_b[:, 4:]).abs().max().item() > 1e-4)


# ------------------------------------------------------------------------ 12. MoE
def test_moe() -> None:
    print("\n12. MoE: бюджет total/active и aux-loss-free балансировка (M7 §5, §8)")
    torch.manual_seed(0)          # тест не должен зависеть от порядка запуска
    # A deliberately tiny MoE keeps this regression test fast on CPU while
    # exercising the same routing and out-of-graph bias update contract.
    cfg = MiniLLMConfig.profile(
        "S", vocab_size=256, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, num_local_experts=8,
        num_experts_per_tok=2, num_shared_experts=1,
        first_k_dense_replace=1, router_bias_speed=5e-3)
    model = MiniLLM(cfg)
    check("бюджет total сходится",
          cfg.parameter_report()["total"] == model.num_parameters(),
          f"{cfg.parameter_report()['total']:,} == {model.num_parameters():,}")
    total, active = cfg.parameter_report()["total"], cfg.active_parameters()
    check("активных заметно меньше total", active < total,
          f"{active:,} из {total:,} = {active / total:.1%}")

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(ids, labels=ids)
    check("forward MoE конечен", torch.isfinite(loss).item(), f"loss={loss.item():.3f}")

    moe = model.layers[-1].ffn
    # A single batch can remain imbalanced even when the loss-free controller
    # balances traffic over time: top-k is discrete. Test the documented
    # long-run contract on a stationary, deliberately tied routing problem.
    fixed_hidden = torch.zeros(4, 16, cfg.hidden_size)

    def mean_selected_load(*, update: bool, rounds: int = 64) -> torch.Tensor:
        moe.expert_bias.zero_()
        total = torch.zeros(cfg.num_local_experts)
        for _ in range(rounds):
            with torch.no_grad():
                moe(fixed_hidden)
            total += moe.selected_load.cpu()
            if update:
                moe.update_expert_bias()
        return total / rounds

    baseline = mean_selected_load(update=False)
    balanced = mean_selected_load(update=True)
    before, after = baseline.std().item(), balanced.std().item()
    check("долговременный разброс выбранной нагрузки упал", after < 0.25 * before,
          f"std средней нагрузки: {before:.4f} → {after:.4f}")
    check("средняя нагрузка приблизилась к target", 
          torch.allclose(balanced, torch.full_like(balanced, cfg.num_experts_per_tok / cfg.num_local_experts), atol=0.04),
          f"load={balanced.tolist()}")
    check("смещение ушло от нуля", moe.expert_bias.abs().max().item() > 0,
          f"max|b|={moe.expert_bias.abs().max().item():.4f}")
    check("смещение не попало в параметры (оно вне графа)",
          all("expert_bias" not in n for n, _ in model.named_parameters()))


# ----------------------------------------------------------------- 13. HF-паспорт
def test_moe_capacity() -> None:
    print("\n12b. Ёмкость эксперта и отбрасывание токенов (M7 §12)")
    torch.manual_seed(0)
    base = dict(vocab_size=64, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                max_position_embeddings=64, num_local_experts=8,
                num_experts_per_tok=2)
    ids = torch.randint(0, 64, (4, 32))
    tokens = ids.numel()

    def run(**extra):
        cfg = MiniLLMConfig.profile("S", **{**base, **extra})
        torch.manual_seed(0)
        model = MiniLLM(cfg).eval()
        with torch.no_grad():
            model(ids)
        return cfg, model.layers[0].ffn

    cfg_none, moe_none = run()
    check("без лимита ничего не отбрасывается",
          cfg_none.expert_capacity(tokens) is None
          and moe_none.dropped_fraction.item() == 0.0)

    _, moe_wide = run(capacity_factor=2.0)
    check("щедрый лимит тоже не отбрасывает", moe_wide.dropped_fraction.item() == 0.0,
          "запас вдвое покрывает неравномерность роутера")

    cfg_tight, moe_tight = run(capacity_factor=1.0)
    dropped = moe_tight.dropped_fraction.item()
    check("лимит «ровно по среднему» уже отбрасывает заметную долю", dropped > 0.10,
          f"{dropped:.1%} назначений при capacity={cfg_tight.expert_capacity(tokens)} — "
          "маршрутизация никогда не бывает равномерной")
    cap_share = cfg_tight.expert_capacity(tokens) / tokens
    check("нагрузка каждого эксперта обрезана лимитом",
          bool((moe_tight.expert_load <= cap_share + 1e-6).all()),
          f"потолок {cap_share:.3f}")

    _, moe_half = run(capacity_factor=0.5)
    check("вдвое меньший лимит отбрасывает вдвое больше",
          moe_half.dropped_fraction.item() > 2 * dropped,
          f"{moe_half.dropped_fraction.item():.1%} против {dropped:.1%}")

    # --- перенормировка выживших гейтов
    cfg, moe = run(capacity_factor=1.0)
    x = torch.randn(4, 32, cfg.hidden_size)
    with torch.no_grad():
        cfg.renormalize_on_drop = True
        with_renorm = moe(x)
        cfg.renormalize_on_drop = False
        without = moe(x)
    ratio = without.norm().item() / with_renorm.norm().item()
    check("без перенормировки норма выхода систематически меньше", ratio < 0.95,
          f"×{ratio:.3f} — гейты суммировались в единицу, а слагаемое выпало")
    check("перенормировка не трогает случай без отбрасывания",
          torch.allclose(run(capacity_factor=2.0)[1](x),
                         run(capacity_factor=2.0, renormalize_on_drop=False)[1](x)))

    # --- политика отбрасывания
    kept_weight = {}
    for policy in ("position", "score"):
        cfg_p = MiniLLMConfig.profile("S", **{**base, "capacity_factor": 1.0,
                                              "drop_policy": policy})
        torch.manual_seed(0)
        model = MiniLLM(cfg_p).eval()
        moe_p = model.layers[0].ffn
        with torch.no_grad():
            model(ids)
        kept_weight[policy] = moe_p.expert_load.sum().item()
    check("политика score сохраняет не меньше нагрузки, чем position",
          kept_weight["score"] >= kept_weight["position"] - 1e-6,
          f"{kept_weight['score']:.3f} против {kept_weight['position']:.3f}")

    # --- главное: авария тихая
    cfg_loud, model_loud = MiniLLMConfig.profile("S", **{**base, "capacity_factor": 0.5}), None
    torch.manual_seed(0)
    model_loud = MiniLLM(cfg_loud)
    _, loss = model_loud(ids, labels=ids)
    check("при половине отброшенных назначений лосс конечен и выглядит обычно",
          torch.isfinite(loss).item() and abs(loss.item() - math.log(64)) < 1.0,
          f"loss={loss.item():.3f} при ln(V)={math.log(64):.3f} — "
          "ни исключения, ни всплеска: долю отброшенных надо писать в лог отдельно")


def test_config_roundtrip() -> None:
    print("\n13. config.json читается паспортным скриптом модуля 1")
    cfg = MiniLLMConfig.profile("M", num_local_experts=4, attention_sinks=True)
    with tempfile.TemporaryDirectory(prefix="minillm-config-") as tmp:
        path = cfg.save_pretrained(tmp)
        restored = MiniLLMConfig.from_pretrained(tmp)
    check("round-trip конфига", restored.to_dict() == cfg.to_dict(), str(path))
    keys = set(cfg.to_dict())
    passport = {"hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "rope_theta",
                "num_local_experts", "num_experts_per_tok", "tie_word_embeddings",
                "sliding_window", "layer_types", "rms_norm_eps", "intermediate_size"}
    check("все паспортные поля модуля 1 присутствуют", passport <= keys,
          f"отсутствуют: {sorted(passport - keys) or '—'}")


def main() -> int:
    print("Проверки MiniLLM (reference implementation капстоуна)")
    test_parameter_accounting()
    test_moe_twins()
    test_forward_backward()
    test_kv_cache_equivalence()
    test_kv_cache_size_formula()
    test_rope_relativity()
    test_rope_conventions_differ()
    test_yarn_spectrum()
    test_norm_placements()
    test_depth_variance()
    test_attention_sinks()
    test_sliding_window()
    test_packed_dialogue_isolation()
    test_moe()
    test_moe_capacity()
    test_config_roundtrip()
    print(f"\nИтог: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    for name in FAILED:
        print("  провал:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
