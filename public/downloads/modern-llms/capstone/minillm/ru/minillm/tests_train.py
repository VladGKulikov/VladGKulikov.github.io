"""Проверки тренировочного контура. Запуск: python3 -m minillm.tests_train

Каждая проверка соответствует утверждению модуля 9. Прогон на CPU занимает
пару минут — это осознанный выбор: тест, который никто не запускает, не тест.
"""
from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import torch

from .config import MiniLLMConfig
from .data import ByteTokenizer, TokenStream, toy_corpus
from .model import MiniLLM, MoELayer
from .precision import resolve_compute_dtype
from .train import (Muon, TrainConfig, TrainState, build_optimizer, evaluate,
                    load_checkpoint, load_model_artifact, memory_report,
                    optimizer_partition, save_checkpoint, save_model_artifact,
                    train, wsd_lr_scale, zeropower_parameter,
                    zeropower_via_newtonschulz)

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _tiny_setup(**overrides):
    tok = ByteTokenizer()
    # Small CPU geometry: these are unit tests for contracts, not quality runs.
    mcfg = MiniLLMConfig.profile("S", vocab_size=tok.vocab_size, hidden_size=64,
                                 num_hidden_layers=1, head_dim=16,
                                 num_attention_heads=4, num_key_value_heads=2,
                                 max_position_embeddings=64)
    defaults = dict(steps=20, seq_len=32, micro_batch_size=4, log_every=100)
    tcfg = TrainConfig(**{**defaults, **overrides})
    stream = TokenStream.from_text(toy_corpus(60), tok, tcfg.seq_len)
    return mcfg, tcfg, stream


# --------------------------------------------------------------- 1. расписание
def test_precision_policy() -> None:
    print("\n0. Precision policy: storage, compute and cache are distinct")
    check("CPU auto resolves to fp32", resolve_compute_dtype("cpu", "auto") == "fp32")
    try:
        resolve_compute_dtype("cpu", "bf16")
        rejected = False
    except ValueError:
        rejected = True
    check("CPU reference path rejects explicit mixed precision", rejected)


def test_wsd_schedule() -> None:
    print("\n1. WSD-расписание: warmup → полка → спад (M9 §11)")
    cfg = TrainConfig(steps=100, warmup_frac=0.1, decay_frac=0.2, min_lr_ratio=0.1)
    scales = [wsd_lr_scale(s, cfg) for s in range(cfg.steps)]
    check("warmup растёт от малого к единице",
          scales[0] < scales[5] < scales[9] and abs(scales[9] - 1.0) < 1e-9,
          f"{scales[0]:.2f} → {scales[9]:.2f}")
    check("полка ровно единица", all(abs(s - 1.0) < 1e-9 for s in scales[10:80]))
    check("последний шаг точно равен min_lr_ratio",
          abs(scales[-1] - cfg.min_lr_ratio) < 1e-12, f"последний={scales[-1]:.3f}")
    check("расписание монотонно на спаде",
          all(scales[i] >= scales[i + 1] for i in range(80, 99)))
    short = TrainConfig(steps=5, warmup_frac=0.2, decay_frac=0.2, min_lr_ratio=0.1)
    check("одиночный decay-step состоит из конечной точки",
          abs(wsd_lr_scale(4, short) - 0.1) < 1e-12)
    ten = TrainConfig(steps=10, warmup_frac=0.1, decay_frac=0.2, min_lr_ratio=0.1)
    check("короткое расписание тоже достигает endpoint",
          abs(wsd_lr_scale(9, ten) - 0.1) < 1e-12)


# ------------------------------------------------------------------- 2. Muon
def test_newton_schulz_orthogonalizes() -> None:
    print("\n2. Newton–Schulz выравнивает спектр обновления (M9 §9)")
    torch.manual_seed(0)
    G = torch.randn(64, 48)
    G = G @ torch.diag(torch.linspace(1.0, 0.01, 48))   # заведомо перекошенный спектр
    before = torch.linalg.svdvals(G)
    after = torch.linalg.svdvals(zeropower_via_newtonschulz(G))
    check("разброс сингулярных чисел резко падает",
          after.max() / after.min() < before.max() / before.min() / 10,
          f"было ×{before.max() / before.min():.1f}, стало ×{after.max() / after.min():.2f}")
    check("сингулярные числа близки к единице",
          bool(((after - 1.0).abs() < 0.35).all()),
          f"диапазон [{after.min():.3f}, {after.max():.3f}]")
    check("форма сохранена", zeropower_via_newtonschulz(G).shape == G.shape)
    check("работает и для «широких» матриц",
          zeropower_via_newtonschulz(torch.randn(16, 96)).shape == (16, 96))
    bank = torch.randn(4, 16, 24)
    check("3D expert bank обрабатывается как пакет независимых матриц",
          zeropower_parameter(bank).shape == bank.shape)


def test_optimizer_partition() -> None:
    print("\n3. Гибрид Muon+AdamW: кто куда попал и сколько это стоит по памяти (M9 §10, §12)")
    mcfg, tcfg, _ = _tiny_setup(optimizer="muon")
    model = MiniLLM(mcfg)
    opt = build_optimizer(model, tcfg)
    assigned = sum(len(g["params"]) for o in opt.optimizers for g in o.param_groups)
    check("каждый параметр ровно у одного оптимизатора",
          assigned == len([p for p in model.parameters() if p.requires_grad]),
          f"{assigned} назначено")
    muon = opt.optimizers[opt.names.index("muon")]
    muon_params = [p for g in muon.param_groups for p in g["params"]]
    check("под Muon только 2D-матрицы и 3D expert banks",
          all(p.ndim in (2, 3) for p in muon_params), f"{len(muon_params)} параметров")
    embed = model.embed_tokens.weight
    check("эмбеддинги остались под AdamW",
          not any(p is embed for p in muon_params))

    moe_cfg = MiniLLMConfig.profile("S-MoE")
    moe_model = MiniLLM(moe_cfg)
    partition = optimizer_partition(moe_model, TrainConfig(optimizer="muon"))
    muon_names = {name for name, _ in partition["muon"]}
    check("3D routed expert banks принадлежат Muon",
          any(name.endswith("gate_w") for name in muon_names) and
          any(name.endswith("up_w") for name in muon_names) and
          any(name.endswith("down_w") for name in muon_names))
    check("router и embeddings остаются под AdamW",
          not any("router" in name or "embed_tokens" in name for name in muon_names))

    # Состояния: у Muon один буфер на параметр, у AdamW — два.
    ids = torch.randint(0, mcfg.vocab_size, (2, 32))
    _, loss = model(ids, labels=ids)
    loss.backward()
    opt.step()
    per_param = {}
    for name, o in zip(opt.names, opt.optimizers):
        tensors = [sum(1 for v in st.values() if torch.is_tensor(v)) for st in o.state.values()]
        per_param[name] = max(tensors) if tensors else 0
    check("Muon хранит один тензор состояния на параметр", per_param["muon"] == 1,
          f"muon={per_param['muon']}, adamw={per_param['adamw']}")


def test_memory_report() -> None:
    print("\n4. Учёт постоянной памяти следует фактическому optimizer ownership (M9 §12)")
    dense = MiniLLMConfig.profile("L")
    adam = memory_report(dense, optimizer="adamw")
    muon = memory_report(dense, optimizer="muon")
    check("AdamW: FP32 weights+grads+two moments = 16 bytes/parameter",
          abs(adam["bytes_per_parameter_average"] - 16.0) < 1e-9, str(adam))
    check("Muon-hybrid экономит только на реально принадлежащих ему параметрах",
          12.0 <= muon["bytes_per_parameter_average"] < 16.0, str(muon))
    check("ownership fractions sum to one",
          abs(muon["ownership"]["muon_fraction"] +
              muon["ownership"]["adamw_fraction"] - 1.0) < 1e-12)

    moe_cfg = MiniLLMConfig.profile("L-MoE")
    moe = memory_report(moe_cfg, optimizer="muon")
    check("L-MoE 3D expert banks are included in Muon ownership",
          moe["ownership"]["muon_fraction"] > 0.9,
          f"{moe['ownership']['muon_fraction']:.2%} under Muon")
    check("reported payload equals weights/gradients plus optimizer states",
          abs(moe["payload_gib"] -
              round(moe["weights_and_gradients_gib"] + moe["optimizer_state_gib"], 2)) <= 0.01)



# ------------------------------------------------------- 5. накопление градиента
def test_grad_accumulation_equivalence() -> None:
    print("\n5. Накопление градиента == один большой батч (M9 §13)")
    mcfg, tcfg, stream = _tiny_setup()
    torch.manual_seed(0)
    model = MiniLLM(mcfg)
    big = stream.batch(torch.arange(8))

    model.zero_grad()
    _, loss = model(big, labels=big)
    loss.backward()
    reference = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    for half in (big[:4], big[4:]):
        _, l = model(half, labels=half)
        (l / 2).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    worst = max((a - b).abs().max().item() for a, b in zip(reference, accumulated))
    check("градиенты совпадают", worst < 1e-5, f"max|Δ| = {worst:.2e}")


def test_moe_balance_uses_the_effective_batch() -> None:
    print("\n5b. Балансировка MoE агрегирует все microbatch эффективного батча")
    tok = ByteTokenizer()
    mcfg = MiniLLMConfig.profile(
        "S", vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, num_local_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=64)
    stream = TokenStream.from_text(toy_corpus(20), tok, seq_len=32)
    biases = {}
    for label, micro, accum in (("big", 8, 1), ("accumulated", 4, 2)):
        torch.manual_seed(0)
        model = MiniLLM(mcfg)
        cfg = TrainConfig(steps=1, seq_len=32, micro_batch_size=micro,
                          grad_accum=accum, lr=0.0, weight_decay=0.0,
                          seed=7, log_every=1)
        train(model, stream, cfg)
        biases[label] = model.layers[0].ffn.expert_bias.clone()
    check("bias-update совпадает с одним большим батчем",
          torch.equal(biases["big"], biases["accumulated"]),
          f"big={biases['big'].tolist()}, accum={biases['accumulated'].tolist()}")


def test_gradient_checkpointing_equivalence() -> None:
    print("\n6. Пересчёт активаций не меняет градиенты (M9 §14)")
    mcfg, tcfg, stream = _tiny_setup()
    ids = stream.batch(torch.arange(4))
    grads = {}
    for flag in (False, True):
        torch.manual_seed(0)
        model = MiniLLM(mcfg)
        model.train()
        model.gradient_checkpointing = flag
        model.zero_grad()
        _, loss = model(ids, labels=ids)
        loss.backward()
        grads[flag] = [p.grad.clone() for p in model.parameters()]
    worst = max((a - b).abs().max().item() for a, b in zip(grads[False], grads[True]))
    check("градиенты совпадают", worst < 1e-5, f"max|Δ| = {worst:.2e}")


# ----------------------------------------------------------------- 7. обучение
def test_training_reduces_loss() -> None:
    print("\n7. Модель действительно учится обеими ветками оптимизатора")
    for optimizer in ("adamw", "muon"):
        mcfg, tcfg, stream = _tiny_setup(optimizer=optimizer, steps=40,
                                         log_every=1000, seed=0)
        torch.manual_seed(0)
        model = MiniLLM(mcfg)
        start = evaluate(model, stream, tcfg, batches=4)
        state = train(model, stream, tcfg)
        end = evaluate(model, stream, tcfg, batches=4)
        uniform = math.log(mcfg.vocab_size)
        check(f"{optimizer}: лосс упал заметно ниже ln(V)", end < 0.5 * uniform,
              f"{start:.3f} → {end:.3f} при ln(V)={uniform:.3f}")
        check(f"{optimizer}: история пишется", len(state.history) >= 1)


def test_moe_training_balances() -> None:
    print("\n8. При обучении MoE смещение экспертов реально двигается (M7 §8)")
    tok = ByteTokenizer()
    mcfg = MiniLLMConfig.profile("S", vocab_size=tok.vocab_size, hidden_size=64,
                                 num_hidden_layers=1, head_dim=16,
                                 num_attention_heads=4, num_key_value_heads=2,
                                 max_position_embeddings=64,
                                 num_local_experts=4, num_experts_per_tok=2,
                                 moe_intermediate_size=64)
    tcfg = TrainConfig(steps=16, seq_len=32, micro_batch_size=4, log_every=1000)
    stream = TokenStream.from_text(toy_corpus(30), tok, tcfg.seq_len)
    torch.manual_seed(0)
    model = MiniLLM(mcfg)
    state = train(model, stream, tcfg)
    moe = model.layers[0].ffn
    check("слой действительно MoE", isinstance(moe, MoELayer))
    check("смещение сдвинулось", moe.expert_bias.abs().max().item() > 0,
          f"max|b| = {moe.expert_bias.abs().max().item():.4f}")
    check("selected load сохраняет ровно k назначений на токен",
          abs(moe.selected_load.sum().item() - mcfg.num_experts_per_tok) < 1e-6,
          f"sum={moe.selected_load.sum().item():.3f}, k={mcfg.num_experts_per_tok}")
    check("без capacity limit selected и kept load совпадают",
          torch.allclose(moe.selected_load, moe.expert_load),
          f"selected={moe.selected_load.tolist()}, kept={moe.expert_load.tolist()}")
    check("история содержит диагностический CV роутера",
          "selected_load_cv" in state.history[-1], str(state.history[-1]))


def test_capacity_is_visible_in_log() -> None:
    print("\n8b. Переполнение ёмкости попадает в лог — иначе авария тихая (M7 §12)")
    tok = ByteTokenizer()
    common = dict(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=1,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=64, num_local_experts=8,
                  num_experts_per_tok=2, moe_intermediate_size=64)
    tcfg = TrainConfig(steps=12, seq_len=32, micro_batch_size=4, optimizer="muon",
                       seed=0, log_every=6)
    stream = TokenStream.from_text(toy_corpus(30), tok, 32)

    histories = {}
    for capacity in (None, 1.0):
        cfg = MiniLLMConfig.profile("S", **{**common, "capacity_factor": capacity})
        torch.manual_seed(0)
        model = MiniLLM(cfg)
        histories[capacity] = train(model, stream, tcfg).history

    check("поле доли отброшенных есть в истории обоих прогонов",
          all("dropped_fraction" in h[-1] for h in histories.values()))
    check("без лимита отброшенных нет", histories[None][-1]["dropped_fraction"] == 0.0)
    tight = histories[1.0][-1]["dropped_fraction"]
    check("с лимитом «по среднему» отбрасывается заметная доля", tight > 0.2,
          f"{tight:.1%} назначений")

    # Вот в чём тихость: лосс падает в обоих случаях, и по нему ничего не видно.
    loose_loss = histories[None][-1]["loss"]
    tight_loss = histories[1.0][-1]["loss"]
    check("лосс с потерями всё равно убывает и не выглядит аварийным",
          tight_loss < math.log(tok.vocab_size),
          f"{tight_loss:.4f} против {loose_loss:.4f} без лимита при "
          f"ln(V)={math.log(tok.vocab_size):.3f}")


# --------------------------------------------------------------- 9. чекпойнты
def test_resume_is_exact() -> None:
    print("\n9. Возобновление после разрыва даёт тот же результат (M9 §16)")
    mcfg, tcfg, stream = _tiny_setup(steps=16, log_every=1000, seed=3)

    torch.manual_seed(0)
    reference = MiniLLM(mcfg)
    train(reference, stream, tcfg)

    torch.manual_seed(0)
    interrupted = MiniLLM(mcfg)
    state = train(interrupted, stream, tcfg, stop_at_step=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt"
        save_checkpoint(path, interrupted, state.optimizer, tcfg, state)
        model2, _, cfg2, state2 = load_checkpoint(path)
        check("шаг восстановлен", state2.step == 8, f"step={state2.step}")
        train(model2, stream, tcfg, state=state2)

    worst = max((a - b).abs().max().item()
                for a, b in zip(reference.state_dict().values(),
                                model2.state_dict().values()))
    check("веса совпадают побитово с непрерывным забегом", worst == 0.0,
          f"max|Δ| = {worst:.3e}")
    check("конфиг обучения пережил round-trip", cfg2.seed == tcfg.seed)
    check("запрошенное устройство переопределяет stale checkpoint device", cfg2.device == "cpu")


def test_inference_artifact_has_no_stale_optimizer() -> None:
    print("\n10. Inference-артефакт хранит веса и JSON provenance, но не чужой optimizer")
    mcfg, _, _ = _tiny_setup()
    torch.manual_seed(11)
    model = MiniLLM(mcfg)
    with tempfile.TemporaryDirectory() as tmp:
        save_model_artifact(tmp, model, {"stage": "dpo", "step": 17})
        restored, metadata = load_model_artifact(tmp)
        files = {p.name for p in Path(tmp).iterdir()}
    worst = max((a - b).abs().max().item()
                for a, b in zip(model.state_dict().values(), restored.state_dict().values()))
    check("веса inference-артефакта совпадают", worst == 0.0, f"max|Δ|={worst:.2e}")
    check("provenance пережил JSON round-trip", metadata == {"stage": "dpo", "step": 17})
    check("optimizer state намеренно отсутствует",
          files == {"config.json", "metadata.json", "model.pt"}, str(sorted(files)))



class _FixedExamples:
    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def loader(self, batch_size, seed=0, shuffle=False, start_batch=0):
        batches = [(self.ids[i:i + batch_size], self.labels[i:i + batch_size])
                   for i in range(0, len(self.ids), batch_size)]
        index = start_batch
        while True:
            yield batches[index % len(batches)]
            index += 1


def test_masked_token_mean() -> None:
    print("\n11. Маски целей: оценка и train() сохраняют среднее по токенам")
    torch.manual_seed(37)
    model = MiniLLM(MiniLLMConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        intermediate_size=16, initializer_range=0.3))
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [2, 4, 6, 8, 10]])
    labels = ids.clone()
    labels[0, :4] = -100
    labels[1, 0] = -100
    labels[2] = -100
    stream = _FixedExamples(ids, labels)
    # Независимый эталон: сумма CE пяти целей, без усреднения потерь батчей.
    logits, _ = model(ids)
    oracle = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum") / 5
    oracle.backward()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    expected_loss = oracle.item()
    losses, gradients = [], []
    for micro, accum in ((3, 1), (1, 3)):
        cfg = TrainConfig(steps=1, micro_batch_size=micro, grad_accum=accum,
                          lr=0.0, weight_decay=0.0, grad_clip=1e9, log_every=1)
        current = copy.deepcopy(model)
        measured = evaluate(current, stream, cfg, batches=accum)
        check("оценка совпадает с независимой суммой потерь целей",
              abs(measured - expected_loss) < 1e-6, f"batch={micro}, loss={measured:.7f}")
        state = train(current, stream, cfg)
        losses.append(state.history[-1]["loss"])
        gradients.append([p.grad.clone() for p in current.parameters()])
        check("после сдвига учтены пять обучаемых позиций",
              state.supervised_tokens == 5)
        check("учтены только позиции, фактически переданные модели",
              state.tokens == (15 if micro == 3 else 10))
    worst = max((a - b).abs().max().item()
                for result in gradients for a, b in zip(result, expected_grads))
    check("градиенты совпадают при разных масках и накоплении",
          worst < 2e-6, f"max|delta|={worst:.2e}")
    check("loss в журнале усреднён по тем же токенам",
          all(abs(loss - expected_loss) < 1e-6 for loss in losses))
    empty = _FixedExamples(ids, torch.full_like(labels, -100))
    for operation in (evaluate, train):
        current = copy.deepcopy(model).train()
        try:
            operation(current, empty, TrainConfig(steps=1, micro_batch_size=3))
        except ValueError as exc:
            rejected = "no supervised next-token targets" in str(exc)
        else:
            rejected = False
        check("отсутствие обучаемых целей вызывает явную ошибку", rejected, operation.__name__)
        check("режим модели сохранён при ошибке оценки", current.training)


def test_variable_length_counters() -> None:
    print("\n12. Счётчики входных позиций и целей сохраняются при продолжении")
    mcfg, _, _ = _tiny_setup()
    class VariableStream:
        def loader(self, batch_size, seed=0, shuffle=False, start_batch=0):
            index = start_batch
            while True:
                length = (4, 7)[index % 2]
                ids = torch.arange(length).repeat(batch_size, 1)
                labels = ids.clone()
                labels[:, :2] = -100
                yield ids, labels
                index += 1
    cfg = TrainConfig(steps=2, seq_len=64, micro_batch_size=2, log_every=1,
                      peak_tflops=1.0, lr=0.0)
    model = MiniLLM(mcfg)
    state = train(model, VariableStream(), cfg, stop_at_step=1)
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(tmp, model, state.optimizer, cfg, state)
        restored, _, restored_cfg, resumed = load_checkpoint(tmp)
        train(restored, VariableStream(), restored_cfg, state=resumed)
        check("накопленные счётчики включают оба батча разной длины",
              resumed.tokens == 22 and resumed.supervised_tokens == 14)
        last = resumed.history[-1]
        check("счётчики вызова не включают работу до продолжения",
              last["run_tokens"] == 14 and last["run_supervised_tokens"] == 10)
        check("единица скорости явно задана входными позициями",
              last["token_basis"] == "input_positions")
        check("MFU использует измеренную скорость входных позиций",
              abs(last["mfu"] - 6 * mcfg.active_parameters() * last["tokens_per_s"] / 1e12) < 1e-12)
        blob = torch.load(Path(tmp) / "state.pt", weights_only=False)
        del blob["tokens"], blob["supervised_tokens"]
        torch.save(blob, Path(tmp) / "state.pt")
        old_model, _, old_cfg, old_state = load_checkpoint(tmp)
        train(old_model, VariableStream(), old_cfg, state=old_state)
        check("для старого чекпойнта итог неизвестен, текущий вызов измерен",
              old_state.tokens is None and old_state.supervised_tokens is None
              and old_state.history[-1]["run_tokens"] == 14)


def main() -> int:
    print("Проверки тренировочного контура MiniLLM")
    test_precision_policy()
    test_wsd_schedule()
    test_newton_schulz_orthogonalizes()
    test_optimizer_partition()
    test_memory_report()
    test_grad_accumulation_equivalence()
    test_moe_balance_uses_the_effective_batch()
    test_gradient_checkpointing_equivalence()
    test_training_reduces_loss()
    test_moe_training_balances()
    test_capacity_is_visible_in_log()
    test_resume_is_exact()
    test_inference_artifact_has_no_stale_optimizer()
    test_masked_token_mean()
    test_variable_length_counters()
    print(f"\nИтог: {len(PASSED)} пройдено, {len(FAILED)} провалено")
    for name in FAILED:
        print("  провал:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
