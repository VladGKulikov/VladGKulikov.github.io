"""Training-loop checks. Run: python3 -m minillm.tests_train

Every check corresponds to a module 9 claim. The CPU run takes a couple of
minutes by design: a test that nobody runs is not a test.
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


# ------------------------------------------------------------------ 1. schedule
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
    print("\n1. WSD schedule: warmup → plateau → decay (M9 §11)")
    cfg = TrainConfig(steps=100, warmup_frac=0.1, decay_frac=0.2, min_lr_ratio=0.1)
    scales = [wsd_lr_scale(s, cfg) for s in range(cfg.steps)]
    check("warmup rises from a small value to one",
          scales[0] < scales[5] < scales[9] and abs(scales[9] - 1.0) < 1e-9,
          f"{scales[0]:.2f} → {scales[9]:.2f}")
    check("plateau is exactly one", all(abs(s - 1.0) < 1e-9 for s in scales[10:80]))
    check("final step equals min_lr_ratio exactly",
          abs(scales[-1] - cfg.min_lr_ratio) < 1e-12, f"last={scales[-1]:.3f}")
    check("schedule is monotonic during decay",
          all(scales[i] >= scales[i + 1] for i in range(80, 99)))
    short = TrainConfig(steps=5, warmup_frac=0.2, decay_frac=0.2, min_lr_ratio=0.1)
    check("a one-step decay consists only of the endpoint",
          abs(wsd_lr_scale(4, short) - 0.1) < 1e-12)
    ten = TrainConfig(steps=10, warmup_frac=0.1, decay_frac=0.2, min_lr_ratio=0.1)
    check("short schedule also reaches the endpoint",
          abs(wsd_lr_scale(9, ten) - 0.1) < 1e-12)


# ------------------------------------------------------------------- 2. Muon
def test_newton_schulz_orthogonalizes() -> None:
    print("\n2. Newton–Schulz flattens the update spectrum (M9 §9)")
    torch.manual_seed(0)
    G = torch.randn(64, 48)
    G = G @ torch.diag(torch.linspace(1.0, 0.01, 48))   # deliberately skewed spectrum
    before = torch.linalg.svdvals(G)
    after = torch.linalg.svdvals(zeropower_via_newtonschulz(G))
    check("singular-value spread drops sharply",
          after.max() / after.min() < before.max() / before.min() / 10,
          f"before ×{before.max() / before.min():.1f}, after ×{after.max() / after.min():.2f}")
    check("singular values are close to one",
          bool(((after - 1.0).abs() < 0.35).all()),
          f"range [{after.min():.3f}, {after.max():.3f}]")
    check("shape is preserved", zeropower_via_newtonschulz(G).shape == G.shape)
    check("also works for wide matrices",
          zeropower_via_newtonschulz(torch.randn(16, 96)).shape == (16, 96))
    bank = torch.randn(4, 16, 24)
    check("3D expert bank is treated as a batch of independent matrices",
          zeropower_parameter(bank).shape == bank.shape)


def test_optimizer_partition() -> None:
    print("\n3. Muon+AdamW hybrid: ownership and memory cost (M9 §10, §12)")
    mcfg, tcfg, _ = _tiny_setup(optimizer="muon")
    model = MiniLLM(mcfg)
    opt = build_optimizer(model, tcfg)
    assigned = sum(len(g["params"]) for o in opt.optimizers for g in o.param_groups)
    check("every parameter belongs to exactly one optimizer",
          assigned == len([p for p in model.parameters() if p.requires_grad]),
          f"{assigned} assigned")
    muon = opt.optimizers[opt.names.index("muon")]
    muon_params = [p for g in muon.param_groups for p in g["params"]]
    check("Muon owns only 2D matrices and 3D expert banks",
          all(p.ndim in (2, 3) for p in muon_params), f"{len(muon_params)} parameters")
    embed = model.embed_tokens.weight
    check("embeddings remain under AdamW",
          not any(p is embed for p in muon_params))

    moe_cfg = MiniLLMConfig.profile("S-MoE")
    moe_model = MiniLLM(moe_cfg)
    partition = optimizer_partition(moe_model, TrainConfig(optimizer="muon"))
    muon_names = {name for name, _ in partition["muon"]}
    check("3D routed expert banks belong to Muon",
          any(name.endswith("gate_w") for name in muon_names) and
          any(name.endswith("up_w") for name in muon_names) and
          any(name.endswith("down_w") for name in muon_names))
    check("router and embeddings remain under AdamW",
          not any("router" in name or "embed_tokens" in name for name in muon_names))

    # State: Muon keeps one buffer per parameter; AdamW keeps two.
    ids = torch.randint(0, mcfg.vocab_size, (2, 32))
    _, loss = model(ids, labels=ids)
    loss.backward()
    opt.step()
    per_param = {}
    for name, o in zip(opt.names, opt.optimizers):
        tensors = [sum(1 for v in st.values() if torch.is_tensor(v)) for st in o.state.values()]
        per_param[name] = max(tensors) if tensors else 0
    check("Muon stores one state tensor per parameter", per_param["muon"] == 1,
          f"muon={per_param['muon']}, adamw={per_param['adamw']}")


def test_memory_report() -> None:
    print("\n4. Persistent-memory accounting follows actual optimizer ownership (M9 §12)")
    dense = MiniLLMConfig.profile("L")
    adam = memory_report(dense, optimizer="adamw")
    muon = memory_report(dense, optimizer="muon")
    check("AdamW: FP32 weights+grads+two moments = 16 bytes/parameter",
          abs(adam["bytes_per_parameter_average"] - 16.0) < 1e-9, str(adam))
    check("Muon hybrid saves only on parameters it actually owns",
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



# ---------------------------------------------------- 5. gradient accumulation
def test_grad_accumulation_equivalence() -> None:
    print("\n5. Gradient accumulation == one large batch (M9 §13)")
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
    check("gradients match", worst < 1e-5, f"max|Δ| = {worst:.2e}")


def test_moe_balance_uses_the_effective_batch() -> None:
    print("\n5b. MoE balancing aggregates every microbatch in the effective batch")
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
    check("bias update matches one large batch",
          torch.equal(biases["big"], biases["accumulated"]),
          f"big={biases['big'].tolist()}, accum={biases['accumulated'].tolist()}")


def test_gradient_checkpointing_equivalence() -> None:
    print("\n6. Activation recomputation does not change gradients (M9 §14)")
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
    check("gradients match", worst < 1e-5, f"max|Δ| = {worst:.2e}")


# ----------------------------------------------------------------- 7. training
def test_training_reduces_loss() -> None:
    print("\n7. Model genuinely learns under both optimizer branches")
    for optimizer in ("adamw", "muon"):
        mcfg, tcfg, stream = _tiny_setup(optimizer=optimizer, steps=40,
                                         log_every=1000, seed=0)
        torch.manual_seed(0)
        model = MiniLLM(mcfg)
        start = evaluate(model, stream, tcfg, batches=4)
        state = train(model, stream, tcfg)
        end = evaluate(model, stream, tcfg, batches=4)
        uniform = math.log(mcfg.vocab_size)
        check(f"{optimizer}: loss fell well below ln(V)", end < 0.5 * uniform,
              f"{start:.3f} → {end:.3f} at ln(V)={uniform:.3f}")
        check(f"{optimizer}: history is recorded", len(state.history) >= 1)


def test_moe_training_balances() -> None:
    print("\n8. During MoE training, expert bias actually moves (M7 §8)")
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
    check("layer is genuinely MoE", isinstance(moe, MoELayer))
    check("bias moved", moe.expert_bias.abs().max().item() > 0,
          f"max|b| = {moe.expert_bias.abs().max().item():.4f}")
    check("selected load preserves exactly k assignments per token",
          abs(moe.selected_load.sum().item() - mcfg.num_experts_per_tok) < 1e-6,
          f"sum={moe.selected_load.sum().item():.3f}, k={mcfg.num_experts_per_tok}")
    check("without a capacity limit, selected and kept loads match",
          torch.allclose(moe.selected_load, moe.expert_load),
          f"selected={moe.selected_load.tolist()}, kept={moe.expert_load.tolist()}")
    check("history contains the router CV diagnostic",
          "selected_load_cv" in state.history[-1], str(state.history[-1]))


def test_capacity_is_visible_in_log() -> None:
    print("\n8b. Capacity overflow is logged; otherwise failure is silent (M7 §12)")
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

    check("dropped-fraction field is present in both run histories",
          all("dropped_fraction" in h[-1] for h in histories.values()))
    check("without a limit, nothing is dropped", histories[None][-1]["dropped_fraction"] == 0.0)
    tight = histories[1.0][-1]["dropped_fraction"]
    check("an average-load limit drops a noticeable share", tight > 0.2,
          f"{tight:.1%} of assignments")

    # This is why failure is quiet: loss falls in both cases and reveals nothing.
    loose_loss = histories[None][-1]["loss"]
    tight_loss = histories[1.0][-1]["loss"]
    check("loss with dropping still falls and does not look catastrophic",
          tight_loss < math.log(tok.vocab_size),
          f"{tight_loss:.4f} vs {loose_loss:.4f} without a limit at "
          f"ln(V)={math.log(tok.vocab_size):.3f}")


# ------------------------------------------------------------- 9. checkpoints
def test_resume_is_exact() -> None:
    print("\n9. Resume after interruption gives the same result (M9 §16)")
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
        check("step is restored", state2.step == 8, f"step={state2.step}")
        train(model2, stream, tcfg, state=state2)

    worst = max((a - b).abs().max().item()
                for a, b in zip(reference.state_dict().values(),
                                model2.state_dict().values()))
    check("weights match the uninterrupted run bit-for-bit", worst == 0.0,
          f"max|Δ| = {worst:.3e}")
    check("training config survived round trip", cfg2.seed == tcfg.seed)
    check("requested device overrides stale checkpoint device", cfg2.device == "cpu")


def test_inference_artifact_has_no_stale_optimizer() -> None:
    print("\n10. Inference artifact stores weights and JSON provenance, not a stale optimizer")
    mcfg, _, _ = _tiny_setup()
    torch.manual_seed(11)
    model = MiniLLM(mcfg)
    with tempfile.TemporaryDirectory() as tmp:
        save_model_artifact(tmp, model, {"stage": "dpo", "step": 17})
        restored, metadata = load_model_artifact(tmp)
        files = {p.name for p in Path(tmp).iterdir()}
    worst = max((a - b).abs().max().item()
                for a, b in zip(model.state_dict().values(), restored.state_dict().values()))
    check("inference-artifact weights match", worst == 0.0, f"max|Δ|={worst:.2e}")
    check("provenance survived JSON round trip", metadata == {"stage": "dpo", "step": 17})
    check("optimizer state is deliberately absent",
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
    print("\n11. Masked targets: evaluation and actual train() preserve the token mean")
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
    # Independent oracle: sum CE over five targets, not a mean of model losses.
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
        check("evaluation matches the summed-token oracle",
              abs(measured - expected_loss) < 1e-6, f"batch={micro}, loss={measured:.7f}")
        state = train(current, stream, cfg)
        losses.append(state.history[-1]["loss"])
        gradients.append([p.grad.clone() for p in current.parameters()])
        check("five supervised positions are counted after shifting",
              state.supervised_tokens == 5)
        check("only input positions actually forwarded are counted",
              state.tokens == (15 if micro == 3 else 10))
    worst = max((a - b).abs().max().item()
                for result in gradients for a, b in zip(result, expected_grads))
    check("train gradients match across unequal masks and accumulation",
          worst < 2e-6, f"max|delta|={worst:.2e}")
    check("logged loss uses the same token mean",
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
        check("all-masked work raises an explicit error", rejected, operation.__name__)
        check("model mode is preserved on failed evaluation", current.training)


def test_variable_length_counters() -> None:
    print("\n12. Actual input and target counters survive checkpoint/resume")
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
        check("cumulative counts include both variable-length batches",
              resumed.tokens == 22 and resumed.supervised_tokens == 14)
        last = resumed.history[-1]
        check("run counters exclude work before resume",
              last["run_tokens"] == 14 and last["run_supervised_tokens"] == 10)
        check("throughput names its input-position basis",
              last["token_basis"] == "input_positions")
        check("MFU uses measured input throughput",
              abs(last["mfu"] - 6 * mcfg.active_parameters() * last["tokens_per_s"] / 1e12) < 1e-12)
        blob = torch.load(Path(tmp) / "state.pt", weights_only=False)
        del blob["tokens"], blob["supervised_tokens"]
        torch.save(blob, Path(tmp) / "state.pt")
        old_model, _, old_cfg, old_state = load_checkpoint(tmp)
        train(old_model, VariableStream(), old_cfg, state=old_state)
        check("legacy checkpoint totals stay unknown; current run is measured",
              old_state.tokens is None and old_state.supervised_tokens is None
              and old_state.history[-1]["run_tokens"] == 14)


def main() -> int:
    print("MiniLLM training-loop checks")
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
    print(f"\nResult: {len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print("  failed:", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
