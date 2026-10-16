"""MiniLLM training: optimizers, schedules, memory, and reproducibility.

This implements exactly the mechanisms discussed in module 9:

  Muon (Newton–Schulz orthogonalization) vs AdamW ........ M9 §9–10
  hybrid “Muon for 2D, AdamW for the rest” ............... M9 §10
  WSD schedule (warmup–stable–decay) ..................... M9 §11
  gradient accumulation and effective batch .............. M9 §13
  activation checkpointing .............................. M9 §14
  optimizer-state accounting in bytes .................... M9 §12
  MFU measurement ........................................ M9 §15

MFU here means model FLOPs utilization: 6·N_active·tokens/s divided by the
accelerator's peak throughput. The factor 6 is 2 (forward) + 4 (backward).
Activation recomputation under gradient checkpointing adds roughly 2·N more;
it is tracked separately so overhead is not presented as useful work.

Loss is averaged over supervised next-token targets across the effective
batch. tokens/s counts actual input positions passed to the model, including
PAD and the first context position. supervised_tokens separately counts targets
after shifting. Fully masked microbatches are skipped.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from .config import MiniLLMConfig
from .data import TokenStream
from .model import MiniLLM, MoELayer
from .precision import (autocast_context, make_grad_scaler, model_parameter_dtype,
                        resolve_compute_dtype)


# =============================================================================
# Muon (M9 §9)
# =============================================================================
@torch.no_grad()
def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate matrix orthogonalization via Newton–Schulz iteration.

    Return something close to U·Vᵀ from an SVD without computing the SVD. In
    module 9 terms, the update receives approximately equal singular values: a
    step that gives equal weight to every direction rather than AdamW's
    coordinate-wise scaling.

    Coefficients (3.4445, −4.7750, 2.0315) come from Keller Jordan's
    github.com/KellerJordan/Muon implementation. The iteration intentionally
    does not converge to exact orthogonalization; it is tuned to be fast in
    five steps.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def zeropower_parameter(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Apply Muon's matrix transform to 2D weights or each matrix in a 3D expert bank."""
    if G.ndim == 2:
        return zeropower_via_newtonschulz(G, steps)
    if G.ndim == 3:
        return torch.stack([zeropower_via_newtonschulz(g, steps) for g in G], dim=0)
    raise ValueError(f"Muon supports 2D matrices and 3D expert banks, got ndim={G.ndim}")


class Muon(torch.optim.Optimizer):
    """Muon for 2D matrices and 3D MoE expert banks.

    A 3D parameter ``[experts, in, out]`` is treated as a batch of independent
    matrices. Embeddings, norms, biases and router matrices remain under AdamW.
    State consists of one momentum tensor per parameter.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim not in (2, 3):
                    raise ValueError("Muon принимает 2D-матрицы и 3D expert banks")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = zeropower_parameter(g, group["ns_steps"])
                rows, cols = p.shape[-2], p.shape[-1]
                scale = max(1.0, rows / cols) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


class OptimizerGroup:
    """Expose multiple optimizers as one object for the Muon+AdamW hybrid."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], names: list[str]):
        self.optimizers = optimizers
        self.names = names

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def set_lr(self, scales: dict[str, float]) -> None:
        for name, opt in zip(self.names, self.optimizers):
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scales[name]

    def state_dict(self) -> dict:
        return {n: o.state_dict() for n, o in zip(self.names, self.optimizers)}

    def load_state_dict(self, state: dict) -> None:
        for name, opt in zip(self.names, self.optimizers):
            opt.load_state_dict(state[name])

    def state_bytes(self) -> int:
        """Optimizer-state bytes, matching the accounting in M9 §12."""
        total = 0
        for opt in self.optimizers:
            for st in opt.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        total += v.numel() * v.element_size()
        return total


# =============================================================================
# Training configuration
# =============================================================================
@dataclass
class TrainConfig:
    steps: int = 200
    seq_len: int = 128
    micro_batch_size: int = 8
    grad_accum: int = 1

    optimizer: str = "adamw"          # "adamw" | "muon"
    lr: float = 3e-3                  # AdamW branch
    muon_lr: float = 0.02             # 2D parameters owned by Muon
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_weight_decay: float | None = None  # None -> same contract as weight_decay
    grad_clip: float = 1.0

    # WSD: fractions of the total step count (M9 §11)
    warmup_frac: float = 0.05
    decay_frac: float = 0.20
    min_lr_ratio: float = 0.1

    compute_dtype: str = "auto"       # auto | fp32 | bf16 | fp16
    gradient_checkpointing: bool = False
    seed: int = 0

    log_every: int = 25
    ckpt_every: int = 0               # 0 disables intermediate checkpoints
    out_dir: str = "runs/minillm"
    device: str = "cpu"
    peak_tflops: float | None = None  # for MFU; None disables the metric

    @property
    def tokens_per_step(self) -> int:
        """Planned next-token targets for a fixed-length stream, not a measured count."""
        return self.micro_batch_size * self.grad_accum * self.seq_len


def wsd_lr_scale(step: int, cfg: TrainConfig) -> float:
    """Warmup--stable--decay with an explicit final endpoint.

    Steps are indexed ``0 .. cfg.steps-1``. If a decay phase exists, its first
    step starts at scale 1 and the final training step reaches ``min_lr_ratio``.
    A one-step decay therefore consists solely of the endpoint.
    """
    if cfg.steps <= 0 or not 0 <= step < cfg.steps:
        raise ValueError(f"step={step} outside [0,{cfg.steps})")
    warmup = min(cfg.steps, max(1, int(cfg.warmup_frac * cfg.steps)))
    decay = max(0, int(cfg.decay_frac * cfg.steps))
    stable_end = cfg.steps - decay
    if step < warmup:
        return (step + 1) / warmup
    if decay == 0 or step < stable_end:
        return 1.0
    if decay == 1:
        return cfg.min_lr_ratio
    progress = (step - stable_end) / (decay - 1)
    return 1.0 - (1.0 - cfg.min_lr_ratio) * progress


def optimizer_partition(model: MiniLLM, cfg: TrainConfig) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Return named parameter ownership for the actual optimizer contract."""
    groups = {"muon": [], "adamw_decay": [], "adamw_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_transform = p.ndim in (2, 3)
        is_embedding = "embed_tokens" in name or "lm_head" in name
        is_router = "router" in name
        if cfg.optimizer == "muon" and is_transform and not is_embedding and not is_router:
            groups["muon"].append((name, p))
        elif p.ndim >= 2:
            groups["adamw_decay"].append((name, p))
        else:
            groups["adamw_nodecay"].append((name, p))
    return groups


def build_optimizer(model: MiniLLM, cfg: TrainConfig) -> OptimizerGroup:
    """Build AdamW or the real Muon+AdamW hybrid used by this capstone."""
    partition = optimizer_partition(model, cfg)
    decay_params = [p for _, p in partition["adamw_decay"]]
    nodecay_params = [p for _, p in partition["adamw_nodecay"]]
    muon_params = [p for _, p in partition["muon"]]

    optimizers, names = [], []
    adamw = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": cfg.weight_decay, "base_lr": cfg.lr},
         {"params": nodecay_params, "weight_decay": 0.0, "base_lr": cfg.lr}],
        lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    optimizers.append(adamw)
    names.append("adamw")
    if muon_params:
        muon_decay = cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay
        muon = Muon([{"params": muon_params, "base_lr": cfg.muon_lr,
                      "weight_decay": muon_decay}],
                    lr=cfg.muon_lr, weight_decay=muon_decay)
        optimizers.append(muon)
        names.append("muon")
    return OptimizerGroup(optimizers, names)


# =============================================================================
# Training loop
# =============================================================================
@dataclass
class TrainState:
    step: int = 0
    history: list[dict] = field(default_factory=list)
    # Actual input positions (including PAD) and supervised targets after shifting.
    # None marks an unknown lifetime total from an older checkpoint.
    tokens: int | None = 0
    supervised_tokens: int | None = 0
    # Keep the optimizer in state rather than recreating it on every train()
    # call. Otherwise resume silently loses moments and continues from a cold
    # optimizer state—something the logs do not reveal.
    optimizer: "OptimizerGroup | None" = field(default=None, repr=False, compare=False)
    grad_scaler_state: dict | None = field(default=None, repr=False, compare=False)


def _unpack_batch(batch):
    if not isinstance(batch, tuple):
        return batch, batch, None
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch
    raise ValueError(f"unsupported batch tuple length: {len(batch)}")


def _target_count(labels: torch.Tensor) -> int:
    """Count labels after the same causal shift used by the model."""
    return int((labels[:, 1:] != -100).sum().item())


@torch.no_grad()
def evaluate(model: MiniLLM, stream: TokenStream, cfg: TrainConfig,
             batches: int = 8) -> float:
    """Token-mean loss over all supervised targets in the selected batches."""
    if batches <= 0:
        raise ValueError("batches must be positive")
    was_training = model.training
    model.eval()
    loss_sum, target_count = 0.0, 0
    loader = stream.loader(cfg.micro_batch_size, seed=1234, shuffle=False)
    compute_dtype = resolve_compute_dtype(cfg.device, cfg.compute_dtype)
    try:
        for _ in range(batches):
            ids, labels, segments = _unpack_batch(next(loader))
            count = _target_count(labels)
            if count == 0:
                continue
            ids, labels = ids.to(cfg.device), labels.to(cfg.device)
            segments = None if segments is None else segments.to(cfg.device)
            with autocast_context(cfg.device, compute_dtype):
                _, loss = model(ids, labels=labels, segment_ids=segments)
            loss_sum += loss.item() * count
            target_count += count
        if target_count == 0:
            raise ValueError("evaluation contains no supervised next-token targets")
        return loss_sum / target_count
    finally:
        model.train(was_training)


def train(model: MiniLLM, stream: TokenStream, cfg: TrainConfig,
          eval_stream: TokenStream | None = None,
          state: TrainState | None = None,
          stop_at_step: int | None = None) -> TrainState:
    """Main loop; return the state whose history the notebook plots.

    ``cfg.steps`` always means the full planned horizon and therefore defines
    the WSD schedule. ``stop_at_step`` simulates an interrupted session without
    changing that horizon; this is the contract needed for exact resume.
    """
    if cfg.grad_accum <= 0 or cfg.micro_batch_size <= 0:
        raise ValueError("grad_accum and micro_batch_size must be positive")
    state = state or TrainState()
    if state.step == 0:
        torch.manual_seed(cfg.seed)   # on resume, RNG state comes from the checkpoint
    model.to(cfg.device)
    model.gradient_checkpointing = cfg.gradient_checkpointing
    model.train()

    optimizer = state.optimizer or build_optimizer(model, cfg)
    state.optimizer = optimizer
    compute_dtype = resolve_compute_dtype(cfg.device, cfg.compute_dtype)
    scaler = make_grad_scaler(cfg.device, compute_dtype)
    if state.grad_scaler_state:
        scaler.load_state_dict(state.grad_scaler_state)
    # Derive the data cursor from the step number: resume continues at the same
    # position instead of replaying earlier batches. This is essential for a
    # sharded profile-L stream—a terabyte cannot be reread on every resume.
    loader = stream.loader(cfg.micro_batch_size, seed=cfg.seed,
                           start_batch=state.step * cfg.grad_accum)
    out_dir = Path(cfg.out_dir)
    active = model.cfg.active_parameters()
    run_tokens = run_supervised_tokens = 0
    t0 = time.time()
    moe_layers = [layer.ffn for layer in model.layers if isinstance(layer.ffn, MoELayer)]

    end_step = cfg.steps if stop_at_step is None else min(cfg.steps, stop_at_step)
    if end_step < state.step:
        raise ValueError(f"stop_at_step={end_step} precedes resumed step={state.step}")
    for step in range(state.step, end_step):
        optimizer.set_lr({n: wsd_lr_scale(step, cfg) for n in optimizer.names})
        optimizer.zero_grad()

        # One denominator for the effective batch. Buffer only input tensors;
        # activation graphs are still built and released one microbatch at a time.
        microbatches = [_unpack_batch(next(loader)) for _ in range(cfg.grad_accum)]
        target_counts = [_target_count(labels) for _, labels, _ in microbatches]
        step_targets = sum(target_counts)
        if step_targets == 0:
            raise ValueError("effective batch contains no supervised next-token targets")
        total_loss = 0.0
        step_tokens = 0
        selected_sums = [torch.zeros_like(moe.selected_load) for moe in moe_layers]
        kept_sums = [torch.zeros_like(moe.expert_load) for moe in moe_layers]
        dropped_sums = [torch.zeros_like(moe.dropped_load) for moe in moe_layers]
        drop_maxima = [torch.zeros_like(moe.dropped_fraction) for moe in moe_layers]
        for (ids, labels, segments), count in zip(microbatches, target_counts):
            if count == 0:
                continue
            input_count = ids.numel()
            step_tokens += input_count
            ids, labels = ids.to(cfg.device), labels.to(cfg.device)
            segments = None if segments is None else segments.to(cfg.device)
            with autocast_context(cfg.device, compute_dtype):
                _, loss = model(ids, labels=labels, segment_ids=segments)
            scaled_loss = loss * (count / step_targets)
            scaler.scale(scaled_loss).backward()
            total_loss += loss.item() * (count / step_targets)
            for i, moe in enumerate(moe_layers):
                selected_sums[i].add_(moe.selected_load, alpha=input_count)
                kept_sums[i].add_(moe.expert_load, alpha=input_count)
                dropped_sums[i].add_(moe.dropped_load, alpha=input_count)
                drop_maxima[i] = torch.maximum(drop_maxima[i], moe.dropped_fraction)

        for moe, selected, kept, dropped, drop_max in zip(
                moe_layers, selected_sums, kept_sums, dropped_sums, drop_maxima):
            moe.selected_load = selected / step_tokens
            moe.expert_load = kept / step_tokens
            moe.dropped_load = dropped / step_tokens
            moe.dropped_fraction = drop_max

        for opt in optimizer.optimizers:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizer.optimizers:
            scaler.step(opt)
        scaler.update()
        state.grad_scaler_state = scaler.state_dict()
        # Expert balancing runs after the optimizer step and outside the graph (M7 §8).
        model.update_expert_biases()

        state.step = step + 1
        run_tokens += step_tokens
        run_supervised_tokens += step_targets
        if state.tokens is not None:
            state.tokens += step_tokens
        if state.supervised_tokens is not None:
            state.supervised_tokens += step_targets
        if state.step % cfg.log_every == 0 or state.step == cfg.steps:
            elapsed = time.time() - t0
            record = {
                "step": state.step,
                "loss": total_loss,
                "grad_norm": float(grad_norm),
                "lr_scale": wsd_lr_scale(step, cfg),
                "tokens": state.tokens,
                "supervised_tokens": state.supervised_tokens,
                "run_supervised_tokens": run_supervised_tokens,
                "token_basis": "input_positions",
                "run_tokens": run_tokens,
                "tokens_per_s": run_tokens / max(elapsed, 1e-9),
                "optimizer_state_bytes": optimizer.state_bytes(),
                "parameter_dtype": model_parameter_dtype(model),
                "compute_dtype": compute_dtype,
            }
            # Fraction of capacity-dropped assignments (M7 §12). Without this
            # line, buffer overflow is invisible in both loss and validation.
            dropped = [layer.ffn.dropped_fraction.item() for layer in model.layers
                       if isinstance(layer.ffn, MoELayer)]
            if dropped:
                record["dropped_fraction"] = round(max(dropped), 5)
                record["selected_load_cv"] = round(max(
                    float(layer.ffn.selected_load.std() /
                          layer.ffn.selected_load.mean().clamp(min=1e-9))
                    for layer in model.layers if isinstance(layer.ffn, MoELayer)), 5)
            if cfg.peak_tflops:
                flops_per_s = 6 * active * record["tokens_per_s"]
                record["mfu"] = flops_per_s / (cfg.peak_tflops * 1e12)
                if cfg.gradient_checkpointing:
                    # Activation recomputation is overhead, not useful work.
                    record["mfu_with_recompute"] = (8 * active * record["tokens_per_s"]
                                                    / (cfg.peak_tflops * 1e12))
            if eval_stream is not None:
                record["eval_loss"] = evaluate(model, eval_stream, cfg)
            state.history.append(record)
            print(f"  step {record['step']:>5} | loss {record['loss']:.4f}"
                  + (f" | eval {record['eval_loss']:.4f}" if eval_stream else "")
                  + f" | |g| {record['grad_norm']:.3f}"
                  + (f" | drop {record['dropped_fraction']:.1%}"
                     if "dropped_fraction" in record else "")
                  + f" | {record['tokens_per_s']:.0f} tok/s")

        if cfg.ckpt_every and state.step % cfg.ckpt_every == 0:
            save_checkpoint(out_dir / f"step_{state.step}", model, optimizer, cfg, state)

    return state


# =============================================================================
# Checkpoints (M9 §16): an overnight run must survive session interruption
# =============================================================================
def save_checkpoint(path: str | Path, model: MiniLLM, optimizer: OptimizerGroup,
                    cfg: TrainConfig, state: TrainState) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.cfg.save_pretrained(path)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": state.step,
        "history": state.history,
        "tokens": state.tokens,
        "supervised_tokens": state.supervised_tokens,
        "train_config": asdict(cfg),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "grad_scaler_state": state.grad_scaler_state,
    }, path / "state.pt")
    return path


def load_checkpoint(path: str | Path, device: str = "cpu"
                    ) -> tuple[MiniLLM, OptimizerGroup, TrainConfig, TrainState]:
    path = Path(path)
    blob = torch.load(path / "state.pt", map_location=device, weights_only=False)
    cfg = TrainConfig(**blob["train_config"])
    cfg.device = device
    model = MiniLLM(MiniLLMConfig.from_pretrained(path)).to(device)
    model.load_state_dict(blob["model"])
    optimizer = build_optimizer(model, cfg)
    optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob.get("torch_rng_state", blob.get("rng_state")))
    if torch.cuda.is_available() and blob.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(blob["cuda_rng_states"])
    if blob.get("python_rng_state") is not None:
        random.setstate(blob["python_rng_state"])
    if blob.get("numpy_rng_state") is not None:
        np.random.set_state(blob["numpy_rng_state"])
    state = TrainState(step=blob["step"], history=blob["history"], optimizer=optimizer,
                       tokens=blob.get("tokens"), supervised_tokens=blob.get("supervised_tokens"),
                       grad_scaler_state=blob.get("grad_scaler_state"))
    return model, optimizer, cfg, state


def save_model_artifact(path: str | Path, model: MiniLLM,
                        metadata: dict[str, Any] | None = None) -> Path:
    """Save inference weights without a misleading optimizer state.

    After DPO, weights no longer correspond to the SFT optimizer. This artifact
    is deliberately separate from resumable `save_checkpoint`: `model.pt` can
    be loaded safely with `weights_only=True`, while JSON records stage provenance.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.cfg.save_pretrained(path)
    torch.save({"model": model.state_dict()}, path / "model.pt")
    (path / "metadata.json").write_text(
        json.dumps(metadata or {}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return path


def load_model_artifact(path: str | Path, device: str = "cpu"
                        ) -> tuple[MiniLLM, dict[str, Any]]:
    """Load an inference artifact and its non-pickled metadata."""
    path = Path(path)
    blob = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model = MiniLLM(MiniLLMConfig.from_pretrained(path)).to(device)
    model.load_state_dict(blob["model"])
    metadata_path = path / "metadata.json"
    metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists() else {})
    return model, metadata


# =============================================================================
# Memory accounting (M9 §12)—before launch, not after OOM
# =============================================================================
def memory_report(model_cfg: MiniLLMConfig, optimizer: str = "adamw",
                  train_cfg: TrainConfig | None = None) -> dict[str, float | dict]:
    """Estimate persistent training memory from actual optimizer ownership.

    MiniLLM parameters and gradients are FP32 in the reference runtime. AdamW
    stores two FP32 moments; Muon stores one FP32 momentum tensor. Activations and
    temporary workspaces are intentionally reported separately by the notebooks.
    """
    cfg = train_cfg or TrainConfig(optimizer=optimizer)
    cfg.optimizer = optimizer
    with torch.device("meta"):
        model = MiniLLM(model_cfg)
    partition = optimizer_partition(model, cfg)
    counts = {name: sum(p.numel() for _, p in params) for name, params in partition.items()}
    total = sum(counts.values())
    # FP32 parameter + FP32 gradient for all parameters.
    base = total * 8
    adamw_count = counts["adamw_decay"] + counts["adamw_nodecay"]
    state = adamw_count * 8 + counts["muon"] * 4
    payload = base + state
    ownership = {
        "muon_parameters": counts["muon"],
        "adamw_parameters": adamw_count,
        "muon_fraction": counts["muon"] / max(total, 1),
        "adamw_fraction": adamw_count / max(total, 1),
    }
    return {
        "parameters": total,
        "parameter_dtype": "fp32",
        "bytes_per_parameter_average": payload / max(total, 1),
        "payload_gib": round(payload / 1024 ** 3, 2),
        "payload_gb": round(payload / 1000 ** 3, 2),
        "ownership": ownership,
        "weights_and_gradients_gib": round(base / 1024 ** 3, 2),
        "optimizer_state_gib": round(state / 1024 ** 3, 2),
    }


# =============================================================================
# Demonstration: profile S on the toy corpus
# =============================================================================
def demo(steps: int = 300, optimizer: str = "adamw") -> TrainState:
    from .data import ByteTokenizer, toy_corpus

    tok = ByteTokenizer()
    cfg = MiniLLMConfig.profile("S", vocab_size=tok.vocab_size,
                                max_position_embeddings=256)
    tcfg = TrainConfig(steps=steps, seq_len=128, micro_batch_size=8,
                       optimizer=optimizer, log_every=max(1, steps // 6))
    stream = TokenStream.from_text(toy_corpus(), tok, tcfg.seq_len)
    model = MiniLLM(cfg)
    print(f"MiniLLM-S: {model.num_parameters():,} parameters; "
          f"corpus {stream.tokens.numel():,} tokens, sha {stream.checksum()}")
    print(f"start: loss should fall well below ln(V) = "
          f"{math.log(cfg.vocab_size):.3f}")
    return train(model, stream, tcfg, eval_stream=stream)


if __name__ == "__main__":
    demo()
