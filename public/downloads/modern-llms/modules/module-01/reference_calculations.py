#!/usr/bin/env python3
"""Reviewed numerical core and frozen reference data for Module 1.

The learner notebooks use a module-owned snapshot instead of mutable lecture prose or an
implicit live Hub fallback. Optional Hub retrieval supports both an exact commit and a deliberately
live branch/tag lookup that is immediately resolved to a full commit before ``config.json`` is
fetched. Both paths run the same schema validation as the snapshot and never fall back silently.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

MODULE_ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = MODULE_ROOT / "data" / "reference_metrics.json"


# BEGIN EDUCATIONAL CORE: validation_and_json

class ConfigSchemaError(ValueError):
    """Raised when a model configuration violates the supported teaching schema."""


class ConfigIntegrityError(ValueError):
    """Raised when a config digest differs from the explicitly expected digest."""


def _positive_int(name: str, value: object) -> int:
    """Return a strictly positive integer and reject booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigSchemaError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ConfigSchemaError(f"{name} must be positive")
    return result


def _non_negative_int(name: str, value: object) -> int:
    """Return a non-negative integer and reject booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigSchemaError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ConfigSchemaError(f"{name} must be non-negative")
    return result


def _finite_positive_real(name: str, value: object) -> float:
    """Return a positive finite real number."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigSchemaError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigSchemaError(f"{name} must be finite and positive")
    return result


def _string(name: str, value: object) -> str:
    """Return a non-empty string stripped of surrounding whitespace."""

    if not isinstance(value, str):
        raise ConfigSchemaError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ConfigSchemaError(f"{name} must be non-empty")
    return result


def _mapping(name: str, value: object) -> dict[str, Any]:
    """Return a shallow dictionary copy of a string-keyed mapping."""

    if not isinstance(value, Mapping):
        raise ConfigSchemaError(f"{name} must be an object")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ConfigSchemaError(f"{name} keys must be strings")
    return result


def _synonym_value(
    cfg: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: object = None,
) -> object:
    """Return one synonym value and reject conflicting aliases."""

    present = [(key, cfg[key]) for key in keys if key in cfg and cfg[key] is not None]
    if not present:
        return default
    first_key, first_value = present[0]
    conflicts = [key for key, value in present[1:] if value != first_value]
    if conflicts:
        names = ", ".join([first_key, *conflicts])
        raise ConfigSchemaError(f"conflicting synonym fields: {names}")
    return first_value


def strict_json_loads(text: object) -> dict[str, Any]:
    """Parse a JSON object while rejecting duplicate keys and non-finite constants."""

    if not isinstance(text, str) or not text.strip():
        raise ConfigSchemaError("config JSON must be a non-empty string")

    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Build one object while rejecting a repeated key at its own nesting level."""

        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigSchemaError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def parse_constant(value: str) -> object:
        """Reject JavaScript-style non-finite constants accepted by Python's parser."""

        raise ConfigSchemaError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs_hook,
            parse_constant=parse_constant,
        )
    except ConfigSchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise ConfigSchemaError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ConfigSchemaError("config JSON must contain a top-level object")
    return payload


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible data in one deterministic UTF-8 representation."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    """Return SHA-256 of canonical JSON bytes."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# END EDUCATIONAL CORE: validation_and_json


# BEGIN EDUCATIONAL CORE: config_schema

EXPERT_KEYS = ("num_local_experts", "n_routed_experts", "num_experts")
TOP_K_KEYS = ("num_experts_per_tok", "num_experts_per_token")
SHARED_EXPERT_KEYS = ("n_shared_experts", "num_shared_experts")
ROPE_TYPE_KEYS = ("rope_type", "type")
MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


def validate_repo_id(repo_id: object) -> str:
    """Validate a two-component Hugging Face repository identifier."""

    value = _string("repo_id", repo_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ConfigSchemaError("repo_id must have the form 'namespace/name'")
    return value


def validate_commit_revision(revision: object) -> str:
    """Require an immutable full 40-hex Git commit revision."""

    value = _string("revision", revision)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ConfigSchemaError("revision must be an exact 40-hex commit, not 'main' or a tag")
    return value.lower()


def validate_source_revision(revision: object) -> str:
    """Validate a branch, tag, PR ref, or full commit used for an exploratory Hub lookup.

    This intentionally accepts only the common Hub forms used in the lesson while rejecting
    whitespace, control characters, repeated path separators, and ambiguous Git-ref constructs.
    """

    value = _string("revision", revision)
    if len(value) > 200 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value):
        raise ConfigSchemaError("source revision contains unsupported characters")
    if ".." in value or "//" in value or value.endswith(".lock"):
        raise ConfigSchemaError("source revision contains an unsafe or ambiguous construct")
    return value


def expert_count(cfg: Mapping[str, Any]) -> int | None:
    """Return the routed-expert count after validating synonym consistency."""

    value = _synonym_value(cfg, EXPERT_KEYS)
    return None if value is None else _positive_int("expert count", value)


def experts_per_token(cfg: Mapping[str, Any]) -> int | None:
    """Return top-k routed experts after validating synonym consistency."""

    value = _synonym_value(cfg, TOP_K_KEYS)
    return None if value is None else _positive_int("experts per token", value)


def shared_expert_count(cfg: Mapping[str, Any]) -> int:
    """Return the non-negative number of always-on shared experts."""

    value = _synonym_value(cfg, SHARED_EXPERT_KEYS, default=0)
    return _non_negative_int("shared expert count", value)


def resolved_head_dim(cfg: Mapping[str, Any]) -> int:
    """Return explicit head_dim or infer it only when hidden/heads divide exactly."""

    if cfg.get("head_dim") is not None:
        return _positive_int("head_dim", cfg["head_dim"])
    hidden = _positive_int("hidden_size", cfg.get("hidden_size"))
    heads = _positive_int("num_attention_heads", cfg.get("num_attention_heads"))
    if hidden % heads:
        raise ConfigSchemaError("head_dim is missing and hidden_size is not divisible by heads")
    return hidden // heads


def validate_config(config: object) -> dict[str, Any]:
    """Validate fields used by the Module 1 passport and return a defensive copy.

    Unknown fields are retained because real configs evolve. The validator is strict only for
    fields consumed by the teaching formulas; unsupported architectures should fail loudly
    instead of being silently forced into the nearest familiar family.
    """

    cfg = _mapping("config", config)
    hidden = _positive_int("hidden_size", cfg.get("hidden_size"))
    layers = _positive_int("num_hidden_layers", cfg.get("num_hidden_layers"))
    heads = _positive_int("num_attention_heads", cfg.get("num_attention_heads"))
    _positive_int("vocab_size", cfg.get("vocab_size"))
    _positive_int("max_position_embeddings", cfg.get("max_position_embeddings"))
    resolved_head_dim(cfg)

    kv_heads = cfg.get("num_key_value_heads", heads)
    kv_heads = _positive_int("num_key_value_heads", kv_heads)
    if kv_heads > heads or heads % kv_heads:
        raise ConfigSchemaError("num_key_value_heads must divide num_attention_heads")

    if "tie_word_embeddings" in cfg and not isinstance(cfg["tie_word_embeddings"], bool):
        raise ConfigSchemaError("tie_word_embeddings must be boolean")

    architectures = cfg.get("architectures")
    if architectures is not None:
        if not isinstance(architectures, list) or not architectures:
            raise ConfigSchemaError("architectures must be a non-empty list")
        for index, architecture in enumerate(architectures):
            _string(f"architectures[{index}]", architecture)

    rope_scaling = cfg.get("rope_scaling")
    if rope_scaling is not None:
        rs = _mapping("rope_scaling", rope_scaling)
        rope_type = _synonym_value(rs, ROPE_TYPE_KEYS)
        if rope_type is not None:
            _string("rope_scaling type", rope_type)
        if rs.get("factor") is not None:
            _finite_positive_real("rope_scaling.factor", rs["factor"])
        if rs.get("original_max_position_embeddings") is not None:
            _positive_int(
                "rope_scaling.original_max_position_embeddings",
                rs["original_max_position_embeddings"],
            )

    experts = expert_count(cfg)
    top_k = experts_per_token(cfg)
    shared = shared_expert_count(cfg)
    if experts is None:
        if top_k is not None or shared:
            raise ConfigSchemaError("top-k/shared experts require a routed-expert count")
        _positive_int("intermediate_size", cfg.get("intermediate_size"))
    else:
        if top_k is None:
            raise ConfigSchemaError("MoE config requires num_experts_per_tok")
        if top_k > experts:
            raise ConfigSchemaError("experts per token cannot exceed routed experts")
        inter_value = cfg.get("moe_intermediate_size", cfg.get("intermediate_size"))
        _positive_int("MoE intermediate size", inter_value)
        dense_prefix = _non_negative_int(
            "first_k_dense_replace",
            cfg.get("first_k_dense_replace", 0),
        )
        if dense_prefix > layers:
            raise ConfigSchemaError("first_k_dense_replace cannot exceed layer count")
        if dense_prefix and cfg.get("intermediate_size") is None:
            raise ConfigSchemaError("dense prefix requires intermediate_size")

    has_any_mla = any(cfg.get(field) is not None for field in MLA_FIELDS)
    if has_any_mla:
        missing = [field for field in MLA_FIELDS if cfg.get(field) is None]
        if missing:
            raise ConfigSchemaError(f"incomplete MLA signature; missing {', '.join(missing)}")
        for field in MLA_FIELDS:
            _positive_int(field, cfg[field])

    if hidden <= 0:  # retained as an explicit local invariant for readability
        raise ConfigSchemaError("hidden_size must be positive")
    return copy.deepcopy(cfg)


@dataclass(frozen=True)
class SnapshotModel:
    """One validated model fixture and the convention used for its active count."""

    repo_id: str
    display_name: str
    config: dict[str, Any]
    config_sha256: str
    claimed_total_b: float
    claimed_active_b: float
    active_convention: str
    total_tolerance_pct: float
    active_tolerance_pct: float


@dataclass(frozen=True)
class ReferenceBundle:
    """Validated Module 1 snapshot, assumptions, and context milestones."""

    snapshot_date: str
    models: tuple[SnapshotModel, ...]
    context_milestones: tuple[dict[str, Any], ...]
    cost_assumptions: dict[str, float]


def load_reference_bundle(payload: object) -> ReferenceBundle:
    """Validate the frozen module payload and return typed immutable records."""

    root = _mapping("reference payload", payload)
    if root.get("schema_version") != 1:
        raise ConfigSchemaError("unsupported reference schema_version")
    snapshot_date = _string("snapshot_date", root.get("snapshot_date"))
    raw_models = root.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigSchemaError("models must be a non-empty list")

    models: list[SnapshotModel] = []
    seen_repo_ids: set[str] = set()
    seen_display_names: set[str] = set()
    for index, raw in enumerate(raw_models):
        item = _mapping(f"models[{index}]", raw)
        repo_id = validate_repo_id(item.get("repo_id"))
        display_name = _string("display_name", item.get("display_name"))
        if repo_id in seen_repo_ids:
            raise ConfigSchemaError(f"duplicate repo_id in snapshot: {repo_id}")
        if display_name in seen_display_names:
            raise ConfigSchemaError(f"duplicate display_name in snapshot: {display_name}")
        seen_repo_ids.add(repo_id)
        seen_display_names.add(display_name)
        config = validate_config(item.get("config"))
        expected_hash = _string("config_sha256", item.get("config_sha256"))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ConfigSchemaError("config_sha256 must be lowercase 64-hex")
        if sha256_json(config) != expected_hash:
            raise ConfigSchemaError(f"config checksum mismatch for {repo_id}")
        convention = _string("active_convention", item.get("active_convention"))
        if convention not in {
            "total",
            "core_plus_unembedding",
            "core_plus_embeddings",
        }:
            raise ConfigSchemaError(f"unsupported active convention: {convention}")
        models.append(
            SnapshotModel(
                repo_id=repo_id,
                display_name=display_name,
                config=config,
                config_sha256=expected_hash,
                claimed_total_b=_finite_positive_real(
                    "claimed_total_b", item.get("claimed_total_b")
                ),
                claimed_active_b=_finite_positive_real(
                    "claimed_active_b", item.get("claimed_active_b")
                ),
                active_convention=convention,
                total_tolerance_pct=_finite_positive_real(
                    "total_tolerance_pct", item.get("total_tolerance_pct")
                ),
                active_tolerance_pct=_finite_positive_real(
                    "active_tolerance_pct", item.get("active_tolerance_pct")
                ),
            )
        )

    raw_milestones = root.get("context_milestones")
    if not isinstance(raw_milestones, list) or not raw_milestones:
        raise ConfigSchemaError("context_milestones must be a non-empty list")
    milestones: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_milestones):
        item = _mapping(f"context_milestones[{index}]", raw)
        year = _finite_positive_real("milestone year", item.get("year"))
        tokens = _positive_int("milestone tokens", item.get("tokens"))
        model = _string("milestone model", item.get("model"))
        milestones.append({"year": year, "tokens": tokens, "model": model})
    if milestones != sorted(milestones, key=lambda row: row["year"]):
        raise ConfigSchemaError("context_milestones must be sorted by year")

    assumptions_raw = _mapping("cost_assumptions", root.get("cost_assumptions"))
    assumptions = {
        "bf16_bits": _finite_positive_real("bf16_bits", assumptions_raw.get("bf16_bits")),
        "four_bit_effective_bits": _finite_positive_real(
            "four_bit_effective_bits", assumptions_raw.get("four_bit_effective_bits")
        ),
        "card_capacity_gb_decimal": _finite_positive_real(
            "card_capacity_gb_decimal", assumptions_raw.get("card_capacity_gb_decimal")
        ),
    }
    return ReferenceBundle(
        snapshot_date=snapshot_date,
        models=tuple(models),
        context_milestones=tuple(milestones),
        cost_assumptions=assumptions,
    )


# END EDUCATIONAL CORE: config_schema


# BEGIN EDUCATIONAL CORE: classification


def attention_kind(config: object) -> str:
    """Classify standard MHA/GQA/MQA or MLA after schema validation."""

    cfg = validate_config(config)
    if "kv_lora_rank" in cfg:
        return f"MLA (kv_rank={cfg['kv_lora_rank']}, q_rank={cfg['q_lora_rank']})"
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    if kv_heads == heads:
        label = "MHA"
    elif kv_heads == 1:
        label = "MQA"
    else:
        label = f"GQA {heads}/{kv_heads}"
    window = cfg.get("sliding_window")
    if window is not None:
        label += f" + sliding {_positive_int('sliding_window', window)}"
    return label


def moe_kind(config: object) -> str:
    """Return a readable dense/MoE classifier from validated synonym fields."""

    cfg = validate_config(config)
    experts = expert_count(cfg)
    if experts is None:
        return "dense"
    label = f"MoE {experts} × top-{experts_per_token(cfg)}"
    shared = shared_expert_count(cfg)
    if shared:
        label += f" + {shared} shared"
    return label


def context_info(config: object) -> str:
    """Summarize maximum positions and optional RoPE scaling."""

    cfg = validate_config(config)
    maximum = int(cfg["max_position_embeddings"])
    scaling = cfg.get("rope_scaling")
    if not scaling:
        return f"{maximum:,}"
    rope_type = _synonym_value(scaling, ROPE_TYPE_KEYS, default="unspecified")
    factor = scaling.get("factor", "?")
    original = scaling.get("original_max_position_embeddings")
    suffix = f"{rope_type} ×{factor}"
    if original is not None:
        suffix += f" from {int(original):,}"
    return f"{maximum:,} ({suffix})"


# END EDUCATIONAL CORE: classification


# BEGIN EDUCATIONAL CORE: parameter_accounting

@dataclass(frozen=True)
class ParameterBreakdown:
    """Approximate parameter accounting for one supported Transformer family."""

    total: int
    active_core: int
    active_plus_unembedding: int
    active_plus_embeddings: int
    attention: int
    ffn_total: int
    embedding: int
    unembedding: int
    norms: int
    extras: int

    def active(self, convention: str) -> int:
        """Return the active count under one explicitly declared convention."""

        if convention == "total":
            return self.total
        if convention == "core_plus_unembedding":
            return self.active_plus_unembedding
        if convention == "core_plus_embeddings":
            return self.active_plus_embeddings
        raise ConfigSchemaError(f"unsupported active convention: {convention}")


def _architecture_name(cfg: Mapping[str, Any]) -> str:
    """Return the first declared architecture name or an empty string."""

    architectures = cfg.get("architectures")
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    return ""


def attention_parameter_count(config: object) -> int:
    """Count attention parameters per layer for the supported teaching families.

    For gpt-oss the public implementation uses fused QKV and output biases. For
    DeepSeek-V3 the public configuration/implementation also exposes Q- and latent-KV
    normalization scales. Those small terms matter because this module promises an
    explicit reconstruction rather than a matrix-only headline.
    """

    cfg = validate_config(config)
    architecture = _architecture_name(cfg)
    heads = int(cfg["num_attention_heads"])
    hidden = int(cfg["hidden_size"])
    if "kv_lora_rank" in cfg:
        q_rank = int(cfg["q_lora_rank"])
        kv_rank = int(cfg["kv_lora_rank"])
        d_nope = int(cfg["qk_nope_head_dim"])
        d_rope = int(cfg["qk_rope_head_dim"])
        d_value = int(cfg["v_head_dim"])
        query = hidden * q_rank + q_rank * heads * (d_nope + d_rope)
        key_value = hidden * (kv_rank + d_rope) + kv_rank * heads * (d_nope + d_value)
        output = heads * d_value * hidden
        q_kv_norm_scales = q_rank + kv_rank if architecture == "DeepseekV3ForCausalLM" else 0
        return query + key_value + output + q_kv_norm_scales
    head_dim = resolved_head_dim(cfg)
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    query = hidden * heads * head_dim
    key_value = 2 * hidden * kv_heads * head_dim
    output = heads * head_dim * hidden
    biases = 0
    if architecture == "GptOssForCausalLM":
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        biases = q_dim + 2 * kv_dim + hidden
    # Qwen3 applies a learned RMSNorm to Q and K, each with head_dim weights.
    qk_norms = 2 * head_dim if architecture in {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"} else 0
    return query + key_value + output + biases + qk_norms


def ffn_parameter_count(config: object, *, dense_layer: bool) -> tuple[int, int]:
    """Return total and per-token-active FFN/router parameters for one layer."""

    cfg = validate_config(config)
    hidden = int(cfg["hidden_size"])
    experts = expert_count(cfg)
    if experts is None or dense_layer:
        dense = 3 * hidden * int(cfg["intermediate_size"])
        return dense, dense
    intermediate_value = cfg.get("moe_intermediate_size")
    if intermediate_value is None:
        intermediate_value = cfg.get("intermediate_size")
    intermediate = int(intermediate_value)
    architecture = _architecture_name(cfg)
    expert = 3 * hidden * intermediate
    if architecture == "GptOssForCausalLM":
        expert += 2 * intermediate + hidden
    router = hidden * experts
    if architecture in {"GptOssForCausalLM", "DeepseekV3ForCausalLM"}:
        router += experts
    shared = shared_expert_count(cfg) * expert
    total = experts * expert + shared + router
    active = int(experts_per_token(cfg)) * expert + shared + router
    return total, active


def count_parameters(config: object) -> ParameterBreakdown:
    """Count approximate total and convention-dependent active parameters."""

    cfg = validate_config(config)
    supported = {"GptOssForCausalLM", "LlamaForCausalLM", "Qwen3ForCausalLM",
                 "Qwen3MoeForCausalLM", "MixtralForCausalLM", "DeepseekV3ForCausalLM"}
    if _architecture_name(cfg) not in supported:
        raise ConfigSchemaError("parameter accounting requires a supported architecture contract")
    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    vocab = int(cfg["vocab_size"])
    embedding = vocab * hidden
    unembedding = 0 if cfg.get("tie_word_embeddings", False) else vocab * hidden
    dense_prefix = int(cfg.get("first_k_dense_replace", 0)) if expert_count(cfg) else layers
    attention = attention_parameter_count(cfg) * layers
    ffn_total = 0
    ffn_active = 0
    for layer in range(layers):
        total_layer, active_layer = ffn_parameter_count(
            cfg,
            dense_layer=layer < dense_prefix,
        )
        ffn_total += total_layer
        ffn_active += active_layer
    norms = (2 * layers + 1) * hidden
    architecture = _architecture_name(cfg)
    extras = layers * int(cfg["num_attention_heads"]) if architecture == "GptOssForCausalLM" else 0
    active_core = attention + ffn_active + norms + extras
    total = embedding + unembedding + attention + ffn_total + norms + extras
    return ParameterBreakdown(
        total=total,
        active_core=active_core,
        active_plus_unembedding=active_core + unembedding,
        active_plus_embeddings=active_core + unembedding + embedding,
        attention=attention,
        ffn_total=ffn_total,
        embedding=embedding,
        unembedding=unembedding,
        norms=norms,
        extras=extras,
    )


# END EDUCATIONAL CORE: parameter_accounting


# BEGIN EDUCATIONAL CORE: reconciliation_and_costs

@dataclass(frozen=True)
class ReconciliationRow:
    """One model's computed passport and declared-card reconciliation."""

    repo_id: str
    model: str
    attention: str
    ffn: str
    context: str
    total_params: int
    active_params: int
    claimed_total_b: float
    claimed_active_b: float
    delta_total_pct: float
    delta_active_pct: float
    active_convention: str


def reconcile_model(model: SnapshotModel) -> ReconciliationRow:
    """Compute one snapshot passport without choosing a convention post hoc."""

    breakdown = count_parameters(model.config)
    active = breakdown.active(model.active_convention)
    total_b = breakdown.total / 1e9
    active_b = active / 1e9
    delta_total = 100.0 * (total_b - model.claimed_total_b) / model.claimed_total_b
    delta_active = 100.0 * (active_b - model.claimed_active_b) / model.claimed_active_b
    if abs(delta_total) > model.total_tolerance_pct:
        raise AssertionError(
            f"{model.repo_id}: total differs by {delta_total:.3f}% "
            f"(limit {model.total_tolerance_pct:.3f}%)"
        )
    if abs(delta_active) > model.active_tolerance_pct:
        raise AssertionError(
            f"{model.repo_id}: active differs by {delta_active:.3f}% "
            f"(limit {model.active_tolerance_pct:.3f}%)"
        )
    return ReconciliationRow(
        repo_id=model.repo_id,
        model=model.display_name,
        attention=attention_kind(model.config),
        ffn=moe_kind(model.config),
        context=context_info(model.config),
        total_params=breakdown.total,
        active_params=active,
        claimed_total_b=model.claimed_total_b,
        claimed_active_b=model.claimed_active_b,
        delta_total_pct=delta_total,
        delta_active_pct=delta_active,
        active_convention=model.active_convention,
    )


@dataclass(frozen=True)
class WeightCost:
    """Weight-payload floor and dense-matmul FLOP approximation for one model."""

    payload_bytes: float
    payload_gb_decimal: float
    payload_gib_binary: float
    gflops_per_token: float
    payload_only_cards: int


def weight_cost(
    *,
    total_params: int,
    active_params: int,
    bits_per_parameter: Real,
    card_capacity_gb_decimal: Real,
) -> WeightCost:
    """Return payload-only memory and a ``2 × active`` matmul FLOP approximation."""

    total = _positive_int("total_params", total_params)
    active = _positive_int("active_params", active_params)
    bits = _finite_positive_real("bits_per_parameter", bits_per_parameter)
    card_gb = _finite_positive_real("card_capacity_gb_decimal", card_capacity_gb_decimal)
    payload_bytes = total * bits / 8.0
    return WeightCost(
        payload_bytes=payload_bytes,
        payload_gb_decimal=payload_bytes / 1e9,
        payload_gib_binary=payload_bytes / (2**30),
        gflops_per_token=(2.0 * active) / 1e9,
        payload_only_cards=math.ceil(payload_bytes / (card_gb * 1e9)),
    )


def flatten_config(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested config objects into dot-separated paths for drift reports."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_config(value[key], path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            result.update(flatten_config(item, path))
        return result
    return {prefix: value}


def config_diff(snapshot: object, live: object) -> list[dict[str, object]]:
    """Return deterministic path-level changes between two validated configs."""

    left = flatten_config(validate_config(snapshot))
    right = flatten_config(validate_config(live))
    changes: list[dict[str, object]] = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            changes.append({"path": path, "snapshot": left.get(path), "live": right.get(path)})
    return changes


# END EDUCATIONAL CORE: reconciliation_and_costs


# BEGIN EDUCATIONAL CORE: live_fetch

ConfigTransport = Callable[[str, str, str | None], str]
RevisionResolver = Callable[[str, str, str | None], str]


@dataclass(frozen=True)
class FetchedConfig:
    """A schema-validated config tied to an immutable repository revision."""

    repo_id: str
    revision: str
    config: dict[str, Any]
    config_sha256: str


@dataclass(frozen=True)
class SourceFetchedConfig:
    """A live-source config plus the mutable name and immutable commit that produced it."""

    repo_id: str
    requested_revision: str
    resolved_commit: str
    config: dict[str, Any]
    config_sha256: str


def fetch_config_with_transport(
    repo_id: object,
    revision: object,
    transport: ConfigTransport,
    *,
    token: str | None = None,
    expected_config_sha256: str | None = None,
) -> FetchedConfig:
    """Fetch and validate one config through an injected exact-commit text transport.

    There is deliberately no implicit fallback. Snapshot mode and live mode are separate
    operations so a network, authentication, or revision error cannot silently change the data
    source used by a published calculation.
    """

    repository = validate_repo_id(repo_id)
    commit = validate_commit_revision(revision)
    if not callable(transport):
        raise TypeError("transport must be callable")
    text = transport(repository, commit, token)
    config = validate_config(strict_json_loads(text))
    digest = sha256_json(config)
    if expected_config_sha256 is not None:
        expected = _string("expected_config_sha256", expected_config_sha256).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ConfigSchemaError("expected_config_sha256 must be lowercase 64-hex")
        if digest != expected:
            raise ConfigIntegrityError(
                f"config checksum mismatch: expected {expected}, got {digest}"
            )
    return FetchedConfig(
        repo_id=repository,
        revision=commit,
        config=config,
        config_sha256=digest,
    )


def fetch_source_config_with_transports(
    repo_id: object,
    requested_revision: object,
    resolver: RevisionResolver,
    transport: ConfigTransport,
    *,
    token: str | None = None,
    expected_config_sha256: str | None = None,
) -> SourceFetchedConfig:
    """Resolve a live branch/tag to a commit, then fetch and validate that exact commit.

    ``requested_revision='main'`` is useful for a live illustration but is not reproducible by
    itself. The returned ``resolved_commit`` is the provenance record that makes the observed
    config reproducible later. Resolver, download, access, schema, and checksum failures remain
    visible; this function never falls back to the frozen course snapshot.
    """

    repository = validate_repo_id(repo_id)
    requested = validate_source_revision(requested_revision)
    if not callable(resolver):
        raise TypeError("resolver must be callable")
    if not callable(transport):
        raise TypeError("transport must be callable")
    resolved = validate_commit_revision(resolver(repository, requested, token))
    fetched = fetch_config_with_transport(
        repository,
        resolved,
        transport,
        token=token,
        expected_config_sha256=expected_config_sha256,
    )
    return SourceFetchedConfig(
        repo_id=repository,
        requested_revision=requested,
        resolved_commit=resolved,
        config=fetched.config,
        config_sha256=fetched.config_sha256,
    )


def huggingface_hub_transport(repo_id: str, revision: str, token: str | None) -> str:
    """Download only ``config.json`` at an exact commit and return its UTF-8 text."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "live mode requires 'huggingface_hub'; install it explicitly before use"
        ) from exc
    path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
        token=token,
    )
    return Path(path).read_text(encoding="utf-8")


def huggingface_hub_revision_resolver(
    repo_id: str,
    revision: str,
    token: str | None,
) -> str:
    """Resolve a Hub branch, tag, PR ref, or commit to the repository's full commit SHA."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "live mode requires 'huggingface_hub'; install it explicitly before use"
        ) from exc
    info = HfApi(token=token).model_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError("the Hub response did not contain a resolved commit SHA")
    return str(info.sha)


def fetch_hf_config(
    repo_id: object,
    revision: object,
    *,
    token: str | None = None,
    expected_config_sha256: str | None = None,
) -> FetchedConfig:
    """Fetch one Hugging Face config with exact revision pinning and no silent fallback."""

    return fetch_config_with_transport(
        repo_id,
        revision,
        huggingface_hub_transport,
        token=token,
        expected_config_sha256=expected_config_sha256,
    )


def fetch_hf_config_from_source(
    repo_id: object,
    revision: object = "main",
    *,
    token: str | None = None,
    expected_config_sha256: str | None = None,
) -> SourceFetchedConfig:
    """Fetch a current/source config, recording the immutable commit actually observed.

    Use ``revision='main'`` or a tag only for an opt-in live illustration. Save the returned
    ``resolved_commit`` and pass it to :func:`fetch_hf_config` when reproducibility matters.
    """

    return fetch_source_config_with_transports(
        repo_id,
        revision,
        huggingface_hub_revision_resolver,
        huggingface_hub_transport,
        token=token,
        expected_config_sha256=expected_config_sha256,
    )


# END EDUCATIONAL CORE: live_fetch


# Maintainer input: exact upstream revisions and a documented schema projection.
VERIFIED_INPUTS = json.loads((MODULE_ROOT / "data/verified_model_inputs.json").read_text(encoding="utf-8"))
RAW_MODELS: list[dict[str, Any]] = VERIFIED_INPUTS["models"]


CONTEXT_MILESTONES = [
    {"year": 2019.10, "tokens": 1_024, "model": "GPT-2"},
    {"year": 2020.40, "tokens": 2_048, "model": "GPT-3"},
    {"year": 2022.90, "tokens": 4_096, "model": "GPT-3.5"},
    {"year": 2023.20, "tokens": 32_768, "model": "GPT-4"},
    {"year": 2023.40, "tokens": 100_000, "model": "Claude"},
    {"year": 2023.90, "tokens": 128_000, "model": "GPT-4 Turbo"},
    {"year": 2024.15, "tokens": 1_000_000, "model": "Gemini 1.5 Pro"},
    {"year": 2024.20, "tokens": 200_000, "model": "Claude 3"},
    {"year": 2025.27, "tokens": 10_000_000, "model": "Llama 4 Scout"},
    {"year": 2026.57, "tokens": 1_048_576, "model": "Kimi K3"},
    {"year": 2026.69, "tokens": 1_000_000, "model": "DeepSeek V4.1 Flash"},
    {"year": 2026.74, "tokens": 1_050_000, "model": "GPT-6.1 Sol"},
]

COST_ASSUMPTIONS = {
    "bf16_bits": 16.0,
    "four_bit_effective_bits": 4.25,
    "card_capacity_gb_decimal": 80.0,
}


def build_reference_payload() -> dict[str, Any]:
    """Build the frozen snapshot and exact expected calculations."""

    models: list[dict[str, Any]] = []
    for raw in RAW_MODELS:
        config = validate_config(raw["config"])
        fixture = SnapshotModel(
            repo_id=validate_repo_id(raw["repo_id"]),
            display_name=_string("display_name", raw["display_name"]),
            config=config,
            config_sha256=sha256_json(config),
            claimed_total_b=float(raw["claimed_total_b"]),
            claimed_active_b=float(raw["claimed_active_b"]),
            active_convention=str(raw["active_convention"]),
            total_tolerance_pct=float(raw["total_tolerance_pct"]),
            active_tolerance_pct=float(raw["active_tolerance_pct"]),
        )
        reconciliation = reconcile_model(fixture)
        breakdown = count_parameters(config)
        item = {
            **{key: value for key, value in raw.items() if key != "config"},
            "config": config,
            "config_sha256": fixture.config_sha256,
            "expected": {
                "attention": attention_kind(config),
                "ffn": moe_kind(config),
                "context": context_info(config),
                "total_params": breakdown.total,
                "active_params": breakdown.active(fixture.active_convention),
                "delta_total_pct": reconciliation.delta_total_pct,
                "delta_active_pct": reconciliation.delta_active_pct,
            },
        }
        models.append(item)
    payload = {
        "schema_version": 1,
        "snapshot_date": VERIFIED_INPUTS["as_of"],
        "models": models,
        "context_milestones": CONTEXT_MILESTONES,
        "cost_assumptions": COST_ASSUMPTIONS,
    }
    load_reference_bundle(payload)
    return payload


def validate_reference_payload(payload: object) -> ReferenceBundle:
    """Validate payload checksums and all exact expected calculations."""

    bundle = load_reference_bundle(payload)
    raw_models = _mapping("reference payload", payload)["models"]
    by_repo = {item["repo_id"]: item for item in raw_models}
    for model in bundle.models:
        expected = _mapping("expected", by_repo[model.repo_id].get("expected"))
        breakdown = count_parameters(model.config)
        actual = {
            "attention": attention_kind(model.config),
            "ffn": moe_kind(model.config),
            "context": context_info(model.config),
            "total_params": breakdown.total,
            "active_params": breakdown.active(model.active_convention),
        }
        for key, value in actual.items():
            if expected.get(key) != value:
                raise AssertionError(
                    f"{model.repo_id}: expected[{key!r}]={expected.get(key)!r}, got {value!r}"
                )
        reconcile_model(model)
    return bundle


def write_reference(path: Path = REFERENCE_PATH) -> str:
    """Write canonical reference JSON and return its SHA-256."""

    payload = build_reference_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_json(payload)


def load_reference(path: Path = REFERENCE_PATH) -> tuple[dict[str, Any], ReferenceBundle]:
    """Load and validate committed reference JSON."""

    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    return payload, validate_reference_payload(payload)


def run_self_tests() -> None:
    """Exercise important schema, accounting, live-mode, and edge-case failures."""

    payload = build_reference_payload()
    bundle = validate_reference_payload(payload)
    assert len(bundle.models) == 8
    by_repo = {model.repo_id: model for model in bundle.models}
    assert attention_kind(by_repo["deepseek-ai/DeepSeek-V3"].config).startswith("MLA")
    assert moe_kind(by_repo["Qwen/Qwen3-32B"].config) == "dense"
    assert count_parameters(by_repo["openai/gpt-oss-120b"].config).total == 116_829_156_672

    fake_revision = "a" * 40
    text = canonical_json(by_repo["openai/gpt-oss-20b"].config)
    calls: list[tuple[str, str, str | None]] = []

    def fake_transport(repo_id: str, revision: str, token: str | None) -> str:
        calls.append((repo_id, revision, token))
        return text

    fetched = fetch_config_with_transport(
        "openai/gpt-oss-20b",
        fake_revision,
        fake_transport,
        token="secret",
    )
    assert fetched.revision == fake_revision and len(calls) == 1

    source_calls: list[tuple[str, str, str | None]] = []

    def fake_resolver(repo_id: str, revision: str, token: str | None) -> str:
        """Resolve an exploratory name to the deterministic fake commit."""

        source_calls.append((repo_id, revision, token))
        return fake_revision

    source_fetched = fetch_source_config_with_transports(
        "openai/gpt-oss-20b",
        "main",
        fake_resolver,
        fake_transport,
        token="secret",
    )
    assert source_fetched.requested_revision == "main"
    assert source_fetched.resolved_commit == fake_revision
    assert source_fetched.config_sha256 == fetched.config_sha256
    assert source_calls == [("openai/gpt-oss-20b", "main", "secret")]
    assert len(calls) == 2

    def failing_transport(repo_id: str, revision: str, token: str | None) -> str:
        """Simulate a visible network/authentication failure without fallback."""

        raise ConnectionError("simulated live-mode failure")

    try:
        fetch_config_with_transport(
            "openai/gpt-oss-20b",
            fake_revision,
            failing_transport,
        )
    except ConnectionError:
        pass
    else:
        raise AssertionError("live transport failure was unexpectedly hidden")

    failures: list[Callable[[], object]] = [
        lambda: validate_commit_revision("main"),
        lambda: validate_source_revision(""),
        lambda: validate_source_revision("bad revision"),
        lambda: strict_json_loads('{"x": 1, "x": 2}'),
        lambda: strict_json_loads('{"x": NaN}'),
        lambda: validate_config({"hidden_size": 8}),
        lambda: validate_config(
            {
                **by_repo["openai/gpt-oss-20b"].config,
                "num_local_experts": 32,
                "num_experts": 64,
            }
        ),
        lambda: validate_config(
            {
                **by_repo["Qwen/Qwen3-8B"].config,
                "num_key_value_heads": 7,
            }
        ),
    ]
    for failure in failures:
        try:
            failure()
        except (ConfigSchemaError, TypeError):
            pass
        else:
            raise AssertionError("negative test unexpectedly succeeded")


def main() -> None:
    """Validate the committed reference or intentionally regenerate it."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite reference_metrics.json")
    args = parser.parse_args()
    run_self_tests()
    if args.write:
        digest = write_reference()
        print(f"wrote {REFERENCE_PATH} (canonical payload sha256={digest})")
    else:
        payload, bundle = load_reference()
        print(
            f"validated {len(bundle.models)} Module 1 fixtures; "
            f"canonical payload sha256={sha256_json(payload)}"
        )


if __name__ == "__main__":
    main()
