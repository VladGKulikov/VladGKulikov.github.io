"""A minimal model passport from a validated config.json."""
import json

EXAMPLE_CONFIG_JSON = r"""
{
  "hidden_size": 2880,
  "num_hidden_layers": 24,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "num_local_experts": 32,
  "num_experts_per_tok": 4,
  "max_position_embeddings": 131072,
  "vocab_size": 201088,
  "rope_scaling": {"rope_type": "yarn", "factor": 32}
}
"""


def positive_int(config: dict, key: str) -> int:
    """Read one required positive integer field."""

    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def one_of(config: dict, *keys: str):
    """Read one alias field without hiding conflicting values."""

    found = [(key, config[key]) for key in keys if config.get(key) is not None]
    if not found:
        return None
    if any(type(value) is not type(found[0][1]) or value != found[0][1]
           for _, value in found[1:]):
        raise ValueError(f"conflicting aliases: {[key for key, _ in found]}")
    return found[0][1]


def attention_kind(config: dict) -> str:
    """Classify MHA (Multi-Head Attention), MQA (Multi-Query Attention), GQA, or MLA from validated fields."""

    if "kv_lora_rank" in config:  # check the MLA signature first
        return f"MLA (kv_rank={positive_int(config, 'kv_lora_rank')})"
    heads = positive_int(config, "num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    if isinstance(kv_heads, bool) or not isinstance(kv_heads, int) or kv_heads <= 0:
        raise ValueError("num_key_value_heads must be a positive integer")
    if kv_heads > heads or heads % kv_heads:
        raise ValueError("KV heads must divide Q heads")
    if kv_heads == heads:
        return f"MHA ({heads} heads)"
    if kv_heads == 1:
        return f"MQA ({heads} Q heads / 1 KV)"
    return f"GQA ({heads} Q heads / {kv_heads} KV)"


def moe_kind(config: dict) -> str:
    """Classify dense/MoE across common field aliases."""

    experts = one_of(config, "num_local_experts", "n_routed_experts", "num_experts")
    if experts is None:
        if any(config.get(k) is not None for k in ("num_experts_per_tok", "num_experts_per_token", "n_shared_experts", "num_shared_experts")):
            raise ValueError("expert settings require an expert count")
        return "dense"
    if isinstance(experts, bool) or not isinstance(experts, int) or experts <= 0:
        raise ValueError("expert count must be a positive integer")
    top_k = one_of(config, "num_experts_per_tok", "num_experts_per_token")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError("top-k must be in [1, expert count]")
    shared = one_of(config, "n_shared_experts", "num_shared_experts")
    shared = 0 if shared is None else shared
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        raise ValueError("n_shared_experts must be a non-negative integer")
    return f"MoE: {experts} experts, top-{top_k}" + (f" + {shared} shared" if shared else "")


def passport(config: dict) -> dict:
    """Return a minimal structural passport without a network call."""

    return {
        "layers_x_hidden": (
            positive_int(config, "num_hidden_layers"),
            positive_int(config, "hidden_size"),
        ),
        "attention": attention_kind(config),
        "ffn": moe_kind(config),
        "context": positive_int(config, "max_position_embeddings"),
        "vocabulary": positive_int(config, "vocab_size"),
    }
