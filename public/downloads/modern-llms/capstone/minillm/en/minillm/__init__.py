"""MiniLLM — the course's educational end-to-end reference implementation."""
from .config import MiniLLMConfig, gqa_variants, kv_bytes_per_token
from .model import MiniLLM, KVCache, MoELayer, RMSNorm, RotaryEmbedding, apply_rope
from .data import ByteTokenizer, ShardStream, TokenStream, toy_corpus, write_shards
from .tokenizer import BBPETokenizer
from .corpus import FilterConfig, dedup_exact, minhash_dedup, normalize, pipeline
from .chat import (CHAT_SPECIALS, ChatTemplate, Message, SFTDataset,
                   pack_first_fit_decreasing)
from .preference import (DPOConfig, PreferencePair, dpo_loss, freeze_reference,
                         sequence_logprob, train_dpo)
from .probes import RepeatTask, chance_level, repeat_accuracy
from .precision import (PrecisionPolicy, autocast_context, model_parameter_dtype,
                        native_cuda_bf16_supported, resolve_compute_dtype)
from .train import (Muon, TrainConfig, TrainState, build_optimizer, evaluate,
                    load_checkpoint, load_model_artifact, memory_report,
                    save_checkpoint, save_model_artifact, train, wsd_lr_scale,
                    zeropower_via_newtonschulz)
from .generate import (GenerationStats, SamplingConfig, bits_per_weight, generate,
                       logits_to_probs, quantize_fp4, quantize_int8_per_channel,
                       quantize_model, speculative_generate)

__all__ = [
    "MiniLLMConfig", "MiniLLM", "KVCache", "MoELayer", "RMSNorm", "RotaryEmbedding",
    "apply_rope", "gqa_variants", "kv_bytes_per_token",
    "ByteTokenizer", "TokenStream", "ShardStream", "toy_corpus", "write_shards",
    "BBPETokenizer", "FilterConfig", "normalize", "dedup_exact", "minhash_dedup",
    "pipeline",
    "CHAT_SPECIALS", "ChatTemplate", "Message", "SFTDataset",
    "pack_first_fit_decreasing",
    "DPOConfig", "PreferencePair", "dpo_loss", "freeze_reference",
    "sequence_logprob", "train_dpo",
    "RepeatTask", "chance_level", "repeat_accuracy",
    "PrecisionPolicy", "resolve_compute_dtype", "native_cuda_bf16_supported",
    "autocast_context", "model_parameter_dtype",
    "Muon", "TrainConfig", "TrainState", "build_optimizer", "evaluate",
    "load_checkpoint", "load_model_artifact", "memory_report", "save_checkpoint",
    "save_model_artifact", "train", "wsd_lr_scale", "zeropower_via_newtonschulz",
    "GenerationStats", "SamplingConfig", "bits_per_weight", "generate",
    "logits_to_probs", "quantize_fp4", "quantize_int8_per_channel",
    "quantize_model", "speculative_generate",
]
