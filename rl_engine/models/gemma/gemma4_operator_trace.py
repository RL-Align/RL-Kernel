# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gemma 4 per-layer operator trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.models.gemma.gemma4_workload import Gemma4Manifest, load_manifest

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"
_LAYER_PERIOD = 6
_FULL_SLOT = 5  # layer i is full_attention iff i % _LAYER_PERIOD == _FULL_SLOT

OFFICIAL_FINGERPRINT: dict[str, Any] = {
    "num_hidden_layers": 60,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "num_global_key_value_heads": 4,
    "global_head_dim": 512,
    "sliding_window": 1024,
    "attention_k_eq_v": True,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "enable_moe_block": False,
    "use_double_wide_mlp": False,
    "attention_bias": False,
    "tie_word_embeddings": True,
    "vocab_size": 262144,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "hidden_activation": "gelu_pytorch_tanh",
    "final_logit_softcapping": 30.0,
    "use_bidirectional_attention": "vision",
    "rope_parameters": {
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 10000.0},
        FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}

# Layer structure the node list below assumes; pinned so a manifest that disagrees is
# rejected, the way Qwen3 pins ``qk_norm`` and ``swiglu`` without reading them at runtime.
_OFFICIAL_QKV_NORM = {
    "q": {"norm": "rmsnorm", "with_scale": True, "rope": True},
    "k": {"norm": "rmsnorm", "with_scale": True, "rope": True},
    "v": {"norm": "rmsnorm", "with_scale": False, "rope": False},
}
_OFFICIAL_NORM_RESIDUAL_ORDER = [
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "residual_add",
    "pre_feedforward_layernorm",
    "mlp",
    "post_feedforward_layernorm",
    "residual_add",
]

NODE_KINDS = (
    "scaled_embedding",
    "rms_norm",
    "det_gemm",
    "qkv_norm",
    "rope_sliding",
    "rope_global",
    "attention_sliding",
    "attention_global",
    "gelu_tanh_mul",
    "tied_lm_head",
    "logit_softcap",
    "logprob",
)

# Kinds the chain computes directly as plain tensor ops; they never resolve to a backend
# candidate.
NO_BACKEND_KINDS = ("residual_add", "layer_scale")


@dataclass(frozen=True)
class Gemma4Spec:
    """Pinned official Gemma-4-31B-it text-model identity. Shrinking any field is forbidden."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_global_key_value_heads: int
    global_head_dim: int
    sliding_window: int
    attention_k_eq_v: bool
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    hidden_activation: str
    final_logit_softcapping: float
    use_bidirectional_attention: str
    tie_word_embeddings: bool
    layer_types: tuple[str, ...]
    rope_parameters: dict[str, dict[str, Any]]
    workload_id: str
    model_id: str
    revision: str
    weight_content_hash: str

    @classmethod
    def from_manifest(cls, manifest: Gemma4Manifest | None = None) -> Gemma4Spec:
        m = manifest if manifest is not None else load_manifest()
        ident = m.model_identity
        fp = ident["config_fingerprint"]
        text = fp["text_config"]
        for key, expected in OFFICIAL_FINGERPRINT.items():
            got = text.get(key)
            if got != expected:
                raise ValueError(
                    f"gemma4_operator_trace forbids architecture shrink/drift: text_config "
                    f"{key}={got!r} != official {expected!r}"
                )
        layer_types = tuple(str(t) for t in text.get("layer_types") or ())
        expected_types = tuple(
            FULL_ATTENTION if i % _LAYER_PERIOD == _FULL_SLOT else SLIDING_ATTENTION
            for i in range(int(OFFICIAL_FINGERPRINT["num_hidden_layers"]))
        )
        if layer_types != expected_types:
            raise ValueError(
                "gemma4_operator_trace forbids architecture shrink/drift: layer_types deviates "
                "from the official 5x sliding + 1x full schedule"
            )
        for key, expected in (
            ("attention_qkv_norm", _OFFICIAL_QKV_NORM),
            ("norm_residual_order", _OFFICIAL_NORM_RESIDUAL_ORDER),
        ):
            got = fp.get(key)
            if got != expected:
                raise ValueError(
                    f"gemma4_operator_trace forbids architecture shrink/drift: "
                    f"config_fingerprint {key}={got!r} != official {expected!r}"
                )
        return cls(
            num_hidden_layers=int(text["num_hidden_layers"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            num_global_key_value_heads=int(text["num_global_key_value_heads"]),
            global_head_dim=int(text["global_head_dim"]),
            sliding_window=int(text["sliding_window"]),
            attention_k_eq_v=bool(text["attention_k_eq_v"]),
            vocab_size=int(text["vocab_size"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            hidden_activation=str(text["hidden_activation"]),
            final_logit_softcapping=float(text["final_logit_softcapping"]),
            use_bidirectional_attention=str(text["use_bidirectional_attention"]),
            tie_word_embeddings=bool(text["tie_word_embeddings"]),
            layer_types=layer_types,
            rope_parameters={k: dict(v) for k, v in text["rope_parameters"].items()},
            workload_id=str(m.workload_id),
            model_id=str(ident["model_id"]),
            revision=str(ident["revision"]),
            weight_content_hash=str(ident["weight_snapshot"]["content_hash"]),
        )

    def layer_type(self, index: int) -> str:
        return self.layer_types[index]

    def has_v_proj(self, layer_type: str) -> bool:
        return not (self.attention_k_eq_v and layer_type == FULL_ATTENTION)

    def node_names(self) -> tuple[str, ...]:
        names: list[str] = ["embedding"]
        for index in range(self.num_hidden_layers):
            prefix = f"layers.{index}"
            suffixes = _layer_node_suffixes(self, self.layer_type(index))
            names.extend(f"{prefix}.{suffix}" for suffix in suffixes)
        names.extend(["final_layernorm", "lm_head", "logit_softcap", "logprob"])
        return tuple(names)

    def node_kind(self, node_name: str) -> str:
        layer, suffix = _split_name(node_name)
        layer_type = self.layer_type(layer) if layer is not None else None
        if suffix == "embedding" and layer is None:
            return "scaled_embedding"
        if suffix.endswith("layernorm"):
            return "rms_norm"
        if layer_type is not None:
            if suffix in ("q_norm", "k_norm", "v_norm"):
                return "qkv_norm"
            if suffix in ("rope_q", "rope_k"):
                return "rope_sliding" if layer_type == SLIDING_ATTENTION else "rope_global"
            if suffix == "attn":
                return (
                    "attention_sliding" if layer_type == SLIDING_ATTENTION else "attention_global"
                )
            if suffix in (
                "q_proj",
                "k_proj",
                "v_proj",
                "kv_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                if suffix in ("k_proj", "v_proj") and not self.has_v_proj(layer_type):
                    raise KeyError(
                        f"{node_name!r}: full-attention layers project K and V once, as kv_proj"
                    )
                if suffix == "kv_proj" and self.has_v_proj(layer_type):
                    raise KeyError(f"{node_name!r}: sliding layers have separate k_proj and v_proj")
                return "det_gemm"
            if suffix == "gelu_tanh_mul":
                return "gelu_tanh_mul"
            if suffix in ("residual_attn", "residual_mlp"):
                return "residual_add"
            if suffix == "layer_scalar":
                return "layer_scale"
        if layer is None:
            if suffix == "lm_head":
                return "tied_lm_head"
            if suffix == "logit_softcap":
                return "logit_softcap"
            if suffix == "logprob":
                return "logprob"
        raise KeyError(f"unknown node {node_name!r}")


def _layer_node_suffixes(spec: Gemma4Spec, layer_type: str) -> tuple[str, ...]:
    attention = ["q_proj", "q_norm", "rope_q"]
    if spec.has_v_proj(layer_type):
        attention.extend(["k_proj", "v_proj"])
    else:
        # One projection (k_proj.weight) feeds both K, via k_norm and RoPE, and V, via v_norm.
        attention.append("kv_proj")
    attention.extend(["k_norm", "rope_k", "v_norm", "attn", "o_proj"])
    return (
        "input_layernorm",
        *attention,
        "post_attention_layernorm",
        "residual_attn",
        "pre_feedforward_layernorm",
        "gate_proj",
        "up_proj",
        "gelu_tanh_mul",
        "down_proj",
        "post_feedforward_layernorm",
        "residual_mlp",
        "layer_scalar",
    )


def _split_name(node_name: str) -> tuple[int | None, str]:
    parts = node_name.split(".")
    if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit():
        return int(parts[1]), parts[2]
    if len(parts) == 1:
        return None, parts[0]
    raise KeyError(f"unknown node {node_name!r}")


__all__ = [
    "FULL_ATTENTION",
    "Gemma4Spec",
    "NODE_KINDS",
    "NO_BACKEND_KINDS",
    "OFFICIAL_FINGERPRINT",
    "SLIDING_ATTENTION",
]
