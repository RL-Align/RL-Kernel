# SPDX-License-Identifier: Apache-2.0
"""Pinned Qwen3.6-35B-A3B identity. Dense Qwen3 config must not load as this model."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MODEL_ID = "Qwen3.6-35B-A3B"
ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
MODEL_TYPE = "qwen3_5_moe"
TEXT_MODEL_TYPE = "qwen3_5_moe_text"
TEXT_WEIGHT_PREFIX = "model.language_model"

NUM_HIDDEN_LAYERS = 40
FULL_ATTENTION_INTERVAL = 4
HIDDEN_SIZE = 2048
VOCAB_SIZE = 248320
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE_SIZE = 512
SHARED_EXPERT_INTERMEDIATE_SIZE = 512

# Text post-training scores the language stack. The checkpoint still contains
# the vision tower and one MTP layer; those stays out of this forward.
TEXT_FORWARD_INCLUDES_VISION = False
TEXT_FORWARD_INCLUDES_MTP = False

REQUIRED_OPERATORS = frozenset(
    {
        "rms_norm",
        "qk_norm",
        "det_gemm",
        "grouped_gemm",
        "swiglu",
        "mrope",
        "attention",
        "attn_output_gate",
        "causal_conv",
        "gdn",
        "gated_rms_norm",
        "router",
        "topk",
        "dispatch",
        "combine",
        "logprob",
    }
)

_LINEAR = "linear_attention"
_FULL = "full_attention"
OFFICIAL_LAYER_TYPES = tuple(
    _FULL if (index + 1) % FULL_ATTENTION_INTERVAL == 0 else _LINEAR
    for index in range(NUM_HIDDEN_LAYERS)
)

_TEXT_FINGERPRINT = {
    "attention_bias": False,
    "attn_output_gate": True,
    "dtype": "bfloat16",
    "full_attention_interval": FULL_ATTENTION_INTERVAL,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": HIDDEN_SIZE,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 16,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_key_value_heads": 2,
    "output_router_logits": False,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
    "tie_word_embeddings": False,
    "vocab_size": VOCAB_SIZE,
}
_ROPE_FINGERPRINT = {
    "mrope_interleaved": True,
    "mrope_section": (11, 11, 10),
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}
_VISION_FINGERPRINT = {
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "intermediate_size": 4304,
    "model_type": MODEL_TYPE,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": HIDDEN_SIZE,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


def official_config() -> dict:
    """Config object matching the pinned Qwen3.6-35B-A3B config.json."""

    text = dict(_TEXT_FINGERPRINT)
    text["layer_types"] = list(OFFICIAL_LAYER_TYPES)
    text["model_type"] = TEXT_MODEL_TYPE
    text["rope_parameters"] = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    vision = dict(_VISION_FINGERPRINT)
    vision["deepstack_visual_indexes"] = []
    return {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "text_config": text,
        "tie_word_embeddings": False,
        "vision_config": vision,
    }


@dataclass(frozen=True)
class Qwen3MoeSpec:
    """Identity, layer schedule, and parameter layout for one checkpoint.

    Weight names follow ``Qwen3_5MoeForConditionalGeneration``: the language
    stack lives under ``model.language_model`` and the output projection is
    ``lm_head``. This module does not pin shard hashes; pass the checkpoint
    key set to :meth:`missing_parameters`.
    """

    model_id: str
    architecture: str
    model_type: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_output_gate: bool
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    partial_rotary_factor: float
    rope_theta: int
    mrope_interleaved: bool
    mrope_section: tuple[int, int, int]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    router_aux_loss_coef: float
    output_router_logits: bool
    rms_norm_eps: float
    hidden_act: str
    dtype: str
    tie_word_embeddings: bool
    max_position_embeddings: int
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    layer_types: tuple[str, ...]
    text_forward_includes_vision: bool
    text_forward_includes_mtp: bool

    @classmethod
    def from_config(cls, config: Mapping) -> Qwen3MoeSpec:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        architectures = config.get("architectures")
        if architectures != [ARCHITECTURE]:
            raise ValueError(
                f"{MODEL_ID} requires architectures {[ARCHITECTURE]!r}, got {architectures!r}"
            )
        _require_equal(config, "model_type", MODEL_TYPE)
        _require_equal(config, "tie_word_embeddings", False)
        text = config.get("text_config")
        vision = config.get("vision_config")
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ValueError(
                f"{MODEL_ID} requires text_config and vision_config; "
                "a Qwen3 dense config cannot be substituted"
            )
        _require_equal(text, "model_type", TEXT_MODEL_TYPE)
        for key, expected in _TEXT_FINGERPRINT.items():
            _require_equal(text, key, expected)
        rope = text.get("rope_parameters")
        if not isinstance(rope, Mapping):
            raise ValueError("text_config.rope_parameters must be a mapping")
        for key, expected in _ROPE_FINGERPRINT.items():
            _require_equal(rope, key, expected if key != "mrope_section" else list(expected))
        layer_types = tuple(text.get("layer_types") or ())
        if layer_types != OFFICIAL_LAYER_TYPES:
            raise ValueError(
                f"layer_types must be the pinned {NUM_HIDDEN_LAYERS}-layer "
                "linear/full schedule"
            )
        for key, expected in _VISION_FINGERPRINT.items():
            _require_equal(vision, key, expected)
        if list(vision.get("deepstack_visual_indexes") or []) != []:
            raise ValueError("vision deepstack_visual_indexes must be empty for this pin")
        return cls(
            model_id=MODEL_ID,
            architecture=ARCHITECTURE,
            model_type=MODEL_TYPE,
            text_model_type=TEXT_MODEL_TYPE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            hidden_size=HIDDEN_SIZE,
            vocab_size=VOCAB_SIZE,
            head_dim=int(text["head_dim"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            attn_output_gate=bool(text["attn_output_gate"]),
            linear_num_key_heads=int(text["linear_num_key_heads"]),
            linear_num_value_heads=int(text["linear_num_value_heads"]),
            linear_key_head_dim=int(text["linear_key_head_dim"]),
            linear_value_head_dim=int(text["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
            partial_rotary_factor=float(text["partial_rotary_factor"]),
            rope_theta=int(rope["rope_theta"]),
            mrope_interleaved=bool(rope["mrope_interleaved"]),
            mrope_section=_ROPE_FINGERPRINT["mrope_section"],
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
            shared_expert_intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
            router_aux_loss_coef=float(text["router_aux_loss_coef"]),
            output_router_logits=bool(text["output_router_logits"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            hidden_act=str(text["hidden_act"]),
            dtype=str(text["dtype"]),
            tie_word_embeddings=False,
            max_position_embeddings=int(text["max_position_embeddings"]),
            mtp_num_hidden_layers=int(text["mtp_num_hidden_layers"]),
            mtp_use_dedicated_embeddings=bool(text["mtp_use_dedicated_embeddings"]),
            layer_types=layer_types,
            text_forward_includes_vision=TEXT_FORWARD_INCLUDES_VISION,
            text_forward_includes_mtp=TEXT_FORWARD_INCLUDES_MTP,
        )

    def layer_type(self, layer_index: int) -> str:
        return self.layer_types[layer_index]

    def node_names(self) -> tuple[str, ...]:
        names = ["embedding"]
        for index, layer_type in enumerate(self.layer_types):
            prefix = f"layers.{index}"
            names.extend(_attention_nodes(prefix, layer_type))
            names.extend(_moe_nodes(prefix))
        names.extend(["final_layernorm", "lm_head", "logprob", "loss"])
        return tuple(names)

    def node_kind(self, node_name: str) -> str:
        if node_name == "embedding":
            return "embedding"
        if node_name in {"final_layernorm"} or node_name.endswith("layernorm"):
            return "rms_norm"
        if node_name.endswith(".gdn_norm"):
            return "gated_rms_norm"
        if node_name.endswith((".q_norm", ".k_norm")):
            return "qk_norm"
        if node_name.endswith(".conv1d"):
            return "causal_conv"
        if node_name.endswith(".gdn"):
            return "gdn"
        if node_name.endswith((".rope_q", ".rope_k")):
            return "mrope"
        if node_name.endswith(".attn"):
            return "attention"
        if node_name.endswith(".attn_output_gate"):
            return "attn_output_gate"
        if node_name.endswith(".router"):
            return "router"
        if node_name.endswith(".topk"):
            return "topk"
        if node_name.endswith(".dispatch"):
            return "dispatch"
        if node_name.endswith(".combine"):
            return "combine"
        if node_name.endswith(".swiglu"):
            return "swiglu"
        if node_name.endswith((".expert_gate_up", ".expert_down")):
            return "grouped_gemm"
        if node_name.endswith(
            (
                ".in_proj_qkv",
                ".in_proj_z",
                ".in_proj_a",
                ".in_proj_b",
                ".out_proj",
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
                ".shared_gate_proj",
                ".shared_up_proj",
                ".shared_down_proj",
                ".shared_expert_gate",
            )
        ):
            return "det_gemm"
        if node_name.endswith((".residual_attn", ".residual_mlp")):
            return "residual_add"
        if node_name == "lm_head":
            return "lm_head"
        if node_name == "logprob":
            return "logprob"
        if node_name == "loss":
            return "masked_loss"
        raise KeyError(f"unknown node {node_name!r}")

    def missing_operators(self, available: Iterable[str]) -> tuple[str, ...]:
        present = set(available)
        return tuple(sorted(REQUIRED_OPERATORS - present))

    def parameter_names(self) -> tuple[str, ...]:
        names = [
            f"{TEXT_WEIGHT_PREFIX}.embed_tokens.weight",
            f"{TEXT_WEIGHT_PREFIX}.norm.weight",
            "lm_head.weight",
        ]
        for index, layer_type in enumerate(self.layer_types):
            prefix = f"{TEXT_WEIGHT_PREFIX}.layers.{index}"
            names.extend(f"{prefix}.{suffix}" for suffix in _layer_parameter_suffixes(layer_type))
        return tuple(names)

    def parameter_shape(self, name: str) -> tuple[int, ...]:
        if name == f"{TEXT_WEIGHT_PREFIX}.embed_tokens.weight":
            return (self.vocab_size, self.hidden_size)
        if name == f"{TEXT_WEIGHT_PREFIX}.norm.weight":
            return (self.hidden_size,)
        if name == "lm_head.weight":
            return (self.vocab_size, self.hidden_size)
        marker = f"{TEXT_WEIGHT_PREFIX}.layers."
        if not name.startswith(marker):
            raise KeyError(f"unknown parameter {name!r}")
        rest = name[len(marker):]
        layer_text, _, suffix = rest.partition(".")
        layer_index = int(layer_text)
        shapes = self._layer_parameter_shapes(self.layer_type(layer_index))
        if suffix not in shapes:
            raise KeyError(f"unknown parameter {name!r}")
        return shapes[suffix]

    def missing_parameters(self, keys: Iterable[str]) -> tuple[str, ...]:
        present = set(keys)
        return tuple(name for name in self.parameter_names() if name not in present)

    def _layer_parameter_shapes(self, layer_type: str) -> dict[str, tuple[int, ...]]:
        hidden = self.hidden_size
        shapes: dict[str, tuple[int, ...]] = {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "mlp.gate.weight": (self.num_experts, hidden),
            "mlp.experts.gate_up_proj": (
                self.num_experts,
                hidden,
                2 * self.moe_intermediate_size,
            ),
            "mlp.experts.down_proj": (
                self.num_experts,
                self.moe_intermediate_size,
                hidden,
            ),
            "mlp.shared_expert.gate_proj.weight": (
                self.shared_expert_intermediate_size,
                hidden,
            ),
            "mlp.shared_expert.up_proj.weight": (
                self.shared_expert_intermediate_size,
                hidden,
            ),
            "mlp.shared_expert.down_proj.weight": (
                hidden,
                self.shared_expert_intermediate_size,
            ),
            "mlp.shared_expert_gate.weight": (1, hidden),
        }
        if layer_type == _LINEAR:
            key_dim = self.linear_num_key_heads * self.linear_key_head_dim
            value_dim = self.linear_num_value_heads * self.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            shapes.update(
                {
                    "linear_attn.in_proj_qkv.weight": (conv_dim, hidden),
                    "linear_attn.in_proj_z.weight": (value_dim, hidden),
                    "linear_attn.in_proj_a.weight": (self.linear_num_value_heads, hidden),
                    "linear_attn.in_proj_b.weight": (self.linear_num_value_heads, hidden),
                    "linear_attn.conv1d.weight": (
                        conv_dim,
                        1,
                        self.linear_conv_kernel_dim,
                    ),
                    "linear_attn.dt_bias": (self.linear_num_value_heads,),
                    "linear_attn.A_log": (self.linear_num_value_heads,),
                    "linear_attn.norm.weight": (self.linear_value_head_dim,),
                    "linear_attn.out_proj.weight": (hidden, value_dim),
                }
            )
        elif layer_type == _FULL:
            query_dim = self.num_attention_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            shapes.update(
                {
                    "self_attn.q_proj.weight": (query_dim * 2, hidden),
                    "self_attn.k_proj.weight": (kv_dim, hidden),
                    "self_attn.v_proj.weight": (kv_dim, hidden),
                    "self_attn.o_proj.weight": (hidden, query_dim),
                    "self_attn.q_norm.weight": (self.head_dim,),
                    "self_attn.k_norm.weight": (self.head_dim,),
                }
            )
        else:
            raise ValueError(f"unknown layer type {layer_type!r}")
        return shapes


def _require_equal(config: Mapping, key: str, expected) -> None:
    got = config.get(key)
    if got != expected:
        raise ValueError(f"config {key}={got!r} != pinned {expected!r}")


def _attention_nodes(prefix: str, layer_type: str) -> tuple[str, ...]:
    if layer_type == _LINEAR:
        body = (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "conv1d",
            "gdn",
            "gdn_norm",
            "out_proj",
        )
    elif layer_type == _FULL:
        body = (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_norm",
            "k_norm",
            "rope_q",
            "rope_k",
            "attn",
            "attn_output_gate",
            "o_proj",
        )
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    return (f"{prefix}.input_layernorm", *(f"{prefix}.{name}" for name in body), f"{prefix}.residual_attn")


def _moe_nodes(prefix: str) -> tuple[str, ...]:
    body = (
        "post_attention_layernorm",
        "router",
        "topk",
        "dispatch",
        "expert_gate_up",
        "expert_swiglu",
        "expert_down",
        "shared_gate_proj",
        "shared_up_proj",
        "shared_swiglu",
        "shared_down_proj",
        "shared_expert_gate",
        "combine",
        "residual_mlp",
    )
    return tuple(f"{prefix}.{name}" for name in body)


def _layer_parameter_suffixes(layer_type: str) -> tuple[str, ...]:
    shared = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    if layer_type == _LINEAR:
        attention = (
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.A_log",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
        )
    elif layer_type == _FULL:
        attention = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        )
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    return attention + shared
