# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse Glimmer per-layer operator trace.

Lists every operator of the 52-layer text model in execution order and maps each one to
the RFC #501 work-item row that owns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.models.muse_glimmer.workload import (
    FULL_ATTENTION,
    OFFICIAL_FINGERPRINT,
    OFFICIAL_PARALLEL_LAYOUT,
    SLIDING_ATTENTION,
    MuseGlimmerManifest,
    expected_layer_types,
    load_manifest,
)

# Node kind -> RFC #501 work-item row.
NODE_KIND_ROWS: dict[str, str] = {
    "normalized_embedding": "normalized_embedding",
    "centered_rms_norm": "fused_add_rmsnorm",
    "final_rms_norm": "fused_add_rmsnorm",
    "residual_add": "fused_add_rmsnorm",
    "det_gemm": "dense_gemm_qualification",
    "qk_norm": "qknorm_scaled_rope",
    "rope_local": "qknorm_scaled_rope",
    "attention_local": "local_gated_attention_train",
    "attention_global": "global_gated_attention",
    "output_gate_local": "local_gated_attention_train",
    "output_gate_global": "global_gated_attention",
    "swiglu": "swiglu_ffn",
    "lm_head": "lm_head_softcap_logprob",
    "logit_softcap": "lm_head_softcap_logprob",
    "logprob": "lm_head_softcap_logprob",
}

NODE_KINDS = tuple(NODE_KIND_ROWS)

# Top-level nodes in execution order; the embedding precedes the layers.
_TOP_LEVEL_KINDS = {
    "embedding": "normalized_embedding",
    "final_norm": "final_rms_norm",
    "lm_head": "lm_head",
    "logit_softcap": "logit_softcap",
    "logprob": "logprob",
}

_GEMMS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_gate_proj",
    "o_proj",
    "mlp_gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TraceNode:
    name: str
    kind: str
    row: str
    layer: int | None


@dataclass(frozen=True)
class MuseGlimmerSpec:
    """Pinned official Muse-Glimmer-30B text-model identity. Shrinking any field is forbidden."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_window: int
    vocab_size: int
    padded_vocab_size: int
    canonical_tp: int
    max_position_embeddings: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    rope_theta: float
    layer_types: tuple[str, ...]
    workload_id: str
    model_id: str
    revision: str
    weight_content_hash: str

    @classmethod
    def from_manifest(cls, manifest: MuseGlimmerManifest | None = None) -> MuseGlimmerSpec:
        m = manifest if manifest is not None else load_manifest()
        ident = m.model_identity
        text = m.text_config
        for key, expected in OFFICIAL_FINGERPRINT.items():
            if text.get(key) != expected:
                raise ValueError(
                    f"museglimmer_operator_trace forbids architecture shrink/drift: text_config "
                    f"{key}={text.get(key)!r} != official {expected!r}"
                )
        layer_types = tuple(str(t) for t in text.get("layer_types") or ())
        if layer_types != tuple(expected_layer_types()):
            raise ValueError(
                "museglimmer_operator_trace forbids architecture shrink/drift: layer_types "
                "deviates from the official 3x sliding + 1x full schedule"
            )
        layout = m.raw["parallel_layout"]
        if layout != OFFICIAL_PARALLEL_LAYOUT:
            raise ValueError("museglimmer_operator_trace: parallel_layout differs from the pin")
        return cls(
            num_hidden_layers=int(text["num_hidden_layers"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            sliding_window=int(text["sliding_window"]),
            vocab_size=int(text["vocab_size"]),
            padded_vocab_size=int(layout["padded_vocab_size"]),
            canonical_tp=int(layout["canonical_tp"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            post_norm_eps=float(text["post_norm_eps"]),
            qk_scale_factor=float(text["qk_scale_factor"]),
            output_multiplier=float(text["output_multiplier"]),
            final_logit_softcapping=float(text["final_logit_softcapping"]),
            rope_theta=float(text["rope_parameters"]["rope_theta"]),
            layer_types=layer_types,
            workload_id=str(m.workload_id),
            model_id=str(ident["model_id"]),
            revision=str(ident["revision"]),
            weight_content_hash=str(ident["weight_snapshot"]["content_hash"]),
        )

    def layer_type(self, index: int) -> str:
        return self.layer_types[index]

    def is_local(self, index: int) -> bool:
        return self.layer_types[index] == SLIDING_ATTENTION

    def gemm_shapes(self) -> dict[str, tuple[int, int]]:
        """(N, K) of every per-layer GEMM, as qualified by dense_gemm_qualification."""
        h, f = self.hidden_size, self.intermediate_size
        q = self.num_attention_heads * self.head_dim
        kv = self.num_key_value_heads * self.head_dim
        return {
            "q_proj": (q, h),
            "k_proj": (kv, h),
            "v_proj": (kv, h),
            "attn_gate_proj": (q, h),
            "o_proj": (h, q),
            "mlp_gate_proj": (f, h),
            "up_proj": (f, h),
            "down_proj": (h, f),
        }

    def trace(self) -> tuple[TraceNode, ...]:
        nodes = [_node("embedding", "normalized_embedding", None)]
        for index in range(self.num_hidden_layers):
            for suffix in _layer_node_suffixes(self.is_local(index)):
                nodes.append(
                    _node(f"layers.{index}.{suffix}", self._layer_kind(index, suffix), index)
                )
        nodes.extend(
            _node(name, kind, None)
            for name, kind in _TOP_LEVEL_KINDS.items()
            if name != "embedding"
        )
        return tuple(nodes)

    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.trace())

    def node_kind(self, node_name: str) -> str:
        parts = node_name.split(".")
        if len(parts) == 3 and parts[0] == "layers" and parts[1].isdigit():
            index = int(parts[1])
            if index >= self.num_hidden_layers or parts[2] not in _layer_node_suffixes(
                self.is_local(index)
            ):
                raise KeyError(f"unknown node {node_name!r}")
            return self._layer_kind(index, parts[2])
        if node_name in _TOP_LEVEL_KINDS:
            return _TOP_LEVEL_KINDS[node_name]
        raise KeyError(f"unknown node {node_name!r}")

    def _layer_kind(self, index: int, suffix: str) -> str:
        scope = "local" if self.is_local(index) else "global"
        if suffix.endswith("layernorm"):
            return "centered_rms_norm"
        if suffix in _GEMMS:
            return "det_gemm"
        if suffix in ("q_norm", "k_norm"):
            return "qk_norm"
        if suffix in ("rope_q", "rope_k"):
            return "rope_local"
        if suffix == "attn":
            return f"attention_{scope}"
        if suffix == "output_gate":
            return f"output_gate_{scope}"
        if suffix == "swiglu":
            return "swiglu"
        if suffix in ("residual_attn", "residual_mlp"):
            return "residual_add"
        raise KeyError(f"unknown node layers.{index}.{suffix}")

    def row_layers(self) -> dict[str, Any]:
        """Which layers each attention row covers."""
        return {
            "local_gated_attention_train": [
                i for i in range(self.num_hidden_layers) if self.is_local(i)
            ],
            "global_gated_attention": [
                i for i in range(self.num_hidden_layers) if not self.is_local(i)
            ],
        }


def _node(name: str, kind: str, layer: int | None) -> TraceNode:
    return TraceNode(name=name, kind=kind, row=NODE_KIND_ROWS[kind], layer=layer)


def _layer_node_suffixes(local: bool) -> tuple[str, ...]:
    rope = ("rope_q", "rope_k") if local else ()
    return (
        "input_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_gate_proj",
        "q_norm",
        "k_norm",
        *rope,
        "attn",
        "output_gate",
        "o_proj",
        "post_attention_layernorm",
        "residual_attn",
        "pre_feedforward_layernorm",
        "mlp_gate_proj",
        "up_proj",
        "swiglu",
        "down_proj",
        "post_feedforward_layernorm",
        "residual_mlp",
    )


__all__ = [
    "FULL_ATTENTION",
    "MuseGlimmerSpec",
    "NODE_KINDS",
    "NODE_KIND_ROWS",
    "SLIDING_ATTENTION",
    "TraceNode",
]
