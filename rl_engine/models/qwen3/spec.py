# SPDX-License-Identifier: Apache-2.0
"""Pinned Qwen3 Dense topology and operator inventory."""
from __future__ import annotations

from dataclasses import dataclass

from rl_engine.config.workload import WS1Manifest, load_manifest

OFFICIAL_FINGERPRINT = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "swiglu": True,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "qk_norm": True,
}


NODE_KINDS = (
    "embedding",
    "rms_norm",
    "det_gemm",
    "qk_norm",
    "rope",
    "attention",
    "swiglu",
    "lm_head",
    "logprob",
)


@dataclass(frozen=True)
class Qwen3DenseSpec:
    """Pinned official Qwen3-8B Dense identity. Shrinking any field is forbidden."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    swiglu: bool
    tie_word_embeddings: bool
    attention_bias: bool
    qk_norm: bool
    workload_id: str
    model_id: str
    revision: str
    weight_content_hash: str
    weight_index_file: str
    weight_index_sha256: str
    weight_shards: tuple[tuple[str, str, int], ...]

    @classmethod
    def from_manifest(cls, manifest: WS1Manifest | None = None) -> Qwen3DenseSpec:
        m = manifest if manifest is not None else load_manifest()
        ident = m.model_identity
        fp = ident["config_fingerprint"]
        for key, expected in OFFICIAL_FINGERPRINT.items():
            got = fp.get(key)
            if got != expected:
                raise ValueError(
                    f"C9 forbids architecture shrink/drift: fingerprint {key}={got!r} "
                    f"!= official {expected!r}"
                )
        snap = ident["weight_snapshot"]
        shards = tuple(
            (
                str(item["filename"]),
                str(item["sha256"]),
                int(item["size_bytes"]),
            )
            for item in snap["shards"]
        )
        return cls(
            num_hidden_layers=int(fp["num_hidden_layers"]),
            hidden_size=int(fp["hidden_size"]),
            intermediate_size=int(fp["intermediate_size"]),
            num_attention_heads=int(fp["num_attention_heads"]),
            num_key_value_heads=int(fp["num_key_value_heads"]),
            head_dim=int(fp["head_dim"]),
            vocab_size=int(fp["vocab_size"]),
            max_position_embeddings=int(fp["max_position_embeddings"]),
            rope_theta=float(fp["rope_theta"]),
            rms_norm_eps=float(fp["rms_norm_eps"]),
            hidden_act=str(fp["hidden_act"]),
            swiglu=bool(fp["swiglu"]),
            tie_word_embeddings=bool(fp["tie_word_embeddings"]),
            attention_bias=bool(fp["attention_bias"]),
            qk_norm=bool(fp["qk_norm"]),
            workload_id=m.workload_id,
            model_id=str(ident["model_id"]),
            revision=str(ident["revision"]),
            weight_content_hash=str(snap["content_hash"]),
            weight_index_file=str(snap["index_file"]),
            weight_index_sha256=str(snap["index_sha256"]),
            weight_shards=shards,
        )

    def node_names(self) -> tuple[str, ...]:
        names: list[str] = ["embedding"]
        for index in range(self.num_hidden_layers):
            prefix = f"layers.{index}"
            names.extend(
                [
                    f"{prefix}.input_layernorm",
                    f"{prefix}.q_proj",
                    f"{prefix}.k_proj",
                    f"{prefix}.v_proj",
                    f"{prefix}.q_norm",
                    f"{prefix}.k_norm",
                    f"{prefix}.rope_q",
                    f"{prefix}.rope_k",
                    f"{prefix}.attn",
                    f"{prefix}.o_proj",
                    f"{prefix}.residual_attn",
                    f"{prefix}.post_attention_layernorm",
                    f"{prefix}.gate_proj",
                    f"{prefix}.up_proj",
                    f"{prefix}.swiglu",
                    f"{prefix}.down_proj",
                    f"{prefix}.residual_mlp",
                ]
            )
        names.extend(["final_layernorm", "lm_head", "logprob", "loss"])
        return tuple(names)

    def node_kind(self, node_name: str) -> str:
        if node_name == "embedding":
            return "embedding"
        if node_name in {"final_layernorm"} or node_name.endswith("layernorm"):
            return "rms_norm"
        if node_name.endswith((".q_norm", ".k_norm")):
            return "qk_norm"
        if node_name.endswith((".rope_q", ".rope_k")):
            return "rope"
        if node_name.endswith(".attn"):
            return "attention"
        if node_name.endswith(".swiglu"):
            return "swiglu"
        if node_name.endswith(
            (
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
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
