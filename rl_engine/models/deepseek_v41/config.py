# SPDX-License-Identifier: Apache-2.0
"""Parse the released composite config without silently applying V4 defaults."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
OFFICIAL_REVISION = "08c8657f02291e11b6f6b63e0e16014f8e748518"


@dataclass(frozen=True)
class LayerPlan:
    original_layer_id: int
    compress_ratio: int
    engram: bool
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    candidate_source_layer_id: int | None

    @property
    def owns_kv(self) -> bool:
        return self.kv_source_layer_id == self.original_layer_id

    @property
    def owns_index(self) -> bool:
        return self.index_source_layer_id == self.original_layer_id


class DeepSeekV41Config:
    """Validated official HF configuration, including non-text metadata.

    Layer IDs always refer to the released backbone. Selecting four layers never
    renumbers them or replaces source dependencies with fabricated empty caches.
    """

    def __init__(self, config: dict[str, Any]):
        raw = copy.deepcopy(config)
        if raw.get("model_type") != "deepseek_v41":
            raise ValueError("expected composite deepseek_v41 config, not a V4 config")
        text = raw.get("text_config", {})
        if text.get("model_type") != "deepseek_v41_text":
            raise ValueError("missing deepseek_v41_text text_config")
        required = (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "qk_rope_head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "rms_norm_eps",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "scoring_func",
            "topk_method",
            "norm_topk_prob",
            "routed_scaling_factor",
            "swiglu_limit",
            "sliding_window",
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "candidate_source_layer_id",
            "candidate_topk_blocks",
            "candidate_block_size",
            "engram_layer_ids",
            "engram_num_embeddings",
            "engram_max_ngram_size",
            "engram_vocab_size",
            "engram_n_heads",
            "engram_head_dim",
            "engram_pad_token_id",
            "engram_compressed_vocab_size",
            "hc_mult",
            "hc_eps",
            "hc_sinkhorn_iters",
            "num_nextn_predict_layers",
            "rope_theta",
            "compress_rope_theta",
            "rope_scaling",
        )
        for key in required:
            if key not in text:
                raise ValueError(f"missing text_config.{key}")
        layers = text["num_hidden_layers"]
        if not isinstance(layers, int) or layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        ratios = text["compress_ratios"]
        if len(ratios) != layers + text["num_nextn_predict_layers"]:
            raise ValueError("compress_ratios must include backbone and MTP layers")
        if any(type(r) is not int or r not in (0, 1, 2) for r in ratios):
            raise ValueError("V4.1 compression profiles are 0, 1 and 2")
        for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
            ids = text[key]
            if ids != sorted(set(ids)) or any(
                type(i) is not int or not 0 <= i < layers for i in ids
            ):
                raise ValueError(f"{key} must contain unique ordered backbone IDs")
        kv_ids = text["kv_source_layer_ids"]
        index_ids = text["index_source_layer_ids"]
        if not set(kv_ids) <= set(index_ids):
            raise ValueError("every KV source must also produce index keys")
        for layer, ratio in enumerate(ratios[:layers]):
            if not ratio:
                continue
            source = max((i for i in kv_ids if i <= layer), default=None)
            index = max((i for i in index_ids if i <= layer), default=None)
            if source is None or index is None or ratios[source] != ratio or ratios[index] != ratio:
                raise ValueError(f"layer {layer} has no compatible KV/index source")
        if len(text["engram_layer_ids"]) != len(text["engram_num_embeddings"]):
            raise ValueError("each Engram layer needs a table size")
        eps = text["rms_norm_eps"]
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        quant = raw.get("quantization_config", {})
        expected = {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
            "activation_scheme": "dynamic",
        }
        for key, value in expected.items():
            if quant.get(key) != value:
                raise ValueError(f"unsupported quantization_config.{key}: {quant.get(key)!r}")
        self._raw = raw

    @classmethod
    def from_json(cls, path: str | Path) -> DeepSeekV41Config:
        return cls(json.loads(Path(path).read_text()))

    @property
    def text(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw["text_config"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    def layer_plan(self, layer_id: int) -> LayerPlan:
        t = self._raw["text_config"]
        if type(layer_id) is not int or not 0 <= layer_id < t["num_hidden_layers"]:
            raise ValueError(f"not a backbone layer ID: {layer_id}")
        ratio = t["compress_ratios"][layer_id]

        def source(key):
            return max((i for i in t[key] if i <= layer_id), default=None) if ratio else None

        candidate = t["candidate_source_layer_id"]
        return LayerPlan(
            layer_id,
            ratio,
            layer_id in t["engram_layer_ids"],
            source("kv_source_layer_ids"),
            source("index_source_layer_ids"),
            candidate if ratio and 0 <= candidate <= layer_id else None,
        )

    def plan_layers(self, layer_ids: tuple[int, ...]) -> tuple[LayerPlan, ...]:
        if not layer_ids or list(layer_ids) != sorted(set(layer_ids)):
            raise ValueError("select nonempty unique ascending original layer IDs")
        plans = tuple(self.layer_plan(i) for i in layer_ids)
        selected = set(layer_ids)
        for p in plans:
            for source in (
                p.kv_source_layer_id,
                p.index_source_layer_id,
                p.candidate_source_layer_id,
            ):
                if source is not None and source not in selected:
                    raise ValueError(f"layer {p.original_layer_id} requires source layer {source}")
        return plans

    def validate_topology(
        self,
        *,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        ep: int = 1,
        cp: int = 1,
        world_size: int = 1,
    ) -> None:
        """Validate arithmetic only; this does not advertise execution support."""
        if any(type(n) is not int or n <= 0 for n in (tp, pp, dp, ep, cp, world_size)):
            raise ValueError("parallel sizes must be positive integers")
        if tp * pp * dp * cp != world_size or dp % ep:
            raise ValueError("world_size must be TP*PP*DP*CP and EP must divide DP")
        t = self._raw["text_config"]
        for key in ("num_attention_heads", "index_n_heads", "o_groups", "n_routed_experts"):
            if t[key] % tp:
                raise ValueError(f"TP{tp} does not divide {key}")
        if t["n_routed_experts"] % ep:
            raise ValueError("EP must divide expert count")
