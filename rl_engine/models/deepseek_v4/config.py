# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 model config (the Hugging Face ``config.json`` fields this model reads).

:meth:`Dsv4Config.check_supported` fails closed on any value the batch-invariant kernel
set does not implement (the mHC, router, shared-expert and P5 constants are fixed in
their kernels).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from rl_engine.models.deepseek_v4.attention import Dsv4AttentionConfig


@dataclass(frozen=True)
class Dsv4Config:
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    compress_ratios: tuple[int, ...] = ()
    num_hash_layers: int = 3
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict = field(default_factory=dict)
    expert_dtype: str = "fp4"
    quantization_config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Dsv4Config":
        names = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in names}
        if "compress_ratios" in kw:
            kw["compress_ratios"] = tuple(int(r) for r in kw["compress_ratios"])
        cfg = cls(**kw)
        if len(cfg.compress_ratios) < cfg.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(cfg.compress_ratios)} entries for "
                f"{cfg.num_hidden_layers} layers"
            )
        return cfg

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path) -> "Dsv4Config":
        return cls.from_dict(json.loads((pathlib.Path(path) / "config.json").read_text()))

    def compress_ratio(self, layer: int) -> int:
        """1 (sliding window only), 4 (C4 + indexer) or 128 (C128); the checkpoint's 0 is 1."""
        return max(1, self.compress_ratios[layer])

    def hash_layer(self, layer: int) -> bool:
        return layer < self.num_hash_layers

    def attention_config(self, layer: int, max_model_len: int) -> Dsv4AttentionConfig:
        return Dsv4AttentionConfig(
            hidden=self.hidden_size,
            heads=self.num_attention_heads,
            padded_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            rope_dim=self.qk_rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            window=self.sliding_window,
            compress_ratio=self.compress_ratio(layer),
            index_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            eps=self.rms_norm_eps,
            max_model_len=max_model_len,
        )

    def check_supported(self) -> None:
        """Raise ``ValueError`` on anything the kernel set does not implement."""
        from rl_engine.kernels.ops.cuda.router.dsv4_router import TOP_K
        from rl_engine.kernels.ops.pytorch.mhc_dsv4 import (
            HC_EPS,
            NORM_EPS,
            POST_MULT,
            RMS_EPS,
            SINKHORN_ITERS,
        )

        q = self.quantization_config or {}
        checks = [
            ("hc_mult", self.hc_mult, 4),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters, SINKHORN_ITERS),
            ("hc_eps", self.hc_eps, HC_EPS),
            ("rms_norm_eps (mHC)", self.rms_norm_eps, RMS_EPS),
            ("rms_norm_eps (norms)", self.rms_norm_eps, NORM_EPS),
            ("num_experts_per_tok", self.num_experts_per_tok, TOP_K),
            ("scoring_func", self.scoring_func, "sqrtsoftplus"),
            ("swiglu_limit", float(self.swiglu_limit), 10.0),
            ("n_shared_experts", self.n_shared_experts, 1),
            ("expert_dtype", self.expert_dtype, "fp4"),
            ("quant fmt", q.get("fmt"), "e4m3"),
            ("quant scale_fmt", q.get("scale_fmt"), "ue8m0"),
            ("quant weight_block_size", list(q.get("weight_block_size", [])), [128, 128]),
            ("rope_scaling type", (self.rope_scaling or {}).get("type"), "yarn"),
        ]
        bad = [f"{n}={v!r} (expected {e!r})" for n, v, e in checks if v != e]
        bad += [
            f"compress_ratios[{i}]={r}"
            for i, r in enumerate(self.compress_ratios[: self.num_hidden_layers])
            if r not in (0, 1, 4, 128)
        ]
        if self.n_routed_experts not in (256, 384):
            bad.append(
                f"n_routed_experts={self.n_routed_experts} (the router kernel takes 256 or 384)"
            )
        if POST_MULT != 2.0:
            bad.append(f"mHC post multiplier {POST_MULT}")
        if self.hidden_size > 4096 or self.hidden_size % 128:
            bad.append(f"hidden_size={self.hidden_size} (mHC needs a multiple of 128, <= 4096)")
        if bad:
            raise ValueError("unsupported DeepSeek-V4 config: " + "; ".join(bad))
