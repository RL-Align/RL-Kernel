# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4.1-Flash model config: the ``config.json`` fields this model reads.

Field names and their meaning follow vLLM's DSv4.1 implementation (``vllm/models/
deepseek_v41`` at ``3f86db3``); each helper below cites the vLLM code it mirrors. The
defaults are the release's (``release_config.json``: ``deepseek-ai/DeepSeek-V4.1-Flash``
``config.json`` at ``314113f8``); its ``text_config`` holds the model, the top level the
quantization config. Note ``rms_norm_eps = 1e-20`` (DeepSeek's ``norm_eps``): every RMSNorm,
the mHC controller's rms and the Engram gate use it; ``hc_eps = 1e-6`` stays. It is a
separate class from :class:`rl_engine.models.deepseek_v4.config.Dsv4Config` so DSv4's checks
stay strict, and because DSv4.1's ``compress_ratios`` mean something else: 0 is sliding
window only, 1 a full-length compressed cache, 2 a ratio-2 compressed cache (DSv4's
internal "1 = window only" does not apply).

Topology (``docs/architecture/dsv41-design.md`` 2.3): ``kv_source_layer_ids`` own a
compressor and the compressed KV / indexer K caches, ``index_source_layer_ids`` own an
indexer and publish top-k ids; a compressed layer uses the nearest source at or below it,
which must have the same ratio. ``candidate_source_layer_id`` additionally publishes the
top ``candidate_topk_blocks`` blocks of ``candidate_block_size`` positions, which later
indexers mask their logits to.

Not read (D9, D10 and the rollout-only fields): DSpark (``dspark_*``, ``n_mtp_layers``),
vision (``vision_*``), ``num_nextn_predict_layers``. ``num_hash_layers`` is read but has
no effect: vLLM's DSv4.1 MoE always passes ``num_hash_layers=0``.

:meth:`Dsv41Config.check_supported` fails closed on anything the kernel set does not
implement.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import NamedTuple

WINDOW, FULL, REINDEX, REUSE = "window", "full", "reindex", "reuse"


class LayerRole(NamedTuple):
    """What one layer's attention owns and reads."""

    ratio: int  # 0 window only, 1 or 2 compressed
    mode: str  # WINDOW, FULL (KV and index source), REINDEX (index source), REUSE
    kv_source: int | None  # layer whose compressed KV / indexer K cache this layer reads
    index_source: int | None  # layer whose top-k ids this layer reads
    candidate_source: bool  # publishes candidate blocks
    uses_candidates: bool  # its indexer masks to the candidate blocks
    engram: int | None  # index into engram_layer_ids, if an Engram is injected here


@dataclass(frozen=True)
class Dsv41Config:
    model_type: str = "deepseek_v41_text"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    compress_ratios: tuple[int, ...] = ()
    num_hash_layers: int = 0
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2304
    n_shared_experts: int = 1
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-20
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    max_position_embeddings: int = 1048576
    rope_parameters: dict = field(default_factory=dict)  # vLLM's name; flat or {main, compress}
    rope_scaling: dict = field(default_factory=dict)  # DSv4-style fallback
    expert_dtype: str = "fp4"
    quantization_config: dict = field(default_factory=dict)
    tie_word_embeddings: bool = False
    # KV / index sharing and two-level candidates
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # Engram
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_vocab_size: int = 0
    engram_compressed_vocab_size: int = 0
    engram_pad_token_id: int = 2

    _TUPLES = (
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "engram_layer_ids",
        "engram_num_embeddings",
    )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict) -> "Dsv41Config":
        """Build from a ``config.json`` dict. A nested ``text_config`` (the release layout) is
        used with the top-level ``quantization_config``; unknown keys are ignored. Raises
        ``ValueError`` on an inconsistent topology."""
        if isinstance(d.get("text_config"), dict):
            top = d
            d = dict(d["text_config"])
            if "quantization_config" not in d and "quantization_config" in top:
                d["quantization_config"] = top["quantization_config"]
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in names}
        for k in cls._TUPLES:
            if k in kw:
                kw[k] = tuple(int(v) for v in (kw[k] or ()))
        for k in ("rope_parameters", "rope_scaling", "quantization_config"):
            if kw.get(k) is None:
                kw.pop(k, None)
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path) -> "Dsv41Config":
        return cls.from_dict(json.loads((pathlib.Path(path) / "config.json").read_text()))

    # ------------------------------------------------------------------ topology
    def validate(self) -> None:
        """The topology rules of vLLM's ``nvidia/pipeline.py`` and ``attention.py``."""
        n = self.num_hidden_layers
        r = self.compress_ratios
        errors = []
        if len(r) < n or any(x not in (0, 1, 2) for x in r[:n]):
            errors.append(f"compress_ratios must give 0, 1 or 2 for all {n} layers")
        else:
            for name in ("kv_source_layer_ids", "index_source_layer_ids"):
                s = getattr(self, name)
                if s != tuple(sorted(set(s))) or any(not 0 <= x < n or r[x] == 0 for x in s):
                    errors.append(f"{name} must be sorted, unique, in range, compressed layers")
            if not set(self.kv_source_layer_ids) <= set(self.index_source_layer_ids):
                errors.append("KV sources must also be index sources")
            if not errors:
                for layer in range(n):
                    if r[layer] == 0:
                        continue
                    for kind in ("kv", "index"):
                        src = self._source(kind, layer)
                        if src is None:
                            errors.append(f"layer {layer} has no preceding {kind} source")
                        elif r[src] != r[layer]:
                            errors.append(
                                f"layer {layer} and its {kind} source {src} disagree on ratio"
                            )
            c = self.candidate_source_layer_id
            if self.candidates_enabled and c not in self.index_source_layer_ids:
                errors.append(f"candidate_source_layer_id {c} must be an index source")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            errors.append("engram_layer_ids and engram_num_embeddings differ in length")
        if any(not 0 <= x < n for x in self.engram_layer_ids):
            errors.append("engram_layer_ids out of range")
        if errors:
            raise ValueError("inconsistent DeepSeek-V4.1 config: " + "; ".join(errors))

    def _source(self, kind: str, layer: int) -> int | None:
        ids = self.kv_source_layer_ids if kind == "kv" else self.index_source_layer_ids
        preceding = [s for s in ids if s <= layer]
        return preceding[-1] if preceding else None

    @property
    def candidates_enabled(self) -> bool:
        return self.candidate_source_layer_id >= 0 and self.candidate_topk_blocks > 0

    def compress_ratio(self, layer: int) -> int:
        """0 (sliding window only), 1 or 2, as in the checkpoint."""
        return self.compress_ratios[layer]

    def hash_layer(self, layer: int) -> bool:
        """Always False: DSv4.1's MoE has no hash-routed layers."""
        return False

    def layer_role(self, layer: int) -> LayerRole:
        ratio = self.compress_ratio(layer)
        engram = self.engram_layer_ids.index(layer) if layer in self.engram_layer_ids else None
        if ratio == 0:
            return LayerRole(0, WINDOW, None, None, False, False, engram)
        is_kv = layer in self.kv_source_layer_ids
        is_index = layer in self.index_source_layer_ids
        mode = FULL if is_kv else REINDEX if is_index else REUSE
        c = self.candidate_source_layer_id
        return LayerRole(
            ratio,
            mode,
            self._source("kv", layer),
            self._source("index", layer),
            self.candidates_enabled and layer == c,
            self.candidates_enabled and is_index and 0 <= c < layer,
            engram,
        )

    def attention_config(self, layer: int, max_model_len: int):
        """The layer's ``Dsv41AttentionConfig`` (also what DSv4's ``attach_lora`` reads)."""
        from rl_engine.models.deepseek_v41.attention import attention_configs

        return attention_configs(self, max_model_len)[layer]

    def roles(self) -> list[LayerRole]:
        return [self.layer_role(i) for i in range(self.num_hidden_layers)]

    # ------------------------------------------------------------------ RoPE
    def rope_for_layer(self, layer: int) -> dict:
        """The RoPE of ``layer`` as vLLM's ``common/rope.py`` builds it: plain RoPE at
        ``rope_theta`` on window-only layers, YaRN at ``compress_rope_theta`` on compressed
        layers (ratio 1 too). Returns ``{"theta", "kind", ...}`` with ``kind`` ``"plain"``,
        ``"yarn"`` or ``"llama_scaling"`` (``attention_factor == 1``) and, unless plain,
        ``factor``, ``original_max_position_embeddings``, ``beta_fast``, ``beta_slow``."""
        ratio = self.compress_ratio(layer)
        params = self.rope_parameters or self.rope_scaling or {}
        if isinstance(params.get("main"), dict) and isinstance(params.get("compress"), dict):
            params = params["compress" if ratio > 0 else "main"]
        params = dict(params)
        rope_type = params.get("rope_type", params.get("type", "default"))
        theta = float(self.compress_rope_theta if ratio > 0 else self.rope_theta)
        if ratio == 0 or rope_type == "default":
            return {"theta": theta, "kind": "plain"}
        return {
            "theta": theta,
            "kind": "llama_scaling" if params.get("attention_factor") == 1.0 else "yarn",
            "factor": float(params.get("factor", 1.0)),
            "original_max_position_embeddings": int(
                params.get("original_max_position_embeddings", self.max_position_embeddings)
            ),
            "beta_fast": float(params.get("beta_fast", 32)),
            "beta_slow": float(params.get("beta_slow", 1)),
        }

    # ------------------------------------------------------------------ Engram
    def engram_primes(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Bucket sizes ``[layer][n-gram size - 2][head]`` (vLLM ``EngramLayout``): primes
        above ``engram_vocab_size - 1``, drawn in order and never reused across layers."""
        seen: set[int] = set()
        layers = []
        for _ in self.engram_layer_ids:
            per_ngram = []
            for _ in range(self.engram_max_ngram_size - 1):
                sizes, current = [], self.engram_vocab_size - 1
                for _ in range(self.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            layers.append(tuple(per_ngram))
        return tuple(layers)

    # ------------------------------------------------------------------ support
    def check_supported(self) -> None:
        """Raise ``ValueError`` on anything the DSv4.1 kernel set does not implement."""
        from rl_engine.kernels.ops.cuda.router.dsv4_router import TOP_K
        from rl_engine.kernels.ops.cuda.topk.stable_topk import MAX_TOP_K
        from rl_engine.kernels.ops.pytorch.mhc import HC_EPS, POST_MULT, SINKHORN_ITERS

        q = self.quantization_config or {}
        checks = [
            ("hc_mult", self.hc_mult, 4),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters, SINKHORN_ITERS),
            ("hc_eps", self.hc_eps, HC_EPS),
            ("num_experts_per_tok", self.num_experts_per_tok, TOP_K),
            ("scoring_func", self.scoring_func, "sqrtsoftplus"),
            ("swiglu_limit", float(self.swiglu_limit), 10.0),
            ("n_shared_experts", self.n_shared_experts, 1),
            ("expert_dtype", q.get("expert_dtype", self.expert_dtype), "fp4"),
            ("num_attention_heads", self.num_attention_heads, 64),
            ("head_dim", self.head_dim, 512),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 64),
            ("index_head_dim", self.index_head_dim, 128),
            ("sliding_window", self.sliding_window, 128),
            ("quant_method", q.get("quant_method"), "fp8"),
            ("quant fmt", q.get("fmt", "e4m3"), "e4m3"),
            ("quant scale_fmt", q.get("scale_fmt"), "ue8m0"),
            ("quant weight_block_size", list(q.get("weight_block_size", [])), [32, 32]),
        ]
        bad = [f"{n}={v!r} (expected {e!r})" for n, v, e in checks if v != e]
        if self.model_type not in ("deepseek_v41", "deepseek_v41_text"):
            bad.append(f"model_type={self.model_type!r}")
        if not self.rms_norm_eps > 0:
            bad.append(f"rms_norm_eps={self.rms_norm_eps} (must be positive)")
        if self.hidden_size > 5120 or self.hidden_size % 128:
            bad.append(f"hidden_size={self.hidden_size} (mHC / Engram: a multiple of 128, <= 5120)")
        if self.n_routed_experts not in (256, 384):
            bad.append(f"n_routed_experts={self.n_routed_experts} (the router takes 256 or 384)")
        if self.index_topk > MAX_TOP_K:
            bad.append(f"index_topk={self.index_topk} (stable_topk takes <= {MAX_TOP_K})")
        if POST_MULT != 2.0:
            bad.append(f"mHC post multiplier {POST_MULT}")
        for layer in range(self.num_hidden_layers):
            rope = self.rope_for_layer(layer)
            if rope["kind"] == "llama_scaling":
                bad.append(
                    f"layer {layer}: llama-scaling RoPE (attention_factor 1) not implemented"
                )
                break
        if self.candidates_enabled:
            if self.candidate_topk_blocks > MAX_TOP_K:
                bad.append(f"candidate_topk_blocks={self.candidate_topk_blocks} (> {MAX_TOP_K})")
            if self.candidate_block_size <= 0:
                bad.append("candidate_block_size must be positive")
            c = self.candidate_source_layer_id
            for layer, role in enumerate(self.roles()):
                if role.uses_candidates and role.kv_source != self._source("kv", c):
                    bad.append(f"layer {layer} uses candidates from layer {c} of another KV group")
        if self.engram_layer_ids:
            bad += self._check_engram()
        if bad:
            raise ValueError("unsupported DeepSeek-V4.1 config: " + "; ".join(bad))

    def _check_engram(self) -> list[str]:
        q = self.quantization_config or {}
        bad = []
        if q.get("engram_embed_quant", "fp8") != "fp8":
            bad.append(f"engram_embed_quant={q.get('engram_embed_quant')!r} (FP8 tables only)")
        if q.get("engram_embed_block_size", 32) != 32:
            bad.append("engram_embed_block_size must be 32")
        if q.get("engram_embed_scale_dtype", "ue8m0") != "ue8m0":
            bad.append("engram_embed_scale_dtype must be ue8m0")
        for name in (
            "engram_max_ngram_size",
            "engram_n_heads",
            "engram_head_dim",
            "engram_vocab_size",
            "engram_compressed_vocab_size",
        ):
            if getattr(self, name) <= 0:
                bad.append(f"{name} must be set")
        if self.engram_max_ngram_size < 2:
            bad.append("engram_max_ngram_size must be >= 2")
        if not bad:
            for i, primes in enumerate(self.engram_primes()):
                rows = sum(p for per in primes for p in per)
                if self.engram_num_embeddings[i] < rows:
                    bad.append(
                        f"engram_num_embeddings[{i}]={self.engram_num_embeddings[i]} < "
                        f"{rows} bucket rows"
                    )
        return bad


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (vLLM ``common/engram.py``)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


# --------------------------------------------------------------------------- release
RELEASE_CONFIG = pathlib.Path(__file__).with_name("release_config.json")


def release_config() -> dict:
    """The release ``config.json`` (``deepseek-ai/DeepSeek-V4.1-Flash`` at ``314113f8``)."""
    return json.loads(RELEASE_CONFIG.read_text())


def toy_config(vocab_size: int = 129280, engram_vocab_size: int = 1009) -> dict:
    """The release's dimensions with a short layer map holding one layer of every role
    (design doc 4.1): window, Full r2, Reuse r2, Full r1 (the candidate source), Reuse r1,
    Reindex r1, and one Engram layer; small Engram vocabularies. Flat (no ``text_config``)."""
    release = release_config()
    cfg = dict(release["text_config"])
    cfg["quantization_config"] = release["quantization_config"]
    cfg.update(
        vocab_size=vocab_size,
        num_hidden_layers=7,
        compress_ratios=[0, 2, 2, 1, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        candidate_source_layer_id=3,
        engram_layer_ids=[1],
        engram_vocab_size=engram_vocab_size,
        engram_compressed_vocab_size=min(vocab_size, cfg["engram_compressed_vocab_size"]),
        num_nextn_predict_layers=0,
    )
    for k in [k for k in cfg if k.startswith("dspark_")]:
        del cfg[k]
    probe = Dsv41Config.from_dict({**cfg, "engram_num_embeddings": [0]})
    cfg["engram_num_embeddings"] = [
        sum(p for per in layer for p in per) for layer in probe.engram_primes()
    ]
    return cfg
