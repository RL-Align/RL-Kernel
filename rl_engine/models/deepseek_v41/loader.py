# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4.1-Flash checkpoint layout (``deepseek-ai/DeepSeek-V4.1-Flash``).

Formats, as stored in the release (safetensors headers at ``314113f8``):

- Dense linears (attention ``wq_a`` / ``wkv`` / ``wq_b`` / ``wo_a`` / ``wo_b``, the shared
  expert, the indexer's ``wq_b``, Engram's ``wkv``): MXFP8, ``<name>.weight`` E4M3 ``[N, K]``
  + ``<name>.scale`` E8M0 ``[N/32, K/32]`` (one power of two per 32 x 32 block). Under
  design decision D12 they are dequantized once to BF16 (:func:`mxfp8_dequant`, exact).
- Routed experts: MXFP4 as in DSv4, ``<name>.weight`` int8 ``[N, K/2]`` + E8M0 ``[N, K/32]``.
- Engram tables: ``engram.embed.weight`` E4M3 ``[rows, 256]`` + E8M0 ``[rows, 8]`` (MXFP8
  rows, one scale per 32 values; ``engram_table_device="mmap"`` maps them from the shards);
  ``engram.q_weight`` / ``k_weight`` BF16 ``[4, H]``.
- Compressor ``wkv`` / ``wgate``, indexer ``wk`` / ``weights_proj``, router gate, norms,
  ``embed`` / ``head``: BF16. ``attn_sink``, the router bias and the mHC parameters: FP32.

Layer-dependent tensors follow the topology (:meth:`Dsv41Config.layer_role`): a KV source
has the compressor (``wgate`` only at ratio > 1) and the indexer's ``wk`` / ``k_norm``; an
index source has the indexer's ``wq_b`` / ``weights_proj``; window-only and reuse layers
have neither. Not loaded (D9, D10): ``mtp.*`` (DSpark), ``vision.*``, ``aligner.*``,
``image_*`` and the router's ``bias_vl``. There is no ``hc_head`` and no compressor ``ape``.
"""

from __future__ import annotations

import json
import re

import torch
from torch import Tensor

from rl_engine.models.deepseek_v41.config import FULL, REINDEX, Dsv41Config

E8M0 = torch.float8_e8m0fnu
E4M3 = torch.float8_e4m3fn
MX_BLOCK = 32
FP8_MAX = 448.0


def _mxfp8(spec: dict, name: str, n: int, k: int) -> None:
    spec[f"{name}.weight"] = ((n, k), E4M3)
    spec[f"{name}.scale"] = ((-(-n // MX_BLOCK), -(-k // MX_BLOCK)), E8M0)


def _fp4(spec: dict, name: str, n: int, k: int) -> None:
    spec[f"{name}.weight"] = ((n, k // 2), torch.int8)
    spec[f"{name}.scale"] = ((n, k // 32), E8M0)


def checkpoint_spec(
    c: Dsv41Config, layers: int | None = None, experts: int | None = None
) -> dict[str, tuple]:
    """``{name: (shape, dtype)}`` of every tensor the model reads, for the first ``layers``
    layers (all by default) and the first ``experts`` routed experts of each (all)."""
    bf, f32 = torch.bfloat16, torch.float32
    h, hc = c.hidden_size, c.hc_mult
    mix = (2 + hc) * hc
    spec: dict[str, tuple] = {
        "embed.weight": ((c.vocab_size, h), bf),
        "head.weight": ((c.vocab_size, h), bf),
        "norm.weight": ((h,), bf),
    }
    n_layers = c.num_hidden_layers if layers is None else layers
    n_experts = c.n_routed_experts if experts is None else experts
    for i in range(n_layers):
        p, a = f"layers.{i}.", f"layers.{i}.attn."
        role = c.layer_role(i)
        spec[a + "attn_sink"] = ((c.num_attention_heads,), f32)
        spec[a + "q_norm.weight"] = ((c.q_lora_rank,), bf)
        spec[a + "kv_norm.weight"] = ((c.head_dim,), bf)
        _mxfp8(spec, a + "wq_a", c.q_lora_rank, h)
        _mxfp8(spec, a + "wkv", c.head_dim, h)
        _mxfp8(spec, a + "wq_b", c.num_attention_heads * c.head_dim, c.q_lora_rank)
        _mxfp8(spec, a + "wo_a", c.o_groups * c.o_lora_rank,
               c.num_attention_heads * c.head_dim // c.o_groups)  # fmt: skip
        _mxfp8(spec, a + "wo_b", h, c.o_groups * c.o_lora_rank)
        if role.mode == FULL:
            cp, ix, d = a + "compressor.", a + "indexer.", c.index_head_dim
            spec[cp + "norm.weight"] = ((c.head_dim,), bf)
            spec[cp + "wkv.weight"] = ((c.head_dim, h), bf)
            if role.ratio > 1:
                spec[cp + "wgate.weight"] = ((c.head_dim, h), bf)
            spec[ix + "wk.weight"] = ((d, c.head_dim), bf)
            spec[ix + "k_norm.weight"] = ((d,), bf)
        if role.mode in (FULL, REINDEX):
            ix, d = a + "indexer.", c.index_head_dim
            spec[ix + "weights_proj.weight"] = ((c.index_n_heads, h), bf)
            _mxfp8(spec, ix + "wq_b", c.index_n_heads * d, c.q_lora_rank)
        if role.engram is not None:
            e = p + "engram."
            rows = c.engram_num_embeddings[role.engram]
            cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim
            spec[e + "embed.weight"] = ((rows, c.engram_head_dim), E4M3)
            spec[e + "embed.scale"] = ((rows, c.engram_head_dim // MX_BLOCK), E8M0)
            spec[e + "q_weight"] = ((hc, h), bf)
            spec[e + "k_weight"] = ((hc, h), bf)
            _mxfp8(spec, e + "wkv", (hc + 1) * h, cols)
        spec[p + "attn_norm.weight"] = ((h,), bf)
        spec[p + "ffn_norm.weight"] = ((h,), bf)
        spec[p + "ffn.gate.weight"] = ((c.n_routed_experts, h), bf)
        spec[p + "ffn.gate.bias"] = ((c.n_routed_experts,), f32)
        f = c.moe_intermediate_size
        for e in range(n_experts):
            x = f"{p}ffn.experts.{e}."
            _fp4(spec, x + "w1", f, h)
            _fp4(spec, x + "w3", f, h)
            _fp4(spec, x + "w2", h, f)
        fs = f * c.n_shared_experts
        _mxfp8(spec, p + "ffn.shared_experts.w1", fs, h)
        _mxfp8(spec, p + "ffn.shared_experts.w3", fs, h)
        _mxfp8(spec, p + "ffn.shared_experts.w2", h, fs)
        for site in ("attn", "ffn"):
            spec[f"{p}hc_{site}_fn"] = ((mix, hc * h), f32)
            spec[f"{p}hc_{site}_base"] = ((mix,), f32)
            spec[f"{p}hc_{site}_scale"] = ((3,), f32)
    return spec


def skipped(name: str) -> bool:
    """Release tensors this model does not read (DSpark, vision, the VL router bias)."""
    return bool(
        re.match(r"(mtp|vision|aligner|image_)", name) or name.endswith(".ffn.gate.bias_vl")
    )


# --------------------------------------------------------------------------- MXFP8 (D12)
def mxfp8_dequant(q: Tensor, scale: Tensor) -> Tensor:
    """BF16 ``W`` from E4M3 codes ``[N, K]`` and E8M0 block scales ``[ceil(N/32), ceil(K/32)]``.

    Exact: an E4M3 value times a power of two is a BF16 value (BF16 has E4M3's mantissa bits
    and a wider exponent) unless the product leaves BF16's range, which a checkpoint scale
    does not do. E8M0 code 255 (NaN) gives NaN.
    """
    if q.dtype != E4M3 or scale.dtype != E8M0:
        raise TypeError(f"expected E4M3 codes and E8M0 scales, got {q.dtype} / {scale.dtype}")
    n, k = q.shape
    s = scale.float().repeat_interleave(MX_BLOCK, 0)[:n].repeat_interleave(MX_BLOCK, 1)[:, :k]
    return (q.float() * s).to(torch.bfloat16)


def mxfp8_quantize(w: Tensor) -> tuple[Tensor, Tensor]:
    """E4M3 codes and E8M0 scales of ``w`` [N, K] per 32 x 32 block: scale = the smallest
    power of two with ``amax / scale <= 448`` (amax floored at 1e-4), codes saturated."""
    n, k = w.shape
    pn, pk = -(-n // MX_BLOCK) * MX_BLOCK, -(-k // MX_BLOCK) * MX_BLOCK
    wf = torch.nn.functional.pad(w.float(), (0, pk - k, 0, pn - n))
    blocks = wf.view(pn // MX_BLOCK, MX_BLOCK, pk // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    exp = torch.ceil(torch.log2(amax / FP8_MAX))
    scale = torch.exp2(exp)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    return q.view(pn, pk)[:n, :k].contiguous(), scale.to(E8M0)


def _mxfp8_rows(w: Tensor) -> tuple[Tensor, Tensor]:
    """Engram rows: one E8M0 scale per 32 consecutive values of a row."""
    n, k = w.shape
    blocks = w.float().view(n, k // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(-1).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (blocks / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    return q.view(n, k), scale.to(E8M0)


# --------------------------------------------------------------------------- random weights
def random_checkpoint(
    c: Dsv41Config,
    layers: int | None = None,
    experts: int | None = None,
    seed: int = 0,
    device="cuda",
) -> dict[str, Tensor]:
    """Random tensors for every name in :func:`checkpoint_spec`, in the release's formats.

    Linears N(0, 1/sqrt(K)) quantized like the checkpoint (MXFP8 32 x 32, MXFP4); norms
    1 + N(0, 0.1); mHC ``fn`` N(0, 1/sqrt(K)); Engram tables N(0, 1) as MXFP8 rows and
    ``q`` / ``k`` 1 + N(0, 0.1); other floats N(0, 0.1).
    """
    from rl_engine.moe.mx_format import mx_quantize

    g = torch.Generator(device=device).manual_seed(seed)
    spec = checkpoint_spec(c, layers, experts)
    out: dict[str, Tensor] = {}

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    for name, (shape, dtype) in spec.items():
        if name in out or dtype == E8M0:
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if name.endswith("engram.embed.weight"):
            out[name], out[base + ".scale"] = _mxfp8_rows(randn(*shape))
        elif dtype == E4M3:
            n, k = shape
            out[name], out[base + ".scale"] = mxfp8_quantize(randn(n, k) / k**0.5)
        elif dtype == torch.int8:
            n, k = shape[0], shape[1] * 2
            mx = mx_quantize(randn(n, k) / k**0.5, "e2m1")
            out[name] = mx.codes.view(torch.int8)
            out[base + ".scale"] = mx.scales.view(E8M0)
        elif re.search(r"norm\.weight$|engram\.[qk]_weight$", name):
            out[name] = (1 + 0.1 * randn(*shape)).to(dtype)
        elif re.search(r"hc_\w+_fn$", name):
            out[name] = (randn(*shape) / shape[1] ** 0.5).to(dtype)
        elif dtype == torch.bfloat16 and len(shape) == 2:
            out[name] = (randn(*shape) / shape[1] ** 0.5).to(dtype)
        else:
            out[name] = (0.1 * randn(*shape)).to(dtype)
    return out


# --------------------------------------------------------------------------- build the model
def _attention(src, c: Dsv41Config, i: int, max_model_len: int, device, rope_cache: dict):
    from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear
    from rl_engine.models.deepseek_v4.rope import rope_cos_sin
    from rl_engine.models.deepseek_v41.attention import (
        Bf16OutputProjection,
        Dsv41Attention,
        Dsv41AttentionWeights,
        attention_configs,
    )

    a = f"layers.{i}.attn."
    ac = attention_configs(c, max_model_len)[i]

    def t(name):
        return src[a + name].to(device)

    def dq(name):
        return mxfp8_dequant(t(name + ".weight"), t(name + ".scale"))

    w = Dsv41AttentionWeights(
        fused_wqa_wkv=Bf16Linear(torch.cat([dq("wq_a"), dq("wkv")]).contiguous(), True),
        q_norm=t("q_norm.weight").contiguous(),
        kv_norm=t("kv_norm.weight").contiguous(),
        wq_b=Bf16Linear(dq("wq_b"), True),
        attn_sink=t("attn_sink").float().contiguous(),
        o_proj=Bf16OutputProjection(dq("wo_a"), Bf16Linear(dq("wo_b"), True), c.o_groups),
    )
    if ac.is_kv_source:
        comp = [t("compressor.wkv.weight")]
        if ac.ratio > 1:
            comp.append(t("compressor.wgate.weight"))
        w.comp_wkv = Bf16Linear(torch.cat(comp).to(torch.bfloat16).contiguous())
        w.comp_norm = t("compressor.norm.weight").contiguous()
        w.idx_wk = Bf16Linear(t("indexer.wk.weight").to(torch.bfloat16).contiguous())
        w.idx_k_norm = t("indexer.k_norm.weight").contiguous()
    if ac.is_index_source:
        w.idx_wq_b = Bf16Linear(dq("indexer.wq_b"), True)
        w.idx_weights_proj = t("indexer.weights_proj.weight").to(torch.bfloat16).contiguous()
    kind = 2 if ac.ratio else 1  # rope_cos_sin: 1 = plain (window only), > 1 = YaRN
    if kind not in rope_cache:
        rope_cache[kind] = rope_cos_sin(_rope_view(c, max_model_len), kind, max_model_len, device)
    return Dsv41Attention(ac, w, rope_cache[kind])


def _rope_view(c: Dsv41Config, max_model_len: int):
    """The fields DSv4's ``rope_cos_sin`` reads, with the compressed layers' YaRN parameters
    as :meth:`Dsv41Config.rope_for_layer` resolves them (``rope_parameters`` first)."""
    from types import SimpleNamespace

    compressed = [i for i in range(c.num_hidden_layers) if c.compress_ratio(i)]
    r = c.rope_for_layer(compressed[0]) if compressed else {"kind": "plain"}
    scaling = {
        "factor": r.get("factor", 1.0),
        "original_max_position_embeddings": r.get(
            "original_max_position_embeddings", max_model_len
        ),
        "beta_fast": r.get("beta_fast", 32),
        "beta_slow": r.get("beta_slow", 1),
    }
    return SimpleNamespace(
        rope_scaling=scaling,
        qk_rope_head_dim=c.qk_rope_head_dim,
        rope_theta=c.rope_theta,
        compress_rope_theta=c.compress_rope_theta,
    )


def _moe(src, c: Dsv41Config, i: int, device, ep_group, experts_limit: int | None = None):
    import torch.distributed as dist

    from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear
    from rl_engine.models.deepseek_v4.loader import _mx
    from rl_engine.models.deepseek_v4.moe import Dsv4MoE
    from rl_engine.models.deepseek_v4.router import Dsv4Router
    from rl_engine.models.deepseek_v41.moe import Bf16SharedExpert

    p = f"layers.{i}.ffn."

    def t(name):
        return src[p + name].to(device)

    router = Dsv4Router(
        t("gate.weight").contiguous(),
        bias=t("gate.bias").float().contiguous(),
        routed_scaling_factor=c.routed_scaling_factor,
    )
    world = 1 if ep_group is None else dist.get_world_size(ep_group)
    rank = 0 if ep_group is None else dist.get_rank(ep_group)
    if c.n_routed_experts % world:
        raise ValueError(f"{c.n_routed_experts} experts do not split over {world} ranks")
    local = c.n_routed_experts // world
    experts = range(rank * local, (rank + 1) * local)
    h, f = c.hidden_size, c.moe_intermediate_size
    w1 = _mx(
        [torch.cat([t(f"experts.{e}.w1.weight"), t(f"experts.{e}.w3.weight")]) for e in experts],
        [torch.cat([t(f"experts.{e}.w1.scale"), t(f"experts.{e}.w3.scale")]) for e in experts],
        2 * f,
        h,
    )
    w2 = _mx(
        [t(f"experts.{e}.w2.weight") for e in experts],
        [t(f"experts.{e}.w2.scale") for e in experts],
        h,
        f,
    )
    s = "shared_experts."

    def dq(name):
        return mxfp8_dequant(t(s + name + ".weight"), t(s + name + ".scale"))

    shared = Bf16SharedExpert(
        Bf16Linear(torch.cat([dq("w1"), dq("w3")]).contiguous(), True), Bf16Linear(dq("w2"), True)
    )
    return Dsv4MoE(router=router, w1=w1, w2=w2, shared_provider=shared, ep_group=ep_group)


_ST_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def _mmap_tensor(src, name: str) -> Tensor:
    """``name`` as a zero-copy, copy-on-write map of its safetensors shard: the pages are the
    file's (shared by every process that maps it) and only the rows a lookup touches are
    read. For the Engram tables (94 GiB each in the release)."""
    import struct

    import numpy as np

    path = str(src.path / src._where[name])
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
    begin, end = meta["data_offsets"]
    raw = np.memmap(path, dtype=np.uint8, mode="c", offset=8 + n + begin, shape=(end - begin,))
    return torch.from_numpy(raw).view(_ST_DTYPES[meta["dtype"]]).view(meta["shape"])


def load_model(
    src,
    config: Dsv41Config | None = None,
    *,
    device: str | torch.device = "cuda",
    max_model_len: int = 65536,
    num_layers: int | None = None,
    ep_group=None,
    load_head: bool = True,
    token_map: list[int] | None = None,
    engram_table_device: str | torch.device | None = None,
):
    """Build :class:`Dsv41Model` from ``src`` (a checkpoint directory or a ``name -> tensor``
    mapping). ``num_layers`` keeps the first layers; ``ep_group`` slices the routed experts.

    Engram: ``token_map`` (``engram.build_compressed_token_map``) enables the hasher; the
    tables stay on ``engram_table_device`` (default: the model's device; pass ``"cpu"`` for
    the release's 196B-value tables, or ``"mmap"`` to map them from the shards, shared by
    every process)."""
    from collections.abc import Mapping

    from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear
    from rl_engine.models.deepseek_v4.loader import SafetensorsDir
    from rl_engine.models.deepseek_v4.model import MhcParams
    from rl_engine.models.deepseek_v41.engram import EngramWeights, NgramHasher
    from rl_engine.models.deepseek_v41.model import Dsv41DecoderLayer, Dsv41Model

    if not isinstance(src, Mapping):
        if config is None:
            config = Dsv41Config.from_pretrained(src)
        src = SafetensorsDir(src)
    if config is None:
        raise ValueError("config is required when src is a mapping")
    config.check_supported()
    device = torch.device(device)
    mmap_tables = engram_table_device == "mmap"
    if mmap_tables and not isinstance(src, SafetensorsDir):
        raise ValueError('engram_table_device="mmap" needs a checkpoint directory')
    table_device = (
        device
        if engram_table_device is None or mmap_tables
        else torch.device(engram_table_device)
    )

    def table(name):
        return _mmap_tensor(src, name) if mmap_tables else src[name].to(table_device)
    n = config.num_hidden_layers if num_layers is None else num_layers

    def t(name):
        return src[name].to(device)

    def mhc(prefix: str, norm: str) -> MhcParams:
        return MhcParams(
            t(prefix + "_fn").float().contiguous(),
            t(prefix + "_scale").float().contiguous(),
            t(prefix + "_base").float().contiguous(),
            t(norm).to(torch.bfloat16).contiguous(),
        )

    rope_cache: dict = {}
    layers = []
    for i in range(n):
        role = config.layer_role(i)
        engram = None
        if role.engram is not None:
            e = f"layers.{i}.engram."
            engram = EngramWeights(
                table=table(e + "embed.weight"),
                table_scale=table(e + "embed.scale"),
                wkv=Bf16Linear(mxfp8_dequant(t(e + "wkv.weight"), t(e + "wkv.scale")), True),
                q_weight=t(e + "q_weight").to(torch.bfloat16).contiguous(),
                k_weight=t(e + "k_weight").to(torch.bfloat16).contiguous(),
                hash_index=role.engram,
            )
        layers.append(
            Dsv41DecoderLayer(
                index=i,
                attn=_attention(src, config, i, max_model_len, device, rope_cache),
                moe=_moe(src, config, i, device, ep_group),
                hc_attn=mhc(f"layers.{i}.hc_attn", f"layers.{i}.attn_norm.weight"),
                hc_ffn=mhc(f"layers.{i}.hc_ffn", f"layers.{i}.ffn_norm.weight"),
                eps=config.rms_norm_eps,
                engram=engram,
            )
        )
    hasher = None
    if token_map is not None and config.engram_layer_ids:
        hasher = NgramHasher(config, token_map, device=device)
    return Dsv41Model(
        config=config,
        embed=t("embed.weight").contiguous(),
        layers=layers,
        norm=t("norm.weight").to(torch.bfloat16).contiguous(),
        head=t("head.weight").contiguous() if load_head else None,
        max_model_len=max_model_len,
        hasher=hasher,
    )
