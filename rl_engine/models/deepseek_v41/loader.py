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
  rows, one scale per 32 values); ``engram.q_weight`` / ``k_weight`` BF16 ``[4, H]``.
- Compressor ``wkv`` / ``wgate``, indexer ``wk`` / ``weights_proj``, router gate, norms,
  ``embed`` / ``head``: BF16. ``attn_sink``, the router bias and the mHC parameters: FP32.

Layer-dependent tensors follow the topology (:meth:`Dsv41Config.layer_role`): a KV source
has the compressor (``wgate`` only at ratio > 1) and the indexer's ``wk`` / ``k_norm``; an
index source has the indexer's ``wq_b`` / ``weights_proj``; window-only and reuse layers
have neither. Not loaded (D9, D10): ``mtp.*`` (DSpark), ``vision.*``, ``aligner.*``,
``image_*`` and the router's ``bias_vl``. There is no ``hc_head`` and no compressor ``ape``.
"""

from __future__ import annotations

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
