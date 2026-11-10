# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Build a :class:`Dsv4Model` from a DeepSeek-V4 checkpoint (Hugging Face layout).

Tensor names and formats are those of ``deepseek-ai/DeepSeek-V4-Flash``:

- FP8 linears: ``<name>.weight`` E4M3 ``[N, K]`` + ``<name>.scale`` E8M0
  ``[N/128, K/128]``.
- Routed experts: ``<name>.weight`` int8 ``[N, K/2]`` (two E2M1 per byte, low nibble =
  even column) + ``<name>.scale`` E8M0 ``[N, K/32]``.
- Everything else BF16 or FP32 as stored.

Fusions follow vLLM's ``load_weights``: ``fused_wqa_wkv = [wq_a; wkv]``, the
compressors' ``[wkv; wgate]``, the experts' ``w1 = [w1 (gate); w3 (up)]``. The
weights are used as stored (no requantization). ``checkpoint_spec`` lists every
tensor a config implies; :func:`random_checkpoint` fills it for tests.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Mapping

import torch
from torch import Tensor

from rl_engine.models.deepseek_v4.config import Dsv4Config

E8M0 = torch.float8_e8m0fnu


def _fp8(spec: dict, name: str, n: int, k: int) -> None:
    spec[f"{name}.weight"] = ((n, k), torch.float8_e4m3fn)
    spec[f"{name}.scale"] = ((-(-n // 128), -(-k // 128)), E8M0)


def _fp4(spec: dict, name: str, n: int, k: int) -> None:
    spec[f"{name}.weight"] = ((n, k // 2), torch.int8)
    spec[f"{name}.scale"] = ((n, k // 32), E8M0)


def checkpoint_spec(c: Dsv4Config, layers: int | None = None) -> dict[str, tuple]:
    """``{name: (shape, dtype)}`` of every tensor of the first ``layers`` layers + top."""
    bf, f32 = torch.bfloat16, torch.float32
    h, hc = c.hidden_size, c.hc_mult
    mix = (2 + hc) * hc
    spec: dict[str, tuple] = {
        "embed.weight": ((c.vocab_size, h), bf),
        "head.weight": ((c.vocab_size, h), bf),
        "norm.weight": ((h,), bf),
        "hc_head_fn": ((hc, hc * h), f32),
        "hc_head_base": ((hc,), f32),
        "hc_head_scale": ((1,), f32),
    }
    for i in range(c.num_hidden_layers if layers is None else layers):
        p, a = f"layers.{i}.", f"layers.{i}.attn."
        ratio = c.compress_ratio(i)
        spec[a + "attn_sink"] = ((c.num_attention_heads,), f32)
        spec[a + "q_norm.weight"] = ((c.q_lora_rank,), bf)
        spec[a + "kv_norm.weight"] = ((c.head_dim,), bf)
        _fp8(spec, a + "wq_a", c.q_lora_rank, h)
        _fp8(spec, a + "wkv", c.head_dim, h)
        _fp8(spec, a + "wq_b", c.num_attention_heads * c.head_dim, c.q_lora_rank)
        _fp8(spec, a + "wo_a", c.o_groups * c.o_lora_rank,
             c.num_attention_heads * c.head_dim // c.o_groups)  # fmt: skip
        _fp8(spec, a + "wo_b", h, c.o_groups * c.o_lora_rank)
        if ratio > 1:
            width = (2 if ratio == 4 else 1) * c.head_dim
            spec[a + "compressor.ape"] = ((ratio, width), f32)
            spec[a + "compressor.norm.weight"] = ((c.head_dim,), bf)
            spec[a + "compressor.wkv.weight"] = ((width, h), bf)
            spec[a + "compressor.wgate.weight"] = ((width, h), bf)
        if ratio == 4:
            ix, d = a + "indexer.", c.index_head_dim
            spec[ix + "compressor.ape"] = ((4, 2 * d), f32)
            spec[ix + "compressor.norm.weight"] = ((d,), bf)
            spec[ix + "compressor.wkv.weight"] = ((2 * d, h), bf)
            spec[ix + "compressor.wgate.weight"] = ((2 * d, h), bf)
            spec[ix + "weights_proj.weight"] = ((c.index_n_heads, h), bf)
            _fp8(spec, ix + "wq_b", c.index_n_heads * d, c.q_lora_rank)
        spec[p + "attn_norm.weight"] = ((h,), bf)
        spec[p + "ffn_norm.weight"] = ((h,), bf)
        spec[p + "ffn.gate.weight"] = ((c.n_routed_experts, h), bf)
        if c.hash_layer(i):
            spec[p + "ffn.gate.tid2eid"] = ((c.vocab_size, c.num_experts_per_tok), torch.int64)
        else:
            spec[p + "ffn.gate.bias"] = ((c.n_routed_experts,), f32)
        f = c.moe_intermediate_size
        for e in range(c.n_routed_experts):
            x = f"{p}ffn.experts.{e}."
            _fp4(spec, x + "w1", f, h)
            _fp4(spec, x + "w3", f, h)
            _fp4(spec, x + "w2", h, f)
        fs = f * c.n_shared_experts
        _fp8(spec, p + "ffn.shared_experts.w1", fs, h)
        _fp8(spec, p + "ffn.shared_experts.w3", fs, h)
        _fp8(spec, p + "ffn.shared_experts.w2", h, fs)
        for site in ("attn", "ffn"):
            spec[f"{p}hc_{site}_fn"] = ((mix, hc * h), f32)
            spec[f"{p}hc_{site}_base"] = ((mix,), f32)
            spec[f"{p}hc_{site}_scale"] = ((3,), f32)
    return spec


class SafetensorsDir(Mapping):
    """Lazy ``name -> tensor`` view of a sharded safetensors checkpoint directory."""

    def __init__(self, path: str | pathlib.Path, device: str | torch.device = "cpu"):
        from safetensors import safe_open

        self.path = pathlib.Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self._where = json.loads(index.read_text())["weight_map"]
        else:
            self._where = {}
            for shard in sorted(self.path.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt") as f:
                    self._where.update({k: shard.name for k in f.keys()})
        self._device = str(device)
        self._open: dict = {}

    def __getitem__(self, name: str) -> Tensor:
        from safetensors import safe_open

        shard = self._where[name]
        if shard not in self._open:
            self._open[shard] = safe_open(
                str(self.path / shard), framework="pt", device=self._device
            )
        return self._open[shard].get_tensor(name)

    def __iter__(self):
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)


def _scale(s: Tensor) -> Tensor:
    """E8M0 block scales -> FP32 (exact: every E8M0 code is a power of two)."""
    if s.dtype != E8M0:
        raise TypeError(f"expected E8M0 scales, got {s.dtype}")
    return s.float().contiguous()


def _mx(codes: list[Tensor], scales: list[Tensor], n: int, k: int):
    from rl_engine.moe.mx_format import MXTensor

    q = torch.stack(codes).view(torch.uint8).contiguous()
    s = torch.stack(scales).view(torch.uint8).contiguous()
    return MXTensor(q, s, "e2m1", (len(codes), n, k))


def _attention(src, c: Dsv4Config, i: int, max_model_len: int, device):
    from rl_engine.kernels.ops.cuda import dsv4_attention as core
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8BlockWeight, Fp8Linear
    from rl_engine.models.deepseek_v4.attention import Dsv4Attention, Dsv4AttentionWeights
    from rl_engine.models.deepseek_v4.rope import rope_cos_sin

    a = f"layers.{i}.attn."

    def t(name):
        return src[a + name].to(device)

    def fp8(name):
        return Fp8Linear.from_fp8(t(name + ".weight"), _scale(t(name + ".scale")))

    def cat(*names):
        return torch.cat([t(n) for n in names]).contiguous()

    cfg = c.attention_config(i, max_model_len)
    sink = torch.full((cfg.padded_heads,), float("-inf"), dtype=torch.float32, device=device)
    sink[: cfg.heads] = t("attn_sink").float()
    w = Dsv4AttentionWeights(
        fused_wqa_wkv=Fp8Linear.from_fp8(
            cat("wq_a.weight", "wkv.weight"), _scale(cat("wq_a.scale", "wkv.scale"))
        ),
        q_norm=t("q_norm.weight"),
        kv_norm=t("kv_norm.weight"),
        wq_b=fp8("wq_b"),
        attn_sink=sink,
        o_proj=core.OutputProjection(
            Fp8BlockWeight(t("wo_a.weight").contiguous(), _scale(t("wo_a.scale"))),
            fp8("wo_b"),
            cfg.o_groups,
        ),
    )
    if cfg.compress_ratio > 1:
        w.wkv_wgate = cat("compressor.wkv.weight", "compressor.wgate.weight")
        w.ape = t("compressor.ape").float().contiguous()
        w.comp_norm = t("compressor.norm.weight")
    if cfg.compress_ratio == 4:
        w.idx_wq_b = fp8("indexer.wq_b")
        w.idx_weights_proj = t("indexer.weights_proj.weight").contiguous()
        w.idx_wkv_wgate = cat("indexer.compressor.wkv.weight", "indexer.compressor.wgate.weight")
        w.idx_ape = t("indexer.compressor.ape").float().contiguous()
        w.idx_norm = t("indexer.compressor.norm.weight")
    cos_sin = rope_cos_sin(c, cfg.compress_ratio, max_model_len, device)
    return Dsv4Attention(cfg, w, cos_sin)


def _moe(src, c: Dsv4Config, i: int, device, ep_group):
    import torch.distributed as dist

    from rl_engine.models.deepseek_v4.moe import Dsv4MoE
    from rl_engine.models.deepseek_v4.router import Dsv4Router

    p = f"layers.{i}.ffn."

    def t(name):
        return src[p + name].to(device)

    if c.hash_layer(i):
        router = Dsv4Router(
            t("gate.weight").contiguous(),
            tid2eid=t("gate.tid2eid").contiguous(),
            routed_scaling_factor=c.routed_scaling_factor,
        )
    else:
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
    shared = (
        (
            torch.cat([t(s + "w1.weight"), t(s + "w3.weight")]).contiguous(),
            _scale(torch.cat([t(s + "w1.scale"), t(s + "w3.scale")])),
        ),
        (t(s + "w2.weight").contiguous(), _scale(t(s + "w2.scale"))),
    )
    return Dsv4MoE(router=router, w1=w1, w2=w2, shared_fp8=shared, ep_group=ep_group)


def load_model(
    src: Mapping | str | pathlib.Path,
    config: Dsv4Config | None = None,
    *,
    device: str | torch.device = "cuda",
    max_model_len: int = 65536,
    num_layers: int | None = None,
    ep_group=None,
    load_head: bool = True,
):
    """Build the model from ``src`` (a checkpoint directory or a ``name -> tensor``
    mapping). ``num_layers`` keeps only the first layers (for the layerwise gate);
    with ``ep_group`` each rank keeps its slice of the routed experts."""
    from rl_engine.models.deepseek_v4.model import Dsv4DecoderLayer, Dsv4Model, MhcParams

    if not isinstance(src, Mapping):
        if config is None:
            config = Dsv4Config.from_pretrained(src)
        src = SafetensorsDir(src)
    if config is None:
        raise ValueError("config is required when src is a mapping")
    config.check_supported()
    device = torch.device(device)
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

    layers = []
    for i in range(n):
        layers.append(
            Dsv4DecoderLayer(
                index=i,
                attn=_attention(src, config, i, max_model_len, device),
                moe=_moe(src, config, i, device, ep_group),
                hc_attn=mhc(f"layers.{i}.hc_attn", f"layers.{i}.attn_norm.weight"),
                hc_ffn=mhc(f"layers.{i}.hc_ffn", f"layers.{i}.ffn_norm.weight"),
                hash_routing=config.hash_layer(i),
            )
        )
    return Dsv4Model(
        config=config,
        embed=t("embed.weight").contiguous(),
        layers=layers,
        hc_head=mhc("hc_head", "norm.weight"),
        head=t("head.weight").contiguous() if load_head else None,
        max_model_len=max_model_len,
    )


def random_checkpoint(
    c: Dsv4Config, layers: int | None = None, seed: int = 0, device="cuda"
) -> dict[str, Tensor]:
    """Random tensors for every name in :func:`checkpoint_spec`, in checkpoint formats.

    Linears are N(0, 1/sqrt(K)) quantized like the checkpoint (FP8 128x128 UE8M0 blocks,
    MXFP4); norms 1 + N(0, 0.1); mHC ``fn`` N(0, 1 / sqrt(K)), so ``fn . x / rms(x)`` is
    O(1); other floats N(0, 0.1). Routing tables draw distinct experts per token.
    """
    from rl_engine.kernels.ops.cuda.fp8_linear import quantize_weight
    from rl_engine.moe.mx_format import mx_quantize

    g = torch.Generator(device=device).manual_seed(seed)
    spec = checkpoint_spec(c, layers)
    out: dict[str, Tensor] = {}

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device)

    for name, (shape, dtype) in spec.items():
        if name in out or dtype == E8M0:
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if dtype == torch.float8_e4m3fn:
            n, k = shape
            q = quantize_weight((randn(n, k) / k**0.5).to(torch.bfloat16))
            out[name], out[base + ".scale"] = q.q, q.scale.to(E8M0)
        elif dtype == torch.int8:
            n, k = shape[0], shape[1] * 2
            q = mx_quantize(randn(n, k) / k**0.5, "e2m1")
            out[name] = q.codes.view(torch.int8)
            out[base + ".scale"] = q.scales.view(E8M0)
        elif name.endswith("tid2eid"):
            r = torch.rand(shape[0], c.n_routed_experts, generator=g, device=device)
            out[name] = r.argsort(dim=1)[:, : shape[1]].long()
        elif re.search(r"norm\.weight$", name):
            out[name] = (1 + 0.1 * randn(*shape)).to(dtype)
        elif re.search(r"hc_\w+_fn$", name):
            out[name] = (randn(*shape) / shape[1] ** 0.5).to(dtype)
        elif dtype == torch.bfloat16 and len(shape) == 2:
            out[name] = (randn(*shape) / shape[1] ** 0.5).to(dtype)
        else:
            out[name] = (0.1 * randn(*shape)).to(dtype)
    return out
