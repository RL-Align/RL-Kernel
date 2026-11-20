# SPDX-License-Identifier: Apache-2.0
"""V4.1 source-owned text attention. Library GEMMs plus explicit ROCm kernels.

Caches retain their autograd edges; source ownership is checked on every use.
Indexer selection is discrete, as in the released inference implementation.
"""

import math
import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention
from rl_engine.backends.rocm.attention.v41_transforms import fp4_roundtrip, fp8_roundtrip, rotary
from rl_engine.backends.rocm.norm.v41_rmsnorm import rmsnorm
from .compressor import Compressor
from .loading import projection, grouped_weight
from .parallel import AttentionTP


def frequencies(config, ratio, positions, batch):
    t = config.text
    rd = t["qk_rope_head_dim"]
    base = t["compress_rope_theta"] if ratio else t["rope_theta"]
    freq = 1.0 / (
        base ** (torch.arange(0, rd, 2, device=positions.device, dtype=torch.float32) / rd)
    )
    if ratio:
        r = t["rope_scaling"]

        def corrected(turns):
            return (
                rd
                * math.log(r["original_max_position_embeddings"] / (turns * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected(r["beta_fast"])), 0)
        high = min(math.ceil(corrected(r["beta_slow"])), rd - 1)
        smooth = 1 - (
            (torch.arange(rd // 2, device=positions.device, dtype=torch.float32) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        freq = freq / r["factor"] * (1 - smooth) + freq * smooth
    angle = positions.float()[:, None] * freq[None, :]
    return tuple(v[None].expand(batch, -1, -1).contiguous() for v in (angle.cos(), angle.sin()))


def candidate_blocks(scores, lengths, topk, block):
    width = scores.shape[-1]
    blocks = (
        F.pad(scores, (0, -width % block), value=-torch.inf).unflatten(-1, (-1, block)).amax(-1)
    )
    last = (lengths - 1) // block
    blocks = blocks.masked_fill(
        torch.arange(blocks.shape[-1], device=scores.device) == last, torch.inf
    )
    values, indices = blocks.topk(min(topk, blocks.shape[-1]), dim=-1)
    return (
        torch.zeros_like(blocks, dtype=torch.bool)
        .scatter_(-1, indices, values > -torch.inf)
        .repeat_interleave(block, -1)[..., :width]
    )


class Attention(torch.nn.Module):
    def __init__(self, config, checkpoint, layer_id, device, tp=None):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.t = config.text
        self.tp = tp or AttentionTP()
        self.plan = config.layer_plan(layer_id)
        prefix = f"layers.{layer_id}.attn"
        for name in ("wq_a", "wq_b", "wkv", "wo_b"):
            setattr(
                self,
                name,
                projection(
                    checkpoint,
                    prefix + "." + name,
                    device,
                    tp=self.tp if name in ("wq_b", "wo_b") else None,
                    shard_dim=1 if name == "wo_b" else 0,
                ),
            )
        for name, key in (
            ("q_gain", "q_norm.weight"),
            ("kv_gain", "kv_norm.weight"),
            ("sink", "attn_sink"),
        ):
            weight = checkpoint.tensor(prefix + "." + key, device=device)
            if name == "sink":
                weight = self.tp.shard(weight)
            setattr(self, name, torch.nn.Parameter(weight))
        self.wo_a = torch.nn.Parameter(
            self.tp.shard(grouped_weight(checkpoint, prefix + ".wo_a", device))
        )
        if self.plan.owns_kv:
            c = prefix + ".compressor"
            self.compressor = Compressor(
                config,
                layer_id,
                checkpoint.tensor(c + ".wkv.weight", device=device),
                checkpoint.tensor(c + ".norm.weight", device=device),
                checkpoint.tensor(c + ".wgate.weight", device=device)
                if self.plan.compress_ratio == 2
                else None,
            )
        if self.plan.owns_index:
            c = prefix + ".indexer"
            self.index_q = projection(checkpoint, c + ".wq_b", device, tp=self.tp)
            self.index_weights = projection(checkpoint, c + ".weights_proj", device, tp=self.tp)
            if self.plan.owns_kv:
                self.index_k = projection(checkpoint, c + ".wk", device)
                self.index_gain = torch.nn.Parameter(
                    checkpoint.tensor(c + ".k_norm.weight", device=device)
                )
        self.sparse = NativeSparseAttention()
        self.reset()

    def reset(self):
        self.window = None
        self.compressed = None
        self.index_keys = None
        if hasattr(self, "compressor"):
            self.compressor.reset()

    def forward(self, x, start_pos, state, trace):
        b, s, _ = x.shape
        t = self.t
        dev = x.device
        p = self.plan
        freqs = frequencies(
            self.config, p.compress_ratio, torch.arange(start_pos, start_pos + s, device=dev), b
        )
        qr = self.tp.copy(rmsnorm(self.wq_a(x), self.q_gain, eps=t["rms_norm_eps"]))
        q = rotary(
            self.wq_b(qr).unflatten(-1, (t["num_attention_heads"] // self.tp.size, t["head_dim"])),
            *freqs,
        )
        raw = fp8_roundtrip(
            rotary(rmsnorm(self.wkv(x), self.kv_gain, eps=t["rms_norm_eps"]), *freqs)
        )
        win = t["sliding_window"]
        if start_pos == 0:
            self.window = torch.zeros((b, win, t["head_dim"]), device=dev, dtype=x.dtype)
            for pos in range(max(0, s - win), s):
                self.window[:, pos % win] = raw[:, pos]
            kv = raw
            end = torch.arange(s, device=dev)[:, None]
            idx = (end - win + 1).clamp_min(0) + torch.arange(min(s, win), device=dev)
            idx = torch.where(idx > end, -1, idx)
        else:
            self.window = self.window.clone()
            self.window[:, start_pos % win] = raw[:, 0]
            kv = self.window
            oldest = start_pos % win + 1
            idx = torch.cat(
                (torch.arange(oldest, win, device=dev), torch.arange(oldest, device=dev))
            )[None]
            idx = torch.where(idx > start_pos, -1, idx)
        indices = idx.int()[None].expand(b, -1, -1).contiguous()
        trace(f"layers.{self.layer_id}.attention.q", q)
        trace(f"layers.{self.layer_id}.attention.raw_kv", raw)
        if p.compress_ratio:
            ratio = p.compress_ratio
            count = (start_pos + s) // ratio
            latent = self.compressor(x, start_pos) if p.owns_kv else None
            if latent is not None:
                positions = torch.arange(latent.shape[1], device=dev) * ratio + (
                    0 if start_pos == 0 else start_pos + 1 - ratio
                )
                cf = frequencies(self.config, ratio, positions, b)
                key = fp4_roundtrip(
                    rotary(
                        rmsnorm(self.index_k(latent), self.index_gain, eps=t["rms_norm_eps"]), *cf
                    )
                )
                self.index_keys = (
                    key
                    if start_pos == 0
                    else torch.cat((self.index_keys, key), 1)
                    if self.index_keys is not None
                    else key
                )
                compressed = fp4_roundtrip(rotary(latent, *cf), group=16, e4m3_scale=True)
                self.compressed = (
                    compressed
                    if start_pos == 0
                    else torch.cat((self.compressed, compressed), 1)
                    if self.compressed is not None
                    else compressed
                )
                trace(f"layers.{self.layer_id}.attention.compressor_latent", latent)
            if p.owns_kv:
                if self.compressed is None:
                    self.compressed = x.new_empty((b, 0, t["head_dim"]))
                    self.index_keys = x.new_empty((b, 0, t["index_head_dim"]))
                state["kv"] = (self.layer_id, self.compressed)
                state["index_k"] = (self.layer_id, self.index_keys)
            if (
                state.get("kv", (None,))[0] != p.kv_source_layer_id
                or state.get("index_k", (None,))[0] != p.kv_source_layer_id
            ):
                raise RuntimeError("missing or incorrect original KV/index-key source")
            if p.owns_index:
                if count:
                    iq = fp4_roundtrip(
                        rotary(
                            self.index_q(qr).unflatten(
                                -1, (t["index_n_heads"] // self.tp.size, t["index_head_dim"])
                            ),
                            *freqs,
                        )
                    )
                    weights = self.index_weights(x) * (
                        t["index_head_dim"] ** -0.5 * t["index_n_heads"] ** -0.5
                    )
                    score = (
                        torch.einsum("bshd,btd->bsht", iq, state["index_k"][1]).relu()
                        * weights[..., None]
                    ).sum(2)
                    score = self.tp.reduce(score)
                    lengths = (
                        (torch.arange(1, s + 1, device=dev) // ratio)[:, None]
                        if start_pos == 0
                        else count
                    )
                    score = score.masked_fill(
                        torch.arange(count, device=dev) >= lengths, -torch.inf
                    )
                    if self.layer_id == t["candidate_source_layer_id"]:
                        state["candidates"] = (
                            self.layer_id,
                            candidate_blocks(
                                score,
                                lengths,
                                t["candidate_topk_blocks"],
                                t["candidate_block_size"],
                            ),
                        )
                    elif p.candidate_source_layer_id is not None:
                        if state.get("candidates", (None,))[0] != p.candidate_source_layer_id:
                            raise RuntimeError("missing original candidate source")
                        score = score.masked_fill(~state["candidates"][1], -torch.inf)
                    picked = (
                        score.topk(min(t["index_topk"], count), dim=-1, sorted=False)
                        .indices.sort(-1)
                        .values
                    )
                    ci = torch.where(picked < lengths, picked + kv.shape[1], -1).int()
                    trace(f"layers.{self.layer_id}.attention.index_scores", score)
                else:
                    ci = torch.empty((b, s, 0), device=dev, dtype=torch.int32)
                state["indices"] = (self.layer_id, ci)
            if state.get("indices", (None,))[0] != p.index_source_layer_id:
                raise RuntimeError("missing or incorrect original index source")
            kv = torch.cat((kv, state["kv"][1]), 1)
            indices = torch.cat((indices, state["indices"][1]), -1)
        trace(f"layers.{self.layer_id}.attention.indices", indices)
        kv = self.tp.copy(kv)
        out = torch.stack(
            [
                self.sparse(
                    q[i].contiguous(),
                    kv[i].contiguous(),
                    self.sink,
                    indices[i].contiguous(),
                    t["head_dim"] ** -0.5,
                )
                for i in range(b)
            ]
        )
        trace(f"layers.{self.layer_id}.attention.sparse", out)
        out = fp8_roundtrip(
            rotary(out, *freqs, inverse=True).reshape(b, s, t["o_groups"] // self.tp.size, -1)
        )
        out = torch.einsum(
            "bsgd,grd->bsgr",
            out,
            self.wo_a.reshape(t["o_groups"] // self.tp.size, t["o_lora_rank"], -1),
        )
        return self.tp.reduce(self.wo_b(out.flatten(2)))
