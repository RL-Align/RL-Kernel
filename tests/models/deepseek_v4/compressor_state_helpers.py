# SPDX-License-Identifier: Apache-2.0
"""Independent state/cache reference; used only through explicit test injection."""

import torch

from rl_engine.contracts.operators.compressor_state import CompressorStateContract
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache


def tensors(contract, *, seed=21, device="cpu"):
    c = contract
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((c.max_sequence_tokens, c.width * 2), generator=generator).bfloat16().to(device)
    ape = (torch.randn((c.ratio, c.width), generator=generator) * 0.1).to(device)
    norm = (torch.randn(512, generator=generator) * 0.1 + 1).to(device)
    angles = torch.arange(c.max_sequence_tokens).double()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2).double() / 64)
    )
    return x, ape, norm, angles.cos().bfloat16().to(device), angles.sin().bfloat16().to(device)


def make_cache(ratio=4, *, backend=None, device="cpu", **kwargs):
    c = CompressorStateContract(ratio=ratio, **kwargs)
    x, ape, norm, cos, sin = tensors(c, device=device)
    engine = CompressorStateCache(
        c,
        ape=ape,
        norm=norm,
        cos=cos,
        sin=sin,
        version="v1",
        writer_rank=0,
        expected_arithmetic=c.arithmetic_id,
        backend=backend or ReferenceStateBackend(),
    )
    return engine, x, (ape, norm, cos, sin)


class ReferenceStateBackend:
    device_type = "cpu"
    supports_graph = False

    def __init__(self):
        self.calls = 0

    def execute(self, arena):
        self.calls += 1
        c = arena.contract
        for ragged, slot, position, historical in arena._plan.tolist():
            if position < 0:
                continue
            contents, scores = [], []
            for k in range(c.state_rows):
                absolute = position - c.state_rows + 1 + k
                col = 512 if c.ratio == 4 and k >= 4 else 0
                if absolute < 0:
                    contents.append(torch.zeros(512))
                    scores.append(torch.full((512,), -torch.inf))
                elif k < historical:
                    row = arena._state[slot, absolute % c.state_rows]
                    contents.append(row[col : col + 512])
                    scores.append(row[c.width + col : c.width + col + 512])
                else:
                    row = arena._input[ragged - (c.state_rows - 1 - k)].float()
                    contents.append(row[col : col + 512])
                    scores.append(
                        row[c.width + col : c.width + col + 512]
                        + arena._ape[k % c.ratio, col : col + 512]
                    )
            score = torch.stack(scores).double()
            weights = (score - score.amax(0, keepdim=True)).exp()
            weights /= weights.sum(0, keepdim=True)
            pooled = (torch.stack(contents).double() * weights).sum(0)
            out = (
                pooled * torch.rsqrt(pooled.square().mean() + arena.rms_eps) * arena._norm.double()
            )
            group = position // c.ratio
            cos, sin = arena._cos[group * c.ratio].double(), arena._sin[group * c.ratio].double()
            even, odd = out[-64::2], out[-63::2]
            rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten()
            out = torch.cat((out[:-64], rotated))
            block = arena._blocks[slot, group // 64]
            arena._cache[block, group % 64].copy_(out)
        for row, physical in enumerate(arena._write_slots.tolist()):
            if physical < 0:
                continue
            slot, offset = divmod(physical, c.state_rows)
            source = arena._input[row].float()
            position = int(arena._positions[row])
            arena._state[slot, offset, : c.width] = source[: c.width]
            arena._state[slot, offset, c.width :] = (
                source[c.width :] + arena._ape[position % c.ratio]
            )


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
