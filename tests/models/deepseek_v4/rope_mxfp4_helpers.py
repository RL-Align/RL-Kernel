# SPDX-License-Identifier: Apache-2.0
"""Test-only oracles for items 11/15; never a production fallback."""

import torch

from rl_engine.contracts.operators.indexer_q_preprocess import INDEXER_Q_BACKEND
from rl_engine.contracts.operators.partial_rope import PARTIAL_ROPE_BACKEND
from .indexer_c4_helpers import hadamard64
from .o_projection_helpers import rotation

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


class ReferenceRoPEBackend:
    backend_id = PARTIAL_ROPE_BACKEND
    device_type = "cpu"

    @staticmethod
    def plan(contract):
        return {"reference": "fp64"}

    def rotate(self, x, cos, sin):
        return rotation(x, cos, sin, inverse=False).bfloat16()

    def rotate_vjp(self, grad, cos, sin):
        return rotation(grad, cos, sin, inverse=True).bfloat16()


class ReferenceIndexerQBackend(ReferenceRoPEBackend):
    backend_id = INDEXER_Q_BACKEND

    def forward(self, q, cos, sin, contract):
        values = (self.rotate(q, cos, sin).double() @ hadamard64(q.device).T).bfloat16()
        packed, scales = mxfp4_even(values)
        return values, packed, scales

    def vjp(self, grad, cos, sin):
        return self.rotate_vjp((grad.double() @ hadamard64(grad.device)).bfloat16(), cos, sin)


def angles(positions, *, dtype=torch.bfloat16, device="cpu"):
    table = torch.arange(positions, device=device).double()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2, device=device).double() / 64)
    )
    return table.cos().to(dtype), table.sin().to(dtype)


def selected(cache, positions):
    return cache.index_select(0, positions.clamp_min(0)).float().reshape(-1, 1, 1, 32)


def rope_fp64(x, positions, cos, sin, *, inverse=False):
    keep = (positions >= 0)[:, None, None]
    out = rotation(x, selected(cos, positions), selected(sin, positions), inverse=inverse)
    return torch.where(keep, out, 0.0)


def mxfp4_even(values):
    """#624's rule: E8M0 = floor(log2(round_up_mantissa(amax))) - 2; E2M1 RNE; low nibble even."""
    x = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).cpu()
    bits = (amax.view(torch.int32).long() + 0x200000) & 0x7F800000
    exponent = ((bits >> 23) - 127 - 2).clamp(-127, 127)
    scaled = x.cpu() * torch.exp2(-exponent.float())
    sign = (scaled.view(torch.int32) < 0).long() << 3
    magnitude = scaled.abs().double()
    middle = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
    code = torch.searchsorted(middle, magnitude.contiguous())
    tie = (code < 7) & (magnitude == middle[code.clamp(max=6)]) & (code % 2 == 1)
    codes = ((code + tie.long()) | sign).reshape(values.shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed.to(values.device), (exponent + 127).squeeze(-1).to(torch.uint8).to(values.device)


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
