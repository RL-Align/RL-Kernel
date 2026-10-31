# SPDX-License-Identifier: Apache-2.0
"""Independent CPU/FP64 test oracle, never a production fallback."""

import torch

from rl_engine.contracts.operators.kv_compressor_c128 import C128_BACKEND_ID


def reference(x, ape, norm, groups, cos, sin, eps):
    """Explicit stable exp/pool/sqrt/interleaved rotation, independent of mLite."""
    shape = (groups.numel(), 128, 1024)
    v = x.reshape(shape)
    gates = v[..., 512:] + ape
    shifted = gates - gates.amax(1, keepdim=True)
    unnormalized = shifted.exp()
    weighted = v[..., :512] * (unnormalized / unnormalized.sum(1, keepdim=True))
    pool = weighted.sum(1)
    normalized = (pool / (pool.square().sum(-1, keepdim=True) / 512 + eps).sqrt()) * norm
    co, si = (
        cos.index_select(0, groups * 128).to(x.dtype),
        sin.index_select(0, groups * 128).to(x.dtype),
    )
    tail = normalized[:, -64:].reshape(-1, 32, 2)
    rotated = torch.stack(
        (tail[..., 0] * co - tail[..., 1] * si, tail[..., 0] * si + tail[..., 1] * co), -1
    ).flatten(1)
    return torch.cat((normalized[:, :-64], rotated), -1)


def fixture(
    rows=256, *, request=7, start=0, seed=18, device="cpu", requires_grad=False, table_rows=None
):
    gen = torch.Generator().manual_seed(seed)
    values = (
        torch.randn((rows, 1024), generator=gen).bfloat16().to(device).requires_grad_(requires_grad)
    )
    ape = (torch.randn((128, 512), generator=gen) * 0.1).to(device).requires_grad_(requires_grad)
    norm = (torch.randn((512,), generator=gen) * 0.1 + 1).to(device).requires_grad_(requires_grad)
    positions = torch.arange(start, start + rows, dtype=torch.int64, device=device)
    ids = torch.stack((torch.full_like(positions, request), positions), -1)
    length = max(1, start + rows) if table_rows is None else table_rows
    angles = torch.arange(length, dtype=torch.float32).unsqueeze(1) * (
        10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    )
    cos, sin = angles.cos().bfloat16().to(device), angles.sin().bfloat16().to(device)
    return values, ids, ape, norm, cos, sin


def byte_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class ReferenceC128Backend:
    backend_id = C128_BACKEND_ID
    device_type = "cpu"

    def __init__(self):
        self.calls = 0

    def forward(self, bound, ape, norm, cos, sin, contract):
        self.calls += 1
        return reference(
            bound.values.float(),
            ape,
            norm,
            bound.group_ids[:, 1],
            cos,
            sin,
            contract.rms_eps,
        ).bfloat16()

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        xs, aa, nn = [], [], []
        with torch.enable_grad():
            for i in range(len(bound.logical_keys)):
                x = bound.values[i * 128 : (i + 1) * 128].float().detach().requires_grad_()
                a, n = ape.detach().requires_grad_(), norm.detach().requires_grad_()
                y = reference(x, a, n, bound.group_ids[i : i + 1, 1], cos, sin, contract.rms_eps)
                dx, da, dn = torch.autograd.grad(y, (x, a, n), grad[i : i + 1].float())
                xs.append(dx)
                aa.append(da)
                nn.append(dn)
        return torch.cat(xs), torch.stack(aa), torch.stack(nn)
