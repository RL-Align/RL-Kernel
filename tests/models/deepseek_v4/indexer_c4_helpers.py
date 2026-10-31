# SPDX-License-Identifier: Apache-2.0
"""Test-only independent math; production always reuses upstream providers."""

import torch

from rl_engine.contracts.operators.indexer_compressor_c4 import INDEXER_C4_BACKEND_ID


def fixture(groups=4, *, request=7, start=0, seed=20, device="cpu", requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((groups * 4, 512), generator=generator).bfloat16().to(device)
    ape = (torch.randn((4, 256), generator=generator) * 0.1).to(device)
    norm = (torch.randn((128,), generator=generator) * 0.1 + 1).to(device)
    for t in (x, ape, norm):
        t.requires_grad_(requires_grad)
    positions = torch.arange(start, start + x.shape[0], dtype=torch.int64, device=device)
    ids = torch.stack((torch.full_like(positions, request), positions), -1)
    angle = torch.arange(max(4, start + x.shape[0])).double()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2).double() / 64)
    )
    return x, ids, ape, norm, angle.cos().bfloat16().to(device), angle.sin().bfloat16().to(device)


def hadamard64(device):
    h = torch.ones((1, 1), dtype=torch.float64, device=device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / 128**0.5


def pre_rotation_reference(context, ape, norm, group, cos, sin, eps):
    # Float64 callers remain float64 all the way through, including gradients.
    x = context if context.dtype == torch.float64 else context.float()
    dim = norm.numel()
    content = torch.cat((x[:4, :dim], x[4:, dim : 2 * dim]), 0)
    gate = torch.cat((x[:4, 2 * dim : 3 * dim] + ape[:, :dim], x[4:, 3 * dim :] + ape[:, dim:]), 0)
    if group == 0:
        gate = torch.cat((torch.full_like(gate[:4], -torch.inf), gate[4:]), 0)
    weights = torch.exp(gate - gate.amax(0, keepdim=True))
    weights = weights / weights.sum(0, keepdim=True)
    pooled = (content * weights).sum(0)
    normed = pooled * torch.rsqrt(pooled.square().mean() + eps) * norm
    c, s = cos[group * 4].to(x.dtype), sin[group * 4].to(x.dtype)
    even, odd = normed[-64::2], normed[-63::2]
    return torch.cat(
        (normed[:-64], torch.stack((even * c - odd * s, odd * c + even * s), -1).flatten())
    )


def rotate_oracle(x):
    return (x.double() @ hadamard64(x.device).T).to(x.dtype)


class ReferenceIndexerC4Backend:
    backend_id = INDEXER_C4_BACKEND_ID
    device_type = "cpu"

    def forward(self, bound, ape, norm, cos, sin, contract):
        if not bound.logical_keys:
            return torch.empty((0, 128), dtype=torch.bfloat16)
        out = torch.stack(
            [
                pre_rotation_reference(ctx.float(), ape, norm, group, cos, sin, contract.rms_eps)
                for ctx, (_req, group) in zip(bound.context_values, bound.logical_keys, strict=True)
            ]
        ).bfloat16()
        return rotate_oracle(out)

    def quantize(self, values, contract):
        # CPU contract tests only. Numerical quantization is tested with the
        # actual upstream GPU providers, never this dummy layout implementation.
        fp4 = contract.precision == "mxfp4"
        return (
            torch.zeros(
                (values.shape[0], 64 if fp4 else 128),
                dtype=torch.uint8 if fp4 else torch.float8_e4m3fnuz,
            ),
            torch.ones(
                (values.shape[0], 4 if fp4 else 1), dtype=torch.uint8 if fp4 else torch.float32
            ),
        )

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        contributions, aa, nn = [], [], []
        grad = rotate_oracle(grad.bfloat16()).float()
        with torch.enable_grad():
            for i, (_req, group) in enumerate(bound.logical_keys):
                x = bound.context_values[i].float().detach().requires_grad_()
                a, n = ape.detach().requires_grad_(), norm.detach().requires_grad_()
                y = pre_rotation_reference(x, a, n, group, cos, sin, contract.rms_eps)
                dx, da, dn = torch.autograd.grad(y, (x, a, n), grad[i])
                contributions.append((bound.context_indices_host[i], dx))
                aa.append(da)
                nn.append(dn)
        return contributions, torch.stack(aa), torch.stack(nn)
