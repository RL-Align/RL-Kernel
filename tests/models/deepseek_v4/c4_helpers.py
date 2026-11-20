# SPDX-License-Identifier: Apache-2.0
"""Independent ratio-4 overlap oracle, not a production fallback."""

import torch

from rl_engine.contracts.operators.kv_compressor_c4 import C4_BACKEND_ID


def context_reference(context, ape, norm, group, cos, sin, eps):
    if context.dtype != torch.float64:
        context = context.float()
    # Each source row is [content1024, score1024]. The overlap window uses
    # previous group's first half and current group's second half.
    previous = context[:4, :1024]
    previous_gate = context[:4, 1024:]
    current = context[4:, :1024]
    current_gate = context[4:, 1024:]
    if group == 0:
        content = torch.cat((torch.zeros_like(previous[:, :512]), current[:, 512:]), dim=0)
        gate = torch.cat(
            (
                torch.full_like(previous_gate[:, :512], float("-inf")),
                current_gate[:, 512:],
            ),
            dim=0,
        )
    else:
        content = torch.cat((previous[:, :512], current[:, 512:]), dim=0)
        gate = torch.cat((previous_gate[:, :512], current_gate[:, 512:]), dim=0)
    ape_window = torch.cat((ape[:, :512], ape[:, 512:]), dim=0)
    gate = gate + ape_window
    weights = torch.softmax(gate, dim=0)
    pooled = (content * weights).sum(0)
    pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps)
    pooled = pooled * norm.to(context.dtype)
    selected_cos = cos[group * 4].to(context.dtype)
    selected_sin = sin[group * 4].to(context.dtype)
    tail = pooled[-64:].reshape(32, 2)
    rotated = torch.stack(
        (
            tail[:, 0] * selected_cos - tail[:, 1] * selected_sin,
            tail[:, 0] * selected_sin + tail[:, 1] * selected_cos,
        ),
        dim=-1,
    ).flatten()
    return torch.cat((pooled[:-64], rotated), dim=0)


def reference(values, ape, norm, groups, cos, sin, eps):
    values = values.reshape(-1, 4, 2048)
    outputs = []
    for group in range(values.shape[0]):
        if group == 0:
            context = torch.cat((torch.zeros_like(values[group]), values[group]), dim=0)
        else:
            context = torch.cat((values[group - 1], values[group]), dim=0)
        outputs.append(context_reference(context, ape, norm, int(groups[group]), cos, sin, eps))
    return (
        torch.stack(outputs).to(values.dtype)
        if outputs
        else torch.empty((0, 512), dtype=values.dtype, device=values.device)
    )


def fixture(groups=4, *, request=7, start=0, seed=19, device="cpu", requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    rows = groups * 4
    values = torch.randn((rows, 2048), generator=generator).bfloat16().to(device)
    values.requires_grad_(requires_grad)
    positions = torch.arange(start, start + rows, dtype=torch.int64, device=device)
    ids = torch.stack((torch.full_like(positions, request), positions), dim=-1)
    ape = (
        (torch.randn((4, 1024), generator=generator) * 0.1).to(device).requires_grad_(requires_grad)
    )
    norm = (
        (torch.randn((512,), generator=generator) * 0.1 + 1)
        .to(device)
        .requires_grad_(requires_grad)
    )
    length = max(4, start + rows)
    angles = torch.arange(length, dtype=torch.float32).unsqueeze(1) * (
        10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    )
    cos = angles.cos().bfloat16().to(device)
    sin = angles.sin().bfloat16().to(device)
    return values, ids, ape, norm, cos, sin


def byte_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


class ReferenceC4Backend:
    backend_id = C4_BACKEND_ID
    device_type = "cpu"

    def __init__(self):
        self.calls = 0

    def forward(self, bound, ape, norm, cos, sin, contract):
        self.calls += 1
        if not bound.logical_keys:
            return torch.empty((0, 512), dtype=torch.bfloat16)
        return torch.stack(
            [
                context_reference(
                    bound.context_values[index].float(),
                    ape,
                    norm,
                    group,
                    cos,
                    sin,
                    contract.rms_eps,
                )
                for index, (_request, group) in enumerate(bound.logical_keys)
            ],
            dim=0,
        ).bfloat16()

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        contributions, ape_rows, norm_rows = [], [], []
        with torch.enable_grad():
            for index, (_request, group) in enumerate(bound.logical_keys):
                context = bound.context_values[index].float().detach().requires_grad_()
                a, n = ape.detach().requires_grad_(), norm.detach().requires_grad_()
                y = context_reference(context, a, n, group, cos, sin, contract.rms_eps)
                dx, da, dn = torch.autograd.grad(
                    y, (context, a, n), grad_outputs=grad[index].float()
                )
                contributions.append((bound.context_indices[index], dx))
                ape_rows.append(da)
                norm_rows.append(dn)
        return contributions, torch.stack(ape_rows), torch.stack(norm_rows)
