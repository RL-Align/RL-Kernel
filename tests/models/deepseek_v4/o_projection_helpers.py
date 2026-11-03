"""Independent FP64 test oracle; never used as a production fallback."""

import torch

from rl_engine.contracts.operators.o_projection import O_PROJECTION_BACKEND, OProjectionContract
from rl_engine.models.deepseek_v4.o_projection import OProjection, OProjectionRows


class ReferenceBackend:
    backend_id = O_PROJECTION_BACKEND
    device_type = "cpu"
    provenance = {"test_only": "explicit FP64 oracle"}

    def grouped(self, x, weight):
        return torch.bmm(x.double(), weight.double().transpose(1, 2)).to(torch.bfloat16)

    def linear(self, x, weight, *, output_dtype, role):
        return (x.double() @ weight.double().t()).to(output_dtype)

    def inverse_rope(self, x, cos, sin):
        return rotation(x, cos, sin, inverse=True).to(torch.bfloat16)

    def inverse_rope_gradient(self, x, cos, sin):
        return rotation(x, cos, sin, inverse=False).to(torch.bfloat16)

    def input_gradient(self, grad, weight):
        return (grad.double() @ weight.double()).to(torch.bfloat16)

    def weight_gradient(self, rows, grads):
        return (grads.double().t() @ rows.double()).to(torch.bfloat16)


def rotation(x, cos, sin, *, inverse):
    dim = cos.shape[-1] * 2
    y = x.double().clone()
    c, s = cos[:, 0].double(), sin[:, 0].double()
    a, b = x[..., -dim::2].double(), x[..., -dim + 1 :: 2].double()
    sign = -1 if inverse else 1
    y[..., -dim:] = torch.stack((a * c - sign * b * s, b * c + sign * a * s), -1).flatten(-2)
    return y


def make_case(*, device="cpu", m=5, train=False, contract=None, backend=None, padding=False):
    torch.manual_seed(230)
    c = contract or OProjectionContract(16, 128, 64, 128, 128)
    op = OProjection(c, layer_id="layer23", backend=backend or ReferenceBackend())
    a = (
        (
            torch.randn(c.local_groups, c.o_lora_rank, c.group_width, device=device)
            / c.group_width**0.5
        )
        .bfloat16()
        .requires_grad_(train)
    )
    b = (
        (
            torch.randn(c.hidden_size, c.local_groups * c.o_lora_rank, device=device)
            / (c.local_groups * c.o_lora_rank) ** 0.5
        )
        .bfloat16()
        .requires_grad_(train)
    )
    x = torch.randn(m, c.local_heads, c.head_dim, device=device).bfloat16().requires_grad_(train)
    keys = torch.stack(
        (torch.arange(m, device=device) % 3, torch.arange(m, device=device) * 3), 1
    ).long()
    if padding and m:
        keys[-1] = -1
    angles = torch.arange(max(m * 3, 1), device=device).float()[:, None] * (
        10000.0 ** (-torch.arange(0, c.rope_dim, 2, device=device).float() / c.rope_dim)
    )
    rows = OProjectionRows.bind(
        c, keys, angles.cos(), angles.sin(), head_ids=c.head_ids, version="v1"
    )
    weights = op.prepare_weights(a, b, version="v1")
    return op, x, weights, rows


def select_rows(c, rows, index):
    from rl_engine.models.deepseek_v4.indexer_projection import IndexerLogicalRows, tensor_stamp

    cos, sin = rows.cos[index].clone(), rows.sin[index].clone()
    return OProjectionRows(
        IndexerLogicalRows.bind(rows.logical.keys[index]),
        cos,
        sin,
        (tensor_stamp(cos), tensor_stamp(sin)),
        c.head_ids,
        rows.version,
        c.fingerprint,
    )


def oracle_forward(c, x, weights, rows):
    masked = torch.where((rows.logical.keys[:, 0] >= 0)[:, None, None], x, 0)
    inv = rotation(masked, rows.cos, rows.sin, inverse=True).bfloat16()
    grouped = inv.reshape(x.shape[0], c.local_groups, c.group_width).transpose(0, 1)
    z = torch.bmm(grouped.double(), weights.a.double().transpose(1, 2)).bfloat16()
    z = z.transpose(0, 1).contiguous()
    y = z.flatten(1).double() @ weights.b.double().t()
    return inv, z, y.float()


def oracle_backward(c, result, weights, grad):
    dy = torch.where(result.rows.logical.keys[:, :1] >= 0, grad.bfloat16(), 0)
    dz = (
        (dy.double() @ weights.b.double())
        .bfloat16()
        .reshape(dy.shape[0], c.local_groups, c.o_lora_rank)
        .transpose(0, 1)
    )
    inv = result.inverse_rope.reshape(dy.shape[0], c.local_groups, c.group_width).transpose(0, 1)
    da = torch.bmm(dz.double().transpose(1, 2), inv.double()).bfloat16()
    db = (dy.double().t() @ result.groups.flatten(1).double()).bfloat16()
    dinv = (
        torch.bmm(dz.double(), weights.a.double())
        .bfloat16()
        .transpose(0, 1)
        .reshape(dy.shape[0], c.local_heads, c.head_dim)
    )
    dx = rotation(dinv, result.rows.cos, result.rows.sin, inverse=False).bfloat16()
    return dx, da, db


def same_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
