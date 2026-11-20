# SPDX-License-Identifier: Apache-2.0
"""Test-only independent numerical oracle; never a production fallback."""

import torch

from rl_engine.contracts.operators.indexer_icv import ICV_BACKEND_ID

FP4_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_fp4(data, scales, dim):
    # Independent CPU float64 oracle includes every nibble and exponent.
    raw, scale = data.cpu(), scales.cpu()
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    table = torch.tensor(FP4_TABLE, dtype=torch.float64)
    values = table[codes.long()]
    return (values * torch.exp2(scale.double() - 127).repeat_interleave(32, -1))[..., :dim]


def decode(data, scales, contract):
    if contract.precision == "mxfp4":
        return decode_fp4(data, scales, contract.head_dim)
    validate_fp8(data, scales, contract)
    return data.cpu().double() * scales.cpu().double()


def validate_fp8(data, scales, contract):
    """Independent oracle checks; FP8 scales are multipliers, NOT exponent bytes."""
    if not isinstance(scales, torch.Tensor) or scales.dtype != torch.float32:
        raise ValueError("FP8 reference requires FP32 scale multipliers")
    if data.dtype != getattr(torch, "float8_" + contract.fp8_encoding):
        raise ValueError("FP8 reference encoding mismatch")
    if data.ndim not in (2, 3) or data.shape[-1] != contract.head_dim:
        raise ValueError("FP8 reference data shape mismatch")
    if tuple(scales.shape) != (*data.shape[:-1], 1):
        raise ValueError("FP8 reference scale shape mismatch")
    if not bool(torch.isfinite(data.float()).all()):
        raise ValueError("FP8 reference payload must be finite")
    s = scales.cpu().double()
    exponent = torch.log2(s)
    if not bool((torch.isfinite(exponent) & (exponent >= -60) & (exponent <= 30)).all()):
        raise ValueError("FP8 reference scale out of domain")
    if not bool((exponent == exponent.round()).all()):
        raise ValueError("FP8 reference scale must be a power of two")


class ReferenceICVBackend:
    backend_id = ICV_BACKEND_ID
    device_type = "cpu"

    def __init__(self):
        self.calls = 0

    @staticmethod
    def plan(contract):
        return {"test_only": True}

    def prepare(self, data, scales, contract):
        if contract.precision == "mxfp4":
            return decode(data, scales, contract).bfloat16(), None
        validate_fp8(data, scales, contract)
        # Keep multipliers separate to test the production epilogue placement.
        # The old oracle pre-scaled BF16 operands; that applied scales, but was
        # not faithful to FP8 dot-then-scale at the small-exponent boundary.
        return data.cpu().float(), scales.cpu().clone()

    def dot(self, q, k, q_scale, k_scale, contract):
        self.calls += 1
        out = torch.einsum("mhd,nd->mhn", q.float(), k.float())
        if contract.precision == "fp8":
            out = out * (q_scale * k_scale.transpose(0, 1))
        return out


def payload(contract, rows, *, q=False, seed=0, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    leading = (rows, contract.num_heads) if q else (rows,)
    if contract.precision == "mxfp4":
        data = torch.randint(
            0, 256, (*leading, contract.padded_dim // 2), generator=generator, dtype=torch.uint8
        )
        scales = torch.randint(
            123, 131, (*leading, contract.padded_dim // 32), generator=generator, dtype=torch.uint8
        )
    else:
        dtype = getattr(torch, "float8_" + contract.fp8_encoding)
        data = torch.randn((*leading, contract.head_dim), generator=generator).to(dtype)
        scales = torch.exp2(torch.randint(-3, 4, (*leading, 1), generator=generator).float())
    return data.to(device), scales.to(device)


def ids(request, rows, *, start=0, device="cpu"):
    return torch.tensor(
        [[request, i] for i in range(start, start + rows)], dtype=torch.int64, device=device
    ).reshape(rows, 2)


def byte_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
