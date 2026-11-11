# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA clamped SwiGLU: byte-exact with the PyTorch golden.

Two orthogonal properties: the fused forward emits the bytes of
``fp8_group_quant(silu_and_mul_clamp(z))`` (h, FP8 values and scales) and the
backward the bytes of ``silu_and_mul_clamp_backward``, over every BF16 gate
value and edge up values; and a row's output does not depend on the batch.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp import (
    silu_and_mul_clamp,
    silu_and_mul_clamp_backward,
)
from rl_engine.kernels.ops.pytorch.quant import fp8_group_quant


def _kernel_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError:
        return False
    return hasattr(ext, "silu_and_mul_clamp_quant_forward")


pytestmark = pytest.mark.skipif(not _kernel_available(), reason="CUDA SwiGLU kernel not built")

UPS = [0.0, -0.0, 1.0, -3.5, 9.9375, 10.0, -10.0, 12.0, -12.0, float("inf"), float("-inf")]
UPS += [float("nan"), 1e-30]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _every_gate(ffn: int = 256) -> torch.Tensor:
    values = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return values.view(-1, ffn)


def _cases():
    gates = _every_gate()
    gen = torch.Generator().manual_seed(0)
    for up in UPS + ["random"]:
        if up == "random":
            ups = (torch.randn(gates.shape, generator=gen) * 8).to(torch.bfloat16)
        else:
            ups = torch.full(gates.shape, up).to(torch.bfloat16)
        yield up, torch.cat([gates, ups], dim=1).contiguous().cuda()


@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_forward_matches_golden_for_every_gate(scale_ue8m0):
    from rl_engine.kernels.ops.cuda.activation.silu_and_mul_clamp import (
        silu_and_mul_clamp_quant_cuda,
    )

    for up, z in _cases():
        q, s, h = silu_and_mul_clamp_quant_cuda(z, scale_ue8m0=scale_ue8m0, return_h=True)
        h_ref = silu_and_mul_clamp(z)
        q_ref, s_ref = fp8_group_quant(h_ref, scale_ue8m0=scale_ue8m0)
        assert torch.equal(_bytes(h), _bytes(h_ref)), up
        assert torch.equal(_bytes(q), _bytes(q_ref)), up
        assert s.stride() == s_ref.stride()
        assert torch.equal(s.view(torch.int32), s_ref.view(torch.int32)), up


def test_backward_matches_golden_for_every_gate():
    from rl_engine.kernels.ops.cuda.activation.silu_and_mul_clamp import (
        silu_and_mul_clamp_backward_cuda,
    )

    gen = torch.Generator().manual_seed(1)
    for up, z in _cases():
        for dh in (
            torch.ones(z.shape[0], z.shape[1] // 2, dtype=torch.bfloat16, device="cuda"),
            (torch.randn(z.shape[0], z.shape[1] // 2, generator=gen) * 3).to(torch.bfloat16).cuda(),
        ):
            got = silu_and_mul_clamp_backward_cuda(dh, z)
            assert torch.equal(_bytes(got), _bytes(silu_and_mul_clamp_backward(dh, z))), up


@pytest.mark.parametrize("rows", [1, 3, 300])
@pytest.mark.parametrize("ffn", [128, 2048])
def test_shapes(rows, ffn):
    from rl_engine.kernels.ops.cuda.activation.silu_and_mul_clamp import (
        silu_and_mul_clamp_backward_cuda,
        silu_and_mul_clamp_quant_cuda,
    )

    gen = torch.Generator().manual_seed(rows + ffn)
    z = (torch.randn(rows, 2 * ffn, generator=gen) * 6).to(torch.bfloat16).cuda()
    dh = torch.randn(rows, ffn, generator=gen).to(torch.bfloat16).cuda()
    q, s = silu_and_mul_clamp_quant_cuda(z)
    q_ref, s_ref = fp8_group_quant(silu_and_mul_clamp(z))
    assert torch.equal(_bytes(q), _bytes(q_ref)) and torch.equal(s, s_ref)
    got = silu_and_mul_clamp_backward_cuda(dh, z)
    assert torch.equal(_bytes(got), _bytes(silu_and_mul_clamp_backward(dh, z)))


def test_rows_do_not_depend_on_the_batch():
    from rl_engine.kernels.ops.cuda.activation.silu_and_mul_clamp import (
        silu_and_mul_clamp_backward_cuda,
        silu_and_mul_clamp_quant_cuda,
    )

    gen = torch.Generator().manual_seed(2)
    z = (torch.randn(1024, 4096, generator=gen) * 6).to(torch.bfloat16).cuda()
    dh = torch.randn(1024, 2048, generator=gen).to(torch.bfloat16).cuda()
    q, s = silu_and_mul_clamp_quant_cuda(z)
    dz = silu_and_mul_clamp_backward_cuda(dh, z)
    for start, stop in ((0, 1), (511, 512), (100, 357)):
        q_part, s_part = silu_and_mul_clamp_quant_cuda(z[start:stop].contiguous())
        assert torch.equal(_bytes(q_part), _bytes(q[start:stop]))
        assert torch.equal(s_part, s[start:stop])
        dz_part = silu_and_mul_clamp_backward_cuda(
            dh[start:stop].contiguous(), z[start:stop].contiguous()
        )
        assert torch.equal(_bytes(dz_part), _bytes(dz[start:stop]))


def test_rejects_unsupported_inputs():
    from rl_engine.kernels.ops.cuda.activation.silu_and_mul_clamp import (
        silu_and_mul_clamp_backward_cuda,
        silu_and_mul_clamp_quant_cuda,
    )

    with pytest.raises(TypeError, match="BF16"):
        silu_and_mul_clamp_quant_cuda(torch.zeros(1, 256, device="cuda"))
    with pytest.raises(ValueError, match="multiple of 256"):
        silu_and_mul_clamp_quant_cuda(torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="dh must be"):
        silu_and_mul_clamp_backward_cuda(
            torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda"),
        )
