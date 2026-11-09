# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 dense FP8 linear on DeepGEMM.

Properties: the forward is the vLLM rollout composition (golden FP8 group
quantization + DeepGEMM ``fp8_gemm_nt``) byte for byte; forward and dX are
batch-invariant at the DSv4 projection shapes; dX equals a forward through the
block-quantized ``W.T``; dW is run-to-run identical (fresh and accumulated);
all three stay within FP8 error of FP32; empty batches and bad inputs are
handled without launching.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.pytorch.quant import fp8_group_quant


def _ready() -> bool:
    if not torch.cuda.is_available() or not deepgemm_available():
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "fp8_group_quant_forward")


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90 + pinned DeepGEMM + CUDA 12.8 JIT + rl_engine._C required"
)

# (name, N, K) at TP=1, as in scripts/probe_dsv4_deepgemm_bi.py.
DSV4_SHAPES = (
    ("fused_wqa_wkv", 1536, 4096),
    ("wq_b", 32768, 1024),
    ("wo_a_group", 1024, 4096),
    ("wo_b", 4096, 8192),
    ("indexer_wq_b", 8192, 1024),
)
FP8_REL_ERR = 0.06  # E4M3 operand rounding gives ~3.8% on Gaussian data


def _linear(n: int, k: int, seed: int = 0):
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear

    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=gen) / k**0.5).to(torch.bfloat16).cuda()
    return Fp8Linear.from_bf16(w), w


def _rand(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=gen).to(torch.bfloat16).cuda()


def _bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(-1), b.view(-1))


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want).norm() / want.norm())


def test_forward_is_the_rollout_composition_byte_for_byte():
    import deep_gemm

    lin, _ = _linear(1536, 4096)
    x = _rand(257, 4096, seed=1)
    q, s = fp8_group_quant(x)  # PyTorch golden of vLLM's per_token_group_fp8_quant
    want = torch.empty(257, 1536, device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((q, s), lin.weight.pair(), want, disable_ue8m0_cast=True)
    assert _bytes_equal(lin(x), want)


@pytest.mark.parametrize("name,n,k", DSV4_SHAPES, ids=[s[0] for s in DSV4_SHAPES])
def test_forward_and_dx_are_batch_invariant(name, n, k):
    lin, _ = _linear(n, k)
    x = _rand(512, k, seed=2)
    dy = _rand(512, n, seed=3)
    y, dx = lin(x), lin.backward_dx(dy)
    assert _bytes_equal(lin(x), y) and _bytes_equal(lin.backward_dx(dy), dx), "run to run"
    for start, stop in ((0, 1), (0, 3), (5, 6), (127, 257), (511, 512)):
        assert _bytes_equal(lin(x[start:stop]), y[start:stop]), (name, start, stop)
        assert _bytes_equal(lin.backward_dx(dy[start:stop]), dx[start:stop]), (name, start, stop)


def test_dx_equals_forward_through_transposed_weight():
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear

    lin, w = _linear(1024, 4096)
    dy = _rand(130, 1024, seed=4)
    via_t = Fp8Linear.from_bf16(w.t().contiguous())
    assert _bytes_equal(lin.backward_dx(dy), via_t(dy))


@pytest.mark.parametrize("tokens", [1, 100, 1000])
def test_dw_is_deterministic_and_accumulates(tokens):
    lin, _ = _linear(1536, 4096)
    x, dy = _rand(tokens, 4096, seed=5), _rand(tokens, 1536, seed=6)
    first = lin.backward_dw(dy, x)
    assert first.dtype == torch.float32 and tuple(first.shape) == (1536, 4096)
    for _ in range(3):
        assert torch.equal(lin.backward_dw(dy, x).view(torch.int32), first.view(torch.int32))

    acc0 = torch.randn(1536, 4096, device="cuda")
    runs = [lin.backward_dw(dy, x, out=acc0.clone()) for _ in range(3)]
    for run in runs[1:]:
        assert torch.equal(run.view(torch.int32), runs[0].view(torch.int32))
    torch.testing.assert_close(runs[0], acc0 + first, rtol=1e-5, atol=1e-4)

    out = acc0.clone()
    assert lin.backward_dw(dy, x, out=out) is out


def test_accuracy_against_fp32():
    lin, _ = _linear(1536, 4096)
    w32 = lin.weight.dequantize().float()
    x, dy = _rand(512, 4096, seed=7), _rand(512, 1536, seed=8)
    x32, dy32 = x.float(), dy.float()
    assert _rel_err(lin(x), x32 @ w32.t()) < FP8_REL_ERR
    assert _rel_err(lin.backward_dx(dy), dy32 @ w32) < FP8_REL_ERR
    assert _rel_err(lin.backward_dw(dy, x), dy32.t() @ x32) < FP8_REL_ERR


def test_empty_batches_launch_nothing():
    lin, _ = _linear(1536, 4096)
    assert tuple(lin(_rand(0, 4096, seed=9)).shape) == (0, 1536)
    assert tuple(lin.backward_dx(_rand(0, 1536, seed=9)).shape) == (0, 4096)
    dw = lin.backward_dw(_rand(0, 1536, seed=9), _rand(0, 4096, seed=9))
    assert torch.count_nonzero(dw) == 0
    torch.cuda.synchronize()


def test_checkpoint_weights_and_input_validation():
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear

    lin, _ = _linear(256, 512)
    again = Fp8Linear.from_fp8(*lin.weight.pair())
    x = _rand(4, 512, seed=10)
    assert _bytes_equal(again(x), lin(x))

    with pytest.raises(TypeError):
        Fp8Linear.from_fp8(lin.weight.q.float(), lin.weight.scale)
    with pytest.raises(ValueError):
        Fp8Linear.from_fp8(lin.weight.q, lin.weight.scale[:1])
    with pytest.raises(ValueError):
        Fp8Linear.from_bf16(torch.zeros(100, 512, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        lin(x.float())
    with pytest.raises(ValueError):
        lin(_rand(4, 256, seed=10))
    with pytest.raises(ValueError):
        lin.backward_dw(_rand(4, 256, seed=10), _rand(5, 512, seed=10))
    with pytest.raises(ValueError):
        out = torch.zeros(256, 512, device="cuda", dtype=torch.bfloat16)  # wgrad is FP32 only
        lin.backward_dw(_rand(4, 256, seed=10), x, out=out)
