# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Invariance + correctness tests for det_gemm (WS1).

Runs against both deterministic backends — the hand-written CUDA kernel and the
Triton path — each of which must independently satisfy the invariance contract.
The PyTorch path (torch.matmul) is intentionally NOT tested here: it is the
non-deterministic reference baseline and would fail batch-invariance by design.
"""
import pytest
import torch

from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm

try:
    from rl_engine.kernels.ops.triton.matmul import deterministic_gemm_triton

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="det_gemm requires CUDA SM80+",
)

# Each deterministic backend is validated independently.
_BACKENDS = [("cuda", deterministic_gemm)]
if _HAS_TRITON:
    _BACKENDS.append(("triton", deterministic_gemm_triton))


def _rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_batch_invariance(name, gemm):
    # A row's output must not change when other rows join the batch.
    torch.manual_seed(0)
    K, N = 4096, 4096
    b = _rand(K, N)
    row = _rand(1, K)
    out1 = gemm(row, b)
    big = _rand(512, K)
    big[0] = row[0]
    outN = gemm(big, b)
    assert torch.equal(out1[0], outN[0]), f"{name}: forward batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_chunked_prefill(name, gemm):
    # Splitting M then concatenating must match the full GEMM bitwise.
    torch.manual_seed(1)
    M, K, N = 256, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    full = gemm(a, b)
    chunked = torch.cat([gemm(a[:100], b), gemm(a[100:], b)], dim=0)
    assert torch.equal(full, chunked), f"{name}: chunked-prefill broke invariance"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_padding_invariance(name, gemm):
    # Padding rows must not affect valid rows' output.
    torch.manual_seed(2)
    M, K, N = 100, 4096, 4096
    a, b = _rand(M, K), _rand(K, N)
    base = gemm(a, b)
    a_pad = torch.cat([a, _rand(28, K)], dim=0)
    padded = gemm(a_pad, b)
    assert torch.equal(base, padded[:M]), f"{name}: padding changed valid-row output"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_forward_correctness(name, gemm):
    # vs FP32 reference. Placeholder tolerance; PR3 swaps for #108 contract.
    torch.manual_seed(3)
    M, K, N = 128, 2048, 2048
    a, b = _rand(M, K), _rand(K, N)
    out = gemm(a, b).float()
    ref = a.float() @ b.float()
    assert (out - ref).abs().max().item() < 1.0  # TODO(#108): contract threshold


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_batch_invariance(name, gemm):
    # dA for a row must be invariant to the surrounding batch.
    torch.manual_seed(4)
    K, N = 2048, 2048
    b = _rand(K, N)
    row = _rand(1, K).requires_grad_(True)
    gemm(row, b).sum().backward()
    g1 = row.grad.clone()
    big = _rand(256, K)
    big[0] = row.detach()[0]
    big.requires_grad_(True)
    gemm(big, b).sum().backward()
    assert torch.equal(g1[0], big.grad[0]), f"{name}: backward dA batch-invariance broken"


@pytest.mark.parametrize("name,gemm", _BACKENDS)
def test_backward_correctness(name, gemm):
    # dA / dB vs FP32 reference gradients. Placeholder tolerance; PR3 -> #108.
    torch.manual_seed(5)
    M, K, N = 64, 1024, 1024
    a = _rand(M, K).requires_grad_(True)
    b = _rand(K, N).requires_grad_(True)
    g = _rand(M, N)
    gemm(a, b).backward(g)
    af = a.detach().float().requires_grad_(True)
    bf = b.detach().float().requires_grad_(True)
    (af @ bf).backward(g.float())
    assert (a.grad.float() - af.grad).abs().max().item() < 2.0  # TODO(#108)
    assert (b.grad.float() - bf.grad).abs().max().item() < 2.0  # TODO(#108)


@pytest.mark.parametrize("name,gemm", _BACKENDS)
@pytest.mark.parametrize(
    "shape",
    [
        (4096, 4096, 12288),  # qkv
        (4096, 4096, 4096),  # o_proj
        (4096, 4096, 14336),  # mlp_up
        (4096, 14336, 4096),  # mlp_dn
        (4096, 4096, 32000),  # lm_head
    ],
)
def test_target_shapes_invariance(name, gemm, shape):
    # Standard-Transformer projection shapes stay batch-invariant.
    torch.manual_seed(6)
    M, K, N = shape
    b = _rand(K, N)
    row = _rand(1, K)
    big = _rand(64, K)
    big[0] = row[0]
    assert torch.equal(
        gemm(row, b)[0], gemm(big, b)[0]
    ), f"{name}: batch-invariance broken at shape {shape}"
