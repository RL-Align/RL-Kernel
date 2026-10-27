# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real ROCm acceptance. Set RLK_REQUIRE_ROCM=1 to fail instead of skipping."""

import os

import pytest
import torch

from rl_engine.backends.rocm.moe.lora_delta import RocmLoRADelta
from rl_engine.ops.moe.lora_delta import shared_grouped_lora_delta


@pytest.fixture(scope="module", autouse=True)
def require_rocm():
    reason = None
    if torch.version.hip is None or not torch.cuda.is_available():
        reason = "requires a ROCm PyTorch build and a visible AMD GPU"
    else:
        props = torch.cuda.get_device_properties(0)
        arch = str(props.gcnArchName).split(":")[0]
        if arch not in ("gfx942", "gfx950"):
            reason = f"requires gfx942/gfx950, got {arch}"
        else:
            print(f"GPU={props.name} arch={arch} torch={torch.__version__} HIP={torch.version.hip}")
    if reason:
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    import triton

    print(f"Triton={triton.__version__} profile={RocmLoRADelta.numeric_profile}")


def tensors(m, k, n, rank):
    gen = torch.Generator().manual_seed(45)
    return tuple(
        (torch.randn(shape, generator=gen) * 0.125).to(device="cuda", dtype=torch.bfloat16)
        for shape in ((m, k), (rank, k), (n, rank))
    )


def assert_bytes(lhs, rhs):
    assert lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
    assert torch.equal(lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8))


@pytest.mark.parametrize("k,n,rank", [(4096, 2048, 8), (2048, 4096, 16), (65, 67, 33)])
def test_same_row_across_batches_and_positions(k, n, rank):
    op = RocmLoRADelta()
    x, a, b = tensors(65, k, n, rank)
    row = x[:1].clone()
    dy_row = torch.linspace(-0.2, 0.2, n, device="cuda")[None, :]
    baseline_y, baseline_u = op.forward(row, a, b, 0.375)
    baseline_dx = op.backward(dy_row, row, a, b, 0.375, baseline_u)[0]
    for m in (1, 16, 17, 64, 65):
        for index in sorted({0, m // 2, m - 1}):
            batch = x[:m].clone()
            batch[index] = row[0]
            dy = torch.full((m, n), 0.1, device="cuda")
            dy[index] = dy_row[0]
            y, u = op.forward(batch, a, b, 0.375)
            grads = op.backward(dy, batch, a, b, 0.375, u)
            assert_bytes(y[index : index + 1], baseline_y)
            assert_bytes(u[index : index + 1], baseline_u)
            assert_bytes(grads[0][index : index + 1], baseline_dx)
            # Parameter gradients depend on all rows: compare the SAME batch.
            again = op.backward(dy, batch, a, b, 0.375, u)
            for actual, repeated in zip(grads, again):
                assert_bytes(actual, repeated)


@pytest.mark.parametrize("k,n,rank", [(4096, 2048, 16), (2048, 4096, 8), (65, 67, 33)])
def test_staged_fp64_accuracy(k, n, rank):
    x, a, b = tensors(17, k, n, rank)
    op = RocmLoRADelta()
    y, u = op.forward(x, a, b, 0.375)
    xx, aa, bb = (t.cpu().double() for t in (x, a, b))
    uu = (xx @ aa.t()).bfloat16().double()
    expected_y = (uu @ bb.t()) * 0.375
    dy = torch.linspace(-0.2, 0.2, y.numel(), device="cuda").reshape_as(y)
    dv = (dy.cpu().double() * 0.375).bfloat16().double()
    du = (dv @ bb).bfloat16().double()
    expected = (expected_y, du @ aa, du.t() @ xx, dv.t() @ uu)
    actual = (y, *op.backward(dy, x, a, b, 0.375, u))
    assert u.dtype == torch.bfloat16
    for value, target in zip(actual, expected):
        assert value.dtype == torch.float32
        error = value.cpu().double() - target
        # BF16 intermediate rounding can move one U/dU value by one ULP.
        assert error.norm() <= 0.01 * target.norm() + 1e-7
        assert error.abs().max() <= 0.01 * target.abs().max() + 1e-7


def test_empty_batch_and_noncontiguous_inputs():
    op = RocmLoRADelta()
    x, a, b = tensors(17, 65, 67, 9)
    y, u = op.forward(x, a, b, 0.375)
    strided = tuple(t.t().contiguous().t() for t in (x, a, b))
    strided_y, strided_u = op.forward(*strided, 0.375)
    assert_bytes(strided_y, y)
    assert_bytes(strided_u, u)
    dy = torch.ones_like(y).t().contiguous().t()
    for lhs, rhs in zip(
        op.backward(dy, *strided, 0.375, strided_u),
        op.backward(dy.contiguous(), x, a, b, 0.375, u),
    ):
        assert_bytes(lhs, rhs)
    empty_y, empty_u = op.forward(x[:0], a, b, 0.375)
    assert empty_y.shape == (0, b.shape[0])
    dx, da, db = op.backward(empty_y, x[:0], a, b, 0.375, empty_u)
    assert dx.shape == x[:0].shape
    assert torch.count_nonzero(da) == 0 and torch.count_nonzero(db) == 0


def test_chunks_and_zero_padding():
    x, a, b = tensors(33, 65, 67, 9)
    op = RocmLoRADelta()
    full, _ = op.forward(x, a, b, 0.375)
    chunks = torch.cat([op.forward(chunk, a, b, 0.375)[0] for chunk in x.split(16)])
    assert_bytes(full, chunks)
    padded = torch.cat((x, torch.zeros_like(x)))
    padded_y, _ = op.forward(padded, a, b, 0.375)
    assert_bytes(full, padded_y[: x.shape[0]])


def test_autograd_save_recompute_and_weight_updates():
    x, a, b = tensors(17, 65, 67, 9)
    op = RocmLoRADelta()
    for step in range(3):
        expected_y, u = op.forward(x, a, b, 0.375)
        dy = torch.full_like(expected_y, 0.2)
        expected_grads = op.backward(dy, x, a, b, 0.375, u)
        for recompute in (False, True):
            inputs = tuple(t.detach().clone().requires_grad_() for t in (x, a, b))
            y = shared_grouped_lora_delta(*inputs, 0.375, recompute=recompute)
            y.backward(dy)
            assert_bytes(y, expected_y)
            for operand, grad in zip(inputs, expected_grads):
                assert_bytes(operand.grad, grad.bfloat16())
        if step < 2:
            a.add_(0.015625)
            b.sub_(0.0078125)
