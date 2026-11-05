# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 output selection: copies, ordered VJP, accuracy and supported invariances."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.minimax_h3.packed_index_select import NativePackedIndexSelectOp

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def vjp(op, x, index, grad):
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


class TestReference:
    @pytest.mark.parametrize(
        "shape,count", [((2, 0, 32), 0), ((2, 5, 96), 0), ((0, 5, 32), 3), ((2, 5, 0), 3)]
    )
    def test_empty(self, shape, count):
        x = torch.zeros(shape)
        index = torch.zeros(count, dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2])
        output, dx = vjp(NativePackedIndexSelectOp(), x, index, grad)
        assert output.shape == grad.shape
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_duplicate_contract(self, dtype):
        x = torch.randn(2, 5, 3, dtype=dtype)
        index = torch.tensor([4, 1, 4, 0, 4], dtype=torch.int32)
        grad = torch.arange(30).reshape(2, 5, 3).to(dtype)
        output, dx = vjp(NativePackedIndexSelectOp(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        expected = torch.zeros_like(x)
        expected[:, 0] = grad[:, 3]
        expected[:, 1] = grad[:, 1]
        expected[:, 4] = (grad[:, 0].float() + grad[:, 2].float() + grad[:, 4].float()).to(dtype)
        assert_bits(dx, expected)

    def test_fp32_rounding_order_is_observable(self):
        x = torch.zeros(1, 1, 1)
        index = torch.zeros(3, dtype=torch.int64)
        grad = torch.tensor([2**24, 1, -(2**24)], dtype=torch.float32).view(1, 3, 1)
        op = NativePackedIndexSelectOp()
        assert vjp(op, x, index, grad)[1].item() == 0
        # FP64 accuracy oracle deliberately differs on adversarial cancellation.
        assert vjp(op.forward_fp32, x, index, grad)[1].item() == 1
        assert vjp(op, x, index, grad[:, [0, 2, 1]])[1].item() == 1

    @pytest.mark.parametrize(
        "x,index,error",
        [
            (torch.zeros(2, 3), torch.tensor([0]), ValueError),
            (torch.zeros(1, 3, 2).double(), torch.tensor([0]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([[0]]), ValueError),
            (torch.zeros(1, 3, 2), torch.tensor([0.0]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([True]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([-1]), IndexError),
            (torch.zeros(1, 3, 2), torch.tensor([3]), IndexError),
            (torch.zeros(1, 0, 2), torch.tensor([0]), IndexError),
        ],
    )
    def test_invalid_inputs(self, x, index, error):
        with pytest.raises(error):
            NativePackedIndexSelectOp()(x, index)


@requires_cuda
@pytest.mark.cuda_only
class TestCuda:
    @staticmethod
    def op():
        from rl_engine.backends.shared.triton.embedding.packed_index_select import (
            TritonPackedIndexSelectOp,
        )

        return TritonPackedIndexSelectOp()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "width,index_dtype",
        [(32, torch.int32), (96, torch.int64), (7, torch.int64), (513, torch.int32)],
    )
    def test_strides_repeats_accuracy(self, dtype, width, index_dtype):
        torch.manual_seed(420)
        x = torch.randn(2, 67, 2 * width, device="cuda", dtype=dtype)[..., ::2]
        index = torch.randint(0, 67, (258,), device="cuda", dtype=index_dtype)[::2]
        grad = torch.randn(2, width, 129, device="cuda", dtype=dtype).transpose(1, 2)
        output, dx = vjp(self.op(), x, index, grad)
        assert output.is_contiguous()
        assert_bits(output, x.index_select(1, index))
        ordered = vjp(NativePackedIndexSelectOp(), x, index, grad)[1]
        assert_bits(dx, ordered)
        gold = vjp(NativePackedIndexSelectOp().forward_fp32, x.float(), index, grad.float())[1]
        tolerance = resolve_tolerance(
            load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=dtype
        )
        torch.testing.assert_close(dx.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        for _ in range(3):
            assert_bits(vjp(self.op(), x, index, grad)[1], dx)

    @pytest.mark.parametrize(
        "shape,count",
        [((2, 0, 32), 0), ((2, 5, 96), 0), ((0, 5, 32), 3), ((2, 5, 0), 3), ((1, 1, 1), 1)],
    )
    def test_empty_and_boundary(self, shape, count):
        x = torch.empty(shape, device="cuda")
        index = torch.zeros(count, device="cuda", dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2], device="cuda")
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, vjp(NativePackedIndexSelectOp(), x, index, grad)[1])

    @pytest.mark.parametrize("width", [32, 96])
    @pytest.mark.parametrize("sequence", [38734, 110750, 367986])
    def test_realistic_h3_unique_selection(self, width, sequence):
        # Full mixed-row counts from the merged output-head qualification.
        x = torch.randn(1, sequence, width, device="cuda")
        index = torch.arange(sequence - 1, -1, -3, device="cuda")
        grad = torch.randn(1, index.numel(), width, device="cuda")
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        expected = torch.zeros_like(x)
        expected.index_copy_(1, index, grad)
        assert_bits(dx, expected)

    def test_batch_train_infer_checkpoint_and_row_relabeling(self):
        from torch.utils.checkpoint import checkpoint

        x = torch.randn(3, 73, 96, device="cuda")
        index = torch.tensor([72, 1, 0, 72, 2, 1], device="cuda")
        grad = torch.randn(3, 6, 96, device="cuda")
        op = self.op()
        output, dx = vjp(op, x, index, grad)
        with torch.inference_mode():
            assert_bits(op(x, index), output)
        for batch in range(3):
            single_y, single_dx = vjp(op, x[batch : batch + 1], index, grad[batch : batch + 1])
            assert_bits(single_y, output[batch : batch + 1])
            assert_bits(single_dx, dx[batch : batch + 1])
        permutation = torch.tensor([2, 0, 1], device="cuda")
        py, pdx = vjp(op, x[permutation], index, grad[permutation])
        assert_bits(py, output[permutation])
        assert_bits(pdx, dx[permutation])
        # Relabel input rows while preserving duplicate occurrence order.
        rows = torch.randperm(73, device="cuda")
        ry, rdx = vjp(op, x[:, rows], torch.argsort(rows)[index], grad)
        assert_bits(ry, output)
        assert_bits(rdx, dx[:, rows])
        cy, cdx = vjp(lambda a, i: checkpoint(op, a, i, use_reentrant=False), x, index, grad)
        assert_bits(cy, output)
        assert_bits(cdx, dx)
        changed = x.clone()
        changed[0] *= 123
        assert_bits(op(changed, index)[1:], output[1:])

    def test_special_value_copy_and_high_multiplicity(self):
        x = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda")
        x = x.view(1, 5, 1)
        index = torch.tensor([4, 1, 3, 0, 2], device="cuda")
        assert_bits(self.op()(x, index), x.index_select(1, index))
        x = torch.zeros(2, 3, 32, device="cuda")
        index = torch.ones(1025, device="cuda", dtype=torch.int64)
        grad = torch.randn(2, 1025, 32, device="cuda")
        assert_bits(
            vjp(self.op(), x, index, grad)[1], vjp(NativePackedIndexSelectOp(), x, index, grad)[1]
        )

    def test_expanded_input_and_gradient(self):
        x = torch.randn(1, 1, 7, device="cuda").expand(2, 9, 7)
        index = torch.tensor([8, 0, 8], device="cuda")
        grad = torch.ones(1, 1, 1, device="cuda").expand(2, 3, 7)
        actual = vjp(self.op(), x, index, grad)
        expected = vjp(NativePackedIndexSelectOp(), x, index, grad)
        for a, b in zip(actual, expected, strict=True):
            assert_bits(a, b)

    def test_device_guards_deterministic_mode_and_runtime(self):
        from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
        from rl_engine.runtime.registry import KernelRegistry

        op = KernelRegistry().get_op("packed_index_select", device="cuda")
        x = torch.randn(1, 7, 32, device="cuda")
        index = torch.tensor([6, 0, 6], device="cuda")
        grad = torch.randn(1, 3, 32, device="cuda")
        expected = vjp(op, x, index, grad)
        old = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = vjp(op, x, index, grad)
        finally:
            torch.use_deterministic_algorithms(old)
        for a, b in zip(actual, expected, strict=True):
            assert_bits(a, b)
        event = snapshot_backward_runtime()["packed_index_select"]
        assert event["family"] == "triton" and "stable_segment" in event["impl"]
        with pytest.raises(ValueError, match="same device"):
            op(x, index.cpu())
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            op(x.cpu(), index.cpu())

    @pytest.mark.parametrize("bad_index", [-1, 7, 2**63 - 1])
    def test_invalid_cuda_index_fails_in_isolated_process(self, bad_index):
        code = f"""
import torch
from rl_engine.backends.shared.triton.embedding.packed_index_select import TritonPackedIndexSelectOp
x = torch.zeros(1, 7, 32, device='cuda')
i = torch.tensor([{bad_index}], device='cuda')
TritonPackedIndexSelectOp()(x, i)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=90
        )
        assert result.returncode != 0
        assert "device-side assert" in result.stderr or "index values must be" in result.stderr
