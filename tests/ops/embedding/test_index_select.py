# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unique index selection, its one-writer gradient, and train/inference invariance."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")


def assert_bits(actual, expected):
    """Check shape and dtype before comparing the bytes of each logical element."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def vjp(op, x, index, grad, *, dim=1):
    """Differentiate a fresh leaf with a fixed upstream gradient for this selection."""
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index, dim=dim, unique_indices=True)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


def expected_gradient(x, index, grad_y, *, dim=1):
    """Build the expected FP32 gradient one selected slice at a time on CPU."""
    grad_x = torch.zeros(x.shape, dtype=torch.float32, device="cpu")
    grad_y = grad_y.to(device="cpu", dtype=torch.float32)
    for position, source in enumerate(index.cpu().tolist()):
        grad_x.select(dim, source).add_(grad_y.select(dim, position))
    return grad_x.to(x.device)


class TestReference:
    @pytest.mark.parametrize(
        "shape,count",
        [((2, 0, 32), 0), ((2, 5, 96), 0), ((0, 5, 32), 3), ((2, 5, 0), 3)],
    )
    def test_empty(self, shape, count):
        x = torch.zeros(shape)
        index = torch.arange(count, dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2])
        output, dx = vjp(NativeIndexSelectOp(), x, index, grad)
        assert output.shape == grad.shape
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize(
        "x,index,error",
        [
            (torch.tensor(0.0), torch.tensor([0]), ValueError),
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
            NativeIndexSelectOp()(x, index, dim=1, unique_indices=True)

    def test_dim_is_required_and_keyword_only(self):
        op = NativeIndexSelectOp()
        x, index = torch.zeros(2, 3), torch.tensor([0])
        with pytest.raises(TypeError, match="dim"):
            op(x, index, unique_indices=True)
        with pytest.raises(TypeError):
            op(x, index, 0, unique_indices=True)

    def test_forward_fp32_requires_explicit_uniqueness(self):
        op = NativeIndexSelectOp()
        x = torch.empty(0, dtype=torch.bfloat16, requires_grad=True)
        index = torch.empty(0, dtype=torch.int64)
        with pytest.raises(NotImplementedError, match="unique_indices=True"):
            op.forward_fp32(x, index, dim=0)
        with pytest.raises(NotImplementedError, match="unique_indices=True"):
            op.forward_fp32(x, index, dim=x.ndim, unique_indices=False)
        output = op.forward_fp32(x, index, dim=0, unique_indices=True)
        assert output.dtype == torch.float32
        (dx,) = torch.autograd.grad(output.sum(), x)
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize(
        "dtype,tiny_grad", [(torch.float16, -(2.0**-30)), (torch.bfloat16, -(2.0**-136))]
    )
    def test_forward_fp32_rounds_gradient_after_scatter(self, dtype, tiny_grad):
        x = torch.tensor([1.0, -0.0, 2.5, -7.5], dtype=dtype, requires_grad=True)
        index = torch.tensor([3, 1])
        output = NativeIndexSelectOp().forward_fp32(x, index, dim=0, unique_indices=True)
        assert_bits(output, torch.tensor([-7.5, -0.0], dtype=torch.float32))
        # The second contribution is negative in FP32 and rounds to -0 in x.dtype.
        grad = torch.tensor([5.0, tiny_grad], dtype=torch.float32)
        (dx,) = torch.autograd.grad(output, x, grad)
        assert_bits(dx, torch.tensor([0.0, -0.0, 0.0, 5.0], dtype=dtype))


@pytest.mark.parametrize(
    "backend",
    [
        "pytorch",
        pytest.param("cuda", marks=requires_cuda),
        *(
            name
            for name in ("xpu", "musa")
            if hasattr(torch, name) and getattr(torch, name).is_available()
        ),
    ],
)
class TestSharedAxes:
    @staticmethod
    def operator(backend):
        if backend == "pytorch":
            return NativeIndexSelectOp(), torch.device("cpu")
        from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

        return TritonIndexSelectOp(), torch.device(backend)

    @pytest.mark.parametrize("axis_size", [2**31, 2**31 + 1])
    def test_int32_indices_on_large_axis(self, backend, axis_size):
        op, device = self.operator(backend)
        # The expanded view represents every row with one stored element.
        x = torch.ones(1, device=device).expand(axis_size)
        index = torch.tensor([0, 2**31 - 1], dtype=torch.int32, device=device)
        output = op(x, index, dim=0, unique_indices=True)
        assert_bits(output, torch.ones(2, device=device))

    def test_requires_explicit_uniqueness(self, backend):
        op, device = self.operator(backend)
        x = torch.zeros(5, device=device, requires_grad=True)
        # False rejects the mode itself, even for empty or actually unique indices.
        for rows in ([], [3, 1], [3, 1, 3]):
            index = torch.tensor(rows, device=device, dtype=torch.int64)
            with pytest.raises(NotImplementedError, match="unique_indices=True"):
                op(x, index, dim=0)
            with pytest.raises(NotImplementedError, match="unique_indices=True"):
                op.forward(x, index, dim=0, unique_indices=False)
        index = torch.tensor([3, 1], device=device)
        # Unsupported mode takes precedence over an invalid axis.
        with pytest.raises(NotImplementedError, match="unique_indices=True"):
            op.forward(x, index, dim=x.ndim, unique_indices=False)
        output = op(x, index, dim=0, unique_indices=True)
        (dx,) = torch.autograd.grad(output, x, torch.tensor([2.0, 5.0], device=device))
        assert_bits(dx, torch.tensor([0.0, 5.0, 0.0, 2.0, 0.0], device=device))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_copied_bits_and_single_gradient_contribution(self, backend, dtype):
        op, device = self.operator(backend)
        x = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=dtype)
        index = torch.tensor([4, 1, 3, 0], device=device)
        # Include the smallest positive subnormal and -0 at different selected rows.
        bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        smallest = torch.tensor(1, dtype=bits_dtype).view(dtype).item()
        grad = torch.tensor([smallest, -0.0, -1.0, 2.0], device=device, dtype=dtype)
        output, dx = vjp(op, x.to(device), index, grad, dim=0)
        assert_bits(output.cpu(), x.index_select(0, index.cpu()))
        assert_bits(dx.cpu(), torch.tensor([2.0, 0.0, 0.0, -1.0, smallest], dtype=dtype))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((7,), 0),
            ((7, 9), 0),
            ((2, 7), -1),
            ((2, 7, 9), 1),
            ((7, 3, 9), 0),
            ((2, 3, 7, 9), 2),
            ((7, 2, 3, 9), -4),
            ((2, 3, 7), -1),
        ],
    )
    def test_axes_strides_unique_vjp_and_accuracy(self, backend, dtype, shape, dim):
        """H3, BAGEL and arbitrary axes use the same bitwise and accuracy contracts."""
        op, device = self.operator(backend)
        generator = torch.Generator(device=device).manual_seed(680)
        # Reverse physical axes and slice with a storage offset: neither prefix
        # nor suffix dimensions can be assumed contiguous by the kernel.
        storage_shape = list(reversed(shape))
        storage_shape[-1] *= 2
        x = torch.randn(storage_shape, device=device, dtype=dtype, generator=generator)
        x = x[..., 1::2].permute(tuple(reversed(range(len(shape)))))
        index = torch.tensor([0, 6, 0, 1, 0, 4, 0, 0], device=device, dtype=torch.int32)[1::2]
        output_shape = list(shape)
        output_shape[dim] = index.numel()
        grad = torch.randn(
            tuple(reversed(output_shape)),
            device=device,
            dtype=dtype,
            generator=generator,
        ).permute(tuple(reversed(range(len(shape)))))
        output, dx = vjp(op, x, index, grad, dim=dim)
        assert output.is_contiguous() and dx.dtype == x.dtype
        assert_bits(output, torch.index_select(x, dim, index))
        assert_bits(dx, expected_gradient(x, index, grad, dim=dim).to(dtype))
        # Also exercise the public FP32 reference at the project's tolerance.
        gold = vjp(NativeIndexSelectOp().forward_fp32, x.float(), index, grad.float(), dim=dim)[1]
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(dx.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        with torch.inference_mode():
            assert_bits(op(x, index, dim=dim, unique_indices=True), output)
        assert_bits(vjp(op, x, index, grad, dim=dim)[1], dx)

    @pytest.mark.parametrize("contiguous_x", [True, False])
    @pytest.mark.parametrize("contiguous_index", [True, False])
    def test_input_index_and_gradient_layouts_are_independent(
        self, backend, contiguous_x, contiguous_index
    ):
        op, device = self.operator(backend)
        x = torch.arange(2 * 3 * 5 * 7 * 6, device=device, dtype=torch.float32)
        x = x.reshape(2, 3, 5, 7, 6)[..., 1::2]
        # Both index views select [4, 0, 2] and start at a nonzero storage offset.
        if contiguous_index:
            index = torch.tensor([1, 4, 0, 2], device=device)[1:]
        else:
            index = torch.tensor([1, 4, 1, 0, 1, 2], device=device)[1::2]
        # Both views start out strided. Make exactly one contiguous so each
        # launch must choose its address calculation from the tensor it reads.
        grad = x[:, :, 1:4]
        if contiguous_x:
            x = x.contiguous()
        else:
            grad = grad.contiguous()
        y, dx = vjp(op, x, index, grad, dim=-3)
        assert_bits(y, x.index_select(-3, index))
        assert_bits(dx, expected_gradient(x, index, grad, dim=-3))

    def test_batch_train_infer_and_row_relabeling(self, backend):
        op, device = self.operator(backend)
        x = torch.randn(3, 73, 96, device=device)
        index = torch.tensor([72, 1, 0, 54, 2, 35], device=device)
        grad = torch.randn(3, 6, 96, device=device)
        output, dx = vjp(op, x, index, grad)
        with torch.inference_mode():
            assert_bits(op(x, index, dim=1, unique_indices=True), output)
        for batch in range(3):
            single_y, single_dx = vjp(op, x[batch : batch + 1], index, grad[batch : batch + 1])
            assert_bits(single_y, output[batch : batch + 1])
            assert_bits(single_dx, dx[batch : batch + 1])
        permutation = torch.tensor([2, 0, 1], device=device)
        py, pdx = vjp(op, x[permutation], index, grad[permutation])
        assert_bits(py, output[permutation])
        assert_bits(pdx, dx[permutation])
        # Relabel the source rows and map the original positions to their new rows.
        rows = torch.randperm(73, device=device)
        ry, rdx = vjp(op, x[:, rows], torch.argsort(rows)[index], grad)
        assert_bits(ry, output)
        assert_bits(rdx, dx[:, rows])
        changed = x.clone()
        changed[0] *= 123
        assert_bits(op(changed, index, dim=1, unique_indices=True)[1:], output[1:])

    @pytest.mark.parametrize(
        "shape,dim,count",
        [
            ((0,), 0, 0),
            ((7,), -1, 0),
            ((0, 7), 1, 2),
            ((7, 0), 0, 2),
            ((2, 0, 3), 1, 0),
        ],
    )
    def test_empty_axes(self, backend, shape, dim, count):
        op, device = self.operator(backend)
        x = torch.zeros(shape, device=device)
        index = torch.arange(count, device=device, dtype=torch.int64)
        expected = x.index_select(dim, index)
        output, dx = vjp(op, x, index, torch.ones_like(expected), dim=dim)
        assert_bits(output, expected)
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize("dim", [-3, 2])
    def test_invalid_dim(self, backend, dim):
        op, device = self.operator(backend)
        with pytest.raises(IndexError, match="dim must be"):
            op(
                torch.zeros(2, 7, device=device),
                torch.tensor([0], device=device),
                dim=dim,
                unique_indices=True,
            )


@requires_cuda
class TestCuda:
    @staticmethod
    def op():
        from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

        return TritonIndexSelectOp()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "width,index_dtype",
        [(32, torch.int32), (96, torch.int64), (7, torch.int64), (513, torch.int32)],
    )
    def test_strided_unique_indices_and_partial_tiles(self, dtype, width, index_dtype):
        torch.manual_seed(420)
        x = torch.randn(2, 67, 2 * width, device="cuda", dtype=dtype)[..., ::2]
        index = torch.randperm(67, device="cuda", dtype=index_dtype)[1::2]
        grad = torch.randn(2, width, index.numel(), device="cuda", dtype=dtype).transpose(1, 2)
        output, dx = vjp(self.op(), x, index, grad)
        assert output.is_contiguous()
        assert_bits(output, x.index_select(1, index))
        gold = expected_gradient(x, index, grad)
        assert_bits(dx, gold.to(dtype))
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(dx.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        for _ in range(3):
            assert_bits(vjp(self.op(), x, index, grad)[1], dx)

    @pytest.mark.parametrize(
        "shape,count",
        [
            ((2, 0, 32), 0),
            ((2, 5, 96), 0),
            ((0, 5, 32), 3),
            ((2, 5, 0), 3),
            ((1, 1, 1), 1),
        ],
    )
    def test_empty_and_boundary(self, shape, count):
        x = torch.empty(shape, device="cuda")
        index = torch.arange(count, device="cuda", dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2], device="cuda")
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, expected_gradient(x, index, grad))

    @pytest.mark.parametrize("width", [32, 96])
    @pytest.mark.parametrize("sequence", [38734, 110750, 367986])
    def test_realistic_h3_unique_selection(self, width, sequence):
        # Full mixed-row counts from the merged output-head qualification.
        x = torch.randn(1, sequence, width, device="cuda")
        index = torch.arange(sequence - 1, -1, -3, device="cuda")
        grad = torch.randn(1, index.numel(), width, device="cuda")
        grad[0, 0, 0] = -0.0
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        # PyTorch adds each contribution to +0, including the selected negative zero.
        expected = torch.zeros_like(x)
        expected.index_add_(1, index, grad)
        assert_bits(dx, expected)

    def test_checkpoint_recomputation(self):
        from torch.utils.checkpoint import checkpoint

        op = self.op()
        x = torch.randn(2, 7, 9, device="cuda")
        index = torch.tensor([6, 1, 0], device="cuda")
        grad = torch.randn_like(x.index_select(1, index))
        output, dx = vjp(op, x, index, grad)
        cy, cdx = vjp(
            lambda a, i, **kwargs: checkpoint(op, a, i, **kwargs, use_reentrant=False),
            x,
            index,
            grad,
        )
        assert_bits(cy, output)
        assert_bits(cdx, dx)

    def test_expanded_input_and_gradient(self):
        x = torch.randn(1, 1, 7, device="cuda").expand(2, 9, 7)
        index = torch.tensor([8, 0, 4], device="cuda")
        grad = torch.ones(1, 1, 1, device="cuda").expand(2, 3, 7)
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, expected_gradient(x, index, grad))

    def test_capped_grid_covers_rows_and_tiles(self, monkeypatch):
        from rl_engine.backends.shared.triton.embedding import index_select

        # Eight logical rows and three tiles per row must fit a (3, 2) launch.
        monkeypatch.setattr(index_select, "_MAX_GRID_X", 3)
        monkeypatch.setattr(index_select, "_MAX_GRID_Y", 2)
        width = 2 * index_select._BLOCK_SIZE + 1
        x = torch.arange(2 * 6 * width, device="cuda", dtype=torch.float32).reshape(2, 6, width)
        index = torch.tensor([5, 1, 4, 0], device=x.device)
        grad = x[:, 1:5]

        output, dx = vjp(self.op(), x, index, grad)
        expected_dx = torch.zeros_like(x)
        expected_dx.index_copy_(1, index, grad)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, expected_dx)

    def test_inner_tiles_exceed_cuda_grid_y_limit(self):
        # Shared selection can copy much larger slices than H3's head widths.
        width = 65535 * 256 + 1
        x = torch.ones(1, 2, 1, device="cuda", dtype=torch.float16).expand(1, 2, width)
        index = torch.tensor([1], device="cuda")
        grad = torch.ones(1, 1, 1, device="cuda", dtype=x.dtype).expand(1, 1, width)
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, grad)
        assert_bits(dx[:, 1:2], grad)
        assert torch.count_nonzero(dx[:, 0]) == 0

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(2, 67, 96, device="cuda")
            index = torch.arange(66, -1, -2, device="cuda")
            grad = torch.randn(2, index.numel(), 96, device="cuda")
            output, dx = vjp(self.op(), x, index, grad)
        torch.cuda.current_stream().wait_stream(stream)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, expected_gradient(x, index, grad))

    def test_device_guards_and_deterministic_mode(self):
        from rl_engine.runtime.registry import KernelRegistry

        op = KernelRegistry().get_op("index_select", device="cuda")
        x = torch.randn(1, 7, 32, device="cuda")
        index = torch.tensor([6, 0, 3], device="cuda")
        grad = torch.randn(1, 3, 32, device="cuda")
        expected = vjp(op, x, index, grad)
        old = torch.are_deterministic_algorithms_enabled()
        old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = vjp(op, x, index, grad)
        finally:
            torch.use_deterministic_algorithms(old, warn_only=old_warn_only)
        for a, b in zip(actual, expected, strict=True):
            assert_bits(a, b)
        with pytest.raises(ValueError, match="same device"):
            op(x, index.cpu(), dim=1, unique_indices=True)
        with pytest.raises(RuntimeError, match="got 'cpu'"):
            op(x.cpu(), index.cpu(), dim=1, unique_indices=True)

    def test_h3_example_uses_the_registered_operator(self, capsys):
        """Exercise both H3 projection widths through the documented registry entry."""
        from examples.models.minimax_h3.packed_index_select import main

        main()
        output = capsys.readouterr().out
        assert "video=(2, 5, 96), audio=(2, 4, 32)" in output
        assert "backend=TritonIndexSelectOp" in output

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA or ROCm GPUs")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_noncurrent_device_and_restoration(self, dtype, dim):
        op = self.op()
        with torch.cuda.device(0):
            x = torch.randn(7, 7, 9, device="cuda:1", dtype=dtype)
            index = torch.tensor([6, 0, 3], device="cuda:1")
            grad = torch.randn_like(x.index_select(dim, index))
            output, dx = vjp(op, x, index, grad, dim=dim)
            assert torch.cuda.current_device() == 0
            assert output.device == dx.device == x.device
            assert_bits(output, x.index_select(dim, index))
            assert_bits(dx, expected_gradient(x, index, grad, dim=dim).to(dtype))

    @pytest.mark.parametrize(
        "shape,bad_index",
        [
            ((1, 7, 32), -1),
            ((1, 7, 32), 7),
            ((1, 7, 32), 2**63 - 1),
            ((0, 7, 32), 7),
            ((1, 7, 0), -1),
        ],
    )
    def test_invalid_cuda_index_fails_in_isolated_process(self, shape, bad_index):
        code = f"""
import torch
from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp
x = torch.zeros({shape!r}, device='cuda')
i = torch.tensor([{bad_index}], device='cuda')
TritonIndexSelectOp()(x, i, dim=1, unique_indices=True)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "TRITON_DEBUG": "0"},
        )
        assert result.returncode != 0
        assert "device-side assert" in result.stderr or "index values must be" in result.stderr
