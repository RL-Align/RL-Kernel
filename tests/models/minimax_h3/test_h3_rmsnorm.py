# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``h3_rmsnorm``: RMSNorm (eps 1e-5) and its fused AdaLN modulation.

* forward is bitwise equal to ``nn.RMSNorm`` and to diffusers'
  ``norm(x) * (1.0 + scale[i]) + shift[i]`` (block: ``adaln_indices``;
  ``norm_out``: ``timestep_indices``) on the pinned norm weights;
* rows are independent of batch size and position;
* backward is deterministic and checked against an FP64 golden;
* malformed inputs fail closed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.reference.minimax_h3.rmsnorm import NativeH3RMSNormOp
from rl_engine.validation.models.h3_cases import h3_packed_layout
from rl_engine.validation.models.h3_provider import provider_norm_modulate

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
NORMS = (
    "transformer_blocks.0.norm1.weight",
    "transformer_blocks.0.norm2.weight",
    "token_refiner.final_norm.weight",
    "norm_out.norm.weight",
)


def _cuda_op():
    from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import (
        H3RMSNormCudaOp,
        h3_rmsnorm_available,
    )

    if not h3_rmsnorm_available():
        pytest.skip("rl_engine._C lacks h3_rmsnorm_*")
    return H3RMSNormCudaOp()


def _x(shape, dtype=torch.bfloat16, seed=0, scale=2.0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(device=device, dtype=dtype)


def _table(rows, hidden, chunks, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    table = (torch.randn(rows, chunks * hidden, generator=g) * 0.5).to(dtype).cuda()
    return table.chunk(chunks, dim=-1)  # strided (R, H) views, like the AdaLN outputs


def _fp64_modulated(x, weight, shift, scale, index, eps=1e-5):
    x64 = x.double()
    n = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps) * weight.double()
    return n * (1 + scale.double().index_select(0, index)) + shift.double().index_select(0, index)


class TestReference:
    def test_golden_close_to_provider(self):
        x, w = _x((3, 64), torch.float32, device="cpu"), torch.rand(64) + 0.5
        op = NativeH3RMSNormOp()
        torch.testing.assert_close(op.forward(x, w), op.forward_fp32(x, w), atol=1e-6, rtol=1e-6)

    def test_modulated_with_prevalidated_indices(self):
        x, w = _x((2, 8), torch.float32, device="cpu"), torch.ones(8)
        shift, scale = torch.randn(3, 8), torch.randn(3, 8)
        index = torch.tensor([2, 0])
        op = NativeH3RMSNormOp()
        expected = op.forward(x, w) * (1 + scale[index]) + shift[index]
        torch.testing.assert_close(
            op.forward_modulated(x, w, shift, scale, index, check_range=False),
            expected,
        )

    def test_modulation_backward_is_bitwise_the_per_row_scan_it_replaced(self):
        from rl_engine.reference.minimax_h3.rmsnorm import _DeterministicModulation

        g = torch.Generator(device="cpu").manual_seed(11)
        n = torch.randn(300, 16, generator=g)
        shift, scale = torch.randn(7, 16, generator=g), torch.randn(7, 16, generator=g)
        index = torch.randint(0, 5, (300,), generator=g)  # rows 5 and 6 are never referenced
        grad = torch.randn(300, 16, generator=g)
        leaves = [t.clone().requires_grad_(True) for t in (n, shift, scale)]
        _DeterministicModulation.apply(*leaves, index).backward(grad)

        per_shift = grad.reshape(-1, 300, 16).sum(dim=0)
        per_scale = (grad * n).reshape(-1, 300, 16).sum(dim=0)
        d_shift, d_scale = torch.zeros(7, 16), torch.zeros(7, 16)
        for row in range(7):  # the loop of the earlier implementation
            positions = torch.nonzero(index == row).flatten()
            if positions.numel():
                d_shift[row] = per_shift.index_select(0, positions).sum(dim=0)
                d_scale[row] = per_scale.index_select(0, positions).sum(dim=0)
        assert torch.equal(leaves[1].grad, d_shift) and torch.equal(leaves[2].grad, d_scale)

    def test_rejects_bad_inputs(self):
        op = NativeH3RMSNormOp()
        x, w = torch.randn(2, 8), torch.ones(8)
        with pytest.raises(TypeError):
            op.forward(x, w.double())
        with pytest.raises(TypeError):
            op.forward(x, w.bfloat16())
        with pytest.raises(ValueError):
            op.forward(x, torch.ones(7))
        with pytest.raises(ValueError):
            op.forward(x, w, eps=0.0)
        shift, scale = torch.zeros(3, 8), torch.zeros(3, 8)
        with pytest.raises(IndexError):
            op.forward_modulated(x, w, shift, scale, torch.tensor([0, 3]))
        with pytest.raises(IndexError):  # the golden takes check_range like the provider path
            op.forward_modulated_fp32(x, w, shift, scale, torch.tensor([0, 3]))
        op.forward_modulated_fp32(x, w, shift, scale, torch.tensor([0, 2]), check_range=False)
        with pytest.raises(ValueError, match="is on meta"):  # a device mismatch is a ValueError
            op.forward_modulated(x, w, shift.to("meta"), scale, torch.tensor([0, 1]))
        with pytest.raises(TypeError, match="dtype"):  # a dtype mismatch stays a TypeError
            op.forward_modulated(x, w, shift.double(), scale, torch.tensor([0, 1]))
        with pytest.raises(ValueError):  # one index per position S
            op.forward_modulated(x, w, shift, scale, torch.tensor([0]))


@requires_cuda
class TestCudaForward:
    @pytest.mark.parametrize("name", NORMS)
    def test_bitwise_equal_to_nn_rmsnorm_on_pinned_weights(self, h3_weights_cpu, name):
        weight = h3_weights_cpu[name].cuda()
        module = torch.nn.RMSNorm(5376, eps=1e-5).cuda().bfloat16()
        module.weight.data.copy_(weight)
        x = _x((2, 777, 5376), seed=1)
        assert torch.equal(_cuda_op()(x, weight), module(x))

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("hidden", [12, 256, 4096, 5376])
    def test_bitwise_equal_to_f_rms_norm(self, dtype, hidden):
        x = _x((5, 33, hidden), dtype, seed=hidden, scale=30.0)
        w = (torch.rand(hidden, device="cuda") * 2).to(dtype)
        ref = torch.nn.functional.rms_norm(x, (hidden,), w, 1e-5)
        assert torch.equal(_cuda_op()(x, w), ref)

    def test_block_modulation_bitwise_equal_to_diffusers(self, h3_weights_cpu):
        weight = h3_weights_cpu["transformer_blocks.0.norm1.weight"].cuda()
        shift, scale, *_ = _table(9, 5376, 6, seed=2)  # (3T, H) rows, T = 3
        ti, tags = h3_packed_layout(4097, 3, seed=2)
        index = ti * 3 + tags
        x = _x((2, 4097, 5376), seed=2)
        ours = _cuda_op().forward_modulated(x, weight, shift, scale, index)
        assert torch.equal(ours, provider_norm_modulate(x, weight, shift, scale, index))

    def test_norm_out_modulation_bitwise_equal_to_diffusers(self, h3_weights_cpu):
        weight = h3_weights_cpu["norm_out.norm.weight"].cuda()
        shift, scale = _table(3, 5376, 2, seed=3)  # (T, H) rows indexed by timestep
        ti, _ = h3_packed_layout(1000, 3, seed=3)
        x = _x((1, 1000, 5376), seed=3)
        ours = _cuda_op().forward_modulated(x, weight, shift, scale, ti)
        assert torch.equal(ours, provider_norm_modulate(x, weight, shift, scale, ti))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_modulation_bitwise_in_every_dtype(self, dtype):
        """FP32 must not contract the separately rounded eager ops into an FMA."""

        w = (torch.rand(5376, device="cuda") + 0.5).to(dtype)
        shift, scale, *_ = _table(9, 5376, 6, dtype=dtype, seed=9)
        ti, tags = h3_packed_layout(333, 3, seed=9)
        index = ti * 3 + tags
        x = _x((2, 333, 5376), dtype, seed=9)
        ours = _cuda_op().forward_modulated(x, w, shift, scale, index)
        assert torch.equal(ours, provider_norm_modulate(x, w, shift, scale, index))

    def test_overlapping_table_rows_are_copied_and_wider_table_views_are_not(self):
        w = (torch.rand(5376, device="cuda") + 0.5).bfloat16()
        table = _x((9, 2 * 5376), seed=21, scale=0.5)
        wide_shift, wide_scale = table.chunk(2, dim=-1)  # row stride 2H, handed over as views
        ti, tags = h3_packed_layout(100, 3, seed=21)
        index = ti * 3 + tags
        x = _x((1, 100, 5376), seed=21)
        op = _cuda_op()
        expected = op.forward_modulated(
            x, w, wide_shift.contiguous(), wide_scale.contiguous(), index
        )
        assert torch.equal(op.forward_modulated(x, w, wide_shift, wide_scale, index), expected)
        # An expanded table (row stride 0) is a table of equal rows: copied, not rejected.
        row = _x((1, 5376), seed=22, scale=0.5)
        expanded = row.expand(9, 5376)
        assert expanded.stride(0) == 0
        dense = row.repeat(9, 1)
        assert torch.equal(
            op.forward_modulated(x, w, expanded, expanded, index),
            op.forward_modulated(x, w, dense, dense, index),
        )

    def test_rows_are_batch_and_position_invariant(self):
        w = (torch.rand(5376, device="cuda") + 0.5).bfloat16()
        shift, scale, *_ = _table(9, 5376, 6, seed=4)
        ti, tags = h3_packed_layout(300, 3, seed=4)
        index = ti * 3 + tags
        x = _x((3, 300, 5376), seed=4)
        full = _cuda_op().forward_modulated(x, w, shift, scale, index)
        part = _cuda_op().forward_modulated(x[2:3, 50:90], w, shift, scale, index[50:90])
        assert torch.equal(part[0], full[2, 50:90])

    def test_rejects_unsupported(self):
        w = torch.ones(6, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="multiple of 4"):
            _cuda_op()(_x((2, 6)), w)
        with pytest.raises(ValueError):
            _cuda_op()(_x((2, 8)).cpu(), torch.ones(8).bfloat16())

    def test_rejects_a_row_wider_than_the_int_the_kernels_count_with(self):
        """N goes to the kernels as ``int``: forward and backward both refuse it up front."""

        from rl_engine.backends.extension import _C

        _cuda_op()
        n = 2**31  # a multiple of 4 above INT_MAX
        if torch.cuda.mem_get_info()[0] < 10 * 2**30:
            pytest.skip("needs about 8 GiB of free device memory for the two operands")
        x = torch.empty(1, n, device="cuda", dtype=torch.float16)
        w = torch.empty(n, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="too large"):
            _C.h3_rmsnorm_forward(x, w, 1e-6)
        with pytest.raises(RuntimeError, match="too large"):
            _C.h3_rmsnorm_backward(x, x, w, torch.empty(1, device="cuda"))


_DEVICE_MESSAGES = {
    "weight": "with x's dtype and device",
    "index": r"row index must be a non-empty contiguous int64 tensor on cuda:0",
    "rstd": "on x's device",
}


@requires_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
class TestCudaNativeDeviceValidation:
    @pytest.mark.parametrize("argument", ["weight", "index"])
    def test_forward_rejects_different_device(self, argument):
        from rl_engine.backends.extension import _C

        _cuda_op()
        x = torch.ones(2, 8, device="cuda:0", dtype=torch.bfloat16)
        weight = torch.ones(8, device="cuda:0", dtype=x.dtype)
        shift, scale = torch.zeros(2, 16, device="cuda:0", dtype=x.dtype).chunk(2, dim=-1)
        index = torch.tensor([0, 1], device="cuda:0")
        if argument == "weight":
            weight = weight.to("cuda:1")
        else:
            index = index.to("cuda:1")
        with pytest.raises(RuntimeError, match=_DEVICE_MESSAGES[argument]):
            _C.h3_rmsnorm_forward(x, weight, 1e-5, shift, scale, index)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

    @pytest.mark.parametrize("argument", ["weight", "index", "rstd"])
    def test_backward_rejects_different_device(self, argument):
        from rl_engine.backends.extension import _C

        _cuda_op()
        x = torch.ones(2, 8, device="cuda:0", dtype=torch.bfloat16)
        weight = torch.ones(8, device="cuda:0", dtype=x.dtype)
        shift, scale = torch.zeros(2, 16, device="cuda:0", dtype=x.dtype).chunk(2, dim=-1)
        index = torch.tensor([0, 1], device="cuda:0")
        _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5, shift, scale, index)
        if argument == "weight":
            weight = weight.to("cuda:1")
        elif argument == "index":
            index = index.to("cuda:1")
        else:
            rstd = rstd.to("cuda:1")
        with pytest.raises(RuntimeError, match=_DEVICE_MESSAGES[argument]):
            _C.h3_rmsnorm_backward(torch.ones_like(x), x, weight, rstd, shift, scale, index)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)


@requires_cuda
class TestCudaBackward:
    def _grads(self, fn, tensors, grad):
        leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
        fn(*leaves).backward(grad)
        return [leaf.grad for leaf in leaves]

    def test_modulated_backward_against_fp64(self):
        g = torch.Generator(device="cpu").manual_seed(5)
        w = (torch.rand(5376, generator=g) + 0.5).bfloat16().cuda()
        table = (torch.randn(9, 2 * 5376, generator=g) * 0.5).bfloat16().cuda()
        ti, tags = h3_packed_layout(2049, 3, seed=5)
        index = ti * 3 + tags
        x = _x((2, 2049, 5376), seed=5)
        grad = _x((2, 2049, 5376), seed=6, scale=1.0)
        op = _cuda_op()

        def ours(x_, w_, t_):
            shift, scale = t_.chunk(2, dim=-1)
            return op.forward_modulated(x_, w_, shift, scale, index)

        def golden(x_, w_, t_):
            shift, scale = t_.chunk(2, dim=-1)
            return _fp64_modulated(x_, w_, shift, scale, index)

        got = self._grads(ours, (x, w, table), grad)
        ref = self._grads(golden, (x.double(), w.double(), table.double()), grad.double())
        for name, g_, r in zip(("dx", "dweight", "dtable"), got, ref):
            rel = ((g_.double() - r).abs().max() / r.abs().max()).item()
            # Two BF16 roundings remain: the eager VJP uses round(1 + scale), and
            # every gradient is rounded once to BF16 (2^-9 relative each).
            assert rel < 1e-2, (name, rel)
        again = self._grads(ours, (x, w, table), grad)
        assert all(torch.equal(a, b) for a, b in zip(got, again))

    @pytest.mark.parametrize("seq", [257, 1024, 4097])
    def test_modulated_gradients_meet_contract_against_golden(self, seq):
        # Against forward_modulated_fp32 before it stored norm(x) and 1 + scale
        # in BF16, d_weight missed from S=1024 and d_scale at S=4097.
        from rl_engine.contracts.numerical import load_contract, resolve_tolerance

        spec = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=torch.bfloat16,
        )
        g = torch.Generator(device="cpu").manual_seed(seq)
        w = (torch.rand(5376, generator=g) + 0.5).bfloat16().cuda()
        table = (torch.randn(9, 2 * 5376, generator=g) * 0.1).bfloat16().cuda()
        ti, tags = h3_packed_layout(seq, 3, seed=seq)
        index = ti * 3 + tags
        x = _x((1, seq, 5376), seed=seq)
        grad = _x((1, seq, 5376), seed=seq + 1, scale=1.0)
        op, native = _cuda_op(), NativeH3RMSNormOp()

        def run(fn, upstream):
            def call(x_, w_, t_):
                shift, scale = t_.chunk(2, dim=-1)
                return fn(x_, w_, shift, scale, index)

            return self._grads(call, (x, w, table), upstream)

        got = run(op.forward_modulated, grad)
        ref = run(native.forward_modulated_fp32, grad.float())
        for name, g_, r in zip(("dx", "dweight", "dtable"), got, ref):
            torch.testing.assert_close(
                g_.float(), r.float(), atol=spec.atol, rtol=spec.rtol, msg=lambda m: f"{name}: {m}"
            )

    def test_plain_backward_against_fp64_and_row_local_dx(self):
        w = (torch.rand(5376, generator=torch.Generator().manual_seed(7)) + 0.5).bfloat16().cuda()
        x = _x((4, 64, 5376), seed=7)
        grad = _x((4, 64, 5376), seed=8, scale=1.0)
        op = _cuda_op()
        got = self._grads(lambda a, b: op(a, b), (x, w), grad)

        def golden(a, b):
            return a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-5) * b

        ref = self._grads(golden, (x.double(), w.double()), grad.double())
        for g, r in zip(got, ref):
            assert ((g.double() - r).abs().max() / r.abs().max()).item() < 5e-3
        part = self._grads(lambda a, b: op(a, b), (x[1:2, 10:20], w), grad[1:2, 10:20])
        assert torch.equal(part[0][0], got[0][1, 10:20])

    def test_registry_dispatches_cuda(self):
        from rl_engine.runtime.registry import KernelRegistry

        _cuda_op()
        assert (
            type(KernelRegistry().get_op("h3_rmsnorm", device="cuda")).__name__ == "H3RMSNormCudaOp"
        )


def _index_probe(entry: str, bad: int):
    """Run one native entry point with a table-row index outside the table, in a child process."""

    probe = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine.backends.extension import _C

        entry, bad = sys.argv[1], int(sys.argv[2])
        rows, hidden, table_rows = 600, 64, 4
        x = torch.randn(rows, hidden, device="cuda").bfloat16()
        w = torch.ones(hidden, device="cuda").bfloat16()
        table = torch.randn(table_rows, 2 * hidden, device="cuda").bfloat16()
        shift, scale = table.chunk(2, dim=-1)
        index = torch.arange(rows, device="cuda") % table_rows
        index[rows - 3] = bad  # the last tile of rows, past the first dx blocks
        if entry == "forward":
            _C.h3_rmsnorm_forward(x, w, 1e-5, shift, scale, index)
        else:
            from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
                _segment_tiles,
            )
            valid = index.clamp(0, table_rows - 1)
            y, rstd = _C.h3_rmsnorm_forward(x, w, 1e-5, shift, scale, valid)
            tiles = _segment_tiles(valid, table_rows)
            _C.h3_rmsnorm_backward(torch.ones_like(x), x, w, rstd, shift, scale, index, *tiles)
        torch.cuda.synchronize()
        """
    )
    return subprocess.run(
        [sys.executable, "-c", probe, entry, str(bad)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )


@requires_cuda
@pytest.mark.parametrize("entry", ["forward", "backward"])
@pytest.mark.parametrize("bad", [-1, 4, 10**9], ids=["negative", "past-table", "far"])
def test_a_modulation_index_outside_the_table_stops_the_kernels(entry, bad):
    """check_range=False and direct calls get a device-side assert, not an out-of-bounds read.

    The dweight partials read the index inside their row loop, where an assert costs about 11% of
    the backward; the entry point checks the index values once before launching them.
    """

    _cuda_op()
    assert _index_probe(entry, 1).returncode == 0  # the control: the same call with a valid index
    done = _index_probe(entry, bad)
    assert done.returncode != 0, entry
    assert "device-side assert triggered" in done.stderr, done.stderr
