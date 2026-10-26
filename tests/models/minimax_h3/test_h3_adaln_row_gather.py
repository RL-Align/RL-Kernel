# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``adaln_row_gather``: ``timestep_index * 3 + token_tag`` lookup of six tensors.

* forward is a copy: bitwise equal to diffusers' six ``index_select`` calls,
  for every packing, length and index dtype;
* semantic indices fail closed (out-of-range tags or timesteps, RFC probes
  H2/H3), and flipping a tag or offsetting a timestep picks a different row;
* backward is a deterministic FP32 segmented sum: repeat-bitwise and
  correctly rounded, unlike the atomic BF16 ``index_select`` backward;
* batch invariance: a position's output does not depend on which other
  positions share the call, and a row's gradient does not depend on the
  positions that reference other rows.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.reference.minimax_h3.adaln_row_gather import NativeH3AdaLNRowGatherOp
from rl_engine.validation.models.h3_cases import h3_packed_layout
from rl_engine.validation.models.h3_provider import provider_adaln_row_gather

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _cuda_op():
    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
        H3AdaLNRowGatherCudaOp,
        adaln_row_gather_available,
    )

    if not adaln_row_gather_available():
        pytest.skip("rl_engine._C lacks h3_adaln_row_gather_*")
    return H3AdaLNRowGatherCudaOp()


def _rows(num_timesteps, hidden=5376, dtype=torch.bfloat16, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(3 * num_timesteps, 6 * hidden, generator=g).to(dtype).to(device)


class TestReference:
    def test_matches_diffusers_index_select(self):
        rows = _rows(2, hidden=4, device="cpu")
        ti, tags = h3_packed_layout(11, 2, device="cpu")
        index = ti * 3 + tags
        chunks = rows.chunk(6, dim=-1)
        for ours, chunk in zip(NativeH3AdaLNRowGatherOp().forward(rows, ti, tags), chunks):
            assert torch.equal(ours, chunk.index_select(0, index))

    @pytest.mark.parametrize(
        "mutate, error",
        [
            (lambda ti, tags: (ti, tags.clone().fill_(3)), IndexError),  # H2: no 4th modality
            (lambda ti, tags: (ti, tags - 1), IndexError),
            (lambda ti, tags: (ti + 2, tags), IndexError),  # H3: timestep offset past T
            (lambda ti, tags: (ti, tags[:-1]), ValueError),
            (lambda ti, tags: (ti[:0], tags[:0]), ValueError),
            (lambda ti, tags: (ti.float(), tags), TypeError),
            (lambda ti, tags: (ti.int(), tags), TypeError),  # mixed index dtypes
        ],
    )
    def test_rejects_invalid_indices(self, mutate, error):
        rows = _rows(2, hidden=4, device="cpu")
        ti, tags = mutate(*h3_packed_layout(9, 2, device="cpu"))
        with pytest.raises(error):
            NativeH3AdaLNRowGatherOp().forward(rows, ti, tags)

    def test_rejects_bad_rows(self):
        ti, tags = h3_packed_layout(4, 1, device="cpu")
        with pytest.raises(ValueError):
            NativeH3AdaLNRowGatherOp().forward(torch.zeros(4, 24), ti, tags)  # not 3 per t
        with pytest.raises(ValueError):
            NativeH3AdaLNRowGatherOp().forward(torch.zeros(3, 25), ti, tags)  # not 6 * H


def test_backward_positions_are_a_whole_number_of_index_repeats():
    from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import _repeat_positions

    index = torch.tensor([2, 0, 1])
    assert _repeat_positions(index, 6).tolist() == [2, 0, 1, 2, 0, 1]
    with pytest.raises(ValueError, match="whole number of repeats"):
        _repeat_positions(index, 7)
    with pytest.raises(ValueError, match="whole number of repeats"):
        _repeat_positions(index[:0], 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_positions_by_row_equal_the_per_row_scan(device):
    from rl_engine.reference.minimax_h3.adaln_row_gather import positions_by_row

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    g = torch.Generator(device="cpu").manual_seed(0)
    index = torch.randint(0, 7, (500,), generator=g).to(device)  # rows 7..11 are never referenced
    positions = positions_by_row(index, 12)
    assert len(positions) == 12
    for row, found in enumerate(positions):
        assert torch.equal(found, torch.nonzero(index == row).flatten())


@requires_cuda
def test_reference_backward_is_bitwise_the_per_row_scan_it_replaced():
    from rl_engine.reference.minimax_h3.adaln_row_gather import _DeterministicRowGather

    rows = _rows(3, hidden=64, seed=7)
    ti, tags = h3_packed_layout(777, 3, seed=7)
    index = ti.long() * 3 + tags.long()
    gen = torch.Generator(device="cuda").manual_seed(7)
    grad = torch.randn(6, 777, 64, device="cuda", generator=gen).bfloat16()

    leaf = rows.detach().clone().requires_grad_(True)
    outputs = _DeterministicRowGather.apply(leaf, index)
    outputs.backward(grad)

    grad32 = grad.float()
    expected = grad32.new_zeros((rows.shape[0], 6, 64))
    for row in range(rows.shape[0]):  # the loop of the earlier implementation
        positions = torch.nonzero(index == row).flatten()
        if positions.numel():
            expected[row] = grad32.index_select(1, positions).sum(dim=1)
    assert torch.equal(leaf.grad, expected.reshape(rows.shape[0], -1).to(rows.dtype))


@requires_cuda
class TestCudaForward:
    @pytest.mark.parametrize("seq", [1, 2, 3, 257, 4097, 32768])
    @pytest.mark.parametrize("num_timesteps", [1, 3])
    def test_bitwise_equal_to_index_select(self, seq, num_timesteps):
        rows = _rows(num_timesteps, seed=seq)
        ti, tags = h3_packed_layout(seq, num_timesteps, seed=seq)
        ours = _cuda_op()(rows, ti, tags)
        ref = provider_adaln_row_gather(rows.chunk(6, dim=-1), ti, tags)
        assert len(ours) == 6
        for a, b in zip(ours, ref):
            assert a.shape == (seq, 5376) and a.is_contiguous() and torch.equal(a, b)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("hidden", [24, 5, 5376])
    def test_dtypes_and_unaligned_hidden(self, dtype, hidden):
        rows = _rows(2, hidden=hidden, dtype=dtype)
        ti, tags = h3_packed_layout(33, 2)
        for a, b in zip(
            _cuda_op()(rows, ti.int(), tags.int()), NativeH3AdaLNRowGatherOp()(rows, ti, tags)
        ):
            assert torch.equal(a, b)

    def test_strided_rows_view(self):
        table = _rows(4, hidden=16)
        rows = table[::2]  # row stride 2 * 6H, still 3 rows per timestep after slicing 12 -> 6
        ti, tags = h3_packed_layout(20, 2)
        for a, b in zip(_cuda_op()(rows, ti, tags), NativeH3AdaLNRowGatherOp()(rows, ti, tags)):
            assert torch.equal(a, b)

    def test_tag_flip_and_timestep_offset_select_other_rows(self):
        rows = _rows(2, hidden=16)
        ti, tags = h3_packed_layout(30, 2)
        base = _cuda_op()(rows, ti, tags)
        flipped = _cuda_op()(rows, ti, (tags + 1) % 3)  # H2
        shifted = _cuda_op()(rows, 1 - ti, tags)  # H3 within range
        assert not torch.equal(base[0], flipped[0])
        assert not torch.equal(base[0], shifted[0])
        # Every output row is exactly the addressed table row, nothing else.
        index = ti * 3 + tags
        for c, out in enumerate(base):
            assert torch.equal(out, rows[index, c * 16 : (c + 1) * 16])

    def test_rejects_out_of_range_on_device(self):
        rows = _rows(1, hidden=16)
        ti, tags = h3_packed_layout(8, 1)
        with pytest.raises(IndexError):
            _cuda_op()(rows, ti, tags + 3)

    @pytest.mark.parametrize("entrypoint", ["native", "unchecked_wrapper"])
    @pytest.mark.parametrize(
        "timestep, tag, index_dtype",
        [
            pytest.param(-1, 0, "int32", id="negative-timestep"),
            pytest.param(2, 0, "int64", id="timestep-past-table"),
            pytest.param(0, 3, "int32", id="tag-past-modality-valid-row"),
            pytest.param(1, -1, "int64", id="negative-tag-valid-row"),
            # Multiplication by 3 wraps this value to row 2 in signed int64.
            pytest.param((2**64 + 2) // 3, 0, "int64", id="timestep-overflow-valid-row"),
            pytest.param(0, 2**63 - 1, "int64", id="int64-max-tag"),
        ],
    )
    def test_native_bounds_assertions(self, entrypoint, timestep, tag, index_dtype):
        """Reject semantic index violations without poisoning pytest's CUDA context."""

        _cuda_op()
        probe = textwrap.dedent(
            """
            import sys
            import torch
            from rl_engine.backends.extension import _C
            from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
                H3AdaLNRowGatherCudaOp,
            )

            entrypoint, timestep, tag, index_dtype = sys.argv[1:]
            rows = torch.zeros((6, 6), dtype=torch.float32, device="cuda")
            dtype = getattr(torch, index_dtype)
            ti = torch.tensor([int(timestep)], dtype=dtype, device="cuda")
            tags = torch.tensor([int(tag)], dtype=dtype, device="cuda")
            if entrypoint == "native":
                _C.h3_adaln_row_gather_forward(rows, ti, tags, 6, 3)
            else:
                H3AdaLNRowGatherCudaOp().forward(rows, ti, tags, check_range=False)
            torch.cuda.synchronize()
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe, entrypoint, str(timestep), str(tag), index_dtype],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode != 0, "native gather accepted invalid semantic indices"
        assert (
            "device-side assert triggered" in completed.stderr
        ), f"expected a CUDA bounds assertion, got:\n{completed.stdout}\n{completed.stderr}"

    def test_gather_chunks_drop_in(self):
        rows = _rows(2, hidden=16)
        ti, tags = h3_packed_layout(10, 2)
        chunks = rows.chunk(6, dim=-1)
        for a, b in zip(_cuda_op().gather_chunks(chunks, ti, tags), _cuda_op()(rows, ti, tags)):
            assert torch.equal(a, b)


@requires_cuda
class TestCudaBackward:
    def _grads(self, seq, hidden, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return [torch.randn(seq, hidden, device="cuda", generator=g).bfloat16() for _ in range(6)]

    def _cuda_grad(self, rows, ti, tags, grads):
        leaf = rows.detach().clone().requires_grad_(True)
        torch.autograd.backward(list(_cuda_op()(leaf, ti, tags)), grads)
        return leaf.grad

    def test_correctly_rounded_fp32_segment_sum(self):
        rows = _rows(3, seed=1)
        ti, tags = h3_packed_layout(4097, 3, seed=1)
        grads = self._grads(4097, 5376)
        ours = self._cuda_grad(rows, ti, tags, grads)
        ref = rows.detach().double().requires_grad_(True)
        torch.autograd.backward(
            list(NativeH3AdaLNRowGatherOp().forward_fp32(ref, ti, tags)), [g.float() for g in grads]
        )
        assert ours.dtype == torch.bfloat16
        # tolerance_contract.json gradient_accuracy / elementwise / bfloat16.
        torch.testing.assert_close(ours.double(), ref.grad, atol=2e-2, rtol=1.6e-2)
        assert (ours == ref.grad.bfloat16()).float().mean() > 0.9999

    def test_repeat_bitwise(self):
        rows = _rows(2, hidden=512, seed=2)
        ti, tags = h3_packed_layout(5000, 2, seed=2)
        grads = self._grads(5000, 512, seed=2)
        first = self._cuda_grad(rows, ti, tags, grads)
        for _ in range(3):
            assert torch.equal(self._cuda_grad(rows, ti, tags, grads), first)

    def test_forward_rows_invariant_to_batch_size_and_position(self):
        rows = _rows(3, hidden=512, seed=5)
        ti, tags = h3_packed_layout(4097, 3, seed=5)
        full = _cuda_op()(rows, ti, tags)
        perm = torch.randperm(4097, generator=torch.Generator().manual_seed(5)).cuda()
        for pick in (
            torch.tensor([0]),
            torch.tensor([2048]),
            torch.arange(100, 357),
            torch.arange(4000, 4097),
            perm,
        ):
            pick = pick.cuda()
            part = _cuda_op()(rows, ti[pick], tags[pick])
            for a, b in zip(part, full):
                assert torch.equal(a, b[pick])

    def test_backward_row_gradient_independent_of_other_rows_tokens(self):
        # Each row's segment is tiled on its own, so a row's gradient from the
        # full packing equals the one from a packing of only its own positions
        # (same order), even though ~455 positions per row cross tile boundaries.
        rows = _rows(3, hidden=512, seed=6)
        ti, tags = h3_packed_layout(4097, 3, seed=6)
        grads = self._grads(4097, 512, seed=6)
        full = self._cuda_grad(rows, ti, tags, grads)
        flat = ti * 3 + tags
        for r in range(rows.shape[0]):
            own = (flat == r).nonzero().squeeze(1)
            assert own.numel() > 256
            alone = self._cuda_grad(rows, ti[own], tags[own], [g[own] for g in grads])
            assert torch.equal(alone[r], full[r])
            assert torch.count_nonzero(alone[torch.arange(rows.shape[0], device="cuda") != r]) == 0

    def test_unreferenced_rows_get_zero_grad(self):
        rows = _rows(3, hidden=16)
        ti = torch.zeros(7, dtype=torch.long, device="cuda")
        tags = torch.zeros(7, dtype=torch.long, device="cuda")  # only row 0 is used
        grad = self._cuda_grad(rows, ti, tags, self._grads(7, 16))
        assert torch.count_nonzero(grad[1:]) == 0
        assert torch.count_nonzero(grad[0]) > 0

    def test_segments_longer_than_one_tile(self):
        from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import BACKWARD_TILE

        seq = 3 * BACKWARD_TILE + 17
        rows = _rows(1, hidden=8, dtype=torch.float32)
        ti = torch.zeros(seq, dtype=torch.long, device="cuda")
        tags = torch.zeros(seq, dtype=torch.long, device="cuda")
        grads = [torch.ones(seq, 8, device="cuda") for _ in range(6)]
        grad = self._cuda_grad(rows, ti, tags, grads)
        assert torch.equal(grad[0], torch.full((48,), float(seq), device="cuda"))

    def test_reference_backward_is_deterministic_and_accurate(self):
        rows = _rows(2, hidden=256, seed=4)
        ti, tags = h3_packed_layout(3000, 2, seed=4)
        grads = self._grads(3000, 256, seed=4)

        def ref_grad():
            leaf = rows.detach().clone().requires_grad_(True)
            torch.autograd.backward(list(NativeH3AdaLNRowGatherOp()(leaf, ti, tags)), grads)
            return leaf.grad

        first = ref_grad()
        assert torch.equal(ref_grad(), first)
        torch.testing.assert_close(
            first.float(), self._cuda_grad(rows, ti, tags, grads).float(), atol=1e-1, rtol=2e-2
        )

    @staticmethod
    def _backward_probe(mutation: str):
        """Run the native backward on corrupted tile metadata in a child process."""

        probe = textwrap.dedent(
            """
            import sys
            import torch
            from rl_engine.backends.extension import _C
            from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import (
                _segment_tiles,
            )

            mutation = sys.argv[1]
            seq, hidden = 40, 8
            index = (torch.arange(seq, device="cuda") % 3)
            grad = torch.ones(6, seq, hidden, device="cuda")
            sorted_pos, begin, end, first = _segment_tiles(index, 3)
            if mutation == "tile_end_past_seq":
                end = end.clone(); end[-1] = seq + 5
            elif mutation == "position_past_seq":
                sorted_pos = sorted_pos.clone(); sorted_pos[0] = seq
            elif mutation == "segments_past_tiles":
                first = first.clone(); first[-1] = begin.numel() + 3
            elif mutation == "no_tiles":
                begin = begin[:0]; end = end[:0]
            _C.h3_adaln_row_gather_backward(grad, sorted_pos, begin, end, first, torch.float32)
            torch.cuda.synchronize()
            """
        )
        return subprocess.run(
            [sys.executable, "-c", probe, mutation],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_the_backward_stops_on_metadata_that_does_not_describe_the_tiles(self):
        _cuda_op()
        control = self._backward_probe("none")
        assert control.returncode == 0, control.stderr
        for mutation in (
            "tile_end_past_seq",
            "position_past_seq",
            "segments_past_tiles",
            "no_tiles",
        ):
            done = self._backward_probe(mutation)
            assert done.returncode != 0, mutation
            assert "device-side assert triggered" in done.stderr, (mutation, done.stderr)

    def test_backward_with_no_table_rows_returns_an_empty_table_and_rejects_no_boundaries(self):
        from rl_engine.backends.extension import _C

        _cuda_op()
        grad = torch.ones(6, 4, 8, device="cuda")
        zero = torch.zeros(0, dtype=torch.int64, device="cuda")
        pos = torch.arange(4, device="cuda")
        one = torch.zeros(1, dtype=torch.int64, device="cuda")  # R = 0
        empty = _C.h3_adaln_row_gather_backward(grad, pos, zero, zero, one, torch.float32)
        assert empty.shape == (0, 48)
        with pytest.raises(RuntimeError, match="R \\+ 1 entries"):
            _C.h3_adaln_row_gather_backward(grad, pos, zero, zero, zero, torch.float32)

    def test_forward_rejects_more_chunks_than_one_grid_dimension_holds(self):
        from rl_engine.backends.extension import _C

        _cuda_op()
        rows = torch.zeros(3, 65536, device="cuda")
        ti = torch.zeros(1, dtype=torch.int64, device="cuda")
        with pytest.raises(RuntimeError, match="grid.y"):
            _C.h3_adaln_row_gather_forward(rows, ti, ti, 65536, 3)

    def test_registry_dispatches_cuda(self):
        from rl_engine.runtime.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("adaln_row_gather", device="cuda")
        assert type(op).__name__ == "H3AdaLNRowGatherCudaOp"


def _backward_args(device="cuda"):
    kw = {"device": device, "dtype": torch.int64}
    return {
        "grad": torch.ones((6, 4, 8), device="cuda"),
        "sorted_pos": torch.tensor([0, 2, 1, 3], **kw),
        "tile_begin": torch.tensor([0, 2], **kw),
        "tile_end": torch.tensor([2, 4], **kw),
        "seg_first_tile": torch.tensor([0, 1, 2], **kw),
        "out_dtype": torch.float32,
    }


@requires_cuda
class TestNativeBackwardMetadata:
    def test_a_valid_call_runs(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        assert _C.h3_adaln_row_gather_backward(**_backward_args()).shape == (2, 48)

    def test_empty_segment_table_is_rejected(self):
        _cuda_op()
        from rl_engine.backends.extension import _C

        args = _backward_args()
        args["seg_first_tile"] = args["seg_first_tile"][:0]
        with pytest.raises(RuntimeError, match="seg_first_tile"):
            _C.h3_adaln_row_gather_backward(**args)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
    @pytest.mark.parametrize("name", ["sorted_pos", "tile_begin", "tile_end", "seg_first_tile"])
    def test_metadata_on_another_device_than_grad_is_rejected(self, name):
        _cuda_op()
        from rl_engine.backends.extension import _C

        args = _backward_args()
        args[name] = args[name].to("cuda:1")
        with pytest.raises(RuntimeError, match="tile metadata"):
            _C.h3_adaln_row_gather_backward(**args)
