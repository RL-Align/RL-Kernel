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
        probe = textwrap.dedent("""
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
            """)
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

    def test_registry_dispatches_cuda(self):
        from rl_engine.runtime.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("adaln_row_gather", device="cuda")
        assert type(op).__name__ == "H3AdaLNRowGatherCudaOp"
