# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 per-head Q/K RMSNorm (RFC #420 ``h3_qk_rmsnorm_d128``): reference, kernel, invariance."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.reference.minimax_h3.qk_rmsnorm import (
    H3_QK_HEAD_DIM,
    H3_QK_HEADS,
    NativeH3QKRMSNormOp,
)
from rl_engine.reference.minimax_h3.rmsnorm import NativeH3RMSNormOp

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _x(shape, dtype=torch.bfloat16, seed=0, device="cpu", scale=2.0):
    gen = torch.Generator("cpu").manual_seed(seed)
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(device)


def _w(dtype=torch.bfloat16, seed=1, device="cpu"):
    gen = torch.Generator("cpu").manual_seed(seed)
    return (torch.rand(H3_QK_HEAD_DIM, generator=gen) + 0.5).to(dtype).to(device)


def _op():
    from rl_engine.backends.cuda.model_specific.minimax_h3.qk_rmsnorm import H3QKRMSNormCudaOp

    try:
        return H3QKRMSNormCudaOp()
    except RuntimeError as exc:
        pytest.skip(str(exc))


class TestReference:
    def test_golden_matches_the_provider_in_fp32(self):
        x, w = _x((3, 7, 8, 128), torch.float32), _w(torch.float32)
        op = NativeH3QKRMSNormOp()
        torch.testing.assert_close(op.forward(x, w), op.forward_fp32(x, w), atol=1e-6, rtol=1e-6)

    def test_it_is_the_row_of_h3_rmsnorm_at_128(self):
        x, w = _x((2, 5, 14, 128)), _w()
        assert _same(NativeH3QKRMSNormOp()(x, w), NativeH3RMSNormOp()(x, w))

    @pytest.mark.parametrize("heads", [1, 7, 8, 14, 28, 56])
    def test_head_counts_that_divide_the_heads_are_accepted(self, heads):
        x = _x((1, 3, heads, 128))
        assert NativeH3QKRMSNormOp()(x, _w()).shape == x.shape

    @pytest.mark.parametrize(
        ("x", "w", "eps", "error"),
        [
            ((2, 5, 56, 64), (128,), 1e-5, ValueError),  # head dim
            ((2, 5, 56, 128), (64,), 1e-5, ValueError),  # weight shape
            ((2, 5, 5, 128), (128,), 1e-5, ValueError),  # heads do not divide 56
            ((2, 128), (128,), 1e-5, ValueError),  # no head axis
            ((2, 5, 56, 128), (128,), 1e-6, ValueError),  # eps is pinned
            ((2, 0, 56, 128), (128,), 1e-5, ValueError),  # empty
        ],
    )
    def test_bad_shapes_are_refused(self, x, w, eps, error):
        with pytest.raises(error):
            NativeH3QKRMSNormOp()(torch.zeros(x).bfloat16(), torch.ones(w).bfloat16(), eps)

    def test_mixed_dtypes_are_refused(self):
        with pytest.raises(TypeError):
            NativeH3QKRMSNormOp()(_x((1, 2, 56, 128)), _w(torch.float32))


class TestRegistry:
    def test_the_row_resolves_and_is_the_cuda_op_first(self):
        from rl_engine.runtime.registry import KernelRegistry

        registry = KernelRegistry()
        chain = registry._priority_map["cuda"]["h3_qk_rmsnorm_d128"]
        assert [b.name for b in chain] == ["CUDA_H3_QK_RMSNORM", "PYTORCH_H3_QK_RMSNORM"]


@needs_gpu
class TestCuda:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
    @pytest.mark.parametrize("shape", [(1, 1, 56, 128), (2, 17, 56, 128), (1, 925, 56, 128)])
    def test_forward_is_bitwise_the_providers(self, shape, dtype):
        op, x, w = _op(), _x(shape, dtype, 3, "cuda"), _w(dtype, 5, "cuda")
        want = F.rms_norm(x, (128,), w, 1e-5)
        assert _same(op(x, w), want)

    @pytest.mark.parametrize("heads", [7, 14, 28])
    def test_a_head_shard_is_the_slice_of_the_full_result(self, heads):
        op, x, w = _op(), _x((2, 33, 56, 128), seed=4, device="cuda"), _w(device="cuda")
        full = op(x, w)
        for r in range(56 // heads):
            part = op(x[:, :, r * heads : (r + 1) * heads].contiguous(), w)
            assert _same(part, full[:, :, r * heads : (r + 1) * heads])

    def test_a_strided_view_gives_the_contiguous_result(self):
        op, x, w = _op(), _x((2, 56, 9, 128), seed=5, device="cuda"), _w(device="cuda")
        view = x.transpose(1, 2)  # (B, S, H, D) from a (B, H, S, D) tensor, not contiguous
        assert not view.is_contiguous()
        assert _same(op(view, w), op(view.contiguous(), w))

    def test_rows_do_not_depend_on_batch_size_position_or_other_rows(self):
        op, w = _op(), _w(device="cuda")
        x = _x((9, 11, 56, 128), seed=6, device="cuda")
        full = op(x, w)
        for b in (1, 2, 5):
            for i in (0, 3, 9 - b):
                assert _same(op(x[i : i + b], w), full[i : i + b])
        assert _same(op(x.flip(0), w), full.flip(0))
        noisy = x.clone()
        noisy[4:] = _x((5, 11, 56, 128), seed=77, device="cuda")
        assert _same(op(noisy, w)[:4], full[:4])  # other rows, other bytes

    def test_repeat_is_bitwise(self):
        op, x, w = _op(), _x((2, 13, 56, 128), seed=8, device="cuda"), _w(device="cuda")
        assert _same(op(x, w), op(x, w))

    def test_backward_repeats_and_dx_is_row_local(self):
        op = _op()
        x = _x((4, 7, 56, 128), seed=9, device="cuda").requires_grad_(True)
        w = _w(device="cuda").requires_grad_(True)
        g = _x((4, 7, 56, 128), seed=10, device="cuda")
        op(x, w).backward(g)
        dx, dw = x.grad.clone(), w.grad.clone()
        for _ in range(2):
            x.grad, w.grad = None, None
            op(x, w).backward(g)
            assert _same(x.grad, dx) and _same(w.grad, dw)
        # each sample's dx is the same alone
        for i in range(4):
            xi = x.detach()[i : i + 1].clone().requires_grad_(True)
            op(xi, w.detach()).backward(g[i : i + 1])
            assert _same(xi.grad, dx[i : i + 1])

    @pytest.mark.parametrize("seq", [3, 64, 257])
    def test_gradients_against_fp64(self, seq):
        op = _op()
        x = _x((2, seq, 56, 128), torch.float32, 11, "cuda").requires_grad_(True)
        w = _w(torch.float32, 12, "cuda").requires_grad_(True)
        g = _x((2, seq, 56, 128), torch.float32, 13, "cuda")
        op(x, w).backward(g)
        x64, w64 = (
            x.detach().double().requires_grad_(True),
            w.detach().double().requires_grad_(True),
        )
        rstd = torch.rsqrt(x64.square().mean(-1, keepdim=True) + 1e-5)
        (x64 * rstd * w64).backward(g.double())
        for got, want in ((x.grad, x64.grad), (w.grad, w64.grad)):
            rel = (got.double() - want).abs().max() / want.abs().max()
            assert rel < 5e-6, (seq, float(rel))

    def test_the_backward_records_the_rows_kernel(self):
        from rl_engine.ops.autograd.backward_runtime import (
            reset_backward_runtime,
            snapshot_backward_runtime,
        )

        op = _op()
        reset_backward_runtime()
        x = _x((1, 2, 56, 128), device="cuda").requires_grad_(True)
        op(x, _w(device="cuda")).sum().backward()
        assert "h3_qk_rmsnorm_d128" in snapshot_backward_runtime()

    def test_other_devices_and_dtypes_are_refused(self):
        op = _op()
        with pytest.raises(ValueError, match="CUDA"):
            op(_x((1, 1, 56, 128)), _w())
        with pytest.raises(TypeError):
            op(_x((1, 1, 56, 128), device="cuda"), _w(torch.float32, device="cuda"))
        with pytest.raises(ValueError, match="weight is on"):
            op(_x((1, 1, 56, 128), device="cuda"), _w())

    def test_block_0s_weights(self, h3_block_weights_cuda):
        op = _op()
        names = ("attn.norm_q.weight", "attn.norm_k.weight")
        x = _x((1, 257, 56, 128), seed=21, device="cuda")
        for name in names:
            w = h3_block_weights_cuda[f"transformer_blocks.0.{name}"]
            assert w.shape == (128,) and w.dtype == torch.bfloat16
            assert _same(op(x, w), F.rms_norm(x, (128,), w, 1e-5))
            assert torch.isfinite(op(x, w)).all()


@needs_gpu
def test_the_strict_block_and_refiner_bind_the_row(h3_block_weights_cuda):
    from rl_engine.validation.models import h3_block, h3_refiner

    try:
        ops = h3_block.CandidateOps()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert type(ops.qk_norm).__name__ == "H3QKRMSNormCudaOp"
    for nodes in (h3_block.NODES, h3_refiner.NODES):
        bound = {n.name: n.binding for n in nodes}
        for name in ("q_norm", "k_norm"):
            assert (bound[name].status, bound[name].rfc_row) == ("row", "h3_qk_rmsnorm_d128")


def test_heads_constants_match_the_released_model():
    assert H3_QK_HEADS * H3_QK_HEAD_DIM == 7168
