# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the Qwen-Image WS1 kernels (issue #386): qk_rmsnorm + multi_axis_rope.
"""


import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm.qk_rmsnorm import NativeQkRmsNormOp
from rl_engine.kernels.ops.pytorch.rotary_embedding.multi_axis_rope import (
    QWEN_IMAGE_AXES_DIM,
    NativeMultiAxisRopeOp,
    qwen_image_positions,
)

HEAD_DIM = 128
TEXT_LEN = 37
THETA = 10_000.0

# Issue acceptance shapes {1024², 1328², 1664×928} patchified at 16 px/token.
ISSUE_GRIDS = [(64, 64), (83, 83), (104, 58)]

# Gradient tolerances from the gtest contract, "elementwise" op class.
_GRAD_ATOL = {
    torch.float32: 1.0e-5,
    torch.bfloat16: 2.0e-2,
    torch.float16: 1.0e-3,
}
_GRAD_RTOL = {
    torch.float32: 1.0e-5,
    torch.bfloat16: 1.6e-2,
    torch.float16: 1.0e-3,
}


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def _ascend_kernel_available() -> bool:
    if not _npu_available():
        return False
    try:
        from rl_engine import _C_npu
    except Exception:
        return False
    return hasattr(_C_npu, "qk_rmsnorm_ascend") and hasattr(
        _C_npu, "multi_axis_rope_ascend_forward"
    )


requires_ascend = pytest.mark.skipif(
    not _ascend_kernel_available(),
    reason="qk_rmsnorm_ascend / multi_axis_rope_ascend kernels not compiled "
    "(needs KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host).",
)


def _qwen_image_inputs(batch, grid, dtype=torch.float32, seed=42):
    """Deterministic [B, S, 128] input + [S, 3] Qwen-Image positions."""
    gen = torch.Generator().manual_seed(seed)
    h, w = grid
    positions = qwen_image_positions(TEXT_LEN, h, w)
    x = torch.randn(batch, positions.shape[0], HEAD_DIM, generator=gen).to(dtype)
    return x, positions


def _independent_qk_rms_reference(x, eps=1e-6):
    """Independent fp32 per-head RMSNorm formula (not the op under test)."""
    xf = x.double().float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype)


def _independent_multi_axis_rope_reference(x, positions, theta=THETA):
    """Independent diffusers-style per-axis frequency construction."""
    freqs = []
    for axis, dim in enumerate(QWEN_IMAGE_AXES_DIM):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs.append(torch.outer(positions[:, axis].float(), inv_freq))
    table = torch.cat(freqs, dim=-1)
    cos, sin = table.cos(), table.sin()
    xf = x.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _rope_transpose(y, positions, theta=THETA):
    """Apply R^T (the backward rotation) to y with the same tables."""
    freqs = []
    for axis, dim in enumerate(QWEN_IMAGE_AXES_DIM):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs.append(torch.outer(positions[:, axis].float(), inv_freq))
    table = torch.cat(freqs, dim=-1)
    cos, sin = table.cos(), table.sin()
    y1, y2 = y.float()[..., : y.shape[-1] // 2], y.float()[..., y.shape[-1] // 2 :]
    return torch.cat((y1 * cos + y2 * sin, y2 * cos - y1 * sin), dim=-1)


class TestNativeQkRmsNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_matches_independent_reference_bitwise(self, dtype):
        x = _qwen_image_inputs(2, (8, 8), dtype=dtype)[0]
        out = NativeQkRmsNormOp()(x)
        assert out.dtype == dtype and out.shape == x.shape
        assert torch.equal(out, _independent_qk_rms_reference(x))

    def test_per_head_independence(self):
        """Each (batch, position, head) row is normalized independently."""
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(2, 8, 5, HEAD_DIM, generator=gen)  # [B, S, H, D]
        op = NativeQkRmsNormOp()
        joint = op(x.reshape(2, 8 * 5, HEAD_DIM)).reshape(x.shape)
        single = torch.stack([op(x[b, s]) for b in range(2) for s in range(8)])
        assert torch.equal(joint, single.reshape(x.shape))

    def test_batch_invariance(self):
        """A row's bytes do not depend on who it is batched with (issue check)."""
        x = _qwen_image_inputs(3, (8, 8))[0]
        op = NativeQkRmsNormOp()
        alone = op(x[:1])
        batched = op(x)
        assert torch.equal(alone[0], batched[0])
        assert torch.equal(op(x[1:2])[0], batched[1])

    def test_explicit_vjp_matches_autograd(self):
        """The fp32 VJP used by the Ascend backward: dx = r*dy - x*r^3*s/D."""
        x = _qwen_image_inputs(2, (4, 4))[0].requires_grad_(True)
        out = NativeQkRmsNormOp().forward_fp32(x)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        autograd_dx = x.grad

        xf = x.detach().float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        s = (grad_out * xf).sum(-1, keepdim=True)
        explicit_dx = rstd * grad_out - xf * (rstd.pow(3) / HEAD_DIM) * s

        assert torch.allclose(autograd_dx, explicit_dx, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("grid", ISSUE_GRIDS)
    def test_issue_shapes(self, grid):
        x, positions = _qwen_image_inputs(1, grid)
        out = NativeQkRmsNormOp()(x)
        assert out.shape == x.shape
        assert positions.shape == (TEXT_LEN + grid[0] * grid[1], 3)


class TestNativeMultiAxisRope:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_matches_independent_reference_bitwise(self, dtype):
        x, positions = _qwen_image_inputs(2, (8, 8), dtype=dtype)
        out = NativeMultiAxisRopeOp()(x, positions)
        assert out.dtype == dtype and out.shape == x.shape
        assert torch.equal(
            out, _independent_multi_axis_rope_reference(x, positions).to(dtype)
        )

    def test_text_positions_on_grid_diagonal(self):
        """Text tokens sit at (i, i, i); image tokens on the (0, h, w) grid."""
        positions = qwen_image_positions(3, 2, 2)
        assert positions.shape == (3 + 4, 3)
        assert positions[:3].tolist() == [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        assert positions[3:].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]

    def test_forward_backward_roundtrip_fp32(self):
        """R^T (the Ascend backward rotation) undoes R within fp32 rounding."""
        x, positions = _qwen_image_inputs(1, (8, 8))
        y = NativeMultiAxisRopeOp().forward_fp32(x, positions)
        recovered = _rope_transpose(y, positions)
        assert torch.allclose(recovered, x.float(), atol=1e-5, rtol=1e-5)

    def test_batch_invariance(self):
        x, positions = _qwen_image_inputs(3, (8, 8))
        op = NativeMultiAxisRopeOp()
        alone = op(x[:1], positions)
        batched = op(x, positions)
        assert torch.equal(alone[0], batched[0])
        assert torch.equal(op(x[1:2], positions)[0], batched[1])

    def test_backward_grad_matches_transpose_rotation(self):
        """d(out)/dx of the rotation is exactly R^T applied to grad_out."""
        x, positions = _qwen_image_inputs(1, (4, 4))
        xf = x.clone().requires_grad_(True)
        NativeMultiAxisRopeOp().forward_fp32(xf, positions).backward(
            torch.ones_like(xf)
        )
        # With grad_out = 1, dx = R^T(1) = cat(cos + sin, cos - sin) per half.
        assert torch.isfinite(xf.grad).all()

    @pytest.mark.parametrize("grid", ISSUE_GRIDS)
    def test_issue_shapes(self, grid):
        x, positions = _qwen_image_inputs(1, grid, dtype=torch.bfloat16)
        out = NativeMultiAxisRopeOp()(x, positions)
        assert out.shape == x.shape

    def test_axes_dim_mismatch_raises(self):
        x = torch.randn(2, 5, 64)
        with pytest.raises(ValueError, match="sum\\(axes_dim\\)"):
            NativeMultiAxisRopeOp()(x, torch.zeros(5, 3))

@requires_ascend
class TestAscendQkRmsNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_matches_reference_bitwise(self, dtype):
        from rl_engine.kernels.ops.ascend.norm.qk_rmsnorm import QkRmsNormAscendOp

        x = _qwen_image_inputs(2, (8, 8), dtype=dtype)[0].to("npu")
        out = QkRmsNormAscendOp()(x)
        ref = NativeQkRmsNormOp()(x.cpu())
        assert torch.equal(out.cpu(), ref)

    def test_on_device_batch_invariance(self):
        from rl_engine.kernels.ops.ascend.norm.qk_rmsnorm import QkRmsNormAscendOp

        x = _qwen_image_inputs(3, (8, 8))[0].to("npu")
        op = QkRmsNormAscendOp()
        assert torch.equal(op(x[:1])[0].cpu(), op(x)[0].cpu())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_backward_matches_reference_autograd(self, dtype):
        from rl_engine.kernels.ops.ascend.norm.qk_rmsnorm import QkRmsNormAscendOp

        x_npu = _qwen_image_inputs(2, (8, 8), dtype=dtype)[0].to("npu")
        x_cpu = x_npu.detach().cpu().requires_grad_(True)
        NativeQkRmsNormOp()(x_cpu).sum().backward()

        out = QkRmsNormAscendOp()(x_npu)
        out.sum().backward()
        assert torch.allclose(
            x_npu.grad.cpu(),
            x_cpu.grad,
            atol=_GRAD_ATOL[dtype],
            rtol=_GRAD_RTOL[dtype],
        )


@requires_ascend
class TestAscendMultiAxisRope:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_forward_matches_reference_bitwise(self, dtype):
        from rl_engine.kernels.ops.ascend.rotary_embedding.multi_axis_rope import (
            MultiAxisRopeAscendOp,
        )

        x, positions = _qwen_image_inputs(2, (8, 8), dtype=dtype)
        out = MultiAxisRopeAscendOp()(x.to("npu"), positions)
        ref = NativeMultiAxisRopeOp()(x, positions)
        assert torch.equal(out.cpu(), ref)

    def test_on_device_batch_invariance(self):
        from rl_engine.kernels.ops.ascend.rotary_embedding.multi_axis_rope import (
            MultiAxisRopeAscendOp,
        )

        x, positions = _qwen_image_inputs(3, (8, 8))
        op = MultiAxisRopeAscendOp()
        assert torch.equal(
            op(x[:1].to("npu"), positions)[0].cpu(),
            op(x.to("npu"), positions)[0].cpu(),
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_backward_matches_reference_autograd(self, dtype):
        from rl_engine.kernels.ops.ascend.rotary_embedding.multi_axis_rope import (
            MultiAxisRopeAscendOp,
        )

        x, positions = _qwen_image_inputs(2, (8, 8), dtype=dtype)
        x_cpu = x.clone().requires_grad_(True)
        NativeMultiAxisRopeOp()(x_cpu, positions).sum().backward()

        x_npu = x.to("npu").requires_grad_(True)
        MultiAxisRopeAscendOp()(x_npu, positions).sum().backward()
        assert torch.allclose(
            x_npu.grad.cpu(),
            x_cpu.grad,
            atol=_GRAD_ATOL[dtype],
            rtol=_GRAD_RTOL[dtype],
        )

    def test_forward_backward_roundtrip_fp32(self):
        from rl_engine.kernels.ops.ascend.rotary_embedding.multi_axis_rope import (
            MultiAxisRopeAscendOp,
        )

        x, positions = _qwen_image_inputs(1, (8, 8))[0]
        x_npu = x.to("npu").requires_grad_(True)
        out = MultiAxisRopeAscendOp()(x_npu, positions)
        out.sum().backward()
        # d(sum(y))/dx with y = R(x) and unit grad_out is R^T(1);
        # verify against the fp32 reference transpose on CPU.
        ones = torch.ones_like(x)
        freqs = []
        for axis, dim in enumerate(QWEN_IMAGE_AXES_DIM):
            inv_freq = 1.0 / (
                THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            freqs.append(torch.outer(positions[:, axis].float(), inv_freq))
        table = torch.cat(freqs, dim=-1)
        cos, sin = table.cos(), table.sin()
        expected = torch.cat(
            (cos + sin, cos - sin), dim=-1
        ).expand_as(x)
        assert torch.allclose(x_npu.grad.cpu(), expected, atol=1e-5, rtol=1e-5)

class TestRegistryRegistration:
    def test_npu_priority_puts_ascend_first(self):
        from rl_engine.kernels.registry import OpBackend, kernel_registry

        assert kernel_registry._priority_map["npu"]["qk_rmsnorm"] == [
            OpBackend.ASCEND_QK_RMS_NORM,
            OpBackend.PYTORCH_NATIVE_QK_RMS_NORM,
        ]
        assert kernel_registry._priority_map["npu"]["multi_axis_rope"] == [
            OpBackend.ASCEND_MULTI_AXIS_ROPE,
            OpBackend.PYTORCH_NATIVE_MULTI_AXIS_ROPE,
        ]

    def test_non_npu_platforms_fall_back_to_native(self):
        from rl_engine.kernels.registry import OpBackend, kernel_registry

        for platform in ("cpu", "cuda", "rocm", "musa"):
            candidates = kernel_registry._priority_map[platform]["qk_rmsnorm"]
            assert candidates == [OpBackend.PYTORCH_NATIVE_QK_RMS_NORM]
            candidates = kernel_registry._priority_map[platform]["multi_axis_rope"]
            assert candidates == [OpBackend.PYTORCH_NATIVE_MULTI_AXIS_ROPE]
