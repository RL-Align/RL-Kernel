# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.ops.attention.preprocess import (
    MANDATED_ATTENTION_PREPROCESS_BACKENDS,
    TE_ROCM_QK_RMSNORM_BACKEND_ID,
    H100AttentionPreprocessor,
)
from rl_engine.reference.norm.rms_norm import NativeRMSNormOp
from rl_engine.reference.rope.rope import NativeRoPEOp


def _has_h100_preprocess() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        required = (
            "rmsnorm_forward",
            "rmsnorm_backward_dx",
            "rmsnorm_backward_dw",
            "rope_apply_sm90",
        )
        return bool(_EXT_AVAILABLE and all(hasattr(_C, name) for name in required))
    except ImportError:
        return False


requires_h100_preprocess = pytest.mark.skipif(
    not _has_h100_preprocess(),
    reason="Hopper with compiled RMSNorm and RoPE CUDA kernels is required",
)


def test_h100_preprocessor_uses_common_backend_ids_for_fallback():
    assert dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS) == {
        "qk_rmsnorm": "rlkernel.cuda.rmsnorm",
        "rope": "rlkernel.cuda.rope_sm90",
    }


def test_preprocessor_can_reuse_qk_norm_without_fusing_rope():
    def rmsnorm(x, weight, *, eps):
        del eps
        return (x.float() * weight.float()).to(torch.bfloat16)

    def rope(x, positions, *, theta):
        del positions, theta
        return x

    op = object.__new__(H100AttentionPreprocessor)
    op.device = torch.device("cpu")
    op.device_capability = (0, 0)
    op.rmsnorm = rmsnorm
    op.rope = rope
    op.deterministic_backend_ids = {
        "qk_rmsnorm": "rlkernel.rocm.triton_rmsnorm",
        "rope": "rlkernel.rocm.deterministic_rope",
    }
    op.native_qk_norm = rmsnorm
    op.native_rope = None
    op.require_native_qk_norm = True
    op.native_qk_norm_backend_id = TE_ROCM_QK_RMSNORM_BACKEND_ID
    op.native_rope_backend_id = "unused"
    op.policy_id = "test"
    q = torch.randn(2, 4, 3, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int64)

    result = op(q, k, weight, weight, positions)

    assert result.fallback is False
    assert result.backend_ids == {
        "qk_rmsnorm": TE_ROCM_QK_RMSNORM_BACKEND_ID,
        "rope": "rlkernel.rocm.deterministic_rope",
    }


def test_preprocessor_falls_back_atomically_when_vendor_qk_norm_drifts():
    def deterministic(x, weight, *, eps):
        del weight, eps
        return x

    def drifting(x, weight, *, eps):
        del weight, eps
        return (x.float() + 1).to(torch.bfloat16)

    def rope(x, positions, *, theta):
        del positions, theta
        return x

    op = object.__new__(H100AttentionPreprocessor)
    op.device = torch.device("cpu")
    op.device_capability = (0, 0)
    op.rmsnorm = deterministic
    op.rope = rope
    op.deterministic_backend_ids = dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS)
    op.native_qk_norm = drifting
    op.native_rope = None
    op.require_native_qk_norm = False
    op.native_qk_norm_backend_id = "transformer_engine.cuda.rmsnorm"
    op.native_rope_backend_id = "unused"
    op.policy_id = "test"
    q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.bfloat16)
    positions = torch.arange(3, dtype=torch.int64)

    result = op(q, k, weight, weight, positions)

    assert result.fallback is True
    assert result.backend_ids == MANDATED_ATTENTION_PREPROCESS_BACKENDS
    assert result.fallback_reason == "native_preprocess_bitwise_probe_failed"


def test_h100_preprocessor_fails_before_dispatch_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an available CUDA runtime"):
        H100AttentionPreprocessor()


def test_h100_preprocessor_rejects_non_hopper_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    with pytest.raises(RuntimeError, match="requires Hopper SM90"):
        H100AttentionPreprocessor()


def _inputs():
    torch.manual_seed(7)
    device = torch.device("cuda")
    q = torch.randn(2, 4, 8, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 8, 128, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    positions = torch.tensor(
        [[0, 7, 2, 9, 4, 11, 6, 13], [100, 107, 102, 109, 104, 111, 106, 113]],
        device=device,
        dtype=torch.int64,
    )
    return q, k, q_weight, k_weight, positions


@requires_h100_preprocess
def test_h100_preprocessor_executes_cuda_qk_norm_and_zigzag_rope():
    q, k, q_weight, k_weight, positions = _inputs()
    result = H100AttentionPreprocessor()(q, k, q_weight, k_weight, positions)
    result = H100AttentionPreprocessor(reuse_transformer_engine_qk_norm=False)(
        q, k, q_weight, k_weight, positions
    )

    norm = NativeRMSNormOp()
    rope = NativeRoPEOp()
    q_ref = rope(norm(q, q_weight), positions)
    k_ref = rope(norm(k, k_weight), positions)

    assert result.fallback is True
    assert dict(result.backend_ids) == dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS)
    assert result.fallback_reason == "native_preprocess_not_supplied"
    assert result.probe_id
    assert result.readback_fields() == {
        "preprocess_backends": dict(MANDATED_ATTENTION_PREPROCESS_BACKENDS),
        "preprocess_fallback": True,
        "preprocess_fallback_reason": "native_preprocess_not_supplied",
        "preprocess_probe_id": result.probe_id,
        "preprocess_policy_id": result.policy_id,
    }
    torch.testing.assert_close(result.q.float(), q_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result.k.float(), k_ref.float(), atol=2e-2, rtol=2e-2)


@requires_h100_preprocess
def test_h100_preprocessor_is_bitwise_batch_invariant_for_2d_positions():
    q, k, q_weight, k_weight, positions = _inputs()
    op = H100AttentionPreprocessor()
    op = H100AttentionPreprocessor(reuse_transformer_engine_qk_norm=False)
    full = op(q, k, q_weight, k_weight, positions)

    for batch_index in range(q.shape[0]):
        single = op(
            q[batch_index : batch_index + 1],
            k[batch_index : batch_index + 1],
            q_weight,
            k_weight,
            positions[batch_index : batch_index + 1],
        )
        assert torch.equal(full.q[batch_index], single.q[0])
        assert torch.equal(full.k[batch_index], single.k[0])
