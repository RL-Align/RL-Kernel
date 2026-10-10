# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from rl_engine.ops.attention.projection import (
    O_PROJ_COLLECTIVE_CONTRACT,
    QKV_COLLECTIVE_CONTRACT,
    ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID,
    AttentionProjectionOp,
    split_qkv,
)


def _deterministic(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(x.float(), weight.float()).to(torch.bfloat16)


def _inputs():
    torch.manual_seed(19)
    return (
        torch.randn(7, 8, dtype=torch.bfloat16),
        torch.randn(8, 12, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    ("projection", "collective"),
    [("qkv", QKV_COLLECTIVE_CONTRACT), ("o_proj", O_PROJ_COLLECTIVE_CONTRACT)],
)
def test_projection_falls_back_to_common_deterministic_path(projection, collective):
    x, weight = _inputs()
    result = AttentionProjectionOp(projection, deterministic=_deterministic)(x, weight)

    assert torch.equal(result.output, _deterministic(x, weight))
    assert result.plan.backend_id == "rlkernel.cuda.det_gemm"
    assert result.plan.fallback is True
    assert result.plan.fallback_reason == "native_projection_not_supplied"
    assert result.plan.split_k is False
    assert result.plan.accumulation_dtype == "torch.float32"
    assert dict(result.plan.collective) == collective.to_dict()


def test_projection_accepts_native_only_after_bitwise_probe():
    x, weight = _inputs()
    result = AttentionProjectionOp(
        "qkv",
        native=_deterministic,
        native_backend_id="megatron.te.qkv",
        deterministic=_deterministic,
    )(x, weight)

    assert torch.equal(result.output, _deterministic(x, weight))
    assert result.plan.backend_id == "megatron.te.qkv"
    assert result.plan.fallback is False
    assert result.plan.fallback_reason is None


def test_projection_records_rocm_deterministic_backend():
    x, weight = _inputs()
    result = AttentionProjectionOp(
        "qkv",
        deterministic=_deterministic,
        deterministic_backend_id=ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID,
    )(x, weight)

    assert result.plan.backend_id == ROCM_DETERMINISTIC_PROJECTION_BACKEND_ID
    assert result.plan.fallback is True


def test_projection_rejects_native_drift_and_records_reason():
    x, weight = _inputs()

    def drifting_native(a, b):
        return (_deterministic(a, b).float() + 1.0).to(torch.bfloat16)

    result = AttentionProjectionOp("o_proj", native=drifting_native, deterministic=_deterministic)(
        x, weight
    )

    assert result.plan.fallback is True
    assert result.plan.fallback_reason == "native_projection_bitwise_probe_failed"
    assert torch.equal(result.output, _deterministic(x, weight))


def test_split_qkv_is_fixed_contiguous_q_k_v_order():
    projected = torch.arange(2 * 16, dtype=torch.bfloat16).reshape(2, 16)
    q, k, v = split_qkv(projected, q_heads=2, kv_heads=1, head_dim=4)

    assert q.shape == (2, 8)
    assert k.shape == (2, 4)
    assert v.shape == (2, 4)
    assert torch.equal(torch.cat((q, k, v), dim=-1), projected)


def test_projection_requires_bf16_and_compatible_k():
    x, weight = _inputs()
    with pytest.raises(TypeError, match="BF16"):
        AttentionProjectionOp("qkv", deterministic=_deterministic)(x.float(), weight)
    with pytest.raises(ValueError, match="K dimensions"):
        AttentionProjectionOp("qkv", deterministic=_deterministic)(x, weight[:-1])


def test_o_proj_collective_contract_includes_sp_scatter_gather_and_tp_reduction():
    assert O_PROJ_COLLECTIVE_CONTRACT.sp_forward == "reduce_scatter"
    assert O_PROJ_COLLECTIVE_CONTRACT.sp_backward == "all_gather"
    assert O_PROJ_COLLECTIVE_CONTRACT.reduction_forward == "all_reduce"
    assert O_PROJ_COLLECTIVE_CONTRACT.reduction_backward == "none"
