# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContract,
    AttentionContractError,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
)
from rl_engine.reference.attention.ablation import (
    BACKEND_ID,
    REFERENCE_BACKEND_ID,
    AttentionAblationConfig,
    AttentionAblationOp,
)


def _contract(*, split_kv: SplitKVSpec | None = None) -> AttentionContract:
    sharding = ShardingSpec(
        tp_rank=0,
        tp_world_size=1,
        cp_rank=0,
        cp_world_size=1,
        global_q_heads=2,
        global_kv_heads=1,
        local_q_head_start=0,
        local_q_heads=2,
        local_kv_head_start=0,
        local_kv_heads=1,
        global_sequence_length=4,
        local_sequence_length=4,
        global_block_indices=(0,),
        global_block_token_starts=(0,),
        local_block_offsets=(0, 4),
    )
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=4,
        head_dim=4,
        causal=True,
        causal_offsets=(0,),
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=split_kv or SplitKVSpec.disabled(),
    )


def _qkv():
    torch.manual_seed(0)
    return (
        torch.randn(1, 2, 4, 4, dtype=torch.bfloat16),
        torch.randn(1, 1, 4, 4, dtype=torch.bfloat16),
        torch.randn(1, 1, 4, 4, dtype=torch.bfloat16),
    )


def _cp2_contract() -> AttentionContract:
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=2,
        head_dim=4,
        causal=True,
        causal_offsets=(0,),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            cp_rank=0,
            cp_world_size=2,
            global_q_heads=2,
            global_kv_heads=1,
            local_q_head_start=0,
            local_q_heads=2,
            local_kv_head_start=0,
            local_kv_heads=1,
            global_sequence_length=4,
            local_sequence_length=2,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, 2),
        ),
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
    )


def test_attention_wrapper_has_unified_result_and_provenance():
    q, k, v = _qkv()
    result = AttentionAblationOp()(q, k, v, contract=_contract())

    assert result.backend_id == REFERENCE_BACKEND_ID
    assert result.deterministic
    assert result.out.shape == q.shape
    assert result.lse is not None
    assert result.lse.dtype is torch.float32
    assert result.provenance["semantic_operator"] == "attention"
    assert result.provenance["split_kv"]["mode"] == "disabled"
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_SCHEDULE_ID
    assert result.readback()["out_shape"] == list(q.shape)


def test_attention_wrapper_supports_explicit_injected_backend():
    q, k, v = _qkv()

    class FakeBackend:
        backend_id = "test.attention.backend"

        def forward_with_lse(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            return q.clone(), torch.zeros(q.shape[:3], dtype=torch.float32)

    result = AttentionAblationOp()(
        q,
        k,
        v,
        contract=_contract(),
        backend=FakeBackend(),
        deterministic=False,
    )
    assert result.backend_id == "test.attention.backend"
    assert torch.equal(result.out, q)


def test_attention_wrapper_binds_rocm_runtime_once(monkeypatch):
    import rl_engine.backends.rocm.attention.strict_runtime as rocm_runtime

    calls = []

    class Runtime:
        def __init__(self, *, process_group=None):
            calls.append(process_group)

    monkeypatch.setattr(rocm_runtime, "StrictRocmAttentionRuntime", Runtime)
    group = object()
    operator = AttentionAblationOp()

    first = operator.bind_rocm_runtime(process_group=group)
    second = operator.bind_runtime(platform="rocm", process_group=group)

    assert first is second
    assert calls == [group]
    assert operator.core is first and operator.cp_backend is first


def test_attention_wrapper_rejects_runtime_platform_or_group_drift(monkeypatch):
    import rl_engine.backends.rocm.attention.strict_runtime as rocm_runtime

    class Runtime:
        def __init__(self, *, process_group=None):
            self.process_group = process_group

    monkeypatch.setattr(rocm_runtime, "StrictRocmAttentionRuntime", Runtime)
    group = object()
    operator = AttentionAblationOp()
    operator.bind_rocm_runtime(process_group=group)

    with pytest.raises(AttentionContractError, match="platform"):
        operator.bind_cuda_runtime(process_group=group)
    with pytest.raises(AttentionContractError, match="process group"):
        operator.bind_rocm_runtime(process_group=object())


def test_wrapper_owned_deterministic_core_does_not_require_external_provenance():
    q, k, v = _qkv()

    class WrapperOwnedCore:
        backend_id = BACKEND_ID

        def __call__(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            return q.clone(), torch.zeros(q.shape[:3], dtype=torch.float32)

    result = AttentionAblationOp()(
        q,
        k,
        v,
        contract=_contract(),
        backend=WrapperOwnedCore(),
    )

    assert result.backend_id == BACKEND_ID
    assert result.deterministic is True


def test_cp_production_configuration_fails_closed_without_ag_rs_backend():
    q, k, v = _qkv()
    with pytest.raises(AttentionContractError, match="injected AG/RS backend"):
        AttentionAblationOp(communication_backend="self_owned_cuda_ag_rs")(
            q[:, :, :2], k[:, :, :2], v[:, :, :2], contract=_cp2_contract()
        )


def test_cp_backend_must_explicitly_declare_cp_world_size():
    q, k, v = _qkv()

    class KwargsOnlyBackend:
        backend_id = "test.kwargs_only"
        core_id = STRICT_ATTENTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_SCHEDULE_ID

        def __call__(self, q, k, v, **kwargs):
            del k, v, kwargs
            return q, torch.zeros(q.shape[:3], dtype=torch.float32)

    with pytest.raises(AttentionContractError, match="explicitly accepts cp_world_size"):
        AttentionAblationOp(
            cp_backend=KwargsOnlyBackend(),
            communication_backend="cuda_ag_rs",
        )(
            q[:, :, :2],
            k[:, :, :2],
            v[:, :, :2],
            contract=_cp2_contract(),
        )


@pytest.mark.parametrize("communication_backend", ["cuda_ag_rs", "rccl_ag_rs"])
def test_cp_wrapper_accepts_exact_platform_vendor_core(communication_backend):
    q, k, v = _qkv()

    class VendorCPBackend:
        backend_id = "vendor.strict_attention"
        core_id = "vendor.strict_core.v1"
        strict_schedule = "vendor.strict_schedule.v1"

        def __call__(self, q, k, v, *, causal, scale, cp_world_size):
            del k, v, causal, scale
            assert cp_world_size == 2
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros(q.shape[:3], dtype=torch.float32),
                provenance={
                    "strict_core_id": self.core_id,
                    "strict_schedule": self.strict_schedule,
                    "actual_backend": self.backend_id,
                    "communication_backend": communication_backend,
                    "production_ready": True,
                    "native_attention_arithmetic": True,
                    "fallback": False,
                    "reference_only": False,
                },
            )

    result = AttentionAblationOp(
        cp_backend=VendorCPBackend(),
        communication_backend=communication_backend,
    )(
        q[:, :, :2],
        k[:, :, :2],
        v[:, :, :2],
        contract=_cp2_contract(),
        config=AttentionAblationConfig(
            strict_core_id=VendorCPBackend.core_id,
            strict_schedule=VendorCPBackend.strict_schedule,
        ),
    )

    assert result.provenance["actual_backend"] == VendorCPBackend.backend_id
    assert result.provenance["communication_backend"] == communication_backend
    assert result.provenance["native_attention_arithmetic"] is True


def test_cp_production_wrapper_preserves_runtime_backend_provenance():
    q, k, v = _qkv()

    class StrictCPBackend:
        backend_id = "injected.strict_cp_backend"
        core_id = STRICT_ATTENTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_SCHEDULE_ID

        def __call__(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros(q.shape[:3], dtype=torch.float32),
                provenance={
                    "strict_core_id": STRICT_ATTENTION_CORE_ID,
                    "strict_schedule": STRICT_ATTENTION_SCHEDULE_ID,
                    "actual_backend": self.backend_id,
                    "communication_backend": "self_owned_cuda_ag_rs",
                    "production_ready": True,
                    "native_attention_arithmetic": False,
                    "fallback": False,
                    "reference_only": False,
                },
            )

    result = AttentionAblationOp(
        cp_backend=StrictCPBackend(),
        communication_backend="self_owned_cuda_ag_rs",
    )(
        q,
        k,
        v,
        contract=_contract(),
        backend=StrictCPBackend(),
    )
    assert result.provenance["actual_backend"] == StrictCPBackend.backend_id
    assert result.provenance["communication_backend"] == "self_owned_cuda_ag_rs"
    assert result.provenance["production_ready"] is True


def test_deterministic_attention_rejects_runtime_split_kv_auto():
    q, k, v = _qkv()
    contract = _contract(split_kv=SplitKVSpec.auto(strict_consistency=False))
    with pytest.raises(AttentionContractError, match="Split-KV"):
        AttentionAblationOp()(q, k, v, contract=contract)


@pytest.mark.parametrize(
    ("missing_field", "replacement"),
    [
        ("production_ready", None),
        ("fallback", True),
        ("reference_only", True),
    ],
)
def test_vendor_production_core_fails_closed_without_exact_provenance(missing_field, replacement):
    q, k, v = _qkv()

    class VendorCore:
        backend_id = "vendor.strict_attention"
        core_id = "vendor.strict_core.v1"
        strict_schedule = "vendor.strict_schedule.v1"

        def __call__(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            provenance = {
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "actual_backend": self.backend_id,
                "production_ready": True,
                "native_attention_arithmetic": True,
                "fallback": False,
                "reference_only": False,
            }
            if replacement is None:
                del provenance[missing_field]
            else:
                provenance[missing_field] = replacement
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros(q.shape[:3], dtype=torch.float32),
                provenance=provenance,
            )

    with pytest.raises(AttentionContractError, match="runtime provenance"):
        AttentionAblationOp()(
            q,
            k,
            v,
            contract=_contract(),
            backend=VendorCore(),
            config=AttentionAblationConfig(
                strict_core_id=VendorCore.core_id,
                strict_schedule=VendorCore.strict_schedule,
            ),
        )


@pytest.mark.parametrize(
    "split_kv",
    [SplitKVSpec.auto(strict_consistency=False), SplitKVSpec.fixed(2)],
)
def test_deterministic_attention_requires_split_kv_disabled(split_kv):
    q, k, v = _qkv()
    contract = _contract(split_kv=split_kv)
    with pytest.raises(AttentionContractError, match="Split-KV to be disabled"):
        AttentionAblationOp()(q, k, v, contract=contract)


def test_explicit_cp_callable_cannot_bypass_ag_rs_requirement():
    q, k, v = _qkv()

    class VendorCPBackend:
        backend_id = "vendor.strict_attention"
        core_id = "vendor.strict_core.v1"
        strict_schedule = "vendor.strict_schedule.v1"

        def __call__(self, q, k, v, *, causal, scale, cp_world_size):
            del k, v, causal, scale, cp_world_size
            return q, torch.zeros(q.shape[:3], dtype=torch.float32)

    with pytest.raises(AttentionContractError, match="requires an explicit CUDA AG/RS"):
        AttentionAblationOp()(
            q[:, :, :2],
            k[:, :, :2],
            v[:, :, :2],
            contract=_cp2_contract(),
            backend=VendorCPBackend(),
            config=AttentionAblationConfig(
                strict_core_id=VendorCPBackend.core_id,
                strict_schedule=VendorCPBackend.strict_schedule,
            ),
        )


def test_vendor_core_actual_backend_must_match_selected_backend():
    q, k, v = _qkv()

    class VendorCore:
        backend_id = "vendor.strict_attention"
        core_id = "vendor.strict_core.v1"
        strict_schedule = "vendor.strict_schedule.v1"

        def __call__(self, q, k, v, *, causal, scale):
            del k, v, causal, scale
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros(q.shape[:3], dtype=torch.float32),
                provenance={
                    "strict_core_id": self.core_id,
                    "strict_schedule": self.strict_schedule,
                    "actual_backend": "vendor.other_attention",
                    "production_ready": True,
                    "native_attention_arithmetic": True,
                    "fallback": False,
                    "reference_only": False,
                },
            )

    with pytest.raises(AttentionContractError, match="actual_backend"):
        AttentionAblationOp()(
            q,
            k,
            v,
            contract=_contract(),
            backend=VendorCore(),
            config=AttentionAblationConfig(
                strict_core_id=VendorCore.core_id,
                strict_schedule=VendorCore.strict_schedule,
            ),
        )


def test_deterministic_native_backend_requires_explicit_native_callable():
    q, k, v = _qkv()
    with pytest.raises(AttentionContractError, match="native Attention backend"):
        AttentionAblationOp()(q, k, v, contract=_contract(), backend="native")


def test_attention_wrapper_can_return_dq_dk_dv_from_reference_backend():
    q, k, v = _qkv()
    result = AttentionAblationOp()(
        q,
        k,
        v,
        contract=_contract(),
        return_gradients=True,
        dout=torch.ones_like(q),
    )

    assert result.dq is not None and result.dq.shape == q.shape
    assert result.dk is not None and result.dk.shape == k.shape
    assert result.dv is not None and result.dv.shape == v.shape


def test_attention_backend_is_registered_for_pr230_semantic_resolution():
    from rl_engine.runtime.registry import kernel_registry
    from rl_engine.runtime.semantic_registry import OperatorRequirements

    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="attention",
        requested_backend=BACKEND_ID,
        target="training",
        requirements=OperatorRequirements(
            device="cpu",
            dtype="bfloat16",
            topology={"world_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1},
            alignment_properties={"deterministic": True},
        ),
    )
    instance = session.instantiate(resolution)
    assert isinstance(instance, AttentionAblationOp)
    provenance = session.instance_provenance(resolution, instance)
    assert provenance.backend_id == BACKEND_ID


def test_strict_wrapper_is_bitwise_invariant_to_batch_shape():
    q, k, v = _qkv()
    noise_q, noise_k, noise_v = _qkv()
    contract = _contract()
    batch_contract = AttentionContract(
        role=contract.role,
        mode=contract.mode,
        dtype=contract.dtype,
        batch_size=2,
        query_sequence_length=contract.query_sequence_length,
        head_dim=contract.head_dim,
        causal=contract.causal,
        causal_offsets=(0, 0),
        sharding=contract.sharding,
        reduction=contract.reduction,
        split_kv=contract.split_kv,
    )
    single = AttentionAblationOp()(q, k, v, contract=contract)
    batched = AttentionAblationOp()(
        torch.cat((q, noise_q), dim=0),
        torch.cat((k, noise_k), dim=0),
        torch.cat((v, noise_v), dim=0),
        contract=batch_contract,
    )
    assert torch.equal(single.out[0], batched.out[0])
    assert torch.equal(single.lse[0], batched.lse[0])
