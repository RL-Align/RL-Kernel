# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_engine.reference.logprob.logp import NativeLogpOp
from rl_engine.runtime.operators import (
    OperatorBridge,
    OperatorOverride,
    selected_logprobs_with_operator,
)
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    OperatorResolutionPolicy,
    SemanticOperatorCatalog,
)
from rl_engine.runtime.semantic_registry import (
    implementation_fingerprint as fingerprint_implementation,
)
from rl_engine.validation.common import selected_logprobs_reference
from rl_engine.validation.cross_config.planner import V1_KNOBS
from rl_engine.validation.cross_config.runtime import (
    AdapterMaterialization,
    KnobApplication,
    RuntimeBinding,
    RuntimeMaterializationError,
    RuntimeTools,
)
from rl_engine.validation.cross_config.schema import (
    ExperimentCase,
    IsolationScope,
    MaterializationStatus,
    SemanticIdentitySpec,
)
from rl_engine.validation.fixtures.cpu_cross_config import CpuSmokeMaterializer
from rl_engine.validation.fixtures.smoke_ops import (
    SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
    SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
    SmokeOnlyLogpOffset,
    register_smoke_operators,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TEMPORARY_DOCSTRING = "TEMPORARY TEST SCAFFOLD - NOT A PRODUCTION RL-KERNEL OPERATOR"
_TOPOLOGIES = {
    "rollout": {
        "world_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
    },
    "training": {"world_size": 1, "sharding": "unsharded"},
}


def _identity() -> SemanticIdentitySpec:
    return SemanticIdentitySpec(
        checkpoint_id="tiny-cpu-checkpoint",
        model_version="weights-v1",
        tokenizer_policy="synthetic-tokenizer-v1",
        token_ids=((1, 2, 3),),
        selected_token_ids=((0, 2, 3),),
        active_mask=((False, True, True),),
        attention_mask=((True, True, True),),
        pre_update_state="iteration-0",
    )


def _requested(**overrides):
    requested = {
        "batch": {"size": 2},
        "rollout": {
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "dtype": "float32",
            "enable_prefix_caching": False,
            "enforce_eager": True,
        },
        "training": {
            "attention_backend": "eager",
            "compute_dtype": "float32",
            "sharding": "unsharded",
        },
        "logp": {"backend": "rlkernel.reference_logp"},
    }
    for path, value in overrides.items():
        current = requested
        parts = path.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
    return requested


def _case(
    *,
    case_id: str = "case-1",
    changed_paths=(),
    requested=None,
    execution_binding=None,
) -> ExperimentCase:
    return ExperimentCase(
        case_id=case_id,
        experiment_id="runtime-test",
        scenario_id="S0",
        identity=_identity(),
        requested=requested or _requested(),
        changed_paths=changed_paths,
        execution_binding=execution_binding or {},
        contract_fingerprint="contract-sha",
        scenario_fingerprint="scenario-sha",
    )


def _value_at(requested, path: str):
    current = requested
    for part in path.split("."):
        current = current[part]
    return current


def _readback(requested):
    return {path: _value_at(requested, path) for path in V1_KNOBS if path != "batch.size"}


class _RuntimeTestAdapter:
    """Small observable fake kept beside the lifecycle tests that need it."""

    runtime_kind = "test_runtime"

    def __init__(self, *, actual_readback=None):
        self.actual_readback = dict(actual_readback or {})

    @property
    def implementation_fingerprint(self):
        return fingerprint_implementation(
            type(self),
            instance=self,
            entrypoints=("materialize",),
        )

    def materialize(self, normalized, descriptors):
        applications = []
        for path, descriptor in descriptors.items():
            requested = _value_at(normalized, path)
            materialized = requested
            actual = self.actual_readback.get(path)
            status = MaterializationStatus.UNOBSERVABLE
            reason = "no runtime readback is available"

            unsupported = (path == "rollout.context_parallel_size" and requested != 1) or (
                path == "training.sharding" and requested != "unsharded"
            )
            if unsupported:
                materialized = actual = None
                status = MaterializationStatus.UNSUPPORTED
                reason = "the test adapter does not support this topology"
            elif path == "batch.size":
                actual = requested
                status = MaterializationStatus.APPLIED
                reason = "batch size is observed at scorer invocation"
            elif path in self.actual_readback:
                status = (
                    MaterializationStatus.APPLIED
                    if actual == requested
                    else MaterializationStatus.FALLBACK
                )
                reason = "runtime readback was captured"

            applications.append(
                KnobApplication(
                    path=path,
                    requested=requested,
                    materialized=materialized,
                    actual=actual,
                    lifecycle=descriptor.lifecycle,
                    status=status,
                    evidence={"reason": reason},
                    critical=descriptor.critical,
                )
            )

        backend = _value_at(normalized, "logp.backend")
        return AdapterMaterialization(
            applications=tuple(applications),
            binding=RuntimeBinding(
                batch_size=_value_at(normalized, "batch.size"),
                side_configs={"rollout": {}, "training": {}},
                topology={
                    "rollout": {
                        "world_size": 1,
                        "tensor_parallel_size": _value_at(
                            normalized, "rollout.tensor_parallel_size"
                        ),
                        "context_parallel_size": _value_at(
                            normalized, "rollout.context_parallel_size"
                        ),
                    },
                    "training": {
                        "world_size": 1,
                        "sharding": _value_at(normalized, "training.sharding"),
                    },
                },
                scorer={},
                operator_backends={"rollout": backend, "training": backend},
                runtime_kind=self.runtime_kind,
            ),
        )


def _cpu_materializer() -> CpuSmokeMaterializer:
    backends = {
        "rollout": "rlkernel.reference_logp",
        "training": "rlkernel.reference_logp",
    }
    return CpuSmokeMaterializer(
        requested_operator_backends=backends,
        actual_operator_backends=backends,
    )


def _requirements(
    *,
    target: str = "rollout",
    device: str = "cpu",
) -> OperatorRequirements:
    return OperatorRequirements(
        device=device,
        dtype="float32",
        topology=_TOPOLOGIES[target],
        alignment_properties={"deterministic": True},
    )


def _catalog() -> SemanticOperatorCatalog:
    """Clone repository descriptors so each test owns registration state."""

    return SemanticOperatorCatalog(OperatorBridge().catalog.backend_descriptors())


def test_cpu_materialization_records_all_ten_knobs_across_three_stages():
    case = _case()
    materialization = RuntimeTools().materialize(case, _cpu_materializer())
    applications = {application.path: application for application in materialization.applications}

    assert len(V1_KNOBS) == 10
    assert set(applications) == set(V1_KNOBS)
    assert materialization.materialized_case.status is MaterializationStatus.APPLIED
    assert materialization.executable_in_strict_mode
    RuntimeTools.require_executable(materialization, strict=True)

    for path, descriptor in V1_KNOBS.items():
        application = applications[path]
        assert application.requested == _value_at(case.requested, path)
        assert application.lifecycle is descriptor.lifecycle
        assert application.status is MaterializationStatus.APPLIED
        assert application.evidence["reason"]

    provenance = materialization.provenance
    assert provenance.requested == case.requested
    assert provenance.normalized == case.requested
    assert provenance.materialized["batch"]["size"] == 2
    assert provenance.materialized["rollout"] == {
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "dtype": "float32",
        "enable_prefix_caching": False,
        "enforce_eager": True,
    }
    assert provenance.materialized["training"] == {
        "attention_backend": "eager",
        "compute_dtype": "float32",
        "sharding": "unsharded",
    }
    assert provenance.materialized["logp"]["backend"] == {
        "rollout": "rlkernel.reference_logp",
        "training": "rlkernel.reference_logp",
    }
    assert provenance.actual == provenance.materialized
    assert provenance.implementation_fingerprint == _cpu_materializer().implementation_fingerprint
    assert provenance.evidence["adapter_implementation_fingerprint"] == (
        provenance.implementation_fingerprint
    )

    binding = materialization.binding
    assert binding.runtime_kind == "cpu_smoke"
    assert binding.side_configs["rollout"]["device"] == "cpu"
    assert binding.side_configs["training"]["device"] == "cpu"
    assert binding.side_configs["training"]["dtype"] == "float32"
    assert binding.side_configs["rollout"] == {
        "device": "cpu",
        "dtype": "float32",
        "enable_prefix_caching": False,
        "enforce_eager": True,
    }
    assert binding.topology["rollout"] == {
        "world_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
    }
    assert binding.topology["training"] == {"world_size": 1, "sharding": "unsharded"}
    assert binding.scorer == {
        "mode": "reference",
        "use_cache": False,
        "attention_backend": "eager",
        "output_dtype": "float32",
    }
    with pytest.raises(TypeError):
        binding.side_configs["rollout"]["device"] = "cuda"
    with pytest.raises(TypeError):
        applications["batch.size"].evidence["reason"] = "changed after fingerprinting"


def test_lifecycle_fingerprints_allow_request_reuse_and_isolate_engine_and_process_changes(
    monkeypatch,
):
    tools = RuntimeTools()
    baseline_case = _case(case_id="baseline")
    baseline_adapter = _RuntimeTestAdapter(actual_readback=_readback(baseline_case.requested))
    baseline = tools.materialize(
        baseline_case,
        baseline_adapter,
    )

    batch_requested = _requested(**{"batch.size": 1})
    batch = tools.materialize(
        _case(
            case_id="batch",
            requested=batch_requested,
            changed_paths=("batch.size",),
        ),
        _RuntimeTestAdapter(actual_readback=_readback(batch_requested)),
    )
    assert batch.materialized_case.isolation_scope is IsolationScope.REQUEST
    assert batch.materialized_case.construction_fingerprint == (
        baseline.materialized_case.construction_fingerprint
    )
    assert batch.materialized_case.distributed_context_fingerprint == (
        baseline.materialized_case.distributed_context_fingerprint
    )
    assert batch.materialized_case.process_fingerprint == (
        baseline.materialized_case.process_fingerprint
    )
    assert tools.can_reuse(baseline, batch)

    dtype_requested = _requested(**{"rollout.dtype": "bfloat16"})
    dtype = tools.materialize(
        _case(
            case_id="dtype",
            requested=dtype_requested,
            changed_paths=("rollout.dtype",),
        ),
        _RuntimeTestAdapter(actual_readback=_readback(dtype_requested)),
    )
    assert dtype.materialized_case.isolation_scope is IsolationScope.ENGINE_CONSTRUCTION
    assert dtype.materialized_case.construction_fingerprint != (
        baseline.materialized_case.construction_fingerprint
    )
    assert dtype.materialized_case.process_fingerprint == (
        baseline.materialized_case.process_fingerprint
    )
    assert not tools.can_reuse(baseline, dtype)

    topology_requested = _requested(**{"rollout.tensor_parallel_size": 2})
    topology = tools.materialize(
        _case(
            case_id="topology",
            requested=topology_requested,
            changed_paths=("rollout.tensor_parallel_size",),
        ),
        _RuntimeTestAdapter(actual_readback=_readback(topology_requested)),
    )
    assert topology.materialized_case.isolation_scope is IsolationScope.PROCESS
    assert topology.materialized_case.process_fingerprint != (
        baseline.materialized_case.process_fingerprint
    )
    assert not tools.can_reuse(baseline, topology)

    rebound_case = replace(
        baseline_case,
        case_id="rebound",
        execution_binding={"operator_case": {"rollout_options": {"offset": 0.1}}},
    )
    rebound = tools.materialize(
        rebound_case,
        _RuntimeTestAdapter(actual_readback=_readback(rebound_case.requested)),
    )
    assert rebound.materialized_case.construction_fingerprint == (
        baseline.materialized_case.construction_fingerprint
    )
    assert not tools.can_reuse(baseline, rebound)

    changed_identity_case = replace(
        baseline_case,
        case_id="changed-identity",
        identity=replace(baseline_case.identity, pre_update_state="iteration-1"),
    )
    changed_identity = tools.materialize(changed_identity_case, baseline_adapter)
    assert changed_identity.materialized_case.construction_fingerprint == (
        baseline.materialized_case.construction_fingerprint
    )
    assert not tools.can_reuse(baseline, changed_identity)

    original_materialize = _RuntimeTestAdapter.materialize

    def materialize_with_same_result(self, normalized, descriptors):
        return original_materialize(self, normalized, descriptors)

    monkeypatch.setattr(
        _RuntimeTestAdapter,
        "materialize",
        materialize_with_same_result,
    )
    changed_adapter = tools.materialize(
        baseline_case,
        _RuntimeTestAdapter(actual_readback=_readback(baseline_case.requested)),
    )
    assert changed_adapter.provenance.actual == baseline.provenance.actual
    assert (
        changed_adapter.provenance.implementation_fingerprint
        != baseline.provenance.implementation_fingerprint
    )
    assert not tools.can_reuse(baseline, changed_adapter)


def test_materialization_fails_closed_for_fallback_unobservable_and_unsupported_paths():
    tools = RuntimeTools()

    incomplete_descriptors = {
        path: descriptor for path, descriptor in V1_KNOBS.items() if path != "batch.size"
    }
    with pytest.raises(RuntimeMaterializationError, match="missing descriptors"):
        RuntimeTools(incomplete_descriptors).materialize(
            _case(case_id="missing-descriptor"),
            _RuntimeTestAdapter(actual_readback=_readback(_requested())),
        )

    fallback_requested = _requested(**{"training.attention_backend": "flash_attention_2"})
    fallback_readback = _readback(fallback_requested)
    fallback_readback["training.attention_backend"] = "eager"
    fallback = tools.materialize(
        _case(
            case_id="fallback",
            requested=fallback_requested,
            changed_paths=("training.attention_backend",),
        ),
        _RuntimeTestAdapter(actual_readback=fallback_readback),
    )
    assert fallback.materialized_case.status is MaterializationStatus.FALLBACK
    with pytest.raises(RuntimeMaterializationError, match=r"training\.attention_backend"):
        tools.require_executable(fallback, strict=True)
    tools.require_executable(fallback, strict=False)

    unobservable = tools.materialize(
        _case(case_id="unobservable"),
        _RuntimeTestAdapter(),
    )
    assert unobservable.materialized_case.status is MaterializationStatus.UNOBSERVABLE
    with pytest.raises(RuntimeMaterializationError, match="no runtime readback"):
        tools.require_executable(unobservable, strict=False)

    unsupported_requested = _requested(
        **{
            "rollout.context_parallel_size": 4,
            "training.sharding": "fsdp",
        }
    )
    unsupported = tools.materialize(
        _case(
            case_id="unsupported",
            requested=unsupported_requested,
            changed_paths=("rollout.context_parallel_size", "training.sharding"),
        ),
        _RuntimeTestAdapter(),
    )
    unsupported_paths = {
        application.path
        for application in unsupported.applications
        if application.status is MaterializationStatus.UNSUPPORTED
    }
    assert unsupported_paths == {"rollout.context_parallel_size", "training.sharding"}
    assert unsupported.materialized_case.status is MaterializationStatus.UNSUPPORTED
    with pytest.raises(RuntimeMaterializationError, match=r"rollout\.context_parallel_size"):
        tools.require_executable(unsupported, strict=False)

    cpu_only_requested = _requested(**{"rollout.tensor_parallel_size": 2})
    cpu_only = tools.materialize(
        _case(
            case_id="cpu-only",
            requested=cpu_only_requested,
            changed_paths=("rollout.tensor_parallel_size",),
        ),
        _cpu_materializer(),
    )
    assert cpu_only.materialized_case.status is MaterializationStatus.UNSUPPORTED
    assert cpu_only.binding.side_configs["rollout"]["device"] == "cpu"
    assert cpu_only.binding.side_configs["training"]["device"] == "cpu"


def test_operator_binding_selects_rollout_training_and_both_without_side_leakage():
    bridge = OperatorBridge()
    requirements = {
        "rollout": _requirements(target="rollout"),
        "training": _requirements(target="training"),
    }
    assert requirements["rollout"].to_dict()["schema_version"] == (
        "rlkernel.semantic_operator.requirements.v1"
    )

    for target, expected_targets in (
        ("rollout", {"rollout"}),
        ("training", {"training"}),
        ("both", {"rollout", "training"}),
    ):
        resolved = bridge.resolve_override(
            OperatorOverride.for_target(
                semantic_op="selected_logprob",
                backend_id="rlkernel.reference_logp",
                target=target,
            ),
            requirements=requirements,
            strict=True,
        )
        selected_targets = {
            side for side in ("rollout", "training") if resolved.for_target(side) is not None
        }
        assert selected_targets == expected_targets

        instances = {}
        for side in expected_targets:
            resolution = resolved.for_target(side)
            assert resolution is not None
            assert resolution.to_dict()["schema_version"] == (
                "rlkernel.semantic_operator.resolution.v1"
            )
            assert resolution.descriptor.to_dict()["schema_version"] == (
                "rlkernel.semantic_operator.backend_descriptor.v1"
            )
            assert resolution.trace.to_dict()["schema_version"] == (
                "rlkernel.semantic_operator.resolution_trace.v1"
            )
            instance = bridge.instantiate(resolved, target=side)
            instances[side] = instance
            assert isinstance(instance, NativeLogpOp)
            provenance = bridge.instance_provenance(
                resolved,
                target=side,
                instance=instance,
            )
            assert provenance.backend_id == "rlkernel.reference_logp"
            assert provenance.target == side
            assert provenance.instance_fingerprint
            assert provenance.to_dict()["schema_version"] == (
                "rlkernel.semantic_operator.instance_provenance.v1"
            )
            with pytest.raises(TypeError):
                provenance.factory_options["unexpected"] = True
        if target == "both":
            assert instances["rollout"] is not instances["training"]

    catalog = _catalog()
    descriptor = catalog.backend_descriptor("selected_logprob", "rlkernel.reference_logp")
    assert descriptor is not None
    catalog.register_backend(
        replace(descriptor, supported_topologies={"*": "*"}),
        replace=True,
    )
    asymmetric = OperatorBridge(catalog).resolve_override(
        OperatorOverride.for_target(
            semantic_op="selected_logprob",
            backend_id="rlkernel.reference_logp",
            target="both",
        ),
        requirements={
            "rollout": OperatorRequirements(
                device="cpu",
                dtype="float32",
                topology={"world_size": 2, "tensor_parallel_size": 2},
            ),
            "training": OperatorRequirements(
                device="cpu",
                dtype="float32",
                topology={"world_size": 1, "sharding": "fsdp"},
            ),
        },
    )
    assert asymmetric.rollout is not None and asymmetric.training is not None
    assert asymmetric.rollout.requirements.topology != asymmetric.training.requirements.topology

    strict_session = _catalog().session()
    with pytest.raises(OperatorResolutionError, match="topology"):
        strict_session.resolve(
            semantic_op="selected_logprob",
            requested_backend="rlkernel.reference_logp",
            target="rollout",
            requirements=OperatorRequirements(device="cpu", dtype="float32", topology={}),
            strict=True,
        )
    session = catalog.session()
    with pytest.raises(OperatorResolutionError, match="not registered") as unsupported:
        session.resolve(
            semantic_op="selected_logprob",
            requested_backend="missing.backend",
            target="rollout",
            requirements=_requirements(),
            strict=True,
        )
    assert unsupported.value.trace.status == "unsupported"
    assert unsupported.value.trace.fallback_attempts == ()

    native_requirements = OperatorRequirements(
        device="cpu",
        dtype="float32",
        topology=_TOPOLOGIES["training"],
    )
    with pytest.raises(OperatorResolutionError, match="not exactly observable"):
        session.resolve(
            semantic_op="selected_logprob",
            requested_backend="native",
            target="training",
            requirements=native_requirements,
            strict=True,
        )
    native = session.resolve(
        semantic_op="selected_logprob",
        requested_backend="native",
        target="training",
        requirements=native_requirements,
        strict=False,
    )
    assert native.trace.status == "unobservable"
    assert native.trace.concrete_backend is None


@pytest.mark.smoke_operator
def test_smoke_package_is_temporary_cpu_only_and_disabled_without_two_explicit_opt_ins():
    from rl_engine.validation.fixtures.smoke_ops import (
        smoke_only_logp_offset,
        smoke_only_logp_reference,
    )

    assert inspect.getdoc(smoke_only_logp_reference) == _TEMPORARY_DOCSTRING
    assert inspect.getdoc(smoke_only_logp_offset) == _TEMPORARY_DOCSTRING

    manifest = (
        _REPOSITORY_ROOT / "rl_engine/validation/fixtures/smoke_ops/SMOKE_OPERATORS.md"
    ).read_text(encoding="utf-8")
    for required_text in (
        "temporary test scaffolding",
        "smoke_only_logp_reference.py",
        "smoke_only_logp_offset.py",
        "allow_smoke_operators=True",
        "delete this package",
    ):
        assert required_text in manifest

    catalog = _catalog()
    for backend_id in (
        SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
        SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
    ):
        assert catalog.backend_descriptor("selected_logprob", backend_id) is None

    with pytest.raises(PermissionError, match="allow_smoke_operators=True"):
        register_smoke_operators(catalog)

    descriptors = register_smoke_operators(catalog, allow_smoke_operators=True)
    assert {descriptor.backend_id for descriptor in descriptors} == {
        SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
        SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
    }
    for descriptor in descriptors:
        assert descriptor.supported_devices == frozenset({"cpu"})
        assert descriptor.is_smoke_only is True
        disabled_session = catalog.session(
            OperatorResolutionPolicy(strict=True, allow_test_backends=False)
        )
        with pytest.raises(OperatorResolutionError, match="explicit opt-in"):
            disabled_session.resolve(
                semantic_op="selected_logprob",
                requested_backend=descriptor.backend_id,
                target="training",
                requirements=_requirements(target="training"),
            )
        enabled_session = catalog.session(
            OperatorResolutionPolicy(strict=True, allow_test_backends=True)
        )
        with pytest.raises(OperatorResolutionError) as error:
            enabled_session.resolve(
                semantic_op="selected_logprob",
                requested_backend=descriptor.backend_id,
                target="training",
                requirements=_requirements(target="training", device="cuda"),
            )
        failed = {
            decision.capability
            for decision in error.value.trace.capability_decisions
            if not decision.passed
        }
        assert failed == {"device"}

    assert SmokeOnlyLogpOffset().offset == 0.0
    with pytest.raises(PermissionError, match="allow_smoke_operators=True"):
        SmokeOnlyLogpOffset(offset=0.01)


@pytest.mark.smoke_operator
def test_explicit_smoke_opt_in_runs_both_sides_on_cpu_with_sealed_provenance():
    catalog = _catalog()
    register_smoke_operators(catalog, allow_smoke_operators=True)
    bridge = OperatorBridge(
        catalog,
        policy=OperatorResolutionPolicy(strict=True, allow_test_backends=True),
    )
    resolved = bridge.resolve_override(
        OperatorOverride.for_target(
            semantic_op="selected_logprob",
            backend_id=SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
            target="both",
        ),
        requirements={
            "rollout": _requirements(target="rollout"),
            "training": _requirements(target="training"),
        },
        strict=True,
    )
    instances = {
        target: bridge.instantiate(resolved, target=target) for target in ("rollout", "training")
    }
    logits = torch.tensor(
        [[[1.0, 2.0, -1.0], [0.0, 3.0, 1.0], [2.0, 0.0, 4.0]]],
        device="cpu",
    )
    token_ids = torch.tensor([[1, -100, 2]], device="cpu")
    active_mask = torch.tensor([[True, False, True]], device="cpu")
    expected = selected_logprobs_reference(logits, token_ids, mask=active_mask)

    outputs = {}
    for target, instance in instances.items():
        output = selected_logprobs_with_operator(
            instance,
            logits,
            token_ids,
            active_mask=active_mask,
        )
        outputs[target] = output
        assert output.device.type == "cpu"
        torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)
        assert torch.count_nonzero(output[~active_mask]).item() == 0

        provenance = bridge.instance_provenance(
            resolved,
            target=target,
            instance=instance,
        )
        assert provenance.backend_id == SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID
        assert provenance.target == target
        assert provenance.concrete_implementation.endswith("SmokeOnlyLogpReference")
        assert provenance.descriptor_fingerprint
        assert provenance.instance_fingerprint

    for invalid_temperature in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite and greater than zero"):
            selected_logprobs_with_operator(
                instances["rollout"],
                logits,
                token_ids,
                active_mask=active_mask,
                temperature=invalid_temperature,
            )

    assert instances["rollout"] is not instances["training"]
    torch.testing.assert_close(outputs["rollout"], outputs["training"], atol=0.0, rtol=0.0)
