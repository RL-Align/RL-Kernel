# SPDX-License-Identifier: Apache-2.0
"""Check launch side effects and failures on CPU; ROCm completion is mocked."""

from copy import deepcopy

import pytest

from rl_engine.contracts.strict_provider import (
    CallContext,
    KernelPlan,
    ProviderLock,
    SnapshotActivationReceipt,
    StrictProviderError,
    WorkerObservation,
    fingerprint,
)
from rl_engine.integrations.common.provider_observer import CompletionReceipt, ResolvedCall
from rl_engine.runtime.strict_provider import StrictProviderSession


def make_plan(**overrides):
    data = {
        "node": "test_operator",
        "phase": "decode",
        "implementation": "fake.Operator",
        "entrypoint": "fake.forward",
        "code": {
            "source_inventory_sha256": "a" * 64,
            "binary_inventory_sha256": "b" * 64,
            "compiled_kernel_id": "fake-binary-1",
            "compile_key": "fake-key-1",
        },
        "io": {
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "layout": "BHD",
            "shape_class": "single-token",
            "workspace_policy": "private",
        },
        "numerics": {
            "accumulator_dtype": "float32",
            "split_policy": "disabled",
            "split_count": 1,
            "reduction_spec": "fixed-v1",
            "cast_spec": "bf16-final",
            "atomic_policy": "forbidden",
        },
        "launch": {"tile_config": {"block": 64}, "graph_state": "eager", "tuning": "fixed"},
    }
    data.update(overrides)
    return KernelPlan.from_dict(data)


def make_lock(plan, *, platform_kind="cpu"):
    return ProviderLock.from_dict(
        {
            "schema_version": "rl_kernel.strict_provider.v1",
            "profile_id": f"fake-{platform_kind}-v1",
            "scope": f"component.{plan.node}",
            "model": {
                "model_id": "fake",
                "revision": "r1",
                "config_sha256": "a" * 64,
                "weight_index_sha256": "b" * 64,
                "shard_inventory_sha256": "c" * 64,
            },
            "platform": {
                "kind": platform_kind,
                "gpu_arch": "gfx942" if platform_kind == "rocm" else "cpu",
                "torch_build": "fake-build",
                "hip_runtime": "fake-hip",
                "driver_identity": "fake-driver",
                "triton_build": "fake-triton",
            },
            "execution": {
                "topology": {"tp": 1, "ep": 1, "cp": 1, "pp": 1},
                "activation_dtype": "bfloat16",
                "graph_mode": "eager",
                "tuning_policy": "fixed",
                "fallback_policy": "error",
                "runtime_settings": {},
            },
            "nodes": {
                plan.node: {
                    "semantics_ref": f"{plan.node}-v1",
                    "bindings": {
                        role: {"implementation": plan.implementation, "artifacts_ref": "d" * 64}
                        for role in ("rollout", "replay")
                    },
                    "allowed_plans": {role: [plan.plan_id] for role in ("rollout", "replay")},
                    "alignment": {"policy": "identical_plan"},
                }
            },
        }
    )


def receipt(snapshot_id="s0", generation=0, worker_id="w0", **changes):
    data = {
        "snapshot_id": snapshot_id,
        "generation": generation,
        "worker_id": worker_id,
        "expected_shards": ["shard0"],
        "verified_shards": ["shard0"],
        "required_refresh": ["derived-cache"],
        "refreshed_components": ["derived-cache"],
        "content_digest": "e" * 64,
        "completed": True,
        "evidence_kind": "worker_load",
    }
    data.update(changes)
    return SnapshotActivationReceipt.from_dict(data)


def observation(lock, **changes):
    data = {name: lock.to_dict()[name] for name in ("model", "platform", "execution")}
    data.update(
        {
            "worker_id": "w0",
            "role": "rollout",
            "rank": 0,
            "weight_snapshot_id": "s0",
            "bindings": {
                node: dict(spec["bindings"]["rollout"]) for node, spec in lock.nodes.items()
            },
        }
    )
    data.update(changes)
    return WorkerObservation.from_dict(data)


def context(**changes):
    data = {
        "case_id": "case0",
        "call_id": "decode0",
        "node": "test_operator",
        "phase": "decode",
        "role": "rollout",
        "worker_id": "w0",
        "run_id": "run0",
        "attempt": "1",
        "weight_snapshot_id": "s0",
        "layer_id": 0,
    }
    data.update(changes)
    return CallContext.from_dict(data)


@pytest.fixture
def setup_session():
    plan = make_plan()
    lock = make_lock(plan)
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    session.bind("test_operator", "rollout", observation(lock))
    return plan, lock, session


def fake_call(plan, launches, metadata=None, completion=None, call_context=None, generation=0):
    metadata = {"split": 1} if metadata is None else metadata

    def launch():
        launches.append(deepcopy(metadata))
        return 42

    return ResolvedCall(
        plan,
        launch,
        call_context=call_context or context(),
        snapshot_generation=generation,
        recheck=lambda: metadata,
        completion=completion,
    )


def test_execute_owns_authorization_and_completion(setup_session):
    plan, lock, session = setup_session
    launches = []
    before = lock.to_dict()
    assert session.execute(fake_call(plan, launches), context()) == 42
    assert launches == [{"split": 1}]
    summary = session.finish()
    assert summary["calls"][0]["states"] == ["prepared", "authorized", "launched", "completed"]
    assert summary["footer"] == {"complete": True, "call_count": 1}
    summary_id = summary.pop("summary_id")
    assert summary_id == fingerprint(summary)
    assert "manifest_enforcement" not in summary and "numerical_parity" not in summary
    assert lock.to_dict() == before


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("code", "compile_key", "another-key"),
        ("code", "compiled_kernel_id", "another-binary"),
        ("numerics", "accumulator_dtype", "float16"),
        ("launch", "tile_config", {"block": 128}),
        ("io", "input_dtype", "float16"),
        ("io", "layout", "HBD"),
        ("io", "workspace_policy", "shared"),
    ],
)
def test_plan_drift_blocks_before_launch(setup_session, section, field, value):
    plan, lock, session = setup_session
    data = plan.to_dict()
    data.pop("plan_id")
    data.pop("alignment_id")
    data[section][field] = value
    launches = []
    before = lock.to_dict()
    with pytest.raises(StrictProviderError, match="not sealed") as error:
        session.execute(fake_call(KernelPlan.from_dict(data), launches), context())
    assert error.value.code == "plan_mismatch"
    assert launches == []
    assert session.state == "failed" and lock.to_dict() == before


def test_changed_entrypoint_and_split_rejected(setup_session):
    plan, _, session = setup_session
    numerics = dict(plan.to_dict()["numerics"], split_policy="fixed", split_count=4)
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(
            fake_call(make_plan(entrypoint="fake.other", numerics=numerics), launches), context()
        )
    assert error.value.code == "plan_mismatch" and launches == []


def test_mutable_metadata_rejected_before_launch(setup_session):
    plan, _, session = setup_session
    metadata = {"indices": [7], "split": 1}
    launches = []
    call = fake_call(plan, launches, metadata)
    metadata["indices"][0] = 23
    with pytest.raises(StrictProviderError) as error:
        session.execute(call, context())
    assert error.value.code == "metadata_changed" and launches == []
    with pytest.raises(StrictProviderError) as later:
        session.finish()
    assert later.value.code == "session_terminal"


@pytest.mark.parametrize(
    "changed",
    [
        {"worker_id": "w1"},
        {"role": "replay"},
        {"attempt": "2"},
        {"run_id": "other"},
        {"weight_snapshot_id": "s1"},
    ],
)
def test_cross_context_rejected(setup_session, changed):
    plan, _, session = setup_session
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context(**changed))
    assert error.value.code == "context_mismatch" and launches == []


def test_one_shot_call_cannot_cross_session(setup_session):
    plan, lock, session = setup_session
    launches = []
    call = fake_call(plan, launches)
    session.execute(call, context())
    other = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    other.bind("test_operator", "rollout", observation(lock))
    with pytest.raises(StrictProviderError) as error:
        other.execute(call, context(call_id="decode1"))
    assert error.value.code == "call_reused" and len(launches) == 1
    assert other.state == "failed"


def test_duplicate_context_and_finished_session_rejected(setup_session):
    plan, _, session = setup_session
    launches = []
    session.execute(fake_call(plan, launches), context())
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "call_reused" and len(launches) == 1


def test_finish_is_terminal_and_empty_summary_is_not_acceptance(setup_session):
    plan, _, session = setup_session
    summary = session.finish()
    assert summary["calls"] == []
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "session_terminal" and launches == []


@pytest.mark.parametrize("mode", ["pending", "raises", "invalid"])
def test_pending_failed_completion_cannot_finish(setup_session, mode):
    plan, _, session = setup_session
    launches = []

    def complete(output):
        assert output == 42
        if mode == "raises":
            raise RuntimeError("async device error")
        return CompletionReceipt(False) if mode == "pending" else True

    with pytest.raises((StrictProviderError, RuntimeError)):
        session.execute(fake_call(plan, launches, completion=complete), context())
    assert len(launches) == 1 and session.state == "failed"
    with pytest.raises(StrictProviderError):
        session.finish()


def test_completed_callback_is_checked(setup_session):
    plan, _, session = setup_session
    seen = []
    launches = []

    def complete(output):
        seen.append(output)
        return CompletionReceipt(True, kind="fake-event")

    session.execute(fake_call(plan, launches, completion=complete), context())
    assert seen == [42]
    assert session.finish()["calls"][0]["completion"] == {
        "kind": "fake-event",
        "status": "completed",
    }


def test_snapshot_update_requires_complete_worker_receipt(setup_session):
    plan, _, session = setup_session
    assert session.begin_weight_update("s1") == 1
    session.activate_snapshot(receipt("s1", 1))
    launches = []
    updated_context = context(weight_snapshot_id="s1")
    session.execute(
        fake_call(plan, launches, call_context=updated_context, generation=1), updated_context
    )
    summary = session.finish()
    assert summary["snapshot_generation"] == 1
    assert summary["calls"][0]["snapshot_generation"] == 1
    assert len(summary["snapshot_history"]) == 2


@pytest.mark.parametrize("mode", ["string", "disk", "wrong_worker", "old_generation", "partial"])
def test_invalid_snapshot_activation_is_terminal(setup_session, mode):
    _, _, session = setup_session
    session.begin_weight_update("s1")
    changes = {
        "disk": {"evidence_kind": "disk_checkpoint"},
        "wrong_worker": {"worker_id": "w1"},
        "old_generation": {"generation": 0},
        "partial": {"completed": False, "refreshed_components": []},
    }
    values = {"snapshot_id": "s1", "generation": 1} | changes.get(mode, {})
    value = "s1" if mode == "string" else receipt(**values)
    with pytest.raises(StrictProviderError):
        session.activate_snapshot(value)
    assert session.state == "failed"


def test_snapshot_inactive_blocks_launch(setup_session):
    plan, _, session = setup_session
    session.begin_weight_update("s1")
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context(weight_snapshot_id="s1"))
    assert error.value.code == "snapshot_inactive" and launches == []


def test_unbound_provider_blocks_launch(setup_session):
    plan, lock, _ = setup_session
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "unbound_provider" and launches == []


def test_graph_plan_is_rejected_even_if_sealed():
    plan = make_plan(
        launch={"tile_config": {"block": 64}, "graph_state": "graph", "tuning": "fixed"}
    )
    lock = make_lock(plan)
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    session.bind("test_operator", "rollout", observation(lock))
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "unsupported_graph" and launches == []


def test_actual_binary_identity_mismatch_at_bind(setup_session):
    _, lock, _ = setup_session
    session = StrictProviderSession(
        lock, role="rollout", worker_id="w0", run_id="run0", attempt="1", weight_snapshot_id="s0"
    )
    wrong = observation(
        lock,
        bindings={"test_operator": {"implementation": "fake.Operator", "artifacts_ref": "f" * 64}},
    )
    with pytest.raises(StrictProviderError) as error:
        session.bind("test_operator", "rollout", wrong)
    assert error.value.field == "bindings.test_operator.artifacts_ref"
    assert session.state == "failed"


def test_initial_snapshot_name_does_not_authorize_execution(setup_session):
    plan, lock, _ = setup_session
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
    )
    session.bind("test_operator", "rollout", observation(lock))
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "snapshot_inactive" and launches == []


def test_old_prepared_dispatch_cannot_cross_weight_generation(setup_session):
    plan, _, session = setup_session
    launches = []
    old_call = fake_call(plan, launches)
    # A repeated snapshot name still represents a new load generation.
    session.begin_weight_update("s0")
    session.activate_snapshot(receipt("s0", 1))
    with pytest.raises(StrictProviderError) as error:
        session.execute(old_call, context())
    assert error.value.field == "resolved_call.snapshot_generation"
    assert launches == []


def test_prepared_dispatch_cannot_change_logical_layer(setup_session):
    plan, _, session = setup_session
    launches = []
    call = fake_call(plan, launches)
    with pytest.raises(StrictProviderError) as error:
        session.execute(call, context(layer_id=1))
    assert error.value.field == "resolved_call.context_id" and launches == []


def test_gpu_requires_completion_before_launch(setup_session):
    plan, lock, _ = setup_session
    lock_data = lock.to_dict()
    lock_data.pop("lock_id")
    lock_data["platform"]["kind"] = "rocm"
    lock_data["platform"]["gpu_arch"] = "gfx942"
    gpu_lock = ProviderLock.from_dict(lock_data)
    session = StrictProviderSession(
        gpu_lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    session.bind("test_operator", "rollout", observation(gpu_lock))
    launches = []
    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(plan, launches), context())
    assert error.value.code == "unobservable_completion" and launches == []


@pytest.mark.parametrize(
    "field", ["gpu_arch", "torch_build", "hip_runtime", "driver_identity", "triton_build"]
)
def test_rocm_worker_platform_must_match_the_lock(field):
    plan = make_plan()
    lock = make_lock(plan, platform_kind="rocm")
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    actual = observation(lock).to_dict()
    actual["platform"][field] = "different-runtime"
    with pytest.raises(StrictProviderError) as error:
        session.bind(plan.node, "rollout", WorkerObservation.from_dict(actual))
    assert error.value.code == "binding_mismatch"
    assert error.value.field == f"observation.platform.{field}"
    assert session.state == "failed"


def test_rocm_session_keeps_operator_bindings_and_calls_separate():
    """Three mock operators share a session; no ROCm kernels are executed."""
    nodes = ("full_attention", "dense_gemm", "moe_combine")
    plans = {
        node: make_plan(
            node=node, implementation=f"fake.{node}.Provider", entrypoint=f"fake.{node}"
        )
        for node in nodes
    }
    data = make_lock(plans[nodes[0]], platform_kind="rocm").to_dict()
    data.pop("lock_id")
    data["scope"] = "component.test_operators"
    data["nodes"] = {node: make_lock(plan).to_dict()["nodes"][node] for node, plan in plans.items()}
    lock = ProviderLock.from_dict(data)
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    actual = observation(lock)
    for node in nodes:
        session.bind(node, "rollout", actual)
    launches = []
    completions = []

    def complete(output):
        completions.append(output)
        return CompletionReceipt(True, kind="mock_event")

    for node, plan in plans.items():
        call_context = context(node=node)
        call = fake_call(
            plan,
            launches,
            metadata={"operator": node, "split": 1},
            completion=complete,
            call_context=call_context,
        )
        assert session.execute(call, call_context) == 42
    summary = session.finish()
    assert [launch["operator"] for launch in launches] == list(nodes)
    assert len(completions) == len(nodes)
    assert set(summary["bindings"]) == set(nodes)
    assert [call["context"]["node"] for call in summary["calls"]] == list(nodes)
    assert [call["plan_id"] for call in summary["calls"]] == [p.plan_id for p in plans.values()]


def test_launch_exception_cannot_be_reused_or_finished(setup_session):
    plan, _, session = setup_session
    launches = []

    def launch():
        launches.append(True)
        raise RuntimeError("provider failed")

    call = ResolvedCall(plan, launch, call_context=context(), recheck=lambda: {})
    with pytest.raises(RuntimeError, match="provider failed"):
        session.execute(call, context())
    assert launches == [True]
    with pytest.raises(StrictProviderError) as error:
        session.execute(call, context())
    assert error.value.code == "session_terminal"
    with pytest.raises(StrictProviderError):
        session.finish()


def test_unknown_dispatch_recheck_blocks_before_launch(setup_session):
    plan, _, session = setup_session
    launches = []
    known = True

    def recheck():
        if not known:
            raise StrictProviderError("unobservable_plan", "Branch identity is unavailable")
        return {"branch": "fixed"}

    call = ResolvedCall(
        plan, lambda: launches.append(True), call_context=context(), recheck=recheck
    )
    known = False
    with pytest.raises(StrictProviderError) as error:
        session.execute(call, context())
    assert error.value.code == "unobservable_plan" and launches == []


def test_same_call_name_is_legal_for_different_layers(setup_session):
    plan, _, session = setup_session
    launches = []
    session.execute(fake_call(plan, launches), context())
    next_layer = context(layer_id=1)
    session.execute(fake_call(plan, launches, call_context=next_layer), next_layer)
    assert len(session.finish()["calls"]) == 2


def test_identical_runtime_facts_have_distinct_call_tickets(setup_session):
    plan, _, session = setup_session
    launches = []
    session.execute(fake_call(plan, launches), context())
    second_context = context(call_id="decode1")
    session.execute(fake_call(plan, launches, call_context=second_context), second_context)
    left, right = session.finish()["calls"]
    assert left["runtime_id"] == right["runtime_id"]
    assert left["call_ticket_id"] != right["call_ticket_id"]


@pytest.mark.parametrize("completion_kind", [None, "synchronous"])
def test_plan_cpu_label_cannot_override_gpu_completion_requirement(setup_session, completion_kind):
    plan, lock, _ = setup_session
    plan_data = plan.to_dict()
    plan_data.pop("plan_id")
    plan_data.pop("alignment_id")
    plan_data["launch"]["device_kind"] = "cpu"
    overridden_plan = KernelPlan.from_dict(plan_data)
    lock_data = make_lock(overridden_plan).to_dict()
    lock_data.pop("lock_id")
    lock_data["platform"]["kind"] = "rocm"
    lock_data["platform"]["gpu_arch"] = "gfx942"
    gpu_lock = ProviderLock.from_dict(lock_data)
    session = StrictProviderSession(
        gpu_lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    session.bind("test_operator", "rollout", observation(gpu_lock))
    launches = []
    completion = None
    if completion_kind is not None:

        def completion(output):
            return CompletionReceipt(True, kind=completion_kind)

    with pytest.raises(StrictProviderError) as error:
        session.execute(fake_call(overridden_plan, launches, completion=completion), context())
    assert error.value.code == "unobservable_completion"
    assert len(launches) == (0 if completion_kind is None else 1)
    with pytest.raises(StrictProviderError):
        session.finish()


def test_binding_extensions_are_locked_while_diagnostics_are_not(setup_session):
    _, lock, _ = setup_session
    session = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    observed = observation(lock, diagnostics={"backend_display_name": "AITER"})
    session.bind("test_operator", "rollout", observed)
    assert session.finish()["observations"]["test_operator"]["diagnostics"] == {
        "backend_display_name": "AITER"
    }
    other = StrictProviderSession(
        lock,
        role="rollout",
        worker_id="w0",
        run_id="run0",
        attempt="1",
        weight_snapshot_id="s0",
        snapshot_receipt=receipt(),
    )
    observed_data = observed.to_dict()
    observed_data["bindings"]["test_operator"]["backend_display_name"] = "AITER"
    with pytest.raises(StrictProviderError) as error:
        other.bind("test_operator", "rollout", WorkerObservation.from_dict(observed_data))
    assert error.value.field == "bindings.test_operator.backend_display_name"
