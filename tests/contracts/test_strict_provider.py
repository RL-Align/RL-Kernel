# SPDX-License-Identifier: Apache-2.0
"""CPU tests for identity stability, strict schemas and immutable evidence."""

import copy

import pytest

from rl_engine.contracts.strict_provider import (
    SCHEMA_VERSION,
    CallContext,
    CaseSpec,
    KernelPlan,
    ProviderLock,
    ScopeSpec,
    SnapshotActivationReceipt,
    StrictProviderError,
    WorkerObservation,
    canonical_json,
    fingerprint,
    parse_json,
)

pytestmark = pytest.mark.unit
SHA = "a" * 64


def plan_data():
    return {
        "node": "full_attention",
        "phase": "decode",
        "implementation": "adapter.Rollout",
        "entrypoint": "native.attention",
        "code": {
            "source_inventory_sha256": SHA,
            "binary_inventory_sha256": SHA,
            "compiled_kernel_id": "kernel.v1",
            "compile_key": "key.v1",
        },
        "io": {
            "input_dtype": "bf16",
            "output_dtype": "bf16",
            "layout": "bshd",
            "shape_class": {"head_dim": 256},
            "workspace_policy": "private",
        },
        "numerics": {
            "accumulator_dtype": "fp32",
            "split_policy": "disabled",
            "split_count": 1,
            "reduction_spec": {"rank_tree": [0, 2, 1, 3]},
            "cast_spec": "final_write",
            "atomic_policy": "forbidden",
        },
        "launch": {"tile_config": {"block_m": 16}, "graph_state": "eager", "tuning": "fixed"},
    }


def lock_data():
    plan = KernelPlan.from_dict(plan_data())
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": "cpu.mock.v1",
        "scope": "component.full_attention",
        "model": {
            "model_id": "test/model",
            "revision": "revision-1",
            "config_sha256": SHA,
            "weight_index_sha256": SHA,
            "shard_inventory_sha256": SHA,
        },
        "platform": {
            "kind": "cpu",
            "gpu_arch": "cpu",
            "torch_build": "test-build",
            "hip_runtime": {"status": "not_applicable", "reason": "CPU fixture"},
            "driver_identity": "test-driver",
            "triton_build": "test-build",
        },
        "execution": {
            "topology": {"tp": 1, "ep": 1, "cp": 1, "pp": 1},
            "activation_dtype": "bf16",
            "graph_mode": "eager",
            "tuning_policy": "fixed",
            "fallback_policy": "error",
            "runtime_settings": {},
        },
        "nodes": {
            "full_attention": {
                "semantics_ref": "attention.contract.v1",
                "bindings": {
                    "rollout": {"implementation": "adapter.Rollout", "artifacts_ref": SHA}
                },
                "allowed_plans": {"rollout": [plan.plan_id]},
                "alignment": {"policy": "identical_plan"},
            }
        },
    }


def test_canonical_identity_and_semantic_order():
    original = plan_data()
    reversed_keys = dict(reversed(list(original.items())))
    assert fingerprint(original) == fingerprint(reversed_keys)
    plan = KernelPlan.from_dict(original)
    original["numerics"]["reduction_spec"]["rank_tree"].reverse()
    assert plan.plan_id != KernelPlan.from_dict(original).plan_id
    assert plan.numerics["reduction_spec"]["rank_tree"] == (0, 2, 1, 3)
    with pytest.raises(TypeError):
        plan.numerics["split_count"] = 2
    with pytest.raises(AttributeError):
        plan.phase = "prefill"
    exported = plan.to_dict()
    exported["launch"]["tile_config"]["block_m"] = 32
    assert plan.launch["tile_config"]["block_m"] == 16


def test_role_wrapper_is_excluded_only_from_alignment():
    rollout = KernelPlan.from_dict(plan_data())
    replay_data = plan_data()
    replay_data["implementation"] = "adapter.Replay"
    replay = KernelPlan.from_dict(replay_data)
    assert rollout.plan_id != replay.plan_id
    assert rollout.alignment_id == replay.alignment_id
    replay_data["numerics"]["cast_spec"] = "before_reduce"
    assert rollout.alignment_id != KernelPlan.from_dict(replay_data).alignment_id
    replay_data = plan_data()
    replay_data["code"]["binary_inventory_sha256"] = "b" * 64
    assert rollout.alignment_id != KernelPlan.from_dict(replay_data).alignment_id


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "unknown", "<actual>"])
def test_reject_unresolved_and_non_json_identity(value):
    with pytest.raises(StrictProviderError) as error:
        canonical_json({"nested": {"value": value}})
    assert error.value.field == "$.nested.value"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version="future.v9"),
        lambda d: d.update(unrecognized=True),
        lambda d: d.pop("model"),
        lambda d: d["execution"].update(fallback_policy="allow"),
        lambda d: d["execution"]["topology"].update(tp=True),
        lambda d: d["model"].update(config_sha256="not-a-hash"),
    ],
)
def test_reject_invalid_contract(change):
    data = lock_data()
    change(data)
    with pytest.raises(StrictProviderError) as error:
        ProviderLock.from_dict(data)
    assert error.value.to_dict()["code"] == "invalid_contract"
    assert error.value.field


def test_roundtrip_verifies_supplied_ids_and_does_not_mutate_inputs():
    data = lock_data()
    before = copy.deepcopy(data)
    lock = ProviderLock.from_dict(data)
    assert data == before
    assert ProviderLock.from_dict(lock.to_dict()).lock_id == lock.lock_id
    tampered = lock.to_dict()
    tampered["model"]["revision"] = "revision-2"
    with pytest.raises(StrictProviderError, match="does not match"):
        ProviderLock.from_dict(tampered)
    plan = KernelPlan.from_dict(plan_data()).to_dict()
    plan["launch"]["tile_config"]["block_m"] = 32
    with pytest.raises(StrictProviderError):
        KernelPlan.from_dict(plan)


def test_duplicate_json_keys_are_not_silently_overwritten():
    with pytest.raises(StrictProviderError, match="Duplicate"):
        parse_json('{"split":1,"split":4}')


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("io", "input_dtype", "imaginary_dtype"),
        ("io", "shape_class", {}),
        ("numerics", "split_count", True),
        ("numerics", "reduction_spec", {}),
        ("code", "compile_key", {}),
        ("launch", "tile_config", {}),
    ],
)
def test_missing_or_unknown_numerical_facts_cannot_be_sealed(section, key, value):
    data = plan_data()
    data[section][key] = value
    with pytest.raises(StrictProviderError) as error:
        KernelPlan.from_dict(data)
    assert error.value.field == f"$.{section}.{key}"


def test_worker_call_scope_and_case_records():
    lock = lock_data()
    observation = {key: lock[key] for key in ("model", "platform", "execution")}
    observation.update(
        worker_id="worker-0",
        role="rollout",
        rank=0,
        bindings=lock["nodes"]["full_attention"]["bindings"],
    )
    observation["bindings"] = {"full_attention": observation["bindings"]["rollout"]}
    assert WorkerObservation.from_dict(observation).role == "rollout"
    context = dict(
        case_id="case-1",
        call_id="decode-0",
        node="full_attention",
        phase="decode",
        role="rollout",
        worker_id="worker-0",
        run_id="run-1",
        attempt="1",
        weight_snapshot_id="snapshot-1",
        layer_id=3,
    )
    assert CallContext.from_dict(context).layer_id == 3
    scope = ScopeSpec.from_dict(
        {
            "scope_id": lock["scope"],
            "kind": "component",
            "required_nodes": ["full_attention"],
            "inventory_fingerprint": SHA,
        }
    )
    assert scope.required_nodes == ("full_attention",)
    case = {
        "case_id": "case-1",
        "comparison_context": {"tokens": [1, 2]},
        "required_calls": [
            {
                "node": "full_attention",
                "phase": "decode",
                "layer_id": 3,
                "call_id": "decode-0",
                "role": "rollout",
                "rank": 0,
                "count": 1,
            }
        ],
    }
    assert CaseSpec.from_dict(case).case_id == "case-1"
    # Acceptance policies belong to the numerical harness, not manifest cases.
    for field, value in (
        ("output_names", ["selected_logprob"]),
        ("numerical_policy", {"comparison": "exact"}),
    ):
        with pytest.raises(StrictProviderError, match="Unknown field"):
            CaseSpec.from_dict({**case, field: value})
    case["required_calls"].append(copy.deepcopy(case["required_calls"][0]))
    with pytest.raises(StrictProviderError, match="Duplicate required call"):
        CaseSpec.from_dict(case)


def test_incomplete_snapshot_cannot_claim_completed():
    receipt = dict(
        snapshot_id="snapshot-1",
        generation=0,
        worker_id="worker-0",
        expected_shards=["a", "b"],
        verified_shards=["a"],
        required_refresh=["packed"],
        refreshed_components=[],
        content_digest=SHA,
        completed=True,
        evidence_kind="worker_load",
    )
    with pytest.raises(StrictProviderError, match="Incomplete"):
        SnapshotActivationReceipt.from_dict(receipt)
    receipt["completed"] = False
    incomplete = SnapshotActivationReceipt.from_dict(receipt)
    assert not incomplete.load_complete
    with pytest.raises(StrictProviderError, match="incomplete"):
        incomplete.validate()
