# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from rl_engine.contracts.strict_provider import (
    KernelPlan,
    ProviderLock,
    SnapshotActivationReceipt,
    StrictProviderError,
    fingerprint,
)
from rl_engine.validation.common.strict_provider import (
    artifact_reference,
    build_expectation,
    validate_run,
)

from tests.helpers.strict_provider import Bundle


@pytest.fixture
def bundle(tmp_path):
    return Bundle(tmp_path)


def test_manifest_evidence_does_not_claim_numerical_or_hardware_acceptance(bundle):
    result = bundle.validate()
    assert result["manifest_enforcement"] == "passed"
    assert result["coverage"] == "passed"
    assert result["validation_scope"] == "identity_configuration_provenance"
    assert result["call_count"] == 4
    assert result["numerical_parity"] == "not_evaluated"
    assert result["hardware_validation"] == "not_evaluated"
    assert "compared_elements" not in result
    assert "numeric_evidence_format" not in result
    assert "output_artifacts" not in result["evidence"]


def test_independent_coverage_catches_missing_layer(bundle):
    for summary in bundle.summaries:
        summary["calls"] = summary["calls"][:1]
    bundle.save()
    with pytest.raises(StrictProviderError, match="multiplicities"):
        bundle.validate()


def test_expectation_cannot_omit_an_entire_inventory_layer(bundle):
    expectation = deepcopy(bundle.expectation)
    expectation["cases"][0]["required_calls"] = [
        c for c in expectation["cases"][0]["required_calls"] if c["layer_id"] == "0"
    ]
    expectation.pop("expectation_id")
    expectation["expectation_id"] = fingerprint(expectation)
    with pytest.raises(StrictProviderError, match="omits a selected"):
        validate_run(bundle.lock, expectation, bundle.summary_refs, bundle.root)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "rank", "phase", "duplicate_call"])
def test_worker_roster_multiplicity_and_branch_coverage(bundle, mutation):
    if mutation == "missing":
        bundle.summaries.pop()
    elif mutation == "duplicate":
        bundle.summaries.append(deepcopy(bundle.summaries[0]))
    elif mutation == "rank":
        bundle.summaries[0]["rank"] = 1
    elif mutation == "phase":
        bundle.summaries[0]["calls"][0]["context"]["phase"] = "decode"
    else:
        bundle.summaries[0]["calls"].append(deepcopy(bundle.summaries[0]["calls"][0]))
    bundle.save()
    with pytest.raises(StrictProviderError):
        bundle.validate()


@pytest.mark.parametrize("field", ["lock_id", "run_id", "attempt", "weight_snapshot_id"])
def test_stale_execution_context_is_rejected(bundle, field):
    bundle.summaries[0][field] = "previous_execution"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "stale_evidence"


@pytest.mark.parametrize(
    "mutation", ["footer", "state", "pending", "unauthorized", "generation", "disk_receipt"]
)
def test_incomplete_and_unauthorized_execution_is_rejected(bundle, mutation):
    summary = bundle.summaries[0]
    if mutation == "footer":
        summary["footer"] = {"complete": False, "call_count": 2}
    elif mutation == "state":
        summary["state"] = "failed"
    elif mutation == "pending":
        summary["calls"][0]["completion"]["status"] = "pending"
    elif mutation == "unauthorized":
        summary["calls"][0]["states"].remove("authorized")
    elif mutation == "generation":
        summary["calls"][0]["snapshot_generation"] = 1
    else:
        receipt = summary["snapshot_history"][-1]
        receipt["evidence_kind"] = "disk_checkpoint"
        receipt.pop("receipt_id")
    bundle.save(update_footer=mutation != "footer")
    with pytest.raises(StrictProviderError):
        bundle.validate()


def test_original_artifact_content_change_cannot_be_hidden_by_same_size(bundle):
    original = bundle.original.read_bytes()
    bundle.original.write_bytes(b"X" + original[1:])
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "evidence_hash_mismatch"


@pytest.mark.parametrize(
    "mutation", ["truncate", "outside", "other_attempt", "duplicate_reference"]
)
def test_evidence_content_and_attempt_ownership(bundle, mutation):
    if mutation == "truncate":
        path = bundle.root / bundle.summary_refs[0]["path"]
        path.write_bytes(path.read_bytes()[:-2])
    elif mutation in {"outside", "other_attempt"}:
        path = bundle.root / ("outside.json" if mutation == "outside" else "run1/attempt0/old.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle.summaries[0]), encoding="utf-8")
        bundle.summary_refs[0] = artifact_reference(path, bundle.root)
    else:
        bundle.summary_refs.append(bundle.summary_refs[0])
    with pytest.raises(StrictProviderError):
        bundle.validate()


@pytest.mark.parametrize(
    "field,value",
    [("tests", 0), ("skipped", 1), ("errors", 1), ("failures", 1), ("outcome", "skipped")],
)
def test_mandatory_tests_no_empty_or_skip_pass(bundle, field, value):
    bundle.report["cases"][0][field] = value
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "mandatory_case_failed"


def test_zero_cases_and_duplicate_cases_are_rejected(bundle):
    for cases in ([], [bundle.case, bundle.case]):
        with pytest.raises(StrictProviderError):
            build_expectation(
                bundle.lock,
                bundle.scope,
                cases,
                bundle.roster,
                bundle.inventory,
                run_id="run1",
                attempt="attempt1",
                weight_snapshot_id="snapshot1",
                identity_artifacts=bundle.expectation["identity_artifacts"],
            )


@pytest.mark.parametrize(
    "mutation", ["tokens", "positions", "mask", "state_boundary", "logprob_semantics"]
)
def test_call_context_must_match_predeclared_case(bundle, mutation):
    context = bundle.summaries[1]["calls"][0]["context"]
    context["metadata"]["comparison_context"][mutation] = "different_context"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "execution_context_mismatch"


@pytest.mark.parametrize("value", [None, {}, [], "unverified_context"])
def test_missing_or_invalid_call_context_cannot_pass(bundle, value):
    metadata = bundle.summaries[0]["calls"][0]["context"]["metadata"]
    if value is None:
        metadata.pop("comparison_context")
    else:
        metadata["comparison_context"] = value
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "execution_context_mismatch"


def test_undeclared_case_cannot_supply_its_own_execution_context(bundle):
    bundle.summaries[0]["calls"][0]["context"]["case_id"] = "unregistered_case"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "invalid_coverage"


def test_component_cannot_claim_full_model_by_changing_label(bundle):
    inventory = {**bundle.inventory, "2": ["moe"]}
    scope = {**bundle.scope, "kind": "full_model", "inventory_fingerprint": fingerprint(inventory)}
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            bundle.lock,
            scope,
            [bundle.case],
            bundle.roster,
            inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_scope"


@pytest.mark.parametrize("node", ["full_attention", "dense_gemm", "moe_combine"])
@pytest.mark.parametrize("platform_kind", ["cpu", "rocm"])
def test_operator_sessions_produce_consumable_evidence(tmp_path, node, platform_kind):
    """Exercise both workers with callbacks, not native operator/GPU kernels."""
    from rl_engine.contracts.strict_provider import CallContext, WorkerObservation
    from rl_engine.integrations.common.provider_observer import CompletionReceipt, ResolvedCall
    from rl_engine.runtime.strict_provider import StrictProviderSession

    bundle = Bundle(tmp_path, node=node, platform_kind=platform_kind)
    summaries = []
    launches = []
    completions = []

    def complete(output):
        completions.append(output)
        return CompletionReceipt(True, kind=bundle.completion_kind)

    for template in bundle.summaries:
        worker = {k: template[k] for k in ("worker_id", "role", "rank")}
        receipt = SnapshotActivationReceipt.from_dict(template["snapshot_history"][-1])
        session = StrictProviderSession(
            bundle.lock,
            **worker,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            snapshot_receipt=receipt,
        )
        session.bind(
            node, worker["role"], WorkerObservation.from_dict(template["observations"][node])
        )
        for call in template["calls"]:
            call["context"]["call_id"] = f"prompt_layer_{call['context']['layer_id']}"
            context = CallContext.from_dict(call["context"])
            output = object()

            def launch(output=output):
                launches.append(output)
                return output

            resolved = ResolvedCall(
                bundle.plan,
                launch,
                call_context=context,
                recheck=lambda: {"actual_shape": [2]},
                completion=complete,
            )
            assert session.execute(resolved, context) is output
        summaries.append(session.finish())
    bundle.summaries = summaries
    for call in bundle.expectation["cases"][0]["required_calls"]:
        call["call_id"] = f"prompt_layer_{call['layer_id']}"
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    bundle.save()
    result = bundle.validate()
    assert result["manifest_enforcement"] == "passed"
    assert result["call_count"] == 4
    assert len(launches) == 4 and completions == launches
    assert result["numerical_parity"] == result["hardware_validation"] == "not_evaluated"


@pytest.mark.parametrize("mutation", ["inventory_id", "unrelated_file", "logical_path"])
def test_artifact_inventory_must_cover_actual_locked_binding(bundle, mutation):
    inventory = deepcopy(bundle.artifact_inventory)
    if mutation == "inventory_id":
        inventory["inventory_sha256"] = "f" * 64
    elif mutation == "logical_path":
        inventory["files"][0]["logical_path"] = "../provider.bin"
    else:
        path = bundle.directory / "unrelated.bin"
        path.write_bytes(b"different provider")
        reference = artifact_reference(path, bundle.root)
        inventory["files"] = [{"logical_path": "unrelated.bin", "reference": reference}]
        inventory["inventory_sha256"] = fingerprint(
            [
                {
                    "path": "unrelated.bin",
                    "size": reference["size_bytes"],
                    "sha256": reference["sha256"],
                }
            ]
        )
    ref = bundle.write("artifact_inventory.json", inventory)
    bundle.expectation["identity_artifacts"] = [ref]
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    with pytest.raises(StrictProviderError):
        bundle.validate()


def test_plan_code_identity_must_have_original_file_evidence(bundle):
    call = bundle.summaries[0]["calls"][0]
    plan = deepcopy(call["plan"])
    plan["code"]["binary_inventory_sha256"] = "f" * 64
    plan.pop("plan_id")
    plan.pop("alignment_id")
    updated = KernelPlan.from_dict(plan)
    call.update(plan=updated.to_dict(), plan_id=updated.plan_id, alignment_id=updated.alignment_id)
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "missing_artifact_evidence"


def test_full_model_scope_rejected_even_with_matching_self_declared_inventory(bundle):
    lock_data = bundle.lock.to_dict()
    lock_data.pop("lock_id")
    lock_data["scope"] = "full_model"
    lock = ProviderLock.from_dict(lock_data)
    scope = {**bundle.scope, "scope_id": "full_model", "kind": "full_model"}
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            lock,
            scope,
            [bundle.case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_scope"


def test_multiplicity_two_rejected_until_runner_assigns_distinct_call_ids(bundle):
    case = deepcopy(bundle.case)
    case["required_calls"][0]["count"] = 2
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            bundle.lock,
            bundle.scope,
            [case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "invalid_coverage"


def test_repeated_logical_call_rejected_even_with_new_ticket(bundle):
    copied = deepcopy(bundle.summaries[0]["calls"][0])
    copied["call_ticket_id"] = "new_ticket"
    bundle.summaries[0]["calls"].append(copied)
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "duplicate_call"


def test_gpu_lock_rejects_synchronous_completion_from_worker_summary(bundle):
    lock_data = bundle.lock.to_dict()
    lock_data.pop("lock_id")
    lock_data["platform"]["kind"] = "rocm"
    bundle.lock = ProviderLock.from_dict(lock_data)
    bundle.artifact_inventory["lock_id"] = bundle.lock.lock_id
    inventory_ref = bundle.write("artifact_inventory.json", bundle.artifact_inventory)
    bundle.expectation["lock_id"] = bundle.lock.lock_id
    bundle.expectation["identity_artifacts"] = [inventory_ref]
    bundle.expectation.pop("expectation_id")
    bundle.expectation["expectation_id"] = fingerprint(bundle.expectation)
    for summary in bundle.summaries:
        summary["lock_id"] = bundle.lock.lock_id
        summary["observations"]["test_operator"]["platform"]["kind"] = "rocm"
    bundle.save()
    with pytest.raises(StrictProviderError) as exc:
        bundle.validate()
    assert exc.value.code == "incomplete_calls"


def test_graph_lock_cannot_be_accepted_without_a_graph_execution_protocol(bundle):
    lock = bundle.lock.to_dict()
    lock.pop("lock_id")
    lock["execution"]["graph_mode"] = "graph"
    graph_lock = ProviderLock.from_dict(lock)
    with pytest.raises(StrictProviderError) as exc:
        build_expectation(
            graph_lock,
            bundle.scope,
            [bundle.case],
            bundle.roster,
            bundle.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=bundle.expectation["identity_artifacts"],
        )
    assert exc.value.code == "unsupported_graph"
