# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T04 reference acceptance; fixed goldens, injected faults, and owner-local EP mocks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rl_engine.alignment.cross_config.artifacts import ArtifactError, ArtifactStore
from rl_engine.kernels.ops.pytorch.moe import (
    MergeContext,
    MergeContractError,
    MergeIdentity,
    MergeSource,
    shared_residual_merge_fwd,
    tensor_sha256,
)
from rl_engine.kernels.ops.pytorch.moe.shared_residual_merge import BOUNDARIES, ORDER

FIXTURE = Path(__file__).parent / "fixtures" / "p6_t04_merge.json"
FIXTURE_CHECKSUM = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
CASES = json.loads(FIXTURE.read_text())["cases"]


def test_fixture_checksum():
    assert FIXTURE_CHECKSUM == "79813b6999c56489b4e08575fd71ed62d6d65cdd7ad61a90d7727a83bceb20d4"


def make_inputs(case=CASES[0], *, device="cpu", dtype=torch.float32):
    return tuple(
        torch.tensor(case[role], dtype=torch.float32 if role == "routed" else dtype, device=device)
        for role in ("routed", "shared", "residual")
    )


def make_context(inputs, *, case_id="T04-LOCAL-MERGE", rank=0, ep=1, tokens=None):
    identity = MergeIdentity(
        case_id=case_id,
        run_id="synthetic-t04",
        pass_id="forward-0",
        checkpoint_fingerprint="synthetic-no-checkpoint",
        weight_fingerprint="synthetic-no-weights",
        route_plan_fingerprint="synthetic-route",
        exchange_plan_fingerprint="synthetic-exchange",
        combine_plan_fingerprint="synthetic-combine",
        fixture_checksum=FIXTURE_CHECKSUM,
        global_token_ids=tokens or tuple(range(inputs[0].shape[0])),
    )
    sources = tuple(
        MergeSource(role, f"synthetic-{role}", identity, tensor_sha256(value))
        for role, value in zip(("routed", "shared", "residual"), inputs, strict=True)
    )
    return MergeContext(
        identity=identity,
        expected_source_ids=tuple(s.source_id for s in sources),
        sources=sources,
        upstream_events=("route_weight",),
        rank=rank,
        token_owner_ranks=(rank,) * len(identity.global_token_ids),
        shared_replica_ranks=tuple(range(ep)),
    )


def assert_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8),
        b.detach().cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_frozen_golden_and_debug(case):
    inputs = make_inputs(case)
    context = make_context(inputs, case_id=case["id"])
    result = shared_residual_merge_fwd(*inputs, context=context, debug=True)
    golden = torch.tensor(case["bf16_words"], dtype=torch.uint16).view(torch.bfloat16)
    assert_bytes(result.token_moe_row_bf16, golden)
    for key, name in zip(BOUNDARIES[:2], ("after_shared", "after_residual"), strict=True):
        assert_bytes(result.debug_boundaries[key], torch.tensor(case[name], dtype=torch.float32))
    quiet = shared_residual_merge_fwd(*inputs, context=context)
    assert quiet.debug_boundaries == {}
    assert quiet.receipt == result.receipt
    assert_bytes(quiet.token_moe_row_bf16, golden)
    for count in (
        "route_weight_applied_count",
        "shared_applied_count",
        "residual_applied_count",
        "local_downcast_count",
    ):
        assert result.receipt[count] == 1
    assert result.receipt["executed_events"] == list(ORDER)


def test_wrong_arithmetic_is_detectable():
    r, s, x = make_inputs(CASES[1])
    good = shared_residual_merge_fwd(r, s, x, context=make_context((r, s, x)))
    wrong_order = (r + (s + x)).bfloat16()
    early_cast = ((r + s).bfloat16().float() + x).bfloat16()
    assert wrong_order[0, 0].item() == 1 and good.token_moe_row_bf16[0, 0].item() == 0
    assert early_cast[0, 1].item() == 0 and good.token_moe_row_bf16[0, 1].item() == 1


@pytest.mark.parametrize(
    "event,status",
    [
        ("shared", "SHARED_APPLIED_NOT_ONCE"),
        ("residual", "RESIDUAL_APPLIED_NOT_ONCE"),
        ("cast_bf16", "EARLY_OR_MULTIPLE_DOWNCAST"),
        ("route_weight", "ROUTE_WEIGHT_APPLIED_TWICE"),
        ("mhc_post", "INVALID_COMBINE_PLAN"),
    ],
)
def test_reject_prior_application(event, status):
    inputs = make_inputs()
    context = make_context(inputs)
    context = replace(context, upstream_events=context.upstream_events + (event,))
    with pytest.raises(MergeContractError, match=status):
        shared_residual_merge_fwd(*inputs, context=context)


@pytest.mark.parametrize(
    "role,status",
    [("shared", "SHARED_APPLIED_NOT_ONCE"), ("residual", "RESIDUAL_APPLIED_NOT_ONCE")],
)
@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_duplicate_source(role, status, duplicate):
    inputs = make_inputs()
    context = make_context(inputs)
    source = next(s for s in context.sources if s.role == role)
    sources = (
        context.sources + (source,)
        if duplicate
        else tuple(s for s in context.sources if s.role != role)
    )
    with pytest.raises(MergeContractError, match=status):
        shared_residual_merge_fwd(*inputs, context=replace(context, sources=sources))


@pytest.mark.parametrize(
    "field",
    [
        "case_id",
        "run_id",
        "pass_id",
        "checkpoint_fingerprint",
        "weight_fingerprint",
        "route_plan_fingerprint",
        "exchange_plan_fingerprint",
        "combine_plan_fingerprint",
        "fixture_checksum",
        "global_token_ids",
    ],
)
def test_identity_precedes_numerics(field):
    inputs = make_inputs()
    context = make_context(inputs)
    source = context.sources[1]
    identity = replace(
        source.identity, **{field: (42,) if field == "global_token_ids" else "wrong"}
    )
    context = replace(
        context,
        sources=(context.sources[0], replace(source, identity=identity), context.sources[2]),
    )
    inputs[0].fill_(float("nan"))
    with pytest.raises(MergeContractError, match="IDENTITY_DRIFT: shared.identity"):
        shared_residual_merge_fwd(*inputs, context=context)


@pytest.mark.parametrize(
    "mutation,status",
    [
        ({"merge_order": ("residual", "shared", "cast_bf16")}, "ADDITION_ORDER_MISMATCH"),
        ({"upstream_events": ()}, "INVALID_COMBINE_PLAN"),
        ({"token_owner_ranks": (1,)}, "SHARED_APPLIED_NOT_ONCE"),
        ({"rank": -1}, "MISSING_PROVENANCE"),
        ({"shared_replica_ranks": (0, 0)}, "INVALID_COMBINE_PLAN"),
        ({"foundation_abi": "wrong"}, "SCHEMA_MISMATCH"),
        ({"receipt_version": "wrong"}, "SCHEMA_MISMATCH"),
        ({"input_row_weighted": False}, "INVALID_COMBINE_PLAN"),
        ({"route_weight_owner": "P6"}, "INVALID_COMBINE_PLAN"),
        (
            {"expected_source_ids": ("wrong", "synthetic-shared", "synthetic-residual")},
            "IDENTITY_DRIFT",
        ),
    ],
)
def test_invalid_context(mutation, status):
    inputs = make_inputs()
    with pytest.raises(MergeContractError, match=status):
        shared_residual_merge_fwd(*inputs, context=replace(make_context(inputs), **mutation))


def test_mhc_post_input_rejected():
    inputs = make_inputs()
    context = make_context(inputs)
    context = replace(
        context, sources=(*context.sources[:2], replace(context.sources[2], role="mhc_post"))
    )
    with pytest.raises(MergeContractError, match="INVALID_COMBINE_PLAN"):
        shared_residual_merge_fwd(*inputs, context=context)


def test_checksum_detects_stale_tensor():
    inputs = make_inputs()
    context = make_context(inputs)
    inputs[1][0, 0] += 1
    with pytest.raises(MergeContractError, match="IDENTITY_DRIFT: shared.tensor_checksum"):
        shared_residual_merge_fwd(*inputs, context=context)


@pytest.mark.parametrize("role", range(3))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(role, value):
    inputs = make_inputs()
    inputs[role][0, 0] = value
    with pytest.raises(MergeContractError, match="NON_FINITE"):
        shared_residual_merge_fwd(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("stage", ["after_shared", "after_residual", "final_bf16"])
def test_intermediate_and_cast_overflow(stage):
    maxval = torch.finfo(torch.float32).max
    values = {
        "after_shared": (maxval, maxval, -maxval),
        "after_residual": (maxval, 0.0, maxval),
        "final_bf16": (maxval, 0.0, 0.0),
    }[stage]
    inputs = tuple(torch.tensor([[v]], dtype=torch.float32) for v in values)
    with pytest.raises(MergeContractError, match=f"NON_FINITE: P6.merge.{stage}"):
        shared_residual_merge_fwd(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("kind", ["routed_bf16", "shared_fp16", "broadcast", "strided"])
def test_unsupported_payloads(kind):
    inputs = list(make_inputs())
    if kind == "routed_bf16":
        inputs[0] = inputs[0].bfloat16()
    elif kind == "shared_fp16":
        inputs[1] = inputs[1].half()
    elif kind == "broadcast":
        inputs[1] = inputs[1][:, :1].contiguous()
    else:
        inputs[1] = torch.ones(1, 6)[:, ::2]
    with pytest.raises(MergeContractError):
        shared_residual_merge_fwd(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_ep_replication_owner_local_mock(ep):
    # No collectives: replicas exist on every mock rank; exactly one owner merges
    # each token. This is WS2 semantic evidence, not multi-GPU certification.
    inputs = tuple(v.repeat(16, 1) for v in make_inputs())
    canonical = shared_residual_merge_fwd(*inputs, context=make_context(inputs))
    gathered = torch.empty_like(canonical.token_moe_row_bf16)
    for rank in range(ep):
        tokens = tuple(range(rank, 16, ep))
        local = tuple(v[list(tokens)].contiguous() for v in inputs)
        context = make_context(local, rank=rank, ep=ep, tokens=tokens)
        result = shared_residual_merge_fwd(*local, context=context)
        gathered[list(tokens)] = result.token_moe_row_bf16
        assert result.receipt["shared_applied_count"] == 1
        assert result.receipt["merge_order_hash"] == canonical.receipt["merge_order_hash"]
        if ep > 1:
            with pytest.raises(MergeContractError, match="SHARED_APPLIED_NOT_ONCE"):
                shared_residual_merge_fwd(*local, context=replace(context, rank=(rank + 1) % ep))
    assert_bytes(gathered, canonical.token_moe_row_bf16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_immutability_and_autocast(dtype):
    inputs = make_inputs(dtype=dtype)
    snapshots = tuple(v.clone() for v in inputs)
    context = make_context(inputs)
    baseline = shared_residual_merge_fwd(*inputs, context=context)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = shared_residual_merge_fwd(*inputs, context=context)
    assert_bytes(result.token_moe_row_bf16, baseline.token_moe_row_bf16)
    for value, saved in zip(inputs, snapshots, strict=True):
        assert_bytes(value, saved)
    operations = result.receipt["actual_provenance"]["observed_aten_operations"]
    assert [op["output_dtype"] for op in operations if op["operator"] == "aten.add.Tensor"] == [
        "torch.float32",
        "torch.float32",
    ]
    assert sum(op["output_dtype"] == "torch.bfloat16" for op in operations) == 1


@pytest.mark.parametrize("width", [1, 17, 129, 4096])
def test_batch_chunk_padding_permutation(width):
    generator = torch.Generator().manual_seed(71)
    inputs = tuple(torch.randn(7, width, generator=generator) for _ in range(3))
    expected = shared_residual_merge_fwd(*inputs, context=make_context(inputs)).token_moe_row_bf16
    permutation = [6, 2, 0, 1, 5, 3, 4]
    permuted = tuple(v[permutation] for v in inputs)
    result = shared_residual_merge_fwd(
        *permuted, context=make_context(permuted, tokens=tuple(permutation))
    )
    assert_bytes(result.token_moe_row_bf16, expected[permutation])
    for start, end in ((0, 1), (1, 4), (4, 7)):
        chunk = tuple(v[start:end] for v in inputs)
        result = shared_residual_merge_fwd(
            *chunk, context=make_context(chunk, tokens=tuple(range(start, end)))
        )
        assert_bytes(result.token_moe_row_bf16, expected[start:end])
    padded = tuple(torch.cat((v, torch.zeros(3, width))) for v in inputs)
    result = shared_residual_merge_fwd(*padded, context=make_context(padded))
    assert_bytes(result.token_moe_row_bf16[:7], expected)


def test_duplicate_and_empty_token_batch():
    inputs = tuple(v.repeat(2, 1) for v in make_inputs())
    context = make_context(inputs)
    context = replace(context, identity=replace(context.identity, global_token_ids=(1, 1)))
    with pytest.raises(MergeContractError, match="AMBIGUOUS_GLOBAL_TOKEN_MAPPING"):
        shared_residual_merge_fwd(*inputs, context=context)
    empty = tuple(v[:0] for v in inputs)
    with pytest.raises(MergeContractError, match="AMBIGUOUS_GLOBAL_TOKEN_MAPPING"):
        shared_residual_merge_fwd(*empty, context=make_context(empty))


def test_reference_registry_entry():
    from rl_engine.kernels.ops.pytorch.moe.shared_residual_merge import BACKEND_ID
    from rl_engine.kernels.registry import KernelRegistry
    from rl_engine.kernels.semantic_registry import (
        OperatorRequirements,
        OperatorResolutionError,
        OperatorSession,
    )

    session = OperatorSession(KernelRegistry().semantic)
    resolution = session.resolve(
        semantic_op="shared_residual_merge",
        requested_backend=BACKEND_ID,
        target="rollout",
        requirements=OperatorRequirements(device="cpu", dtype="float32"),
    )
    op = session.instantiate(resolution)
    inputs = make_inputs()
    result = op(*inputs, context=make_context(inputs))
    assert result.receipt["scope"] == "local_forward_reference"
    assert resolution.descriptor.determinism_or_alignment_properties["reference_only"] is True
    with pytest.raises(OperatorResolutionError):
        session.resolve(
            semantic_op="shared_residual_merge",
            requested_backend=BACKEND_ID,
            target="rollout",
            requirements=OperatorRequirements(
                device="cpu", dtype="float32", alignment_properties={"gpu_launch_observable": True}
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/ROCm GPU: not certified")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_gpu_golden(case):
    inputs = make_inputs(case, device="cuda")
    result = shared_residual_merge_fwd(
        *inputs, context=make_context(inputs, case_id=case["id"]), debug=True
    )
    golden = torch.tensor(case["bf16_words"], dtype=torch.uint16).view(torch.bfloat16)
    assert_bytes(result.token_moe_row_bf16, golden)
    for key, field in zip(BOUNDARIES[:2], ("after_shared", "after_residual"), strict=True):
        assert_bytes(result.debug_boundaries[key], torch.tensor(case[field], dtype=torch.float32))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        quiet = shared_residual_merge_fwd(*inputs, context=make_context(inputs, case_id=case["id"]))
    assert_bytes(quiet.token_moe_row_bf16, result.token_moe_row_bf16)
    assert quiet.receipt == result.receipt


def test_sealed_reference_evidence(tmp_path):
    root = Path(os.environ.get("RL_KERNEL_T04_ARTIFACT_DIR", tmp_path))
    store = ArtifactStore(root)
    required = ("t04_receipt.json", "t04_debug.pt")
    for case in CASES:
        inputs = make_inputs(case)
        context = make_context(inputs, case_id=case["id"])
        result = shared_residual_merge_fwd(*inputs, context=context, debug=True)
        experiment = {
            "context": json.loads(json.dumps(asdict(context))),
            "implementation": result.receipt["actual_provenance"],
        }
        # Identity/build changes cannot silently resume an existing experiment.
        store.initialize_experiment(case["id"], experiment=experiment, plan=[])
        attempt = store.create_attempt(case["id"], case["id"])
        store.write_json(attempt, required[0], result.receipt)
        store.write_tensor_bundle(attempt, required[1], result.debug_boundaries)
        store.complete_attempt(
            attempt,
            required=required,
            summary={
                "schema_version": "cross_config.complete.v1",
                "case_id": case["id"],
                "attempt_id": attempt.name,
                "status": "LOCAL_REFERENCE_ONLY",
            },
        )
        store.validate_completed_attempt(attempt, required=required, expected_case_id=case["id"])
        loaded = store.load_tensor_bundle(attempt / required[1])["tensors"]
        receipt = json.loads((attempt / required[0]).read_text())
        for boundary in receipt["boundaries"]:
            assert tensor_sha256(loaded[boundary["key"]]) == boundary["checksum"]
        with pytest.raises(ArtifactError, match="resume metadata differs"):
            store.initialize_experiment(case["id"], experiment={"wrong_run": True}, plan=[])
    # Exercise corruption on an isolated copy, never alter a sealed report.
    corrupted = tmp_path / "corrupted" / attempt.name
    shutil.copytree(attempt, corrupted)
    (corrupted / "t04_receipt.json").write_text("{}")
    with pytest.raises(ArtifactError, match="artifact hash does not match"):
        store.validate_completed_attempt(corrupted, required=required)
