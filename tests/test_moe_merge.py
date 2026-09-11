# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MoE merge: fixed goldens, injected faults, and owner-local EP checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.alignment.cross_config.artifacts import ArtifactError, ArtifactStore
from rl_engine.alignment.testing.moe_merge import (
    BACKEND_ID,
    BOUNDARIES,
    ORDER,
    MergeContext,
    MergeContractError,
    MergeIdentity,
    MergeSource,
    check_moe_merge,
    tensor_sha256,
)
from rl_engine.kernels.ops.pytorch.moe import MoeMergeOp, shared_residual_merge_fwd

FIXTURE = Path(__file__).parent / "fixtures" / "moe_merge.json"
FIXTURE_CHECKSUM = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA/ROCm GPU"),
        ),
    ]
)
def device(request):
    return request.param


def test_fixture_checksum():
    assert FIXTURE_CHECKSUM == "f92fb938160285f54de88f746dd814471598ff117701103bfac8c55e870f11cc"


def make_inputs(case=CASES[0], *, device="cpu", dtype=torch.float32):
    return tuple(
        torch.tensor(
            case[role],
            dtype=torch.float32 if role == "routed" else dtype,
            device=device,
        )
        for role in ("routed", "shared", "residual")
    )


def make_context(inputs, *, case_id="simple_merge", rank=0, ep=1, tokens=None):
    identity = MergeIdentity(
        case_id=case_id,
        run_id="synthetic-merge",
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
def test_frozen_golden_and_debug(case, device):
    inputs = make_inputs(case, device=device)
    context = make_context(inputs, case_id=case["id"])
    result = check_moe_merge(*inputs, context=context, debug=True)
    golden = torch.tensor(case["bf16_words"], dtype=torch.uint16).view(torch.bfloat16)
    assert_bytes(result.token_moe_row_bf16, golden)
    assert_bytes(shared_residual_merge_fwd(*inputs), golden)
    op = MoeMergeOp()
    assert_bytes(op(*inputs), golden)
    for actual, key in zip(op.forward_with_intermediates(*inputs), BOUNDARIES, strict=True):
        assert_bytes(actual, result.debug_boundaries[key])
    for key, name in zip(BOUNDARIES[:2], ("after_shared", "after_residual"), strict=True):
        assert_bytes(result.debug_boundaries[key], torch.tensor(case[name], dtype=torch.float32))
    quiet = check_moe_merge(*inputs, context=context)
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
    assert result.receipt["reference_events"] == list(ORDER)


def test_wrong_arithmetic_is_detectable(device):
    r, s, x = make_inputs(CASES[1], device=device)
    good = shared_residual_merge_fwd(r, s, x)
    wrong_order = (r + (s + x)).bfloat16()
    early_cast = ((r + s).bfloat16().float() + x).bfloat16()
    assert wrong_order[0, 0].item() == 1 and good[0, 0].item() == 0
    assert early_cast[0, 1].item() == 0 and good[0, 1].item() == 1


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
def test_reject_prior_application(event, status, device):
    inputs = make_inputs(device=device)
    context = make_context(inputs)
    context = replace(context, upstream_events=context.upstream_events + (event,))
    with pytest.raises(MergeContractError, match=status):
        check_moe_merge(*inputs, context=context)


@pytest.mark.parametrize(
    "role,status",
    [("shared", "SHARED_APPLIED_NOT_ONCE"), ("residual", "RESIDUAL_APPLIED_NOT_ONCE")],
)
@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_duplicate_source(role, status, duplicate, device):
    inputs = make_inputs(device=device)
    context = make_context(inputs)
    source = next(s for s in context.sources if s.role == role)
    sources = (
        context.sources + (source,)
        if duplicate
        else tuple(s for s in context.sources if s.role != role)
    )
    with pytest.raises(MergeContractError, match=status):
        check_moe_merge(*inputs, context=replace(context, sources=sources))


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
def test_identity_precedes_numerics(field, device):
    inputs = make_inputs(device=device)
    context = make_context(inputs)
    source = context.sources[1]
    identity = replace(
        source.identity, **{field: (42,) if field == "global_token_ids" else "wrong"}
    )
    context = replace(
        context,
        sources=(
            context.sources[0],
            replace(source, identity=identity),
            context.sources[2],
        ),
    )
    inputs[0].fill_(float("nan"))
    with pytest.raises(MergeContractError, match="IDENTITY_DRIFT: shared.identity"):
        check_moe_merge(*inputs, context=context)


@pytest.mark.parametrize(
    "mutation,status",
    [
        (
            {"merge_order": ("residual", "shared", "cast_bf16")},
            "ADDITION_ORDER_MISMATCH",
        ),
        ({"upstream_events": ()}, "INVALID_COMBINE_PLAN"),
        ({"token_owner_ranks": (1,)}, "SHARED_APPLIED_NOT_ONCE"),
        ({"rank": -1}, "MISSING_PROVENANCE"),
        ({"shared_replica_ranks": (0, 0)}, "INVALID_COMBINE_PLAN"),
        ({"input_row_weighted": False}, "INVALID_COMBINE_PLAN"),
        (
            {
                "expected_source_ids": (
                    "wrong",
                    "synthetic-shared",
                    "synthetic-residual",
                )
            },
            "IDENTITY_DRIFT",
        ),
    ],
)
def test_invalid_context(mutation, status, device):
    inputs = make_inputs(device=device)
    with pytest.raises(MergeContractError, match=status):
        check_moe_merge(*inputs, context=replace(make_context(inputs), **mutation))


def test_mhc_post_input_rejected():
    inputs = make_inputs()
    context = make_context(inputs)
    context = replace(
        context,
        sources=(*context.sources[:2], replace(context.sources[2], role="mhc_post")),
    )
    with pytest.raises(MergeContractError, match="INVALID_COMBINE_PLAN"):
        check_moe_merge(*inputs, context=context)


def test_checksum_detects_stale_tensor():
    inputs = make_inputs()
    context = make_context(inputs)
    inputs[1][0, 0] += 1
    with pytest.raises(MergeContractError, match="IDENTITY_DRIFT: shared.tensor_checksum"):
        check_moe_merge(*inputs, context=context)


@pytest.mark.parametrize("role", range(3))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs(role, value, device):
    inputs = make_inputs(device=device)
    inputs[role][0, 0] = value
    with pytest.raises(MergeContractError, match="NON_FINITE"):
        check_moe_merge(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("stage", ["after_shared", "after_residual", "final_bf16"])
def test_intermediate_and_cast_overflow(stage, device):
    maxval = torch.finfo(torch.float32).max
    values = {
        "after_shared": (maxval, maxval, -maxval),
        "after_residual": (maxval, 0.0, maxval),
        "final_bf16": (maxval, 0.0, 0.0),
    }[stage]
    inputs = tuple(torch.tensor([[v]], dtype=torch.float32, device=device) for v in values)
    with pytest.raises(MergeContractError, match=f"NON_FINITE: {stage}"):
        check_moe_merge(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("kind", ["routed_bf16", "shared_fp16", "broadcast", "strided"])
def test_unsupported_payloads(kind, device):
    inputs = list(make_inputs(device=device))
    if kind == "routed_bf16":
        inputs[0] = inputs[0].bfloat16()
    elif kind == "shared_fp16":
        inputs[1] = inputs[1].half()
    elif kind == "broadcast":
        inputs[1] = inputs[1][:, :1].contiguous()
    else:
        inputs[1] = torch.ones(1, 6, device=device)[:, ::2]
    with pytest.raises((TypeError, ValueError)):
        shared_residual_merge_fwd(*inputs)
    if kind == "routed_bf16":
        with pytest.raises(MergeContractError, match="EARLY_OR_MULTIPLE_DOWNCAST"):
            check_moe_merge(*inputs, context=make_context(inputs))


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_ep_replication_owner_local_mock(ep):
    # No collectives: replicas exist on every mock rank; exactly one owner merges
    # each token. Process-group ownership is covered separately on GPUs.
    inputs = tuple(v.repeat(16, 1) for v in make_inputs())
    canonical = check_moe_merge(*inputs, context=make_context(inputs))
    gathered = torch.empty_like(canonical.token_moe_row_bf16)
    for rank in range(ep):
        tokens = tuple(range(rank, 16, ep))
        local = tuple(v[list(tokens)].contiguous() for v in inputs)
        context = make_context(local, rank=rank, ep=ep, tokens=tokens)
        result = check_moe_merge(*local, context=context)
        gathered[list(tokens)] = result.token_moe_row_bf16
        assert result.receipt["shared_applied_count"] == 1
        assert result.receipt["merge_order_hash"] == canonical.receipt["merge_order_hash"]
        if ep > 1:
            with pytest.raises(MergeContractError, match="SHARED_APPLIED_NOT_ONCE"):
                check_moe_merge(*local, context=replace(context, rank=(rank + 1) % ep))
    assert_bytes(gathered, canonical.token_moe_row_bf16)


@pytest.mark.parametrize("shared_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("residual_dtype", [torch.float32, torch.bfloat16])
def test_input_immutability_and_autocast(shared_dtype, residual_dtype, device):
    routed, shared, residual = make_inputs(CASES[1], device=device)
    inputs = (routed, shared.to(shared_dtype), residual.to(residual_dtype))
    for value in inputs:
        value.requires_grad_(True)
    snapshots = tuple(v.detach().clone() for v in inputs)
    op = MoeMergeOp()
    baseline = op(*inputs)
    with torch.autocast(device, dtype=torch.bfloat16):
        after_shared, after_residual, output = op.forward_with_intermediates(*inputs)
        result = op(*inputs)
    golden = torch.tensor(CASES[1]["bf16_words"], dtype=torch.uint16).view(torch.bfloat16)
    assert_bytes(result, golden)
    assert_bytes(result, baseline)
    assert_bytes(result, output)
    assert after_shared.dtype == after_residual.dtype == torch.float32
    assert not any(v.requires_grad for v in (after_shared, after_residual, output, result))
    for value, saved in zip(inputs, snapshots, strict=True):
        assert_bytes(value, saved)


@pytest.mark.parametrize("width", [1, 17, 129, 4096])
def test_batch_chunk_padding_permutation(width, device):
    generator = torch.Generator().manual_seed(71)
    inputs = tuple(torch.randn(7, width, generator=generator).to(device) for _ in range(3))
    expected = shared_residual_merge_fwd(*inputs)
    permutation = [6, 2, 0, 1, 5, 3, 4]
    permuted = tuple(v[permutation] for v in inputs)
    assert_bytes(shared_residual_merge_fwd(*permuted), expected[permutation])
    for start, end in ((0, 1), (1, 4), (4, 7)):
        chunk = tuple(v[start:end] for v in inputs)
        assert_bytes(shared_residual_merge_fwd(*chunk), expected[start:end])
    padded = tuple(torch.cat((v, torch.zeros(3, width, device=device))) for v in inputs)
    assert_bytes(shared_residual_merge_fwd(*padded)[:7], expected)


def test_duplicate_and_empty_token_batch():
    inputs = tuple(v.repeat(2, 1) for v in make_inputs())
    context = make_context(inputs)
    context = replace(context, identity=replace(context.identity, global_token_ids=(1, 1)))
    with pytest.raises(MergeContractError, match="AMBIGUOUS_GLOBAL_TOKEN_MAPPING"):
        check_moe_merge(*inputs, context=context)
    empty = tuple(v[:0] for v in inputs)
    with pytest.raises(MergeContractError, match="AMBIGUOUS_GLOBAL_TOKEN_MAPPING"):
        check_moe_merge(*empty, context=make_context(empty))


@pytest.mark.parametrize("target", ["rollout", "training"])
def test_reference_registry_entry(target):
    from rl_engine.kernels.registry import KernelRegistry
    from rl_engine.kernels.semantic_registry import OperatorRequirements, OperatorSession

    session = OperatorSession(KernelRegistry().semantic)
    resolution = session.resolve(
        semantic_op="shared_residual_merge",
        requested_backend=BACKEND_ID,
        target=target,
        requirements=OperatorRequirements(
            device="cpu",
            dtype="float32",
            alignment_properties={"deterministic": True, "batch_invariant": True},
        ),
    )
    op = session.instantiate(resolution)
    inputs = make_inputs()
    result = op(*inputs)
    assert isinstance(op, torch.nn.Module)
    assert_bytes(result, shared_residual_merge_fwd(*inputs))
    assert resolution.descriptor.determinism_or_alignment_properties["reference_only"] is True
    provenance = session.instance_provenance(resolution, op)
    assert provenance.backend_id == BACKEND_ID
    assert provenance.concrete_implementation.endswith("MoeMergeOp")
    assert provenance.implementation_fingerprint


@pytest.mark.parametrize(
    "requirements,capability",
    [
        ({"dtype": "bfloat16"}, "dtype"),
        ({"topology": {"tensor_parallel_size": 2}}, "topology"),
        ({"topology": {"expert_parallel_size": 2}}, "topology"),
        ({"alignment_properties": {"cross_tp_bitwise": True}}, "alignment_properties"),
        ({"alignment_properties": {"gpu_launch_observable": True}}, "alignment_properties"),
        ({"alignment_properties": {"reference_only": False}}, "alignment_properties"),
        ({"alignment_properties": {"forward_only": False}}, "alignment_properties"),
    ],
)
def test_reference_registry_rejects_unsupported_requirements(requirements, capability):
    from rl_engine.kernels.registry import KernelRegistry
    from rl_engine.kernels.semantic_registry import (
        OperatorRequirements,
        OperatorResolutionError,
        OperatorSession,
    )

    session = OperatorSession(KernelRegistry().semantic)
    requested = {"device": "cpu", "dtype": "float32", **requirements}
    with pytest.raises(OperatorResolutionError) as error:
        session.resolve(
            semantic_op="shared_residual_merge",
            requested_backend=BACKEND_ID,
            target="rollout",
            requirements=OperatorRequirements(**requested),
        )
    assert {
        decision.capability
        for decision in error.value.trace.capability_decisions
        if not decision.passed
    } == {capability}


def test_sealed_reference_evidence(tmp_path, device):
    root = Path(os.environ.get("RL_KERNEL_MOE_MERGE_ARTIFACT_DIR", tmp_path))
    store = ArtifactStore(root / device)
    required = ("merge_receipt.json", "merge_debug.pt")
    for case in CASES:
        inputs = make_inputs(case, device=device)
        context = make_context(inputs, case_id=case["id"])
        result = check_moe_merge(*inputs, context=context, debug=True)
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
    (corrupted / "merge_receipt.json").write_text("{}")
    with pytest.raises(ArtifactError, match="artifact hash does not match"):
        store.validate_completed_attempt(corrupted, required=required)


def _gpu_owner_worker(rank, ep, rendezvous, artifact_root):
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.05, rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=ep,
        timeout=timedelta(seconds=120),
    )
    try:
        # Every process holds a complete shared replica and merges only its own
        # token rows. Uneven counts and reversed rows catch mapping mistakes.
        # Integer-valued inputs have an independent exact oracle.
        grid = torch.arange(17 * 17, dtype=torch.float32).reshape(17, 17)
        routed = (grid.remainder(31) - 15) * 256
        shared_replica = (grid.remainder(7) + 1).to(rank)
        residual = 3 - routed
        tokens = tuple(reversed(range(rank, 17, ep)))
        local = (
            routed[list(tokens)].to(rank),
            shared_replica[list(tokens)].contiguous(),
            residual[list(tokens)].to(rank),
        )
        context = make_context(
            local,
            case_id=f"owner-local-ep-{ep}",
            rank=dist.get_rank(),
            ep=dist.get_world_size(),
            tokens=tokens,
        )
        result = check_moe_merge(*local, context=context, debug=True)
        for field in (
            "shared_applied_count",
            "residual_applied_count",
            "local_downcast_count",
        ):
            assert result.receipt[field] == 1
        if ep > 1:
            with pytest.raises(MergeContractError, match="SHARED_APPLIED_NOT_ONCE"):
                check_moe_merge(*local, context=replace(context, rank=(rank + 1) % ep))
            wrong = (local[0] + ep * local[1] + local[2]).bfloat16()
            assert not torch.equal(
                wrong.view(torch.int16), result.token_moe_row_bf16.view(torch.int16)
            )

        # This gather belongs solely to the test harness. The merge never calls
        # a collective; the caller owns communication and token assignment.
        record = {
            "rank": dist.get_rank(),
            "device": str(result.token_moe_row_bf16.device),
            "tokens": tokens,
            "shared_replica_checksum": tensor_sha256(shared_replica),
            "output_words": result.token_moe_row_bf16.cpu().view(torch.int16).tolist(),
        }
        gathered = [None] * ep
        dist.all_gather_object(gathered, record)
        assert [item["rank"] for item in gathered] == list(range(ep))
        assert sorted(t for item in gathered for t in item["tokens"]) == list(range(17))
        assert len({item["shared_replica_checksum"] for item in gathered}) == 1
        expected = (grid.remainder(7) + 4).bfloat16().view(torch.int16)
        for item in gathered:
            assert_bytes(
                torch.tensor(item["output_words"], dtype=torch.int16),
                expected[list(item["tokens"])],
            )

        store = ArtifactStore(Path(artifact_root) / f"ep-{ep}")
        case_id = f"rank-{rank}"
        store.initialize_experiment(
            case_id,
            experiment={
                "context": json.loads(json.dumps(asdict(context))),
                "implementation": result.receipt["actual_provenance"],
                "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "world_size": dist.get_world_size(),
                "test_collective_backend": dist.get_backend(),
            },
            plan=[],
        )
        attempt = store.create_attempt(case_id, case_id)
        required = ("merge_receipt.json", "merge_debug.pt", "rank_gather.json")
        store.write_json(attempt, required[0], result.receipt)
        store.write_tensor_bundle(attempt, required[1], result.debug_boundaries)
        store.write_json(attempt, required[2], {"ranks": gathered})
        store.complete_attempt(
            attempt,
            required=required,
            summary={
                "schema_version": "cross_config.complete.v1",
                "case_id": case_id,
                "attempt_id": attempt.name,
                "status": "LOCAL_REFERENCE_ONLY",
            },
        )
        store.validate_completed_attempt(attempt, required=required, expected_case_id=case_id)
        loaded = store.load_tensor_bundle(attempt / required[1])["tensors"]
        for boundary in result.receipt["boundaries"]:
            assert tensor_sha256(loaded[boundary["key"]]) == boundary["checksum"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_gpu_ep_shared_once(ep, tmp_path):
    if torch.cuda.device_count() < ep:
        pytest.skip(f"needs {ep} GPUs for real owner ranks")
    if not dist.is_nccl_available():
        pytest.skip("NCCL/RCCL unavailable")
    root = Path(os.environ.get("RL_KERNEL_MOE_MERGE_ARTIFACT_DIR", tmp_path)) / "distributed"
    workers = mp.spawn(
        _gpu_owner_worker,
        args=(ep, (tmp_path / "rendezvous").as_uri(), str(root)),
        nprocs=ep,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(f"EP={ep} workers did not finish within 180 seconds")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
