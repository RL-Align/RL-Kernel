# SPDX-License-Identifier: Apache-2.0
"""T05 metadata gates plus opt-in NVIDIA kernel conformance."""

import copy
import importlib.util
import inspect
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_engine.p6 import reference
from rl_engine.p6.contract import CombinePlan, ContractError, SavedForward
from rl_engine.p6.fixtures import load_golden, make_case
from rl_engine.p6.mocks import ep_return
from rl_engine.p6.oracle import forward as scalar_forward
from rl_engine.p6.provider import ProviderRegistry, RecordedProvider, compare_envelopes
from rl_engine.p6.recordings import boundary_recordings


def binding():
    from rl_engine.p6 import combine

    return combine


def test_lookup_maps_opaque_tokens_to_physical_rows():
    case = make_case("lookup", n=3, h=7, mode="mixed")
    plan = CombinePlan.from_dict(case["plan"])
    lookup = binding().canonical_row_lookup(plan, plan.context)
    assert len(lookup) == 3
    for t, token in enumerate(plan.token_ids):
        for slot in range(6):
            if plan.valid_slots[t][slot]:
                assert plan.inverse_map[lookup[t][slot]] == (token, slot, True)
            else:
                assert lookup[t][slot] == -1


def test_lookup_rejects_wrong_run_before_device_setup():
    plan = CombinePlan.from_dict(make_case("wrong-run")["plan"])
    with pytest.raises(ContractError, match="STALE_RUN_METADATA"):
        binding().canonical_row_lookup(plan, replace(plan.context, run_id="other"))


def test_lookup_rejects_duplicate_slot():
    plan = CombinePlan.from_dict(make_case("duplicate")["plan"])
    bad = replace(plan, inverse_map=(plan.inverse_map[0],) + plan.inverse_map[:-1])
    with pytest.raises(ContractError, match="INVALID_DISCRETE_PLAN"):
        binding().canonical_row_lookup(bad, plan.context)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing", "INVALID_DISCRETE_PLAN"),
        ("extra-slot", "INVALID_DISCRETE_PLAN"),
        ("padding", "INVALID_DISCRETE_PLAN"),
        ("duplicate-token", "INVALID_DISCRETE_PLAN"),
        ("short-mask", "INVALID_DISCRETE_PLAN"),
        ("policy", "SCHEMA_MISMATCH"),
        ("route-ref", "SCHEMA_MISMATCH"),
        ("exchange-ref", "SCHEMA_MISMATCH"),
        ("hidden-zero", "UNSUPPORTED_GEOMETRY"),
    ],
)
def test_bad_metadata_is_rejected_before_backend_or_allocation(mutation, code):
    plan = CombinePlan.from_dict(make_case("invalid-metadata")["plan"])
    changes = {
        "missing": {"inverse_map": plan.inverse_map[1:]},
        "extra-slot": {"inverse_map": ((plan.token_ids[0], 6, True),) + plan.inverse_map},
        "padding": {"inverse_map": plan.inverse_map + ((0, 0, False),)},
        "duplicate-token": {"token_ids": (plan.token_ids[0],) * 3},
        "short-mask": {"valid_slots": ((True,) * 5,) * 3},
        "policy": {"policy_hash": "0" * 64},
        "route-ref": {"route_ref": ("route", "malformed")},
        "exchange-ref": {"exchange_ref": ("exchange", "malformed")},
        "hidden-zero": {"hidden_size": 0},
    }
    with pytest.raises(ContractError, match=code):
        binding().prepare_combine(replace(plan, **changes[mutation]), plan.context, "cpu")


@pytest.mark.parametrize("field", ["run_id", "microbatch_id", "forward_id", "checkpoint_id"])
def test_context_identity_fields_fail_closed(field):
    plan = CombinePlan.from_dict(make_case("stale-identity")["plan"])
    with pytest.raises(ContractError, match="STALE_RUN_METADATA"):
        binding().prepare_combine(plan, replace(plan.context, **{field: "stale"}), "cpu")


def test_cpu_backend_is_explicitly_unsupported():
    plan = CombinePlan.from_dict(make_case("cpu")["plan"])
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY"):
        binding().prepare_combine(plan, plan.context, "cpu")


def test_empty_lookup_does_not_invent_token_rows():
    plan = CombinePlan.from_dict(make_case("empty", n=0)["plan"])
    assert binding().canonical_row_lookup(plan, plan.context) == ()


@pytest.mark.parametrize("name", ["rows", "shared", "residual"])
@pytest.mark.parametrize(
    "target", ["output", "canonical_fp32", "routed_fp32", "slot_partials_fp32"]
)
def test_launch_rejects_overlapping_output_before_kernel(name, target):
    plan = CombinePlan.from_dict(make_case("alias", n=2, h=17, mode="zero-routes")["plan"])
    plan = replace(
        plan,
        valid_slots=((True, False, False, False, False, False),) * 2,
        inverse_map=tuple((t, 0, True) for t in reversed(plan.token_ids)),
    )
    prepared = binding().PreparedCombine(
        plan,
        torch.device("cpu"),
        SavedForward.capture(plan, plan.context),
        torch.empty((2, 6), dtype=torch.int64),
    )
    # Nonzero, partially overlapping storage offsets also need to fail closed.
    storage = torch.zeros((3, 17), dtype=torch.bfloat16)
    output = storage[1:]
    stages = {}
    alias = storage[:2]
    if target != "output":
        stage = torch.zeros((3, 17), dtype=torch.float32)
        alias = stage.view(torch.bfloat16).reshape(-1)[2:36].reshape(2, 17)
        stages[target] = [stage] if target == "slot_partials_fp32" else stage
    buffers = binding().CombineBuffers(
        output,
        prepared.saved_forward,
        stages,
        torch.full((2,), -1, dtype=torch.int32),
        prepared,
        256,
    )
    inputs = {
        key: torch.zeros((2, 17), dtype=torch.bfloat16) for key in ("rows", "shared", "residual")
    }
    inputs[name] = alias
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY.*overlap"):
        prepared.launch(inputs["rows"], inputs["shared"], inputs["residual"], buffers)


@pytest.mark.parametrize("name", ["rows", "shared", "residual"])
def test_host_tensor_gate_rejects_lazy_negative_storage(name):
    case = make_case("negative-view")
    plan = CombinePlan.from_dict(case["plan"])
    values = {
        key: torch.tensor(case[key], dtype=torch.bfloat16) for key in ("rows", "shared", "residual")
    }
    # Exercise the device-independent tensor gate without pretending to run CUDA.
    prepared = binding().PreparedCombine(
        plan,
        torch.device("cpu"),
        SavedForward.capture(plan, plan.context),
        torch.empty((len(plan.token_ids), 6), dtype=torch.int64),
    )
    values[name] = torch._neg_view(values[name])
    assert values[name].is_contiguous() and values[name].is_neg()
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY"):
        prepared._validate_inputs(values["rows"], values["shared"], values["residual"])


def gpu():
    if os.environ.get("P6_RUN_T05_GPU") != "1":
        pytest.skip("set P6_RUN_T05_GPU=1 to run actual T05 CUDA kernels")
    assert torch.cuda.is_available(), "P6_RUN_T05_GPU=1 requires available CUDA"
    assert torch.version.hip is None, "this T05 backend requires NVIDIA CUDA"
    assert torch.cuda.get_device_capability()[0] >= 8, "T05 requires sm80+"
    assert importlib.util.find_spec("triton"), "P6_RUN_T05_GPU=1 requires Triton"
    return torch.device("cuda", torch.cuda.current_device())


def tensors(case, device):
    plan = CombinePlan.from_dict(case["plan"])
    h, n, p = plan.hidden_size, len(plan.token_ids), len(plan.inverse_map)
    return (
        plan,
        torch.tensor(case["rows"], dtype=torch.bfloat16, device=device).reshape(p, h),
        torch.tensor(case["shared"], dtype=torch.bfloat16, device=device).reshape(n, h),
        torch.tensor(case["residual"], dtype=torch.bfloat16, device=device).reshape(n, h),
    )


GOLDEN = Path(__file__).parent / "data/golden.v1.json"


def candidate_provider(device="cpu"):
    from rl_engine.p6.combine_provider import TritonCombineProvider

    return TritonCombineProvider(device)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("operator", "UNSUPPORTED_CAPABILITY"),
        ("case", "UNSUPPORTED_CAPABILITY"),
        ("input", "IDENTITY_DRIFT"),
        ("context", "STALE_RUN_METADATA"),
        ("extra-field", "SCHEMA_MISMATCH"),
        ("cpu", "UNSUPPORTED_CAPABILITY"),
    ],
)
def test_candidate_provider_rejects_before_execution(mutation, code):
    record = next(r for r in boundary_recordings() if r["operator"] == "fused_moe_combine_fwd")
    inputs = copy.deepcopy(record["inputs"])
    operator, case_id = record["operator"], record["case_id"]
    if mutation == "operator":
        operator = "canonical_unpermute_fwd"
    elif mutation == "case":
        case_id = "unknown-case"
    elif mutation == "input":
        inputs["rows"][0][0] += 1
    elif mutation == "context":
        inputs["context"]["forward_id"] = "stale"
    elif mutation == "extra-field":
        inputs["fallback"] = True
    with pytest.raises(ContractError, match=code):
        candidate_provider().run(operator, case_id, inputs)


def test_candidate_provider_description_is_explicitly_synthetic_forward_only():
    description = candidate_provider().describe()
    assert description["kind"] == "live"
    assert description["production_certified"] is False
    assert len(description["capabilities"]) == 9
    assert {op for op, _ in description["capabilities"]} == {"fused_moe_combine_fwd"}


@pytest.mark.parametrize(
    "record", load_golden(GOLDEN)["payload"]["cases"], ids=lambda r: r["input"]["name"]
)
def test_gpu_frozen_bytes_debug_on_and_off(record):
    device = gpu()
    plan, rows, shared, residual = tensors(record["input"], device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    expected = record["expected"]["forward"]
    canonical = reference.canonical_unpermute_fwd(plan, record["input"]["rows"], plan.context)
    routed = reference.fixed_order_combine_fwd(plan, canonical["canonical_rows"], plan.context)
    merged = reference.shared_residual_merge_fwd(
        plan, routed["routed"], record["input"]["shared"], record["input"]["residual"], plan.context
    )
    serial = {
        "canonical_fp32": canonical["canonical_fp32"],
        "slot_partials_fp32": routed["slot_partials_fp32"],
        "routed_fp32": routed["routed_fp32"],
        **{k: merged[k] for k in ("after_shared_fp32", "precast_fp32", "output_bf16")},
    }
    assert serial == expected
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    fingerprint, order = plan.fingerprint, plan.order_hash
    for block in (128, 256, 512):
        plain = prepared.forward(rows, shared, residual, block_size=block)
        debug = prepared.forward(rows, shared, residual, debug=True, block_size=block)
        assert plain.stages == {}
        assert debug.stage_bytes() == expected
        assert binding().tensor_bytes(plain.output) == expected["output_bf16"]
        assert binding().tensor_bytes(debug.output) == expected["output_bf16"]
        assert debug.saved_forward.restore(plan.context, plan.fingerprint) == plan
        assert debug.trace() == record["expected"]["trace"]
    assert plan.fingerprint == fingerprint and plan.order_hash == order
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))


@pytest.mark.parametrize(
    "record",
    [r for r in boundary_recordings() if r["operator"] == "fused_moe_combine_fwd"],
    ids=lambda r: r["case_id"],
)
def test_gpu_candidate_registry_matches_recorded_bytes(record):
    device = gpu()
    recordings = boundary_recordings()
    registry = ProviderRegistry()
    registry.register("recorded", RecordedProvider(recordings, "a" * 64))
    registry.register("live", candidate_provider(device))
    args = record["operator"], record["case_id"], record["inputs"]
    actual = registry.run("live", *args)
    expected = registry.run("recorded", *args)
    assert compare_envelopes(expected, actual) == {
        "status": "REFERENCE_BYTES_PASS",
        "production_certified": False,
    }
    assert actual["provenance"]["readback_kind"] == "actual"
    assert actual["provenance"]["backend"] == "triton-cuda"


@pytest.mark.parametrize("ep", [1, 2, 4, 8])
def test_gpu_same_exchange_p4_mock_chunk_topology_arrival_and_overlap(ep):
    device = gpu()
    case = make_case("p4-schedule", n=3, h=513, mode="mixed")
    plan, rows, shared, residual = tensors(case, device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    expected = scalar_forward(plan, case["rows"], case["shared"], case["residual"], plan.context)
    for rotation, reverse, chunk, overlap in ((0, False, 1, False), (ep - 1, True, 3, True)):
        returned = ep_return(
            plan, case["rows"], plan.context, ep=ep, placement=rotation, reverse=reverse
        )
        assert returned["plan_fingerprint"] == plan.fingerprint
        assert sum(returned["counts"]) == len(plan.inverse_map)
        if ep == 8:
            assert 0 in returned["counts"]  # zero-count peers and uneven counts
        assert returned["rows"] == case["rows"]
        arrived = torch.full_like(rows, float("nan"))
        batches = [
            returned["arrival"][i : i + chunk] for i in range(0, len(plan.inverse_map), chunk)
        ]
        # Reverse completion order models delayed chunks, without a wall-clock sleep.
        if overlap:
            batches.reverse()
        streams = [torch.cuda.Stream(device=device) for _ in range(2)] if overlap else []
        current = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(current)
        for i, indices in enumerate(batches):
            index = torch.tensor(indices, dtype=torch.int64, device=device)
            stream = streams[i % len(streams)] if overlap else current
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                arrived.index_copy_(0, index, rows.index_select(0, index))
                # Retain indices until their asynchronous reads have finished.
                index.record_stream(stream)
        for stream in streams:
            current.wait_stream(stream)
        result = prepared.forward(arrived, shared, residual, debug=True)
        assert result.stage_bytes() == expected["stages"]
        assert result.saved_forward.fingerprint == plan.fingerprint
        assert result.trace()["order_hash"] == plan.order_hash


def test_gpu_default_buffers_and_launch_do_not_materialize_canonical(monkeypatch):
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("allocation", n=3, h=4097), device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    allocated = []
    empty = torch.empty

    def observed_empty(shape, *args, **kwargs):
        allocated.append(tuple(shape))
        return empty(shape, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", observed_empty)
        buffers = prepared.allocate(debug=False)
    assert allocated == [(3, 4097)]
    assert buffers.stages == {}
    prepared.launch(rows, shared, residual, buffers)
    buffers.check_status()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        prepared.launch(rows, shared, residual, buffers)
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) == before
    buffers.check_status()


def test_gpu_status_rejects_active_nan_and_subnormal_but_ignores_padding():
    device = gpu()
    case = make_case("status", n=2, h=17, mode="mixed")
    plan, rows, shared, residual = tensors(case, device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    clean = prepared.forward(rows, shared, residual).output.clone()
    for i, (_, _, valid) in enumerate(plan.inverse_map):
        if not valid:
            rows[i].fill_(float("nan"))
    assert binding().tensor_bytes(prepared.forward(rows, shared, residual).output) == (
        binding().tensor_bytes(clean)
    )
    index = next(i for i, r in enumerate(plan.inverse_map) if r[2])
    for value, code in [(float("nan"), "NON_FINITE"), (2**-130, "UNSUPPORTED_CAPABILITY")]:
        poisoned = rows.clone()
        poisoned[index, 0] = value
        with pytest.raises(ContractError, match=code):
            prepared.forward(poisoned, shared, residual)


@pytest.mark.parametrize("name", ["rows", "shared", "residual"])
def test_gpu_rejects_layout_dtype_and_autograd(name):
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("inputs"), device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    inputs = {"rows": rows, "shared": shared, "residual": residual}
    value = inputs[name]
    for bad, code in [
        (value.float(), "DTYPE_MISMATCH"),
        (value.T.contiguous().T, "UNSUPPORTED_CAPABILITY"),
        (value.clone().requires_grad_(), "UNSUPPORTED_CAPABILITY"),
    ]:
        changed = {**inputs, name: bad}
        with pytest.raises(ContractError, match=code):
            prepared.forward(changed["rows"], changed["shared"], changed["residual"])


@pytest.mark.parametrize("name", ["shared", "residual"])
@pytest.mark.parametrize(
    "value,code",
    [
        (float("nan"), "NON_FINITE"),
        (float("inf"), "NON_FINITE"),
        (2**-130, "UNSUPPORTED_CAPABILITY"),
    ],
)
def test_gpu_rejects_invalid_branch_values(name, value, code):
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("branch-status"), device)
    inputs = {"rows": rows, "shared": shared, "residual": residual}
    inputs[name][0, 0] = value
    with pytest.raises(ContractError, match=code):
        binding().prepare_combine(plan, plan.context, device).forward(rows, shared, residual)


@pytest.mark.parametrize("name", ["rows", "shared", "residual"])
@pytest.mark.parametrize("debug", [False, True])
def test_gpu_rejects_input_aliasing_output(name, debug):
    device = gpu()
    plan, _, shared, residual = tensors(
        make_case("gpu-alias", n=2, h=17, mode="zero-routes"), device
    )
    plan = replace(
        plan,
        valid_slots=((True, False, False, False, False, False),) * 2,
        inverse_map=tuple((t, 0, True) for t in reversed(plan.token_ids)),
    )
    rows = torch.zeros((2, 17), dtype=torch.bfloat16, device=device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    buffers = prepared.allocate(debug=debug)
    alias = (
        buffers.stages["routed_fp32"].view(torch.bfloat16).reshape(-1)[:34].reshape(2, 17)
        if debug
        else buffers.output
    )
    if name == "rows":
        rows = alias
    elif name == "shared":
        shared = alias
    else:
        residual = alias
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY.*overlap"):
        prepared.launch(rows, shared, residual, buffers)


def test_gpu_candidate_sealed_artifact_roundtrip(tmp_path):
    from rl_engine.p6.artifact import build_payload, publish, read_artifact, verify
    from rl_engine.p6.recordings import golden_records

    device = gpu()
    records = boundary_recordings()
    reference_provider = RecordedProvider(records, "a" * 64)
    candidate = candidate_provider(device)
    checks = []
    for record in records:
        if record["operator"] != "fused_moe_combine_fwd":
            continue
        args = record["operator"], record["case_id"], record["inputs"]
        expected, actual = reference_provider.run(*args), candidate.run(*args)
        checks.append({"comparison": compare_envelopes(expected, actual), "candidate": actual})
    # Reuse the T01 seal/schema, not a competing T05 artifact format. Top-level
    # provenance identifies CPU reference recordings; actual GPU proof is in checks.
    payload = build_payload(
        golden_records(),
        {"provider": "stdlib-scalar-oracle"},
        checks,
        {"scope": "T05 synthetic CUDA candidate", "device": str(device)},
    )
    directory = Path(os.environ.get("P6_T05_ARTIFACT_DIR", str(tmp_path / "t05-attempt")))
    publish(directory, payload)
    assert verify(directory)["gpu_reexecuted"] is False
    reloaded, _ = read_artifact(directory)
    assert len(reloaded["checks"]) == 9
    by_case = {r["case_id"]: r for r in records if r["operator"] == "fused_moe_combine_fwd"}
    for check in reloaded["checks"]:
        actual = check["candidate"]
        r = by_case[actual["case_id"]]
        expected = reference_provider.run(r["operator"], r["case_id"], r["inputs"])
        assert compare_envelopes(expected, actual) == check["comparison"]


def test_gpu_graph_replay_changes_output_and_rechecks_status():
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("graph", h=33), device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    buffers = prepared.allocate(debug=True)
    prepared.launch(rows, shared, residual, buffers)
    buffers.check_status()
    graph = torch.cuda.CUDAGraph()
    # Warm-up compilation and allocator on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.launch(rows, shared, residual, buffers)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        prepared.launch(rows, shared, residual, buffers)
    graph.replay()
    buffers.check_status()
    first = binding().tensor_bytes(buffers.output)
    rows.neg_()
    wanted = prepared.forward(rows, shared, residual, debug=True).stage_bytes()
    graph.replay()
    buffers.check_status()
    assert buffers.stage_bytes() == wanted
    assert binding().tensor_bytes(buffers.output) != first
    rows[0, 0] = float("inf")
    graph.replay()
    with pytest.raises(ContractError, match="NON_FINITE"):
        buffers.check_status()
    rows[0, 0] = 1
    graph.replay()
    buffers.check_status()


def test_gpu_permutation_and_poison_padding_preserve_logical_bytes():
    device = gpu()
    record = next(
        r
        for r in load_golden(GOLDEN)["payload"]["cases"]
        if r["input"]["name"] == "mixed-invalid-overflow"
    )
    plan, rows, shared, residual = tensors(record["input"], device)
    reverse = list(reversed(range(len(plan.inverse_map))))
    # Same upstream identity, different physical layout; canonical order is unchanged.
    moved = replace(plan, inverse_map=tuple(plan.inverse_map[i] for i in reverse))
    padded = replace(moved, inverse_map=moved.inverse_map + ((-1, -1, False),) * 3)
    rows = torch.cat(
        (
            rows[reverse],
            torch.full((3, plan.hidden_size), float("nan"), dtype=torch.bfloat16, device=device),
        )
    )
    for i, (_, _, valid) in enumerate(padded.inverse_map):
        if not valid:
            rows[i].fill_(float("nan"))
    result = (
        binding()
        .prepare_combine(padded, padded.context, device)
        .forward(rows, shared, residual, debug=True)
    )
    assert padded.order_hash == plan.order_hash
    assert result.stage_bytes() == record["expected"]["forward"]


@pytest.mark.parametrize("hidden", [65, 4096])
def test_gpu_batch_partition_preserves_each_token(hidden):
    device = gpu()
    case = make_case("batch-partition", n=5, h=hidden, mode="mixed")
    plan, rows, shared, residual = tensors(case, device)
    full = (
        binding().prepare_combine(plan, plan.context, device).forward(rows, shared, residual).output
    )
    expected = scalar_forward(plan, case["rows"], case["shared"], case["residual"], plan.context)
    assert binding().tensor_bytes(full) == expected["stages"]["output_bf16"]
    for begin, end in [(0, 1), (1, 3), (3, 5)]:
        tokens = plan.token_ids[begin:end]
        selected = [
            i for i, (token, _, valid) in enumerate(plan.inverse_map) if valid and token in tokens
        ]
        sub = replace(
            plan,
            token_ids=tokens,
            valid_slots=plan.valid_slots[begin:end],
            inverse_map=tuple(plan.inverse_map[i] for i in selected),
        )
        result = (
            binding()
            .prepare_combine(sub, sub.context, device)
            .forward(rows[selected].contiguous(), shared[begin:end], residual[begin:end])
        )
        assert binding().tensor_bytes(result.output) == binding().tensor_bytes(full[begin:end])


def fill_final_cast_overflow(plan, rows, shared, residual):
    rows.zero_()
    first = next(i for i, r in enumerate(plan.inverse_map) if r[1] == 0)
    rows[first].fill_(torch.finfo(torch.bfloat16).max)
    shared.fill_(float(2**119))
    residual.zero_()


def test_final_cast_overflow_fixture_is_finite_fp32_and_infinite_bf16():
    plan, rows, shared, residual = tensors(make_case("overflow", n=1, h=3), torch.device("cpu"))
    fill_final_cast_overflow(plan, rows, shared, residual)
    precast = rows.float().sum(dim=0, keepdim=True) + shared.float() + residual.float()
    # Exact finite FP32 halfway between largest finite BF16 and +infinity.
    assert precast.view(torch.int32).tolist() == [[0x7F7F8000] * 3]
    assert precast.to(torch.bfloat16).view(torch.int16).tolist() == [[0x7F80] * 3]


def test_gpu_rejects_fp32_accumulator_and_final_bf16_overflow():
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("overflow", n=1, h=3), device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    largest = torch.finfo(torch.bfloat16).max
    with pytest.raises(ContractError, match="NON_FINITE"):
        prepared.forward(torch.full_like(rows, largest), shared, residual)
    fill_final_cast_overflow(plan, rows, shared, residual)
    # FP32 remains finite, but rounding to BF16 overflows.
    with pytest.raises(ContractError, match="NON_FINITE"):
        prepared.forward(rows, shared, residual)


def test_gpu_rejects_foreign_buffers_shape_and_unlaunched_readback():
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("buffer-owner"), device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    other = binding().prepare_combine(plan, plan.context, device)
    buffers = prepared.allocate()
    with pytest.raises(ContractError, match="INCOMPLETE_ARTIFACT"):
        buffers.check_status()
    with pytest.raises(ContractError, match="IDENTITY_DRIFT"):
        other.launch(rows, shared, residual, buffers)
    with pytest.raises(ContractError, match="UNSUPPORTED_GEOMETRY"):
        prepared.forward(rows, shared[:-1], residual)


def test_gpu_rejects_subnormal_intermediate_with_normal_inputs():
    device = gpu()
    plan, rows, shared, residual = tensors(make_case("tiny-cancellation", n=1, h=1), device)
    rows.zero_()
    shared.zero_()
    residual.zero_()
    for i, (_, slot, _) in enumerate(plan.inverse_map):
        if slot == 0:
            rows[i, 0] = 2**-126
        elif slot == 1:
            rows[i, 0] = -(2**-126 + 2**-133)
    # Two normal BF16 inputs yield a subnormal FP32 difference after slot 1.
    with pytest.raises(ContractError, match="UNSUPPORTED_CAPABILITY"):
        binding().prepare_combine(plan, plan.context, device).forward(rows, shared, residual)


def test_gpu_invalid_slots_preserve_first_valid_negative_zero():
    device = gpu()
    case = make_case("sparse-zero", n=1, h=2)
    plan = CombinePlan.from_dict(case["plan"])
    plan = replace(
        plan,
        valid_slots=((False, False, True, False, False, False),),
        inverse_map=((plan.token_ids[0], 2, True), (-1, -1, False)),
    )
    rows = torch.tensor(
        [[-0.0, 0.0], [float("nan"), float("nan")]], dtype=torch.bfloat16, device=device
    )
    branch = torch.tensor([[-0.0, 0.0]], dtype=torch.bfloat16, device=device)
    prepared = binding().prepare_combine(plan, plan.context, device)
    result = prepared.forward(rows, branch, branch, debug=True)
    stages = result.stage_bytes()
    assert stages["output_bf16"] == "00800000"
    assert stages["routed_fp32"] == "0000008000000000"
    assert stages["slot_partials_fp32"] == [
        "0000000000000000",
        "0000000000000000",
        "0000008000000000",
        "0000008000000000",
        "0000008000000000",
        "0000008000000000",
    ]
    assert binding().tensor_bytes(prepared.forward(rows, branch, branch).output) == "00800000"


def test_lookup_rejects_geometry_outside_kernel_index_range():
    plan = CombinePlan.from_dict(make_case("huge")["plan"])
    with pytest.raises(ContractError, match="UNSUPPORTED_GEOMETRY"):
        binding().canonical_row_lookup(replace(plan, hidden_size=2**31), plan.context)


def test_allocation_rejects_cuda_grid_y_overflow_before_allocation():
    plan = CombinePlan.from_dict(make_case("grid-limit", n=1, mode="zero-routes")["plan"])
    plan = replace(plan, hidden_size=65535 * 256 + 1)
    prepared = binding().PreparedCombine(
        plan,
        torch.device("cpu"),
        SavedForward.capture(plan, plan.context),
        torch.empty((1, 6), dtype=torch.int64),
    )
    with pytest.raises(ContractError, match="UNSUPPORTED_GEOMETRY"):
        prepared.allocate(block_size=256)


@pytest.mark.parametrize("architecture", [80, 90])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("block", [128, 256, 512])
def test_offline_cuda_compile(architecture, debug, block):
    if os.environ.get("P6_COMPILE_T05") != "1":
        pytest.skip("set P6_COMPILE_T05=1 for offline CUDA compilation")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    from rl_engine.kernels.ops.triton.moe.combine import combine_kernel

    def pointer_type(name):
        if name == "Lookup":
            return "*i64"
        if name == "Status":
            return "*i32"
        if name in ("Rows", "Shared", "Residual", "Output"):
            return "*bf16"
        return "*fp32"

    signature = {name: pointer_type(name) for name in combine_kernel.arg_names[:11]}
    constant_key = (
        "constexprs" if "constexprs" in inspect.signature(ASTSource).parameters else "constants"
    )
    kernel = compile(
        ASTSource(
            combine_kernel,
            signature=signature,
            **{constant_key: {"T": 3, "H": 4097, "BLOCK": block, "DEBUG": debug}},
        ),
        target=GPUTarget("cuda", architecture, 32),
        options={"num_warps": 4, "enable_fp_fusion": False},
    )
    assert kernel.asm["cubin"]
