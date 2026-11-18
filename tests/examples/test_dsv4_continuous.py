# SPDX-License-Identifier: Apache-2.0
"""CPU guards for consecutive-run evidence; these are not GPU/model acceptance."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from examples.operators.dsv4_4l.device_allocation import check_selected_identity, resolve_allocation
from examples.operators.dsv4_4l.optimizer_evidence import (
    optimizer_snapshot,
    verify_optimizer_transition,
)
from examples.operators.dsv4_4l.verify_continuous import MATRIX, check_matrix


def test_actual_cpu_adam_moments_and_counters_continue_across_three_updates(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    engine = SimpleNamespace(optimizer=torch.optim.Adam([p], lr=0.1))
    previous = optimizer_snapshot(engine)
    for version in range(1, 4):
        before = optimizer_snapshot(engine)
        engine.optimizer.zero_grad()
        p.square().sum().backward()
        engine.optimizer.step()
        after = optimizer_snapshot(engine)
        assert verify_optimizer_transition(before, after, version=version, previous=previous)
        previous = after
    assert set(after["step_counters"].values()) == {3}


@pytest.mark.parametrize("reset", ["replace", "clear_state", "replay_old", "no_step"])
def test_optimizer_reset_or_replayed_step_fails(reset, monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    engine = SimpleNamespace(optimizer=torch.optim.Adam([p], lr=0.1))
    p.sum().backward()
    engine.optimizer.step()
    previous = optimizer_snapshot(engine)
    if reset == "replace":
        engine.optimizer = torch.optim.Adam([p], lr=0.1)
    elif reset == "clear_state":
        engine.optimizer.state.clear()
    before = optimizer_snapshot(engine)
    if reset != "no_step":
        p.sum().backward()
        engine.optimizer.step()
    after = optimizer_snapshot(engine)
    if reset == "replay_old":
        after = copy.deepcopy(previous)
    with pytest.raises(ValueError):
        verify_optimizer_transition(before, after, version=2, previous=previous)


def matrix_rows(version=2, prefix=""):
    suffix = "-new" if prefix else ""
    cases = [
        dict(id=n + suffix, prompt_token_ids=[0] * length, response_token_ids=[9] * count)
        for n, length, count in [("A", 4, 2), ("B", 2, 3), ("C", 2, 1)]
    ]
    counts = {x["id"]: len(x["response_token_ids"]) for x in cases}
    rows = []
    for label, ids in MATRIX:
        name = f"v{version}-{prefix}{label}-matrix"
        for position, key in enumerate(ids):
            key += suffix
            common = dict(
                case_id=key,
                weight_version=version,
                batch_size=len(ids),
                batch_position=position,
                expected_tokens=counts[key],
                compared_tokens=counts[key],
                mismatch_count=0,
                dtype="torch.float32",
                all_finite=True,
            )
            rows.append(dict(common, comparison_kind="actor_vs_rollout", configuration=name))
            if len(ids) > 1 or label.endswith("repeat"):
                for kind in ["actor_repeat_or_batch_position", "rollout_repeat_or_batch_position"]:
                    rows.append(
                        dict(
                            common,
                            comparison_kind=kind,
                            configuration=name + "-" + key + "-" + kind,
                        )
                    )
    return cases, rows


@pytest.mark.parametrize("prefix", ["", "generated-"])
def test_complete_fixed_and_generated_matrix_and_per_engine_invariance(prefix):
    cases, rows = matrix_rows(prefix=prefix)
    check_matrix(rows, 2, cases, prefix=prefix)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "version", "position", "invariance", "boundary"]
)
def test_missing_or_stale_matrix_cells_cannot_pass(change):
    cases, rows = matrix_rows()
    if change == "missing":
        rows.pop(0)
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "version":
        rows[0]["weight_version"] = 1
    elif change == "position":
        rows[0]["batch_position"] = 1
    elif change == "boundary":
        rows[0]["expected_tokens"] = 1
    else:
        rows = [r for r in rows if r["comparison_kind"] != "actor_repeat_or_batch_position"]
    with pytest.raises(ValueError):
        check_matrix(rows, 2, cases)


def test_render_minors_are_resolved_through_selected_physical_pci_not_arithmetic(tmp_path):
    for minor, bus in [
        (161, "0000:a3:00.0"),
        (169, "0000:ab:00.0"),
        (177, "0000:b3:00.0"),
        (185, "0000:bb:00.0"),
    ]:
        device = tmp_path / bus
        device.mkdir()
        node = tmp_path / f"renderD{minor}"
        node.mkdir()
        (node / "device").symlink_to(device)
    buses = dict(
        zip(
            range(4, 8),
            ["0000:a3:00.0", "0000:ab:00.0", "0000:b3:00.0", "0000:bb:00.0"],
            strict=False,
        )
    )
    records = resolve_allocation([4, 5], [6, 7], drm=tmp_path, buses=buses)
    assert [x["render_node"] for x in records] == [
        f"/dev/dri/renderD{i}" for i in [161, 169, 177, 185]
    ]
    assert check_selected_identity(dict(pci_bus_id="0000:b3:00.0"), 6, records)["physical_gpu"] == 6
    with pytest.raises(RuntimeError):
        check_selected_identity(dict(pci_bus_id="0000:83:00.0"), 6, records)


def test_lossless_trace_store_keeps_bits_shape_and_original_view_stride(monkeypatch, tmp_path):
    from examples.operators.dsv4_4l import trace_store

    original = trace_store.Path
    monkeypatch.setenv("DSV4_TRACE_FORMAT", "cas-v1")
    monkeypatch.setattr(
        trace_store,
        "Path",
        lambda p: (
            tmp_path
            if str(p) == "/outputs"
            else (tmp_path / "trace-tensors" if str(p) == "/outputs/trace-tensors" else original(p))
        ),
    )
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, ::2]
    filename = trace_store.save_trace({"input": x}, tmp_path / "trace.pt")
    record = json.loads((tmp_path / filename).read_text())["input"]
    assert record["stride"] == list(x.stride())
    restored = trace_store.load_tensor(tmp_path, record)
    assert torch.equal(restored.contiguous().view(torch.int32), x.contiguous().view(torch.int32))
    trace_store.save_trace({"input": x}, tmp_path / "trace2.pt")
    assert len(list((tmp_path / "trace-tensors").glob("*.pt.zst"))) == 1


@pytest.mark.parametrize("ids", [[1, 2, 3, 4], [1, 3], [2, 4], [4, 2, 4]])
def test_kfd_query_filter_preserves_real_allowed_aperture_fields_and_zeros_excluded(
    monkeypatch, tmp_path, ids
):
    import ctypes
    import shutil
    import subprocess
    from pathlib import Path

    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("CPU C compiler is required for the KFD aperture visibility regression")
    source = Path(__file__).resolve().parents[2] / "examples/operators/dsv4_4l/isolated_kfd.c"
    library = tmp_path / "isolated-kfd-test.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DRLK_TEST",
            str(source),
            "-o",
            str(library),
            "-ldl",
        ],
        check=True,
    )
    monkeypatch.setenv("RLK_ALLOWED_KFD_GPU_IDS", "2,4")
    module = ctypes.CDLL(str(library))

    class Aperture(ctypes.Structure):
        _fields_ = [
            ("ranges", ctypes.c_uint64 * 6),
            ("gpu_id", ctypes.c_uint32),
            ("pad", ctypes.c_uint32),
        ]

    entries = (Aperture * len(ids))()
    for i, key in enumerate(ids):
        entries[i].gpu_id = key
        entries[i].ranges[:] = [100 * i + j for j in range(6)]
    expected = [bytes(entries[i]) for i, key in enumerate(ids) if key in (2, 4)]
    function = module.rlk_test_filter
    function.argtypes = [ctypes.POINTER(Aperture), ctypes.c_uint32]
    function.restype = ctypes.c_uint32
    kept = function(entries, len(ids))
    assert kept == len(expected)
    assert [bytes(entries[i]) for i in range(kept)] == expected
    assert all(bytes(entries[i]) == bytes(ctypes.sizeof(Aperture)) for i in range(kept, len(ids)))


def complete_evidence(root):
    """Synthetic CPU guard fixtures only; never presented as actual worker validation."""
    from examples.operators.dsv4_4l.distributed_evidence import tensor_record

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    topology = [
        dict(
            rank=i,
            pid=100 + i,
            pp_rank=i,
            dp_rank=0,
            logical_gpu=i,
            pp_ranks=[0, 1],
            dp_ranks=[i],
            original_layer_ids=[0, 1, 2] if i == 0 else [3],
            device_identity=dict(pci_bus_id=str(i), architecture="gfx942"),
        )
        for i in range(2)
    ]
    save("actor-topology.json", topology)
    start = [
        dict(
            pipeline_stage=i,
            pid=200 + i,
            weight_version=0,
            original_layer_ids=[0, 1, 2] if i == 0 else [3],
            device_identity=dict(pci_bus_id=str(i + 2), architecture="gfx942"),
        )
        for i in range(2)
    ]
    save("rollout-start-ack.json", start)
    initial = dict(
        pid=1,
        engine_id=2,
        optimizer_id=3,
        optimizer_objects=[4],
        state_digest="initial",
        step_counters={},
        state_tensors={},
        state_scalars={},
    )
    for rank in range(2):
        save(f"actor-rank-{rank}/optimizer-initial.json", initial)
    fixed, _ = matrix_rows(0)
    generated, _ = matrix_rows(0, "generated-")
    save("cases-v0.json", fixed)
    result = {
        key: True
        for key in [
            "before_passed",
            "after_passed",
            "pure_rlk_passed",
            "update_executed",
            "same_rollout_process",
            "expanded_coverage_passed",
        ]
    }
    result.update(
        continuous_steps=3,
        completed_version=3,
        cross_replica_dp_reduction_executed=False,
        actor_parallelism=dict(TP=1, PP=2, EP=1, CP=1, DP=1),
        rollout_parallelism=dict(TP=1, PP=2, EP=1, CP=1, DP=1),
        rows=[],
    )
    old = initial
    for version in range(4):
        save(f"cases-v{version}-fixed.json", fixed)
        save(f"cases-v{version}-generated.json", generated)
        for prefix, group in [("", fixed), ("generated-", generated)]:
            _, rows = matrix_rows(version, prefix)
            result["rows"].extend(rows)
            lookup = {x["id"]: x for x in group}
            for label, ids in MATRIX:
                name = f"v{version}-{prefix}{label}-matrix"
                ids = [x + ("-new" if prefix else "") for x in ids]
                tokens = [
                    lookup[x]["prompt_token_ids"] + lookup[x]["response_token_ids"] for x in ids
                ]
                save(f"actor-{name}.json", dict(pid=100, weight_version=version))
                save(
                    f"rollout-batches-{name}.json",
                    [
                        dict(
                            case_id=name,
                            weight_version=version,
                            batches=[dict(num_requests=len(ids), input_token_rows=tokens)],
                        ),
                        dict(case_id=name, weight_version=version, batches=[]),
                    ],
                )
        for new in generated:
            result["rows"].append(
                dict(
                    comparison_kind="actor_vs_rollout",
                    case_id=new["id"],
                    configuration=f"generate-{new['id']}-v{version}",
                    weight_version=version,
                    expected_tokens=len(new["response_token_ids"]),
                    compared_tokens=len(new["response_token_ids"]),
                    mismatch_count=0,
                    dtype="torch.float32",
                    all_finite=True,
                )
            )
        acks = [dict(x, weight_version=version) for x in start]
        save(f"rollout-v{version}-ack.json", acks)
        if version:
            directory = f"steps/v{version}"
            new = dict(
                initial,
                state_digest=f"moment-{version}",
                step_counters={"state/step": version},
                state_tensors={"moment": dict(sha256=str(version), finite=True)},
            )
            records, manifest = [], {}
            for rank, names in [
                (0, ["embed.weight"]),
                (1, ["norm.weight", "layers.3.ffn.gate.bias"]),
            ]:
                state = {
                    n: dict(shape=[1], dtype="torch.float32", sha256=f"{n}-{version}")
                    for n in names
                }
                manifest.update(state)
                gradient = dict(
                    shape=[1], dtype="torch.float32", sha256="grad", finite=True, nonzero=True
                )
                save(
                    f"rank-{rank}-v{version}-grad.json",
                    dict(
                        actual_trainable_parameters=["p"],
                        local_gradients={"p": gradient},
                        reduced_gradients={"p": gradient},
                    ),
                )
                save(
                    f"rank-{rank}-v{version}-change.json",
                    dict(
                        changed_parameters=["p"],
                        synchronization_inventory=state,
                        router_bias_exports=state if rank else {},
                    ),
                )
                value = tensor_record(torch.ones(3, 1, 16384, dtype=torch.bfloat16))
                events = [
                    dict(direction=d, microbatch=0, **value)
                    for d in (
                        ["forward_send", "backward_recv"]
                        if rank == 0
                        else ["forward_recv", "backward_send"]
                    )
                ]
                records.append(
                    dict(
                        rank=rank,
                        pid=100 + rank,
                        pp_rank=rank,
                        dp_rank=0,
                        weight_version=version,
                        previous_weight_version=version - 1,
                        before="old",
                        after="new",
                        dp_gradient_reduction_verified=False,
                        dp_replicas=1,
                        different_local_gradients=False,
                        cross_replica_reduction_executed=False,
                        pipeline_boundaries=events,
                        gradient_witness=dict(
                            backward_executed=True, gradient_finite=True, gradient_nonzero=True
                        ),
                        changed_parameters=["p"],
                        local_sync_state=state,
                        optimizer_evidence=dict(before=old, after=new),
                        gradient_evidence=f"rank-{rank}-v{version}-grad.json",
                        state_change_evidence=f"rank-{rank}-v{version}-change.json",
                    )
                )
                acks[rank].update(
                    prior_weight_version=version - 1,
                    input_update_path=f"/outputs/{directory}/actor-update.pt",
                    parameter_load_completed=True,
                    weight_changed=True,
                    cache_generation_before=version - 1,
                    cache_generation_after=version,
                    prefix_caching_enabled=False,
                    verified_original_tensors={
                        n: dict(sha256=v["sha256"]) for n, v in state.items()
                    },
                )
            save(directory + "/actor-distributed-step.json", records)
            save(directory + "/actor-update-manifest.json", manifest)
            save(directory + "/rollout-sync-ack.json", acks)
            save(
                directory + "/actor-step-status.json",
                dict(
                    weight_version=version,
                    optimizer_returned=True,
                    parameter_changed=True,
                    all_stage_updates_verified=True,
                ),
            )
            old = new
    save("results.json", result)
    return save


def test_complete_continuous_evidence_accepts_selected_dp1_without_claiming_dp2(tmp_path):
    from examples.operators.dsv4_4l.verify_continuous import verify_continuous_run

    complete_evidence(tmp_path)
    result = verify_continuous_run(tmp_path, dp_size=1)
    assert result["accepted"] and not result["cross_replica_dp_reduction_executed"]
    assert [x["version"] for x in result["versions"]] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        verify_continuous_run(tmp_path, dp_size=2)


@pytest.mark.parametrize(
    "change",
    [
        "missing_v2",
        "old_manifest",
        "old_ack",
        "pid",
        "optimizer_reset",
        "missing_gradient",
        "router_bias",
        "batch_order",
        "nan",
        "empty",
        "cache",
    ],
)
def test_three_step_verifier_rejects_partial_stale_or_reset_real_run_evidence(tmp_path, change):
    from examples.operators.dsv4_4l.verify_continuous import verify_continuous_run

    save = complete_evidence(tmp_path)

    def read(name):
        return json.loads((tmp_path / name).read_text())

    if change == "missing_v2":
        (tmp_path / "steps/v2/actor-distributed-step.json").unlink()
    elif change == "old_manifest":
        save("steps/v2/actor-update-manifest.json", read("steps/v1/actor-update-manifest.json"))
    elif change in ("old_ack", "pid", "cache"):
        value = read("steps/v2/rollout-sync-ack.json")
        value[0][
            {"old_ack": "weight_version", "pid": "pid", "cache": "cache_generation_after"}[change]
        ] = -1
        save("steps/v2/rollout-sync-ack.json", value)
    elif change == "optimizer_reset":
        value = read("steps/v2/actor-distributed-step.json")
        value[0]["optimizer_evidence"]["before"]["state_digest"] = "initial"
        save("steps/v2/actor-distributed-step.json", value)
    elif change == "missing_gradient":
        value = read("rank-0-v2-grad.json")
        value["local_gradients"] = {}
        save("rank-0-v2-grad.json", value)
    elif change == "router_bias":
        value = read("rank-1-v2-change.json")
        value["router_bias_exports"] = {}
        save("rank-1-v2-change.json", value)
    elif change == "batch_order":
        value = read("rollout-batches-v2-AB-matrix.json")
        value[0]["batches"][0]["input_token_rows"].reverse()
        save("rollout-batches-v2-AB-matrix.json", value)
    else:
        value = read("results.json")
        value["rows"][0]["all_finite" if change == "nan" else "compared_tokens"] = (
            False if change == "nan" else 0
        )
        save("results.json", value)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_continuous_run(tmp_path, dp_size=1)


def test_declared_right_padding_preserves_valid_prefix_keys_and_rejects_hidden_masks():
    from rl_engine.integrations.engines.train.megatron.logical_keys import (
        logical_keys_for_input,
        right_padded_lengths,
    )
    from rl_engine.integrations.engines.train.megatron.native_mhc import (
        normalize_native_causal_mask,
    )

    cases = [
        dict(id="A", prompt_token_ids=[1, 2, 3], response_token_ids=[4, 5]),
        dict(id="B", prompt_token_ids=[1, 6], response_token_ids=[7, 8]),
    ]
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 6, 7, 8, 0]])
    lengths = right_padded_lengths(ids, cases)
    assert lengths == [5, 4]
    mask = torch.arange(5)[None] < torch.tensor(lengths)[:, None]
    assert normalize_native_causal_mask(mask, ids.shape, valid_lengths=lengths) is None
    keys = logical_keys_for_input(ids, cases, valid_lengths=lengths).reshape(5, 2, 2)
    single = logical_keys_for_input(ids[1:2, :4], [cases[1]]).reshape(4, 1, 2)
    assert torch.equal(keys[:4, 1], single[:, 0])
    swapped = logical_keys_for_input(ids.flip(0), cases[::-1], valid_lengths=lengths[::-1]).reshape(
        5, 2, 2
    )
    assert torch.equal(keys.flip(1), swapped)
    with pytest.raises(ValueError, match="padding"):
        normalize_native_causal_mask(mask, ids.shape)
    for broken in [mask.roll(1, 1), mask.logical_not()]:
        with pytest.raises(ValueError, match="right padding"):
            normalize_native_causal_mask(broken, ids.shape, valid_lengths=lengths)
    changed = ids.clone()
    changed[1, -1] = 9
    with pytest.raises(ValueError, match="complete logical case"):
        right_padded_lengths(changed, cases)


def test_native_prefill_order_control_is_opt_in_and_never_reorders_decode():
    from rl_engine.integrations.engines.rollout.vllm.batch_order import requested_prefill_order

    def native(tokens, drafts, decode):
        return sorted(tokens, key=tokens.get)

    tokens = {"A": 131, "B": 130}
    assert requested_prefill_order(tokens, {}, 1, [], native) == ["B", "A"]
    assert requested_prefill_order(tokens, {}, 1, ["A", "B"], native) == ["A", "B"]
    assert requested_prefill_order(tokens, {}, 1, ["B", "A"], native) == ["B", "A"]
    for requested, values, drafts in [
        (["A", "A"], tokens, {}),
        (["A", "C"], tokens, {}),
        (["A", "B"], {"A": 1, "B": 130}, {}),
        (["A", "B"], tokens, {"A": [9]}),
    ]:
        with pytest.raises(ValueError):
            requested_prefill_order(values, drafts, 1, requested, native)


def test_batch_order_binding_changes_only_the_selected_real_runner_instance():
    from rl_engine.integrations.engines.rollout.vllm.batch_order import bind_prefill_order

    namespace = {"sort_batch_req_ids": lambda t, d, q: sorted(t, key=t.get)}
    exec(
        "def gather_batch_req_state(self, tokens): return sort_batch_req_ids(tokens, {}, 1)",
        namespace,
    )
    runner_type = type(
        "RunnerBoundaryFixture", (), {"gather_batch_req_state": namespace["gather_batch_req_state"]}
    )
    bound, untouched = runner_type(), runner_type()
    bind_prefill_order(bound, lambda: ["A", "B"])
    assert bound.gather_batch_req_state({"A": 131, "B": 130}) == ["A", "B"]
    assert untouched.gather_batch_req_state({"A": 131, "B": 130}) == ["B", "A"]


def test_native_prefill_order_maps_randomized_request_ids_without_disabling_uniqueness():
    from rl_engine.integrations.engines.rollout.vllm.batch_order import requested_prefill_order

    def native(tokens, drafts, decode):
        return sorted(tokens, key=tokens.get)

    tokens = {"v2-AB-matrix-B-abcdef12": 130, "v2-AB-matrix-A-12345678": 131}
    assert requested_prefill_order(tokens, {}, 1, ["v2-AB-matrix-A", "v2-AB-matrix-B"], native) == [
        "v2-AB-matrix-A-12345678",
        "v2-AB-matrix-B-abcdef12",
    ]
    for bad in [
        {"v2-AB-matrix-B-abcdef12": 130, "v2-AB-matrix-A-not-hex": 131},
        {"v2-AB-matrix-A-11111111": 130, "v2-AB-matrix-A-22222222": 131},
    ]:
        with pytest.raises(ValueError):
            requested_prefill_order(bad, {}, 1, ["v2-AB-matrix-A", "v2-AB-matrix-B"], native)
