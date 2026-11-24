# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU protocol tests. These are not Megatron/vLLM/GPU acceptance results."""

from __future__ import annotations

import copy
import sys
from functools import partial
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rl_engine.alignment.debug import (
    AutoCapture,
    DecodeCapture,
    EventCapture,
    FusedCapture,
    GraphCapture,
    ModuleBoundary,
    ShardSet,
    TensorShard,
    TrainingCapture,
    load_megatron_model,
    megatron_weight_shards,
    moe_dispatch,
    paged_kv,
    parallel_tensor,
    prepare_auto,
)
from rl_engine.alignment.debug.events import analyze_events, encode_tree, _merge_trees


IDENTITY = {
    "weights": "test-model",
    "tokens": [1, 2],
    "positions": [0, 1],
    "mask": "causal",
    "sampling": {},
}


def prepare(root, boundaries=("point",), ranks=None):
    EventCapture.prepare(
        root,
        [{"id": name, "module": "native", "stage": name} for name in boundaries],
        scope="CPU protocol test, not native framework acceptance",
        ranks=ranks,
    )


def paired(root, values, *, ranks=None):
    prepare(root, ranks=ranks)
    for side, workers in values.items():
        for rank, value in enumerate(workers):
            with EventCapture(root, side, IDENTITY, rank=rank) as capture:
                capture.record("point", value)
    return analyze_events(root)


@pytest.mark.parametrize("cp_layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("tp_stride", [1, 2, 3])
def test_reconstructs_different_tp_cp_topologies(tmp_path, cp_layout, tp_stride):
    full = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    shards = []
    for cp in range(2):
        for tp in range(2):
            view = parallel_tensor(
                torch.zeros(4, 6),
                tp_rank=tp,
                tp_size=2,
                tp_axis=1,
                tp_stride=tp_stride,
                cp_rank=cp,
                cp_size=2,
                cp_axis=0,
                cp_layout=cp_layout,
            )
            local = full[
                torch.tensor(view.indices[0])[:, None], torch.tensor(view.indices[1])[None, :]
            ]
            shards.append(TensorShard(local, view.global_shape, view.indices))
    report = paired(
        tmp_path,
        {"training": shards, "rollout": [full]},
        ranks={"training": [0, 1, 2, 3], "rollout": [0]},
    )
    assert report["status"] == "equal", report["errors"]


@pytest.mark.parametrize("failure", ["gap", "overlap", "shape", "dtype"])
def test_invalid_shards_never_pass(tmp_path, failure):
    full = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    first = TensorShard(full[:, :2], (2, 4), ([0, 1], [0, 1]))
    if failure == "gap":
        second = first
    elif failure == "overlap":
        second = TensorShard(full + 1, (2, 4), ([0, 1], range(4)))
    elif failure == "shape":
        second = TensorShard(full, (3, 4), ([0, 1], range(4)))
    else:
        second = TensorShard(full.double(), (2, 4), ([0, 1], range(4)))
    report = paired(tmp_path, {"training": [ShardSet([first, second])], "rollout": [full]})
    assert report["status"] == "not_comparable"


def test_scalar_and_empty_local_shards():
    scalar = torch.tensor(3.0)
    assert _merge_trees([encode_tree(TensorShard.replicated(scalar))])["value"] == scalar
    empty = TensorShard(torch.empty(0, 2), (2, 2), ([], [0, 1]))
    full = TensorShard.replicated(torch.eye(2))
    assert torch.equal(_merge_trees([encode_tree(empty), encode_tree(full)])["value"], torch.eye(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp_size": 2},
        {"cp_size": 2},
        {"tp_rank": 2},
        {"cp_layout": "guess"},
        {"tp_axis": 1, "cp_axis": 1},
        {"cp_axis": 8},
        {"tp_axis": 0, "tp_stride": 3},
    ],
)
def test_partition_metadata_is_required(kwargs):
    with pytest.raises(ValueError):
        parallel_tensor(torch.ones(2, 4), **kwargs)


def test_sharded_auto_capture_uses_actual_weights(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.data.copy_(torch.eye(2))
    prepare_auto(
        tmp_path,
        model,
        positions=[0, 1],
        scope="sharded native weights",
        sharded_weights=True,
    )
    for side in ("training", "rollout"):

        def reader():
            return {"weight": TensorShard.replicated(model.weight)}

        with AutoCapture(
            tmp_path,
            side,
            IDENTITY,
            model,
            positions=[0, 1],
            sharded_weights=reader,
        ):
            model(torch.ones(2, 2))
    assert analyze_events(tmp_path)["status"] == "equal"


def test_sharded_auto_capture_detects_local_mutation(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    prepare_auto(
        tmp_path,
        model,
        positions=[0, 1],
        scope="mutation rejection",
        sharded_weights=True,
    )
    for side in ("training", "rollout"):
        with AutoCapture(
            tmp_path,
            side,
            IDENTITY,
            model,
            positions=[0, 1],
            sharded_weights=lambda: {"weight": TensorShard.replicated(model.weight)},
        ):
            model(torch.ones(2, 2))
            model.weight.data.add_(1)
    assert analyze_events(tmp_path)["status"] == "not_comparable"


def test_auto_capture_tp_cp_shards_and_weights(tmp_path):
    class Native(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, x):
            return x * self.weight

    full = torch.arange(32.0).reshape(8, 4)
    bindings = {"block": ModuleBoundary("")}
    template = Native(torch.ones_like(full))
    prepare_auto(
        tmp_path,
        template,
        positions=range(8),
        scope="native TP2 CP2 protocol",
        bindings=bindings,
        capture_root=False,
        sharded_weights=True,
        ranks={"training": [0, 1, 2, 3], "rollout": [0]},
    )
    for side, count in (("training", 4), ("rollout", 1)):
        for rank in range(count):
            tp, cp = (rank % 2, rank // 2) if count == 4 else (0, 0)

            def view(t, tp=tp, cp=cp, count=count):
                return parallel_tensor(
                    t,
                    tp_rank=tp,
                    tp_size=2 if count == 4 else 1,
                    tp_axis=1,
                    cp_rank=cp,
                    cp_size=2 if count == 4 else 1,
                    cp_axis=0,
                    cp_layout="zigzag",
                )

            shard = view(torch.ones(4, 2) if count == 4 else torch.ones_like(full))
            x = full[
                torch.tensor(shard.indices[0])[:, None], torch.tensor(shard.indices[1])[None, :]
            ]
            model = Native(torch.ones_like(x))
            mapped = {
                "block": ModuleBoundary(
                    "",
                    inputs=lambda m, a, k: view(a[0]),
                    outputs=lambda m, a, k, o: view(o),
                )
            }
            with AutoCapture(
                tmp_path,
                side,
                IDENTITY,
                model,
                positions=range(8),
                rank=rank,
                bindings=mapped,
                capture_root=False,
                sharded_weights=lambda model=model, view=view: {"weight": view(model.weight)},
            ):
                model(x)
    report = analyze_events(tmp_path)
    assert report["status"] == "equal", report["errors"]


def native_modules(monkeypatch):
    """Fake only the native provider/checkpoint protocol, explicitly not Megatron."""
    root, core, checkpoint, mapping = [
        ModuleType(n)
        for n in (
            "megatron",
            "megatron.core",
            "megatron.core.dist_checkpointing",
            "megatron.core.dist_checkpointing.mapping",
        )
    ]
    root.core, core.dist_checkpointing = core, checkpoint
    mapping.ShardedTensor = type("ShardedTensor", (), {})
    mapping.ShardedTensorFactory = type("ShardedTensorFactory", (), {})
    mapping.ShardedObject = type("ShardedObject", (SimpleNamespace,), {})
    for module in (root, core, checkpoint, mapping):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return checkpoint, mapping


def test_megatron_provider_and_strict_native_load(tmp_path, monkeypatch):
    checkpoint, _ = native_modules(monkeypatch)

    class Native(torch.nn.Linear):
        def sharded_state_dict(self):
            return self.state_dict()

    expected = Native(2, 2)
    checkpoint.load = Mock(return_value={"model": expected.state_dict()})
    provider = Mock(side_effect=lambda width: Native(width, width))
    model = load_megatron_model(
        provider,
        tmp_path,
        provider_kwargs={"width": 2},
        model_key="model",
    )
    provider.assert_called_once_with(width=2)
    assert torch.equal(model.weight, expected.weight)
    assert model.training
    assert checkpoint.load.call_args.kwargs == {
        "validate_access_integrity": True,
        "strict": "raise_all",
    }


def test_megatron_load_failure_does_not_return_model(tmp_path, monkeypatch):
    checkpoint, _ = native_modules(monkeypatch)
    module = torch.nn.Linear(2, 2)
    module.sharded_state_dict = module.state_dict
    checkpoint.load = Mock(side_effect=ValueError("missing weight"))
    with pytest.raises(ValueError, match="missing weight"):
        load_megatron_model(lambda: module, tmp_path)


def test_megatron_stacked_axes_and_factories(monkeypatch):
    _, mapping = native_modules(monkeypatch)

    class NativeShard(SimpleNamespace):
        pass

    class Factory:
        def build(self):
            return [
                NativeShard(
                    key="decoder.weight",
                    data=torch.ones(2, 2) * layer,
                    local_shape=(2, 2),
                    global_shape=(2, 2, 2),
                    global_offset=(layer, 0, 0),
                    prepend_axis_num=1,
                    flattened_range=None,
                    allow_shape_mismatch=True,
                )
                for layer in range(2)
            ]

    mapping.ShardedTensor, mapping.ShardedTensorFactory = NativeShard, Factory
    model = SimpleNamespace(sharded_state_dict=lambda: {"factory": Factory()})
    shards = megatron_weight_shards(model)
    actual = _merge_trees([encode_tree(shards["decoder.weight"])])["value"]
    assert actual.shape == (2, 2, 2)
    assert torch.equal(actual[0], torch.zeros(2, 2))
    assert torch.equal(actual[1], torch.ones(2, 2))


def test_megatron_loader_disables_shape_resize_including_factories(tmp_path, monkeypatch):
    checkpoint, mapping = native_modules(monkeypatch)

    @dataclass
    class NativeShard:
        data: torch.Tensor
        allow_shape_mismatch: bool = True

    @dataclass
    class Factory:
        build_fn: object
        merge_fn: object

        def build(self):
            return self.build_fn()

    mapping.ShardedTensor, mapping.ShardedTensorFactory = NativeShard, Factory
    model = torch.nn.Linear(2, 2)
    descriptor = NativeShard(model.weight)
    factory = Factory(lambda: {"weight": descriptor}, lambda values: values)
    model.sharded_state_dict = lambda: {"direct": descriptor, "factory": factory}

    def load(state, path, **kwargs):
        assert not state["direct"].allow_shape_mismatch
        assert not state["factory"].build()["weight"].allow_shape_mismatch
        assert state["factory"].merge_fn is factory.merge_fn
        assert state["direct"].data is model.weight
        assert descriptor.allow_shape_mismatch
        assert factory.build()["weight"].allow_shape_mismatch
        return model.state_dict()

    checkpoint.load = load
    assert load_megatron_model(lambda: model, tmp_path) is model


@pytest.mark.parametrize(
    ("payload", "empty"),
    [
        (None, True),
        (torch.empty(0, dtype=torch.uint8), True),
        (torch.tensor([0], dtype=torch.uint8), False),
        ({"fp8_history": torch.ones(2)}, False),
    ],
)
def test_megatron_extra_state_must_be_explicitly_empty(monkeypatch, payload, empty):
    _, mapping = native_modules(monkeypatch)
    model = SimpleNamespace(
        sharded_state_dict=lambda: {
            "weight": torch.ones(2),
            "extra": mapping.ShardedObject(data=payload),
        }
    )
    if empty:
        assert set(megatron_weight_shards(model)) == {"weight"}
    else:
        with pytest.raises(ValueError, match="nonempty native extra state"):
            megatron_weight_shards(model)


def physical_cache(order):
    logical = torch.arange(6.0).reshape(3, 2, 1, 1)
    key = torch.full((5, 2, 1, 1), float("nan"))
    for logical_page, physical_page in enumerate(order):
        key[physical_page] = logical[logical_page]
    return key, key + 20, torch.tensor([order]), torch.tensor([5])


def test_paged_decode_ignores_allocator_padding_and_page_ids(tmp_path):
    EventCapture.prepare(tmp_path, DecodeCapture.boundaries(), scope="paged decode protocol")
    for side, order in (("training", [0, 2, 3]), ("rollout", [4, 1, 0])):
        key, value, table, lengths = physical_cache(order)
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observe = DecodeCapture(
                capture,
                partial(paged_kv, key, value, table, lengths, sample_ids=[7]),
            )

            def forward(x, key=key, value=value, order=order, lengths=lengths):
                key[order[-1], 1] = x
                value[order[-1], 1] = x + 20
                lengths.add_(1)
                return x + 1

            result = observe.run(forward, torch.tensor(5.0), query_rows=[(7, 5)])
            assert result.item() == 6
    assert analyze_events(tmp_path)["status"] == "equal"


def test_paged_cache_mutation_localizes_to_decode_output(tmp_path):
    EventCapture.prepare(tmp_path, DecodeCapture.boundaries(), scope="cache mutation")
    for side in ("training", "rollout"):
        key, value, table, lengths = physical_cache([0, 2, 3])
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = DecodeCapture(
                capture,
                partial(paged_kv, key, value, table, lengths, sample_ids=[0]),
            )

            def native(x, side=side, key=key):
                if side == "rollout":
                    key[0, 0].add_(0.5)
                return x

            observer.run(native, torch.tensor([1.0]), query_rows=[(0, 5)])
    report = analyze_events(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "decode.output"
    assert report["first_divergence"]["input_evidence"] == "equal"


@pytest.mark.parametrize("failure", ["page", "length", "duplicate", "dtype", "sample"])
def test_paged_cache_invalid_metadata(failure):
    key, value, table, lengths = physical_cache([0, 2, 3])
    samples = [0]
    if failure == "page":
        table[0, 0] = 9
    elif failure == "length":
        lengths[0] = 7
    elif failure == "duplicate":
        table[0, 1] = 0
    elif failure == "dtype":
        table = table.float()
    else:
        samples = [0, 0]
    with pytest.raises(ValueError):
        paged_kv(key, value, table, lengths, sample_ids=samples)


def routing(permutation=(0, 1, 2, 3), *, dropped=None):
    expert_ids = torch.tensor([[0, 1], [1, 2]])
    token = torch.tensor([0, 0, 1, 1])[list(permutation)]
    slot = torch.tensor([0, 1, 0, 1])[list(permutation)]
    return moe_dispatch(
        torch.tensor([[1.0, 2], [1, 2], [3, 4], [3, 4]])[list(permutation)],
        token_rows=[(0, 0), (0, 1)],
        expert_indices=expert_ids,
        routing_weights=torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
        dropped=torch.zeros(2, 2, dtype=torch.bool) if dropped is None else dropped,
        source_tokens=token,
        source_slots=slot,
        num_experts=3,
    )


def test_moe_dispatch_permutation_and_ep_shards(tmp_path):
    report = paired(
        tmp_path,
        {"training": [routing()], "rollout": [routing([3, 0]), routing([2, 1])]},
        ranks={"training": [0], "rollout": [0, 1]},
    )
    assert report["status"] == "equal", report["errors"]


def test_moe_missing_route_is_not_comparable(tmp_path):
    report = paired(tmp_path, {"training": [routing()], "rollout": [routing([0, 1, 2])]})
    assert report["status"] == "not_comparable"
    assert any("incomplete MoE" in error for error in report["errors"])


def test_moe_router_weight_divergence(tmp_path):
    wrong = routing()
    wrong["weights"][0, 0] += 0.1
    report = paired(tmp_path, {"training": [routing()], "rollout": [wrong]})
    assert report["status"] == "diverged"


def test_all_dropped_routes_are_explicit_not_missing(tmp_path):
    dropped = torch.ones(2, 2, dtype=torch.bool)
    report = paired(
        tmp_path,
        {side: [routing([], dropped=dropped)] for side in ("training", "rollout")},
    )
    assert report["status"] == "equal", report["errors"]


def test_duplicate_dispatch_route_rejected():
    with pytest.raises(ValueError, match="route ownership"):
        routing([0, 0, 1, 2])


def test_graph_staging_preserves_per_replay_snapshots(tmp_path):
    prepare(tmp_path, ("input", "output"))
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = GraphCapture(capture)
            x = torch.tensor([1.0, 2.0])
            observer.declare("input", x)
            observer.declare("output", x, inputs=("input",))

            def native_replay(observer=observer, x=x):
                observer.stage("input", x)
                x.add_(2)
                observer.stage("output", x)

            # CPU fake tests the protocol only. Real CUDA/HIP graph test below.
            graph = SimpleNamespace(replay=native_replay)
            observer.replay(graph)
            observer.replay(graph)
            x.fill_(99)
    report = analyze_events(tmp_path)
    assert report["status"] == "equal", report["errors"]
    saved = torch.load(tmp_path / "events/training-rank0/00000000.pt", weights_only=True)
    assert saved["tree"]["value"].tolist() == [1, 2]


@pytest.mark.parametrize("failure", ["stale", "twice", "order", "shape", "metadata"])
def test_graph_missing_stale_or_changed_probe_fails_closed(tmp_path, failure):
    prepare(tmp_path, ("a", "b"))
    with EventCapture(tmp_path, "training", IDENTITY) as capture:
        observer = GraphCapture(capture)
        x = torch.ones(2)
        observer.declare("a", {"x": x, "layout": "native"})
        observer.declare("b", x, inputs=("a",))

        def replay():
            if failure == "stale":
                return
            if failure == "order":
                observer.stage("b", x)
            observer.stage(
                "a",
                {
                    "x": torch.ones(3) if failure == "shape" else x,
                    "layout": "changed" if failure == "metadata" else "native",
                },
            )
            if failure != "order":
                observer.stage("b", x)
            if failure == "twice":
                observer.stage("b", x)

        with pytest.raises(ValueError):
            observer.replay(SimpleNamespace(replay=replay))
    assert analyze_events(tmp_path)["status"] == "not_comparable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA/HIP device")
def test_real_cuda_or_hip_graph_replay(tmp_path):
    prepare(tmp_path, ("input", "output"))
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = GraphCapture(capture)
            x = torch.tensor([1.0, 2.0], device="cuda")
            observer.declare("input", x)
            observer.declare("output", x, inputs=("input",))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    observer.stage("input", x)
                    observer.stage("output", x * 2)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                observer.stage("input", x)
                y = x * 2
                observer.stage("output", y)
            for i in (1, 4):
                x.fill_(i)
                observer.replay(graph)
                assert torch.equal(y, x * 2)
    report = analyze_events(tmp_path)
    assert report["status"] == "equal", report["errors"]


def test_fused_probes_capture_actual_intermediates_and_preserve_result(tmp_path):
    specs = FusedCapture.boundaries("kernel", ["projection", "activation"])
    EventCapture.prepare(tmp_path, specs, scope="instrumented CPU native callable")
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = FusedCapture(
                capture,
                "kernel",
                ["projection", "activation"],
                kernel_id="native-test",
                probe_contract="linear-relu-v1",
            )
            calls = []

            def execute(emit, calls=calls):
                calls.append(True)
                x = torch.tensor([-1.0, 2])
                emit("projection", x)
                x.relu_()
                emit("activation", x)
                return x

            result = observer.run(execute, inputs=torch.tensor([-1.0, 2]))
            assert result.tolist() == [0, 2]
            assert calls == [True]
    report = analyze_events(tmp_path)
    assert report["status"] == "equal"
    saved = torch.load(tmp_path / "events/training-rank0/00000001.pt", weights_only=True)
    assert saved["tree"]["value"].tolist() == [-1, 2]


@pytest.mark.parametrize("stages", [[], ["hidden", "hidden"], ["wrong"]])
def test_missing_duplicate_or_unknown_fused_probes_cannot_pass(tmp_path, stages):
    EventCapture.prepare(
        tmp_path,
        FusedCapture.boundaries("kernel", ["hidden"]),
        scope="failed probes",
    )
    with EventCapture(tmp_path, "training", IDENTITY) as capture:
        observer = FusedCapture(
            capture,
            "kernel",
            ["hidden"],
            kernel_id="native",
            probe_contract="test-v1",
        )

        def execute(emit):
            for stage in stages:
                emit(stage, torch.ones(2))
            return torch.ones(2)

        with pytest.raises(ValueError):
            observer.run(execute, inputs=torch.ones(2))
    assert analyze_events(tmp_path)["status"] == "not_comparable"


def train(model, optimizer, observer=None):
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for x in (torch.tensor([[1.0, 2]]), torch.tensor([[3.0, 4]])):
            loss = model(x).square().sum()
            if observer is None:
                loss.backward()
            else:
                observer.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.AdamW])
def test_training_capture_preserves_native_accumulation_clipping_and_updates(
    tmp_path,
    optimizer_class,
):
    torch.manual_seed(11)
    original = torch.nn.Linear(2, 2)
    reference = copy.deepcopy(original)
    opt = optimizer_class(reference.parameters(), lr=0.01)
    train(reference, opt)
    TrainingCapture.prepare(tmp_path, scope="CPU training lifecycle")
    for side in ("training", "rollout"):
        model = copy.deepcopy(original)
        optimizer = optimizer_class(model.parameters(), lr=0.01)
        before_pre, before_post = (
            len(optimizer._optimizer_step_pre_hooks),
            len(optimizer._optimizer_step_post_hooks),
        )
        with TrainingCapture(tmp_path, side, IDENTITY, model, optimizer) as observer:
            train(model, optimizer, observer)
        assert len(optimizer._optimizer_step_pre_hooks) == before_pre
        assert len(optimizer._optimizer_step_post_hooks) == before_post
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                model.parameters(),
                reference.parameters(),
                strict=True,
            )
        )
    report = analyze_events(tmp_path)
    assert report["status"] == "equal", report["errors"]
    assert len(report["boundaries"]) == 14


def test_backward_gradient_bug_localizes_without_changing_forward(tmp_path):
    TrainingCapture.prepare(tmp_path, scope="gradient fault")
    original = torch.nn.Linear(2, 1, bias=False)
    for side in ("training", "rollout"):
        model = copy.deepcopy(original)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if side == "rollout":
            model.weight.register_hook(lambda gradient: gradient + 0.25)
        with TrainingCapture(tmp_path, side, IDENTITY, model, optimizer) as observer:
            observer.backward(model(torch.ones(1, 2)).sum())
            optimizer.step()
    report = analyze_events(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "backward.output"
    assert report["first_divergence"]["input_evidence"] == "unknown"


def test_optimizer_fault_localizes_to_update(tmp_path):
    class BrokenSGD(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].add_(0.25)
            return result

    TrainingCapture.prepare(tmp_path, scope="optimizer fault")
    original = torch.nn.Linear(2, 1, bias=False)
    for side in ("training", "rollout"):
        model = copy.deepcopy(original)
        cls = torch.optim.SGD if side == "training" else BrokenSGD
        optimizer = cls(model.parameters(), lr=0.1)
        with TrainingCapture(tmp_path, side, IDENTITY, model, optimizer) as observer:
            observer.backward(model(torch.ones(1, 2)).sum())
            optimizer.step()
    report = analyze_events(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "step.output"


def test_swallowed_backward_failure_and_hooks_cleanup(tmp_path):
    TrainingCapture.prepare(tmp_path, scope="failed backward")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with TrainingCapture(tmp_path, "training", IDENTITY, model, optimizer) as observer:
        with pytest.raises(RuntimeError):
            observer.backward(torch.tensor(1.0))
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks
    assert analyze_events(tmp_path)["status"] == "not_comparable"


def test_step_without_observed_backward_is_rejected(tmp_path):
    TrainingCapture.prepare(tmp_path, scope="no backward")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with TrainingCapture(tmp_path, "training", IDENTITY, model, optimizer):
        with pytest.raises(ValueError, match="observed backward"):
            optimizer.step()


@pytest.mark.parametrize("kind", ["sharded", "routed"])
def test_corrupt_extended_snapshot_is_not_comparable(tmp_path, kind):
    value = TensorShard.replicated(torch.eye(2)) if kind == "sharded" else routing()
    paired(tmp_path, {side: [value] for side in ("training", "rollout")})
    path = tmp_path / "events/rollout-rank0/00000000.pt"
    record = torch.load(path, weights_only=True)
    if kind == "sharded":
        record["tree"]["parts"][0]["value"] = "not a tensor"
    else:
        record["tree"]["items"]["dispatch"]["value"] = "not a tensor"
    torch.save(record, path)
    report = analyze_events(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("must be a tensor" in error for error in report["errors"])


@pytest.mark.parametrize("changed", [False, True])
def test_extended_evidence_uses_existing_offline_cli(tmp_path, changed, capsys):
    from rl_engine.alignment.debug.entry import main

    left = torch.arange(4.0).reshape(2, 2)
    right = left.clone()
    if changed:
        right[1, 1] += 0.5
    paired(
        tmp_path,
        {"training": [TensorShard.replicated(left)], "rollout": [right]},
    )
    assert main([str(tmp_path)]) == (1 if changed else 0)
    assert (tmp_path / "diagnostic-report.json").is_file()
    assert ("diverged" if changed else "equal") in capsys.readouterr().out


def test_graph_module_hooks_are_observational_and_removed(tmp_path):
    prepare(tmp_path, ("relu.input", "relu.output"))
    model = torch.nn.ReLU()
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, IDENTITY) as capture:
            observer = GraphCapture(capture)
            x = torch.tensor([-1.0, 2.0])
            observer.declare("relu.input", x)
            observer.declare("relu.output", x, inputs=("relu.input",))
            bindings = {"relu": ModuleBoundary("", inputs=lambda m, a, k: a[0])}
            with observer.observe(model, bindings):
                graph = SimpleNamespace(replay=partial(model, x))
                observer.replay(graph)
                assert model(x).tolist() == [0, 2]
            assert not model._forward_pre_hooks
            assert not model._forward_hooks
    assert analyze_events(tmp_path)["status"] == "equal"


def test_graph_module_hook_failure_removes_previously_registered_hooks(tmp_path):
    prepare(tmp_path, ("ok.input", "ok.output"))
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Identity())
    with EventCapture(tmp_path, "training", IDENTITY) as capture:
        observer = GraphCapture(capture)
        observer.declare("ok.input", torch.ones(2))
        observer.declare("ok.output", torch.ones(2))
        with pytest.raises(ValueError, match="declare graph slots"):
            with observer.observe(
                model,
                {"ok": ModuleBoundary("0"), "missing": ModuleBoundary("1")},
            ):
                pass
        assert not model[0]._forward_pre_hooks
        assert not model[0]._forward_hooks
    assert analyze_events(tmp_path)["status"] == "not_comparable"
