# SPDX-License-Identifier: Apache-2.0
"""Local payload ownership/readback units, not rollout execution acceptance."""

from types import SimpleNamespace as NS

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import (
    digest,
    pipeline_stage_payload,
    verify_loaded_update,
)


def payload():
    weights = {
        "embed.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "head.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        "norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "hc_head_fn": torch.ones(2, 2),
        "layers.0.hc_attn_fn": torch.ones(2, 2),
        "layers.3.ffn.gate.bias": torch.tensor([0.125, -0.25]),
    }
    return dict(
        weight_version=1,
        weights=weights,
        manifest={
            k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=digest(v))
            for k, v in weights.items()
        },
    )


def wrapper(first):
    return NS(
        model=NS(
            start_layer=0 if first else 3,
            end_layer=3 if first else 4,
            embed_tokens=NS(weight=True) if first else NS(),
            norm=NS() if first else NS(weight=True),
        ),
        lm_head=NS() if first else NS(weight=True),
        config=NS(num_hidden_layers=4),
    )


def test_each_rollout_stage_receives_exactly_its_original_tensor_objects():
    state = payload()
    first = pipeline_stage_payload(wrapper(True), state)
    last = pipeline_stage_payload(wrapper(False), state)
    assert first["weight_version"] == last["weight_version"] == 1
    assert set(first["weights"]) == {"embed.weight", "layers.0.hc_attn_fn"}
    assert set(first["weights"]) | set(last["weights"]) == set(state["weights"])
    assert not set(first["weights"]) & set(last["weights"])
    for shard in [first, last]:
        for name, value in shard["weights"].items():
            assert value is state["weights"][name]


def test_every_owned_parameter_is_read_back_and_any_corruption_is_rejected():
    w = wrapper(False)
    state = pipeline_stage_payload(w, payload())
    actual = {name: torch.nn.Parameter(value.clone()) for name, value in state["weights"].items()}
    w.named_parameters = lambda: actual.items()
    w.hf_to_vllm_mapper = NS(map_name=lambda name: name)
    checked = verify_loaded_update(w, state, set(actual))
    for name, record in checked.items():
        assert record["shape"] == state["manifest"][name]["shape"]
        assert record["dtype"] == state["manifest"][name]["dtype"]
    assert set(checked) == set(state["weights"])
    with torch.no_grad():
        actual["layers.3.ffn.gate.bias"].add_(0.001)
    with pytest.raises(RuntimeError, match="readback differs"):
        verify_loaded_update(w, state, set(actual))


def test_unexpected_loader_names_and_unknown_ownership_are_rejected():
    w = wrapper(True)
    state = pipeline_stage_payload(w, payload())
    actual = {name: value.clone() for name, value in state["weights"].items()}
    w.named_parameters = lambda: actual.items()
    w.hf_to_vllm_mapper = NS(map_name=lambda name: name)
    with pytest.raises(RuntimeError, match="loader names disagree"):
        verify_loaded_update(w, state, set(actual) | {"unused_head"})
    bad = payload()
    bad["weights"]["unknown"] = bad["manifest"]["unknown"] = None
    with pytest.raises(ValueError, match="unknown checkpoint ownership"):
        pipeline_stage_payload(w, bad)


def test_tp8_padded_vocabulary_reads_only_actual_original_rows_and_detects_corruption():
    pytest.importorskip("vllm")
    # Execute only the pinned source's pure index definitions: importing its ROCm
    # platform would initialize a GPU even for this CPU metadata contract.
    import ast
    from dataclasses import dataclass
    from pathlib import Path
    import vllm

    root = Path(vllm.__file__).parent
    source = ast.parse((root / "model_executor/layers/vocab_parallel_embedding.py").read_text())
    utilities = ast.parse((root / "distributed/utils.py").read_text())
    from collections.abc import Sequence

    namespace = {"__name__": __name__, "dataclass": dataclass, "Sequence": Sequence}
    nodes = [
        n
        for n in utilities.body
        if isinstance(n, ast.FunctionDef) and n.name in ("ensure_divisibility", "divide")
    ]
    nodes += [
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in ("vocab_range_from_per_partition_vocab_size", "vocab_range_from_global_vocab_size")
    ]
    nodes += [
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "VocabParallelEmbeddingShardIndices"
    ]
    cls = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "VocabParallelEmbedding"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_indices"
    )
    pure = ast.ClassDef(name="CPUIndices", bases=[], keywords=[], body=[method], decorator_list=[])
    nodes.append(pure)
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, "<pinned-vllm-pure-vocab-indices>", "exec"), namespace)
    VocabParallelEmbedding = namespace["CPUIndices"]

    vocab = 129280
    padded = ((vocab + 8 * 64 - 1) // (8 * 64)) * (8 * 64)
    original = torch.ones(vocab, 2, dtype=torch.bfloat16)
    manifest = {
        "head.weight": dict(
            shape=list(original.shape), dtype=str(original.dtype), sha256=digest(original)
        )
    }
    state = dict(weights={"head.weight": original}, manifest=manifest, weight_version=0)
    intervals = []
    for rank in range(8):
        indices = VocabParallelEmbedding._get_indices(padded, padded, vocab, vocab, rank, 8)
        local = torch.full((padded // 8, 2), 9, dtype=torch.bfloat16)
        local[: indices.num_org_elements] = original[
            indices.org_vocab_start_index : indices.org_vocab_end_index
        ]
        w = NS(
            model=NS(),
            lm_head=NS(shard_indices=indices),
            named_parameters=lambda local=local: [("head.weight", local)],
            hf_to_vllm_mapper=NS(map_name=lambda name: name),
        )
        record = verify_loaded_update(w, state, {"head.weight"})["head.weight"]
        part = record["tp_partition"]
        intervals.append((part["start"], part["stop"]))
        assert record["shape"] == [indices.num_org_elements, 2]
        assert record["logical_sha256"] == digest(original)
        local[0, 0] = 2
        with pytest.raises(RuntimeError, match="readback differs"):
            verify_loaded_update(w, state, {"head.weight"})
    assert intervals[0][0] == 0 and intervals[-1][1] == vocab
    assert all(a[1] == b[0] for a, b in zip(intervals[:-1], intervals[1:], strict=True))
