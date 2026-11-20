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
