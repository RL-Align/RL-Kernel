# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.config.workload import LogicalSample
from rl_engine.ops.rope.multi_axis_rope import build_multi_axis_cos_sin, qwen_image_positions
from rl_engine.validation.operators.gradient_adapters import _row_inputs
from rl_engine.validation.operators.operator_inputs import make_operator_inputs, operator_shape_name
from rl_engine.validation.operators.operator_specs import (
    make_candidate,
    make_operator_case,
    operator_names,
)


def _args(**overrides):
    values = {
        "batch": 1,
        "seq": 2,
        "vocab": 17,
        "seed": 123,
        "input_mode": "constant",
        "constant_value": 0.5,
        "token_value": 3,
        "normalized_dim": 128,
        "n_heads": 32,
        "head_dim": 128,
        "k_dim": 16,
        "n_dim": 32,
        "theta": 1.0e6,
        "eps": 1.0e-6,
        "arch_key": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "op_name",
    [
        "rms_norm",
        "qk_norm",
        "pack",
        "matmul",
        "det_gemm",
        "attention",
        "logp",
        "linear_logp",
        "batch_invariant_logp",
        "rope",
        "silu",
        "swiglu",
        "embedding",
        "lm_head",
        "kv_cache_attention",
    ],
)
def test_operator_inputs_support_all_issue_108_ops(op_name):
    args = _args()
    inputs = make_operator_inputs(op_name, args, torch.float32, torch.device("cpu"))

    assert inputs
    assert operator_shape_name(op_name, args)


def test_constant_logp_inputs_are_deterministic():
    args = _args(input_mode="constant", constant_value=0.5, token_value=3)
    inputs = make_operator_inputs("logp", args, torch.float32, torch.device("cpu"))

    assert torch.equal(inputs["logits"], torch.full((1, 2, 17), 0.5))
    assert torch.equal(inputs["token_ids"], torch.full((1, 2), 3, dtype=torch.long))


def test_constant_batch_invariant_logp_inputs_match_operator_contract():
    args = _args(input_mode="constant", constant_value=0.5, token_value=3)
    inputs = make_operator_inputs("batch_invariant_logp", args, torch.float32, torch.device("cpu"))

    assert torch.equal(inputs["logits"], torch.full((1, 2, 17), 0.5))
    assert torch.equal(inputs["target_ids"], torch.full((1, 2), 3, dtype=torch.long))


def test_random_logp_inputs_are_seeded():
    args = _args(input_mode="random", seed=7)
    first = make_operator_inputs("logp", args, torch.float32, torch.device("cpu"))
    second = make_operator_inputs("logp", args, torch.float32, torch.device("cpu"))

    assert torch.equal(first["logits"], second["logits"])
    assert torch.equal(first["token_ids"], second["token_ids"])


def test_cp_attention_operator_spec_registers_backward_grad_inputs():
    args = _args(op="cp_attention", input_mode="constant", batch=1, seq=2)

    assert "cp_attention" in operator_names()
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    candidate = make_candidate(argparse.Namespace(**{**vars(args), "candidate": "pytorch"}))

    assert case.op_class == "attention"
    assert case.grad_input_names == ("q", "k", "v")
    assert candidate.name == "pytorch-cp_attention"


def test_multi_axis_rope_operator_spec_and_seeded_inputs():
    args = _args(op="multi_axis_rope", batch=2, seq=7, n_heads=3, input_mode="random")
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    inputs = make_operator_inputs("multi_axis_rope", args, torch.float32, torch.device("cpu"))
    repeated = make_operator_inputs("multi_axis_rope", args, torch.float32, torch.device("cpu"))
    assert case.op_class == "elementwise"
    assert case.grad_input_names == ("x",)
    assert inputs["x"].shape == (2, 7, 3, 128)
    assert inputs["cos"].shape == inputs["sin"].shape == (7, 64)
    for name in inputs:
        assert torch.equal(inputs[name], repeated[name])
    assert operator_shape_name("multi_axis_rope", args) == "2x7x3x128"
    with pytest.raises(ValueError, match="128"):
        make_operator_inputs(
            "multi_axis_rope", _args(head_dim=64), torch.float32, torch.device("cpu")
        )


@pytest.mark.parametrize(
    "seq,side", [(0, 1), (1, 1), (7, 3), (15, 4), (16, 4), (4096, 64), (6889, 83)]
)
def test_multi_axis_rope_validation_uses_centered_image_grid_prefix(seq, side):
    inputs = make_operator_inputs(
        "multi_axis_rope", _args(seq=seq, n_heads=1), torch.float32, torch.device("cpu")
    )
    image, _ = qwen_image_positions(side, side, 0)
    cos, sin = build_multi_axis_cos_sin(image[:seq])
    assert torch.equal(inputs["cos"], cos)
    assert torch.equal(inputs["sin"], sin)


@pytest.mark.parametrize("layout", ["packed", "permuted", "chunked", "padded"])
def test_multi_axis_rope_validation_preserves_logical_tables_across_layouts(layout):
    samples = (
        LogicalSample("short", tuple(range(7)), prompt_len=1, seq_len=7),
        LogicalSample("long", tuple(range(15)), prompt_len=2, seq_len=15),
    )
    tokens = {
        (token.sample_id, token.token_position): token
        for sample in samples
        for token in sample.tokens()
    }
    expected = {
        sample.sample_id: make_operator_inputs(
            "multi_axis_rope",
            _args(seq=sample.seq_len, n_heads=1),
            torch.float32,
            torch.device("cpu"),
        )
        for sample in samples
    }
    keys = list(tokens)
    if layout == "permuted":
        keys.reverse()
    elif layout == "chunked":
        keys = [("short", 5), ("short", 6), ("long", 12)]
    elif layout == "padded":
        keys = [None, ("long", 4), None, ("short", 6), ("short", 0), None]
    inputs = _row_inputs(
        "multi_axis_rope",
        keys,
        tokens,
        {},
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden=1,
        vocab_size=17,
        n_heads=1,
        head_dim=128,
    )
    for row, key in enumerate(keys):
        if key is None:
            assert torch.equal(inputs["cos"][row], torch.ones(64))
            assert torch.equal(inputs["sin"][row], torch.zeros(64))
        else:
            sample_id, position = key
            for name in ("cos", "sin"):
                assert torch.equal(inputs[name][row], expected[sample_id][name][position])


def test_constant_linear_logp_inputs_match_operator_contract():
    args = _args(input_mode="constant", constant_value=0.5, token_value=3)
    inputs = make_operator_inputs("linear_logp", args, torch.float32, torch.device("cpu"))

    assert torch.equal(inputs["hidden"], torch.full((1, 2, 128), 0.5))
    assert torch.equal(inputs["lm_head_weight"], torch.full((17, 128), 0.51))
    assert torch.equal(inputs["target_ids"], torch.full((1, 2), 3, dtype=torch.long))
    assert inputs["bias"] is None


def test_constant_embedding_inputs_match_operator_contract():
    args = _args(input_mode="constant", constant_value=0.5, token_value=3)
    inputs = make_operator_inputs("embedding", args, torch.float32, torch.device("cpu"))

    assert torch.equal(inputs["token_ids"], torch.full((1, 2), 3, dtype=torch.long))
    assert inputs["weight"].shape == (17, 128)
    assert inputs["weight"].dtype is torch.float32
    assert torch.equal(inputs["weight"], torch.full((17, 128), 0.5))
    assert operator_shape_name("embedding", args) == "1x2x17x128"


def test_constant_lm_head_inputs_match_operator_contract():
    args = _args(input_mode="constant", constant_value=0.5)
    inputs = make_operator_inputs("lm_head", args, torch.float32, torch.device("cpu"))

    assert inputs["hidden"].shape == (1, 2, 128)
    assert inputs["weight"].shape == (17, 128)
    assert inputs["hidden"].dtype is torch.float32
    assert inputs["weight"].dtype is torch.float32
    assert torch.equal(inputs["hidden"], torch.full((1, 2, 128), 0.5))
    assert torch.equal(inputs["weight"], torch.full((17, 128), 0.51))
    assert inputs["bias"] is None
    assert operator_shape_name("lm_head", args) == "1x2x128x17"
