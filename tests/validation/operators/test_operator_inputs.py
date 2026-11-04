# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.validation.operators.op_checks import run_operator_suite
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


def test_video_input_projection_contract_and_logical_keys():
    args = _args(batch=2, seq=3)
    inputs = make_operator_inputs(
        "video_input_projection_fp32", args, torch.float32, torch.device("cpu")
    )
    assert inputs["x"].shape == (2, 3, 96)
    assert inputs["weight"].shape == (5376, 96)
    assert inputs["bias"].shape == (5376,)
    assert inputs["logical_keys"].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert operator_shape_name("video_input_projection_fp32", args) == "2x3x96x5376"
    with pytest.raises(TypeError, match="FP32"):
        make_operator_inputs(
            "video_input_projection_fp32", args, torch.bfloat16, torch.device("cpu")
        )


def test_video_output_projection_contract_and_logical_keys():
    args = _args(batch=2, seq=3)
    inputs = make_operator_inputs(
        "video_output_projection_fp32", args, torch.float32, torch.device("cpu")
    )
    assert inputs["x"].shape == (2, 3, 5376)
    assert inputs["weight"].shape == (96, 5376)
    assert inputs["bias"].shape == (96,)
    assert inputs["logical_keys"].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    assert operator_shape_name("video_output_projection_fp32", args) == "2x3x5376x96"
    with pytest.raises(TypeError, match="FP32"):
        make_operator_inputs(
            "video_output_projection_fp32", args, torch.bfloat16, torch.device("cpu")
        )


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_h3_mm_rope_checker_keeps_fp32_tables_and_independent_gold(dtype):
    from rl_engine.reference.minimax_h3.block import golden_apply_rotary

    args = _args(op="h3_mm_rope_3axis_partial", candidate="pytorch", n_heads=14)
    case = make_operator_case(args, dtype, torch.device("cpu"))
    candidate = make_candidate(args)
    x, cos, sin = (case.inputs[name] for name in ("x", "cos", "sin"))
    assert x.shape == (1, 2, 14, 128) and x.dtype == dtype
    assert cos.shape == sin.shape == (2, 96)
    assert cos.dtype == sin.dtype == torch.float32
    assert not cos.requires_grad and not sin.requires_grad
    assert case.grad_input_names == ("x",)
    assert operator_shape_name(args.op, args) == "1x2x14x128"

    output = case.gold_fn(**case.inputs)
    assert output.dtype == torch.float32
    assert torch.equal(output, golden_apply_rotary(x, cos, sin))
    if dtype == torch.bfloat16:
        assert not torch.equal(output, candidate.fn(**case.inputs).float())
    report = run_operator_suite(args.op, candidates=[candidate], cases=[case], check_grad=True)
    assert report.passed, report.to_dict()
    assert {entry.judgment for entry in report.candidates[0].cases[0].outputs} == {
        "forward_accuracy",
        "gradient_accuracy",
    }


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
