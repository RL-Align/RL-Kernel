# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
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


@pytest.mark.parametrize("order", tuple(SwiGLUPackedOrder))
@pytest.mark.parametrize("input_mode", ("random", "constant"))
def test_swiglu_packed_case_preserves_operands_and_declares_input_gradient(order, input_mode):
    args = _args(op="swiglu", input_mode=input_mode)
    separate = make_operator_case(args, torch.float32, torch.device("cpu"))
    args.packed_order = order.value
    packed = make_operator_case(args, torch.float32, torch.device("cpu"))

    assert separate.grad_input_names == ("gate", "up")
    assert packed.grad_input_names == ("input",)
    assert packed.inputs.keys() == {"input", "packed_order"}
    assert packed.inputs["packed_order"] is order
    projected = packed.inputs["input"]
    gate, up = separate.inputs["gate"], separate.inputs["up"]
    assert projected.shape == (*gate.shape[:-1], 2 * gate.shape[-1])
    assert projected.is_contiguous()
    first, second = projected.chunk(2, dim=-1)
    if order is SwiGLUPackedOrder.GATE_UP:
        assert torch.equal(first, gate)
        assert torch.equal(second, up)
    else:
        assert torch.equal(first, up)
        assert torch.equal(second, gate)
    shape = "x".join(str(size) for size in projected.shape)
    assert packed.name.endswith(f"{shape}-packed-{order.value}")
    assert packed.name != separate.name


@pytest.mark.parametrize("candidate", ("cuda", "ascend"))
def test_swiglu_packed_rejects_candidates_without_packed_support(candidate):
    args = _args(op="swiglu", candidate=candidate, packed_order="up_gate")
    with pytest.raises(ValueError, match="packed SwiGLU requires a pytorch or triton candidate"):
        make_candidate(args)


def test_cp_attention_operator_spec_registers_backward_grad_inputs():
    args = _args(op="cp_attention", input_mode="constant", batch=1, seq=2)

    assert "cp_attention" in operator_names()
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    candidate = make_candidate(argparse.Namespace(**{**vars(args), "candidate": "pytorch"}))

    assert case.op_class == "attention"
    assert case.grad_input_names == ("q", "k", "v")
    assert candidate.name == "pytorch-cp_attention"


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
