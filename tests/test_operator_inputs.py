# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.kernels.gtest.operator_inputs import make_operator_inputs, operator_shape_name


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
        "k_dim": 16,
        "n_dim": 32,
        "theta": 1.0e6,
        "eps": 1.0e-6,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "op_name",
    [
        "rms_norm",
        "matmul",
        "attention",
        "logp",
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


def test_random_logp_inputs_are_seeded():
    args = _args(input_mode="random", seed=7)
    first = make_operator_inputs("logp", args, torch.float32, torch.device("cpu"))
    second = make_operator_inputs("logp", args, torch.float32, torch.device("cpu"))

    assert torch.equal(first["logits"], second["logits"])
    assert torch.equal(first["token_ids"], second["token_ids"])
