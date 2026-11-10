# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse

import pytest
import torch

from rl_engine.validation.operators.operator_inputs import make_operator_inputs, operator_shape_name
from rl_engine.validation.operators.operator_specs import (
    make_candidate,
    make_operator_case,
    operator_names,
)
from tests.helpers.fused_add_norm_tolerance import assert_within


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
        "fused_post_norm_residual_pre_norm",
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_the_fused_add_norm_gold_is_checked_against_an_independent_recomputation(dtype):
    """The harness's gold for [F1] is the pytorch candidate, so this checks it elsewhere.

    Every spec in ``operator_specs.py`` names the same class as its gold and as its
    ``pytorch`` candidate, and the others get an independent comparison out of it by
    setting ``gold_method="forward_fp32"`` while the candidate is called through
    ``forward``. This operator cannot: ``forward_fp32`` is a different computation
    rather than ``forward`` widened, because the frozen semantics narrow twice and
    ``forward_fp32`` skips both, so the gold is ``forward`` and the ``pytorch`` row
    of the harness compares that method against itself. That row is a tautology and
    saying so is better than implying otherwise.

    What it would miss is a bug in the narrowing, the ``1 + w`` gain or the eps
    pair, which would then only surface as a CUDA or Triton mismatch against a gold
    that is itself wrong. So that is checked here, against an FP64 recomputation
    written out in this file, on the harness's own inputs rather than on shapes
    chosen by a unit test. ``tests/ops/norm/test_fused_add_norm.py`` holds the same
    comparison with negative controls; this is the part that follows the harness.

    The second comparison exists because the first one cannot see the eps. At the
    model's values, 1e-8 and 1e-5 against a mean of squares near 1, swapping them
    moves ``rstd`` by about five parts in a million, which is well inside the
    forward row: measured, the swap leaves this test green. Running the same
    comparison again with 0.5 and 1e-9 makes the two distinguishable, and the swap
    then fails it.
    """
    from rl_engine.reference.norm.fused_add_norm import NativeFusedAddNormOp

    inputs = make_operator_inputs(
        "fused_post_norm_residual_pre_norm",
        _args(input_mode="random", normalized_dim=128, seq=8),
        dtype,
        torch.device("cpu"),
    )
    got_res, got_pre = NativeFusedAddNormOp().forward(
        inputs["hidden"],
        inputs["residual"],
        inputs["post_weight"],
        inputs["pre_weight"],
        post_eps=inputs["post_eps"],
        pre_eps=inputs["pre_eps"],
    )

    # Written out rather than borrowed: mean of squares in FP64, rsqrt, the 1 + w
    # gain, and a cast at each of the two declared narrowing points.
    def norm64(x, weight, eps):
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * rstd * (1.0 + weight.double())

    hidden64 = inputs["hidden"].double()
    post = norm64(hidden64, inputs["post_weight"], inputs["post_eps"])
    residual_out = (inputs["residual"].double() + post.to(dtype).double()).to(dtype)
    pre_normed = norm64(residual_out.double(), inputs["pre_weight"], inputs["pre_eps"]).to(dtype)

    assert_within("forward_accuracy", dtype, got_res, residual_out.double(), "residual_out: ")
    assert_within("forward_accuracy", dtype, got_pre, pre_normed.double(), "pre_normed: ")

    post_eps, pre_eps = 0.5, 1e-9
    got_res2, got_pre2 = NativeFusedAddNormOp().forward(
        inputs["hidden"],
        inputs["residual"],
        inputs["post_weight"],
        inputs["pre_weight"],
        post_eps=post_eps,
        pre_eps=pre_eps,
    )
    post2 = norm64(hidden64, inputs["post_weight"], post_eps)
    residual_out2 = (inputs["residual"].double() + post2.to(dtype).double()).to(dtype)
    pre_normed2 = norm64(residual_out2.double(), inputs["pre_weight"], pre_eps).to(dtype)
    assert_within(
        "forward_accuracy", dtype, got_res2, residual_out2.double(), "residual_out, wide eps: "
    )
    assert_within(
        "forward_accuracy", dtype, got_pre2, pre_normed2.double(), "pre_normed, wide eps: "
    )
