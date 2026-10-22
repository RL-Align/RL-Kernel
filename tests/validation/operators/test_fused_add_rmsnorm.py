# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared validation harness coverage for both outputs and all input gradients."""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backend", ["pytorch", "triton"])
def test_shared_harness_checks_both_outputs_and_all_input_gradients(dtype, backend):
    import argparse

    from rl_engine.validation.operators.op_checks import run_operator_suite
    from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

    if backend == "triton":
        if not torch.cuda.is_available():
            pytest.skip("A CUDA or ROCm GPU is required")
        triton = pytest.importorskip("triton")
        if not hasattr(triton, "jit"):
            pytest.skip("A working Triton runtime is required; type stubs are insufficient")
    args = argparse.Namespace(
        op="fused_add_rmsnorm",
        candidate=backend,
        arch_key=None,
        batch=2,
        seq=3,
        normalized_dim=129,
        eps=1e-5,
        seed=434,
        input_mode="random",
    )
    device = torch.device("cuda" if backend == "triton" else "cpu")
    case = make_operator_case(args, dtype, device)
    assert case.inputs["x"].shape == case.inputs["residual"].shape == (2, 3, 129)
    assert case.inputs["weight"].shape == (129,)
    assert case.inputs["weight"].dtype == torch.float32
    assert case.grad_input_names == ("x", "residual", "weight")
    report = run_operator_suite(
        args.op,
        candidates=[make_candidate(args)],
        cases=[case],
        check_grad=True,
    )
    assert report.passed
    assert [len(check.outputs) for check in report.candidates[0].cases] == [5]
