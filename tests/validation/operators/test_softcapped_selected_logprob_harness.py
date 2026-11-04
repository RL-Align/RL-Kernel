# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared operator harness coverage for softcapped selected logprob."""

import argparse

import pytest
import torch

from rl_engine.validation.operators.op_checks import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("backend", ("pytorch", "triton"))
def test_shared_accuracy_harness(dtype, backend):
    if backend == "triton" and not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    device = torch.device("cuda" if backend == "triton" else "cpu")
    args = argparse.Namespace(
        op="softcapped_selected_logprob",
        candidate=backend,
        arch_key=None,
        batch=2,
        seq=3,
        vocab=1025,
        seed=415,
        input_mode="random",
    )
    case = make_operator_case(args, dtype, device)
    assert case.inputs["logits"].shape == (6, 1025)
    assert case.inputs["token_ids"].shape == (6,)
    assert case.inputs["softcap"] == 30.0
    report = run_operator_suite(
        args.op,
        candidates=[make_candidate(args)],
        cases=[case],
        check_grad=True,
    )
    assert report.passed
