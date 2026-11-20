# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks of the benchmark's CLI contract and ordinary numerical baseline."""

import pytest
import torch

from benchmarks.operators.norm.benchmark_adaln_modulation import layer_norm_baseline, parse_args


def test_benchmark_cli_records_seed_json_and_rejects_empty_measurements():
    args = parse_args(["--seed", "386", "--json", "--repeat", "2"])
    assert args.seed == 386 and args.json and args.repeat == 2
    for argv in (["--repeat", "0"], ["--warmup", "-1"]):
        with pytest.raises(SystemExit) as exc:
            parse_args(argv)
        assert exc.value.code == 2


def test_layer_norm_baseline_keeps_affine_fp32_until_the_output_and_gate_vjp():
    x = torch.tensor([[[-3.0, 3.0]]], dtype=torch.bfloat16, requires_grad=True)
    m = torch.tensor([[-1.0, -1.0, 1 / 256, 1 / 256, 2.0, -3.0]], dtype=x.dtype, requires_grad=True)
    y, gate = layer_norm_baseline(x, m, eps=7.0)
    torch.testing.assert_close(
        y, torch.tensor([[[-1.75, -0.2470703125]]], dtype=x.dtype), atol=0, rtol=0
    )
    gate.backward(torch.tensor([[[1.0, -2.0]]], dtype=x.dtype))
    torch.testing.assert_close(m.grad, torch.tensor([[0, 0, 0, 0, 1, -2]], dtype=x.dtype))
