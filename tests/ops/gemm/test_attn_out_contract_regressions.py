# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regressions for input parity, empty reductions and frozen parameters."""

import importlib
from contextlib import ExitStack
from itertools import product
from unittest.mock import patch

import pytest
import torch

from rl_engine.reference.gemm.attn_out_bias_gemm import NativeAttnOutBiasGemmOp


@pytest.fixture(params=["pytorch", "triton", "cuda"])
def backend(request):
    name = request.param
    if name == "pytorch":
        module = importlib.import_module("rl_engine.reference.gemm.attn_out_bias_gemm")
        return name, NativeAttnOutBiasGemmOp(), "cpu", module
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if name == "triton":
        module = importlib.import_module("rl_engine.backends.shared.triton.gemm.attn_out_bias_gemm")
        if not module._TRITON_AVAILABLE:
            pytest.skip("Triton unavailable")
        op = module.TritonAttnOutBiasGemmOp()
    else:
        module = importlib.import_module("rl_engine.backends.cuda.gemm.attn_out_bias_gemm")
        if not module._cuda_backend_available():
            pytest.skip("CUDA extension unavailable")
        op = module.CudaAttnOutBiasGemmOp()
    return name, op, "cuda", module


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    bit_dtype = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(
        actual.detach().cpu().contiguous().view(bit_dtype),
        expected.detach().cpu().contiguous().view(bit_dtype),
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "fp16",
        "fp64",
        "weight_dtype",
        "bias_dtype",
        "bias_scalar",
        "bias_row",
        "bias_length",
        "weight_rank",
        "input_scalar",
        "input_width",
        "zero_k",
        "zero_n",
    ],
)
def test_public_input_contract(backend, invalid):
    _, op, device, _ = backend
    x = torch.ones(2, 4, device=device)
    weight = torch.ones(3, 4, device=device)
    bias = torch.ones(3, device=device)
    if invalid in ("fp16", "fp64"):
        dtype = torch.float16 if invalid == "fp16" else torch.float64
        x, weight, bias = (t.to(dtype) for t in (x, weight, bias))
    elif invalid == "weight_dtype":
        weight = weight.bfloat16()
    elif invalid == "bias_dtype":
        bias = bias.bfloat16()
    elif invalid == "bias_scalar":
        bias = torch.ones((), device=device)
    elif invalid == "bias_row":
        bias = bias.unsqueeze(0)
    elif invalid == "bias_length":
        bias = bias[:1]
    elif invalid == "weight_rank":
        weight = weight.unsqueeze(0)
    elif invalid == "input_scalar":
        x = torch.ones((), device=device)
    elif invalid == "input_width":
        x = x[:, :2]
    elif invalid == "zero_k":
        x, weight = x[:, :0], weight[:, :0]
    elif invalid == "zero_n":
        weight, bias = weight[:0], bias[:0]
    with pytest.raises(ValueError):
        op(x, weight, bias=bias)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("needs_grad", list(product([False, True], repeat=3))[1:])
@pytest.mark.parametrize("rows", [0, 3])
def test_only_requested_gradients_are_computed(backend, dtype, needs_grad, rows):
    name, op, device, module = backend
    generator = torch.Generator().manual_seed(459)
    values = [
        torch.randn(shape, generator=generator).to(dtype) for shape in ((rows, 5), (4, 5), (4,))
    ]
    upstream = torch.randn(rows, 4, generator=generator).to(dtype)
    gold_inputs = [t.clone().requires_grad_() for t in values]
    NativeAttnOutBiasGemmOp()(gold_inputs[0], gold_inputs[1], bias=gold_inputs[2]).backward(
        upstream
    )
    inputs = [
        t.to(device).detach().requires_grad_(needs)
        for t, needs in zip(values, needs_grad, strict=True)
    ]
    out = op(inputs[0], inputs[1], bias=inputs[2])

    vjp = importlib.import_module("rl_engine.ops.autograd.vjp_fp32")
    if name == "pytorch":
        targets = [
            (module, "tree_gemm_fp32"),
            (module, "_dw_left_fold_reference"),
            (module, "row_local_bias_fp32"),
        ]
    elif name == "triton":
        targets = [
            (module, "triton_tree_gemm"),
            (module, "triton_dw_left_fold"),
            (vjp, "row_local_bias_fp32"),
        ]
    else:
        targets = [
            (module._C, "attn_out_tree_gemm_cuda"),
            (module._C, "attn_out_dw_left_fold_cuda"),
            (vjp, "row_local_bias_fp32"),
        ]
    with ExitStack() as stack:
        spies = [
            stack.enter_context(patch.object(owner, attr, wraps=getattr(owner, attr)))
            for owner, attr in targets
        ]
        out.backward(upstream.to(device))
        assert [spy.call_count for spy in spies] == [int(needs) for needs in needs_grad]

    for actual, expected, needs in zip(inputs, gold_inputs, needs_grad, strict=True):
        if needs:
            assert_bits(actual.grad, expected.grad)
        else:
            assert actual.grad is None


@pytest.mark.parametrize("shape", [(3, 4, 0), (0, 4, 5), (3, 0, 5), (0, 0, 0)])
def test_low_level_empty_tree_gemm(backend, shape):
    name, _, device, module = backend
    rows, cols, reduction = shape
    a = torch.empty(rows, reduction, device=device)
    b = torch.empty(cols, reduction, device=device)
    if name == "cuda":
        result = module._C.attn_out_tree_gemm_cuda(a, b)
    elif name == "triton":
        result = module.triton_tree_gemm(a, b)
    else:
        result = module.tree_gemm_fp32(a, b)
    assert_bits(result, torch.zeros(rows, cols))


@pytest.mark.parametrize("bf16_out", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_cuda_empty_reduction_keeps_bias_and_output_dtype(backend, bf16_out, with_bias):
    name, _, device, module = backend
    if name != "cuda":
        pytest.skip("Direct CUDA forward binding regression")
    x = torch.empty(2, 0, device=device)
    weight = torch.empty(3, 0, device=device)
    bias = torch.tensor([-0.0, 1.003, -2.007], device=device) if with_bias else None
    result = module._C.attn_out_bias_gemm_cuda_forward(x, weight, bias, bf16_out)
    expected = torch.zeros(2, 3, device=device)
    if bias is not None:
        expected = expected + bias
    if bf16_out:
        expected = expected.bfloat16()
    assert_bits(result, expected)
