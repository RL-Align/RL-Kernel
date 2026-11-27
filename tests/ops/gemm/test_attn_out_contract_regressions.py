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


def test_bias_is_added_before_bf16_output_cast(backend):
    _, op, device, _ = backend
    x = torch.tensor([[1 + 2**-7]], dtype=torch.bfloat16, device=device)
    weight = x.clone()
    bias = torch.tensor([2**-8], dtype=torch.bfloat16, device=device)

    # The exact dot product is 1 + 2^-6 + 2^-14. Its low bit puts the
    # biased result above the BF16 midpoint. Casting before bias loses
    # that bit and rounds the tie down to 1.015625 instead.
    expected = torch.tensor([[1.0234375]], dtype=torch.bfloat16)
    assert_bits(op(x, weight, bias=bias), expected)


def test_frozen_inference_bias_allows_input_gradient(backend):
    _, op, device, _ = backend
    x = torch.tensor([[2.0]], device=device, requires_grad=True)
    weight = x.new_tensor([[3.0]])
    with torch.inference_mode():
        bias = x.new_tensor([4.0])

    output = op(x, weight, bias=bias)
    assert_bits(output, x.new_tensor([[10.0]]))
    (dx,) = torch.autograd.grad(output, x, grad_outputs=x.new_tensor([[-2.0]]))
    assert_bits(dx, x.new_tensor([[-6.0]]))


@pytest.mark.parametrize("train_bias", [False, True])
def test_bias_update_after_forward_preserves_gradients(backend, train_bias):
    _, op, device, _ = backend
    x = torch.tensor([[2.0], [-1.0]], device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = x.new_tensor([[3.0]])
    bias = x.new_tensor([4.0], requires_grad=train_bias)
    output = op(x, weight, bias=bias)
    with torch.no_grad():
        bias.add_(8.0)

    assert_bits(output, x.new_tensor([[10.0], [1.0]]))
    output.backward(x.new_tensor([[2.0], [-1.0]]))
    assert_bits(x.grad, x.new_tensor([[6.0], [-3.0]]))
    if train_bias:
        assert_bits(bias.grad, x.new_tensor([1.0]))
    else:
        assert bias.grad is None


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
        if name in ("triton", "cuda") and needs_grad[2]:
            bias_upstream = spies[2].call_args.args[0]
            assert bias_upstream.dtype == torch.float32
            for spy in spies[:2]:
                if spy.called:
                    assert spy.call_args.args[0] is bias_upstream

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


@pytest.mark.parametrize("check_grad", [False, True])
def test_acceptance_reports_candidate_execution(backend, check_grad):
    from argparse import Namespace

    from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
    from rl_engine.validation.operators import run_operator_suite
    from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

    family, op, device, _ = backend
    args = Namespace(
        op="attn_out_bias_gemm",
        candidate=family,
        arch_key=None,
        batch=1,
        seq=3,
        normalized_dim=7,
        seed=386,
    )
    candidate = make_candidate(args, observe_backend=True)
    case = make_operator_case(args, torch.float32, torch.device(device))
    report = run_operator_suite(
        args.op, candidates=[candidate], cases=[case], check_grad=check_grad
    )
    assert report.passed
    trace = report.candidates[0].cases[0].backend_trace
    assert trace["actual_backend"] == family
    assert trace["fallback"] is False and trace["fallback_reason"] is None
    if check_grad:
        assert trace["backward_backend"]["family"] == family
        assert trace["backward_backend"]["impl"] == op.backward_impl
        assert snapshot_backward_runtime()[args.op]["family"] == family
    else:
        assert "backward_backend" not in trace


@pytest.mark.skipif(torch.version.cuda is None, reason="requires an NVIDIA CUDA build")
def test_registry_reports_rejections_only_for_this_selection(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    reference = NativeAttnOutBiasGemmOp()
    monkeypatch.setattr(
        registry,
        "_get_or_create_backend",
        lambda backend: reference if backend == OpBackend.PYTORCH_ATTN_OUT_BIAS_GEMM else None,
    )
    selected = registry.get_op("attn_out_bias_gemm", device="cuda")
    assert selected.trace["actual_backend"] == "pytorch"
    assert selected.trace["fallback"] is True
    for rejected in registry._priority_map[registry._platform_for_device("cuda")][
        "attn_out_bias_gemm"
    ][:-1]:
        assert rejected.name in selected.trace["fallback_reason"]
    assert registry.get_op("attn_out_bias_gemm", device="cpu").trace["fallback"] is False
    assert reference.trace["fallback"] is False


def _negative_view(shape, value, device):
    tensor = torch.full(shape, complex(0, value), dtype=torch.complex64, device=device)
    view = tensor.conj().imag
    assert view.is_neg() and view.is_contiguous()
    return view


@pytest.mark.parametrize("negative", ["x", "weight", "bias", "upstream"])
def test_contiguous_negative_view_forward_and_backward(backend, negative):
    _, op, device, _ = backend
    values = {}
    for name, shape, value in [
        ("x", (1, 1), 2),
        ("weight", (1, 1), 3),
        ("bias", (1,), 1),
        ("upstream", (1, 1), 5),
    ]:
        if name == negative:
            values[name] = _negative_view(shape, value, device)
        else:
            values[name] = torch.full(shape, float(value), device=device)
    x, w, b = (values[name].requires_grad_() for name in ("x", "weight", "bias"))
    g = values["upstream"]
    y = op(x, w, bias=b)
    dx, dw, db = torch.autograd.grad(y, (x, w, b), g)
    assert y.item() == x.item() * w.item() + b.item()
    assert [dx.item(), dw.item(), db.item()] == [g.item() * w.item(), g.item() * x.item(), g.item()]


def test_native_bindings_read_logical_negative_views():
    from rl_engine.backends.cuda.gemm.attn_out_bias_gemm import _C, _cuda_backend_available

    if not torch.cuda.is_available() or not _cuda_backend_available():
        pytest.skip("requires native CUDA attn-out bindings")
    a = _negative_view((1, 1), 2, "cuda")
    b = torch.full_like(a, 3)
    bias = _negative_view((1,), 1, "cuda")
    assert _C.attn_out_bias_gemm_cuda_forward(a, b, bias, False).item() == -7
    assert _C.attn_out_bias_gemm_cuda_forward(b, a, None, False).item() == -6
    for fn in (_C.attn_out_tree_gemm_cuda, _C.attn_out_dw_left_fold_cuda):
        assert fn(a, b).item() == -6
        assert fn(b, a).item() == -6


def test_generic_cpu_reference_recovers_finite_bf16_after_cancellation():
    import os
    import subprocess
    import sys
    import textwrap

    # ATen selects its CPU dispatch path at process startup.
    script = textwrap.dedent(
        """
        import torch
        from rl_engine.reference.gemm.attn_out_bias_gemm import NativeAttnOutBiasGemmOp

        m = torch.finfo(torch.bfloat16).max
        x = torch.tensor([[-m, m]], dtype=torch.bfloat16)
        w = torch.tensor([[1., 2.]], dtype=x.dtype)
        op = NativeAttnOutBiasGemmOp()
        assert op(x, w).item() == m
        weight = torch.ones((1, 1), dtype=x.dtype, requires_grad=True)
        dw, = torch.autograd.grad(op(x.T, weight), weight, grad_outputs=w.T)
        assert dw.item() == m
    """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, ATEN_CPU_CAPABILITY="default"),
        check=True,
    )


@pytest.mark.parametrize(
    "dtype,bit_dtype,expected",
    [
        (torch.float32, torch.int32, 0x00400000),
        (torch.bfloat16, torch.int16, 0x0040),
    ],
)
def test_cpu_reference_requires_gradual_underflow(dtype, bit_dtype, expected):
    tiny = torch.tensor([[2.0**-126]], dtype=torch.float32, device="cpu")
    was_flushing = (tiny * 0.5).item() == 0.0
    if not torch.set_flush_denormal(False):
        pytest.skip("CPU does not support changing denormal handling")
    try:
        x = tiny.to(dtype)
        w = torch.full_like(x, 0.5, requires_grad=True)
        bias = x.new_zeros(1, requires_grad=True)
        op = NativeAttnOutBiasGemmOp()
        y = op(x, w)
        bias_only = op(x, w.detach(), bias=bias)
        assert y.view(bit_dtype).item() == expected
        assert op.forward_fp32(x, w).view(torch.int32).item() == 0x00400000
        (dw,) = torch.autograd.grad(y, w, torch.full_like(y, 0.5), retain_graph=True)
        assert dw.view(bit_dtype).item() == expected

        torch.set_flush_denormal(True)
        error = r"attn_out_bias_gemm CPU reference requires gradual underflow"
        with pytest.raises(RuntimeError, match=error):
            op(x, w)
        with pytest.raises(RuntimeError, match=error):
            op.forward_fp32(x, w)
        with pytest.raises(RuntimeError, match=error):
            torch.autograd.grad(y, w, torch.full_like(y, 0.5))
        with pytest.raises(RuntimeError, match=error):
            torch.autograd.grad(bias_only, bias, torch.ones_like(bias_only))
        assert (tiny * 0.5).item() == 0.0
    finally:
        torch.set_flush_denormal(was_flushing)


def test_registry_limits_attn_out_to_cuda_and_cpu(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    registry = KernelRegistry()
    name = "attn_out_bias_gemm"
    assert {key for key, ops in registry._priority_map.items() if name in ops} == {"cuda"}
    assert isinstance(registry.get_op(name, device="cpu"), NativeAttnOutBiasGemmOp)
    monkeypatch.setattr(torch.version, "hip", "test")
    with pytest.raises(RuntimeError, match=name):
        registry.get_op(name, device="cuda")
    with pytest.raises(RuntimeError, match=name):
        registry.get_op(name, device="meta")
