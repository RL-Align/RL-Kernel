# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
import pytest
import torch

from benchmarks.operators.attention.benchmark_deterministic_attention import H3StrictNative
from benchmarks.operators.attention.benchmark_deterministic_attention import (
    h3_bits_equal as bits_equal,
)
from benchmarks.operators.attention.benchmark_deterministic_attention import (
    h3_value_grad as value_grad,
)
from rl_engine.backends.cuda.attention.h3_full_attention import _C, H3FullAttentionOp
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.attention.standard_attn import H3FullAttentionReference


def case_data(b, s, seed):
    torch.manual_seed(seed)
    return [
        torch.randn(b, s, 56, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(3)
    ]


def assert_accuracy(actual, reference):
    contract = load_contract()
    for i, (a, g) in enumerate(zip(actual, reference)):
        tol = resolve_tolerance(
            contract,
            op_class="attention",
            dtype="bfloat16",
            judgment="forward_accuracy" if i == 0 else "gradient_accuracy",
            backend_profile="cuda_bf16",
        )
        torch.testing.assert_close(a.to(g.dtype), g, atol=tol.atol, rtol=tol.rtol)


@pytest.fixture
def gpu_op():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if torch.version.hip is not None or _C is None:
        pytest.skip("requires the optional strict NVIDIA CUDA extension")
    assert _C.h3_attention_build_info()["profile"] == "h3-cuda-strict-v1"
    return H3FullAttentionOp


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_chunk(value):
    with pytest.raises(ValueError, match="positive integer"):
        H3FullAttentionOp(value)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 2, 55, 128), torch.bfloat16),
        ((1, 2, 56, 128), torch.float32),
        ((1, 0, 56, 128), torch.bfloat16),
    ],
)
def test_invalid_profile(shape, dtype):
    q = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError):
        H3FullAttentionOp()(q, q, q)


def test_cpu_fails_closed():
    q = torch.empty(1, 2, 56, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        H3FullAttentionOp()(q, q, q)


@pytest.mark.parametrize("platform", ["cpu", "rocm", "musa", "ascend"])
def test_registry_rejects_unsupported_platform(monkeypatch, platform):
    from rl_engine.runtime.registry import kernel_registry

    monkeypatch.setattr(kernel_registry, "_platform_for_device", lambda _: platform)
    with pytest.raises(RuntimeError, match="no fallback"):
        kernel_registry.get_op("h3_full_attention", device="cpu")


@pytest.mark.parametrize(
    "seq,chunk,amplitude",
    [
        (1, 1, 1),
        (17, 16, 1),
        (65, 64, 1),
        (129, 128, 1),
        (257, 128, 1),
        (1024, 128, 1),
        (1025, 128, 4),
        (4096, 128, 1),
    ],
)
def test_accuracy_and_chunk_invariance(gpu_op, seq, chunk, amplitude):
    xs = [(x.detach() * amplitude).requires_grad_() for x in case_data(1, seq, 100 + seq)]
    dy = torch.randn_like(xs[0])
    op = gpu_op(chunk)
    actual = value_grad(op, xs, dy)
    golden = H3FullAttentionReference().value_and_grad(*xs, dy)
    references = [golden]
    if seq <= 257:

        def fp64(q, k, v):
            q, k, v = (x.double().transpose(1, 2) for x in (q, k, v))
            return ((q @ k.transpose(-1, -2) / 128**0.5).softmax(-1) @ v).transpose(1, 2)

        references.append(
            value_grad(fp64, [x.detach().double().requires_grad_() for x in xs], dy.double())
        )
    for reference in references:
        assert_accuracy(actual, reference)
    other = value_grad(gpu_op(257), xs, dy)
    assert all(bits_equal(a, b) for a, b in zip(actual, other))
    if _C is not None and torch.version.hip is None:
        native = value_grad(lambda *x: H3StrictNative.apply(*x)[0], xs, dy)
        assert all(bits_equal(a, b) for a, b in zip(actual, native))
        _, lse = op.forward_with_lse(*xs)
        _, native_lse = H3StrictNative.apply(*xs)
        assert bits_equal(lse, native_lse)


def test_batch_checkpoint_and_stream_invariance(gpu_op):
    from torch.utils.checkpoint import checkpoint

    op = gpu_op(64)
    xs = case_data(3, 129, 781)
    dy = torch.randn_like(xs[0])
    expected = value_grad(op, xs, dy)
    for inputs, grad, golden in (
        (xs, dy, expected),
        ([x[1:2].detach().requires_grad_() for x in xs], dy[1:2], [x[1:2] for x in expected]),
    ):
        actual = value_grad(op, inputs, grad)
        assert all(bits_equal(a, b) for a, b in zip(golden, actual))
    perm = torch.tensor([2, 0, 1], device="cuda")
    actual = value_grad(op, [x[perm].detach().requires_grad_() for x in xs], dy[perm])
    assert all(bits_equal(a[perm], b) for a, b in zip(expected, actual))
    unrelated = [x.detach().clone().requires_grad_() for x in xs]
    with torch.no_grad():
        for x in unrelated:
            x[0].fill_(7)
            x[2].fill_(-3)
    changed = value_grad(op, unrelated, dy)
    assert all(bits_equal(a[1], b[1]) for a, b in zip(expected, changed))
    # Real strided views, not merely a logically permuted contiguous copy.
    strided = []
    for x in xs:
        backing = torch.empty(3, 258, 56, 128, device="cuda", dtype=torch.bfloat16)
        view = backing[:, ::2]
        view.copy_(x.detach())
        strided.append(view.requires_grad_())
    with pytest.raises(ValueError, match="contiguous"):
        op(*strided)
    actual = value_grad(lambda *args: checkpoint(op, *args, use_reentrant=False), xs, dy)
    assert all(bits_equal(a, b) for a, b in zip(expected, actual))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = value_grad(op, xs, dy)
    torch.cuda.current_stream().wait_stream(stream)
    assert all(bits_equal(a, b) for a, b in zip(expected, actual))


@pytest.mark.parametrize("seq,chunk", [(1, 1), (7, 3), (17, 16)])
def test_bounded_reference_vjp(seq, chunk):
    torch.manual_seed(241)
    xs = [torch.randn(1, seq, 56, 128, requires_grad=True) for _ in range(3)]
    dy = torch.randn_like(xs[0])
    ref = H3FullAttentionReference()
    out = ref(*xs)
    expected = [out, *torch.autograd.grad(out, xs, dy)]
    for actual, golden in zip(ref.value_and_grad(*xs, dy, chunk), expected):
        torch.testing.assert_close(actual, golden, atol=2e-6, rtol=2e-5)


def test_gtest_registration():
    from argparse import Namespace

    from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

    args = Namespace(op="h3_full_attention", candidate="cuda", arch_key="sm90", batch=1, seq=7)
    case = make_operator_case(args, torch.bfloat16, torch.device("cpu"))
    assert case.inputs["q"].shape == (1, 7, 56, 128)
    assert isinstance(make_candidate(args).fn, H3FullAttentionOp)
    assert case.gold_fn(**case.inputs).dtype == torch.float32


def test_strict_build_flags(monkeypatch):
    from build_tools import envs

    for name in (
        "RL_KERNEL_H3_STRICT",
        "KERNEL_ALIGN_USE_FAST_MATH",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert envs.h3_strict_cuda_flags() == []
    monkeypatch.setenv("RL_KERNEL_H3_STRICT", "1")
    assert set(envs.h3_strict_cuda_flags()) == {
        "-DRL_KERNEL_H3_STRICT=1",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    }
    for name, value in (
        ("KERNEL_ALIGN_USE_FAST_MATH", "1"),
        ("NVCC_PREPEND_FLAGS", "--use_fast_math"),
        ("NVCC_APPEND_FLAGS", "--fmad=true"),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            envs.h3_strict_cuda_flags()
        monkeypatch.delenv(name)


@pytest.fixture
def extension_config(monkeypatch):
    from build_tools import extensions

    for name in (
        "RL_KERNEL_H3_STRICT",
        "KERNEL_ALIGN_USE_FAST_MATH",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(extensions, "_load_torch_extension_tools", lambda: (torch, None, dict))
    monkeypatch.setattr(extensions, "_musa_build_available", lambda _: False)
    monkeypatch.setattr(extensions, "_ascend_extensions", lambda: [])
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    return extensions.get_extensions


def test_strict_extension_isolation(monkeypatch, extension_config):
    baseline = extension_config()
    monkeypatch.setenv("RL_KERNEL_H3_STRICT", "1")
    ordinary, h3 = extension_config()
    assert baseline == [ordinary]
    assert ordinary["name"] == "rl_engine._C"
    assert h3["name"] == "rl_engine._H3_C"
    assert "-DRL_KERNEL_H3_STRICT=1" not in ordinary["extra_compile_args"]["nvcc"]
    flags = h3["extra_compile_args"]["nvcc"]
    assert "-DRL_KERNEL_H3_STRICT=1" in flags
    assert "--fmad=false" in flags and "--ftz=false" in flags
    assert "--use_fast_math" not in flags


def test_strict_extension_rejects_rocm(monkeypatch, extension_config):
    monkeypatch.setenv("RL_KERNEL_H3_STRICT", "1")
    monkeypatch.setattr(torch.version, "hip", "test")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA only"):
        extension_config()


def test_unqualified_extension_rejected(monkeypatch):
    from types import SimpleNamespace

    from rl_engine.backends.cuda.attention import h3_full_attention as h3

    fake = SimpleNamespace(
        h3_attention_forward=lambda: None,
        h3_attention_backward_accumulate=lambda: None,
        h3_attention_probabilities=lambda: None,
        h3_attention_build_info=lambda: {"profile": "unqualified"},
    )
    monkeypatch.setattr(h3, "_C", fake)
    with pytest.raises(RuntimeError, match="RL_KERNEL_H3_STRICT"):
        h3._extension()


@pytest.mark.parametrize(
    "device,dtype,world,cp",
    [
        ("cuda", "bfloat16", 1, 1),
        ("cpu", "bfloat16", 1, 1),
        ("rocm", "bfloat16", 1, 1),
        ("cuda", "float16", 1, 1),
        ("cuda", "bfloat16", 2, 1),
        ("cuda", "bfloat16", 1, 2),
    ],
)
def test_semantic_dispatch_contract(device, dtype, world, cp):
    from rl_engine.runtime.registry import kernel_registry
    from rl_engine.runtime.semantic_registry import OperatorRequirements, OperatorResolutionError

    session = kernel_registry.semantic.session()
    kwargs = dict(
        semantic_op="h3_full_attention",
        requested_backend=H3FullAttentionOp.backend_id,
        target="training",
        requirements=OperatorRequirements(
            device=device,
            dtype=dtype,
            topology={"world_size": world, "tensor_parallel_size": 1, "context_parallel_size": cp},
            alignment_properties={"deterministic": True},
        ),
    )
    if (device, dtype, world, cp) == ("cuda", "bfloat16", 1, 1):
        resolution = session.resolve(**kwargs)
        instance = session.instantiate(resolution)
        assert isinstance(instance, H3FullAttentionOp)
        assert (
            session.instance_provenance(resolution, instance).backend_id
            == H3FullAttentionOp.backend_id
        )
    else:
        with pytest.raises(OperatorResolutionError):
            session.resolve(**kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)
def test_cuda_build_and_invalid_inputs():
    if _C is None:
        with pytest.raises(RuntimeError, match="extension unavailable"):
            H3FullAttentionOp()(*case_data(1, 17, 9))
        return
    assert hasattr(_C, "h3_attention_build_info")
    xs = case_data(1, 17, 9)
    op = H3FullAttentionOp()
    if _C.h3_attention_build_info()["profile"] != "h3-cuda-strict-v1":
        with pytest.raises(RuntimeError, match="RL_KERNEL_H3_STRICT"):
            op(*xs)
        with pytest.raises(RuntimeError, match="RL_KERNEL_H3_STRICT"):
            _C.h3_attention_forward(*(x.transpose(1, 2) for x in xs))
        return
    for bad in (xs[1].cpu(), xs[1].float(), xs[1][..., :64], xs[1][:, :0]):
        with pytest.raises(ValueError):
            op(xs[0], bad, xs[2])
    q, k, v = [x.transpose(1, 2).contiguous() for x in xs]
    with pytest.raises(RuntimeError):
        _C.h3_attention_probabilities(q, k[:, :0], v[:, :0])
    p = _C.h3_attention_probabilities(q, k, v)
    acc = torch.zeros_like(k, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _C.h3_attention_backward_accumulate(q, q, k, v, p, acc, acc)
