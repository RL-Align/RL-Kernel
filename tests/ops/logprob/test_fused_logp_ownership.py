# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU preservation tests for the CUDA/MUSA selected-logprob ownership boundary."""

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.backends.cuda.logprob import logp as cuda
from rl_engine.backends.musa.logprob import logp as musa
from rl_engine.ops.logprob import fused_logp as shared
from rl_engine.runtime.provenance.identity import stable_operator_path
from rl_engine.runtime.registry import KernelRegistry, OpBackend

ROOT = Path(__file__).resolve().parents[3]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@pytest.fixture(scope="module")
def before():
    # A frozen source oracle also works in shallow CI checkouts without the base object.
    source = Path(__file__).with_name("_logp_main_49941801.py").read_text()
    module = ModuleType("logp_before_task19")
    exec(compile(source, "<main-before-task19>", "exec"), module.__dict__)
    return module


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def fused_logp(self, logits, labels):
        self.calls.append(("fused_logp", (logits, labels)))
        return logits.float().log_softmax(-1).gather(1, labels[:, None]).squeeze(1).to(logits.dtype)

    def __getattr__(self, name):
        if not name.startswith("fused_logp_forward_"):
            raise AttributeError(name)

        def variant(logits, labels, *args):
            self.calls.append((name, (logits, labels, *args)))
            values = logits.float().log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
            indexed = "indexed" in name
            output = args[-1] if name.endswith("_out") else torch.zeros_like(values)
            if indexed:
                indices = args[0]
                output[indices] = values[indices].to(output.dtype)
            else:
                output.copy_(values)
            return output

        return variant


def construct(monkeypatch, module, cls_name, backend):
    monkeypatch.setattr(module, "_C", backend)
    monkeypatch.setattr(module, "_EXT_AVAILABLE", True)
    return getattr(module, cls_name)()


def inputs(dtype, layout):
    logits = torch.linspace(-2, 2, 2 * 3 * 7, dtype=dtype).reshape(2, 3, 7)
    if layout == "strided":
        logits = logits.transpose(0, 1)
    elif layout == "2d":
        logits = logits.reshape(-1, 7)
    labels = torch.arange(logits.numel() // 7, dtype=torch.int32).reshape(logits.shape[:-1])
    if layout == "strided":
        labels = labels.t().contiguous().t()
    return logits.detach().requires_grad_(), labels


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().view(torch.uint8),
        expected.detach().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize(
    "module,cls_name", [(cuda, "FusedLogpGenericOp"), (musa, "MusaFusedLogpOp")]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["2d", "3d", "strided"])
def test_forward_backward_preserves_main(monkeypatch, before, module, cls_name, dtype, layout):
    backend = RecordingBackend()
    op = construct(monkeypatch, module, cls_name, backend)
    old = construct(monkeypatch, before, "FusedLogpGenericOp", RecordingBackend())
    x, labels = inputs(dtype, layout)
    old_x = x.detach().clone().requires_grad_()
    result = op(x, labels)
    expected = old(old_x, labels)
    assert_bits(result, expected)
    upstream = torch.linspace(-1, 2, result.numel(), dtype=dtype).reshape(result.shape)
    result.backward(upstream)
    expected.backward(upstream)
    assert_bits(x.grad, old_x.grad)
    ref_x = x.detach().float().requires_grad_()
    reference = ref_x.log_softmax(-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    reference.backward(upstream.float())
    torch.testing.assert_close(x.grad, ref_x.grad.to(dtype), rtol=0.02, atol=0.002)
    call_x, call_labels = backend.calls[0][1]
    assert call_x.shape == (6, 7) and call_x.is_contiguous()
    assert call_labels.dtype == torch.long and call_labels.is_contiguous()
    assert call_labels.device == x.device
    assert_bits(x.detach(), old_x.detach())


def test_import_and_public_identities():
    legacy = importlib.import_module("rl_engine.kernels.ops.cuda.loss.logp")
    assert legacy is cuda
    assert legacy.FusedLogpGenericOp is cuda.FusedLogpGenericOp
    for cls in (cuda.FusedLogpGenericOp, cuda.FusedLogpSM90Op, cuda.DeterministicLogpCUDAOp):
        assert cls.__module__ == cuda.__name__
        assert stable_operator_path(object.__new__(cls)) == (
            f"rl_engine.kernels.ops.cuda.loss.logp.{cls.__name__}"
        )
    assert stable_operator_path(object.__new__(musa.MusaFusedLogpOp)) == (
        "rl_engine.backends.musa.logprob.logp.MusaFusedLogpOp"
    )
    assert not issubclass(musa.MusaFusedLogpOp, cuda.FusedLogpGenericOp)


@pytest.mark.parametrize("target", [shared.__name__, musa.__name__])
def test_import_isolation(target):
    script = """
import importlib, importlib.abc, sys, types
import torch
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("rl_engine.backends.cuda", "triton", "torch_musa")):
            raise AssertionError("unexpected dependency: " + fullname)
        if TARGET.endswith("fused_logp") and fullname == "rl_engine.backends.extension":
            raise AssertionError("shared module linked an extension")
sys.meta_path.insert(0, Block())
importlib.import_module(TARGET)
assert "rl_engine.backends.cuda.logprob.logp" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run(
        [sys.executable, "-c", script.replace("TARGET", repr(target))],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "module,cls_name", [(cuda, "FusedLogpGenericOp"), (musa, "MusaFusedLogpOp")]
)
def test_missing_extension(monkeypatch, module, cls_name):
    for available, extension in [(False, None), (True, SimpleNamespace())]:
        monkeypatch.setattr(module, "_C", extension)
        monkeypatch.setattr(module, "_EXT_AVAILABLE", available)
        with pytest.raises(RuntimeError, match="Base custom kernel 'fused_logp' is unavailable"):
            getattr(module, cls_name)()


def test_registry_construction_and_routing(monkeypatch):
    backend = RecordingBackend()
    for module in (cuda, musa):
        monkeypatch.setattr(module, "_C", backend)
        monkeypatch.setattr(module, "_EXT_AVAILABLE", True)
    registry = KernelRegistry()
    assert type(registry.get_op("logp", device="cuda")) is cuda.FusedLogpGenericOp
    # CPU PyTorch has no registered "musa" device name; exercise platform routing
    # without globally renaming PrivateUse1 or importing torch_musa.
    monkeypatch.setattr(registry, "_platform", lambda: "musa")
    assert type(registry.get_op("logp")) is musa.MusaFusedLogpOp
    from rl_engine.reference.logprob.logp import NativeLogpOp

    assert type(registry.get_op("logp", device="cpu")) is NativeLogpOp
    for kind in ("logp_indexed", "logp_online", "logp_online_indexed"):
        assert registry._priority_map["musa"][kind] == [OpBackend.PYTORCH_NATIVE]
    monkeypatch.setattr(cuda, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(musa, "_EXT_AVAILABLE", False)
    assert KernelRegistry()._get_or_create_backend(OpBackend.CUDA_FUSED_LOGP_GENERIC) is None
    assert KernelRegistry()._get_or_create_backend(OpBackend.MUSA_FUSED_LOGP_GENERIC) is None


@pytest.mark.parametrize(
    "module,cls_name", [(cuda, "FusedLogpGenericOp"), (musa, "MusaFusedLogpOp")]
)
def test_saved_state_and_version_checks(monkeypatch, module, cls_name):
    backend = RecordingBackend()
    op = construct(monkeypatch, module, cls_name, backend)
    x, labels = inputs(torch.float32, "strided")
    original_labels = labels.clone()
    result = op(x, labels)
    ctx = result.grad_fn
    saved_x, saved_labels = ctx.saved_tensors
    assert saved_x is backend.calls[0][1][0]
    assert saved_labels is backend.calls[0][1][1]
    assert ctx.backend is backend
    assert ctx.input_shape == tuple(x.shape) and ctx.input_dtype == x.dtype
    assert len(ctx.saved_tensors) == 2
    assert torch.equal(labels, original_labels)
    with torch.no_grad():
        saved_x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize(
    "module,bridge", [(cuda, "_FusedLogpAutograd"), (musa, "_MusaFusedLogpAutograd")]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_native_musa_backward_contract(monkeypatch, before, module, bridge, dtype):
    # This tests Python selection only; no MUSA tensor or native kernel is executed.
    logits = SimpleNamespace(device=SimpleNamespace(type="musa"))
    labels = torch.tensor([0, 1])
    upstream = torch.arange(4, dtype=dtype).reshape(2, 2).t()
    calls = []

    def native(*args):
        calls.append(args)
        return torch.arange(12, dtype=dtype).reshape(4, 3)

    backend = SimpleNamespace(fused_logp_backward=native)
    ctx = SimpleNamespace(
        saved_tensors=(logits, labels), backend=backend, input_shape=(2, 2, 3), input_dtype=dtype
    )
    result = getattr(module, bridge).backward(ctx, upstream)
    expected = before._FusedLogpAutograd.backward(ctx, upstream)
    assert_bits(result[0], expected[0])
    assert result[1:] == (None, None)
    assert calls[0][0] is logits and calls[0][1] is labels
    assert calls[0][2].dtype == dtype and calls[0][2].is_contiguous()
    assert_bits(calls[0][2], upstream.reshape(-1))


@pytest.mark.parametrize(
    "module,bridge", [(cuda, "_FusedLogpAutograd"), (musa, "_MusaFusedLogpAutograd")]
)
@pytest.mark.parametrize("native_present", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_fallback_selection_and_cast(
    monkeypatch, before, module, bridge, native_present, dtype
):
    # CUDA now selects #495's capability on non-HIP builds, including CPU doubles.
    # Exercise its HIP fallback and MUSA's device-gated fallback without changing either oracle.
    monkeypatch.setattr(torch.version, "hip", "6.4.0")
    x, labels = inputs(dtype, "3d")
    backend = SimpleNamespace()
    if native_present:

        def forbidden(*args):
            pytest.fail("a native symbol alone must not select MUSA backward")

        backend.fused_logp_backward = forbidden
    ctx = SimpleNamespace(
        saved_tensors=(x.reshape(-1, 7), labels.reshape(-1).long()),
        backend=backend,
        input_shape=x.shape,
        input_dtype=dtype,
    )
    upstream = torch.linspace(-1, 2, 6, dtype=dtype).reshape(2, 3)
    actual = getattr(module, bridge).backward(ctx, upstream)[0]
    expected = before._FusedLogpAutograd.backward(ctx, upstream)[0]
    assert_bits(actual, expected)
    assert actual.dtype == dtype


@pytest.mark.parametrize("native_present", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_backward_preserves_evidence_runner_patch_surface(
    monkeypatch, before, native_present, dtype
):
    from benchmarks.operators.logprob import fused_logp_backward_evidence as evidence

    monkeypatch.setattr(torch.version, "hip", None)
    backend = RecordingBackend()
    chunked = cuda.fused_logp_backward_chunked
    calls = []

    def native(logits, labels, grad):
        calls.append("native")
        assert grad.dtype == torch.float32 and grad.is_contiguous()
        return chunked(logits, labels, grad, logits.dtype, chunk_elems=7)

    def fallback(logits, labels, grad, out_dtype):
        calls.append("chunked")
        assert out_dtype == dtype
        return chunked(logits, labels, grad, out_dtype, chunk_elems=7)

    if native_present:
        backend.fused_logp_backward = native
        dispatched_backend = backend
    else:
        monkeypatch.setattr(evidence, "_C", backend)
        dispatched_backend = evidence._ForwardOnly()

    x, labels = inputs(dtype, "strided")
    old_x = x.detach().clone().requires_grad_()
    old = construct(monkeypatch, before, "FusedLogpGenericOp", RecordingBackend())
    upstream = torch.linspace(-1, 2, 6, dtype=dtype).reshape(2, 3).t()
    assert not x.is_contiguous() and not upstream.is_contiguous()
    with evidence._vjp(fallback):
        output = evidence._autograd_fn(dispatched_backend)(x, labels)
        saved_x, saved_labels = output.grad_fn.saved_tensors
        assert saved_x.shape == (6, 7) and saved_x.is_contiguous()
        assert saved_labels.dtype == torch.long and saved_labels.is_contiguous()
        output.backward(upstream)
    old_output = old(old_x, labels)
    old_output.backward(upstream)
    assert calls == (["native"] if native_present else ["chunked"])
    assert_bits(output, old_output)
    assert_bits(x.grad, old_x.grad)
    assert cuda.fused_logp_backward_chunked is chunked


VARIANTS = {
    "apply_fp32": "fused_logp_forward_fp32",
    "out": "fused_logp_forward_out",
    "indexed_out": "fused_logp_forward_indexed_out",
    "indexed_fp32": "fused_logp_forward_indexed_fp32",
    "online_out": "fused_logp_forward_online_out",
    "online_fp32": "fused_logp_forward_online_fp32",
    "online_indexed_out": "fused_logp_forward_online_indexed_out",
    "online_indexed_fp32": "fused_logp_forward_online_indexed_fp32",
}


@pytest.mark.parametrize("method,symbol", VARIANTS.items())
@pytest.mark.parametrize(
    "module,cls_name", [(cuda, "FusedLogpGenericOp"), (musa, "MusaFusedLogpOp")]
)
def test_variant_delegation_and_storage(monkeypatch, before, module, cls_name, method, symbol):
    backend = RecordingBackend()
    op = construct(monkeypatch, module, cls_name, backend)
    old_backend = RecordingBackend()
    old = construct(monkeypatch, before, "FusedLogpGenericOp", old_backend)
    x, labels = inputs(torch.float16, "strided")
    indices = torch.tensor([[0, 99], [4, 99]], dtype=torch.int32)[:, 0]
    out = torch.full(x.shape[:-1], 123.0, dtype=torch.float16)
    old_out = out.clone()
    args = [indices] if "indexed" in method else []
    output_args = [out] if method.endswith("out") else []
    old_output_args = [old_out] if output_args else []
    result = getattr(op, method)(x, labels, *args, *output_args)
    expected = getattr(old, method)(x, labels, *args, *old_output_args)
    assert_bits(result, expected)
    name, received = backend.calls[-1]
    assert name == symbol
    assert_bits(received[0], old_backend.calls[-1][1][0])
    assert received[0].is_contiguous() == old_backend.calls[-1][1][0].is_contiguous()
    assert received[1].dtype == torch.long and received[1].is_contiguous()
    if args:
        assert received[2].dtype == torch.long and received[2].is_contiguous()
    if output_args:
        assert result.data_ptr() == out.data_ptr()
        if args:
            assert torch.equal(
                out.reshape(-1)[[1, 2, 3, 5]], torch.full((4,), 123.0, dtype=out.dtype)
            )
    else:
        assert result.dtype == torch.float32


def test_output_errors_and_missing_variant_preserve_main(monkeypatch, before):
    op = construct(monkeypatch, cuda, "FusedLogpGenericOp", RecordingBackend())
    old = construct(monkeypatch, before, "FusedLogpGenericOp", RecordingBackend())
    x, labels = inputs(torch.float32, "3d")
    for output in (torch.empty(6), torch.empty(3, 2).t()):
        errors = []
        for candidate in (old, op):
            with pytest.raises((ValueError, RuntimeError)) as exc:
                candidate.out(x, labels, output)
            errors.append((type(exc.value), str(exc.value)))
        assert errors[0] == errors[1]
    minimal = SimpleNamespace(fused_logp=lambda *args: None)
    m = construct(monkeypatch, musa, "MusaFusedLogpOp", minimal)
    with pytest.raises(AttributeError, match="fused_logp_forward_fp32"):
        m.apply_fp32(x, labels)


def test_patch_surfaces(monkeypatch):
    backend = RecordingBackend()
    op = construct(monkeypatch, cuda, "FusedLogpGenericOp", backend)
    legacy = importlib.import_module("rl_engine.kernels.ops.cuda.loss.logp")
    sentinel = object()
    monkeypatch.setattr(legacy, "_FusedLogpAutograd", SimpleNamespace(apply=lambda *args: sentinel))
    x, labels = inputs(torch.float32, "2d")
    assert op(x, labels) is sentinel
    # Selection remains lexically owned by the backend module for #495's future helper patch.
    assert cuda.FusedLogpGenericOp.apply.__globals__ is cuda.__dict__
    assert musa._MusaFusedLogpAutograd.backward.__globals__ is musa.__dict__


def test_musa_extension_binding_is_independent(monkeypatch):
    cuda_backend, musa_backend = RecordingBackend(), RecordingBackend()
    construct(monkeypatch, cuda, "FusedLogpGenericOp", cuda_backend)
    op = construct(monkeypatch, musa, "MusaFusedLogpOp", musa_backend)
    x, labels = inputs(torch.float32, "2d")
    op(x, labels)
    assert op._backend is musa_backend and op.op == musa_backend.fused_logp
    assert len(musa_backend.calls) == 1 and not cuda_backend.calls


def test_musa_missing_native_uses_shared_vjp(monkeypatch):
    logits = SimpleNamespace(device=SimpleNamespace(type="musa"))
    labels = torch.tensor([0, 1])
    upstream = torch.ones(2, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        saved_tensors=(logits, labels),
        backend=SimpleNamespace(),
        input_shape=(2, 7),
        input_dtype=upstream.dtype,
    )
    calls = []
    sentinel = torch.zeros(2, 7, dtype=upstream.dtype)

    def fallback(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(musa, "fused_logp_backward_chunked", fallback)
    grad, labels_grad, backend_grad = musa._MusaFusedLogpAutograd.backward(ctx, upstream)
    assert_bits(grad, sentinel)
    assert labels_grad is None and backend_grad is None
    assert calls[0][0] is logits and calls[0][1] is labels and calls[0][2] is upstream
    assert calls[0][3] == ctx.input_dtype


@pytest.mark.parametrize("bridge", [cuda._FusedLogpAutograd, musa._MusaFusedLogpAutograd])
@pytest.mark.parametrize("labels", [torch.tensor([-1, 0]), torch.tensor([7, 0])])
def test_invalid_label_backward_preserves_integrated_main(bridge, labels):
    # #495 changed the inherited fallback for both backends. The pre-#495 frozen
    # oracle cannot establish this newer zero-gradient contract for invalid targets.
    x = torch.full((2, 7), float("-inf"))
    ctx = SimpleNamespace(
        saved_tensors=(x, labels),
        backend=SimpleNamespace(),
        input_shape=x.shape,
        input_dtype=x.dtype,
    )
    grad, labels_grad, backend_grad = bridge.backward(ctx, torch.ones(2))
    assert grad.shape == x.shape and grad.dtype == x.dtype
    assert_bits(grad[0], torch.zeros_like(x[0]))
    assert torch.isnan(grad[1]).all()
    assert labels_grad is None and backend_grad is None


@pytest.mark.parametrize("layout", ["2d", "3d", "strided"])
def test_higher_order_behavior_is_unchanged(monkeypatch, before, layout):
    outcomes = []
    for module in (before, cuda):
        op = construct(monkeypatch, module, "FusedLogpGenericOp", RecordingBackend())
        x, labels = inputs(torch.float32, layout)
        grad = torch.autograd.grad(op(x, labels).sum(), x, create_graph=True)[0]
        if grad.requires_grad:
            second = torch.autograd.grad(
                (grad * torch.arange(grad.numel()).reshape(grad.shape)).sum(), x, allow_unused=True
            )[0]
            outcomes.append((True, second is None))
        else:
            outcomes.append((False, None))
    assert outcomes[0] == outcomes[1]


def test_shared_source_inventory():
    from rl_engine.validation.operators.gradient_adapters import get_adapter, listed_source_paths

    paths = listed_source_paths(get_adapter("logp"))
    assert Path(inspect.getfile(shared)) in paths
    for path in paths:
        assert path.is_file()


@pytest.mark.parametrize(
    "cls",
    [
        cuda.FusedLogpGenericOp,
        cuda.FusedLogpSM90Op,
        cuda.DeterministicLogpCUDAOp,
        musa.MusaFusedLogpOp,
    ],
)
@pytest.mark.parametrize(
    "original,replacement",
    [
        (
            "probs = torch.softmax(logits[r0:r1].float(), dim=-1)",
            "probs = torch.zeros_like(logits[r0:r1].float())",
        ),
        (
            "dtype=torch.long).contiguous()",
            "dtype=torch.int32).contiguous()",
        ),
    ],
    ids=["vjp", "input_preparation"],
)
def test_shared_helper_source_changes_runtime_fingerprint(
    monkeypatch, tmp_path, cls, original, replacement
):
    from rl_engine.runtime.semantic_registry import operator_implementation_fingerprint

    # Alter a temporary copy, leaving the running implementation and checkout intact.
    source = Path(shared.__file__).read_text()
    path = tmp_path / "fused_logp.py"
    path.write_text(source)
    monkeypatch.setattr(shared, "__file__", str(path))
    op = object.__new__(cls)
    before_hash = operator_implementation_fingerprint(cls, op)
    changed_source = source.replace(original, replacement)
    assert changed_source != source
    path.write_text(changed_source)
    after_hash = operator_implementation_fingerprint(cls, op)
    assert before_hash != after_hash


def test_same_module_fingerprint_payload_is_unchanged():
    from rl_engine.runtime import semantic_registry as registry

    class LocalOp:
        def __call__(self):
            pass

    op = LocalOp()
    # The original payload remains byte-identical for same-module entrypoints.
    expected = registry._fingerprint(
        {
            "factory": registry._implementation_identity(LocalOp),
            "concrete_type": registry._implementation_identity(LocalOp),
            "runtime_entrypoints": {"__call__": registry._callable_identity(op.__call__)},
        }
    )
    assert registry.operator_implementation_fingerprint(LocalOp, op) == expected


def test_deterministic_autograd_source_changes_runtime_fingerprint(monkeypatch, tmp_path):
    from rl_engine.backends.cuda.logprob import _deterministic_autograd as helper
    from rl_engine.runtime.semantic_registry import operator_implementation_fingerprint

    source = Path(helper.__file__).read_text()
    path = tmp_path / "_deterministic_autograd.py"
    path.write_text(source)
    monkeypatch.setattr(helper, "__file__", str(path))
    op = object.__new__(cuda.DeterministicLogpCUDAOp)
    before_hash = operator_implementation_fingerprint(type(op), op)
    changed = source.replace("torch.float64", "torch.float32")
    assert changed != source
    path.write_text(changed)
    assert operator_implementation_fingerprint(type(op), op) != before_hash


def test_h100_runner_rejects_missing_cuda(monkeypatch, tmp_path):
    from tools.validation.operators import compare_fused_logp_ownership as runner

    output = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--output", str(output)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError, match="requires a real CUDA device"):
        runner.main()
    assert not output.exists()


def test_h100_runner_rejects_pre495_extension(monkeypatch, tmp_path):
    from tools.validation.operators import compare_fused_logp_ownership as runner

    output = tmp_path / "evidence.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--output", str(output)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(runner, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(runner, "_C", SimpleNamespace(fused_logp=lambda *args: None))
    with pytest.raises(AssertionError, match="requires the merged #495 native backward"):
        runner.main()
    assert not output.exists()
