# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton linear log-prob requires a CUDA device and Triton.",
)


def _sm90_available():
    """SM90 forward needs a Hopper GPU and the kernel compiled into the extension."""
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        if not (_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")):
            return False
    except Exception:  # pragma: no cover
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the "
    "extension built KERNEL_ALIGN_FORCE_SM90=1.",
)

# SM90 forward needs bf16 and a hidden dim that is a multiple of the kernel's K
# slice (32); N / V are deliberately left unaligned to the 64-wide tiles.
_SM90_N = 96
_SM90_D = 128
_SM90_V = 500


def _sm90_inputs(seed, *, bias=True, dtype=torch.bfloat16, lead=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lead = lead or (_SM90_N,)
    hidden = torch.randn(*lead, _SM90_D, generator=gen, device="cuda", dtype=dtype)
    weight = torch.randn(_SM90_V, _SM90_D, generator=gen, device="cuda", dtype=dtype)
    bias_t = torch.randn(_SM90_V, generator=gen, device="cuda", dtype=dtype) if bias else None
    target = torch.randint(0, _SM90_V, lead, generator=gen, device="cuda")
    return hidden, weight, target, bias_t


# Deliberately non-multiples of the kernel block sizes (32 / 64 / 64).
_N = 40
_D = 80
_V = 300


def _inputs(seed, *, device, dtype=torch.float32, bias=True, lead=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    lead = lead or (_N,)
    hidden = torch.randn(*lead, _D, generator=gen, device=device, dtype=dtype)
    weight = torch.randn(_V, _D, generator=gen, device=device, dtype=dtype)
    bias_t = torch.randn(_V, generator=gen, device=device, dtype=dtype) if bias else None
    target = torch.randint(0, _V, lead, generator=gen, device=device)
    return hidden, weight, target, bias_t


def _manual_reference(hidden, weight, target, bias):
    """The semantic definition: materialize logits, log_softmax, gather."""
    logits = torch.nn.functional.linear(
        hidden.float(), weight.float(), None if bias is None else bias.float()
    )
    logp = torch.log_softmax(logits, dim=-1)
    idx = target.reshape(-1).long()
    sel = logp.reshape(-1, logp.size(-1)).gather(-1, idx.unsqueeze(1)).squeeze(1)
    return sel.reshape(target.shape)


def test_native_matches_manual_reference():
    native = NativeLinearLogpOp()
    hidden, weight, target, bias = _inputs(0, device="cpu")
    out = native(hidden, weight, target, bias)
    ref = _manual_reference(hidden, weight, target, bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-5)


def test_native_rejects_shape_mismatch():
    native = NativeLinearLogpOp()
    hidden, weight, _, bias = _inputs(0, device="cpu")
    with pytest.raises(ValueError):
        native(hidden, weight, torch.zeros(_N + 1, dtype=torch.long), bias)


@requires_triton_cuda
def test_triton_forward_matches_native_fp32():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(1, device="cuda")
    ref = native(hidden, weight, target, bias)
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=1e-3)


@requires_triton_cuda
def test_triton_forward_matches_native_bf16():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(2, device="cuda", dtype=torch.bfloat16)
    # The kernel accumulates in fp32, so the oracle uses the fp32-upcast inputs.
    ref = native(hidden.float(), weight.float(), target, bias.float())
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_triton_cuda
@pytest.mark.parametrize("use_bias", [True, False])
def test_triton_backward_matches_native(use_bias):
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(3, device="cuda", bias=use_bias)
    grad_out = torch.randn(_N, device="cuda")

    def run(op, h, w, b):
        h = h.detach().clone().requires_grad_(True)
        w = w.detach().clone().requires_grad_(True)
        b = b.detach().clone().requires_grad_(True) if b is not None else None
        op(h, w, target, b).backward(grad_out)
        return h.grad, w.grad, (b.grad if b is not None else None)

    th, tw, tb = run(trit, hidden, weight, bias)
    nh, nw, nb = run(native, hidden, weight, bias)
    assert torch.allclose(th, nh, atol=2e-3)
    assert torch.allclose(tw, nw, atol=2e-3)
    if use_bias:
        assert torch.allclose(tb, nb, atol=2e-3)


@requires_triton_cuda
def test_triton_gradients_flow_to_inputs_only():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(4, device="cuda")
    hidden = hidden.requires_grad_(True)
    weight = weight.requires_grad_(True)
    bias = bias.requires_grad_(True)
    trit(hidden, weight, target, bias).sum().backward()
    assert hidden.grad is not None and weight.grad is not None and bias.grad is not None
    assert target.grad is None  # integer targets are non-differentiable


@requires_triton_cuda
def test_triton_preserves_leading_shape():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(5, device="cuda", lead=(4, 7))
    out = trit(hidden, weight, target, bias)
    assert out.shape == (4, 7)
    assert torch.allclose(out, native(hidden, weight, target, bias), atol=1e-3)


@requires_triton_cuda
def test_triton_large_vocab_smoke():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden = torch.randn(8, 64, device="cuda")
    weight = torch.randn(50257, 64, device="cuda")
    target = torch.randint(0, 50257, (8,), device="cuda")
    out = trit(hidden, weight, target)
    assert out.shape == (8,) and torch.isfinite(out).all()


@requires_sm90
def test_sm90_forward_matches_native_bf16():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(11)
    # The kernel matmul accumulates in fp32 (tensor cores), so the oracle uses the
    # fp32-upcast inputs -- like the Triton bf16 test.
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, bias.float())
    out = sm90(hidden, weight, target, bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_forward_no_bias():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, _ = _sm90_inputs(12, bias=False)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, None)
    out = sm90(hidden, weight, target)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
@pytest.mark.parametrize("use_bias", [True, False])
def test_sm90_forward_backward_matches_triton(use_bias):
    # The SM90 forward is fp32-accurate and the backward reuses the same
    # deterministic chunked path as the Triton op, so both match very tightly.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    sm90, trit = FusedLinearLogpSM90Op(), TritonLinearLogpOp()
    hidden, weight, target, bias = _sm90_inputs(13, bias=use_bias)
    grad_out = torch.randn(_SM90_N, device="cuda")

    def run(op):
        h = hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True) if bias is not None else None
        out = op(h, w, target, b)
        out.backward(grad_out)
        return out.detach(), h.grad, w.grad, (b.grad if b is not None else None)

    so, sh, sw, sb = run(sm90)
    to, th, tw, tb = run(trit)
    assert torch.allclose(so, to, atol=1e-3)
    assert torch.allclose(sh, th, atol=2e-3)
    assert torch.allclose(sw, tw, atol=2e-3)
    if use_bias:
        assert torch.allclose(sb, tb, atol=2e-3)


@requires_sm90
def test_sm90_preserves_leading_shape():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(14, lead=(6, 5))
    out = sm90(hidden, weight, target, bias)
    assert out.shape == (6, 5)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, bias.float())
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_large_vocab_smoke():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(50257, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, 50257, (40,), device="cuda")
    out = sm90(hidden, weight, target)
    assert out.shape == (40,) and torch.isfinite(out).all()


@requires_sm90
def test_sm90_falls_back_for_unsupported_inputs():
    # fp32 inputs and a hidden dim not divisible by the kernel's K slice are not
    # handled by the compiled forward; the op must fall back instead of erroring.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()

    fp32 = _sm90_inputs(15, dtype=torch.float32)
    out = sm90(*fp32)
    ref = NativeLinearLogpOp()(*fp32)
    assert torch.allclose(out, ref, atol=1e-3)

    # bf16 but D=80 (not a multiple of 32) -> fallback path.
    gen = torch.Generator(device="cuda").manual_seed(16)
    hidden = torch.randn(40, 80, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(300, 80, device="cuda", dtype=torch.bfloat16, generator=gen)
    target = torch.randint(0, 300, (40,), device="cuda", generator=gen)
    out = sm90(hidden, weight, target)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, None)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_rejects_bad_target_and_bias():
    # Shape/device mismatches must be a clean error, not a CUDA illegal access.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(17)

    with pytest.raises((ValueError, RuntimeError)):  # weight on the wrong device
        sm90(hidden, weight.cpu(), target, bias)
    with pytest.raises((ValueError, RuntimeError)):  # wrong target length
        sm90(hidden, weight, target[:-1], bias)
    with pytest.raises((ValueError, RuntimeError)):  # wrong bias length
        sm90(hidden, weight, target, bias[:-1])
    with pytest.raises((ValueError, RuntimeError)):  # bias on the wrong device
        sm90(hidden, weight, target, bias.cpu())


@requires_sm90
def test_sm90_rejects_out_of_range_target():
    # Padding (-100) / out-of-vocab ids must error, not silently corrupt fwd/bwd.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(18)

    pad = target.clone()
    pad[0] = -100  # typical ignore_index
    with pytest.raises(ValueError):
        sm90(hidden, weight, pad, bias)

    oob = target.clone()
    oob[1] = _SM90_V  # == V, one past the last valid id
    with pytest.raises(ValueError):
        sm90(hidden, weight, oob, bias)

    # A valid target (all in [0, V)) still works.
    out = sm90(hidden, weight, target, bias)
    assert out.shape == target.shape and torch.isfinite(out).all()


def test_registry_dispatch_matches_native():
    from rl_engine.kernels.registry import kernel_registry
    from rl_engine.platforms.device import device_ctx

    op = kernel_registry.get_op("linear_logp")
    device = device_ctx.device if device_ctx.device_type == "cuda" else "cpu"
    hidden, weight, target, bias = _inputs(6, device=device)
    out = op(hidden, weight, target, bias)
    ref = NativeLinearLogpOp()(hidden, weight, target, bias)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-3)
