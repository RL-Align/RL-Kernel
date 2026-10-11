# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    QwenImageAdaLNModProjectionCudaOp,
)
from rl_engine.backends.cuda.model_specific.qwenimage.final_adaln_out import (
    QwenImageFinalAdaLNOutCudaOp,
)
from rl_engine.contracts.numerical import load_contract
from rl_engine.reference.model_specific.qwenimage.adaln_mod_projection import reference_projection
from rl_engine.reference.model_specific.qwenimage.final_adaln_out import reference_final

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
KINDS = ["projection", "final"]


def equal_bytes(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def inputs(kind, dtype, batch=2, seq=33, device="cuda:0", seed=19):
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(shape, scale=1):
        return (
            torch.randn(shape, generator=gen, device=device, dtype=dtype) * scale
        ).requires_grad_()

    args = {
        "temb": rand((batch, 3072)),
        "weight": rand(((6 if kind == "projection" else 2) * 3072, 3072), 0.02),
        "bias": rand(((6 if kind == "projection" else 2) * 3072,), 0.01),
    }
    if kind == "final":
        args["x"] = rand((batch, seq, 3072))
    return args


def operator(kind):
    return (
        QwenImageAdaLNModProjectionCudaOp()
        if kind == "projection"
        else QwenImageFinalAdaLNOutCudaOp()
    )


def random_upstream(y, seed=127):
    generator = torch.Generator(device=y.device).manual_seed(seed)
    return torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)


def evaluate(fn, args, grad=None):
    y = fn(**args)
    if grad is None:
        grad = random_upstream(y)
    grads = torch.autograd.grad(y, tuple(args.values()), grad)
    return y.detach(), dict(zip(args, grads))


def assert_accurate(actual, expected, dtype, judgment):
    spec = load_contract()["judgments"][judgment]["by_op_class"]["reduction"][
        str(dtype).split(".")[-1]
    ]
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), atol=spec["atol"], rtol=spec["rtol"]
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq", [33, 257])
def test_cpu_reference_all_gradients(kind, dtype, seq):
    args = inputs(kind, dtype, seq=seq)
    gold = {k: v.detach().cpu().float().requires_grad_() for k, v in args.items()}
    ref = reference_projection if kind == "projection" else reference_final
    y = operator(kind)(**args)
    gy = ref(**gold, execution_dtype=dtype)
    grad = random_upstream(y)
    grads = torch.autograd.grad(y, tuple(args.values()), grad)
    gold_grads = torch.autograd.grad(gy, tuple(gold.values()), grad.cpu().float())
    assert_accurate(y, gy, dtype, "forward_accuracy")
    for a, b in zip(grads, gold_grads):
        assert_accurate(a, b, dtype, "gradient_accuracy")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch_position_and_repeat_bytes(kind, dtype):
    args = inputs(kind, dtype, batch=9)
    op = operator(kind)
    grad = random_upstream(op(**args))
    y, grads = evaluate(op, args, grad)
    y2, grads2 = evaluate(op, args, grad)
    assert equal_bytes(y, y2)
    for k in grads:
        assert equal_bytes(grads[k], grads2[k]), k
    for order in ([0], [4], [8], [8, 0], [8, 4, 0], list(reversed(range(9)))):
        sub = {
            k: v[order].detach().requires_grad_() if k in ("x", "temb") else v
            for k, v in args.items()
        }
        sy, sg = evaluate(op, sub, grad[order])
        assert equal_bytes(sy, y[order])
        for k in ("x", "temb"):
            if k in args:
                assert equal_bytes(sg[k], grads[k][order]), (k, order)


@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 127, 2026])
def test_long_sequence_gradients(seq, dtype, seed):
    args = inputs("final", dtype, batch=1, seq=seq)
    gold = {k: v.detach().cpu().float().requires_grad_() for k, v in args.items()}
    y = operator("final")(**args)
    gy = reference_final(**gold, execution_dtype=dtype)
    grad = random_upstream(y, seed)
    a = torch.autograd.grad(y, tuple(args.values()), grad)
    b = torch.autograd.grad(gy, tuple(gold.values()), grad.float().cpu())
    assert_accurate(y, gy, dtype, "forward_accuracy")
    for k, av, bv in zip(args, a, b):
        try:
            assert_accurate(av, bv, dtype, "gradient_accuracy")
        except AssertionError as exc:
            raise AssertionError(k) from exc


@pytest.mark.parametrize("kind", KINDS)
def test_noncontiguous(kind):
    args = inputs(kind, torch.bfloat16)
    strided = {
        k: torch.stack((v, v), -1)[..., 0].detach().requires_grad_() for k, v in args.items()
    }
    op = operator(kind)
    grad = random_upstream(op(**args))
    a, ag = evaluate(op, args, grad)
    b, bg = evaluate(op, strided, grad)
    assert equal_bytes(a, b)
    for k in ag:
        assert equal_bytes(ag[k], bg[k])


@pytest.mark.parametrize("kind", KINDS)
def test_validation_and_readback(kind):
    args = inputs(kind, torch.float32)
    op = operator(kind)
    assert op.implementation_info(**args)["fallback"] is False
    for name in args:
        bad = dict(args)
        bad[name] = args[name].cpu()
        with pytest.raises(ValueError):
            op(**bad)
    bad = dict(args)
    bad["temb"] = args["temb"][:0]
    with pytest.raises(ValueError):
        op(**bad)
    bad = dict(args)
    bad["weight"] = args["weight"][:-1]
    with pytest.raises(ValueError):
        op(**bad)
    bad = {k: v.half() for k, v in args.items()}
    with pytest.raises(TypeError):
        op(**bad)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("kind", KINDS)
def test_device_one_under_device_zero(kind):
    args = inputs(kind, torch.bfloat16, device="cuda:1")
    op = operator(kind)
    with torch.cuda.device(0):
        y, grads = evaluate(op, args)
    assert y.device.index == 1
    assert all(g.device.index == 1 for g in grads.values())
    args["bias"] = args["bias"].to("cuda:0")
    with pytest.raises(ValueError):
        op(**args)


@pytest.mark.parametrize("kind", KINDS)
def test_registry_and_gtest(kind):
    from argparse import Namespace

    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.validation.operators.op_checks import run_operator_suite
    from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

    name = "qwenimage_adaln_mod_projection" if kind == "projection" else "qwenimage_final_adaln_out"
    registry = KernelRegistry()
    op = registry.get_op(name, device="cuda:0")
    assert isinstance(op, type(operator(kind)))
    args = Namespace(op=name, candidate="cuda", arch_key="sm100", batch=1, seq=33, seed=7)
    case = make_operator_case(args, torch.bfloat16, torch.device("cuda"))
    report = run_operator_suite(
        name, candidates=[make_candidate(args)], cases=[case], check_grad=True
    )
    assert report.passed, report.to_dict()


@pytest.mark.parametrize("kind", KINDS)
def test_missing_extension_fails_closed(kind, monkeypatch):
    import rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection as module

    monkeypatch.setattr(module, "_C", None)
    with pytest.raises(RuntimeError, match="rebuild"):
        operator(kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_order_parameter_contributions(kind, dtype):
    args = inputs(kind, dtype, batch=3)
    op = operator(kind)
    grad = random_upstream(op(**args))
    _, small_grads = evaluate(op, args, grad)
    expanded = {
        k: torch.cat([v, v[:2]], 0).detach().requires_grad_() if k in ("x", "temb") else v
        for k, v in args.items()
    }
    padded_grad = torch.cat([grad, torch.zeros_like(grad[:2])], 0)
    _, large_grads = evaluate(op, expanded, padded_grad)
    for k in ("weight", "bias"):
        assert equal_bytes(small_grads[k], large_grads[k]), k


def test_projection_chunk_order():
    args = inputs("projection", torch.float32, batch=1)
    args["weight"] = torch.zeros_like(args["weight"])
    args["bias"] = torch.arange(6, device="cuda", dtype=torch.float32).repeat_interleave(3072)
    out = operator("projection")(**args)
    for i, chunk in enumerate(out.chunk(6, -1)):
        assert equal_bytes(chunk, torch.full_like(chunk, i))


def test_final_scale_then_shift_and_layernorm():
    args = inputs("final", torch.float32, batch=1)
    args["weight"] = torch.zeros_like(args["weight"])
    args["bias"] = torch.cat(
        [torch.full((3072,), 2.0, device="cuda"), torch.full((3072,), 7.0, device="cuda")]
    )
    args["x"] = torch.full_like(args["x"], 13.0)
    out = operator("final")(**args)
    assert equal_bytes(out, torch.full_like(out, 7.0))
