# SPDX-License-Identifier: Apache-2.0
"""Every production candidate must pass: no selection or rejection fallback here."""

import json

import pytest
import torch

pytest.importorskip("triton")
from benchmarks.tuning.autotuning_cases import OPERATORS, make_case, supported  # noqa: E402
from rl_engine.backends.shared.triton import autotuning  # noqa: E402
from rl_engine.platforms.autotuner import TuneConfig  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm required")


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.fixture
def qualify(monkeypatch, request):
    """Force every raw launch, compare all mutated buffers, then execute fixed output."""
    seen = set()

    def getitem(kernel, grid):
        def launch(*args, **kwargs):
            reference = list(args)
            for i in kernel.outputs:
                reference[i] = args[i].clone()
            kernel.kernel[grid](*reference, **kwargs)
            comparisons = []
            for config in autotuning.launch_candidates(kernel):
                trial = list(args)
                for i in kernel.outputs:
                    trial[i] = args[i].clone()
                options = {**kwargs, **config.to_dict()}
                selected_grid = grid
                if kernel.flat_block is not None and "BLOCK" in config.to_dict():
                    selected_grid = (
                        (args[kernel.flat_block] + options["BLOCK"] - 1) // options["BLOCK"],
                    )
                # OutOfResources is a reported resource limitation, never a numerical pass.
                from triton.runtime.errors import OutOfResources

                try:
                    if request.config.option.verbose > 1:
                        print(kernel.name, config, options, flush=True)
                    kernel.kernel[selected_grid](*trial, **options)
                except OutOfResources as exc:
                    if config == TuneConfig():
                        raise
                    pytest.skip(f"resource limitation: {kernel.name} {config}: {exc}")
                for i in kernel.outputs:
                    left = reference[i].contiguous().view(torch.uint8)
                    right = trial[i].contiguous().view(torch.uint8)
                    comparisons.append((config, i, torch.all(left == right)))
                seen.add((kernel.name, config.options))
            # One synchronization per kernel, rather than one per candidate/output.
            if not torch.stack([check for _, _, check in comparisons]).all().item():
                failures = [(config, i) for config, i, check in comparisons if not check.item()]
                raise AssertionError(f"{kernel.name}: {failures}")
            return kernel.kernel[grid](*args, **kwargs)

        return launch

    monkeypatch.setattr(autotuning.TunableKernel, "__getitem__", getitem)
    yield seen
    request.node.user_properties.append(("qualified_launches", json.dumps(sorted(seen))))


WIDTHS = {
    "silu": (255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049),
    "swiglu": (255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049),
    "rope": (64, 126, 128, 130),
    "embedding": (63, 64, 65, 127, 128, 129),
    "rmsnorm": (127, 128, 129, 255, 256, 257, 4095, 4096, 4097),
    "ratio_kl": (127, 128, 129, 1023, 1024, 1025),
    "logp": (127, 128, 129, 1023, 1024, 1025),
    "grpo": (3, 4, 5, 31, 32, 33, 127, 128, 129),
    "gemm": (31, 32, 33, 63, 64, 65),
    "attention": (128,),
}


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [0, 1, 2026])
@pytest.mark.parametrize("scenario", ["boundaries", "strided", "special", "empty"])
def test_all_candidates(qualify, monkeypatch, request, name, dtype, seed, scenario):
    if not supported(name, dtype):
        pytest.skip("Existing unsupported dtype")
    torch.manual_seed(seed)
    monkeypatch.setenv("RL_KERNEL_MODE", "auto")
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    widths = (
        WIDTHS[name]
        if scenario == "boundaries"
        else ((128 if name == "rope" else WIDTHS[name][-1]),)
    )
    for width in widths:
        rows_list = (
            (1, 31, 32, 33, 253, 254, 255)
            if name == "attention" and scenario == "boundaries"
            else (3,)
        )
        for rows in rows_list:
            case = make_case(
                name,
                dtype,
                rows=0 if scenario == "empty" else rows,
                width=width,
                layout="strided" if scenario == "strided" else "contiguous",
                values="special" if scenario == "special" else "random",
                causal=seed != 1,
            )
            # Establish existing behavior first, including unsupported/empty inputs.
            try:
                with monkeypatch.context() as fixed:
                    fixed.setattr(
                        autotuning.TunableKernel,
                        "__getitem__",
                        lambda kernel, grid: kernel.kernel[grid],
                    )
                    baseline = case.forward()
                    differentiable = tuple(x for x in baseline if x.requires_grad)
                    grad_outputs = tuple(torch.randn_like(x) for x in differentiable)
                    expected_grads = (
                        torch.autograd.grad(differentiable, case.variables, grad_outputs)
                        if case.variables
                        else ()
                    )
            except Exception as error:
                from triton.compiler.errors import CompilationError

                expected_error = None
                if scenario == "empty":
                    expected_error = {
                        "rope": ZeroDivisionError,
                        "gemm": ValueError,
                        "attention": ValueError,
                    }.get(name)
                elif name == "rope" and width in (126, 130):
                    expected_error = CompilationError
                if expected_error is None or type(error) is not expected_error:
                    raise
                with pytest.raises(type(error)) as actual_error:
                    actual = case.forward()
                    if case.variables:
                        differentiable = tuple(x for x in actual if x.requires_grad)
                        torch.autograd.grad(
                            differentiable,
                            case.variables,
                            tuple(torch.ones_like(x) for x in differentiable),
                        )
                assert str(actual_error.value) == str(error)
                request.node.user_properties.append(
                    ("existing_error", f"{name}/{width}: {type(error).__name__}")
                )
                continue
            originals = [x.detach().clone() for x in case.inputs]
            variables = case.variables
            for x in variables:
                x.grad = torch.ones_like(x)
            # All candidates see each new value set, including backward launches.
            outputs = case.forward()
            for actual, expected in zip(outputs, baseline, strict=True):
                bits(actual, expected)
            if variables:
                differentiable = tuple(x for x in outputs if x.requires_grad)
                gradients = torch.autograd.grad(differentiable, variables, grad_outputs)
                for actual, expected in zip(gradients, expected_grads, strict=True):
                    bits(actual, expected)
                for x in variables:
                    bits(x.grad, torch.ones_like(x))
            for actual, original in zip(case.inputs, originals, strict=True):
                bits(actual, original)
    if scenario != "empty":
        assert qualify, "No tunable kernels exercised"


@pytest.mark.parametrize("name", ["ratio_kl", "logp", "attention"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [0, 1, 2026])
def test_fully_masked_candidates(qualify, name, dtype, seed):
    if not supported(name, dtype):
        pytest.skip("Existing unsupported dtype")
    torch.manual_seed(seed)
    case = make_case(name, dtype, width=128, causal=seed != 1)
    if name == "ratio_kl":
        case.inputs[3].zero_()
    elif name == "logp":
        case.inputs[1].fill_(-100)
    else:
        case.kwargs["key_padding_mask"].zero_()
    outputs = case.forward()
    gradients = torch.autograd.grad(
        outputs, case.variables, tuple(torch.ones_like(x) for x in outputs)
    )
    for gradient in gradients:
        bits(gradient, torch.zeros_like(gradient))
    assert qualify


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("gemm", torch.bfloat16),
        ("gemm", torch.float32),
        ("attention", torch.float16),
        ("attention", torch.bfloat16),
        ("logp", torch.float16),
        ("logp", torch.bfloat16),
        ("logp", torch.float32),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2026])
@pytest.mark.parametrize("values", ["random", "special"])
def test_fp32_output_and_lse_variants(qualify, monkeypatch, name, dtype, seed, values):
    from benchmarks.tuning.autotuning_cases import fixed_case

    torch.manual_seed(seed)
    case = make_case(
        name,
        dtype,
        width=128 if name == "attention" else 65,
        layout="strided",
        values=values,
        causal=seed != 1,
    )
    fixed = fixed_case(case, name, dtype)
    if name == "gemm":
        # FP32-input construction already selects the bound forward_fp32 method.
        if dtype == torch.bfloat16:
            case.op, fixed.op = case.op.forward_fp32, fixed.op.forward_fp32
    elif name == "attention":
        case.op, fixed.op = case.op.forward_fp32, fixed.op.forward_fp32
    else:
        case.op, fixed.op = case.op.forward_with_lse, fixed.op.forward_with_lse
        case.backward = fixed.backward = False  # This existing API has no autograd wrapper.
    with monkeypatch.context() as raw:
        raw.setattr(
            autotuning.TunableKernel, "__getitem__", lambda kernel, grid: kernel.kernel[grid]
        )
        expected = fixed.forward()
        grad_outputs = tuple(torch.randn_like(x) for x in expected)
        expected_grads = (
            torch.autograd.grad(expected, fixed.variables, grad_outputs) if fixed.variables else ()
        )
    actual = case.forward()
    actual_grads = (
        torch.autograd.grad(actual, case.variables, grad_outputs) if case.variables else ()
    )
    for left, right in zip(expected + expected_grads, actual + actual_grads, strict=True):
        bits(left, right)
    assert qualify
