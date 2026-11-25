# SPDX-License-Identifier: Apache-2.0
import importlib.util
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.ops.guidance import cfg_combine_renorm
from rl_engine.reference.guidance import NativeCfgCombineRenormOp
from rl_engine.runtime.registry import KernelRegistry

OP_NAME = "cfg_combine_renorm"
CONTRACT = load_contract()
GPU = pytest.mark.skipif(
    torch.version.cuda is None
    or torch.version.hip is not None
    or not torch.cuda.is_available()
    or importlib.util.find_spec("triton") is None,
    reason="Requires an NVIDIA CUDA device and Triton",
)


def test_cpu_can_import_triton_wrapper_without_triton():
    code = """
import importlib.abc
import sys

class BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == 'triton':
            raise ModuleNotFoundError(fullname)

import torch
sys.meta_path.insert(0, BlockTriton())
from rl_engine.backends.shared.triton.guidance.cfg_combine_renorm import TritonCfgCombineRenormOp
assert callable(TritonCfgCombineRenormOp())
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cpu_registry_and_explicit_triton_rejection():
    from rl_engine.backends.shared.triton.guidance.cfg_combine_renorm import (
        TritonCfgCombineRenormOp,
    )

    assert isinstance(KernelRegistry().get_op(OP_NAME, device="cpu"), NativeCfgCombineRenormOp)
    main = torch.ones(1, 2, 3)
    with pytest.raises((ValueError, RuntimeError)):
        TritonCfgCombineRenormOp()(main, main)
    with pytest.raises((ValueError, RuntimeError)):
        cfg_combine_renorm(main, main, backend="triton")


def test_hip_build_rejects_cuda_device(monkeypatch):
    from rl_engine.backends.shared.triton.guidance import TritonCfgCombineRenormOp

    monkeypatch.setattr(torch.version, "hip", "6.3")
    main = Mock(spec=torch.Tensor)
    main.device = torch.device("cuda:0")
    with pytest.raises((ValueError, RuntimeError), match="(?i)HIP|ROCm|NVIDIA"):
        TritonCfgCombineRenormOp()(main, main)
    with pytest.raises((ValueError, RuntimeError)):
        KernelRegistry().get_op(OP_NAME, device="cuda")


@pytest.mark.parametrize("device", ["meta", "xpu"])
def test_unsupported_registry_device_fails_closed(device):
    with pytest.raises((ValueError, RuntimeError)):
        KernelRegistry().get_op(OP_NAME, device=device)


def evaluate(fn, inputs, upstream, **policy):
    leaves = [x.detach().requires_grad_() for x in inputs]
    output = fn(*leaves, **policy)
    gradients = torch.autograd.grad(output, leaves, upstream)
    return [x.detach() for x in (output, *gradients)]


@GPU
def test_gpu_registry_selects_triton():
    from rl_engine.backends.shared.triton.guidance.cfg_combine_renorm import (
        TritonCfgCombineRenormOp,
    )

    assert isinstance(KernelRegistry().get_op(OP_NAME, device="cuda"), TritonCfgCombineRenormOp)


@GPU
@pytest.mark.parametrize("text_scale", [0.5, 1.0])
def test_gpu_identity_profile_with_image(text_scale):
    main = torch.arange(15, device="cuda", dtype=torch.float32).reshape(1, 3, 5)
    ignored = torch.full_like(main, float("nan"))
    upstream = main + 1
    out, dv, dt, di = evaluate(
        cfg_combine_renorm,
        [main, ignored, ignored],
        upstream,
        cfg_text_scale=text_scale,
        cfg_img_scale=2.0,
        timestep=0.7,
    )
    assert torch.equal(out, main) and torch.equal(dv, upstream)
    assert torch.count_nonzero(dt) == 0 and torch.count_nonzero(di) == 0


@GPU
@pytest.mark.parametrize("main,text,minimum", [(0.0, 0.0, 0.0), (1.0, -4.0, 0.5), (4.0, 7.0, 0.0)])
def test_gpu_clamped_gradient_for_text_only(main, text, minimum):
    v = torch.full((1, 1, 1), main, device="cuda")
    t = torch.full_like(v, text, requires_grad=True)
    kwargs = dict(cfg_text_scale=2.0, cfg_renorm_min=minimum, eps=1.0)
    candidate = cfg_combine_renorm(v, t, **kwargs)
    gold = cfg_combine_renorm(v, t, backend="pytorch", **kwargs)
    assert torch.equal(candidate, gold)
    assert torch.equal(
        torch.autograd.grad(candidate.sum(), t)[0], torch.autograd.grad(gold.sum(), t)[0]
    )


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["global", "channel", "text_channel"])
@pytest.mark.parametrize("branches", [2, 3])
def test_gpu_reference_forward_backward_and_interval(dtype, mode, branches):
    rng = torch.Generator().manual_seed(435)
    tensors = [torch.randn(3, 17, 65, generator=rng).to(dtype) for _ in range(branches + 1)]
    times = torch.tensor([0.4, 0.7, 1.0])
    gold = evaluate(
        NativeCfgCombineRenormOp().forward_fp32,
        tensors[:-1],
        tensors[-1].float(),
        timestep=times,
        renorm_type=mode,
    )
    actual = evaluate(
        cfg_combine_renorm,
        [x.cuda() for x in tensors[:-1]],
        tensors[-1].cuda(),
        timestep=times.cuda(),
        renorm_type=mode,
        backend="triton",
    )
    for idx, (result, expected) in enumerate(zip(actual, gold)):
        assert result.dtype == dtype
        tol = resolve_tolerance(
            CONTRACT,
            judgment="forward_accuracy" if idx == 0 else "gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(result.cpu(), expected.to(dtype), atol=tol.atol, rtol=tol.rtol)
    assert torch.equal(actual[0][0].cpu(), tensors[0][0])
    assert torch.equal(actual[1][0].cpu(), tensors[-1][0])
    assert all(torch.count_nonzero(g[0]) == 0 for g in actual[2:])


@GPU
@pytest.mark.parametrize("mode", ["global", "channel", "text_channel"])
def test_gpu_packed_forward_backward_bitwise_invariance(mode):
    rng = torch.Generator().manual_seed(66)
    requests = [[torch.randn(n, 259, generator=rng).cuda() for _ in range(4)] for n in (31, 5, 2)]

    def run(order):
        tensors = [torch.cat([requests[i][k] for i in order]) for k in range(4)]
        offsets = [0]
        for i in order:
            offsets.append(offsets[-1] + requests[i][0].shape[0])
        values = evaluate(
            cfg_combine_renorm,
            tensors[:-1],
            tensors[-1],
            cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
            renorm_type=mode,
            backend="triton",
        )
        idx = order.index(0)
        return [value[offsets[idx] : offsets[idx + 1]] for value in values]

    singleton = run([0])
    for order in ([2, 0, 1], [1, 2, 0]):
        assert all(torch.equal(x, y) for x, y in zip(singleton, run(order)))
