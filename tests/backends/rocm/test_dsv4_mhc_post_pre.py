# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from rl_engine.integrations.common.mhc_worker import P1Worker
from rl_engine.ops.mhc.post_pre import post_pre
from rl_engine.ops.mhc.training import _post_row, _sum


def inputs(t, c, device):
    rng = torch.Generator().manual_seed(575)

    def sample(shape, dtype):
        return torch.randn(shape, generator=rng).to(device=device, dtype=dtype).requires_grad_()

    return tuple(
        sample(shape, dtype)
        for shape, dtype in (
            ((t, c), torch.bfloat16),
            ((t, 4, c), torch.bfloat16),
            ((t, 4), torch.float32),
            ((t, 4, 4), torch.float32),
            ((24, 4 * c), torch.float32),
            ((3,), torch.float32),
            ((24,), torch.float32),
        )
    )


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("c", [1, 8, 64, 4096])
def test_kernel_in_cpu_interpreter(c):
    if os.getenv("TRITON_INTERPRET") != "1":
        pytest.skip("run separately with TRITON_INTERPRET=1")
    pytest.importorskip("triton")
    from rl_engine.backends.rocm.mhc.post_project import post_project

    y, x, p, b, w, *_ = inputs(1, c, "cpu")
    with torch.no_grad():
        res, lin, ss = post_project(y, x, p, b, w)
        want = _post_row(y[0], x[0], p[0], b[0])[None]
        flat = want.float().flatten(1)
        bits(res, want)
        bits(lin, _sum(flat[:, None, :] * w.bfloat16().float()))
        bits(ss, _sum(flat.square()))


def require_gpu():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("missing target ROCm GPU")
        pytest.skip("requires ROCm")
    print(
        "device:",
        torch.cuda.get_device_properties(0).gcnArchName,
        "torch:",
        torch.__version__,
        "HIP:",
        torch.version.hip,
    )


@pytest.mark.parametrize("m", [0, 1, 16, 17])
def test_device_forward_and_vjp(m):
    require_gpu()
    args = inputs(m, 64, "cuda")

    def run(mode):
        out = post_pre(*args, backend="rocm", mode=mode, sinkhorn_repeat=2)
        rng = torch.Generator().manual_seed(76)
        dy = tuple(torch.randn(v.shape, generator=rng).to(v) for v in out)
        return (*out, *torch.autograd.grad(out, args, dy))

    for a, b in zip(run("split"), run("fused"), strict=True):
        bits(a, b)


@pytest.mark.parametrize("framework", ["verl", "vime"])
@pytest.mark.parametrize("role", ["training", "rollout"])
@pytest.mark.parametrize("mode", ["split", "fused"])
def test_versioned_graph(framework, role, mode):
    require_gpu()
    args = inputs(1, 64, "cuda")
    w = P1Worker(framework, role, backend="rocm", mode=mode)

    def step():
        out = w.adjacent(*args, 2)
        if role == "rollout":
            return out
        return (*out, *torch.autograd.grad(out, args, tuple(torch.ones_like(v) for v in out)))

    old = None
    for version in range(3):
        if version:
            with w.updating(version, torch.cuda.synchronize), torch.no_grad():
                args[-1].add_(0.125)
            with pytest.raises(RuntimeError, match="stale"):
                old()
        replay = w.capture(step, version=version, device=torch.device("cuda", 0))
        with w.request(version):
            expected = step()
        for a, b in zip(expected, replay(), strict=True):
            bits(a, b)
        old = replay
