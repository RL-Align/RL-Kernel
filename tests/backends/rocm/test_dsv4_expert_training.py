# SPDX-License-Identifier: Apache-2.0
"""Requires task 43/44/45/46/49 providers, then real MI300X or MI355X hardware.

RLK_EXPERT_BACKEND_FACTORY=your_module:make_backend must construct the explicit
RocmExpertTrainingBackend using the actual task 43/44 and 49 adapters. There is
no built-in stand-in for those missing production dependencies.
"""

import importlib
import os

import pytest
import torch

from rl_engine.backends.rocm.moe.expert_training import RocmExpertTrainingBackend
from rl_engine.contracts.operators.expert_training import ExpertTrainingConfig, PackedExpertWeight
from rl_engine.models.deepseek_v4.expert_training import RoutedExpertMLP
from rl_engine.reference.moe.expert_training import ReferenceExpertTrainingBackend


@pytest.fixture(scope="module")
def backend():
    available = bool(torch.version.hip and torch.cuda.is_available())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] if available else "none"
    if arch not in ("gfx942", "gfx950"):
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail(f"task 48 requires gfx942/gfx950, found {arch}")
        pytest.skip("no target ROCm GPU")
    spec = os.getenv("RLK_EXPERT_BACKEND_FACTORY", "")
    if ":" not in spec:
        pytest.fail("set RLK_EXPERT_BACKEND_FACTORY to a real task 43/44/49 adapter factory")
    module, name = spec.split(":", 1)
    result = getattr(importlib.import_module(module), name)()
    assert isinstance(
        result, RocmExpertTrainingBackend
    ), "GPU acceptance forbids reference fallback"
    print(f"GPU={arch} torch={torch.__version__} hip={torch.version.hip} profile={result.profile}")
    return result


def _module(backend, mode):
    rng = torch.Generator().manual_seed(480)
    config = ExpertTrainingConfig(32, 32, 3, mode, alpha=0.375)
    weights = []
    for n in (64, 32):
        codes = torch.randint(0, 256, (3, n, 16), generator=rng, dtype=torch.uint8).cuda()
        scales = torch.full((3, n, 1), 122, dtype=torch.uint8, device="cuda")
        weights.append(PackedExpertWeight(codes, scales, (3, n, 32)) if config.use_base else None)
    adapters = (
        tuple(
            (torch.randn(shape, generator=rng) * 0.15).to(device="cuda", dtype=torch.bfloat16)
            for shape in ((3, 32), (64, 3), (3, 32), (32, 3))
        )
        if config.use_lora
        else None
    )
    return RoutedExpertMLP(config, backend=backend, w1=weights[0], w2=weights[1], adapters=adapters)


def _inputs(rows):
    rng = torch.Generator().manual_seed(481)
    x = (torch.randn(rows, 64, generator=rng) * 0.25).to(device="cuda", dtype=torch.bfloat16)[
        :, ::2
    ]
    p = torch.full((rows * 2,), 0.37, device="cuda")[::2]
    if rows:
        p[0] = 0
    if rows > 1:
        p[1] = 1e-8
    offsets = torch.tensor([0, rows // 2, rows // 2, rows], device="cuda", dtype=torch.int64)
    return x, p, offsets


def _run(module, x, p, offsets, recompute):
    module.zero_grad(set_to_none=True)
    module.recompute = recompute
    x, p = x.detach().requires_grad_(), p.detach().requires_grad_()
    y = module(x, p, offsets)
    dy = torch.linspace(-0.3, 0.5, y.numel(), device=x.device).reshape(y.shape).bfloat16()
    y.backward(dy)
    return (
        y.detach(),
        x.grad,
        p.grad,
        *(param.grad for param in module.parameters() if param.requires_grad),
    )


def _same_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("mode", ["base_only", "lora_only", "combined"])
@pytest.mark.parametrize("rows", [0, 1, 17])
def test_real_save_recompute_and_cpu_staged_accuracy(backend, mode, rows):
    module = _module(backend, mode)
    x, p, offsets = _inputs(rows)
    saved = _run(module, x, p, offsets, False)
    for actual in (_run(module, x, p, offsets, True), _run(module, x, p, offsets, False)):
        for a, b in zip(saved, actual):
            _same_bytes(a, b)
    with torch.no_grad():
        _same_bytes(saved[0], module(x, p, offsets))
    # Providers may own non-copyable runtime handles; copy only module tensors.
    cpu_weights = tuple(
        None if w is None else PackedExpertWeight(w.codes.cpu(), w.scales.cpu(), w.shape)
        for w in module.base_weights()
    )
    cpu_adapters = (
        tuple(p.detach().cpu() for p in module.adapters()) if module.config.use_lora else None
    )
    cpu = RoutedExpertMLP(
        module.config,
        backend=ReferenceExpertTrainingBackend(),
        w1=cpu_weights[0],
        w2=cpu_weights[1],
        adapters=cpu_adapters,
    )
    reference = _run(cpu, x.cpu(), p.cpu(), offsets.cpu(), False)
    # Different reduction/exp profiles: accuracy is distinct from byte invariance.
    for a, b in zip(saved, reference):
        torch.testing.assert_close(a.cpu().float(), b.float(), rtol=0.06, atol=2e-4)


@pytest.mark.parametrize("mode", ["base_only", "lora_only", "combined"])
def test_real_same_row_forward_dx_and_dp_across_batches(backend, mode):
    module = _module(backend, mode)
    x, p, _ = _inputs(1)
    p.fill_(0.37)

    def run_row_batch(xx, pp, offsets):
        xx, pp = xx.detach().requires_grad_(), pp.detach().requires_grad_()
        y = module(xx, pp, offsets)
        dx, dp = torch.autograd.grad(y, (xx, pp), torch.ones_like(y))
        return y, dx, dp

    expected = run_row_batch(x, p, torch.tensor([0, 0, 0, 1], device="cuda"))
    for m in (1, 16, 17, 64, 65):
        for row in sorted({0, m // 2, m - 1}):
            xx, pp, _ = _inputs(m)
            xx[row], pp[row] = x[0], p[0]
            # Keep the same expert; only change batch geometry and row position.
            got = run_row_batch(xx, pp, torch.tensor([0, 0, 0, m], device="cuda"))
            for a, b in zip(expected, got):
                _same_bytes(a, b[row : row + 1])
