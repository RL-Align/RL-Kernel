# SPDX-License-Identifier: Apache-2.0
"""Task 48 integration checks; run with #601, #607 and #613 available."""

import importlib
import os

import pytest
import torch

from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient
from rl_engine.reference.moe.route_weight_grad import route_weight_grad_reference as reference


def dependency(name, *, required=False):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        if required or os.getenv("RLK_REQUIRE_P5_DEPENDENCIES") == "1":
            pytest.fail(f"missing P5 dependency {name}: {exc}")
        pytest.skip(f"requires P5 dependency {name}")


def make_module(backend, mode, device):
    contract = dependency("rl_engine.contracts.operators.expert_training")
    models = dependency("rl_engine.models.deepseek_v4.expert_training")
    rng = torch.Generator().manual_seed(495)
    config = contract.ExpertTrainingConfig(32, 32, 2, mode, alpha=0.375)
    weights = []
    for n in (64, 32):
        codes = torch.randint(0, 256, (2, n, 16), generator=rng, dtype=torch.uint8).to(device)
        scales = torch.full((2, n, 1), 122, dtype=torch.uint8, device=device)
        weights.append(
            contract.PackedExpertWeight(codes, scales, (2, n, 32)) if config.use_base else None
        )
    adapters = (
        tuple(
            (torch.randn(shape, generator=rng) * 0.2).to(device=device, dtype=torch.bfloat16)
            for shape in ((3, 32), (64, 3), (3, 32), (32, 3))
        )
        if config.use_lora
        else None
    )
    return models.RoutedExpertMLP(
        config, backend=backend, w1=weights[0], w2=weights[1], adapters=adapters
    )


@pytest.mark.parametrize("mode", ["base_only", "lora_only", "combined"])
def test_cpu_wiring_zero_weight_recompute_and_selected_slot_gradients(mode):
    ref_module = dependency("rl_engine.reference.moe.expert_training")
    calls = []

    class CpuRoute(RocmRouteWeightGradient):
        @staticmethod
        def reduce(dh, silu, up):
            calls.append((dh.clone(), silu.clone(), up.clone()))
            return reference(dh, silu, up)

    class CpuExpert(ref_module.ReferenceExpertTrainingBackend):
        def activation_backward(self, dh, gate, up, p, state, need_route_grad):
            dg, du, _ = super().activation_backward(dh, gate, up, p, state, False)
            dp = CpuRoute().backward(dh, gate, up, p, state) if need_route_grad else None
            return dg, du, dp

    module = make_module(CpuExpert(), mode, "cpu")
    x = torch.linspace(-1, 1, 128).reshape(4, 32).bfloat16()
    offsets = torch.tensor([0, 1, 4])
    slots = torch.tensor([4, 0, 3, 1])
    p_all = torch.tensor([0.0, 1e-30, 0.9, 0.37, 1.0], requires_grad=True)
    results = []
    for recompute in (False, True):
        module.recompute = recompute
        y = module(x, p_all[slots], offsets)
        grads = torch.autograd.grad(y.sum(), (p_all, *module.parameters()))
        results.append((y, *grads))
        dh, silu, up = calls[-1]
        dp64 = ((dh.double() * silu.double()) * up.double().clamp(-10, 10)).sum(1)
        torch.testing.assert_close(grads[0][slots].double(), dp64, rtol=2e-5, atol=2e-6)
        assert grads[0].dtype == torch.float32 and grads[0][2] == 0
        assert grads[0][0] != 0  # p_s=0 still receives a route gradient.
    for a, b in zip(*results):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_gpu_lora_training_uses_real_route_gradient_adapter():
    available = bool(torch.version.hip and torch.cuda.is_available())
    if not available:
        if os.getenv("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("ROCm GPU required for P5 integration")
        pytest.skip("no ROCm GPU")
    backend_module = dependency("rl_engine.backends.rocm.moe.expert_training", required=True)
    dependency("rl_engine.backends.rocm.moe.lora_delta", required=True)
    dependency("rl_engine.backends.rocm.moe.clamp_swiglu", required=True)
    route = RocmRouteWeightGradient()
    seen = []

    class ObservedRoute(RocmRouteWeightGradient):
        def backward(self, dh, gate, up, p, state):
            result = route.backward(dh, gate, up, p, state)
            seen.append((dh, state.silu, up, result))
            return result

    module = make_module(
        backend_module.RocmExpertTrainingBackend(route_gradient=ObservedRoute()),
        "lora_only",
        "cuda",
    )
    x = torch.linspace(-1, 1, 128, device="cuda").reshape(4, 32).bfloat16().requires_grad_()
    p = torch.tensor([0.0, 1e-30, 0.37, 1.0], device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 1, 4], device="cuda")
    results = []
    for recompute in (False, True):
        module.recompute = recompute
        y = module(x, p, offsets)
        grads = torch.autograd.grad(y.sum(), (x, p, *module.parameters()))
        dh, silu, up, dp = seen[-1]
        expected = reference(dh.cpu(), silu.cpu(), up.cpu())
        assert torch.equal(grads[1].cpu().view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(grads[1].view(torch.uint8), dp.view(torch.uint8))
        assert grads[1][0] != 0
        results.append((y, *grads))
    for a, b in zip(*results):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
