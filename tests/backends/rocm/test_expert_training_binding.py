# SPDX-License-Identifier: Apache-2.0
"""CPU mocks check dependency gates and the task 45/46/49 calling convention."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.moe import expert_training as binding
from rl_engine.contracts.operators.expert_training import (
    ExpertActivationState,
    ExpertTrainingConfig,
)


@pytest.fixture
def rocm_metadata(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "mock-rocm")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx942:sramecc+"),
    )


def test_real_binding_rejects_cpu_even_without_base():
    backend = binding.RocmExpertTrainingBackend()
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        backend.validate(ExpertTrainingConfig(4, 3, 1, "lora_only"), torch.device("cpu"), False)


def test_missing_base_and_route_providers_fail_before_kernel_load(rocm_metadata):
    backend = binding.RocmExpertTrainingBackend()
    with pytest.raises(RuntimeError, match="tasks 43/44"):
        backend.validate(ExpertTrainingConfig(32, 32, 1), torch.device("cuda"), False)
    with pytest.raises(RuntimeError, match="task 49"):
        backend.validate(ExpertTrainingConfig(4, 3, 1, "lora_only"), torch.device("cuda"), True)


def test_unknown_arch_and_non_identity_ste_are_rejected(rocm_metadata, monkeypatch):
    config = ExpertTrainingConfig(32, 32, 1)
    backend = binding.RocmExpertTrainingBackend(
        base=SimpleNamespace(profile="test", quant_backward="non-identity")
    )
    with pytest.raises(ValueError, match="identity STE"):
        backend.validate(config, torch.device("cuda"), False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx900")
    )
    with pytest.raises(RuntimeError, match="gfx942/gfx950"):
        backend.validate(config, torch.device("cuda"), False)


def test_missing_dependency_is_actionable():
    with pytest.raises(RuntimeError, match="45 / #601 dependency unavailable"):
        binding._load("rl_engine.does_not_exist", "NoSuchClass", "45 / #601")


def test_pinned_task45_and_task46_profiles_are_checked(rocm_metadata, monkeypatch):
    def wrong_activation(module, symbol, task):
        return "different-profile" if symbol == "PROFILE" else object

    monkeypatch.setattr(binding, "_load", wrong_activation)
    backend = binding.RocmExpertTrainingBackend()
    config = ExpertTrainingConfig(4, 3, 1, "lora_only")
    with pytest.raises(RuntimeError, match="task 46 numerical profile"):
        backend.validate(config, torch.device("cuda"), False)

    def wrong_lora(module, symbol, task):
        if symbol == "PROFILE":
            return "dsv4-clamp-swiglu-rocm-fp32-v1"
        return SimpleNamespace(numeric_profile="different-profile")

    monkeypatch.setattr(binding, "_load", wrong_lora)
    with pytest.raises(RuntimeError, match="task 45 numerical profile"):
        backend.validate(config, torch.device("cuda"), False)


def test_activation_state_and_required_route_gradient_are_delegated(monkeypatch):
    seen = []
    x = torch.ones(2, 3)
    p = torch.ones(2)
    state = ExpertActivationState(x * 0.5, x, x * 2)

    class Activation:
        @staticmethod
        def forward(gate, up, weight):
            seen.append(("forward", gate, up, weight))
            return x.bfloat16(), state

        @staticmethod
        def backward(dh, gate, up, weight, saved):
            assert isinstance(saved, ExpertActivationState)
            seen.append(("backward", dh, saved))
            return x * 3, x * 4

    class Route:
        profile = "test-route"

        def backward(self, dh, gate, up, weight, saved):
            seen.append(("route", dh, saved))
            return p * 5

    monkeypatch.setattr(
        binding,
        "_load",
        lambda module, symbol, task: (
            ExpertActivationState if symbol == "ClampSwiGLUState" else Activation
        ),
    )
    backend = binding.RocmExpertTrainingBackend(route_gradient=Route())
    h, saved = backend.activation_forward(x, x, p)
    dg, du, dp = backend.activation_backward(h, x, x, p, saved, True)
    assert torch.equal(dg, x * 3) and torch.equal(du, x * 4) and torch.equal(dp, p * 5)
    assert [call[0] for call in seen] == ["forward", "backward", "route"]
    backend.route_gradient = None
    _, _, dp = backend.activation_backward(h, x, x, p, saved, False)
    assert dp is None
    with pytest.raises(RuntimeError, match="task 49"):
        backend.activation_backward(h, x, x, p, saved, True)
