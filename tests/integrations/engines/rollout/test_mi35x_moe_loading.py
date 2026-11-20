# SPDX-License-Identifier: Apache-2.0
"""The MI35x native loader must not leak its selector into other model loads."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.mi35x_moe import native_w4a16_loading


@pytest.fixture
def selection_api(monkeypatch):
    # No vLLM installation or GPU is needed to check scope isolation. Numerical
    # acceptance belongs to the actual four-GPU native recipe, not these doubles.
    leaves = (
        "vllm.model_executor.layers.fused_moe.config",
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
        "vllm.model_executor.layers.quantization.mxfp4",
        "vllm.model_executor.layers.quantization.utils.quant_utils",
    )
    modules = {}
    for name in leaves:
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            key = ".".join(parts[:count])
            if key not in modules:
                module = ModuleType(key)
                module.__path__ = []
                modules[key] = module
                monkeypatch.setitem(sys.modules, key, module)
                if count > 1:
                    setattr(modules[".".join(parts[: count - 1])], parts[count - 1], module)
    config_api, oracle_api, quant_api, keys = (modules[name] for name in leaves)
    config_api.RoutingMethodType = SimpleNamespace(DeepseekV4="dsv4")
    oracle_api.Mxfp4MoeBackend = SimpleNamespace(TRITON_UNFUSED="unfused-w4a16")
    keys.kMxfp4Static = object()
    calls = []

    class Oracle:
        def __init__(self, config, weight_key, activation_key):
            assert weight_key is keys.kMxfp4Static and activation_key is None
            self.config = config

        def return_or_raise(self, backend, *, activation_key):
            assert activation_key is None
            if not self.config.supported:
                raise ValueError("unsupported deployment")
            calls.append(backend)
            return backend, "kernel"

    oracle_api.Mxfp4MoEKernelOracle = Oracle
    quant_api.select_deepseek_v4_mxfp4_moe_backend = lambda config: ("upstream", "kernel")
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx950")
    )
    return quant_api, calls


@pytest.mark.parametrize(
    "routing,backend", [("dsv4", "auto"), ("other", "auto"), ("dsv4", "aiter")]
)
def test_native_selection_is_scoped_and_keeps_explicit_requests(selection_api, routing, backend):
    api, calls = selection_api
    original = api.select_deepseek_v4_mxfp4_moe_backend
    config = SimpleNamespace(routing_method=routing, moe_backend=backend, supported=True)
    before = vars(config).copy()
    with native_w4a16_loading():
        selected, _ = api.select_deepseek_v4_mxfp4_moe_backend(config)
    expected = "unfused-w4a16" if (routing, backend) == ("dsv4", "auto") else "upstream"
    assert selected == expected and vars(config) == before
    assert calls == ([expected] if expected != "upstream" else [])
    assert api.select_deepseek_v4_mxfp4_moe_backend is original


def test_failed_load_restores_selector_and_does_not_fallback(selection_api):
    api, calls = selection_api
    original = api.select_deepseek_v4_mxfp4_moe_backend
    config = SimpleNamespace(routing_method="dsv4", moe_backend="auto", supported=False)
    with pytest.raises(ValueError, match="unsupported deployment"), native_w4a16_loading():
        api.select_deepseek_v4_mxfp4_moe_backend(config)
    assert api.select_deepseek_v4_mxfp4_moe_backend is original and not calls


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_other_architectures_keep_the_original_selector(selection_api, monkeypatch, arch):
    api, _ = selection_api
    original = api.select_deepseek_v4_mxfp4_moe_backend
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=arch)
    )
    with native_w4a16_loading():
        assert api.select_deepseek_v4_mxfp4_moe_backend is original
