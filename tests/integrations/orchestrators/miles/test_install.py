# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib

import pytest
import torch

from rl_engine.integrations.common.state import (
    clear_active_integration,
    get_active_integration,
)
from rl_engine.integrations.orchestrators.miles.config import MilesConfig
from rl_engine.integrations.orchestrators.miles.install import (
    CUSTOM_MEGATRON_INIT_PATH,
    custom_megatron_init,
    install_from_config,
)
from rl_engine.runtime.plan import IntegrationPlan

_NATIVE_PLAN = IntegrationPlan.from_case_ids(attention="P/P", ffn="P/P", logp="P/P")


class _FakeAttention(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states


@pytest.fixture(autouse=True)
def _clean_megatron_registry():
    clear_active_integration("megatron")
    yield
    clear_active_integration("megatron")


def _enabled_config(plan: IntegrationPlan, readback_dir: str | None = None) -> MilesConfig:
    return MilesConfig(enabled=True, plan=plan, readback_dir=readback_dir)


def _disabled_config() -> MilesConfig:
    return MilesConfig(enabled=False, plan=_NATIVE_PLAN, readback_dir=None)


def test_disabled_config_keeps_process_native(tmp_path):
    config = MilesConfig(enabled=False, plan=_NATIVE_PLAN, readback_dir=str(tmp_path))

    assert install_from_config(config) is None
    assert get_active_integration("megatron") is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_plan_installs_and_registers():
    integration = install_from_config(
        _enabled_config(_NATIVE_PLAN), attention_classes=[], ffn_classes=[]
    )

    assert integration is not None
    assert integration.plan == _NATIVE_PLAN
    assert get_active_integration("megatron") is integration


def test_reinstall_same_plan_is_idempotent():
    config = _enabled_config(_NATIVE_PLAN)

    first = install_from_config(config, attention_classes=[], ffn_classes=[])
    second = install_from_config(config, attention_classes=[], ffn_classes=[])

    assert first is second


def test_reinstall_different_plan_raises():
    install_from_config(_enabled_config(_NATIVE_PLAN), attention_classes=[], ffn_classes=[])
    rerouted = IntegrationPlan.from_case_ids(attention="P/R", ffn="P/P", logp="P/P")

    with pytest.raises(RuntimeError, match="already installed with another plan"):
        install_from_config(_enabled_config(rerouted), attention_classes=[], ffn_classes=[])


def test_disabled_config_conflicts_with_active_integration():
    install_from_config(_enabled_config(_NATIVE_PLAN), attention_classes=[], ffn_classes=[])

    with pytest.raises(RuntimeError, match="disabled but a Megatron integration"):
        install_from_config(_disabled_config())


def test_custom_megatron_init_disabled_stays_native(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "0")
    monkeypatch.delenv("RL_KERNEL_READBACK_DIR", raising=False)

    assert custom_megatron_init(None) is None
    assert get_active_integration("megatron") is None


def test_custom_megatron_init_enabled_installs(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "1")
    monkeypatch.setenv("RL_KERNEL_ATTENTION_CASE", "P/P")
    monkeypatch.setenv("RL_KERNEL_FFN_CASE", "P/P")
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "P/P")
    monkeypatch.delenv("RL_KERNEL_READBACK_DIR", raising=False)

    integration = custom_megatron_init(None)

    assert integration is not None
    assert get_active_integration("megatron") is integration


def test_rl_kernel_logprob_records_hook_and_readback_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_KERNEL_READBACK_DIR", str(tmp_path))
    plan = IntegrationPlan.from_case_ids(attention="P/P", ffn="P/P", logp="R/P")

    integration = install_from_config(
        _enabled_config(plan), attention_classes=[], ffn_classes=[]
    )

    hooks = integration.readback()["installed_hooks"]
    assert "logp" in hooks
    assert "attention" not in hooks
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("megatron-training-")


@pytest.mark.skipif(torch.cuda.is_available(), reason="strict GEMM extension is available")
def test_rl_kernel_attention_install_fails_closed_without_extension():
    plan = IntegrationPlan.from_case_ids(attention="R/P", ffn="P/P", logp="P/P")

    with pytest.raises(RuntimeError, match="refusing non-strict fallback"):
        install_from_config(
            _enabled_config(plan), attention_classes=[_FakeAttention], ffn_classes=[]
        )


def test_custom_megatron_init_path_matches_miles_loader_format():
    module_path, separator, attribute = CUSTOM_MEGATRON_INIT_PATH.rpartition(".")
    assert separator == "."

    module = importlib.import_module(module_path)
    resolved = getattr(module, attribute)

    assert resolved is custom_megatron_init
