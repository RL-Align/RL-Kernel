# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import pytest

from rl_engine.integrations.common.state import get_active_integration
from rl_engine.integrations.orchestrators.miles.config import MilesConfig
from rl_engine.runtime.plan import Implementation


@pytest.fixture(autouse=True)
def clear_integration_environment(monkeypatch):
    for name in (
        "RL_KERNEL_MILES_INTEGRATION",
        "RL_KERNEL_ATTENTION_CASE",
        "RL_KERNEL_FFN_CASE",
        "RL_KERNEL_LOGP_CASE",
        "RL_KERNEL_READBACK_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_config_is_disabled_with_native_plan():
    config = MilesConfig.from_environment()

    assert config.enabled is False
    assert config.readback_dir is None
    assert {name: case.case_id for name, case in config.plan.cases.items()} == {
        "attention": "P/P",
        "ffn": "P/P",
        "logp": "P/P",
    }
    with pytest.raises(FrozenInstanceError):
        config.enabled = True


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        (" off ", False),
    ],
)
def test_enable_flag(monkeypatch, value, enabled):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", value)
    assert MilesConfig.from_environment().enabled is enabled


@pytest.mark.parametrize("value", ["", "2", "enabled", "truth"])
def test_invalid_enable_flag_is_rejected(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", value)
    with pytest.raises(ValueError, match="RL_KERNEL_MILES_INTEGRATION"):
        MilesConfig.from_environment()


@pytest.mark.parametrize("module", ["attention", "ffn", "logp"])
@pytest.mark.parametrize(
    ("case", "training", "rollout"),
    [
        ("P/P", Implementation.PRODUCTION, Implementation.PRODUCTION),
        ("P/R", Implementation.PRODUCTION, Implementation.RL_KERNEL),
        ("R/P", Implementation.RL_KERNEL, Implementation.PRODUCTION),
        ("R/R", Implementation.RL_KERNEL, Implementation.RL_KERNEL),
    ],
)
def test_plan_routes_each_module(monkeypatch, module, case, training, rollout):
    monkeypatch.setenv(f"RL_KERNEL_{module.upper()}_CASE", case)
    config = MilesConfig.from_environment()

    assert config.plan.implementation_for(module, "training") is training
    assert config.plan.implementation_for(module, "rollout") is rollout
    for other in {"attention", "ffn", "logp"} - {module}:
        assert config.plan.cases[other].case_id == "P/P"


def test_configuration_has_no_install_or_filesystem_side_effects(monkeypatch, tmp_path):
    readback_dir = tmp_path / "not-created"
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "1")
    monkeypatch.setenv("RL_KERNEL_READBACK_DIR", str(readback_dir))
    environment = dict(os.environ)
    active_megatron = get_active_integration("megatron")
    active_vllm = get_active_integration("vllm")

    config = MilesConfig.from_environment()

    assert config.enabled is True
    assert config.readback_dir == str(readback_dir)
    assert not readback_dir.exists()
    assert dict(os.environ) == environment
    assert get_active_integration("megatron") is active_megatron
    assert get_active_integration("vllm") is active_vllm
