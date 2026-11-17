# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

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


def _callback_args(**overrides):
    values = {
        "rl_kernel_miles_enabled": None,
        "rl_kernel_attention_case": None,
        "rl_kernel_ffn_case": None,
        "rl_kernel_logp_case": None,
        "rl_kernel_readback_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_resolve_prefers_callback_fields_over_environment(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "0")
    monkeypatch.setenv("RL_KERNEL_ATTENTION_CASE", "R/R")
    monkeypatch.setenv("RL_KERNEL_READBACK_DIR", "/tmp/from-env")

    config = MilesConfig.resolve(
        _callback_args(
            rl_kernel_miles_enabled="1",
            rl_kernel_attention_case="P/R",
            rl_kernel_ffn_case="R/P",
            rl_kernel_logp_case="R/R",
            rl_kernel_readback_dir="/tmp/from-callback",
        )
    )

    assert config.enabled is True
    assert {name: case.case_id for name, case in config.plan.cases.items()} == {
        "attention": "P/R",
        "ffn": "R/P",
        "logp": "R/R",
    }
    assert config.readback_dir == "/tmp/from-callback"


def test_resolve_falls_back_field_by_field(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "1")
    monkeypatch.setenv("RL_KERNEL_FFN_CASE", "R/P")
    monkeypatch.setenv("RL_KERNEL_READBACK_DIR", "/tmp/from-env")

    config = MilesConfig.resolve(_callback_args(rl_kernel_attention_case="P/R"))

    assert config.enabled is True
    assert config.plan.cases["attention"].case_id == "P/R"
    assert config.plan.cases["ffn"].case_id == "R/P"
    assert config.plan.cases["logp"].case_id == "P/P"
    assert config.readback_dir == "/tmp/from-env"


def test_resolve_without_arguments_matches_environment(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "1")
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "R/P")

    resolved = MilesConfig.resolve(None)
    parsed = MilesConfig.from_environment()

    assert resolved.enabled is parsed.enabled
    assert {name: case.case_id for name, case in resolved.plan.cases.items()} == {
        name: case.case_id for name, case in parsed.plan.cases.items()
    }
    assert resolved.readback_dir == parsed.readback_dir


def test_resolve_rejects_invalid_callback_flag():
    with pytest.raises(ValueError, match="rl_kernel_miles_enabled"):
        MilesConfig.resolve(_callback_args(rl_kernel_miles_enabled="maybe"))


def test_resolve_accepts_boolean_callback_flag():
    assert MilesConfig.resolve(_callback_args(rl_kernel_miles_enabled=True)).enabled is True
    assert MilesConfig.resolve(_callback_args(rl_kernel_miles_enabled=False)).enabled is False


def test_resolve_empty_callback_readback_disables_it(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_READBACK_DIR", "/tmp/from-env")

    config = MilesConfig.resolve(_callback_args(rl_kernel_readback_dir=""))

    assert config.readback_dir is None


def test_resolve_strips_callback_readback():
    config = MilesConfig.resolve(_callback_args(rl_kernel_readback_dir=" /tmp/dir "))

    assert config.readback_dir == "/tmp/dir"


def test_resolve_disabled_ignores_invalid_case(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_MILES_INTEGRATION", "0")
    monkeypatch.setenv("RL_KERNEL_ATTENTION_CASE", "invalid")

    config = MilesConfig.resolve(None)

    assert config.enabled is False
    assert config.plan.cases["attention"].case_id == "P/P"
