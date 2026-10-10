# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest

import envs


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", "on"))
def test_env_flag_truthy_values(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_TEST_FLAG", value)

    assert envs.env_flag("RL_KERNEL_TEST_FLAG")


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "no", "off"))
def test_env_flag_falsey_values(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_TEST_FLAG", value)

    assert not envs.env_flag("RL_KERNEL_TEST_FLAG", default=True)


def test_env_flag_uses_default_for_missing_value(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_TEST_FLAG", raising=False)

    assert envs.env_flag("RL_KERNEL_TEST_FLAG", default=True)
    assert not envs.env_flag("RL_KERNEL_TEST_FLAG", default=False)


def test_env_flag_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_TEST_FLAG", "maybe")

    with pytest.raises(ValueError, match="RL_KERNEL_TEST_FLAG"):
        envs.env_flag("RL_KERNEL_TEST_FLAG")
