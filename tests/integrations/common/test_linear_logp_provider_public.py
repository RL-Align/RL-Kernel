# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The logprob provider lives in the common layer; the vime path is a shim."""

from __future__ import annotations

import subprocess
import sys

import pytest

import rl_engine.integrations.common.linear_logp_provider as common_provider
import rl_engine.integrations.orchestrators.vime.providers.linear_logp_provider as vime_provider

_VIME_PACKAGE = "rl_engine.integrations.orchestrators.vime"
_COMMON_PATH = "rl_engine.integrations.common.linear_logp_provider"


def test_vime_provider_path_resolves_to_the_common_module():
    assert vime_provider is common_provider
    assert vime_provider.provider is common_provider.provider
    assert vime_provider.LinearLogpProviderUnavailable is (
        common_provider.LinearLogpProviderUnavailable
    )


def test_legacy_integrations_vime_path_resolves_to_the_common_module():
    import rl_engine.integrations.vime.linear_logp_provider as legacy_provider

    assert legacy_provider is common_provider


def test_megatron_runtime_does_not_import_the_vime_package():
    code = (
        "import sys\n"
        "import rl_engine.integrations.engines.train.megatron.runtime\n"
        f"print('VIME_IMPORTED=' + str({_VIME_PACKAGE!r} in sys.modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    markers = [
        line for line in result.stdout.splitlines() if line.startswith("VIME_IMPORTED=")
    ]
    assert markers == ["VIME_IMPORTED=False"]


def test_megatron_logp_hook_names_the_common_provider():
    from rl_engine.integrations.common.state import clear_active_integration
    from rl_engine.integrations.orchestrators.miles.config import MilesConfig
    from rl_engine.integrations.orchestrators.miles.install import install_from_config
    from rl_engine.runtime.plan import IntegrationPlan

    clear_active_integration("megatron")
    try:
        plan = IntegrationPlan.from_case_ids(attention="P/P", ffn="P/P", logp="R/P")
        integration = install_from_config(
            MilesConfig(enabled=True, plan=plan, readback_dir=None),
            attention_classes=[],
            ffn_classes=[],
        )

        hook = integration.readback()["installed_hooks"]["logp"]
        assert f"{_COMMON_PATH}.provider" in hook
        assert "orchestrators.vime" not in hook
    finally:
        clear_active_integration("megatron")


@pytest.mark.parametrize(
    "module_path",
    ["rl_engine.integrations.common.linear_logp_provider"],
)
def test_common_provider_exposes_the_public_entry_points(module_path):
    module = __import__(module_path, fromlist=["provider"])

    for name in (
        "provider",
        "_provider_impl",
        "LinearLogpResult",
        "LinearLogpProviderUnavailable",
    ):
        assert hasattr(module, name), name
