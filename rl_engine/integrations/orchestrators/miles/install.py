# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-process installation entry for the miles orchestrator integration.

Miles resolves ``custom_megatron_init_path`` through
``miles.utils.function_registry.load_function`` and calls it on every Megatron
worker after Megatron init and before model construction, passing a merged
args namespace.  ``CUSTOM_MEGATRON_INIT_PATH`` is the path string in that
format and ``custom_megatron_init`` is the function it names.

Settings reach the callback two ways.  The orchestration config is the reliable
one: ``custom_megatron_init_path`` may carry a ``config`` mapping whose fields
Miles validates against ``add_arguments`` below and merges into the callback
args, so no process-inheritance assumption is involved.  The ``RL_KERNEL_*``
environment variables remain as the local fallback.  Disabled configuration
returns before any install call, so the worker keeps its native behavior and
no integration state is created.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Any

from rl_engine.integrations.common.state import get_active_integration
from rl_engine.integrations.engines.train.megatron.runtime import (
    initialize_from_environment,
    install_megatron_integration,
)
from rl_engine.integrations.orchestrators.miles.config import (
    CALLBACK_CASE_FIELDS,
    CALLBACK_ENABLED_FIELD,
    CALLBACK_READBACK_FIELD,
    MilesConfig,
)

CUSTOM_MEGATRON_INIT_PATH = (
    "rl_engine.integrations.orchestrators.miles.install.custom_megatron_init"
)

logger = logging.getLogger(__name__)


def callback_arguments(parser: ArgumentParser) -> None:
    """Declare the fields Miles validates for ``custom_megatron_init_path.config``."""

    parser.add_argument("--rl-kernel-miles-enabled", dest=CALLBACK_ENABLED_FIELD, default=None)
    for module, field in CALLBACK_CASE_FIELDS.items():
        parser.add_argument(f"--rl-kernel-{module}-case", dest=field, default=None)
    parser.add_argument("--rl-kernel-readback-dir", dest=CALLBACK_READBACK_FIELD, default=None)


def custom_megatron_init(args: Any = None) -> Any:
    """Install in this Megatron worker when the callback config or environment enables it."""
    config = MilesConfig.resolve(args)
    if not config.enabled:
        logger.info("miles integration disabled; Megatron worker stays native")
        return None
    return initialize_from_environment(
        args,
        plan=config.plan,
        readback_dir=config.readback_dir,
    )


setattr(custom_megatron_init, "add_arguments", callback_arguments)


def install_from_config(
    config: MilesConfig,
    *,
    attention_classes: Iterable[type[Any]] | None = None,
    ffn_classes: Iterable[type[Any]] | None = None,
) -> Any:
    """Install the Megatron routes described by ``config`` in this process."""
    if not config.enabled:
        if get_active_integration("megatron") is not None:
            raise RuntimeError(
                "RL_KERNEL_MILES_INTEGRATION is disabled but a Megatron "
                "integration is already active in this process"
            )
        return None
    integration = install_megatron_integration(
        config.plan,
        attention_classes=attention_classes,
        ffn_classes=ffn_classes,
        readback_dir=config.readback_dir,
    )
    logger.info(
        "miles integration installed: plan=%s readback_dir=%s",
        {module: case.case_id for module, case in config.plan.cases.items()},
        config.readback_dir,
    )
    return integration


__all__ = [
    "CUSTOM_MEGATRON_INIT_PATH",
    "callback_arguments",
    "custom_megatron_init",
    "install_from_config",
]
