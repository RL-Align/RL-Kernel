# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Opt-in registration for temporary Cross-configuration alignment smoke-only operators."""

from __future__ import annotations

from typing import Any

from rl_engine.runtime.semantic_registry import OperatorBackendDescriptor, SemanticOperatorCatalog

SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID = "smoke_only.logp_reference"
SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID = "smoke_only.logp_offset"


def smoke_operator_descriptors() -> tuple[OperatorBackendDescriptor, ...]:
    """Build smoke descriptors lazily without registering them globally."""

    from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_offset import (
        SMOKE_ONLY_LOGP_OFFSET_DESCRIPTOR,
    )
    from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_reference import (
        SMOKE_ONLY_LOGP_REFERENCE_DESCRIPTOR,
    )

    return (
        SMOKE_ONLY_LOGP_REFERENCE_DESCRIPTOR,
        SMOKE_ONLY_LOGP_OFFSET_DESCRIPTOR,
    )


def register_smoke_operators(
    catalog: SemanticOperatorCatalog,
    *,
    allow_smoke_operators: bool = False,
    replace: bool = False,
) -> tuple[OperatorBackendDescriptor, ...]:
    """Register every smoke backend after an explicit per-call opt-in.

    Importing this package never mutates a catalog. Resolution independently
    requires an ``OperatorResolutionPolicy`` that allows test backends; this
    registration guard is the first fail-closed boundary.
    """

    if allow_smoke_operators is not True:
        raise PermissionError(
            "smoke operator registration requires explicit " "allow_smoke_operators=True"
        )
    if not isinstance(catalog, SemanticOperatorCatalog):
        raise TypeError("catalog must be a SemanticOperatorCatalog")

    descriptors = smoke_operator_descriptors()
    for descriptor in descriptors:
        catalog.register_backend(descriptor, replace=replace)
    return descriptors


def __getattr__(name: str) -> Any:
    """Lazily expose implementation classes without default torch imports."""

    if name == "SmokeOnlyLogpReference":
        from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_reference import (
            SmokeOnlyLogpReference,
        )

        return SmokeOnlyLogpReference
    if name == "SmokeOnlyLogpOffset":
        from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_offset import (
            SmokeOnlyLogpOffset,
        )

        return SmokeOnlyLogpOffset
    raise AttributeError(name)


__all__ = [
    "SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID",
    "SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID",
    "SmokeOnlyLogpOffset",
    "SmokeOnlyLogpReference",
    "register_smoke_operators",
    "smoke_operator_descriptors",
]
