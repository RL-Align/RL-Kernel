# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Megatron-side operator boundary owned by RL-Kernel."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rl_engine.integrations.common.runtime import FrameworkOperatorIntegration
from rl_engine.runtime.plan import IntegrationPlan


class MegatronIntegration(FrameworkOperatorIntegration):
    def __init__(
        self,
        plan: IntegrationPlan,
        *,
        rl_kernel_operators: Mapping[str, Callable[..., Any]],
    ) -> None:
        super().__init__(
            framework="megatron",
            target="training",
            plan=plan,
            rl_kernel_operators=rl_kernel_operators,
        )

    def attention(self, native: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.execute("attention", native, *args, **kwargs)

    def ffn(self, native: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.execute("ffn", native, *args, **kwargs)

    def logp(self, native: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.execute("logp", native, *args, **kwargs)


__all__ = ["MegatronIntegration"]
