# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fail-closed protocol for intermediates exported by a native fused kernel."""

from __future__ import annotations

from typing import Callable, Sequence

from .events import EventCapture


class FusedCapture:
    """Observe debug outputs from ONE real kernel execution.

    The callback receives ``emit(stage, actual_tensor_or_tree)``. A kernel with
    debug-output buffers can emit them after launch; an ordinary black-box kernel
    cannot expose its internals and must not use a decomposed reference forward
    to invent them. Debug kernels can perturb scheduling/performance; this API
    certifies only the instrumented execution, not unobserved production code.
    """

    @staticmethod
    def boundaries(prefix: str, stages: Sequence[str]) -> list[dict]:
        stages = list(stages)
        if (
            not stages
            or len(set(stages)) != len(stages)
            or any(not isinstance(s, str) or not s or s in ("input", "output") for s in stages)
        ):
            raise ValueError("declare unique internal stages, excluding input/output")
        return [
            {
                "id": f"{prefix}.{stage}",
                "module": prefix,
                "stage": stage,
                "contracts": ["probe_contract"],
            }
            for stage in ["input", *stages, "output"]
        ]

    def __init__(
        self,
        capture: EventCapture,
        prefix: str,
        stages: Sequence[str],
        *,
        kernel_id: str,
        probe_contract: str,
    ):
        if not kernel_id or not probe_contract:
            raise ValueError("actual kernel ID and shared semantic probe contract are required")
        specs = self.boundaries(prefix, stages)
        if any(spec["id"] not in capture.specs for spec in specs):
            raise ValueError("fused stages differ from prepared scope")
        self.capture, self.prefix, self.stages = capture, prefix, list(stages)
        self.kernel_id, self.probe_contract = kernel_id, probe_contract
        self.calls = 0

    def run(self, execute: Callable, *, inputs, frame: str | None = None):
        frame = f"kernel:{self.calls}" if frame is None else frame
        self.calls += 1
        contracts = {"probe_contract": self.probe_contract}
        seen = []
        try:
            parent = self.capture.record(
                f"{self.prefix}.input",
                inputs,
                frame=frame,
                route=self.kernel_id,
                contracts=contracts,
            )

            def emit(stage, value):
                nonlocal parent
                if len(seen) >= len(self.stages) or stage != self.stages[len(seen)]:
                    self.capture.errors.append("missing, repeated or out-of-order fused probe")
                    raise ValueError("fused probes must follow the declared native stage order")
                parent = self.capture.record(
                    f"{self.prefix}.{stage}",
                    value,
                    frame=frame,
                    inputs=[parent],
                    route=self.kernel_id,
                    contracts=contracts,
                )
                seen.append(stage)

            result = execute(emit)
            if seen != self.stages:
                raise ValueError("kernel did not export every declared internal probe")
            self.capture.record(
                f"{self.prefix}.output",
                result,
                frame=frame,
                inputs=[parent],
                route=self.kernel_id,
                contracts=contracts,
            )
            return result
        except Exception as exc:
            self.capture.errors.append(f"fused kernel observation failed: {exc}")
            raise
