# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Select complete training/rollout recipes, never independent fastest operators.

This module has no framework or GPU imports. Catalog entries are reviewed Python
code shipped with RL-Kernel; environment variables cannot add implementations or
benchmark results to the catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    # Canonical JSON makes profiles immutable, including nested config values.
    config_match: str

    def matches(self, config: dict[str, Any]) -> bool:
        return all(config.get(key) == value for key, value in json.loads(self.config_match).items())


@dataclass(frozen=True)
class RuntimeContext:
    model_id: str
    framework: str
    platform: str
    hardware: tuple[str, ...]
    topology: tuple[int, int, int, int, int]  # world, train TP/CP, rollout TP/CP
    model_config: str
    runtime: str
    workload: str
    route_environment: str

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    model_id: str
    framework: str
    platform: str
    # The adapter is a shipped implementation, not a user-supplied import path.
    adapter: str
    environment: tuple[tuple[str, str], ...]
    evidence: tuple[str, ...]
    reference: bool = False

    @property
    def digest(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Benchmark:
    """One comparable E2E experiment, including its incumbent reference.

    All participating plans must pass WS1, WS2, and E2E equality before the
    report is registered. Seconds are the same E2E metric in the same workload;
    no microbenchmark score or unmatched hardware can enter selection.
    """

    context_digest: str
    report: str
    # (plan content digest, E2E seconds); the reference must be included.
    timings: tuple[tuple[str, float], ...]
    ws1_passed: bool
    ws2_passed: bool
    e2e_mismatch_count: int
    e2e_max_abs_diff: float
    compared_elements: int

    def validate(self) -> None:
        if (
            not self.report
            or not self.ws1_passed
            or not self.ws2_passed
            or self.e2e_mismatch_count != 0
            or self.e2e_max_abs_diff != 0.0
            or self.compared_elements <= 0
        ):
            raise ValueError(
                "BI benchmarks require WS1/WS2 and nonempty exact E2E equality evidence"
            )
        digests = [digest for digest, _ in self.timings]
        if len(digests) < 2 or len(set(digests)) != len(digests):
            raise ValueError("benchmark must contain distinct reference and candidate plans")
        if any(not math.isfinite(seconds) or seconds <= 0 for _, seconds in self.timings):
            raise ValueError("benchmark E2E seconds must be finite and positive")


class Catalog:
    def __init__(self) -> None:
        self.models: dict[str, ModelProfile] = {}
        self.plans: dict[str, ExecutionPlan] = {}
        self.benchmarks: dict[str, Benchmark] = {}

    def register_model(self, profile: ModelProfile) -> None:
        if profile.model_id in self.models:
            raise ValueError(f"duplicate model profile: {profile.model_id}")
        if not json.loads(profile.config_match):
            raise ValueError("model profiles must specify config fields")
        self.models[profile.model_id] = profile

    def identify(self, config: dict[str, Any]) -> str:
        matches = [key for key, profile in self.models.items() if profile.matches(config)]
        if len(matches) != 1:
            raise ValueError(
                f"RL_KERNEL_BI needs exactly one supported model profile; got {matches}"
            )
        return matches[0]

    def register_plan(self, plan: ExecutionPlan) -> None:
        if plan.plan_id in self.plans:
            raise ValueError(f"duplicate BI plan: {plan.plan_id}")
        if plan.model_id not in self.models or not plan.evidence:
            raise ValueError("BI plans require a registered model and reviewable evidence")
        keys = [key for key, _ in plan.environment]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate plan environment key")
        if any(not isinstance(value, str) for _, value in plan.environment):
            raise ValueError("plan environment values must be strings")
        if any(
            not key.startswith(("RL_KERNEL_", "VLLM_", "NVTE_", "CUBLAS", "NCCL_"))
            or key in {"RL_KERNEL_BI", "RL_KERNEL_BI_PLAN", "RL_KERNEL_RUN_ID"}
            or key.endswith(("_DIR", "_ROOT", "_PYTHON"))
            for key in keys
        ):
            raise ValueError(
                "plan environment must contain only numerical/runtime routing settings"
            )
        self.plans[plan.plan_id] = plan

    def register_benchmark(self, benchmark: Benchmark) -> None:
        benchmark.validate()
        if benchmark.context_digest in self.benchmarks:
            raise ValueError("replace the reviewed comparison; do not mix benchmark cohorts")
        known = {plan.digest for plan in self.plans.values()}
        if any(digest not in known for digest, _ in benchmark.timings):
            raise ValueError("benchmark refers to an unknown or modified execution plan")
        self.benchmarks[benchmark.context_digest] = benchmark

    def select(self, context: RuntimeContext) -> tuple[ExecutionPlan, str]:
        compatible = {
            plan.digest: plan
            for plan in self.plans.values()
            if (plan.model_id, plan.framework, plan.platform)
            == (context.model_id, context.framework, context.platform)
        }
        references = [plan for plan in compatible.values() if plan.reference]
        if len(references) != 1:
            raise ValueError("BI selection requires exactly one shipped reference for this stack")
        reference = references[0]
        benchmark = self.benchmarks.get(context.digest)
        if benchmark is None:
            return reference, "shipped_reference; no comparable candidate benchmark"
        benchmark.validate()
        timings = dict(benchmark.timings)
        if reference.digest not in timings or not set(timings) <= compatible.keys():
            raise ValueError("benchmark must compare compatible plans including the reference")
        # Equal performance keeps the incumbent. A stable ID breaks candidate ties.
        winner = min(
            timings,
            key=lambda digest: (
                timings[digest],
                digest != reference.digest,
                compatible[digest].plan_id,
            ),
        )
        return compatible[winner], f"lowest E2E seconds in {benchmark.report}"
