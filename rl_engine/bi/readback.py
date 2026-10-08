# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate that training and rollout workers used the same launch plan."""

from typing import Any, Mapping, Sequence


def validate_plan_readbacks(
    readbacks: Sequence[Mapping[str, Any]], expected: Mapping[str, Any] | None = None
) -> list[str]:
    workers = [value for value in readbacks if value.get("framework") in {"megatron", "vllm"}]
    plans = [value.get("bi_plan") for value in workers]
    if expected is None and not any(plan is not None for plan in plans):
        return []  # Existing non-BI runs retain their validation contract.
    if not workers or any(not isinstance(plan, Mapping) for plan in plans):
        return ["BI readbacks are missing from one or more training/rollout workers"]
    if {value.get("framework") for value in workers} != {"megatron", "vllm"}:
        return ["BI readbacks must cover both training and rollout"]
    reference = expected if expected is not None else plans[0]
    keys = ("plan_id", "plan_digest", "context_digest", "digest")
    if any(not isinstance(reference.get(key), str) or not reference[key] for key in keys):
        return ["BI readback has an incomplete plan identity"]
    if any(any(plan.get(key) != reference[key] for key in keys) for plan in plans):
        return ["BI training/rollout readbacks disagree with the selected launch plan"]
    return []
