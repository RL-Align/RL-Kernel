# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Operator-independent CPU fixtures; ROCm profiles use mocked completion."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from rl_engine.contracts.strict_provider import (
    KernelPlan,
    ProviderLock,
    SnapshotActivationReceipt,
    fingerprint,
)
from rl_engine.validation.common.strict_provider import (
    ARTIFACTS_SCHEMA,
    CASES_SCHEMA,
    SUMMARY_SCHEMA,
    artifact_reference,
    build_expectation,
    validate_run,
)


def make_plan(node="test_operator", *, dtype="float64", accumulator_dtype="float64"):
    return KernelPlan.from_dict(
        {
            "node": node,
            "phase": "prefill",
            "implementation": "tests.MockOperator",
            "entrypoint": f"tests.MockOperator.{node}",
            "code": {
                "source_inventory_sha256": "a" * 64,
                "binary_inventory_sha256": "b" * 64,
                "compiled_kernel_id": "mock_binary_v1",
                "compile_key": "fixed_shape_v1",
            },
            "io": {
                "input_dtype": dtype,
                "output_dtype": dtype,
                "layout": "row_major",
                "shape_class": "two_rows",
                "workspace_policy": "none",
            },
            "numerics": {
                "accumulator_dtype": accumulator_dtype,
                "split_policy": "disabled",
                "split_count": 1,
                "reduction_spec": "left_fold_v1",
                "cast_spec": "output_cast" if dtype != accumulator_dtype else "no_cast",
                "atomic_policy": "forbidden",
            },
            "launch": {"tile_config": [1], "graph_state": "eager", "tuning": "fixed"},
        }
    )


class Bundle:
    """Trusted predeclared runner plus separately persisted mock worker evidence."""

    def __init__(self, root: Path, *, node="test_operator", platform_kind="cpu"):
        self.root = root
        self.node = node
        is_rocm = platform_kind == "rocm"
        self.completion_kind = "mock_event" if is_rocm else "synchronous"
        dtype = "bfloat16" if is_rocm else "float64"
        accumulator_dtype = "float32" if is_rocm else "float64"
        self.directory = root / "run1" / "attempt1"
        self.directory.mkdir(parents=True)
        self.original = self.directory / "provider.bin"
        self.original.write_bytes(b"original mock provider")
        original_ref = artifact_reference(self.original, root)
        self.artifacts_id = fingerprint(
            [
                {
                    "path": "provider.bin",
                    "size": original_ref["size_bytes"],
                    "sha256": original_ref["sha256"],
                }
            ]
        )
        plan = make_plan(node, dtype=dtype, accumulator_dtype=accumulator_dtype).to_dict()
        plan["code"]["source_inventory_sha256"] = self.artifacts_id
        plan["code"]["binary_inventory_sha256"] = self.artifacts_id
        plan.pop("plan_id")
        plan.pop("alignment_id")
        self.plan = KernelPlan.from_dict(plan)
        binding = {"implementation": self.plan.implementation, "artifacts_ref": self.artifacts_id}
        self.lock = ProviderLock.from_dict(
            {
                "schema_version": "rl_kernel.strict_provider.v1",
                "profile_id": f"{platform_kind}_mock",
                "scope": f"component.{node}",
                "model": {
                    "model_id": "mock_model",
                    "revision": "revision1",
                    "config_sha256": "1" * 64,
                    "weight_index_sha256": "2" * 64,
                    "shard_inventory_sha256": "3" * 64,
                },
                "platform": {
                    "kind": platform_kind,
                    "gpu_arch": "gfx942" if is_rocm else "cpu_mock",
                    "torch_build": "cpu_mock",
                    "hip_runtime": "mock_hip" if is_rocm else "not_applicable",
                    "driver_identity": "cpu_mock",
                    "triton_build": "not_applicable",
                },
                "execution": {
                    "topology": {"tp": 1, "ep": 1, "cp": 1, "pp": 1},
                    "activation_dtype": dtype,
                    "graph_mode": "eager",
                    "tuning_policy": "fixed",
                    "fallback_policy": "error",
                    "runtime_settings": {},
                },
                "nodes": {
                    node: {
                        "semantics_ref": f"mock_{node}_v1",
                        "bindings": {
                            "rollout": binding,
                            "replay": binding,
                        },
                        "allowed_plans": {
                            "rollout": [self.plan.plan_id],
                            "replay": [self.plan.plan_id],
                        },
                        "alignment": {"policy": "identical_plan"},
                    }
                },
            }
        )
        self.inventory = {"0": [node], "1": [node]}
        self.scope = {
            "scope_id": f"component.{node}",
            "kind": "component",
            "required_nodes": [node],
            "inventory_fingerprint": fingerprint(self.inventory),
        }
        self.case = {
            "case_id": "prompt_boundary",
            "comparison_context": {
                "tokens": [1, 2],
                "positions": [0, 1],
                "mask": [1, 1],
                "state_boundary": "after_token_1",
                "logprob_semantics": "raw_policy",
            },
            "required_calls": [
                {
                    "node": node,
                    "phase": "prefill",
                    "layer_id": str(layer),
                    "call_id": "prompt_A",
                    "role": role,
                    "rank": 0,
                    "count": 1,
                }
                for role in ("rollout", "replay")
                for layer in range(2)
            ],
        }
        self.roster = [
            {"worker_id": role + "0", "role": role, "rank": 0} for role in ("rollout", "replay")
        ]
        self.artifact_inventory = {
            "schema_version": ARTIFACTS_SCHEMA,
            **self.context(),
            "inventory_sha256": self.artifacts_id,
            "files": [{"logical_path": "provider.bin", "reference": original_ref}],
        }
        inventory_ref = self.write("artifact_inventory.json", self.artifact_inventory)
        self.expectation = build_expectation(
            self.lock,
            self.scope,
            [self.case],
            self.roster,
            self.inventory,
            run_id="run1",
            attempt="attempt1",
            weight_snapshot_id="snapshot1",
            identity_artifacts=[inventory_ref],
        )
        self.summaries = []
        for worker in self.roster:
            observation = {
                **worker,
                "model": self.lock.to_dict()["model"],
                "platform": self.lock.to_dict()["platform"],
                "execution": self.lock.to_dict()["execution"],
                "bindings": {node: binding},
                "weight_snapshot_id": "snapshot1",
            }
            receipt = SnapshotActivationReceipt.from_dict(
                {
                    "snapshot_id": "snapshot1",
                    "generation": 0,
                    "worker_id": worker["worker_id"],
                    "expected_shards": ["weight0"],
                    "verified_shards": ["weight0"],
                    "required_refresh": [],
                    "refreshed_components": [],
                    "content_digest": "d" * 64,
                    "completed": True,
                    "evidence_kind": "worker_load",
                }
            ).to_dict()
            summary = {
                "schema_version": SUMMARY_SCHEMA,
                **self.context(),
                **worker,
                "state": "finished",
                "snapshot_generation": 0,
                "snapshot_history": [receipt],
                "bindings": {node: binding},
                "observations": {node: observation},
                "calls": [],
            }
            for layer in range(2):
                summary["calls"].append(
                    {
                        "context": {
                            **{k: v for k, v in self.context().items() if k != "lock_id"},
                            "case_id": self.case["case_id"],
                            "node": node,
                            "phase": "prefill",
                            "layer_id": str(layer),
                            "call_id": "prompt_A",
                            "role": worker["role"],
                            "worker_id": worker["worker_id"],
                            "metadata": {
                                "comparison_context": deepcopy(self.case["comparison_context"])
                            },
                        },
                        "plan": self.plan.to_dict(),
                        "plan_id": self.plan.plan_id,
                        "alignment_id": self.plan.alignment_id,
                        "snapshot_generation": 0,
                        "runtime_id": f"runtime_{worker['role']}_{layer}",
                        "call_ticket_id": f"ticket_{worker['role']}_{layer}",
                        "status": "completed",
                        "states": ["prepared", "authorized", "launched", "completed"],
                        "completion": {"kind": self.completion_kind, "status": "completed"},
                    }
                )
            self.summaries.append(summary)
        self.report = {
            "schema_version": CASES_SCHEMA,
            **self.context(),
            "cases": [
                {
                    "case_id": self.case["case_id"],
                    "outcome": "passed",
                    "tests": 1,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                }
            ],
        }
        self.save()

    def context(self):
        return {
            "lock_id": self.lock.lock_id,
            "run_id": "run1",
            "attempt": "attempt1",
            "weight_snapshot_id": "snapshot1",
        }

    def write(self, name, payload):
        path = self.directory / name
        path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        return artifact_reference(path, self.root)

    def save(self, *, update_footer=True):
        for summary in self.summaries:
            if update_footer:
                summary["footer"] = {"complete": True, "call_count": len(summary["calls"])}
            summary.pop("summary_id", None)
            for call in summary["calls"]:
                call["context_id"] = fingerprint(call["context"])
            summary["summary_id"] = fingerprint(summary)
        self.summary_refs = [
            self.write(f"worker_{i}.json", s) for i, s in enumerate(self.summaries)
        ]
        self.report_refs = [self.write("cases.json", self.report)]

    def validate(self):
        return validate_run(
            self.lock,
            self.expectation,
            self.summary_refs,
            self.root,
            case_results=self.report_refs,
        )

    def cli_input(self):
        return {
            "lock": self.lock.to_dict(),
            "expectation": self.expectation,
            "worker_summaries": self.summary_refs,
            "case_results": self.report_refs,
        }
