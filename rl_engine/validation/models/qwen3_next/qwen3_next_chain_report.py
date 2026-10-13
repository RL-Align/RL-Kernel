# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3-Next full-model chain as a WS1 chain-gate report.

The Qwen3-Next chain runs as stages
(``tools/validation/models/qwen3_next/qwen3_next_run.py``): L2 on the
pinned weights, then L2 again after each of two optimizer updates and after a
reload. Each L2 stage writes a 16-case raw-bit matrix comparing rollout
(native vLLM prefill + decode) with replay (VIME recompute). This module turns
those matrices into the same report shape as the Qwen3-8B chain gate
(:class:`rl_engine.validation.models.chain_gate.ChainGateReport`), built from the
same dataclasses, so the two models' reports can be read side by side.

Every case of every weight version becomes one cell. Bitwise rollout/replay
equality of logits and selected logprobs becomes one
:class:`TensorComparisonDetail` each, with ``atol = rtol = 0``. The first
divergence names a decoder layer and a sub-node (see
``rl_engine.models.qwen3_next.qwen3_next_subnode``), which this report also maps to
a layer family.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from rl_engine.contracts.numerical import (
    BackendProvenance,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    resolve_comparison_roles,
)
from rl_engine.validation.models.chain_gate import CellOutput, ChainGateReport
from rl_engine.validation.operators.forward_invariance import TensorComparisonDetail

SCHEMA_VERSION = "qwen3-next-chain-v1"
COMPATIBLE_SCHEMA = "ws1-c10-c11-v5"
PROFILE = "qwen3_next_tp4_shared_eager_v1"
WEIGHT_VERSIONS = ("baseline", "update1", "update2", "reload")
FAMILIES = ("embedding", "norm", "gdn", "full_attention", "moe", "lm_head_logprob")
_TRAIN_INFER_KIND = "train_infer_logprob_parity"
DISCLAIMER = (
    "Qwen3-Next-80B-A3B-Instruct, TP4 x EP1, BF16, strict shared providers. Cells are the "
    "16-case raw-bit L2 matrix (prompt 8/64/256/1024 x batch 1/4 x full/chunked prefill, 16 "
    "response tokens) at each weight version; rollout is native vLLM prefill and decode, "
    "replay is VIME recompute. Gradients are judged by the training stages (two updates, "
    "reload and resumption), not by C11 gradient rows, so gradient fields are empty here."
)


def family_of(layer: int | None, node: str | None) -> str | None:
    """Map a first divergence to the layer family that produced it."""
    if node is None:
        return None
    if node == "embedding":
        return "embedding"
    if node in ("logits", "logprobs", "response_tokens", "lm_head_logprob"):
        return "lm_head_logprob"
    if layer is None or layer not in range(48):
        raise ValueError(f"Divergence node {node!r} needs a decoder layer, got {layer!r}")
    if node in ("input_layernorm", "post_attention_layernorm"):
        return "norm"
    if node in ("router", "mlp"):
        return "moe"
    if node in ("mixer", "output", "state"):
        return "full_attention" if (layer + 1) % 4 == 0 else "gdn"
    raise ValueError(f"Unknown divergence node {node!r}")


def first_drift_of(record: Mapping[str, Any]) -> tuple[str | None, int | None, str | None]:
    """``(dense first_drift, layer, family)`` of one L2 case record."""
    if record["passed"]:
        return None, None, None
    divergence = record.get("first_divergent_node")
    if divergence is not None:
        layer, node = divergence["layer"], divergence["node"]
        label = "embedding" if node == "embedding" else f"layer{layer}.{node}"
    else:
        layer = None
        difference = record.get("first_difference") or {}
        node = difference.get("tensor", "lm_head_logprob")
        label = node
    return f"{label}:rollout->recompute", layer, family_of(layer, node)


def _detail(name, cell_id, rollout, recompute, roles) -> TensorComparisonDetail:
    lhs, rhs = rollout.double(), recompute.double()
    error = (lhs - rhs).abs()
    return TensorComparisonDetail(
        tensor_name=name,
        config_pair=(f"{cell_id}/rollout", f"{cell_id}/recompute"),
        shape=tuple(rollout.shape),
        dtype=str(rollout.dtype).replace("torch.", ""),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
        max_rel_error=float((error / rhs.abs().clamp_min(1e-12)).max()),
        atol=0.0,
        rtol=0.0,
        passed=bool(torch.equal(rollout, recompute)),
        judgment="forward_invariance",
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )


def build_report(
    matrices: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    contract: Mapping[str, Any],
    measured_commit: str,
    weight_source: str,
    weight_hash: str,
    device: str,
    gpu_name: str | None,
    compute_capability: str | None,
    bi_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the chain report from L2 matrices and their loaded case artifacts.

    ``matrices[version]`` is an ``l2-matrix.json`` report; ``artifacts[version]
    [case_id][role]`` is the loaded rollout/recompute artifact of that case.
    ``bi_matrix`` (a ``bi-matrix.json`` from the batch-invariance stage) adds a
    ``batch_invariance`` section and must pass for the report to pass.
    """
    if not matrices or set(matrices) - set(WEIGHT_VERSIONS):
        raise ValueError(f"Weight versions must be a subset of {WEIGHT_VERSIONS}")
    roles = resolve_comparison_roles(contract, "forward_invariance")
    train_roles = resolve_comparison_roles(contract, _TRAIN_INFER_KIND)
    clip = default_clip_interval(contract)
    cells, invariance, decode_prefill = {}, [], []
    lhs_all, rhs_all = [], []
    first = None
    for version in WEIGHT_VERSIONS:
        if version not in matrices:
            continue
        matrix = matrices[version]
        for case_id in matrix["case_ids"]:
            record = matrix["cases"][case_id]
            rollout = artifacts[version][case_id]["rollout"]
            recompute = artifacts[version][case_id]["recompute"]
            cell_id = f"{version}/{case_id}"
            drift, layer, family = first_drift_of(record)
            if drift is not None and first is None:
                first = (drift, layer, family, cell_id)
            batch = rollout["logprobs"].shape[0]
            cells[cell_id] = CellOutput(
                cell_id=cell_id,
                selected_logp={
                    (f"sample{row}", position): rollout["logprobs"][row, position]
                    for row in range(batch)
                    for position in range(16)
                },
                loss=None,
                grads={},
                first_drift_node=drift,
                node_digests={
                    "embedding": rollout["embedding_digest"]["sha256"],
                    **{
                        f"layer{item['layer']}": item["output_sha256"]
                        for item in rollout["layer_digests"]
                    },
                },
                requested_backend=PROFILE,
                actual_backend=PROFILE,
            )
            for name in ("logprobs", "logits"):
                invariance.append(_detail(name, cell_id, rollout[name], recompute[name], roles))
            lhs_all.append(rollout["logprobs"].reshape(-1))
            rhs_all.append(recompute["logprobs"].reshape(-1))
            if case_id.endswith("-chunked"):
                full = artifacts[version][case_id.replace("-chunked", "-full")]["rollout"]
                verdict = judge_logprob_aggregates(
                    compute_logprob_aggregates(
                        rollout["logprobs"].reshape(-1),
                        full["logprobs"].reshape(-1),
                        torch.ones(rollout["logprobs"].numel(), dtype=torch.bool),
                        contract=contract,
                        report_kind=_TRAIN_INFER_KIND,
                        clip_interval=clip,
                        comparison_lhs_role=train_roles.comparison_lhs_role,
                        comparison_rhs_role=train_roles.comparison_rhs_role,
                    ),
                    contract,
                    execution_dtype="bf16",
                )
                decode_prefill.append((cell_id, verdict))
    lhs, rhs = torch.cat(lhs_all), torch.cat(rhs_all)
    train_infer = judge_logprob_aggregates(
        compute_logprob_aggregates(
            lhs,
            rhs,
            torch.ones(lhs.numel(), dtype=torch.bool),
            contract=contract,
            report_kind=_TRAIN_INFER_KIND,
            clip_interval=clip,
            comparison_lhs_role=train_roles.comparison_lhs_role,
            comparison_rhs_role=train_roles.comparison_rhs_role,
        ),
        contract,
        execution_dtype="bf16",
    )
    passed = (
        first is None
        and all(matrix["status"] == "passed" for matrix in matrices.values())
        and all(item.passed for item in invariance)
        and (bi_matrix is None or bi_matrix["status"] == "passed")
    )
    report = ChainGateReport(
        backend_profile=PROFILE,
        workload_id="qwen3-next-l2-matrix-16",
        fixture_hash=matrices[next(iter(matrices))]["fixture"]["content_sha256"],
        config_fingerprint={
            "model": "Qwen3-Next-80B-A3B-Instruct",
            "tp": 4,
            "ep": matrices[next(iter(matrices))]
            .get("identity", {})
            .get("topology", {})
            .get("ep", 1),
            "pp": 1,
        },
        weight_source=weight_source,
        weight_hash=weight_hash,
        seed=1234,
        workload_seed=1234,
        device=device,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        backend_provenance=BackendProvenance(
            backend_profile=PROFILE,
            requested_backend=PROFILE,
            actual_backend=PROFILE,
            execution_dtype="bf16",
            accumulation_dtype="fp32",
            output_dtype="bf16",
            reference_dtype="bf16",
            candidate_tf32_enabled=False,
            reference_tf32_enabled=False,
        ),
        runtime_backend_observations={},
        backward_runtime_observations={},
        cells=cells,
        invariance=tuple(invariance),
        gradient_invariance=(),
        accuracy=(),
        gradient_accuracy=(),
        accuracy_aggregates=(),
        train_infer=train_infer,
        train_infer_bn=None,
        decode_prefill=tuple(decode_prefill),
        first_drift=None if first is None else first[0],
        aggregates=train_infer,
        gradient_scope="training_stage_reports",
        required_grad_names=(),
        all_parameter_gradients=False,
        representative_case_ids=tuple(cells),
        workflow_url=None,
        c8_evidence_path="",
        c8_source_commit=None,
        passed=passed,
        backward_executed="update1" in matrices,
        train_infer_executed=True,
        accuracy_executed=False,
        gradient_accuracy_executed=False,
        disclaimer=DISCLAIMER,
    )
    payload = report.to_dict()
    payload.update(
        schema_version=SCHEMA_VERSION,
        compatible_schema=COMPATIBLE_SCHEMA,
        measured_commit=measured_commit,
        weight_versions=[version for version in WEIGHT_VERSIONS if version in matrices],
        first_drift_cell=None if first is None else first[3],
        first_drift_layer=None if first is None else first[1],
        first_drift_family=None if first is None else first[2],
        batch_invariance=None if bi_matrix is None else batch_invariance_section(bi_matrix),
    )
    return payload


def batch_invariance_section(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Per-family pass counts and the first failing cell of a ``bi-matrix.json``."""
    families: dict[str, dict[str, int]] = {}
    for cell_id in matrix["cell_ids"]:
        cell = matrix["cells"][cell_id]
        counts = families.setdefault(cell["family"], {"cells": 0, "passed": 0})
        counts["cells"] += 1
        counts["passed"] += int(cell["passed"])
    first = matrix.get("first_failure")
    return {
        "suite": matrix["suite"],
        "passed": matrix["status"] == "passed",
        "families": families,
        "first_failure": first,
        "first_failure_family": (
            None if first is None else matrix["cells"][first]["first_drift_family"]
        ),
    }
