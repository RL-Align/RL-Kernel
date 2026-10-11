# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The Qwen3-Next chain report keeps the WS1 chain-gate shape and attributes first drift."""

import dataclasses

import pytest
import torch

from rl_engine.contracts.numerical import load_contract
from rl_engine.validation.models import qwen3_next_chain_report as chain
from rl_engine.validation.models.chain_gate import ChainGateReport

CASES = ("p0008-b1-full", "p0008-b1-chunked")


def artifact(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "logprobs": -torch.rand((1, 16), generator=g),
        "logits": torch.randn((1, 16, 64), generator=g).to(torch.bfloat16),
        "embedding_digest": {"sha256": "a" * 64, "sample_sha256": ["a" * 64]},
        "layer_digests": [{"layer": layer, "output_sha256": "b" * 64} for layer in range(48)],
    }


def record(passed=True, node=None, layer=None, tensor=None):
    return {
        "passed": passed,
        "first_divergent_node": None if node is None else {"layer": layer, "node": node},
        "first_difference": None if tensor is None else {"tensor": tensor},
        "artifacts": {},
    }


def inputs(records=None, recompute_seed=1):
    matrices, artifacts = {}, {}
    for version in ("baseline", "update1"):
        matrices[version] = {
            "status": "passed",
            "case_ids": list(CASES),
            "cases": {case: record() for case in CASES},
            "fixture": {"content_sha256": "f" * 64},
        }
        artifacts[version] = {
            case: {"rollout": artifact(1), "recompute": artifact(recompute_seed)} for case in CASES
        }
    for (version, case), value in (records or {}).items():
        matrices[version]["cases"][case] = value
        matrices[version]["status"] = "failed"
    return matrices, artifacts


def build(matrices, artifacts):
    return chain.build_report(
        matrices,
        artifacts,
        contract=load_contract(),
        measured_commit="c" * 40,
        weight_source="Qwen/Qwen3-Next-80B-A3B-Instruct@rev",
        weight_hash="h" * 64,
        device="cuda",
        gpu_name="NVIDIA B200",
        compute_capability="10.0",
    )


def test_report_is_a_chain_gate_report_superset_and_passes_when_bitwise():
    report = build(*inputs())
    fields = {field.name for field in dataclasses.fields(ChainGateReport)}
    assert fields <= set(report)
    assert report["schema_version"] == "qwen3-next-chain-v1"
    assert report["compatible_schema"] == "ws1-c10-c11-v5"
    assert report["passed"] and report["first_drift"] is None
    assert report["first_drift_family"] is None
    assert len(report["cells"]) == 4 and len(report["invariance"]) == 8
    assert all(item["atol"] == 0 and item["passed"] for item in report["invariance"])
    assert report["weight_versions"] == ["baseline", "update1"]
    assert [entry["case_id"] for entry in report["decode_prefill"]] == [
        "baseline/p0008-b1-chunked",
        "update1/p0008-b1-chunked",
    ]


def test_a_single_bit_in_the_replay_fails_the_report():
    matrices, artifacts = inputs()
    replay = artifacts["update1"]["p0008-b1-full"]["recompute"]
    replay["logprobs"] = replay["logprobs"].clone()
    replay["logprobs"][0, 3] = torch.nextafter(replay["logprobs"][0, 3], torch.tensor(0.0))
    report = build(matrices, artifacts)
    assert not report["passed"]
    failing = [item for item in report["invariance"] if not item["passed"]]
    assert [item["config_pair"][0] for item in failing] == ["update1/p0008-b1-full/rollout"]


@pytest.mark.parametrize(
    "node,layer,family,label",
    [
        ("embedding", 0, "embedding", "embedding"),
        ("input_layernorm", 0, "norm", "layer0.input_layernorm"),
        ("post_attention_layernorm", 5, "norm", "layer5.post_attention_layernorm"),
        ("mixer", 2, "gdn", "layer2.mixer"),
        ("state", 7, "full_attention", "layer7.state"),
        ("mixer", 47, "full_attention", "layer47.mixer"),
        ("router", 9, "moe", "layer9.router"),
        ("mlp", 10, "moe", "layer10.mlp"),
    ],
)
def test_fabricated_drift_resolves_to_its_layer_family(node, layer, family, label):
    failing = record(passed=False, node=node, layer=layer)
    report = build(*inputs({("update1", "p0008-b1-chunked"): failing}))
    assert not report["passed"]
    assert report["first_drift"] == f"{label}:rollout->recompute"
    assert report["first_drift_family"] == family
    assert report["first_drift_layer"] == layer
    assert report["first_drift_cell"] == "update1/p0008-b1-chunked"


def test_drift_after_the_last_layer_is_the_lm_head_or_logprob():
    failing = record(passed=False, tensor="logprobs")
    report = build(*inputs({("baseline", "p0008-b1-full"): failing}))
    assert report["first_drift"] == "logprobs:rollout->recompute"
    assert report["first_drift_family"] == "lm_head_logprob"
    assert report["first_drift_layer"] is None


def test_the_earliest_weight_version_and_case_names_the_first_drift():
    report = build(
        *inputs(
            {
                ("update1", "p0008-b1-full"): record(False, "router", 3),
                ("baseline", "p0008-b1-chunked"): record(False, "mixer", 11),
            }
        )
    )
    assert report["first_drift_cell"] == "baseline/p0008-b1-chunked"
    assert report["first_drift_family"] == "full_attention"


def test_unknown_nodes_and_versions_are_rejected():
    with pytest.raises(ValueError, match="Unknown divergence node"):
        chain.family_of(3, "gate")
    with pytest.raises(ValueError, match="needs a decoder layer"):
        chain.family_of(None, "mixer")
    matrices, artifacts = inputs()
    matrices["update3"] = matrices["update1"]
    with pytest.raises(ValueError, match="subset"):
        build(matrices, artifacts)


def test_batch_invariance_section_counts_families_and_gates_the_report():
    matrix = {
        "suite": "quick",
        "status": "failed",
        "cell_ids": ["concurrency-n04-last", "chunk-p0129-c64-f63"],
        "first_failure": "chunk-p0129-c64-f63",
        "cells": {
            "concurrency-n04-last": {"family": "concurrency", "passed": True},
            "chunk-p0129-c64-f63": {
                "family": "chunk",
                "passed": False,
                "first_drift_family": "gdn",
            },
        },
    }
    matrices, artifacts = inputs()
    report = chain.build_report(
        matrices,
        artifacts,
        contract=load_contract(),
        measured_commit="c" * 40,
        weight_source="w",
        weight_hash="h",
        device="cuda",
        gpu_name=None,
        compute_capability=None,
        bi_matrix=matrix,
    )
    section = report["batch_invariance"]
    assert not report["passed"] and not section["passed"]
    assert section["families"] == {
        "concurrency": {"cells": 1, "passed": 1},
        "chunk": {"cells": 1, "passed": 0},
    }
    assert section["first_failure_family"] == "gdn"
    assert build(matrices, artifacts)["batch_invariance"] is None
