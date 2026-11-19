# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.contracts.operators.moe_numerical import (
    MoEContractError,
    MoENumericalContract,
    moe_rule_manifest,
)


@pytest.mark.parametrize(
    "change",
    [
        {"num_experts": True},
        {"top_k": 0},
        {"top_k": 5},
        {"hidden_size": -1},
        {"input_dtype": "float64"},
        {"output_dtype": "float8"},
        {"normalize_selected_weights": 1},
        {"expert_weight_applied": "false"},
        {"routing_scale": float("nan")},
        {"routing_scale": float("inf")},
        {"routing_scale": -1},
        {"routing_scale": 1e-50},
        {"routing_scale": True},
        {"score_transform": "sigmoid"},
        {"tie_break": "arrival_order"},
        {"reduction_tree": "rank_tree"},
        {"accumulation_dtype": "bfloat16"},
        {"rounding": "stochastic"},
        {"quantization": "fp8"},
        {"version": "future"},
        {"shared_expert_mode": "auto"},
        {"residual_mode": "auto"},
        {"subnormal_mode": "flush"},
    ],
)
def test_contract_rejects_unsupported_or_ambiguous_rules(change):
    with pytest.raises(MoEContractError):
        replace(MoENumericalContract(4, 2, 3), **change)


def test_contract_fingerprint_covers_rules_and_is_immutable():
    contract = MoENumericalContract(4, 2, 3)
    expected = hashlib.sha256(
        json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert contract.fingerprint == expected
    for change in (
        {"routing_scale": 2},
        {"expert_weight_applied": True},
        {"output_dtype": "float32"},
        {"normalize_selected_weights": False},
        {"residual_mode": "external"},
    ):
        assert replace(contract, **change).fingerprint != contract.fingerprint
    assert MoENumericalContract(**json.loads(json.dumps(contract.to_dict()))) == contract
    with pytest.raises(FrozenInstanceError):
        contract.top_k = 3
    exported = contract.to_dict()
    exported["top_k"] = 3
    assert contract.top_k == 2
    assert replace(contract, routing_scale=1).fingerprint == contract.fingerprint


def test_manifest_resolves_existing_rules_without_private_tolerances():
    manifest = moe_rule_manifest(MoENumericalContract(4, 2, 3))
    shared = load_contract()
    assert manifest["shared_contract_version"] == shared["version"]
    assert (
        manifest["rules"]["moe_combine"]["forward_accuracy"]
        == resolve_tolerance(
            shared, judgment="forward_accuracy", op_class="reduction", dtype="bfloat16"
        ).to_dict()
    )
    for op in ("attention", "gemm", "norm", "logprob", "moe_combine"):
        assert manifest["rules"][op]["forward_invariance"]["mode"] == "bitwise"
    assert manifest["rules"]["logprob"]["forward_accuracy"]["dtype_name"] == "float32"


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 4, 1)])
def test_logits_require_declared_two_dimensional_expert_domain(shape):
    with pytest.raises(MoEContractError):
        MoENumericalContract(4, 2, 3).validate_logits(torch.ones(shape, dtype=torch.bfloat16))
