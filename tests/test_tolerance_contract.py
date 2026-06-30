# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from rl_engine.kernels.gtest.tolerance import load_contract


def test_load_contract_contains_expected_operator_classes():
    contract = load_contract()
    accuracy = contract["accuracy"]["default"]
    assert set(accuracy) == {"elementwise", "reduction", "logprob"}


def test_load_contract_contains_expected_dtypes():
    contract = load_contract()
    for op_class in ("elementwise", "reduction", "logprob"):
        assert set(contract["accuracy"]["default"][op_class]) == {
            "float32",
            "bfloat16",
            "float16",
        }


def test_logprob_bfloat16_tolerance_covers_observed_reference_drift():
    contract = load_contract()
    tolerance = contract["accuracy"]["default"]["logprob"]["bfloat16"]
    assert tolerance["atol"] >= 5.0e-2
    assert tolerance["rtol"] == 0.0
