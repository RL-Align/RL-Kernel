# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from rl_engine.contracts.numerical import (
    BackendProvenance,
    ContractError,
    ContractResolveError,
    ContractSchemaError,
    load_contract,
    resolve_dtype_policy,
    resolve_tolerance,
    resolve_tolerance_support,
    validate_backend_provenance,
)
from rl_engine.validation.operators.elementwise_inventory import (
    inventory_items,
    unresolved_needs_fix,
)
from rl_engine.validation.operators.forward_invariance import (
    AccuracyReport,
    ConfigSpec,
    ForwardInvarianceReport,
    InvarianceReport,
    LogprobSmokeResult,
    RuntimeObservation,
    TensorComparisonDetail,
    assert_forward_batch_invariant,
    build_config_matrix,
)
from rl_engine.validation.operators.four_judgment_matrix import build_classified_matrix
from rl_engine.validation.operators.gradient_adapters import (
    make_forward_runner,
    required_forward_adapters,
)
from rl_engine.validation.operators.gradient_invariance import (
    GradientInvarianceReport,
    GradientObservation,
    GradientTensorSpec,
    MissingBackwardError,
    assert_gradient_batch_invariant,
)
from rl_engine.validation.operators.kv_consistency import (
    DecodePrefillReport,
    StatefulKVReport,
    assert_decode_prefill_consistent,
    assert_stateful_kv_consistent,
    build_decode_prefill_cases,
)
from rl_engine.validation.operators.op_checks import CandidateSpec, OperatorCase, run_operator_suite

__all__ = [
    "AccuracyReport",
    "CandidateSpec",
    "ConfigSpec",
    "DecodePrefillReport",
    "ForwardInvarianceReport",
    "GradientInvarianceReport",
    "GradientObservation",
    "GradientTensorSpec",
    "MissingBackwardError",
    "InvarianceReport",
    "LogprobSmokeResult",
    "RuntimeObservation",
    "OperatorCase",
    "StatefulKVReport",
    "TensorComparisonDetail",
    "assert_decode_prefill_consistent",
    "assert_forward_batch_invariant",
    "assert_gradient_batch_invariant",
    "assert_stateful_kv_consistent",
    "build_config_matrix",
    "build_decode_prefill_cases",
    "build_classified_matrix",
    "inventory_items",
    "make_forward_runner",
    "required_forward_adapters",
    "unresolved_needs_fix",
    "run_operator_suite",
    "BackendProvenance",
    "ContractError",
    "ContractResolveError",
    "ContractSchemaError",
    "load_contract",
    "resolve_tolerance",
    "resolve_dtype_policy",
    "resolve_tolerance_support",
    "validate_backend_provenance",
]
