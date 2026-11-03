# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sampling-parameters gate for the T02 / P7 consistency contract.

The package root stays small: the versioned canonical sampling record lives in
:mod:`rl_engine.validation.sampling_params.schema` and the fail-closed gate that
consumes it lives in :mod:`rl_engine.validation.sampling_params.gate`.
"""

from rl_engine.validation.sampling_params.gate import (
    SamplingParamsError,
    evaluate_sampling_gate,
    normalize_sampling_layer,
)
from rl_engine.validation.sampling_params.schema import (
    SAMPLING_RECORD_VERSION,
    CanonicalSamplingLayer,
    CanonicalSamplingRecord,
    SamplingGateStatus,
    SamplingLayerName,
    SamplingValueSource,
)

__all__ = [
    "SAMPLING_RECORD_VERSION",
    "CanonicalSamplingLayer",
    "CanonicalSamplingRecord",
    "SamplingGateStatus",
    "SamplingLayerName",
    "SamplingParamsError",
    "SamplingValueSource",
    "evaluate_sampling_gate",
    "normalize_sampling_layer",
]
