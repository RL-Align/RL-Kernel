# SPDX-License-Identifier: Apache-2.0
"""Reference implementations for DSV4 MoE semantic boundaries."""

from .gradient_combine import (
    combine_local_input_gradients,
    local_input_gradient_provenance,
    merge_residual_gradient_after_rms,
)
from .inverse_mapping import inverse_map_expert_output, inverse_mapping_provenance
from .output_combine import (
    MoEOutputCombineReferenceOp,
    PreparedMoEOutputCombine,
    prepare_output_combine,
)

__all__ = [
    "inverse_map_expert_output",
    "inverse_mapping_provenance",
    "combine_local_input_gradients",
    "local_input_gradient_provenance",
    "merge_residual_gradient_after_rms",
    "PreparedMoEOutputCombine",
    "MoEOutputCombineReferenceOp",
    "prepare_output_combine",
]
