# SPDX-License-Identifier: Apache-2.0
"""Versioned task-29 FP32 score-transform contract; no routing selection."""

CONTRACT_ID = "rlkernel.router_sqrt_softplus.fp32.stable_tail.v1"
AITER_CONTRACT_ID = "aiter.router_sqrt_softplus.fp32.stable_tail.v1"
AITER_REVISION = "326cffa38c9eda16bef19375ba55334e4d3acd59"
SUPPORTED_ARCHS = frozenset({"gfx942", "gfx950"})
NEGATIVE_TAIL_CUTOFF = -16.0
DEEP_TAIL_VJP_CUTOFF = -160.0
BLOCK_SIZE = 256
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
