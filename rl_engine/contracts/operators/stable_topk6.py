# SPDX-License-Identifier: Apache-2.0
"""Task-31 ordering over complete global FP32 selection scores."""

CONTRACT_ID = "rlkernel.stable_topk6.fp32.global_id.v1"
AITER_CONTRACT_ID = "aiter.stable_topk6.fp32.global_id.v1"
AITER_REVISION = "326cffa38c9eda16bef19375ba55334e4d3acd59"
K = 6
MIN_EXPERTS = 6
MAX_EXPERTS = 32767
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
