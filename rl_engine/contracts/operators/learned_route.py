# SPDX-License-Identifier: Apache-2.0
"""Task-32 contract, including the explicitly agreed deterministic capacity policy."""

CONTRACT_ID = "rlkernel.learned_route.fp32.bias_selection.capacity_mask.v1"
AITER_CONTRACT_ID = "aiter.learned_route.fp32.bias_selection.capacity_mask.v1"
AITER_REVISION = "326cffa38c9eda16bef19375ba55334e4d3acd59"
CAPACITY_POLICY = "complete_domain.global_logical_token_id_ascending.v1"
EXPERTS = 256
SLOTS = 6
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
CHECKPOINT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
CHECKPOINT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CHECKPOINT_BIAS_SHA256 = "34d941394db670f6de58ccb9bb781158d12407cc27613f8ae90756e2623fb236"
