# SPDX-License-Identifier: Apache-2.0
"""Task-34 explicitly agreed loss profile; opt-in only."""

CONTRACT_ID = "rlkernel.router_aux_loss.preload.postcapacity.v1"
AITER_CONTRACT_ID = "aiter.router_aux_loss.canonical_fp32.v1"
FORMULA = "alpha*256*sum_e((pre_capacity_count[e]/(6*N))*(post_capacity_unscaled_mass[e]/N))"
REDUCTION_ORDER = "global_logical_token_id_ascending.then_global_expert_id_ascending.v1"
LAUNCH = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
