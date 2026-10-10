# SPDX-License-Identifier: Apache-2.0
"""Fuse the existing rank-ordered FP32 logp merge without changing its tree."""
from rl_engine.backends.extension import _C


def ordered_logp_merge(local_lse, local_target):
    return tuple(_C.ordered_logp_merge(local_lse, local_target))
