# SPDX-License-Identifier: Apache-2.0
"""Training-forward ICV after native CP K transport.

mLite's referenced official_indexer_topk uses use_fp4=False and fuses Q scales
into head weights. Its fused outputs alone cannot recover independent ICV.
Use the explicit FP8 profile with separately preserved scales, or the actual
MXFP4 profile after item-15/22 preprocessing. Full autograd is item 26.
"""

from rl_engine.integrations.common.indexer_icv import bind_global_icv_keys, bind_icv_queries

__all__ = ["bind_global_icv_keys", "bind_icv_queries"]
