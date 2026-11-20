# SPDX-License-Identifier: Apache-2.0
"""Bind actual AMD fused/unfused compressor GEMMs; preserve weights and cache writer."""
from types import MethodType

from rl_engine.backends.rocm.attention.compressor_projection import compressor_projection


def bind_rollout_compressor_projection(attention):
    compressor = attention.compressor
    if compressor is None:
        return
    modules = [compressor]
    if attention.indexer is not None:
        modules.append(attention.indexer.compressor)
    for owner in modules:

        def forward(self, x):
            return compressor_projection(x, self.weight)

        owner.fused_wkv_wgate.forward = MethodType(forward, owner.fused_wkv_wgate)
    original = attention._run_parallel_input_projections

    def projections(self, x):
        weight = self._fused_compressor_weight
        splits = self._fused_compressor_split_sizes
        if weight is None:
            return original(x)
        if splits is None or self.indexer is None:
            raise RuntimeError("incomplete AMD compressor fusion state")
        qr_kv = self._fused_wqa_wkv_gemm(x)
        main, indexer = compressor_projection(x, weight).split(splits, dim=-1)
        routing, _ = self.indexer.weights_proj(x)
        return qr_kv, main, indexer, routing

    attention._run_parallel_input_projections = MethodType(projections, attention)
