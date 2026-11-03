# SPDX-License-Identifier: Apache-2.0
"""Compose task 20 postprocessing with task 21 state and native masked scatter."""

from rl_engine.backends.rocm.attention.compressor_state import AiterCompressorStateBackend
from rl_engine.backends.rocm.attention.indexer_compressor_c4 import AiterIndexerC4Backend
from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.kv_compressor_c4 import source_hash


class AiterIndexerStateBackend(AiterCompressorStateBackend):
    def __init__(self):
        super().__init__()
        self.postprocess = AiterIndexerC4Backend()
        self.quant_contract = IndexerC4Contract(precision="mxfp4")
        self.provenance["postprocess"] = self.postprocess.provenance
        self.provenance["indexer_state_bridge"] = source_hash(__file__)
        self.provenance["cache_write"] = "torch.masked_scatter_: unique sorted complete rows only"

    def _execute(self, arena):
        # Only a TEMPORARY buffer is cleared. Persistent invalid cache rows
        # retain deterministic payload=0, scale=127 until their own commit.
        arena._bf16_stage.zero_()
        self._compress_to(arena, arena._bf16_stage)
        compact = arena._bf16_stage.flatten(0, 1).index_select(0, arena._gather_rows)
        rotated = self.postprocess._rotate(compact)
        payload, scale = self.postprocess.quantize(rotated, self.quant_contract)
        self._scatter(arena, payload, scale)
        self._save_history(arena)

    @staticmethod
    def _scatter(arena, payload, scale):
        # The host lease admits one monotonically increasing interval of
        # complete rows. Its dense prefix is in exactly the same order as the
        # true mask entries. Invalid/padded source rows NEVER write cache.
        rows = arena._cache.flatten(0, 1)
        scales = arena._scale_cache.flatten(0, 1)
        rows.masked_scatter_(arena._commit_mask[:, None].expand_as(rows), payload.flatten())
        scales.masked_scatter_(arena._commit_mask[:, None].expand_as(scales), scale.flatten())
