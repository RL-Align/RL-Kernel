# SPDX-License-Identifier: Apache-2.0
"""Task 21 lifecycle reused verbatim, with separate Indexer cache buffers."""

import torch

from rl_engine.contracts.operators.indexer_state import IndexerStateContract
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache


class IndexerStateCache(CompressorStateCache):
    contract_type = IndexerStateContract

    @staticmethod
    def _make_backend():
        from rl_engine.backends.rocm.attention.indexer_state import AiterIndexerStateBackend

        return AiterIndexerStateBackend()

    @staticmethod
    def _specialization_source():
        return __file__

    def _allocate_extra_buffers(self):
        c = self.contract
        pages = c.max_requests * c.pages_per_request
        self._scale_cache = torch.full((pages, 64, 4), 127, dtype=torch.uint8, device=self.device)
        self._bf16_stage = torch.zeros((pages, 64, 128), dtype=torch.bfloat16, device=self.device)
        self._gather_rows = torch.zeros(c.plan_rows, dtype=torch.int64, device=self.device)
        self._commit_mask = torch.zeros(pages * 64, dtype=torch.bool, device=self.device)

    def _prepare_extra(self, handle, start, stop):
        c = self.contract
        first, end = start // 4, stop // 4
        self._gather_rows.zero_()
        self._commit_mask.zero_()
        if end > first:
            physical = torch.arange(first, end, dtype=torch.int64, device=self.device)
            physical += handle.slot * c.pages_per_request * 64
            self._gather_rows[: end - first].copy_(physical)
            self._commit_mask.index_fill_(0, physical, True)

    def _check_empty_outputs(self):
        return super()._check_empty_outputs() and bool((self._scale_cache == 127).all())

    def _clear_outputs(self, slot):
        super()._clear_outputs(slot)
        c = self.contract
        page = slot * c.pages_per_request
        self._scale_cache[page : page + c.pages_per_request].fill_(127)

    def snapshot(self, handle):
        # Keep payload/scales/ring from the SAME committed cursor under the
        # inherited lock, including callers on different threads/streams.
        with self._lock, torch.no_grad():
            result = super().snapshot(handle)
            c = self.contract
            page = handle.slot * c.pages_per_request
            result["packed"] = result["rows"]
            result["scales"] = (
                self._scale_cache[page : page + c.pages_per_request]
                .flatten(0, 1)[: result["tokens"] // 4]
                .clone()
            )
            result["layout"] = c.layout
            self._complete_device()
            return result
