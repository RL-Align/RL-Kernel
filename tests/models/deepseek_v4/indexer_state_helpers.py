# SPDX-License-Identifier: Apache-2.0
"""Fixtures and explicit non-numerical CPU layout backend; native tests check math."""

from rl_engine.contracts.operators.indexer_state import IndexerStateContract
from rl_engine.models.deepseek_v4.indexer_state import IndexerStateCache
from tests.models.deepseek_v4.indexer_c4_helpers import fixture


def make(backend, *, graph=False, step=271, tokens=271, seed=22, device="cpu", **kwargs):
    c = IndexerStateContract(
        graph=graph, max_tokens_per_step=step, max_sequence_tokens=tokens, **kwargs
    )
    x, ids, a, n, co, si = fixture((tokens + 3) // 4, seed=seed, device=device)
    x, ids, co, si = x[:tokens], ids[:tokens], co[:tokens], si[:tokens]
    cache = IndexerStateCache(
        c,
        ape=a,
        norm=n,
        cos=co,
        sin=si,
        version="v1",
        writer_rank=0,
        expected_arithmetic=c.arithmetic_id,
        backend=backend,
    )
    return cache, x, ids, (a, n, co, si)


class LayoutOnlyBackend:
    """No numerical production fallback. Used only in CPU lifecycle tests."""

    device_type = "cpu"
    supports_graph = False

    def execute(self, arena):
        rows = arena._cache.flatten(0, 1)
        scales = arena._scale_cache.flatten(0, 1)
        rows.masked_fill_(arena._commit_mask[:, None].expand_as(rows), 17)
        scales.masked_fill_(arena._commit_mask[:, None].expand_as(scales), 129)
        c = arena.contract
        for i, physical in enumerate(arena._write_slots.tolist()):
            if physical < 0:
                continue
            slot, offset = divmod(physical, c.state_rows)
            source = arena._input[i].float()
            arena._state[slot, offset, : c.width] = source[: c.width]
            arena._state[slot, offset, c.width :] = (
                source[c.width :] + arena._ape[int(arena._positions[i]) % 4]
            )
