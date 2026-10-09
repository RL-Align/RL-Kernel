# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Batch-invariance cells, their pinned manifest and the worker requests built from them."""

import collections

import pytest

from rl_engine.integrations.qwen3_next_native_recompute import packed_layout
from rl_engine.testing import qwen3_next_bi_cases as bi

TEXTS = [list(range(1000 + 50 * i, 1000 + 50 * i + 17 + i)) for i in range(4)]
IDENTITY = {
    "architecture_layers": 48,
    "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
    "checkpoint": "/checkpoint",
    "provider_config": {"native_rollout_kv_cache_memory_bytes": 4 << 30},
}


def test_manifest_pins_the_generated_cells():
    payload = bi.load_manifest()
    assert len(payload["cells"]) == len(bi.cells()) == 97
    families = collections.Counter(cell.family for cell in bi.cells())
    assert families == {"chunk": 66, "concurrency": 13, "interference": 10, "packing": 8}


def test_concurrency_covers_one_to_sixty_four_requests():
    sizes = {1 + len(cell.companions) for cell in bi.cells() if cell.family == "concurrency"}
    assert sizes == set(bi.CONCURRENCY)


@pytest.mark.parametrize(
    "prompt,first,chunk,plan",
    [
        (64, 1, 64, [1, 63]),
        (65, 63, 64, [63, 2]),
        (129, 65, 64, [65, 64]),
        (1024, 1, 128, [1] + [128] * 7 + [127]),
        (63, 65, 128, [63]),
    ],
)
def test_chunk_plan_is_first_chunk_then_fixed_chunks(prompt, first, chunk, plan):
    cell = bi.Cell("c", "chunk", bi.Sequence(0, prompt), first, chunk)
    assert cell.chunk_plan() == plan and sum(plan) == prompt


def test_quick_subset_has_one_cell_per_family_and_the_largest_batch():
    quick = bi.quick_cells()
    assert {cell.family for cell in quick} == {"concurrency", "interference", "chunk", "packing"}
    assert max(1 + len(cell.companions) for cell in quick) == 64


def test_reference_is_the_target_alone_with_the_same_plan():
    cell = next(c for c in bi.cells() if c.cell_id == "concurrency-n16-last")
    reference = cell.reference()
    assert reference.companions == () and reference.chunk_plan() == cell.chunk_plan()
    assert reference.target == cell.target


def test_cells_reject_companions_that_do_not_fit_the_profile():
    cell = bi.Cell(
        "x", "interference", bi.Sequence(0, 256), 64, 64, (bi.Sequence(1, 1030, max_tokens=16),)
    )
    with pytest.raises(ValueError, match="does not fit"):
        bi.validate(cell)


def test_prompts_repeat_and_rotate_the_pinned_text():
    tokens = bi.prompt_tokens(TEXTS, bi.Sequence(1, 40, offset=3))
    row = TEXTS[1]
    assert len(tokens) == 40 and tokens[:5] == row[3:8]
    assert tokens[len(row) - 3] == row[0]


@pytest.mark.parametrize("last", [False, True])
def test_packing_traces_only_the_target(last):
    cell = next(
        c
        for c in bi.cells()
        if c.cell_id == f"packing-four-sequences-{'last' if last else 'first'}"
    )
    target = list(range(271))
    layout = bi.packing_layout(cell, TEXTS, target)
    request = bi.replay_request(
        cell.cell_id, target[:256], target[256:] + [7], IDENTITY, packing=layout
    )
    sequences, traced = packed_layout(request)
    assert len(sequences) == 4 and traced == [3 if last else 0]
    assert sequences[traced[0]] == target
    layout["sequences"][traced[0]] = target[:-1]
    with pytest.raises(ValueError, match="exactly once"):
        packed_layout({**request, "packing": layout})
