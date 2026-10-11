# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Batch-invariance rollout requests against the native rollout contract (imports vLLM)."""

import pytest

from rl_engine.integrations.engines.rollout.vllm import qwen3_next_native_rollout as rollout
from rl_engine.validation.models import qwen3_next_bi_cases as bi

TEXTS = [list(range(1000 + 50 * i, 1000 + 50 * i + 17 + i)) for i in range(4)]
IDENTITY = {
    "architecture_layers": 48,
    "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
    "checkpoint": "/checkpoint",
    "provider_config": {"native_rollout_kv_cache_memory_bytes": 4 << 30},
}


def test_every_rollout_cell_builds_a_valid_rollout_request():
    for cell in bi.cells():
        if cell.engine != "rollout":
            continue
        request = bi.rollout_request(cell, TEXTS, IDENTITY)
        assert rollout.validate_request(request) == (1, cell.target.prompt_length)
        entries = rollout.batch_invariance_descriptors(request)
        assert len(entries) == 1 + len(cell.companions)
        target = [entry for entry in entries if entry[3] == "shared_l2"]
        assert len(target) == 1 and target[0][2]["prefill_chunks"] == cell.chunk_plan()
        assert (entries[-1] if cell.target_last else entries[0])[3] == "shared_l2"


def test_requests_reject_companions_that_do_not_fit_the_profile():
    request = bi.rollout_request(
        bi.Cell("y", "interference", bi.Sequence(0, 256), 64, 64, (bi.Sequence(1, 8),)),
        TEXTS,
        IDENTITY,
    )
    request["companion_max_tokens"] = [2000]
    request["request_sha256"] = bi.json_digest(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError, match="does not fit"):
        rollout.validate_request(request)
