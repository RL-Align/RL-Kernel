# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the paged gather workspace cache, with a fake gather kernel."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.rocm.attention import strict_runtime

pytestmark = pytest.mark.unit


def test_paged_gather_reuses_workspace_and_refreshes_contents(monkeypatch):
    cls = strict_runtime.StrictRocmAttentionRuntime
    runtime = cls(
        core=SimpleNamespace(core_id=cls.core_id, strict_schedule=cls.strict_schedule),
        communication=SimpleNamespace(),
    )

    def gather(k, v, pages, count, *, k_out, v_out):
        for source, target in ((k, k_out), (v, v_out)):
            selected = source[pages.long()].flatten(1, 2).permute(0, 2, 1, 3)
            target.copy_(selected)
        return k_out, v_out

    monkeypatch.setattr(strict_runtime, "fused_paged_kv_gather_bhsd", gather)
    k = torch.arange(24.0).reshape(3, 2, 1, 4)
    v = -k
    first = runtime._gather_paged_rows_fused_bhsd(k, v, torch.tensor([[2, 0]]), 2)
    expected = torch.cat((k[2], k[0]), dim=0).permute(1, 0, 2).unsqueeze(0)
    torch.testing.assert_close(first[0], expected)
    torch.testing.assert_close(first[1], -expected)
    second = runtime._gather_paged_rows_fused_bhsd(k, v, torch.tensor([[1, 2]]), 2)
    assert second[0] is first[0]
    assert second[1] is first[1]
    expected = torch.cat((k[1], k[2]), dim=0).permute(1, 0, 2).unsqueeze(0)
    torch.testing.assert_close(second[0], expected)
    torch.testing.assert_close(second[1], -expected)
    smaller = runtime._gather_paged_rows_fused_bhsd(k, v, torch.tensor([[0]]), 1)
    assert smaller[0].shape == (1, 1, 2, 4)
    assert smaller[0] is not first[0]
