import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm.parallel import restore_interleaved_pages


@pytest.mark.parametrize("cp", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 8, 16])
def test_cp_cache_reconstructs_logical_token_order(cp, interleave):
    planes, requests, localpages, block, heads, dim = 2, 3, 2, 16, 2, 4
    logical = torch.arange(planes * requests * localpages * block * cp * heads * dim).reshape(
        planes, requests, localpages * block * cp, heads, dim
    )
    positions = torch.arange(localpages * block)
    shards = []
    for rank in range(cp):
        indices = ((positions // interleave) * cp + rank) * interleave + positions % interleave
        shards.append(
            logical[:, :, indices].reshape(planes, requests, localpages, block, heads, dim)
        )
    count = localpages * cp - 1
    keys, values = restore_interleaved_pages(
        torch.stack(shards), interleave=interleave, block_size=block, page_count=count
    )
    expected = logical[:, :, : count * block].reshape(planes, requests * count, block, heads, dim)
    assert torch.equal(keys, expected[0])
    assert torch.equal(values, expected[1])
