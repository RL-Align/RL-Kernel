# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.integrations.common.inkling2_weight_sync import (
    WeightSync, validate_loop_aliases, weight_digest,
)


def aliases():
    return dict(enumerate(list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))))


def test_digest_and_aliases():
    weights = {"w": torch.tensor([1.0]).bfloat16()}
    assert weight_digest(weights) != weight_digest({"w": weights["w"].float()})
    validate_loop_aliases(aliases())
    sync = WeightSync(platform="cuda")
    with pytest.raises(RuntimeError, match="GPU"):
        sync.publish(1, weight_digest(weights), reload=lambda: None, readback=lambda: weights,
                     aliases=aliases, reset_cache=lambda: True)
    with pytest.raises(RuntimeError, match="not published"):
        sync.require_version(1)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_gpu_readback_and_failed_publication(platform):
    if not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm"):
        pytest.skip(f"{platform} required")
    weights = {"w": torch.ones(4, device="cuda", dtype=torch.bfloat16)}
    sync = WeightSync(platform=platform)
    kw = dict(reload=lambda: None, readback=lambda: weights, aliases=aliases,
              reset_cache=lambda: True)
    sync.publish(1, weight_digest(weights), **kw)
    sync.require_version(1)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        sync.publish(2, "a" * 64, **kw)
    with pytest.raises(RuntimeError, match="not published"):
        sync.require_version(1)
