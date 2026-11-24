# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries
from rl_engine.validation.ablation.inkling2.shared_gradient import (
    gradient_order_probe,
    validate_shared_owner,
)


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_shared_gradient_order_and_ownership(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    source = pytest.importorskip("rl_engine.models.inkling2.shared_weight_grad")
    fold = source.accumulate_shared_weight_gradients
    gradients = {i: torch.full((4, 2), v, device=device) for i, v in enumerate((1e8, -1e8, 1.0))}
    base = fold(gradients, shard=(1, 3))
    changed = gradient_order_probe(gradients, fold=fold, shard=(1, 3))
    result = compare_boundaries({"gradient": base}, {"gradient": changed})
    assert result["mismatches"] == {"gradient": 4}


def test_shared_owner_and_visit_coverage():
    validate_shared_owner((6, 6, 6))
    with pytest.raises(ValueError, match="physical weight"):
        validate_shared_owner((6, 18, 30))
    with pytest.raises(ValueError, match="coverage"):
        gradient_order_probe({}, fold=lambda *a, **kw: None, order=(0, 2))
