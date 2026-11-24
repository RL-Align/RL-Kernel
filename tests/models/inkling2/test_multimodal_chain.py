# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.multimodal_chain import (
    compare_input_boundary, inject_packed_features,
)


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_packed_injection_and_backward(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    text = torch.ones(5, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    features = torch.full((2, 4), 3., device=device, dtype=torch.bfloat16, requires_grad=True)
    positions = torch.tensor([1, 3], device=device)
    mask = torch.tensor([False, True, False, True, False], device=device)
    out = inject_packed_features(text, features, positions, mask, platform=platform)
    assert torch.equal(out[positions], features)
    out.sum().backward()
    assert features.grad is None and torch.count_nonzero(text.grad[mask]) == 0
    assert torch.equal(text.grad[~mask], torch.ones_like(text.grad[~mask]))
    result = compare_input_boundary(out, out.clone(), train_mask=mask, rollout_mask=mask)
    assert result["row_mismatches"] == 0 and not result["full_chain_certified"]
    with pytest.raises(ValueError, match="ordered"):
        inject_packed_features(text, features, positions.flip(0), mask, platform=platform)
