# SPDX-License-Identifier: Apache-2.0
"""INK-61, INK-63 and INK-64 vision contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.vision_contracts import (
    VisionContractError,
    VisionPatchConfig,
    inject_image_tokens,
    merge_2x2_patches,
    patchify_images,
    preprocess_images,
    project_merged_patches,
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_image_preprocess_and_patch20_layout(device):
    config = VisionPatchConfig(image_size=80, patch_size=20, channels=3, merge_size=2)
    images = torch.arange(2 * 3 * 40 * 60, device=device, dtype=torch.float32).reshape(2, 3, 40, 60)
    processed = preprocess_images(images, config)
    patches = patchify_images(processed, config)
    assert processed.shape == (2, 3, 80, 80)
    assert patches.shape == (2, 16, 1200)
    assert torch.isfinite(patches).all()


def test_2x2_merge_and_projector_preserve_batch_rows():
    config = VisionPatchConfig(image_size=80, patch_size=20, channels=3, merge_size=2)
    features = torch.arange(2 * 16 * 4, dtype=torch.float32).reshape(2, 16, 4).bfloat16()
    merged = merge_2x2_patches(features, config)
    weight = torch.ones(6, 16, dtype=torch.float32)
    projected = project_merged_patches(merged, weight)
    assert merged.shape == (2, 4, 16)
    assert projected.shape == (2, 4, 6)
    assert torch.equal(projected[0], projected[0])
    assert not torch.equal(projected[0], projected[1])


def test_image_token_injection_returns_positions_and_mask():
    text = torch.zeros(2, 7, 4, dtype=torch.bfloat16)
    image = torch.ones(2, 2, 4, dtype=torch.float32)
    positions = torch.tensor([[1, 5], [0, 6]])
    result = inject_image_tokens(text, image, positions)
    assert result.embeddings.dtype == text.dtype
    assert result.image_mask.sum().item() == 4
    assert torch.equal(
        result.image_mask[0], torch.tensor([False, True, False, False, False, True, False])
    )
    assert torch.equal(result.embeddings[1, 0], torch.ones(4, dtype=torch.bfloat16))


def test_duplicate_or_out_of_range_positions_fail_closed():
    text = torch.zeros(1, 4, 3)
    image = torch.ones(1, 2, 3)
    with pytest.raises(VisionContractError, match="unique"):
        inject_image_tokens(text, image, torch.tensor([[1, 1]]))
    with pytest.raises(VisionContractError, match="outside"):
        inject_image_tokens(text, image, torch.tensor([[1, 4]]))
