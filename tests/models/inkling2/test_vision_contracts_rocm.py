# SPDX-License-Identifier: Apache-2.0
"""ROCm vision contract parity (INK-61/63/64): HIP device vs CPU on real Inkling shapes."""

import pytest
import torch

from rl_engine.models.inkling2.vision_contracts import (
    VisionContractError,
    VisionPatchConfig,
    inject_image_tokens,
    merge_2x2_patches,
    patchify_images,
    preprocess_images,
)
from rl_engine.models.inkling2.vision_contracts_rocm import (
    GEMM_RTOL,
    PREPROCESS_ATOL,
    PROJECTION_IN,
    PROJECTION_OUT,
    bitwise_equal,
    project_merged_patches_fixed_order,
    run_vision_parity,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a ROCm build of torch and a HIP device",
)

CFG = VisionPatchConfig(image_size=560, patch_size=20)


def test_real_inkling_vision_parity_hip_vs_cpu():
    report = run_vision_parity()
    assert report.preprocess_max_abs <= PREPROCESS_ATOL
    assert report.preprocess_run_to_run_bitwise
    assert report.patchify_bitwise
    assert report.merge_bitwise
    assert report.inject_bitwise and report.mask_equal
    assert report.projector_max_rel <= GEMM_RTOL
    assert report.projector_run_to_run_bitwise
    assert report.projector_fixed_order_bitwise


def test_preprocess_other_sizes_and_float_input():
    gen = torch.Generator().manual_seed(1)
    for hw in ((560, 560), (37, 61), (1, 1)):
        images = torch.rand(1, 3, *hw, generator=gen)
        cpu = preprocess_images(images, CFG)
        dev = preprocess_images(images.cuda(), CFG)
        assert dev.shape == (1, 3, 560, 560)
        assert float((cpu - dev.cpu()).abs().max()) <= PREPROCESS_ATOL


def test_patchify_merge_awkward_batches_are_bitwise():
    gen = torch.Generator().manual_seed(2)
    for batch in (1, 3):
        images = torch.randn(batch, 3, 560, 560, generator=gen)
        assert bitwise_equal(patchify_images(images, CFG), patchify_images(images.cuda(), CFG))
        feats = torch.randn(batch, CFG.patch_count, 1024, generator=gen)
        assert bitwise_equal(merge_2x2_patches(feats, CFG), merge_2x2_patches(feats.cuda(), CFG))


def test_injection_is_bitwise_in_all_dtypes_and_validates_on_device():
    gen = torch.Generator().manual_seed(3)
    positions = torch.tensor([[5, 1, 9], [0, 2, 4]])
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        text = torch.randn(2, 12, 64, generator=gen).to(dtype)
        image = torch.randn(2, 3, 64, generator=gen).to(dtype)
        cpu = inject_image_tokens(text, image, positions)
        dev = inject_image_tokens(text.cuda(), image.cuda(), positions.cuda())
        assert bitwise_equal(cpu.embeddings, dev.embeddings)
        assert torch.equal(cpu.image_mask, dev.image_mask.cpu())
    dup = torch.tensor([[1, 1, 2], [0, 2, 4]]).cuda()
    oob = torch.tensor([[1, 2, 12], [0, 2, 4]]).cuda()
    with pytest.raises(VisionContractError, match="unique"):
        inject_image_tokens(text.cuda(), image.cuda(), dup)
    with pytest.raises(VisionContractError, match="outside"):
        inject_image_tokens(text.cuda(), image.cuda(), oob)


def test_fixed_order_projector_is_bitwise_across_devices_and_close_to_gemm():
    gen = torch.Generator().manual_seed(4)
    merged = torch.randn(1, 5, PROJECTION_IN, generator=gen)
    weight = torch.randn(PROJECTION_OUT, PROJECTION_IN, generator=gen) * 0.02
    cpu = project_merged_patches_fixed_order(merged, weight)
    dev = project_merged_patches_fixed_order(merged.cuda(), weight.cuda())
    assert bitwise_equal(cpu, dev)
    gemm = torch.nn.functional.linear(merged.cuda(), weight.cuda())
    assert torch.allclose(dev, gemm, rtol=GEMM_RTOL, atol=1e-3)
    with pytest.raises(VisionContractError):
        project_merged_patches_fixed_order(merged.cuda(), weight)
