# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini vision contracts (INK-61/63/64).

ROCm-only helpers that run the CUDA-side contract functions on a HIP device and
compare against CPU.  Precision policy, per op:

* patchify, 2x2 merge, image-token injection: pure data movement -> bitwise.
* preprocess: bilinear resize + normalize is float math whose last bit may differ
  between CPU and HIP kernels -> documented tolerance (``PREPROCESS_ATOL``), plus
  bitwise run-to-run determinism on the device.
* projector: ``F.linear`` is a hipBLAS GEMM whose accumulation order is not the CPU one
  -> documented tolerance (``GEMM_RTOL``/``GEMM_ATOL``) for the library path and bitwise
  run-to-run determinism.  ``project_merged_patches_fixed_order`` is a deterministic
  fixed-k-order path (separate FP32 multiply and add per k, no FMA fusion, no atomics)
  that is bitwise identical on CPU and the HIP device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_engine.models.inkling2.vision_contracts import (
    TokenInjectionResult,
    VisionContractError,
    VisionPatchConfig,
    inject_image_tokens,
    merge_2x2_patches,
    patchify_images,
    preprocess_images,
    project_merged_patches,
)

PREPROCESS_ATOL = 1e-5
GEMM_RTOL = 1e-4
GEMM_ATOL = 1e-3

# Real Inkling shapes.
VIT_HIDDEN = 1024
PROJECTION_IN = 4 * VIT_HIDDEN
PROJECTION_OUT = 2304


def require_hip_device() -> torch.device:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise VisionContractError("a ROCm build of torch with a HIP device is required")
    return torch.device("cuda")


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.cpu().view(torch.uint8), b.cpu().view(torch.uint8))
    )


def project_merged_patches_fixed_order(
    merged: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """FP32 projector with a fixed k = 0..K-1 accumulation order (bitwise across devices)."""
    if merged.ndim != 3 or weight.ndim != 2 or merged.shape[-1] != weight.shape[1]:
        raise VisionContractError("projector shapes are incompatible")
    if weight.device != merged.device:
        raise VisionContractError("projector weight must share the merged device")
    x = merged.float()
    w_t = weight.float().t().contiguous()  # [K, N]
    acc = torch.zeros(*x.shape[:-1], weight.shape[0], device=x.device, dtype=torch.float32)
    for k in range(x.shape[-1]):
        # two separate kernels: product rounded to FP32, then add (no FMA contraction)
        acc = acc + x[..., k : k + 1] * w_t[k]
    if bias is not None:
        acc = acc + bias.float()
    return acc.to(merged.dtype)


@dataclass(frozen=True)
class VisionParity:
    preprocess_max_abs: float
    preprocess_run_to_run_bitwise: bool
    patchify_bitwise: bool
    merge_bitwise: bool
    inject_bitwise: bool
    mask_equal: bool
    projector_max_rel: float
    projector_run_to_run_bitwise: bool
    projector_fixed_order_bitwise: bool


def run_vision_parity(*, seed: int = 0, image_hw: tuple[int, int] = (480, 640)) -> VisionParity:
    """Run every vision contract on the HIP device and on CPU with real Inkling shapes."""
    dev = require_hip_device()
    cfg = VisionPatchConfig(image_size=560, patch_size=20)
    gen = torch.Generator().manual_seed(seed)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    images = torch.randint(0, 256, (2, 3, *image_hw), generator=gen, dtype=torch.uint8)
    pre_cpu = preprocess_images(images, cfg, mean=mean, std=std)
    pre_dev = preprocess_images(images.to(dev), cfg, mean=mean, std=std)
    pre_dev2 = preprocess_images(images.to(dev), cfg, mean=mean, std=std)
    assert pre_dev.device.type == "cuda"
    pre_diff = float((pre_cpu - pre_dev.cpu()).abs().max())

    # Patchify/merge/inject are exact on the same input, so feed the CPU result to both.
    patches_cpu = patchify_images(pre_cpu, cfg)
    patches_dev = patchify_images(pre_cpu.to(dev), cfg)
    assert patches_cpu.shape == (2, cfg.patch_count, 3 * 20 * 20)

    features = torch.randn(2, cfg.patch_count, VIT_HIDDEN, generator=gen)
    merged_cpu = merge_2x2_patches(features, cfg)
    merged_dev = merge_2x2_patches(features.to(dev), cfg)
    assert merged_cpu.shape == (2, cfg.merged_count, PROJECTION_IN)

    weight = torch.randn(PROJECTION_OUT, PROJECTION_IN, generator=gen) * 0.02
    bias = torch.randn(PROJECTION_OUT, generator=gen)
    proj_cpu = project_merged_patches(merged_cpu, weight, bias)
    proj_dev = project_merged_patches(merged_dev, weight.to(dev), bias.to(dev))
    proj_dev2 = project_merged_patches(merged_dev, weight.to(dev), bias.to(dev))
    scale = proj_cpu.abs().max().clamp_min(1e-6)
    proj_rel = float((proj_cpu - proj_dev.cpu()).abs().max() / scale)

    one = merged_cpu[:1, :8]
    fixed_cpu = project_merged_patches_fixed_order(one, weight, bias)
    fixed_dev = project_merged_patches_fixed_order(one.to(dev), weight.to(dev), bias.to(dev))

    text = torch.randn(2, 900, PROJECTION_OUT, generator=gen).bfloat16()
    tokens = proj_cpu.bfloat16()
    positions = torch.stack(
        [torch.randperm(900, generator=gen)[: cfg.merged_count] for _ in range(2)]
    )
    inj_cpu: TokenInjectionResult = inject_image_tokens(text, tokens, positions)
    inj_dev = inject_image_tokens(text.to(dev), tokens.to(dev), positions.to(dev))

    return VisionParity(
        preprocess_max_abs=pre_diff,
        preprocess_run_to_run_bitwise=bitwise_equal(pre_dev, pre_dev2),
        patchify_bitwise=bitwise_equal(patches_cpu, patches_dev),
        merge_bitwise=bitwise_equal(merged_cpu, merged_dev),
        inject_bitwise=bitwise_equal(inj_cpu.embeddings, inj_dev.embeddings),
        mask_equal=bool(torch.equal(inj_cpu.image_mask, inj_dev.image_mask.cpu())),
        projector_max_rel=proj_rel,
        projector_run_to_run_bitwise=bitwise_equal(proj_dev, proj_dev2),
        projector_fixed_order_bitwise=bitwise_equal(fixed_cpu, fixed_dev),
    )
