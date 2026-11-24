# SPDX-License-Identifier: Apache-2.0
"""INK-65: packed text/image input boundary, before the fixed text loop."""

import torch


def inject_packed_features(text, features, positions, image_mask, *, platform):
    """Replace only declared image rows; preserve gradient flow to text rows.

    Frozen vision features are detached deliberately. The image-mask layout
    must be produced by the actual processor, not inferred from token count.
    """
    if platform not in ("cuda", "rocm", "cpu"):
        raise ValueError("unknown platform")
    if platform != "cpu" and (
        not text.is_cuda or (torch.version.hip is not None) != (platform == "rocm")
    ):
        raise RuntimeError("GPU platform mismatch")
    if platform == "cpu" and text.device.type != "cpu":
        raise RuntimeError("CPU is diagnostic only")
    if text.ndim != 2 or features.ndim != 2 or text.shape[1] != features.shape[1]:
        raise ValueError("packed text/feature geometry mismatch")
    if text.dtype != torch.bfloat16 or features.dtype != text.dtype:
        raise TypeError("expected BF16 text and visual features")
    n = text.shape[0]
    if positions.shape != (features.shape[0],) or positions.dtype != torch.int64:
        raise ValueError("invalid image positions")
    if image_mask.shape != (n,) or image_mask.dtype != torch.bool:
        raise ValueError("invalid packed image mask")
    if any(x.device != text.device for x in (features, positions, image_mask)):
        raise ValueError("all inputs must share device")
    expected = image_mask.nonzero().flatten()
    if not torch.equal(positions, expected):
        raise ValueError("image positions must exactly match the ordered image mask")
    return text.index_copy(0, positions, features.detach())


def compare_input_boundary(train, rollout, *, train_mask, rollout_mask):
    if not torch.equal(train_mask, rollout_mask):
        raise ValueError("different image token layouts")
    if train.shape != rollout.shape or train.dtype != rollout.dtype or train.device != rollout.device:
        raise ValueError("incomparable multimodal embeddings")
    a = train.detach().contiguous().view(torch.uint8).reshape(train.shape[0], -1)
    b = rollout.detach().contiguous().view(torch.uint8).reshape(rollout.shape[0], -1)
    mismatch = (a != b).any(-1)
    return {"image_rows": int(train_mask.sum()), "row_mismatches": int(mismatch.sum()),
            "image_row_mismatches": int((mismatch & train_mask).sum()),
            "text_row_mismatches": int((mismatch & ~train_mask).sum()),
            "full_chain_certified": False}
