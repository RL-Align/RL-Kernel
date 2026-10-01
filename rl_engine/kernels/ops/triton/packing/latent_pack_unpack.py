# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.latent_layout import LatentKernelOp


@triton.jit
def _permute(
    X,
    Y,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    UNPACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Integer pointers preserve every input bit, including signaling NaNs.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    n: tl.constexpr = C * H * W
    channel = i // (H * W)
    row, col = i // W % H, i % W
    packed = ((row // 2 * (W // 2) + col // 2) * C + channel) * 4 + row % 2 * 2 + col % 2  # pack * channel * 4pos
    base = tl.program_id(1).to(tl.int64) * n
    src, dst = (packed, i) if UNPACK else (i, packed)
    value = tl.load(X + base + src, i < n, other=0)
    tl.store(Y + base + dst, value, i < n)


def _launch(x, b, c, h, w, unpack):
    shape = (b, c, 1, h, w) if unpack else (b, h // 2 * (w // 2), c * 4)
    y = torch.empty(shape, dtype=x.dtype, device=x.device)
    if b:
        bits = torch.int32 if x.element_size() == 4 else torch.int16
        with torch.cuda.device(x.device):
            _permute[(triton.cdiv(c * h * w, 256), b)](
                x.view(bits), y.view(bits), c, h, w, unpack, 256
            )
    return y


class TritonLatentPackOp(LatentKernelOp):
    kernel = staticmethod(_launch)

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Triton latent permutation requires a GPU")


class TritonLatentUnpackOp(TritonLatentPackOp):
    unpack = True
