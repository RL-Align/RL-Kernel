# SPDX-License-Identifier: Apache-2.0
"""Architecture-independent CUDA implementation; no Hopper-only instructions."""

import torch

from rl_engine.backends import extension  # noqa: F401 - loads the normal native library
from rl_engine.ops.rope._registration import register_rotation
from rl_engine.ops.rope.multi_axis_rope import MultiAxisRoPEMetadata


def _apply(x, cos, sin, transpose):
    return torch.ops.rl_kernel_native.multi_axis_rope(x, cos, sin, transpose)


def native_available():
    return torch.version.hip is None and hasattr(torch.ops.rl_kernel_native, "multi_axis_rope")


_rotation = register_rotation(
    "qwen_image_rope_cuda",
    _apply,
    kernel_id="torch.ops.rl_kernel_native.multi_axis_rope",
    family="cuda",
)


class MultiAxisRoPECudaOp(MultiAxisRoPEMetadata):
    backend_id = "rlkernel.cuda.qwenimage_multi_axis_rope.v1"
    kernel_id = "torch.ops.rl_kernel_native.multi_axis_rope"

    def __init__(self):
        if not native_available():
            raise RuntimeError(
                "CUDA multi_axis_rope extension is unavailable; rebuild rl_engine._C"
            )

    def __call__(self, x, cos, sin):
        if x.device.type != "cuda":
            raise ValueError("CUDA multi_axis_rope requires CUDA tensors")
        return _rotation(x, cos, sin, False)
