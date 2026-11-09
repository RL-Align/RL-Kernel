# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent FP32 projection reference from quantized H3 inputs."""

from contextlib import contextmanager

import torch


@contextmanager
def fp32_math():
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        with (
            torch.autocast("cuda", enabled=False),
            torch.autocast("cpu", enabled=False),
        ):
            yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def reference_projection(x, weight, grad_output):
    """Autograd FP32 oracle from the SAME quantized inputs, retaining FP32 results."""
    with fp32_math(), torch.enable_grad():
        ref_x = x.detach().float().requires_grad_(True)
        ref_weight = weight.detach().float().requires_grad_(True)
        output = torch.nn.functional.linear(ref_x, ref_weight)
        dx, dw = torch.autograd.grad(output, (ref_x, ref_weight), grad_output.detach().float())
    return tuple(value.detach() for value in (output, dx, dw))
