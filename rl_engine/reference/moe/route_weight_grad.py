# SPDX-License-Identifier: Apache-2.0
"""CPU-only ordered reference; mathematical accuracy needs a separate FP64 check."""

import torch

from rl_engine.contracts.operators.route_weight_grad import BLOCK, validate_inputs


@torch.no_grad()
def route_weight_grad_reference(dh, silu, up):
    validate_inputs(dh, silu, up)
    if dh.device.type != "cpu":
        raise RuntimeError("the route-weight reference is CPU-only")
    m, f = dh.shape
    acc = torch.zeros(m, dtype=torch.float32)
    for begin in range(0, f, BLOCK):
        end = min(begin + BLOCK, f)
        values = torch.zeros((m, BLOCK), dtype=torch.float32)
        values[:, : end - begin] = (dh[:, begin:end].float() * silu[:, begin:end]) * up[
            :, begin:end
        ].float().clamp(-10, 10)
        for _ in range(8):
            pairs = values.reshape(m, values.shape[1] // 2, 2)
            values = pairs[:, :, 0] + pairs[:, :, 1]
        acc = acc + values[:, 0]
    return acc
