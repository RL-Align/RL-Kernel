# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 mathematical and mLite diagnostic references for task 29."""

import torch
import torch.nn.functional as F


def sqrt_softplus_fp64(logits):
    x = logits.double()
    # The tail avoids underflow in softplus before sqrt; its relative
    # approximation error below -36 is smaller than FP64 machine epsilon.
    sp = torch.logaddexp(x, torch.zeros_like(x))
    return torch.where(x < -36, torch.exp(0.5 * x), torch.sqrt(sp))


def sqrt_softplus_vjp_fp64(logits, grad):
    x = logits.double()
    y = sqrt_softplus_fp64(x)
    safe_y = torch.where(x < -36, torch.ones_like(y), y)
    slope = torch.where(x < -36, 0.5 * y, torch.sigmoid(x) / (2 * safe_y))
    return grad.double() * slope


def mlite_sqrt_softplus(logits):
    """Pinned mLite 327a238 fixed-route expression, diagnostic only."""
    return torch.sqrt(F.softplus(logits.float()))
