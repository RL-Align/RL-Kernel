# SPDX-License-Identifier: Apache-2.0
"""Independent mathematical oracle, not an emulator of task-29 FP32 arithmetic.

The FP64 oracle deliberately uses a more conservative tail cutoff (-36) than
the production contract (-16). In [-36, -16), it retains logaddexp/sqrt and the
mathematical derivative so validation measures the production approximation's
error rather than duplicating it. Below -36, the score/slope approximation
errors are bounded by exp(-36)/4 and 3*exp(-36)/4, both below FP64 epsilon.
This oracle checks mathematical accuracy; contract branch/order and raw-bit
batch invariance need their own tests. mLite is a separate diagnostic reference.
"""

import torch
import torch.nn.functional as F

MATHEMATICAL_ORACLE_TAIL_CUTOFF = -36.0


def sqrt_softplus_fp64(logits):
    """Approximate the mathematical score to FP64 accuracy, not FP32 execution."""
    x = logits.double()
    # The tail avoids underflow in softplus before sqrt; its relative
    # approximation error below -36 is smaller than FP64 machine epsilon.
    sp = torch.logaddexp(x, torch.zeros_like(x))
    return torch.where(x < MATHEMATICAL_ORACLE_TAIL_CUTOFF, torch.exp(0.5 * x), torch.sqrt(sp))


def sqrt_softplus_vjp_fp64(logits, grad):
    """Mathematical VJP with the same conservative FP64-only tail cutoff."""
    x = logits.double()
    y = sqrt_softplus_fp64(x)
    tail = x < MATHEMATICAL_ORACLE_TAIL_CUTOFF
    safe_y = torch.where(tail, torch.ones_like(y), y)
    slope = torch.where(tail, 0.5 * y, torch.sigmoid(x) / (2 * safe_y))
    return grad.double() * slope


def mlite_sqrt_softplus(logits):
    """Pinned mLite 327a238 fixed-route expression, diagnostic only."""
    return torch.sqrt(F.softplus(logits.float()))
