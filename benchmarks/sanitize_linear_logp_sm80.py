# SPDX-License-Identifier: Apache-2.0
"""Minimal compute-sanitizer target for the production SM80 kernel."""

import math

import torch

from rl_engine.kernels.ops.cuda.loss.linear_logp_sm80 import FusedLinearLogpSM80Op

D, V = 4096, 128256


def main():
    assert torch.cuda.get_device_capability() == (8, 0)
    torch.manual_seed(1234)
    weight = torch.randn(V, D, device="cuda", dtype=torch.bfloat16)
    op = FusedLinearLogpSM80Op()
    for n in (1, 31, 32, 33, 128, 513):
        hidden = torch.randn(n, D, device="cuda", dtype=torch.bfloat16) / math.sqrt(D)
        targets = [0, V - 1, V // 16 - 1, V // 16, V // 32 - 1, V // 32]
        target = torch.tensor(
            [targets[i % len(targets)] for i in range(n)], device="cuda"
        )
        out = op(hidden, weight, target)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        print(f"N={n} backend={op.selected_backend(hidden, weight, target)} ok")


if __name__ == "__main__":
    main()
