# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Dynamic shape of the ROCm mHC ops.

After one warm-up call per op, a token count M that has not been seen must not
compile any kernel again: M is a runtime argument that Triton does not
specialize, and no stride depends on M. The mHC ops are run for inference, then
for training with their backward.
Counted as the number of compiled variants in each kernel's Triton cache.
"""

import pytest
import torch

from tests.helpers.deepseek_v4_mhc import N, rocm_gpu_available, weights

pytestmark = pytest.mark.skipif(not rocm_gpu_available(), reason="ROCm GPU required.")

_C = 4096
_UNSEEN_M = [1, 2, 3, 7, 16, 17, 31, 33, 59, 100, 255, 257, 1000, 1023, 1025, 3000, 4097]


def _variants(kernels):
    """Compiled variants per kernel, over every device."""
    return {k.fn.__name__: sum(len(c[0]) for c in k.device_caches.values()) for k in kernels}


def test_mhc_ops_compile_once_for_every_m():
    from rl_engine.backends.rocm.mhc import triton_mhc

    w = weights(_C)
    pre, post = triton_mhc.TritonMHCPreOp(), triton_mhc.TritonMHCPostOp()

    def run_all(m):
        gen = torch.Generator(device="cuda").manual_seed(m)
        residual = torch.randn(m, N, _C, generator=gen, device="cuda").to(torch.bfloat16)
        # Inference, then training (the forward stores the saved values) and backward.
        with torch.no_grad():
            post_mix, comb_mix, layer_input = pre(
                residual, w["pre_fn"], w["pre_scale"], w["pre_base"]
            )
            post(layer_input, residual, post_mix, comb_mix)
        leaf = residual.clone().requires_grad_(True)
        post_mix, comb_mix, layer_input = pre(leaf, w["pre_fn"], w["pre_scale"], w["pre_base"])
        post(layer_input, leaf, post_mix, comb_mix).float().sum().backward()

    run_all(5)
    warm = _variants(triton_mhc.KERNELS)
    assert all(count >= 1 for count in warm.values()), warm
    for m in _UNSEEN_M:
        run_all(m)
    torch.cuda.synchronize()
    assert _variants(triton_mhc.KERNELS) == warm


