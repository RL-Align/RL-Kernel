# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU launch-contract checks; no CUDA arithmetic is emulated here."""

from types import SimpleNamespace

import torch

from rl_engine.backends.cuda.gemm import h3_projection_fp32 as adapter


def test_historical_import_shares_module_and_binary_cache():
    from rl_engine.backends.shared.triton.gemm import h3_projection_fp32 as legacy

    assert legacy is adapter
    assert legacy._BINARY_HASHES is adapter._BINARY_HASHES


def test_direct_launch_uses_bias_slot_and_fixed_h3_specialization(monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            # Positional binding mirrors the shared kernel's required interface.
            def launch(a, b, c, bias, m, n, k, *strides, **options):
                calls.append((grid, a, b, c, bias, m, n, k, strides, options))
                return SimpleNamespace(
                    hash="cpu-contract-only",
                    asm={"cubin": b"launch contract fixture"},
                    metadata=SimpleNamespace(num_warps=4),
                )
            return launch

    backend = adapter.FixedFP32Backend.__new__(adapter.FixedFP32Backend)
    backend.kernel = Kernel()
    backend.triton = SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                     __version__="contract-only")
    monkeypatch.setattr(adapter, "_BINARY_HASHES", {})
    a = torch.empty_strided((33, 65), (130, 2), dtype=torch.bfloat16)
    b = torch.empty_strided((65, 67), (134, 2), dtype=torch.bfloat16)
    output = backend.gemm(a, b)
    grid, actual_a, actual_b, actual_c, bias, m, n, k, strides, options = calls.pop()
    assert not calls
    assert actual_a is a and actual_b is b and actual_c is output and bias is None
    assert (m, n, k) == (33, 67, 65) and grid == (1, 2)
    assert strides == (*a.stride(), *b.stride(), *output.stride())
    assert options == dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, PROMOTE_INPUTS=False,
        HAS_BIAS=False, INDEX_64=True, COMPENSATED=False, CHUNK_K=0,
        EVEN_K=False, EVEN_N=False, num_warps=4, num_stages=2,
        enable_fp_fusion=False,
    )
    assert output.shape == (33, 67) and output.dtype == torch.bfloat16
