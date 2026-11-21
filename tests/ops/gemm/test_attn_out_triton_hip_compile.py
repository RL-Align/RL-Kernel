# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compile the two attn-out reduction kernels for AMD without launching them."""

import pytest


@pytest.mark.parametrize("kernel_name", ["_tree_gemm_leaf_kernel", "_dw_left_fold_kernel"])
def test_attn_out_gfx942_fma(kernel_name):
    triton = pytest.importorskip("triton")
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if "amd" not in backends:
        pytest.skip("AMD Triton compiler unavailable")

    from rl_engine.backends.shared.triton.gemm import attn_out_bias_gemm

    kernel = getattr(attn_out_bias_gemm, kernel_name)
    tile_sizes = {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}
    signature = {}
    constants = {}
    for name in kernel.arg_names:
        if name in tile_sizes:
            signature[name] = "constexpr"
            constants[name] = tile_sizes[name]
        else:
            signature[name] = "*fp32" if name.endswith("_ptr") else "i32"

    source = ASTSource(kernel, signature, constexprs=constants)
    compiled = triton.compile(source, target=GPUTarget("hip", "gfx942", 64))
    assert compiled.asm["hsaco"]
    assembly = compiled.asm["amdgcn"]
    # LLVM can pack independent FP32 accumulators into a single instruction.
    assert any(op in assembly for op in ("v_fma_f32", "v_pk_fma_f32", "v_fmac_f32"))
