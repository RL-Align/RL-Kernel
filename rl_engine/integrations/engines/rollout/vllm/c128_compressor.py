# SPDX-License-Identifier: Apache-2.0
"""Declare one native C128 cache-writer profile for prefill and decode."""

import types

import torch

from rl_engine.backends.rocm.attention.native_compressor import C128_PROFILE


def bind_single_pass_c128(attention):
    compressor = attention.compressor
    if compressor is None or compressor.compress_ratio != 128:
        return False
    if (
        compressor.head_dim != 512
        or compressor.rope_head_dim != 64
        or compressor.overlap
        or compressor.use_fp4_cache
        or getattr(compressor, "_rlk_c128_profile", None)
        or getattr(compressor, "_use_two_stage_fused_compressor", None) is not True
    ):
        raise ValueError(
            "expected unbound native head512/RoPE64 C128 FP8-DS-MLA compressor"
        )
    device = compressor.norm.weight.device
    if (
        not torch.version.hip
        or device.type != "cuda"
        or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        != "gfx942"
    ):
        raise ValueError(
            "the validated C128 single-pass profile requires MI300X/gfx942"
        )
    original = compressor._compress_norm_rope_store.__func__
    required = {
        "_use_two_stage_fused_compressor",
        "compress_norm_rope_store_triton",
        "compress_norm_rope_store_two_stage_triton",
    }
    if not required.issubset(original.__code__.co_names):
        raise ValueError("unsupported native C128 compressor dispatch ABI")

    def run(self, *args, **kwargs):
        if self._use_two_stage_fused_compressor or self.compress_ratio != 128:
            raise RuntimeError(
                "declared C128 single-pass dispatch changed after binding"
            )
        result = original(self, *args, **kwargs)
        if not getattr(self, "_rlk_c128_executed", False):
            print("RLK_C128_SINGLE_PASS_EXECUTED", __file__, C128_PROFILE, flush=True)
            self._rlk_c128_executed = True
        return result

    # Keep the original forward, projections, metadata, state/cache and writer.
    # This instance flag selects its existing native decode kernel for prefill too.
    compressor._use_two_stage_fused_compressor = False
    compressor._compress_norm_rope_store = types.MethodType(run, compressor)
    compressor._rlk_c128_profile = C128_PROFILE
    print("RLK_C128_SINGLE_PASS_BOUND", __file__, C128_PROFILE, flush=True)
    return True
