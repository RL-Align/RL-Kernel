# SPDX-License-Identifier: Apache-2.0
"""Bind real native main compressors to the pinned production cache writer."""

import types

from rl_engine.backends.rocm.attention.native_compressor import compressed_kv


def bind_native_compressor(compressor, preprocess):
    if compressor.head_dim != 512 or compressor.rotate or compressor.compress_ratio not in (4, 128):
        raise ValueError("original main C4/C128 compressor only; indexer is a separate boundary")

    def forward(self, x):
        if x.shape[0] < self.compress_ratio:
            return None
        kv = self.linear_wkv(x)[0]
        score = self.linear_wgate(x)[0]
        result = compressed_kv(
            kv,
            score,
            self.ape,
            self.norm.weight,
            preprocess.cos_sin_cache,
            self.compress_ratio,
            self.config.layernorm_epsilon,
        )
        if not getattr(self, "_rlk_compressed_executed", False):
            print(
                "RLK_NATIVE_COMPRESSED_KV_EXECUTED",
                __file__,
                "pinned vLLM compress/norm/RoPE/OCP FP8 cache writer",
                self.compress_ratio,
                tuple(result.shape),
                flush=True,
            )
            self._rlk_compressed_executed = True
        return result

    compressor.forward = types.MethodType(forward, compressor)
    return compressor
