# SPDX-License-Identifier: Apache-2.0
"""Original block-FP8 wo_a on the actual gfx942 vLLM attention instance."""

import types
from pathlib import Path

from safetensors import safe_open

from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Group


def bind_original_group_projection(attention, checkpoint, layer, observer=None):
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("original gfx942 wo_a binding requires TP=1")
    if getattr(attention.wo_b, "bias", None) is not None:
        raise ValueError("original gfx942 wo_b binding requires a bias-free projection")
    if attention._wo_a_fp8_weight is not None:
        raise ValueError("gfx942 wo_a binding expects inverse RoPE already settled in forward_mqa")
    paths = list(Path(checkpoint).glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("initial original wo_a importer requires one four-layer checkpoint file")
    device = attention.wo_a.weight.device
    with safe_open(str(paths[0]), framework="pt", device="cpu") as source:
        prefix = f"layers.{layer}.attn.wo_a"
        weight = source.get_tensor(prefix + ".weight").to(device)
        scale = source.get_tensor(prefix + ".scale").to(device)
    attention._rlk_wo_a = CheckpointFP8Group(weight, scale, attention.n_local_groups)

    def project(self, o, positions):
        # forward_mqa performs inverse RoPE exactly once because the native
        # gfx950-only fused-quantized path remains unset on this gfx942 instance.
        grouped = o.reshape(o.shape[0], self.n_local_groups, -1)
        z = self._rlk_wo_a(grouped)
        if observer:
            observer("wo_a.original_fp8.input", grouped)
            observer("wo_a.original_fp8.output", z)
        if not getattr(self, "_rlk_wo_a_executed", False):
            print(
                "RLK_ROLLOUT_ORIGINAL_WO_A_EXECUTED",
                __file__,
                CheckpointFP8Group.__module__,
                prefix,
                "original E4M3/E8M0, AITER CK no-split-K",
                flush=True,
            )
            self._rlk_wo_a_executed = True
        zf = z.flatten(1)
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        return self.wo_b(zf)

    attention._o_proj = types.MethodType(project, attention)
    return attention
