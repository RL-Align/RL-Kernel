# SPDX-License-Identifier: Apache-2.0
"""Bind Q/indexer original FP8 GEMMs to the actor's explicit no-split profile.

Native QNorm owns the actual FP8 codes/scales. Only the GEMM is bound; no
requantization, fabricated BF16 inputs or changes to the loaded model's norms.
Original packed projection weights are frozen on both training and rollout.
"""

import types
from pathlib import Path

from safetensors import safe_open

from rl_engine.backends.rocm.gemm.checkpoint_fp8 import (
    CheckpointFP8Linear,
    original_block_fp8_gemm,
)


def bind_original_q_projection(attention, checkpoint, layer):
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("original Q projection binding requires TP=1")
    paths = list(Path(checkpoint).glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("original Q importer requires one four-layer checkpoint file")
    owners = [(attention, f"layers.{layer}.attn.wq_b")]
    if attention.indexer is not None:
        owners.append((attention.indexer, f"layers.{layer}.attn.indexer.wq_b"))
    for owner, prefix in owners:
        if getattr(owner.wq_b, "bias", None) is not None:
            raise ValueError("original Q projection must be bias free")
        device = owner.wq_b.weight.device
        with safe_open(str(paths[0]), framework="pt", device="cpu") as source:
            weight = source.get_tensor(prefix + ".weight").to(device)
            scales = source.get_tensor(prefix + ".scale").to(device)
        owner._rlk_original_q = CheckpointFP8Linear(weight, scales)

        def project(self, qr, qr_scale=None):
            if qr_scale is None:
                raise ValueError(
                    "original Q GEMM requires actual native QNorm FP8 codes/scales"
                )
            original = self._rlk_original_q
            result = original_block_fp8_gemm(
                qr, qr_scale, original.codes, original.scales
            )
            if not getattr(self, "_rlk_q_executed", False):
                print(
                    "RLK_ROLLOUT_ORIGINAL_Q_EXECUTED",
                    __file__,
                    original.profile,
                    "native QNorm FP8 input; original E4M3/E8M0; CK no-split-K",
                    flush=True,
                )
                self._rlk_q_executed = True
            return result

        owner._wq_b_proj = types.MethodType(project, owner)
    return attention
