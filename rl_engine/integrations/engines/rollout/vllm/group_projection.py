# SPDX-License-Identifier: Apache-2.0
"""Original block-FP8 wo_a on the actual gfx942 vLLM attention instance."""

import types
from pathlib import Path

# N tile of the fixed actor CK kernel (shared_down_fp8.KERNEL: 256x16x64x256 = block/M/N/K).
_CK_N_TILE = 64
_CK_K_TILE = 256


def wo_b_row_partition(hidden, columns, tp, rank):
    """Original wo_b rows/scale rows owned by `rank` at TP>1 (full K), fail closed on misalignment."""
    if tp < 1 or not 0 <= rank < tp or hidden % tp:
        raise ValueError("wo_b rows must split evenly across TP")
    rows = hidden // tp
    if rows % 128 or rows % _CK_N_TILE:
        raise ValueError("wo_b H/TP must be a multiple of 128 and the CK N tile")
    if columns % 128 or columns % _CK_K_TILE or columns % tp:
        raise ValueError("wo_b K must align with block128, the CK K tile and TP")
    start = rank * rows
    return start, start + rows, start // 128, (start + rows) // 128


def bind_original_group_projection(attention, checkpoint, layer, observer=None):
    from safetensors import safe_open
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import (
        CheckpointFP8Group,
        CheckpointFP8Linear,
    )

    tp = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    if tp not in (1, 2, 4, 8) or attention.n_groups % tp:
        raise ValueError("original wo_a requires TP1/2/4/8 and complete local groups")
    if attention.n_local_groups * tp != attention.n_groups:
        raise ValueError("native attention group ownership disagrees with TP")
    if getattr(attention.wo_b, "bias", None) is not None:
        raise ValueError("original gfx942 wo_b binding requires a bias-free projection")
    if attention._wo_a_fp8_weight is not None:
        raise ValueError("gfx942 wo_a binding expects inverse RoPE already settled in forward_mqa")
    paths = list(Path(checkpoint).glob("*.safetensors"))
    if len(paths) != 1:
        raise ValueError("original wo_a importer requires one extracted checkpoint file")
    device = attention.wo_a.weight.device
    with safe_open(str(paths[0]), framework="pt", device="cpu") as source:
        prefix = f"layers.{layer}.attn.wo_a"
        weight_source = source.get_slice(prefix + ".weight")
        scale_source = source.get_slice(prefix + ".scale")
        rows, columns = weight_source.get_shape()
        if rows % attention.n_groups or rows // attention.n_groups % 128 or columns % 128:
            raise ValueError("original wo_a groups must align with block128 scales")
        local_rows = rows // tp
        if scale_source.get_shape() != [rows // 128, columns // 128]:
            raise ValueError("original wo_a scale geometry differs")
        start = rank * local_rows
        # Read only this rank's original groups; never place a full model copy on each rank.
        weight = weight_source[start : start + local_rows].to(device)
        scale = scale_source[start // 128 : (start + local_rows) // 128].to(device)
        if tp > 1:
            # TP-invariant wo_b: this rank's H/TP original rows with the FULL K, so
            # every output element is accumulated exactly as in the TP1 actor.
            b_prefix = f"layers.{layer}.attn.wo_b"
            b_weight = source.get_slice(b_prefix + ".weight")
            b_scale = source.get_slice(b_prefix + ".scale")
            hidden, b_columns = b_weight.get_shape()
            if b_columns != rows or b_scale.get_shape() != [hidden // 128, b_columns // 128]:
                raise ValueError("original wo_b geometry differs from grouped wo_a output")
            b0, b1, s0, s1 = wo_b_row_partition(hidden, b_columns, tp, rank)
            attention._rlk_wo_b = CheckpointFP8Linear(
                b_weight[b0:b1].to(device), b_scale[s0:s1].to(device)
            )
            attention._rlk_wo_b_partition = dict(tp=tp, rank=rank, start=b0, stop=b1)
    attention._rlk_wo_a = CheckpointFP8Group(weight, scale, attention.n_local_groups)
    attention._rlk_wo_a_partition = dict(
        tp=tp, rank=rank, original_rows=rows, start=start, stop=start + local_rows
    )

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
        if tp > 1:
            # Rank r owns groups [r*G/TP, (r+1)*G/TP) (contiguous wo_a rows above), so the
            # rank-ordered gather reproduces the TP1 K order. No vLLM reduction runs here.
            from rl_engine.integrations.engines.rollout.vllm.tp_exact import gather_last_dim

            full = gather_last_dim(zf.contiguous())
            return gather_last_dim(self._rlk_wo_b(full)[0].contiguous())
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        return self.wo_b(zf)

    attention._o_proj = types.MethodType(project, attention)
    return attention
