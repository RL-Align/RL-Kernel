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


def bind_original_input_projection(attention, checkpoint_file, layer):
    """Replace the replicated fused wq_a|wkv GEMM with the actor's two original projections.

    vLLM's gfx942 path runs this GEMM through AITER bpreshuffle, whose kernel is picked
    from tuned (gfx, cu_num, M, N, K) rows; on cu_num=80 those rows switch between
    triton/CK/asm/opus kernels by M and break parity with the actor's fixed CK no-split-K.
    """
    import torch
    from safetensors import safe_open

    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

    device = attention.fused_wqa_wkv.weight.device
    projections = []
    with safe_open(str(checkpoint_file), framework="pt", device="cpu") as source:
        for key in ("wq_a", "wkv"):
            prefix = f"layers.{layer}.attn.{key}"
            projections.append(
                CheckpointFP8Linear(
                    source.get_tensor(prefix + ".weight").to(device),
                    source.get_tensor(prefix + ".scale").to(device),
                )
            )
    widths = [p.codes.shape[0] for p in projections]
    if widths != [attention.q_lora_rank, attention.head_dim]:
        raise ValueError(f"original wq_a/wkv rows {widths} differ from the fused projection")
    attention._rlk_wq_a, attention._rlk_wkv = projections

    def project(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not x.shape[0]:
            return x.new_empty((*hidden_states.shape[:-1], sum(widths)))
        out = torch.cat([self._rlk_wq_a(x)[0], self._rlk_wkv(x)[0]], dim=-1)
        if not getattr(self, "_rlk_input_projection_executed", False):
            print(
                "RLK_ROLLOUT_ORIGINAL_WQA_WKV_EXECUTED",
                __file__,
                f"layers.{layer}.attn.wq_a|wkv",
                "original E4M3/E8M0, AITER CK no-split-K",
                flush=True,
            )
            self._rlk_input_projection_executed = True
        return out.reshape(*hidden_states.shape[:-1], out.shape[-1])

    attention._fused_wqa_wkv_gemm = types.MethodType(project, attention)
    return attention


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
    bind_original_input_projection(attention, paths[0], layer)
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
        # This rank's H/TP original wo_b rows with the FULL K (all rows at TP1), so every
        # output element is accumulated exactly as in the TP1 actor. TP1 must not fall
        # back to vLLM's cu_num-tuned bpreshuffle GEMM either.
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
        return self._rlk_wo_b(zf)[0]

    attention._o_proj = types.MethodType(project, attention)
    return attention
