# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Frozen T06 constants and schema identities for P2 CSA/HCA attention.

Layer table, cr/coff, and cache format are model identity, not ablation knobs.
Changing Hq/Hkv/D/scale is a breaking contract change, not a runtime option.
"""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = "p2-task-contract.v1"
FOUNDATION_ABI = "foundation-attention-boundary.v1"
SCHEMA_VERSION_ATTENTION = "p2.t06.mqa_joint_attention_sink.v1"
SCHEMA_VERSION_O_PROJ = "p2.t06.o_proj_grouped.v1"
SCHEMA_VERSION_CANDIDATE_PLAN = "p2.t06.candidate_plan.v1"
SCHEMA_VERSION_STATE_GATE = "p2.t06.state_gate.v1"
TASK_ID = "T06"

HIDDEN_SIZE = 4096
N_Q_HEADS = 64
N_KV_HEADS = 1
HEAD_DIM = 512
ATTENTION_SCALE = HEAD_DIM**-0.5
RECENT_WINDOW = 128
N_O_PROJ_GROUPS = 8
HEADS_PER_GROUP = 8
O_LORA_RANK = 1024
GROUP_FLAT_DIM = HEADS_PER_GROUP * HEAD_DIM  # 4096
CONCAT_Z_DIM = N_O_PROJ_GROUPS * O_LORA_RANK  # 8192

ROPE_DIM = 64
MAIN_ROTARY_START = 448
MAIN_ROTARY_END = 512
INDEX_ROTARY_START = 64
INDEX_ROTARY_END = 128

SOFTMAX_MODE_ONE_DENOMINATOR = "one_denominator"
CANDIDATE_ORDER_COMPRESSED_THEN_RECENT = "compressed_prefix_then_recent128"
SPLIT_KV_DISABLED = "disabled"
REDUCTION_TREE_SEQUENTIAL_J = "sink_then_sequential_j_0_to_n_minus_1"
REDUCTION_TREE_SEQUENTIAL_D = "sequential_d_0_to_511"
REDUCTION_TREE_SEQUENTIAL_H = "sequential_h_0_to_63"
DKV_TOKEN_THEN_HEAD = "sequential_t_then_h"
SOFTMAX_CTA_NOTE = "ws1_reference_one_thread_per_row_sequential_j"
QK_LAUNCH = "one_thread_per_t_h_j_sequential_d"
DOWNCAST_FINAL_WRITE = "final_write"

KERNEL_ID_ORACLE = "rlkernel.p2.mqa_joint_attention_sink.oracle.v1"
KERNEL_ID_CUDA = "rlkernel.p2.mqa_joint_attention_sink.cuda_reference.v1"
KERNEL_ID_O_PROJ_ORACLE = "rlkernel.p2.o_proj_grouped.oracle.v1"
KERNEL_ID_O_PROJ_DET_GEMM = "rlkernel.p2.o_proj_grouped.det_gemm.v1"
KERNEL_ID_O_PROJ_TORCH_FP32 = "rlkernel.p2.o_proj_grouped.torch_fp32.v1"
CUDA_MAX_TOKENS = 65535

# CPU sequential-tree oracle vs CUDA scalar expf/FMA is not bitwise.
# Same CUDA launch is bitwise. These are the WS1 comparison budgets.
CUDA_VS_ORACLE_FWD_ATOL = 1e-7
CUDA_VS_ORACLE_BWD_ATOL = 1e-5


class LayerType(str, Enum):
    C0 = "C0"
    C4 = "C4"
    C128 = "C128"


class AttentionKind(str, Enum):
    C0 = "C0"
    CSA = "CSA"
    HCA = "HCA"


class ExecutionMode(str, Enum):
    TRAINING = "training"
    PREFILL = "prefill"
    EAGER_DECODE = "eager_decode"
    GRAPH_DECODE = "graph_decode"


class RoundPoint(str, Enum):
    QK_FP32_ACC = "qk_fp32_acc"
    SOFTMAX_FP32 = "softmax_fp32"
    PV_FP32_ACC = "pv_fp32_acc"
    OUTPUT_FINAL_WRITE = "output_final_write"
    DEQUANT_BEFORE_QK = "dequant_before_qk"
    BWD_FP32_REFERENCE = "bwd_fp32_reference"


def attention_kind_for_layer(layer_type: LayerType) -> AttentionKind:
    if layer_type is LayerType.C0:
        return AttentionKind.C0
    if layer_type is LayerType.C4:
        return AttentionKind.CSA
    return AttentionKind.HCA
