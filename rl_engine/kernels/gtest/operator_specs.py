# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.kernels.gtest.op_checks import CandidateSpec, OperatorCase
from rl_engine.kernels.gtest.operator_inputs import make_operator_inputs, operator_shape_name


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    op_class: str
    gold_path: str
    gold_method: str
    candidate_paths: dict[str, str]
    grad_input_names: tuple[str, ...] = ()


def _load_object(path: str) -> Any:
    module_path, object_name = path.rsplit(".", 1)
    # dynamic loading ops
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


OP_SPECS = {
    "rms_norm": OperatorSpec(
        name="rms_norm",
        op_class="reduction",
        gold_path="rl_engine.kernels.ops.pytorch.norm.rms_norm.NativeRMSNormOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.norm.rms_norm.NativeRMSNormOp",
            "triton": "rl_engine.kernels.ops.triton.rmsnorm_triton.RMSNormTritonOp",
            "cuda": "rl_engine.kernels.ops.cuda.norm.rmsnorm.RMSNormCudaOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.norm.rmsnorm.RMSNormCudaOp",
        },
        grad_input_names=("x", "weight"),
    ),
    "qk_norm": OperatorSpec(
        name="qk_norm",
        op_class="reduction",
        gold_path="rl_engine.kernels.ops.pytorch.norm.rms_norm.NativeRMSNormOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.norm.rms_norm.NativeRMSNormOp",
            "triton": "rl_engine.kernels.ops.triton.rmsnorm_triton.RMSNormTritonOp",
            "cuda": "rl_engine.kernels.ops.cuda.norm.rmsnorm.RMSNormCudaOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.norm.rmsnorm.RMSNormCudaOp",
        },
        grad_input_names=("x", "weight"),
    ),
    "attention": OperatorSpec(
        name="attention",
        op_class="attention",
        gold_path="rl_engine.kernels.ops.pytorch.attention.standard_attn.NativeAttentionOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.attention.standard_attn.NativeAttentionOp",
            "triton": (
                "rl_engine.kernels.ops.triton.attention.standard_attn."
                "TritonBatchInvariantAttentionOp"
            ),
            "cuda": (
                "rl_engine.kernels.ops.cuda.attention.deterministic_attn."
                "DeterministicAttentionOp"
            ),
        },
        grad_input_names=("q", "k", "v"),
    ),
    "cp_attention": OperatorSpec(
        name="cp_attention",
        op_class="attention",
        gold_path=(
            "rl_engine.kernels.ops.pytorch.attention.cp_attention."
            "DeterministicCPAttentionReferenceOp"
        ),
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": (
                "rl_engine.kernels.ops.pytorch.attention.cp_attention."
                "DeterministicCPAttentionReferenceOp"
            ),
        },
        grad_input_names=("q", "k", "v"),
    ),
    "logp": OperatorSpec(
        name="logp",
        op_class="logprob",
        gold_path="rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp",
            "triton": "rl_engine.kernels.ops.triton.loss.logp.TritonLogpOp",
            "cuda": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp",
            "cuda-generic": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op",
        },
        grad_input_names=("logits",),
    ),
    "linear_logp": OperatorSpec(
        name="linear_logp",
        op_class="logprob",
        gold_path="rl_engine.kernels.ops.pytorch.loss.linear_logp.NativeLinearLogpOp",
        gold_method="apply",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.loss.linear_logp.NativeLinearLogpOp",
            "triton": "rl_engine.kernels.ops.triton.loss.linear_logp.TritonLinearLogpOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.loss.linear_logp.FusedLinearLogpSM90Op",
        },
        grad_input_names=("hidden", "lm_head_weight"),
    ),
    "embedding": OperatorSpec(
        name="embedding",
        op_class="elementwise",
        gold_path="rl_engine.kernels.ops.pytorch.linear.embedding.NativeEmbeddingOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.linear.embedding.NativeEmbeddingOp",
            "triton": "rl_engine.kernels.ops.triton.linear.embedding.TritonEmbeddingOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.linear.embedding.SM90EmbeddingOp",
        },
        grad_input_names=("weight",),
    ),
    "lm_head": OperatorSpec(
        name="lm_head",
        op_class="reduction",
        gold_path="rl_engine.kernels.ops.pytorch.linear.lm_head.NativeLMHeadOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.linear.lm_head.NativeLMHeadOp",
            "triton": "rl_engine.kernels.ops.triton.linear.lm_head.TritonLMHeadOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.linear.lm_head.SM90LMHeadOp",
        },
        grad_input_names=("hidden", "weight"),
    ),
    "det_gemm": OperatorSpec(
        name="det_gemm",
        op_class="reduction",
        gold_path="rl_engine.kernels.ops.pytorch.matmul.det_gemm.NativeGemmOp",
        gold_method="__call__",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.matmul.det_gemm.NativeGemmOp",
            "cuda": "rl_engine.kernels.ops.cuda.matmul.det_gemm.DetGemmOp",
            "triton": "rl_engine.kernels.ops.triton.matmul.det_gemm.TritonDetGemmOp",
        },
        grad_input_names=("a", "b"),
    ),
    "rope": OperatorSpec(
        name="rope",
        op_class="elementwise",
        gold_path="rl_engine.kernels.ops.pytorch.rotary_embedding.rope.NativeRoPEOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.rotary_embedding.rope.NativeRoPEOp",
            "triton": "rl_engine.kernels.ops.triton.rotary_embedding.rope.TritonRoPEOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.rotary_embedding.rope.RoPESM90Op",
            "ascend": "rl_engine.kernels.ops.ascend.rotary_embedding.rope.RoPEAscendOp",
        },
        grad_input_names=("x",),
    ),
    "silu": OperatorSpec(
        name="silu",
        op_class="elementwise",
        gold_path="rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSiLUOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSiLUOp",
            "triton": "rl_engine.kernels.ops.triton.activation.swiglu.TritonSiLUOp",
            "cuda": "rl_engine.kernels.ops.cuda.activation.swiglu.SiLUCudaOp",
        },
        grad_input_names=("x",),
    ),
    "swiglu": OperatorSpec(
        name="swiglu",
        op_class="elementwise",
        gold_path="rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSwiGLUOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSwiGLUOp",
            "triton": "rl_engine.kernels.ops.triton.activation.swiglu.TritonSwiGLUOp",
            "cuda": "rl_engine.kernels.ops.cuda.activation.swiglu.SwiGLUCudaOp",
        },
        grad_input_names=("gate", "up"),
    ),
    "batch_invariant_logp": OperatorSpec(
        name="batch_invariant_logp",
        op_class="logprob",
        gold_path="rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp."
        "NativeBatchInvariantLogpOp",
        gold_method="apply",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp."
            "NativeBatchInvariantLogpOp",
            "triton": "rl_engine.kernels.ops.triton.loss.batch_invariant_logp."
            "TritonBatchInvariantLogpOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.loss.batch_invariant_logp."
            "BatchInvariantLogpSM90Op",
            "ascend": "rl_engine.kernels.ops.ascend.loss.batch_invariant_logp."
            "BatchInvariantLogpAscendOp",
        },
        grad_input_names=("logits",),
    ),
    "pack": OperatorSpec(
        name="pack",
        op_class="elementwise",
        gold_path="rl_engine.kernels.gtest.operator_specs.GtestPackOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.kernels.gtest.operator_specs.GtestPackOp",
        },
        grad_input_names=("x",),
    ),
}


class GtestPackOp:
    """gtest view of NativePackOp: compare the packed rows, not cu_seqlens."""

    op_class = "elementwise"

    def __init__(self) -> None:
        from rl_engine.kernels.ops.pytorch.packing.pack import NativePackOp

        self._op = NativePackOp()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed, _cu_seqlens = self._op(x, mask)
        return packed

    def forward_fp32(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed, _cu_seqlens = self._op(x.float(), mask)
        return packed


class _LogpSM90CandidateAdapter:
    def __init__(self, candidate: Any) -> None:
        self._candidate = candidate

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, logits.size(-1))
        labels_1d = token_ids.contiguous().view(-1)
        return self._candidate(logits_2d, labels_1d).view(orig_shape)


def operator_names() -> tuple[str, ...]:
    return tuple(OP_SPECS)


def make_operator_case(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> OperatorCase:
    spec = OP_SPECS[args.op]
    gold_op = _load_object(spec.gold_path)()
    gold_fn = getattr(gold_op, spec.gold_method)
    return OperatorCase(
        name=f"{args.op}-{dtype}-{operator_shape_name(args.op, args)}",
        op_class=spec.op_class,
        dtype=dtype,
        inputs=make_operator_inputs(args.op, args, dtype, device),
        gold_fn=gold_fn,
        grad_input_names=spec.grad_input_names,
    )


def make_candidate(args: argparse.Namespace) -> CandidateSpec:
    spec = OP_SPECS[args.op]
    candidate_name = "pytorch" if args.candidate == "native" else args.candidate

    if candidate_name in spec.candidate_paths:
        candidate_op = _load_object(spec.candidate_paths[candidate_name])()
        if args.op == "logp" and candidate_name == "cuda-sm90":
            candidate_op = _LogpSM90CandidateAdapter(candidate_op)
        return CandidateSpec(
            name=f"{candidate_name}-{args.op}",
            backend=candidate_name,
            arch_key=args.arch_key,
            fn=candidate_op,
        )

    supported = sorted([*spec.candidate_paths, "native"])
    raise ValueError(
        f"unsupported candidate {args.candidate!r} for op {args.op!r}; "
        f"supported candidates: {', '.join(supported)}"
    )
