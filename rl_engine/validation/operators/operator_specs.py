# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.validation.operators.op_checks import CandidateSpec, OperatorCase
from rl_engine.validation.operators.operator_inputs import make_operator_inputs, operator_shape_name


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
            "ascend": "rl_engine.kernels.ops.ascend.norm.rmsnorm.RMSNormAscendOp",
        },
        grad_input_names=("x", "weight"),
    ),
    "fused_post_norm_residual_pre_norm": OperatorSpec(
        name="fused_post_norm_residual_pre_norm",
        op_class="reduction",
        gold_path="rl_engine.kernels.ops.pytorch.norm.fused_add_norm.NativeFusedAddNormOp",
        # The only entry here whose gold is ``forward``, because for this operator
        # ``forward_fp32`` is a different computation rather than ``forward`` widened.
        # The frozen semantics narrow twice -- the post-norm casts before the residual
        # add, and the sum is cast again before the pre-norm reads it -- and
        # ``forward_fp32`` deliberately skips both. Worst absolute error over all
        # six outputs, both candidates, five shapes and four seeds:
        #
        #     dtype   vs forward_fp32   vs forward   differing vs forward
        #     fp32    5.7e-06           5.7e-06      11.0M-13.3M of 32.9M
        #     bf16    1.8e-01           3.1e-02      132-191 of 32.9M
        #     fp16    2.0e-02           7.8e-03      1603-1808 of 32.9M
        #
        # Against forward_fp32 the harness measures the narrowing contract rather
        # than the kernels. At fp32 the narrowing is a no-op and the two references
        # agree, which is why this was invisible until a dtype that narrows was run.
        # FP64-exactness of the frozen semantics is covered separately by
        # test_fused_add_norm.py's FP64 reference tests and their negative controls.
        #
        # An earlier version of this comment quoted one shape and one seed and read
        # 0.0e+00 for bf16 against forward. That was not representative: the kernels
        # are not bitwise equal to forward at bf16, they differ on roughly five
        # elements per million. The conclusion is unchanged and the margin is wider
        # than the single sample suggested.
        #
        # What this costs, said plainly rather than left to be discovered: every
        # entry in this file names one class as both its gold and its ``pytorch``
        # candidate, and the others get an independent comparison out of it because
        # their gold is ``forward_fp32`` while the candidate is called through
        # ``forward``. This one cannot, so the ``pytorch`` row compares
        # ``NativeFusedAddNormOp.forward`` against itself and is a tautology: zero
        # error by construction, not a check. The CUDA and Triton rows are real, and
        # the gold itself is held to an independent FP64 recomputation on these very
        # inputs by test_the_fused_add_norm_gold_is_checked_against_an_independent_
        # recomputation in tests/validation/operators/test_operator_inputs.py, which
        # fails if the ``1 + w`` offset, a narrowing or the eps pair moves.
        gold_method="forward",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.norm.fused_add_norm.NativeFusedAddNormOp",
            "triton": "rl_engine.kernels.ops.triton.norm.fused_add_norm.TritonFusedAddNormOp",
            "cuda": "rl_engine.kernels.ops.cuda.norm.fused_add_norm.CudaFusedAddNormOp",
            "cuda-sm90": "rl_engine.kernels.ops.cuda.norm.fused_add_norm.CudaFusedAddNormOp",
        },
        grad_input_names=("hidden", "residual", "post_weight", "pre_weight"),
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
            "ascend": "rl_engine.kernels.ops.ascend.norm.rmsnorm.RMSNormAscendOp",
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
                (
                    "rl_engine.kernels.ops.triton.attention.standard_attn.TritonBatchInvariantAt"
                    "tentionOp"
                )
            ),
            "cuda": (
                "rl_engine.kernels.ops.cuda.attention.deterministic_attn.DeterministicAttentionOp"
            ),
            "ascend": (
                (
                    "rl_engine.kernels.ops.ascend.attention.deterministic_attn.DeterministicAtte"
                    "ntionAscendOp"
                )
            ),
        },
        grad_input_names=("q", "k", "v"),
    ),
    # GRPO decode: every G group attends over one shared K/V sequence
    # ([B, Skv, D] instead of [B, Hkv, Skv, D]). Forward-only (no backward),
    # same surface as the CUDA PrefixSharedAttentionOp.
    "prefix_shared_attention": OperatorSpec(
        name="prefix_shared_attention",
        op_class="attention",
        gold_path="rl_engine.validation.operators.operator_specs.GtestPrefixSharedAttentionOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.validation.operators.operator_specs.GtestPrefixSharedAttentionOp",
            "cuda": (
                "rl_engine.kernels.ops.cuda.attention.prefix_shared_attn.PrefixSharedAttentionOp"
            ),
            "ascend": (
                (
                    "rl_engine.kernels.ops.ascend.attention.prefix_shared_attn.PrefixSharedAtten"
                    "tionAscendOp"
                )
            ),
        },
    ),
    "cp_attention": OperatorSpec(
        name="cp_attention",
        op_class="attention",
        gold_path=(
            (
                "rl_engine.kernels.ops.pytorch.attention.cp_attention.DeterministicCPAttenti"
                "onReferenceOp"
            )
        ),
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": (
                (
                    "rl_engine.kernels.ops.pytorch.attention.cp_attention.DeterministicCPAttenti"
                    "onReferenceOp"
                )
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
            "ascend": "rl_engine.kernels.ops.ascend.loss.logp.FusedLogpAscendOp",
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
            "ascend": "rl_engine.kernels.ops.ascend.loss.linear_logp.FusedLinearLogpAscendOp",
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
            "ascend": "rl_engine.kernels.ops.ascend.linear.embedding.AscendEmbeddingOp",
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
            "ascend": "rl_engine.kernels.ops.ascend.linear.lm_head.AscendLMHeadOp",
        },
        grad_input_names=("hidden", "weight"),
    ),
    "det_gemm": OperatorSpec(
        name="det_gemm",
        op_class="reduction",
        # The deterministic GEMM rounds every leaf and merge node to BF16, so
        # the accuracy gold must be the same leaf-space tree, not the
        # single-rounding torch.matmul (which fails structurally at
        # near-cancellation outputs on random inputs).
        gold_path="rl_engine.kernels.ops.pytorch.matmul.det_gemm.DetGemmTreeReferenceOp",
        gold_method="__call__",
        candidate_paths={
            "pytorch": "rl_engine.kernels.ops.pytorch.matmul.det_gemm.NativeGemmOp",
            "cuda": "rl_engine.kernels.ops.cuda.matmul.det_gemm.DetGemmOp",
            "triton": "rl_engine.kernels.ops.triton.matmul.det_gemm.TritonDetGemmOp",
            "ascend": "rl_engine.kernels.ops.ascend.matmul.det_gemm.DetGemmAscendOp",
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
            "ascend": "rl_engine.kernels.ops.ascend.activation.silu.SiLUAscendOp",
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
            "ascend": "rl_engine.kernels.ops.ascend.activation.swiglu.SwiGLUAscendOp",
        },
        grad_input_names=("gate", "up"),
    ),
    "batch_invariant_logp": OperatorSpec(
        name="batch_invariant_logp",
        op_class="logprob",
        gold_path=(
            "rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp.NativeBatchInvarian" "tLogpOp"
        ),
        gold_method="apply",
        candidate_paths={
            "pytorch": (
                "rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp.NativeBatchInvarian"
                "tLogpOp"
            ),
            "triton": (
                "rl_engine.kernels.ops.triton.loss.batch_invariant_logp.TritonBatchInvariant"
                "LogpOp"
            ),
            "cuda-sm90": (
                "rl_engine.kernels.ops.cuda.loss.batch_invariant_logp.BatchInvariantLogpSM90" "Op"
            ),
            "ascend": (
                "rl_engine.kernels.ops.ascend.loss.batch_invariant_logp.BatchInvariantLogpAs"
                "cendOp"
            ),
        },
        grad_input_names=("logits",),
    ),
    "pack": OperatorSpec(
        name="pack",
        op_class="elementwise",
        gold_path="rl_engine.validation.operators.operator_specs.GtestPackOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.validation.operators.operator_specs.GtestPackOp",
        },
        grad_input_names=("x",),
    ),
}


class GtestPackOp:
    """gtest view of NativePackOp: compare the packed rows, not cu_seqlens."""

    op_class = "elementwise"

    def __init__(self) -> None:
        from rl_engine.reference.packing.pack import NativePackOp

        self._op = NativePackOp()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed, _cu_seqlens = self._op(x, mask)
        return packed

    def forward_fp32(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed, _cu_seqlens = self._op(x.float(), mask)
        return packed


class GtestPrefixSharedAttentionOp:
    """gtest view of the prefix-shared layout: expand the shared K/V over the
    G groups and reuse the standard fp32 attention reference (non-causal,
    default scale), which is the gold for the CUDA/Ascend prefix-shared ops.
    """

    op_class = "attention"

    def __init__(self) -> None:
        from rl_engine.reference.attention.standard_attn import NativeAttentionOp

        self._op = NativeAttentionOp()

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._op(q, k.unsqueeze(1), v.unsqueeze(1), causal=False)

    def forward_fp32(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self._op.forward_fp32(q, k.unsqueeze(1), v.unsqueeze(1), causal=False)


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
