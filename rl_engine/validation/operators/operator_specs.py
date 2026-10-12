# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.validation.operators.op_checks import CandidateSpec, OperatorCase
from rl_engine.validation.operators.operator_inputs import (
    make_operator_inputs,
    operator_shape_name,
)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    op_class: str
    gold_path: str
    gold_method: str
    candidate_paths: dict[str, str]
    grad_input_names: tuple[str, ...] = ()
    gold_input_dtype: torch.dtype | None = None
    output_matches_input_dtype: bool = False


def _load_object(path: str) -> Any:
    module_path, object_name = path.rsplit(".", 1)
    # dynamic loading ops
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


OP_SPECS = {
    "adaln_modulation": OperatorSpec(
        name="adaln_modulation",
        op_class="reduction",
        gold_path="rl_engine.validation.operators.operator_specs.GtestAdaLNReference",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": (
                "rl_engine.reference.norm.adaln_modulation.NativeAdaLNModulationOp"
            ),
            "triton": (
                "rl_engine.backends.shared.triton.norm.adaln_modulation.TritonAdaLNModulationOp"
            ),
            "cuda": "rl_engine.backends.cuda.norm.adaln_modulation.CudaAdaLNModulationOp",
        },
        grad_input_names=("x", "modulation"),
        gold_input_dtype=torch.float32,
        output_matches_input_dtype=True,
    ),
    "rms_norm": OperatorSpec(
        name="rms_norm",
        op_class="reduction",
        gold_path="rl_engine.reference.norm.rms_norm.NativeRMSNormOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.norm.rms_norm.NativeRMSNormOp",
            "triton": "rl_engine.backends.shared.triton.norm.rmsnorm.RMSNormTritonOp",
            "cuda": "rl_engine.backends.cuda.norm.rmsnorm.RMSNormCudaOp",
            "cuda-sm90": "rl_engine.backends.cuda.norm.rmsnorm.RMSNormCudaOp",
            "ascend": "rl_engine.backends.ascend.norm.rmsnorm.RMSNormAscendOp",
        },
        grad_input_names=("x", "weight"),
    ),
    "qk_norm": OperatorSpec(
        name="qk_norm",
        op_class="reduction",
        gold_path="rl_engine.reference.norm.rms_norm.NativeRMSNormOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.norm.rms_norm.NativeRMSNormOp",
            "triton": "rl_engine.backends.shared.triton.norm.rmsnorm.RMSNormTritonOp",
            "cuda": "rl_engine.backends.cuda.norm.rmsnorm.RMSNormCudaOp",
            "cuda-sm90": "rl_engine.backends.cuda.norm.rmsnorm.RMSNormCudaOp",
            "ascend": "rl_engine.backends.ascend.norm.rmsnorm.RMSNormAscendOp",
        },
        grad_input_names=("x", "weight"),
    ),
    "attention": OperatorSpec(
        name="attention",
        op_class="attention",
        gold_path="rl_engine.reference.attention.standard_attn.NativeAttentionOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.attention.standard_attn.NativeAttentionOp",
            "triton": (
                "rl_engine.backends.shared.triton.attention.standard_attn.TritonBatchInvariantAt"
                "tentionOp"
            ),
            "cuda": (
                "rl_engine.backends.cuda.attention.deterministic_attn.DeterministicAttentionOp"
            ),
            "ascend": (
                "rl_engine.backends.ascend.attention.deterministic_attn.DeterministicAtte"
                "ntionAscendOp"
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
                "rl_engine.backends.cuda.attention.prefix_shared_attn.PrefixSharedAttentionOp"
            ),
            "ascend": (
                "rl_engine.backends.ascend.attention.prefix_shared_attn.PrefixSharedAtten"
                "tionAscendOp"
            ),
        },
    ),
    "cp_attention": OperatorSpec(
        name="cp_attention",
        op_class="attention",
        gold_path=(
            "rl_engine.reference.attention.cp_attention.DeterministicCPAttentionReferenceOp"
        ),
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": (
                "rl_engine.reference.attention.cp_attention.DeterministicCPAttentionReferenceOp"
            ),
        },
        grad_input_names=("q", "k", "v"),
    ),
    "logp": OperatorSpec(
        name="logp",
        op_class="logprob",
        gold_path="rl_engine.reference.logprob.logp.NativeLogpOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.logprob.logp.NativeLogpOp",
            "triton": "rl_engine.backends.shared.triton.logprob.logp.TritonLogpOp",
            "cuda": "rl_engine.backends.cuda.logprob.logp.FusedLogpGenericOp",
            "cuda-generic": "rl_engine.backends.cuda.logprob.logp.FusedLogpGenericOp",
            "cuda-sm90": "rl_engine.backends.cuda.logprob.logp.FusedLogpSM90Op",
            "ascend": "rl_engine.backends.ascend.logprob.logp.FusedLogpAscendOp",
        },
        grad_input_names=("logits",),
    ),
    "linear_logp": OperatorSpec(
        name="linear_logp",
        op_class="logprob",
        gold_path="rl_engine.reference.logprob.linear_logp.NativeLinearLogpOp",
        gold_method="apply",
        candidate_paths={
            "pytorch": "rl_engine.reference.logprob.linear_logp.NativeLinearLogpOp",
            "triton": "rl_engine.backends.shared.triton.logprob.linear_logp.TritonLinearLogpOp",
            "cuda-sm90": "rl_engine.backends.cuda.logprob.linear_logp.FusedLinearLogpSM90Op",
            "ascend": "rl_engine.backends.ascend.logprob.linear_logp.FusedLinearLogpAscendOp",
        },
        grad_input_names=("hidden", "lm_head_weight"),
    ),
    "embedding": OperatorSpec(
        name="embedding",
        op_class="elementwise",
        gold_path="rl_engine.reference.embedding.embedding.NativeEmbeddingOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.embedding.embedding.NativeEmbeddingOp",
            "triton": "rl_engine.backends.shared.triton.embedding.embedding.TritonEmbeddingOp",
            "cuda-sm90": "rl_engine.backends.cuda.embedding.embedding.SM90EmbeddingOp",
            "ascend": "rl_engine.backends.ascend.embedding.embedding.AscendEmbeddingOp",
        },
        grad_input_names=("weight",),
    ),
    "lm_head": OperatorSpec(
        name="lm_head",
        op_class="reduction",
        gold_path="rl_engine.reference.embedding.lm_head.NativeLMHeadOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.embedding.lm_head.NativeLMHeadOp",
            "triton": "rl_engine.backends.shared.triton.embedding.lm_head.TritonLMHeadOp",
            "cuda-sm90": "rl_engine.backends.cuda.embedding.lm_head.SM90LMHeadOp",
            "ascend": "rl_engine.backends.ascend.embedding.lm_head.AscendLMHeadOp",
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
        gold_path="rl_engine.reference.gemm.det_gemm.DetGemmTreeReferenceOp",
        gold_method="__call__",
        candidate_paths={
            "pytorch": "rl_engine.reference.gemm.det_gemm.NativeGemmOp",
            "cuda": "rl_engine.backends.cuda.gemm.det_gemm.DetGemmOp",
            "triton": "rl_engine.backends.shared.triton.gemm.det_gemm.TritonDetGemmOp",
            "ascend": "rl_engine.backends.ascend.gemm.det_gemm.DetGemmAscendOp",
        },
        grad_input_names=("a", "b"),
    ),
    "rope": OperatorSpec(
        name="rope",
        op_class="elementwise",
        gold_path="rl_engine.reference.rope.rope.NativeRoPEOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.rope.rope.NativeRoPEOp",
            "triton": "rl_engine.backends.shared.triton.rope.rope.TritonRoPEOp",
            "cuda-sm90": "rl_engine.backends.cuda.rope.rope.RoPESM90Op",
            "ascend": "rl_engine.backends.ascend.rope.rope.RoPEAscendOp",
        },
        grad_input_names=("x",),
    ),
    "silu": OperatorSpec(
        name="silu",
        op_class="elementwise",
        gold_path="rl_engine.reference.activation.swiglu.NativeSiLUOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.activation.swiglu.NativeSiLUOp",
            "triton": "rl_engine.backends.shared.triton.activation.swiglu.TritonSiLUOp",
            "cuda": "rl_engine.backends.cuda.activation.swiglu.SiLUCudaOp",
            "ascend": "rl_engine.backends.ascend.activation.silu.SiLUAscendOp",
        },
        grad_input_names=("x",),
    ),
    "swiglu": OperatorSpec(
        name="swiglu",
        op_class="elementwise",
        gold_path="rl_engine.reference.activation.swiglu.NativeSwiGLUOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.reference.activation.swiglu.NativeSwiGLUOp",
            "triton": "rl_engine.backends.shared.triton.activation.swiglu.TritonSwiGLUOp",
            "cuda": "rl_engine.backends.cuda.activation.swiglu.SwiGLUCudaOp",
            "ascend": "rl_engine.backends.ascend.activation.swiglu.SwiGLUAscendOp",
        },
        grad_input_names=("gate", "up"),
    ),
    "batch_invariant_logp": OperatorSpec(
        name="batch_invariant_logp",
        op_class="logprob",
        gold_path=("rl_engine.reference.logprob.batch_invariant_logp.NativeBatchInvariantLogpOp"),
        gold_method="apply",
        candidate_paths={
            "pytorch": (
                "rl_engine.reference.logprob.batch_invariant_logp.NativeBatchInvariantLogpOp"
            ),
            "triton": (
                "rl_engine.backends.shared.triton.logprob.batch_invariant_logp.TritonBatchInvariant"
                "LogpOp"
            ),
            "cuda-sm90": (
                "rl_engine.backends.cuda.logprob.batch_invariant_logp.BatchInvariantLogpSM90Op"
            ),
            "ascend": (
                "rl_engine.backends.ascend.logprob.batch_invariant_logp.BatchInvariantLogpAscendOp"
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


class _GtestSharedAdaLNAffine(torch.autograd.Function):
    """Independent CPU affine VJP with the specified ascending FP32 S fold."""

    @staticmethod
    def forward(ctx, norm, modulation):
        shift, scale, gate = modulation.chunk(3, dim=-1)
        ctx.save_for_backward(norm, scale)
        return norm * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]

    @staticmethod
    def backward(ctx, dy, dg):
        norm, scale = ctx.saved_tensors
        shift_grad = torch.zeros_like(scale)
        scale_grad = torch.zeros_like(scale)
        # CPU torch's broadcast backward may reassociate this reduction. The
        # reference follows the declared token order without any kernel helper.
        for token in range(norm.shape[1]):
            shift_grad = shift_grad + dy[:, token, :]
            scale_grad = scale_grad + dy[:, token, :] * norm[:, token, :]
        dm = torch.cat((shift_grad, scale_grad, dg[:, 0, :]), dim=-1)
        return dy * (1 + scale[:, None, :]), dm


class GtestAdaLNReference:
    """Independent CPU FP32 LayerNorm/autograd gold, not the fixed-tree kernel.

    Transfers/casts retain autograd edges to the caller's inputs. Outputs stay
    FP32; no production normalization or backward helper is used. Shared
    affine gradients use a separate CPU VJP for the contract's FP32 token fold.
    """

    def forward_fp32(self, x, modulation, *, eps=1e-6, modulate_index=None):
        x32 = x.to(device="cpu", dtype=torch.float32)
        mod32 = modulation.to(device="cpu", dtype=torch.float32)
        if modulate_index is None:
            norm = torch.nn.functional.layer_norm(x32, (x.shape[-1],), eps=eps)
            return _GtestSharedAdaLNAffine.apply(norm, mod32)
        else:
            batch = x.shape[0]
            rows = torch.arange(batch)[:, None] + modulate_index.cpu().long() * batch
            shift, scale, gate = mod32[rows].chunk(3, dim=-1)
        norm = torch.nn.functional.layer_norm(x32, (x.shape[-1],), eps=eps)
        return norm * (1 + scale) + shift, gate


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
        from rl_engine.reference.attention.standard_attn import (
            NativeAttentionOp,
        )

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
        gold_input_dtype=spec.gold_input_dtype,
        candidate_output_dtype=dtype if spec.output_matches_input_dtype else None,
    )


def make_candidate(args: argparse.Namespace, *, observe_backend: bool = False) -> CandidateSpec:
    spec = OP_SPECS[args.op]
    candidate_name = "pytorch" if args.candidate == "native" else args.candidate

    if candidate_name in spec.candidate_paths:
        candidate_op = _load_object(spec.candidate_paths[candidate_name])()
        if args.op == "adaln_modulation" and observe_backend:
            from rl_engine.runtime.adaln import SHARED_PROFILE, AdaLNDispatch

            candidate_op = AdaLNDispatch(
                candidate_op,
                {
                    "requested_backend": candidate_name,
                    "fallback": False,
                    "fallback_reason": None,
                    "arithmetic_profile": SHARED_PROFILE,
                },
                hidden=args.normalized_dim,
            )
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
