# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MQA attention with a shared denominator for candidates and sink."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.dsv4.attention.oracle import (
    AttentionBackwardState,
    AttentionForwardTensors,
    mqa_joint_attention_sink_bwd,
    mqa_joint_attention_sink_fwd,
)
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import (
    ATTENTION_SCALE,
    CUDA_MAX_TOKENS,
    KERNEL_ID_CUDA,
    KERNEL_ID_ORACLE,
    SCHEMA_VERSION_ATTENTION,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.finite import capture_finite_checks, require_finite
from rl_engine.kernels.dsv4.attention.provenance import ActualProvenance
from rl_engine.kernels.dsv4.attention.state_gate import StateGateVerdict, require_state_gate


def cuda_kernel_available() -> bool:
    return bool(
        _EXT_AVAILABLE
        and _C is not None
        and hasattr(_C, "mqa_joint_attention_sink_forward")
        and hasattr(_C, "mqa_joint_attention_sink_backward")
    )


@dataclass
class AttentionResult:
    o: Tensor
    provenance: ActualProvenance
    debug: dict[str, Tensor] | None = None
    saved: AttentionForwardTensors | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION_ATTENTION,
            "provenance": self.provenance.to_dict(),
            "has_debug": self.debug is not None,
        }


def _oracle_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    *,
    output_dtype: torch.dtype,
    debug: bool,
) -> AttentionResult:
    saved = mqa_joint_attention_sink_fwd(q, k, v, sink, plan)
    o = saved.o if output_dtype == torch.float32 else saved.o.to(output_dtype)
    debug_tensors = None
    if debug:
        debug_tensors = {
            "logits": saved.logits,
            "m": saved.m,
            "e": saved.e,
            "e_sink": saved.e_sink,
            "z": saved.z,
            "p": saved.p,
            "p_sink": saved.p_sink,
        }
    return AttentionResult(
        o=o,
        provenance=ActualProvenance(
            backend="oracle",
            kernel_id=KERNEL_ID_ORACLE,
            device=str(q.device),
            dtype=str(output_dtype),
            debug=debug,
            build_fingerprint=SCHEMA_VERSION_ATTENTION,
        ),
        debug=debug_tensors,
        saved=saved,
    )


def _cuda_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    *,
    output_fp32: bool,
    debug: bool,
) -> AttentionResult:
    if not cuda_kernel_available():
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "CUDA mqa_joint_attention_sink kernel is not in the compiled extension",
        )
    if not q.is_cuda:
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "CUDA backend requires CUDA tensors",
        )
    if q.shape[0] > CUDA_MAX_TOKENS:
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "CUDA reference QK grid.y cannot exceed 65535 tokens",
        )
    plan.validate_for_kv(k, v)
    sink_th = sink.to(device=q.device, dtype=torch.float32)
    if sink_th.dim() == 1:
        sink_th = sink_th.unsqueeze(0).expand(q.shape[0], -1).contiguous()
    else:
        sink_th = sink_th.contiguous()
    results = _C.mqa_joint_attention_sink_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        sink_th,
        plan.valid.to(device=q.device).contiguous(),
        float(ATTENTION_SCALE),
        bool(output_fp32),
    )
    o, p, p_sink, m, z = results[0], results[1], results[2], results[3], results[4]
    require_finite((o, p, p_sink, m, z), "CUDA forward produced non-finite attention values")
    e_sink = p_sink * z
    saved = AttentionForwardTensors(
        o=o.float(),
        p=p,
        p_sink=p_sink,
        m=m,
        z=z,
        logits=p.new_empty(0),
        e=p.new_empty(0),
        e_sink=e_sink,
        sink=sink_th,
        q=q,
        k=k,
        v=v,
        valid=plan.valid.to(device=q.device),
    )
    debug_tensors = None
    if debug:
        debug_tensors = {"m": m, "z": z, "p": p, "p_sink": p_sink, "e_sink": e_sink}
    return AttentionResult(
        o=o,
        provenance=ActualProvenance(
            backend="cuda",
            kernel_id=KERNEL_ID_CUDA,
            device=str(q.device),
            dtype=str(o.dtype),
            debug=debug,
            tile="qk_grid=(N,T)_block=64_pv_block=512",
            unroll="pragma_unroll_8_sequential_d_0_511",
            build_fingerprint=f"{KERNEL_ID_CUDA};unroll8;grid_y_max=65535;device={q.device}",
        ),
        debug=debug_tensors,
        saved=saved,
    )


@dataclass
class CudaFwdWorkspace:
    """Reusable buffers for CUDA graph capture."""

    out: Tensor
    scores: Tensor
    p_sink: Tensor
    m: Tensor
    z: Tensor
    sink_th: Tensor
    valid: Tensor


def make_cuda_fwd_workspace(
    q: Tensor,
    k: Tensor,
    sink: Tensor,
    valid: Tensor,
    *,
    output_fp32: bool = True,
) -> CudaFwdWorkspace:
    tokens, heads, dim = q.shape
    n_cand = int(k.shape[0])
    sink_th = sink.to(device=q.device, dtype=torch.float32)
    if sink_th.dim() == 1:
        sink_th = sink_th.unsqueeze(0).expand(tokens, heads).contiguous()
    else:
        sink_th = sink_th.contiguous()
    return CudaFwdWorkspace(
        out=torch.empty(
            tokens,
            heads,
            dim,
            dtype=torch.float32 if output_fp32 else q.dtype,
            device=q.device,
        ),
        scores=torch.empty(tokens, heads, n_cand, dtype=torch.float32, device=q.device),
        p_sink=torch.empty(tokens, heads, dtype=torch.float32, device=q.device),
        m=torch.empty(tokens, heads, dtype=torch.float32, device=q.device),
        z=torch.empty(tokens, heads, dtype=torch.float32, device=q.device),
        sink_th=sink_th,
        valid=valid.to(device=q.device).contiguous(),
    )


def cuda_forward_into(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    workspace: CudaFwdWorkspace,
    *,
    output_fp32: bool = True,
) -> None:
    """Write attention into static buffers; graph checks use captured scalar flags."""

    if _C is None or not hasattr(_C, "mqa_joint_attention_sink_forward_into"):
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "CUDA-graph path requires mqa_joint_attention_sink_forward_into",
        )
    _C.mqa_joint_attention_sink_forward_into(
        q,
        k,
        v,
        workspace.sink_th,
        workspace.valid,
        float(ATTENTION_SCALE),
        bool(output_fp32),
        workspace.out,
        workspace.scores,
        workspace.p_sink,
        workspace.m,
        workspace.z,
    )
    require_finite(
        (workspace.out, workspace.scores, workspace.p_sink, workspace.m, workspace.z),
        "CUDA forward produced non-finite attention values",
    )


class _MqaJointAttentionSinkFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sink: Tensor,
        plan: CandidatePlan,
        output_dtype_is_fp32: bool,
        use_cuda: bool,
    ) -> Tensor:
        if use_cuda:
            result = _cuda_forward(
                q, k, v, sink, plan, output_fp32=output_dtype_is_fp32, debug=False
            )
        else:
            out_dtype = torch.float32 if output_dtype_is_fp32 else q.dtype
            result = _oracle_forward(q, k, v, sink, plan, output_dtype=out_dtype, debug=False)
        assert result.saved is not None
        ctx.save_for_backward(
            result.saved.q,
            result.saved.k,
            result.saved.v,
            result.saved.sink,
            result.saved.p,
            result.saved.p_sink,
            result.saved.valid,
        )
        ctx.sink_was_shared = sink.dim() == 1
        ctx.use_cuda = use_cuda
        # Autograd workers do not inherit Python context variables.
        ctx.finite_checks = capture_finite_checks()
        return result.o

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        q, k, v, sink, p, p_sink, valid = ctx.saved_tensors
        if ctx.use_cuda:
            if not cuda_kernel_available():
                raise DSv4FailClosedError(
                    DSv4Status.UNSUPPORTED_CAPABILITY,
                    "CUDA forward cannot fall back to oracle backward",
                )
            require_finite(
                (grad_out,),
                "CUDA backward received a non-finite gradient",
                capture_checks=ctx.finite_checks,
            )
            grads = _C.mqa_joint_attention_sink_backward(
                grad_out.float().contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                sink.contiguous(),
                valid.contiguous(),
                p.contiguous(),
                p_sink.contiguous(),
                float(ATTENTION_SCALE),
                bool(ctx.sink_was_shared),
            )
            dq, dk, dv, dsink = grads[0], grads[1], grads[2], grads[3]
            require_finite(
                (dq, dk, dv, dsink),
                "CUDA backward produced non-finite gradients",
                capture_checks=ctx.finite_checks,
            )
        else:
            saved = AttentionBackwardState(q=q, k=k, v=v, p=p, p_sink=p_sink, valid=valid)
            bwd = mqa_joint_attention_sink_bwd(grad_out, saved, sink_was_shared=ctx.sink_was_shared)
            dq, dk, dv, dsink = bwd.dq, bwd.dk, bwd.dv, bwd.dsink
            if q.dtype != torch.float32:
                dq = dq.to(q.dtype)
                dk = dk.to(k.dtype)
                dv = dv.to(v.dtype)
        return dq, dk, dv, dsink, None, None, None


class MqaJointAttentionSinkOp:
    """C0/CSA/HCA joint attention. State gate is required before comparison."""

    op_class = "attention"

    def __init__(self, *, backend: str = "auto") -> None:
        if backend not in {"auto", "oracle", "cuda"}:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                f"backend must be auto|oracle|cuda, got {backend!r}",
            )
        self.backend = backend

    def _resolve_backend(self, q: Tensor) -> str:
        if self.backend == "oracle":
            return "oracle"
        if self.backend == "cuda":
            if not cuda_kernel_available() or not q.is_cuda:
                raise DSv4FailClosedError(
                    DSv4Status.UNSUPPORTED_CAPABILITY,
                    "cuda backend requested but kernel/device unavailable",
                )
            return "cuda"
        if q.is_cuda:
            if not cuda_kernel_available():
                raise DSv4FailClosedError(
                    DSv4Status.UNSUPPORTED_CAPABILITY,
                    "auto/cuda on GPU tensors requires the T06 CUDA kernel; "
                    "refusing oracle fallback",
                )
            return "cuda"
        return "oracle"

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sink: Tensor,
        plan: CandidatePlan,
        *,
        output_dtype: torch.dtype | None = None,
        debug: bool = False,
        state_gate: StateGateVerdict | None = None,
        compare: bool = False,
    ) -> AttentionResult:
        if compare:
            require_state_gate(state_gate)
        resolved = self._resolve_backend(q)
        out_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved == "cuda":
            return _cuda_forward(
                q, k, v, sink, plan, output_fp32=out_dtype == torch.float32, debug=debug
            )
        return _oracle_forward(q, k, v, sink, plan, output_dtype=out_dtype, debug=debug)

    def forward_fp32(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sink: Tensor,
        plan: CandidatePlan,
        *,
        debug: bool = False,
        state_gate: StateGateVerdict | None = None,
        compare: bool = False,
    ) -> AttentionResult:
        return self.forward(
            q,
            k,
            v,
            sink,
            plan,
            output_dtype=torch.float32,
            debug=debug,
            state_gate=state_gate,
            compare=compare,
        )

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sink: Tensor,
        plan: CandidatePlan,
        **kwargs: Any,
    ) -> Tensor:
        return self.forward(q, k, v, sink, plan, **kwargs).o

    def apply_autograd(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sink: Tensor,
        plan: CandidatePlan,
        *,
        output_fp32: bool = False,
        state_gate: StateGateVerdict | None = None,
        compare: bool = False,
    ) -> Tensor:
        if compare:
            require_state_gate(state_gate)
        plan.validate_for_kv(k, v)
        resolved = self._resolve_backend(q)
        return _MqaJointAttentionSinkFn.apply(
            q,
            k,
            v,
            sink,
            plan,
            output_fp32,
            resolved == "cuda",
        )
