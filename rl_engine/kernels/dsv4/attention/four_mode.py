# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measured execution paths for T06, with frozen recorded candidate state."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import (
    MqaJointAttentionSinkOp,
    cuda_forward_into,
    make_cuda_fwd_workspace,
)
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.finite import CheckedCUDAGraph
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.dsv4.attention.state_gate import StateGateVerdict, require_state_gate


def _same_bytes(a: Tensor, b: Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def verify_recorded_four_modes(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    w_a: Tensor,
    w_b: Tensor,
    cos: Tensor,
    sin: Tensor,
    state_gate: StateGateVerdict,
) -> dict:
    """Compare recorded outputs and VJPs across training, prefill and eager/graph decode."""
    if not q.is_cuda or q.shape[0] < 2:
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "four-mode verification requires CUDA and at least two query rows",
        )
    require_state_gate(state_gate)
    plan.validate_for_kv(k, v)
    # Validate kind tags before capture to avoid host synchronization.
    frozen = replace(plan, valid=plan.valid.detach().clone(), kind=None)
    attn = MqaJointAttentionSinkOp(backend="cuda")
    proj = OProjGroupedOp(backend="det_gemm")
    inputs = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v, sink, w_a, w_b))
    qs, ks, vs, ss, wa, wb = inputs
    dy = torch.linspace(-0.5, 0.5, q.shape[0] * w_b.shape[0], device=q.device)
    dy = dy.reshape(q.shape[0], w_b.shape[0]).contiguous()

    def forward(qi, si, ci, ti, *, training):
        if training:
            o = attn.apply_autograd(qi, ks, vs, si, frozen, output_fp32=True)
        else:
            o = attn.forward_fp32(qi, ks, vs, si, frozen).o
        y = proj.forward(o.to(torch.bfloat16), wa, wb, ci, ti).y
        return o, y

    def train():
        with torch.enable_grad():
            o, y = forward(qs, ss, cos, sin, training=True)
            grads = torch.autograd.grad(y, inputs, grad_outputs=dy.to(y.dtype))
        return o, y, grads

    # Warm up forward and backward on a side stream before capture.
    stream = torch.cuda.Stream(device=q.device)
    stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(stream):
        train()
    torch.cuda.current_stream(q.device).wait_stream(stream)
    training_graph = CheckedCUDAGraph()
    with training_graph.capture(stream=stream):
        graph_o, graph_y, graph_grads = train()

    decode_q = qs[:1].detach().clone()
    decode_sink = (ss if ss.ndim == 1 else ss[:1]).detach().clone()
    decode_cos, decode_sin = cos[:1].clone(), sin[:1].clone()

    def decode():
        with torch.no_grad():
            return forward(decode_q, decode_sink, decode_cos, decode_sin, training=False)

    stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(stream):
        decode()
    torch.cuda.current_stream(q.device).wait_stream(stream)
    decode_graph = CheckedCUDAGraph()
    with decode_graph.capture(stream=stream):
        decode_o, decode_y = decode()

    checks = {}
    for replay in range(2):
        if replay:
            with torch.no_grad():
                qs.copy_(-q)
                vs.copy_(-v)
                ss.add_(0.125)
                dy.neg_()
        train_o, train_y, train_grads = train()
        with torch.no_grad():
            prefill_o, prefill_y = forward(qs, ss, cos, sin, training=False)
            eager_rows, graph_rows = [], []
            for row in range(q.shape[0]):
                sr = ss if ss.ndim == 1 else ss[row : row + 1]
                cr, tr = cos[row : row + 1], sin[row : row + 1]
                eager_rows.append(forward(qs[row : row + 1], sr, cr, tr, training=False))
                decode_q.copy_(qs[row : row + 1])
                decode_sink.copy_(sr)
                decode_cos.copy_(cr)
                decode_sin.copy_(tr)
                decode_graph.replay()
                graph_rows.append((decode_o.clone(), decode_y.clone()))
            for mode, rows in (("eager_decode", eager_rows), ("graph_decode", graph_rows)):
                checks[f"replay{replay}.{mode}.O"] = _same_bytes(
                    train_o, torch.cat([r[0] for r in rows])
                )
                checks[f"replay{replay}.{mode}.Y"] = _same_bytes(
                    train_y, torch.cat([r[1] for r in rows])
                )
            checks[f"replay{replay}.prefill.O"] = _same_bytes(train_o, prefill_o)
            checks[f"replay{replay}.prefill.Y"] = _same_bytes(train_y, prefill_y)
        training_graph.replay()
        checks[f"replay{replay}.graph_training.O"] = _same_bytes(train_o, graph_o)
        checks[f"replay{replay}.graph_training.Y"] = _same_bytes(train_y, graph_y)
        for name, ref, actual in zip(
            ("dQ", "dK", "dV", "dsink", "dW_a", "dW_b"), train_grads, graph_grads,
            strict=True,
        ):
            checks[f"replay{replay}.graph_training.{name}"] = _same_bytes(ref, actual)
        if not all(torch.isfinite(t).all().item() for t in (train_o, train_y, *train_grads)):
            raise DSv4FailClosedError(DSv4Status.NON_FINITE, "four-mode reference is non-finite")
    passed = all(checks.values())
    return {
        "scope": "recorded attention + inverse RoPE + grouped DetGemm o-proj",
        "candidate_state": "frozen; no live cache or compressor",
        "tokens": q.shape[0],
        "layer_type": plan.layer_type.value,
        "backward": "DetGemm autograd VJP; explicit FP32 weight VJP is tested separately",
        "graph_includes_backward": True,
        "replays_with_changed_inputs": 2,
        "checks": checks,
        "status": DSv4Status.PASS.value if passed else DSv4Status.BYTE_MISMATCH.value,
        "four_mode_equal": passed,
    }


def eager_vs_cuda_graph_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    state_gate: StateGateVerdict,
) -> tuple[Tensor, Tensor]:
    """Capture a CUDA graph on static addresses and compare to eager bytes."""

    if not q.is_cuda:
        raise DSv4FailClosedError(
            DSv4Status.UNSUPPORTED_CAPABILITY,
            "CUDA-graph four-mode requires CUDA tensors",
        )
    require_state_gate(state_gate)
    plan.validate_for_kv(k, v)

    q_s = q.detach().contiguous()
    k_s = k.detach().contiguous()
    v_s = v.detach().contiguous()
    workspace = make_cuda_fwd_workspace(q_s, k_s, sink, plan.valid, output_fp32=True)

    cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    eager = workspace.out.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    torch.cuda.current_stream().wait_stream(stream)

    graph = CheckedCUDAGraph()
    with graph.capture():
        cuda_forward_into(q_s, k_s, v_s, workspace, output_fp32=True)
    graph.replay()
    if not torch.equal(eager, workspace.out):
        raise DSv4FailClosedError(
            DSv4Status.BYTE_MISMATCH,
            "CUDA-graph decode bytes differ from eager",
        )
    return eager, workspace.out.clone()
