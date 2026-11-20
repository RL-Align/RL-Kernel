# SPDX-License-Identifier: Apache-2.0
"""V4.1 sqrtsoftplus, deterministic top6 and routing-weight VJP on ROCm."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _route(L, BIAS, W, IDS, SAVED, E: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, B)
    logits = tl.load(L + row * E + e, e < E, other=0)
    soft = tl.where(logits > 20, logits, libdevice.log1p(libdevice.exp(logits)))
    scores = tl.sqrt(soft)
    adjusted = tl.where(e < E, scores + tl.load(BIAS + e, e < E, other=0), -float("inf"))
    j = tl.arange(0, 8)
    selected = tl.full((8,), 0, tl.float32)
    for slot in tl.static_range(6):
        best = tl.argmax(adjusted, 0, tie_break_left=True)
        score = tl.sum(tl.where(e == best, scores, 0.0), 0)
        selected = tl.where(j == slot, score, selected)
        tl.store(IDS + row * 6 + slot, best.to(tl.int64))
        adjusted = tl.where(e == best, -float("inf"), adjusted)
    denom = tl.sum(selected, 0) + 1e-20
    tl.store(SAVED + row * 6 + j, selected, j < 6)
    tl.store(W + row * 6 + j, (selected / denom) * SCALE, j < 6)


@triton.jit
def _route_bwd(DW, L, W, IDS, SAVED, DL, E: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, 8)
    ids = tl.load(IDS + row * 6 + j, j < 6, other=0)
    score = tl.load(SAVED + row * 6 + j, j < 6, other=0)
    dw = tl.load(DW + row * 6 + j, j < 6, other=0)
    weight = tl.load(W + row * 6 + j, j < 6, other=0)
    denom = tl.sum(score, 0) + 1e-20
    ds = (dw * SCALE - tl.sum(dw * weight, 0)) / denom
    logits = tl.load(L + row * E + ids, j < 6, other=0)
    sigmoid = 1.0 / (1.0 + tl.exp(-logits))
    derivative = sigmoid / (2.0 * tl.maximum(score, 1e-30))
    tl.store(DL + row * E + ids, ds * derivative, j < 6)


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias, scale):
        if logits.ndim != 2 or logits.dtype != torch.float32 or bias.shape != (logits.shape[1],):
            raise ValueError("router requires FP32 logits[T,E] and correction bias[E]")
        if (
            logits.shape[1] < 6
            or logits.device.type != "cuda"
            or not torch.version.hip
            or bias.device != logits.device
        ):
            raise ValueError("top6 routing requires >=6 experts on AMD ROCm")
        logits, bias = logits.contiguous(), bias.contiguous()
        t, e = logits.shape
        w = torch.empty((t, 6), device=logits.device, dtype=torch.float32)
        ids = torch.empty((t, 6), device=logits.device, dtype=torch.int64)
        scores = torch.empty_like(w)
        with torch.cuda.device(logits.device):
            _route[(t,)](
                logits,
                bias,
                w,
                ids,
                scores,
                e,
                scale,
                triton.next_power_of_2(e),
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(logits, w, ids, scores)
        ctx.scale = scale
        ctx.mark_non_differentiable(ids)
        return w, ids

    @staticmethod
    def backward(ctx, dw, _dids):
        logits, w, ids, scores = ctx.saved_tensors
        dl = torch.zeros_like(logits)
        with torch.cuda.device(logits.device):
            _route_bwd[(logits.shape[0],)](
                dw.contiguous(),
                logits,
                w,
                ids,
                scores,
                dl,
                logits.shape[1],
                ctx.scale,
                enable_fp_fusion=False,
            )
        return dl, None, None


def route_top6(logits, bias, *, scale=1.5):
    return _Route.apply(logits, bias, float(scale))
