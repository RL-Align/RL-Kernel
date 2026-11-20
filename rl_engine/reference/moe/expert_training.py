# SPDX-License-Identifier: Apache-2.0
"""CPU-only task 48 diagnostic provider, adapted from the P5-S0 staged arithmetic.

This is not a production provider or a bitwise oracle for an AMD exponential/GEMM.
The route-gradient calculation below is a test reference, not task 49's GPU kernel.
"""

import torch

from rl_engine.contracts.operators.expert_training import (
    ExpertActivationState,
    QuantizedExpertActivation,
)


def serial_mm(a, b):
    y = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)
    for k in range(a.shape[1]):
        y = y + a[:, k].float()[:, None] * b[k].float()[None, :]
    return y


def decode_scales(codes):
    if bool((codes == 255).any()):
        raise ValueError("invalid E8M0 scale")
    return torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes.int() - 127)


def decode_weights(w):
    nibbles = torch.stack((w.codes & 15, w.codes >> 4), dim=-1).reshape(w.shape)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=w.codes.device)
    elems = table[(nibbles & 7).long()] * torch.where(nibbles < 8, 1.0, -1.0)
    return elems, decode_scales(w.scales)


class ReferenceExpertTrainingBackend:
    profile = "dsv4-expert-training-reference-serial-v1"

    def validate(self, config, device, need_route_grad):
        if device.type != "cpu":
            raise RuntimeError("diagnostic expert provider is CPU-only")

    def quantize(self, x):
        if not bool(torch.isfinite(x).all()):
            raise ValueError("MX activation quantization requires finite inputs")
        blocks = x.float().reshape(x.shape[0], x.shape[1] // 32, 32)
        amax = blocks.abs().amax(-1)
        _, exponent = torch.frexp(amax.clamp_min(torch.finfo(torch.float32).tiny))
        scales = (exponent.int() - 1 - 8).clamp(-127, 127) + 127
        scales = torch.where(amax == 0, 127, scales).to(torch.uint8)
        normalized = blocks / decode_scales(scales).unsqueeze(-1)
        codes = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        return QuantizedExpertActivation(codes.reshape(x.shape), scales)

    def base_forward(self, q, w, offsets):
        ae = q.codes.view(torch.float8_e4m3fn).float()
        sa = decode_scales(q.scales)
        we, sw = decode_weights(w)
        y = torch.zeros((ae.shape[0], w.shape[1]), dtype=torch.float32, device=ae.device)
        for e in range(w.shape[0]):
            lo, hi = int(offsets[e]), int(offsets[e + 1])
            for block in range(w.shape[2] // 32):
                start = block * 32
                partial = serial_mm(ae[lo:hi, start : start + 32], we[e, :, start : start + 32].t())
                y[lo:hi] += (partial * sa[lo:hi, block, None]) * sw[e, :, block][None, :]
        return y

    def base_backward(self, dy, w, offsets):
        we, sw = decode_weights(w)
        wb = (
            (we.reshape(*w.shape[:-1], w.shape[-1] // 32, 32) * sw.unsqueeze(-1))
            .reshape(w.shape)
            .bfloat16()
        )
        dx = torch.zeros((dy.shape[0], w.shape[-1]), dtype=torch.float32, device=dy.device)
        for e in range(w.shape[0]):
            lo, hi = int(offsets[e]), int(offsets[e + 1])
            dx[lo:hi] = serial_mm(dy[lo:hi].bfloat16(), wb[e])
        return dx

    def lora_forward(self, x, a, b, alpha):
        u = serial_mm(x, a.t()).bfloat16()
        return serial_mm(u, b.t()) * alpha, u

    def lora_backward(self, dy, x, a, b, alpha, u):
        dv = (dy.float() * alpha).bfloat16()
        du = serial_mm(dv, b).bfloat16()
        return serial_mm(du, a), serial_mm(du.t(), x), serial_mm(dv.t(), u)

    def activation_forward(self, gate, up, p):
        g, u = gate.clamp(max=10), up.clamp(-10, 10)
        sig = torch.sigmoid(g)
        silu = g * sig
        raw = silu * u
        return (raw * p[:, None]).bfloat16(), ExpertActivationState(sig, silu, raw)

    def activation_backward(self, dh, gate, up, p, state, need_route_grad):
        g, u = gate.clamp(max=10), up.clamp(-10, 10)
        weighted = dh.float() * p[:, None]
        dsilu = state.sigmoid * (1 + g * (1 - state.sigmoid))
        dg = ((weighted * u) * dsilu) * (gate < 10)
        du = (weighted * state.silu) * ((up > -10) & (up < 10))
        dp = None
        if need_route_grad:
            dp = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            for n in range(dh.shape[1]):
                dp = dp + (dh[:, n].float() * state.silu[:, n]) * u[:, n]
        return dg, du, dp
