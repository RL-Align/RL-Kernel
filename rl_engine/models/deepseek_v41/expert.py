# SPDX-License-Identifier: Apache-2.0
"""V4.1 expert assembly preserving BF16 projection boundaries.

Packed FP4 weights/scales stay frozen. Activation quantization uses the stated
identity STE; gradients cover inputs and routing weights. This is a dispatched
single expert, not a 384-expert router/EP implementation.
"""

import torch

from rl_engine.backends.rocm.moe.v41_packed_mx import V41PackedMXBackend
from rl_engine.backends.rocm.moe.clamp_swiglu import RocmClampSwiGLUWeighted
from rl_engine.backends.rocm.moe.route_weight_grad import RocmRouteWeightGradient
from rl_engine.contracts.operators.expert_training import PackedExpertWeight


class _PackedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales):
        n, khalf = codes.shape
        k = khalf * 2
        if x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16:
            raise ValueError("expected BF16 dispatched expert inputs[M,K]")
        backend = V41PackedMXBackend()
        weight = PackedExpertWeight(codes.unsqueeze(0), scales.unsqueeze(0), (1, n, k))
        offsets = torch.tensor([0, x.shape[0]], device=x.device, dtype=torch.int64)
        output = backend.forward(backend.quantize(x.contiguous()), weight, offsets).bfloat16()
        ctx.save_for_backward(codes, scales, offsets)
        return output

    @staticmethod
    def backward(ctx, dy):
        codes, scales, offsets = ctx.saved_tensors
        n, khalf = codes.shape
        weight = PackedExpertWeight(codes.unsqueeze(0), scales.unsqueeze(0), (1, n, khalf * 2))
        dx = V41PackedMXBackend().backward(dy.bfloat16().contiguous(), weight, offsets)
        return dx.bfloat16(), None, None


class V41FP4Linear(torch.nn.Module):
    profile = "dsv41-fp4-block32-bf16-projection-boundary-input-ste-v1"

    def __init__(self, codes, scales):
        super().__init__()
        if codes.dtype not in (torch.int8, torch.uint8) or scales.dtype not in (
            torch.uint8,
            torch.float8_e8m0fnu,
        ):
            raise TypeError("expected original packed I8/uint8 FP4 and E8M0 scales")
        if (
            codes.ndim != 2
            or codes.shape[1] % 16
            or scales.shape != (codes.shape[0], codes.shape[1] // 16)
        ):
            raise ValueError("invalid packed FP4 shape/scale layout")
        if not torch.version.hip or codes.device.type != "cuda" or scales.device != codes.device:
            raise ValueError("packed expert weights/scales must share an AMD GPU")
        if bool((scales.view(torch.uint8) == 255).any()):
            raise ValueError("NaN expert scale")
        self.register_buffer("codes", codes.view(torch.uint8).contiguous())
        self.register_buffer("scales", scales.view(torch.uint8).contiguous())

    def forward(self, x):
        return _PackedLinear.apply(x, self.codes, self.scales)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, weights):
        # Released projection outputs are rounded to BF16 before float() here.
        gate, up = gate.float(), up.float()
        y, state = RocmClampSwiGLUWeighted.forward(gate, up, weights)
        ctx.save_for_backward(gate, up, weights, state.sigmoid, state.silu, state.unweighted)
        return y

    @staticmethod
    def backward(ctx, dy):
        from rl_engine.contracts.operators.clamp_swiglu import ClampSwiGLUState

        gate, up, weights, *saved = ctx.saved_tensors
        state = ClampSwiGLUState(*saved)
        dg, du = RocmClampSwiGLUWeighted.backward(dy, gate, up, weights, state)
        dp = RocmRouteWeightGradient.reduce(dy, state.silu, up.clamp(-10, 10))
        return dg.bfloat16(), du.bfloat16(), dp


class V41Expert(torch.nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.w1, self.w2, self.w3 = w1, w2, w3

    def forward(self, x, routing_weights=None):
        if x.ndim != 2:
            raise ValueError("V4.1 dispatched expert expects X[M,D]")
        if routing_weights is None:
            routing_weights = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        return self.w2(_Activation.apply(self.w1(x), self.w3(x), routing_weights))
