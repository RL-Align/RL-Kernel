# SPDX-License-Identifier: Apache-2.0
"""Original W4A16 expert boundaries for native MCore, retaining its dispatcher."""

import torch

from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16


class _Projection(torch.nn.Module):
    def __init__(self, codes, scales):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)

    def forward(self, x, offsets, gammas=None):
        return packed_w4a16(x, self.codes, self.scales, offsets, gammas)


class _SwiGLU(torch.nn.Module):
    def forward(self, z):
        from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

        out = torch.empty((*z.shape[:-1], z.shape[-1] // 2), device=z.device, dtype=z.dtype)
        # Preserve the actual pinned ROCm production implementation, including
        # its intermediate BF16 conversion and its real autograd derivative.
        swiglu_limit_func(out, z, 10.0)
        return out


class NativeW4A16Experts(torch.nn.Module):
    profile = "dsv4-original-w4a16-bf16-gemm-boundary-gamma-after-down-v1"

    def __init__(self, original):
        super().__init__()
        self.gate_up = _Projection(original.w1_codes, original.w1_scales)
        self.activation = _SwiGLU()
        self.down = _Projection(original.w2_codes, original.w2_scales)
        self.config = original.config

    def forward(self, x, tokens_per_expert, permuted_probs, **kwargs):
        if kwargs:
            raise ValueError("ordinary eager native W4A16 dispatch only")
        offsets = torch.cat(
            (
                torch.zeros(1, device=x.device, dtype=torch.int64),
                tokens_per_expert.to(device=x.device, dtype=torch.int64).cumsum(0),
            )
        )
        gate_up = self.gate_up(x.contiguous(), offsets)
        activated = self.activation(gate_up)
        y = self.down(activated, offsets, permuted_probs.reshape(-1).float().contiguous())
        if not hasattr(self, "_executed"):
            print(
                "RLK_NATIVE_W4A16_EXPERTS_EXECUTED",
                __file__,
                self.profile,
                "original packed E2M1/E8M0 weights; BF16 activations",
                flush=True,
            )
            self._executed = True
        return y, None

    def backward_dw(self):
        pass


def bind_native_w4a16(moe):
    from .checkpoint import NativePackedExperts

    if not isinstance(moe.experts, NativePackedExperts):
        raise ValueError("expected original packed checkpoint experts")
    original = moe.experts
    replacement = NativeW4A16Experts(original)
    assert replacement.gate_up.codes is original.w1_codes
    assert replacement.down.codes is original.w2_codes
    moe.experts = replacement
    return replacement
