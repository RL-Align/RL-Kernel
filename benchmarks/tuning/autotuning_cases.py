# SPDX-License-Identifier: Apache-2.0
"""Deterministic workload construction shared by qualification and benchmarks."""

from dataclasses import dataclass, field
from typing import Callable

import torch

OPERATORS = (
    "silu",
    "swiglu",
    "rope",
    "embedding",
    "rmsnorm",
    "ratio_kl",
    "grpo",
    "gemm",
    "attention",
    "logp",
)


@dataclass
class Workload:
    """An operator call with its inputs, keyword arguments and whether it has a backward."""

    op: Callable
    inputs: tuple
    kwargs: dict = field(default_factory=dict)
    backward: bool = True

    @property
    def variables(self):
        """Return the inputs that require gradients, or ``()`` without a backward."""
        return tuple(x for x in self.inputs if x.requires_grad) if self.backward else ()

    def forward(self):
        """Call the operator and return its outputs, always as a tuple."""
        value = self.op(*self.inputs, **self.kwargs)
        return value if isinstance(value, tuple) else (value,)


def make_case(name, dtype, *, rows=3, width=129, layout="contiguous", values="random", causal=True):
    """Build a deterministic CUDA workload for operator ``name`` with the requested data."""

    def tensor(shape, grad=True):
        """Create a CUDA tensor of ``shape``, optionally strided or filled with special values."""
        storage_shape = (*shape[:-1], shape[-1] * 2) if layout == "strided" else shape
        x = torch.randn(storage_shape, device="cuda", dtype=dtype)
        if layout == "strided":
            x = x[..., ::2]
        if values == "special" and x.numel():
            info = torch.finfo(dtype)
            pattern = torch.tensor(
                [
                    0.0,
                    -0.0,
                    info.tiny,
                    -info.tiny,
                    info.max / 16,
                    -info.max / 16,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                ],
                device="cuda",
                dtype=dtype,
            )
            x.copy_(pattern.repeat((x.numel() + 8) // 9)[: x.numel()].reshape(x.shape))
        return x.requires_grad_(grad)

    if name in ("silu", "swiglu"):
        from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp, TritonSwiGLUOp

        return Workload(
            TritonSiLUOp() if name == "silu" else TritonSwiGLUOp(),
            (
                (tensor((rows, width)),)
                if name == "silu"
                else (tensor((rows, width)), tensor((rows, width)))
            ),
        )
    if name == "rope":
        from rl_engine.backends.shared.triton.rope.rope import TritonRoPEOp

        return Workload(
            TritonRoPEOp(), (tensor((2, rows, width)), torch.arange(rows, device="cuda"))
        )
    if name == "embedding":
        from rl_engine.backends.shared.triton.embedding.embedding import TritonEmbeddingOp

        ids = torch.arange(rows, device="cuda") % 3
        return Workload(TritonEmbeddingOp(), (ids, tensor((7, width))))
    if name == "rmsnorm":
        from rl_engine.backends.shared.triton.norm.rmsnorm import RMSNormTritonOp

        return Workload(RMSNormTritonOp(), (tensor((rows, width)), tensor((width,))))
    if name == "ratio_kl":
        from rl_engine.backends.shared.triton.loss.ratio_kl import TritonRatioKLOp

        ids = torch.arange(rows, device="cuda").reshape(1, rows) % width
        mask = (torch.arange(rows, device="cuda") % 2 == 0).reshape(1, rows)
        return Workload(
            TritonRatioKLOp(),
            (
                tensor((1, rows, width)),
                tensor((1, rows, width), False),
                ids,
                mask,
                tensor((1, rows), False),
            ),
        )
    if name == "logp":
        from rl_engine.backends.shared.triton.tuned import TunedLogpOp

        ids = torch.arange(rows, device="cuda") % width
        ids[1::2] = -100
        return Workload(TunedLogpOp(), (tensor((rows, width)), ids))
    if name == "grpo":
        from rl_engine.backends.shared.triton.loss.grpo_loss import TritonGRPOLossOp

        return Workload(
            TritonGRPOLossOp().group_advantages,
            (tensor((rows * width,), False),),
            {"samples_per_prompt": width},
            backward=False,
        )
    if name == "gemm":
        from rl_engine.backends.shared.triton.tuned import TunedGemmOp

        op = TunedGemmOp()
        return Workload(
            op if dtype == torch.bfloat16 else op.forward_fp32,
            (tensor((rows, width)), tensor((width, 65))),
            backward=dtype == torch.bfloat16,
        )
    if name == "attention":
        from rl_engine.backends.shared.triton.tuned import TunedAttentionOp

        mask = torch.ones((1, rows + 2), device="cuda", dtype=torch.bool)
        mask[:, -1] = False
        return Workload(
            TunedAttentionOp(),
            (
                tensor((1, 2, rows, width)),
                tensor((1, 1, rows + 2, width)),
                tensor((1, 1, rows + 2, width)),
            ),
            {"causal": causal, "key_padding_mask": mask},
        )
    raise ValueError(name)


def supported(name, dtype):
    """Return False for dtypes the existing operators reject: FP16 GEMM, FP32 attention."""
    return not (
        (name == "gemm" and dtype == torch.float16)
        or (name == "attention" and dtype == torch.float32)
    )


def preset_shape(name, preset):
    """Return the ``(rows, width)`` used for ``name`` under the small or throughput preset."""
    if preset == "small":
        return (3, 128) if name in ("rope", "attention") else (3, 129)
    return {
        "gemm": (64, 1024),
        "attention": (129, 128),
        "rope": (1024, 128),
        "grpo": (1024, 8),
        "embedding": (4096, 1024),
    }.get(name, (1024, 4096))


def fixed_case(case, name, dtype):
    """Use the existing fixed public core backend as the performance baseline."""
    if name == "gemm":
        from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

        fixed = TritonDetGemmOp()
        op = fixed if dtype == torch.bfloat16 else fixed.forward_fp32
    elif name == "attention":
        from rl_engine.backends.shared.triton.attention.deterministic_attn import (
            TritonDeterministicAttentionOp,
        )

        op = TritonDeterministicAttentionOp(require_bitwise_libm=False)
    elif name == "logp":
        from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        op = TritonBatchInvariantLogpOp()
    else:
        return case
    return Workload(op, case.inputs, case.kwargs, case.backward)
