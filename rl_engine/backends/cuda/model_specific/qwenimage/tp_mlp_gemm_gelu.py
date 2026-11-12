# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fixed-tree tensor-parallel Qwen-Image MLP with canonical parameter gradients."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
    require_device,
    require_extension,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.canonical_backward import active_session
from rl_engine.reference.model_specific.qwenimage.tp_mlp_gemm_gelu import (
    HIDDEN,
    TP_SIZES,
    shard_mlp_parameters,
    validate_mlp,
)

_CONTRACT = "qwenimage-tp-mlp-fixed-tree"
_PARAMETERS = ("up.weight", "up.bias", "down.weight", "down.bias")


def _tree_and_reduce(a, b, collective, bias=None):
    """Complete the FP32 tree before bias and the single BF16 output cast."""
    capacity = a.shape[0]
    if collective is not None:
        capacity = min(capacity, collective.max_size_bytes // (HIDDEN * 4))
    output = torch.empty((a.shape[0], HIDDEN), dtype=a.dtype, device=a.device)
    for start in range(0, a.shape[0], capacity):
        stop = min(start + capacity, a.shape[0])
        partial = _C.mlp_up_gemm_gelu_cuda_tree_fp32(a[start:stop], b)
        if collective is not None:
            partial = collective.all_reduce(partial)
        if bias is not None:
            partial.add_(bias.float())
        output[start:stop].copy_(partial.to(a.dtype))
    return output


class _TPMlp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, up_weight, up_bias, down_weight, down_bias, keys, parameter_id, collective, previous
    ):
        hidden, pre = _C.mlp_up_gemm_gelu_cuda_forward(x, up_weight, up_bias, True)
        ctx.save_for_backward(x, up_weight, down_weight, hidden, pre)
        ctx.collective = collective
        ctx.session = active_session()
        ctx.ids = [str(parameter_id) + "." + name for name in _PARAMETERS]
        ctx.slots = {}
        for index in range(4):
            if ctx.needs_input_grad[index + 1]:
                ctx.slots[index] = ctx.session.register(ctx.ids[index], keys)
        return _tree_and_reduce(hidden, down_weight, collective, down_bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        # The extra forward edge orders independent microbatch VJPs across ranks.
        # Release the session's tail before executing its first backward use.
        if ctx.collective is not None:
            ctx.session._qwenimage_mlp_tails.pop(ctx.collective, None)
        x, up_weight, down_weight, hidden, pre = ctx.saved_tensors
        grad = grad.contiguous()
        dx = du = da = dd = db = None
        gate = None
        executed = []
        if any(ctx.needs_input_grad[:3]):
            dh = _C.mlp_up_gemm_gelu_cuda_dx(grad, down_weight)
            gate = _C.mlp_up_gemm_gelu_cuda_gate(dh, pre)
            executed.extend(("mlp_up_gemm_gelu_cuda_dx", "mlp_up_gemm_gelu_cuda_gate"))
        if ctx.needs_input_grad[0]:
            dx = _tree_and_reduce(gate, up_weight.t().contiguous(), ctx.collective)
            executed.append("mlp_up_gemm_gelu_cuda_tree_fp32")

        def weight_reducer(rows, grads):
            executed.append("mlp_up_gemm_gelu_cuda_dw")
            return _C.mlp_up_gemm_gelu_cuda_dw(grads, rows)

        def bias_reducer(rows):
            executed.append("mlp_up_gemm_gelu_cuda_db")
            return _C.mlp_up_gemm_gelu_cuda_db(rows).to(x.dtype)

        if ctx.needs_input_grad[1]:
            du = ctx.session.submit_linear(ctx.ids[0], ctx.slots[0], x, gate, weight_reducer)
        if ctx.needs_input_grad[2]:
            da = ctx.session.submit_rows(ctx.ids[1], ctx.slots[1], gate, bias_reducer)
        if ctx.needs_input_grad[3]:
            dd = ctx.session.submit_linear(ctx.ids[2], ctx.slots[2], hidden, grad, weight_reducer)
        if ctx.needs_input_grad[4]:
            db = ctx.session.submit_rows(ctx.ids[3], ctx.slots[3], grad, bias_reducer)
        if executed:
            record_backward(
                "qwenimage_tp_mlp_gemm_gelu",
                kernel_id="+".join("rl_engine._C." + name for name in executed),
                impl=_CONTRACT,
                family="cuda",
            )
        return dx, du, da, dd, db, None, None, None, None


class QwenImageTPMlpGemmGeluCudaOp:
    """One stream's channel-sharded MLP with replicated inputs and outputs.

    All ranks receive identical rows and upstream gradients. Up rows and down
    columns own matching intermediate channels. Parameter IDs distinguish streams
    and layers while remaining shared across a parameter's microbatch uses.
    """

    op_class = "reduction"
    kernel_id = "rl_engine._C.mlp_up_gemm_gelu_cuda_forward"

    def __init__(self, collective=None):
        require_extension()
        symbols = ("forward", "tree_fp32", "gate", "dx", "dw", "db")
        if not all(callable(getattr(_C, "mlp_up_gemm_gelu_cuda_" + s, None)) for s in symbols):
            raise RuntimeError("rebuild rl_engine._C with the fixed-tree MLP entry points")
        self.collective = collective
        self.tp_world = 1 if collective is None else collective.world_size
        if type(self.tp_world) is not int or self.tp_world not in TP_SIZES:
            raise ValueError("TP size must be 1, 2, 4 or 8")
        if collective is not None:
            if not 0 <= collective.rank < self.tp_world:
                raise ValueError("collective rank must be in [0, TP size)")
            if collective.backend_id != "cuda_ipc_fixed_tree":
                raise ValueError("TP MLP requires the existing CUDA deterministic collective")
            if collective.max_size_bytes < HIDDEN * 4:
                raise ValueError("collective capacity cannot hold one FP32 output row")

    def forward(
        self, x, up_weight, up_bias, down_weight, down_bias, *, logical_keys=None, parameter_id=None
    ):
        validate_mlp(x, up_weight, up_bias, down_weight, down_bias, self.tp_world)
        require_device(x)
        if self.collective is not None and self.collective.device != x.device:
            raise ValueError("collective and MLP tensors must use the same device")
        if logical_keys is not None:
            if logical_keys.device != x.device or logical_keys.dtype != torch.int64:
                raise ValueError("logical_keys must be int64 on the input device")
            if (
                logical_keys.ndim != 2
                or logical_keys.shape[0] != x.shape[0]
                or logical_keys.shape[1] not in (2, 3)
            ):
                raise ValueError("logical_keys must have shape [M,2] or [M,3]")
        parameters = (up_weight, up_bias, down_weight, down_bias)
        session = active_session()
        differentiable = torch.is_grad_enabled() and any(t.requires_grad for t in (x, *parameters))
        if self.collective is not None and differentiable and session is None:
            raise RuntimeError("distributed MLP autograd requires a canonical backward session")
        trainable = torch.is_grad_enabled() and any(t.requires_grad for t in parameters)
        if trainable:
            if active_session() is None:
                raise RuntimeError("trainable MLP parameters require a canonical backward session")
            if logical_keys is None or not isinstance(parameter_id, str) or not parameter_id:
                raise ValueError("trainable MLP parameters require logical_keys and parameter_id")
        x, up_weight, up_bias, down_weight, down_bias = (
            t.contiguous() for t in (x, up_weight, up_bias, down_weight, down_bias)
        )
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in (x, *parameters)):
            hidden, _ = _C.mlp_up_gemm_gelu_cuda_forward(x, up_weight, up_bias, False)
            return _tree_and_reduce(hidden, down_weight, self.collective, down_bias)
        previous = None
        if self.collective is not None:
            # A shape signature cannot distinguish two equal-sized requests.
            # An autograd edge, rather than scheduler priority, fixes their order.
            if not hasattr(session, "_qwenimage_mlp_tails"):
                session._qwenimage_mlp_tails = {}
            previous = session._qwenimage_mlp_tails.get(self.collective)
        output = _TPMlp.apply(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            logical_keys,
            parameter_id,
            self.collective,
            previous,
        )
        if self.collective is not None:
            session._qwenimage_mlp_tails[self.collective] = output
        return output

    __call__ = forward

    def implementation_info(self, x, up_weight, up_bias, down_weight, down_bias, **kwargs):
        validate_mlp(x, up_weight, up_bias, down_weight, down_bias, self.tp_world)
        require_device(x)
        return {
            "operator": "qwenimage_tp_mlp_gemm_gelu",
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "fallback": False,
            "contract": _CONTRACT,
            "kernel_id": self.kernel_id,
            "tp_world_size": self.tp_world,
            "transport": "none" if self.collective is None else self.collective.backend_id,
            "accumulation_dtype": "float32",
            "execution_dtype": str(x.dtype),
            "scope": "implementation configuration, not a per-call execution record",
        }


__all__ = ["QwenImageTPMlpGemmGeluCudaOp", "shard_mlp_parameters"]
