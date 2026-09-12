# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.integrations.megatron_runtime import _patch_strict_attention_projections


def test_strict_attention_runtime_core_does_not_pollute_state_dict():
    class CoreAttention(torch.nn.Module):
        def get_extra_state(self):
            return {"runtime": True}

    class ColumnLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.gather_output = False
            self.skip_bias_add = False
            self.bias = None
            self.allreduce_dgrad = True

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input @ weight.t()

    class RowLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(2))
            self.input_is_parallel = True
            self.skip_bias_add = False
            self.return_bias = False
            self.bias = None

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input @ weight.t()

    class SelfAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = ColumnLinear()
            self.linear_proj = RowLinear()
            self.core_attention = CoreAttention()

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=ColumnLinear,
        row_linear_cls=RowLinear,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
        copy_to_tp=lambda value: value,
        reduce_from_tp=lambda value: value,
    )

    attention = SelfAttention()

    assert (
        getattr(
            attention.linear_proj,
            "__rl_kernel_strict_attention_core__",
        )
        is attention.core_attention
    )
    assert not any("__rl_kernel_strict_attention_core__" in key for key in attention.state_dict())
