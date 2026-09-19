# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.vllm_runtime import _patch_qwen3_strict_model


def test_canonical_qkv_compiles_with_vllm_parameter_override(monkeypatch):
    parameter = pytest.importorskip("vllm.model_executor.parameter")
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "8")

    class RMSNorm:
        def forward_cuda(self, x):
            return x

        def forward_native(self, x):
            return x

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.Tensor._make_subclass(
                parameter.BasevLLMParameter, torch.arange(48).reshape(12, 4).float(), False
            )
            self.output_partition_sizes = [8, 2, 2]
            self.tp_size = 4

    class Method:
        def apply(self, layer, x, bias=None):
            return x

    class Attention:
        def __init__(self):
            self.qkv_proj = Layer()
            self.o_proj = SimpleNamespace(tp_size=1)

    _patch_qwen3_strict_model(
        rms_norm_cls=RMSNorm,
        linear_method_cls=Method,
        attention_cls=Attention,
        det_gemm=lambda a, b: a @ b,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = Attention().qkv_proj
            self.method = Method()

        def forward(self, value):
            return self.method.apply(self.projection, value)

    model = Model()
    value = torch.arange(8).reshape(2, 4).float()
    expected = model(value)
    actual = torch.compile(model, backend="eager", fullgraph=True)(value)
    assert torch.equal(actual, expected)
