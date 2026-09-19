# SPDX-License-Identifier: Apache-2.0
import os
import sys
from types import ModuleType, SimpleNamespace

import torch
import pytest

from rl_engine.integrations.vllm_runtime import _configure_strict_ffn_compilation


def test_rocm_aot_cache_separates_canonical_topologies(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", "test")
    vllm = ModuleType("vllm")
    vllm.envs = SimpleNamespace(VLLM_CACHE_ROOT=str(tmp_path))
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = SimpleNamespace(FULL_AND_PIECEWISE="hip_graph")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    compilation = SimpleNamespace(splitting_ops=[], cudagraph_mode=None, inductor_compile_config={})
    namespaces = []
    for canonical_tp, vocab in [(4, 152064), (8, 152064), (8, 152576)]:
        monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", str(canonical_tp))
        monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE", str(vocab))
        _configure_strict_ffn_compilation(SimpleNamespace(compilation_config=compilation))
        namespaces.append(os.environ["VLLM_CACHE_ROOT"])
    assert len(set(namespaces)) == 3
    assert compilation.cudagraph_mode == "hip_graph"


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm compilation contract")
@pytest.mark.parametrize("chunks", [4, 8])
def test_rocm_compiled_canonical_bf16_tree_preserves_intermediate_rounding(
    monkeypatch, tmp_path, chunks
):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    compilation = SimpleNamespace(splitting_ops=[], cudagraph_mode=None, inductor_compile_config={})
    _configure_strict_ffn_compilation(SimpleNamespace(compilation_config=compilation))

    def tree(values):
        while len(values) > 1:
            values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
        return values[0]

    generator = torch.Generator(device="cuda").manual_seed(73)
    values = [
        torch.randn(16, 128, device="cuda", generator=generator).bfloat16() for _ in range(chunks)
    ]
    compiled = torch.compile(tree, fullgraph=True, options=compilation.inductor_compile_config)
    assert torch.equal(tree(values).view(torch.int16), compiled(values).view(torch.int16))
