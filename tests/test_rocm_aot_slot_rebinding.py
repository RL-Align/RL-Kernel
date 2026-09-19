# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

import rl_engine
import rl_engine.kernels.ops.rocm.matmul.det_gemm as gemm


def test_exported_row_projection_graph_resolves_current_process_ipc_handle(monkeypatch, tmp_path):
    observed = []

    def reduce(handle, value, output):
        observed.append(handle)
        output.copy_(value * 2)

    monkeypatch.setattr(
        rl_engine, "_C", SimpleNamespace(deterministic_collective_rocm_ipc_all_reduce_input=reduce)
    )
    monkeypatch.setitem(
        gemm._DIRECT_STAGING_BY_SLOT, 7, (111111, torch.zeros(4, 4), torch.zeros(4, 4))
    )

    class Model(torch.nn.Module):
        def forward(self, value):
            return gemm.row_parallel_reduce_from_slot(value, 7)

    inputs = torch.arange(8).reshape(2, 4).float()
    exported = torch.export.export(Model(), (inputs,))
    path = tmp_path / "model.pt2"
    torch.export.save(exported, path)
    # Simulate loading the same AOT graph with different process-local pointers.
    monkeypatch.setitem(
        gemm._DIRECT_STAGING_BY_SLOT, 7, (222222, torch.zeros(4, 4), torch.zeros(4, 4))
    )
    output = torch.export.load(path).module()(inputs)
    assert observed == [222222]
    assert torch.equal(output, inputs * 2)
