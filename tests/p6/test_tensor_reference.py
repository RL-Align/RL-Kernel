# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from rl_engine.p6.recordings import golden_records
from rl_engine.p6.torch_reference import conformance


def test_cpu_tensor_reference_contiguous_and_strided():
    provenance, outcomes = conformance(golden_records(), "cpu")
    assert provenance["compiled_production_kernel"] is False
    assert len(outcomes) == 18
    assert all(r["verdict"] == "REFERENCE_BYTES_PASS" for r in outcomes)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "cuda-graph"])
def test_opt_in_gpu_reference(graph):
    if os.environ.get("P6_RUN_GPU") != "1":
        pytest.skip("set P6_RUN_GPU=1 to require actual CUDA reference execution")
    assert torch.cuda.is_available(), "requested GPU test must not silently skip"
    assert torch.version.hip is None, "this named slice is NVIDIA CUDA; ROCm cert is separate"
    provenance, outcomes = conformance(golden_records(), "cuda:0", graph=graph)
    assert provenance["gpu_name"]
    assert len(outcomes) == 18
    assert all(r["graph"] == graph for r in outcomes)
