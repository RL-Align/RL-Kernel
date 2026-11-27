# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark():
    script = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_attn_out_bias_gemm.py"
    return runpy.run_path(str(script))


@pytest.mark.parametrize(
    "missing",
    ["attn_out_bias_gemm_cuda_forward", "attn_out_tree_gemm_cuda", "attn_out_dw_left_fold_cuda"],
)
def test_cuda_probe_rejects_incomplete_extension_with_diagnostic(monkeypatch, benchmark, missing):
    module = importlib.import_module("rl_engine.backends.cuda.gemm.attn_out_bias_gemm")
    symbols = {name: object() for name in module._REQUIRED_SYMBOLS if name != missing}
    monkeypatch.setattr(module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(module, "_C", SimpleNamespace(**symbols))
    with pytest.warns(RuntimeWarning, match="Skipping CUDA backend: RuntimeError"):
        assert benchmark["_maybe_cuda_op"]() is None


def test_cuda_probe_accepts_complete_extension(monkeypatch, benchmark):
    module = importlib.import_module("rl_engine.backends.cuda.gemm.attn_out_bias_gemm")
    monkeypatch.setattr(module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(
        module, "_C", SimpleNamespace(**{name: object() for name in module._REQUIRED_SYMBOLS})
    )
    assert isinstance(benchmark["_maybe_cuda_op"](), module.CudaAttnOutBiasGemmOp)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_backend_initialization_error_is_reported(monkeypatch, benchmark, backend):
    path = "cuda" if backend == "cuda" else "shared.triton"
    module = importlib.import_module(f"rl_engine.backends.{path}.gemm.attn_out_bias_gemm")
    class_name = "CudaAttnOutBiasGemmOp" if backend == "cuda" else "TritonAttnOutBiasGemmOp"

    def broken_constructor():
        raise RuntimeError("injected initialization failure")

    monkeypatch.setattr(module, class_name, broken_constructor)
    with pytest.warns(RuntimeWarning, match="injected initialization failure"):
        assert benchmark[f"_maybe_{backend}_op"]() is None
