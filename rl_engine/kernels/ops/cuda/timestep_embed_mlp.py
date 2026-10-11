# SPDX-License-Identifier: Apache-2.0
"""CUDA fixed-order FP32 primitives, lazily built in PyTorch's extension cache."""

import os
from functools import lru_cache
from pathlib import Path

from ..timestep_embed_mlp import TimestepEmbedMLPOp


@lru_cache(maxsize=1)
def _extension():
    import importlib

    if os.environ.get("TIMESTEP_CUDA_JIT_ONLY") != "1":
        try:
            return importlib.import_module("rl_engine._timestep_cuda")
        except ModuleNotFoundError as exc:
            if exc.name != "rl_engine._timestep_cuda":
                raise
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parents[4] / "csrc/cuda/timestep_embed_mlp.cu"
    if not source.is_file():
        raise RuntimeError("CUDA extension is not installed and checkout CUDA source is absent")
    return load(
        name="timestep_official_cuda",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--fmad=false"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class CudaPrimitives:
    def __init__(self):
        self.ext = _extension()

    def mm(self, a, b, trace):
        trace.kernels.append("cuda.timestep_mm")
        return self.ext.mm(a, b)

    def embedding(self, t, f, trace):
        trace.kernels.append("cuda.timestep_embedding")
        return self.ext.embedding(t, f)

    def unary(self, x, derivative, trace):
        trace.kernels.append("cuda.timestep_silu_grad" if derivative else "cuda.timestep_silu")
        return self.ext.unary(x.contiguous(), derivative)

    def timestep_grad(self, de, e, f, trace):
        trace.kernels.append("cuda.timestep_dt")
        return self.ext.timestep_grad(de, e, f)


class CudaTimestepEmbedMLPOp(TimestepEmbedMLPOp):
    def __init__(self, *, allow_fallback=False):
        super().__init__("cuda", allow_fallback=allow_fallback)
