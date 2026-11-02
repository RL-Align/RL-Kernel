"""Platform-selected matrix multiplication operators."""

from rl_engine.ops.gemm.det_gemm import DetGemmOp, deterministic_gemm

__all__ = ["DetGemmOp", "deterministic_gemm"]
