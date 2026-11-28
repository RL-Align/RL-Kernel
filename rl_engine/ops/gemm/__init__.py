"""Platform-selected matrix multiplication operators."""

from rl_engine.ops.gemm.det_gemm import DetGemmOp, deterministic_gemm, fixed_leaf_linear

__all__ = ["DetGemmOp", "deterministic_gemm", "fixed_leaf_linear"]
