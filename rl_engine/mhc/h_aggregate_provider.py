# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from rl_engine.mhc.provider import ReferenceProvider


class _HAggregateProvider(ReferenceProvider):
    def capabilities(self):
        return {**super().capabilities(), "backend": self.name, "devices": ["cuda"],
                "implemented_operators": ["h_aggregate"]}

    def provenance(self):
        return {**super().provenance(), "actual_backend": f"{self.name}+oracle-rest"}

    def h_aggregate_fwd(self, pre, r_old):
        return self.op.forward(r_old, pre)

    def h_aggregate_bwd(self, dh, pre, r_old):
        return self.op.backward_fp32(dh, r_old, pre)


class CudaHAggregateProvider(_HAggregateProvider):
    name = "cuda-h-aggregate"

    def __init__(self):
        from rl_engine.kernels.ops.cuda.mhc import MHCPreHAggregateCudaOp

        self.op = MHCPreHAggregateCudaOp()


class TritonHAggregateProvider(_HAggregateProvider):
    name = "triton-h-aggregate"

    def __init__(self):
        from rl_engine.kernels.ops.triton.mhc import MHCPreHAggregateTritonOp

        self.op = MHCPreHAggregateTritonOp()
