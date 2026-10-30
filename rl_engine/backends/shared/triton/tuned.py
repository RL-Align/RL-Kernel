# SPDX-License-Identifier: Apache-2.0
"""Explicit performance backends; strict contract dispatch never registers these."""

from functools import wraps

from rl_engine.backends.shared.triton.attention.deterministic_attn import (
    TritonDeterministicAttentionOp,
)
from rl_engine.backends.shared.triton.autotuning import TuneState, tuning_scope
from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp
from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import TritonBatchInvariantLogpOp
from rl_engine.platforms.autotuner import autotune_enabled


def _performance_entry(method):
    """Wrap ``method`` to run in a core tuning scope and expose its launch provenance."""

    @wraps(method)
    def call(self, *args, **kwargs):
        """Run the fixed backend method inside a tuning scope and record provenance."""
        state = TuneState(autotune_enabled(), core=True)
        self.autotune_provenance = {
            "backend": self.backend_id,
            "enabled": state.enabled,
            "launches": state.records,
        }
        with tuning_scope(state):
            return method(self, *args, **kwargs)

    return call


class TunedGemmOp(TritonDetGemmOp):
    """Fixed K-tree GEMM with validated launch tuning; no invariance capability claim."""

    backend_id = "rlkernel.triton.tuned_gemm"
    is_batch_invariant = False
    __call__ = _performance_entry(TritonDetGemmOp.__call__)
    forward_fp32 = _performance_entry(TritonDetGemmOp.forward_fp32)
    forward_accum_fp32 = forward_fp32


class TunedAttentionOp(TritonDeterministicAttentionOp):
    """Tune launch parameters while preserving the fixed attention arithmetic."""

    backend_id = "rlkernel.triton.tuned_attention"
    is_batch_invariant = False

    def __init__(self, *, require_bitwise_libm: bool = False):
        """Build the op; ``require_bitwise_libm`` defaults to False for the Triton path."""
        # Compare to the fixed Triton implementation, not the native CUDA libm.
        super().__init__(require_bitwise_libm=require_bitwise_libm)

    forward_with_lse = _performance_entry(TritonDeterministicAttentionOp.forward_with_lse)
    forward_fp32 = _performance_entry(TritonDeterministicAttentionOp.forward_fp32)


class TunedLogpOp(TritonBatchInvariantLogpOp):
    """Selected logprob tuning with the original fixed vocabulary traversal."""

    backend_id = "rlkernel.triton.tuned_logp"
    is_batch_invariant = False
    apply = _performance_entry(TritonBatchInvariantLogpOp.apply)
    forward_with_lse = _performance_entry(TritonBatchInvariantLogpOp.forward_with_lse)
