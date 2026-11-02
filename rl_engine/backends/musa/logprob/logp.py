# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""MUSA-owned fused selected-logprob operator."""

from rl_engine.backends.cuda.logprob.logp import FusedLogpGenericOp


class MusaFusedLogpOp(FusedLogpGenericOp):
    """Use the native MUSA forward and backward functions exposed by ``rl_engine._C``."""
