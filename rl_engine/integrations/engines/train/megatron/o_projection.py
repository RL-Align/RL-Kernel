# SPDX-License-Identifier: Apache-2.0
"""Use shared BF16-master VJPs and a canonical session across microbatches."""

from rl_engine.integrations.common.o_projection import bind_output_projection
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

__all__ = ["bind_output_projection", "canonical_backward_session"]
