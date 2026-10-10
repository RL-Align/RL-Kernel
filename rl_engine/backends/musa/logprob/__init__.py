# SPDX-License-Identifier: Apache-2.0
"""Package extension point; implementations are registered explicitly."""

from .logp import MusaFusedLogpOp

__all__ = ["MusaFusedLogpOp"]
