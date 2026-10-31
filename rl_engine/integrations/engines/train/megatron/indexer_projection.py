# SPDX-License-Identifier: Apache-2.0
"""Preserve mLite [N,K] masters and BF16 VJP casts; bind a strict BI GEMM provider.

Unlike the native TE/torch.mm VJP, this path uses fixed aiter plans and the
existing canonical backward session. It does not label a native TE kernel BI.
"""

from rl_engine.integrations.common.indexer_projection import bind_framework_indexer
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session


def bind_indexer_projection(
    indexer,
    *,
    contract,
    layer_id,
    backend=None,
    packed_q=None,
    expected_implementation=None,
):
    return bind_framework_indexer(
        indexer,
        contract=contract,
        layer_id=layer_id,
        backend=backend,
        packed_q=packed_q,
        expected_implementation=expected_implementation,
    )


__all__ = ["bind_indexer_projection", "canonical_backward_session"]
