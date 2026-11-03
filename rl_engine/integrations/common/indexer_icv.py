# SPDX-License-Identifier: Apache-2.0
"""Explicit quantized boundaries shared by serving and training-forward adapters."""

from rl_engine.models.deepseek_v4.indexer_icv import BoundICVQueries, QuantizedICVRows, require


def bind_icv_queries(operator, values, scales, query_ids, *, layout, version, ownership="borrowed"):
    require(
        scales is not None,
        "raw ICV needs independent Q scales; fused_indexer_q_rope_quant folds them "
        "into head weights. Preserve scales upstream; never use head weights as Q scales.",
    )
    operand = QuantizedICVRows.bind(
        values,
        scales,
        role="q",
        version=version,
        layout=layout,
        contract=operator.contract,
        ownership=ownership,
    )
    return BoundICVQueries.bind(operand, query_ids)


def bind_global_icv_keys(
    operator,
    values,
    scales,
    key_ids,
    *,
    layout,
    version,
    request_key_counts,
    key_owners=None,
    cp_size=1,
    ownership="borrowed",
):
    operand = QuantizedICVRows.bind(
        values,
        scales,
        role="k",
        version=version,
        layout=layout,
        contract=operator.contract,
        ownership=ownership,
    )
    return operator.prepare_keys(
        operand,
        key_ids,
        request_key_counts=request_key_counts,
        key_owners=key_owners,
        cp_size=cp_size,
    )
